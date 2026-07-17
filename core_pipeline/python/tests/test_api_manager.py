from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_ROOT = PROJECT_ROOT / "python" / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from api_manager import ApiManager, ApiProviderAuthLocked, ApiProviderUnavailable, ApiQuotaExhausted, ApiRateLimited


class ApiManagerTests(unittest.TestCase):
    def build_manager(self, env: dict[str, str]) -> ApiManager:
        self.env_patch = patch.dict(os.environ, env, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.load_patch = patch.object(ApiManager, "_load_env", lambda self: None)
        self.load_patch.start()
        self.addCleanup(self.load_patch.stop)
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return ApiManager(Path(temp_dir.name) / "quota_state.json")

    def test_reserve_tracks_usage_without_network(self) -> None:
        manager = self.build_manager(
            {
                "GEMINI_API_KEYS": "key-a,key-b",
                "GEMINI_DAILY_TOKEN_LIMIT": "1000",
                "GEMINI_DAILY_REQUEST_LIMIT": "20",
                "API_MANAGER_SOFT_CAP_RATIO": "0.8",
            }
        )
        lease = manager.reserve_key("gemini", 100)
        self.assertEqual(lease.key_index, 1)
        manager.mark_success(lease, {"usageMetadata": {"totalTokenCount": 120}})
        status = manager.provider_status("gemini")
        self.assertEqual(status["keyCount"], 2)
        self.assertGreater(status["remainingPercent"], 80)

    def test_soft_cap_rotates_before_exhaustion(self) -> None:
        manager = self.build_manager(
            {
                "MISTRAL_API_KEYS": "key-a,key-b",
                "MISTRAL_DAILY_TOKEN_LIMIT": "1000",
                "MISTRAL_DAILY_REQUEST_LIMIT": "20",
                "API_MANAGER_SOFT_CAP_RATIO": "0.8",
                "MISTRAL_LIMIT_SCOPE": "key",
            }
        )
        first = manager.reserve_key("mistral", 700)
        self.assertEqual(first.key_index, 1)
        second = manager.reserve_key("mistral", 200)
        self.assertEqual(second.key_index, 2, "rotation moves on to key 2 rather than needlessly failing key 1")
        # nextKeyStartIndex is now 2, so this call never revisits key 1 -- it is at 700/800 safe
        # tokens (under its own 80% threshold), not yet soft-capped by this sequence alone.
        status_before = manager.provider_status("mistral")
        self.assertEqual(status_before["lockedKeys"], 0, "key 1 has not crossed its own soft-cap threshold yet")

        # Explicitly push key 1 over its own safe threshold to prove the soft-cap lock actually
        # fires once a key's usage genuinely crosses the ratio (the original point of this test).
        with self.assertRaises(ApiQuotaExhausted):
            manager.reserve_key_index("mistral", 1, 150)
        status_after = manager.provider_status("mistral")
        self.assertEqual(status_after["lockedKeys"], 1)

    def test_429_locks_key_and_retries_next_key(self) -> None:
        manager = self.build_manager(
            {
                "OPENROUTER_API_KEYS": "key-a,key-b",
                "OPENROUTER_DAILY_TOKEN_LIMIT": "1000",
                "OPENROUTER_DAILY_REQUEST_LIMIT": "20",
                "OPENROUTER_LIMIT_SCOPE": "key",
            }
        )
        first = manager.reserve_key("openrouter", 100)
        manager.mark_failure(first, 429, "rate limit")
        second = manager.reserve_key("openrouter", 100)
        self.assertEqual(second.key_index, 2)
        status = manager.provider_status("openrouter")
        self.assertEqual(status["lockedKeys"], 1)

    def test_403_access_denied_is_auth_locked_not_quota_text(self) -> None:
        manager = self.build_manager(
            {
                "GROQ_API_KEYS": "key-a",
                "GROQ_DAILY_TOKEN_LIMIT": "1000",
                "GROQ_DAILY_REQUEST_LIMIT": "20",
                "GROQ_LIMIT_SCOPE": "key",
            }
        )
        lease = manager.reserve_key("groq", 100)
        manager.mark_failure(lease, 403, {"error": {"message": "Access denied. Please check your network settings."}})
        status = manager.provider_status("groq")
        self.assertEqual(status["health"], "auth_locked")
        self.assertIn("access denied", status["reason"])
        self.assertNotIn("quota/rate/auth", status["reason"])
        with self.assertRaises(ApiProviderAuthLocked):
            manager.reserve_key("groq", 100)

    def test_targeted_reservation_respects_soft_cap(self) -> None:
        manager = self.build_manager(
            {
                "GEMINI_API_KEYS": "key-a,key-b",
                "GEMINI_DAILY_TOKEN_LIMIT": "1000",
                "GEMINI_DAILY_REQUEST_LIMIT": "20",
                "API_MANAGER_SOFT_CAP_RATIO": "0.8",
                "GEMINI_LIMIT_SCOPE": "key",
            }
        )
        lease = manager.reserve_key_index("gemini", 2, 700)
        self.assertEqual(lease.key_index, 2)
        with self.assertRaises(ApiQuotaExhausted):
            manager.reserve_key_index("gemini", 2, 200)
        status = manager.provider_status("gemini")
        self.assertEqual(status["keys"][1]["status"], "locked")

    def test_cerebras_key_pool_is_supported_without_network(self) -> None:
        manager = self.build_manager(
            {
                "CEREBERAS_API_KEYS": "key-a,key-b,key-c",
                "CEREBRAS_DAILY_TOKEN_LIMIT": "1000",
                "CEREBRAS_DAILY_REQUEST_LIMIT": "20",
            }
        )
        lease = manager.reserve_key_index("cerebras", 3, 100)
        self.assertEqual(lease.provider, "cerebras")
        self.assertEqual(lease.key_index, 3)
        status = manager.provider_status("cerebras")
        self.assertTrue(status["configured"])
        self.assertEqual(status["keyCount"], 3)

    def test_dashscope_is_not_registered_without_alibaba_access(self) -> None:
        manager = self.build_manager(
            {
                "QWEN_API_KEY_1": "key-a",
                "ALIBABA_API_KEY_2": "key-b",
                "DASHSCOPE_API_KEYS": "key-c",
            }
        )
        self.assertNotIn("dashscope", manager.providers)
        with self.assertRaises(ApiProviderUnavailable):
            manager.reserve_key("dashscope", 100)

    def test_openrouter_default_budget_is_provider_scoped_without_live_api(self) -> None:
        manager = self.build_manager(
            {
                "OPENROUTER_API_KEYS": "key-a,key-b",
                "OPENROUTER_DAILY_TOKEN_LIMIT": "1000",
                "OPENROUTER_DAILY_REQUEST_LIMIT": "3",
                "API_MANAGER_SOFT_CAP_RATIO": "0.8",
            }
        )
        first = manager.reserve_key("openrouter", 100)
        self.assertEqual(first.key_index, 1)
        with self.assertRaises(ApiQuotaExhausted):
            manager.reserve_key("openrouter", 100)
        status = manager.provider_status("openrouter")
        self.assertEqual(status["limitScope"], "provider")
        self.assertEqual(status["lockedKeys"], 2)
        self.assertEqual(status["health"], "exhausted")

    def test_key_scope_can_be_enabled_for_verified_separate_accounts(self) -> None:
        manager = self.build_manager(
            {
                "OPENROUTER_API_KEYS": "key-a,key-b",
                "OPENROUTER_DAILY_TOKEN_LIMIT": "1000",
                "OPENROUTER_DAILY_REQUEST_LIMIT": "20",
                "OPENROUTER_LIMIT_SCOPE": "key",
                "API_MANAGER_SOFT_CAP_RATIO": "0.8",
            }
        )
        first = manager.reserve_key("openrouter", 700)
        self.assertEqual(first.key_index, 1)
        second = manager.reserve_key("openrouter", 200)
        self.assertEqual(second.key_index, 2, "rotation moves on to key 2 rather than needlessly failing key 1")
        status = manager.provider_status("openrouter")
        self.assertEqual(status["limitScope"], "key")
        # Rotation never revisited key 1 in this sequence (nextKeyStartIndex moved to 2), so it
        # is under its own 80% threshold, not soft-capped -- see test_soft_cap_rotates_before_
        # exhaustion for a case that actually drives a key over its threshold.
        self.assertEqual(status["lockedKeys"], 0)

    def test_per_minute_limit_throttles_without_daily_exhaustion(self) -> None:
        manager = self.build_manager(
            {
                "GEMINI_API_KEYS": "key-a,key-b",
                "GEMINI_DAILY_TOKEN_LIMIT": "1000",
                "GEMINI_DAILY_REQUEST_LIMIT": "20",
                "GEMINI_TOKENS_PER_MINUTE": "10000",
                "GEMINI_REQUESTS_PER_MINUTE": "1",
                "API_MANAGER_MINUTE_SOFT_CAP_RATIO": "1.0",
            }
        )
        lease = manager.reserve_key("gemini", 100)
        self.assertEqual(lease.key_index, 1)
        with self.assertRaises(ApiRateLimited):
            manager.reserve_key("gemini", 100)
        status = manager.provider_status("gemini")
        self.assertEqual(status["health"], "rate_limited")
        self.assertGreater(status["remainingPercent"], 0)
        self.assertEqual(status["remainingMinuteRequests"], 0)

    def test_all_keys_exhausted_raises_without_live_api(self) -> None:
        manager = self.build_manager(
            {
                "GROQ_API_KEYS": "key-a",
                "GROQ_DAILY_TOKEN_LIMIT": "1000",
                "GROQ_DAILY_REQUEST_LIMIT": "20",
                "API_MANAGER_SOFT_CAP_RATIO": "0.8",
                "GROQ_LIMIT_SCOPE": "key",
            }
        )
        manager.reserve_key("groq", 700)
        with self.assertRaises(ApiQuotaExhausted):
            manager.reserve_key("groq", 200)

    def test_cloudflare_account_ids_match_key_order(self) -> None:
        manager = self.build_manager(
            {
                "CLOUDFLARE_WORKERS_API_KEYS": "key-a,key-b,key-c",
                "CLOUDFLARE_ACCOUNT_ID": "account-a,account-b,account-c",
            }
        )
        self.assertEqual(manager.cloudflare_account_id_for_key_index(2), "account-b")
        status = manager.provider_status("cloudflare")
        self.assertTrue(status["configured"])

    def test_cloudflare_account_id_mismatch_is_unavailable(self) -> None:
        manager = self.build_manager(
            {
                "CLOUDFLARE_WORKERS_API_KEYS": "key-a,key-b,key-c",
                "CLOUDFLARE_ACCOUNT_IDS": "account-a,account-b",
            }
        )
        with self.assertRaises(ApiProviderUnavailable):
            manager.reserve_key("cloudflare", 100)
        status = manager.provider_status("cloudflare")
        self.assertFalse(status["configured"])

    def test_nvidia_nim_supports_vision_ocr_without_network(self) -> None:
        manager = self.build_manager(
            {
                "NVIDIA_NIM_API_KEYS": "key-a,key-b,key-c",
                "NVIDIA_DAILY_TOKEN_LIMIT": "1000",
                "NVIDIA_DAILY_REQUEST_LIMIT": "20",
            }
        )
        lease = manager.reserve_key("nvidia", 100, capability="vision_ocr")
        self.assertEqual(lease.provider, "nvidia")
        status = manager.provider_status("nvidia")
        self.assertTrue(status["configured"])
        self.assertIn("vision_ocr", status["capabilities"])

    def test_daily_counters_reset_on_date_rollover_but_hour_based_locks_survive(self) -> None:
        manager = self.build_manager(
            {
                "MISTRAL_API_KEYS": "key-a",
                "MISTRAL_DAILY_TOKEN_LIMIT": "1000",
                "MISTRAL_DAILY_REQUEST_LIMIT": "20",
                "API_MANAGER_SOFT_CAP_RATIO": "0.8",
                "MISTRAL_LIMIT_SCOPE": "key",
            }
        )
        with self.assertRaises(ApiQuotaExhausted):
            manager.reserve_key("mistral", 850)
        status = manager.provider_status("mistral")
        self.assertEqual(status["lockedKeys"], 1, "setup: key soft-capped today")

        # Simulate the day rolling over by backdating the persisted date field directly (the
        # real state file, not the in-memory object -- _provider_state_locked compares against
        # self._today() on its next call and must reset from what's actually on disk).
        import json as _json

        data = _json.loads(manager._state_path.read_text(encoding="utf-8"))
        data["providers"]["mistral"]["date"] = "2020-01-01"
        manager._state_path.write_text(_json.dumps(data), encoding="utf-8")

        status_after_rollover = manager.provider_status("mistral")
        self.assertEqual(status_after_rollover["lockedKeys"], 0, "a soft-cap (daily-quota) lock must clear on date rollover")
        lease = manager.reserve_key("mistral", 100)
        self.assertEqual(lease.key_index, 1, "the key is usable again after rollover")

        # Now prove the OTHER kind of lock (401/403 auth, hour-based) does NOT clear just because
        # the date rolled over -- it is deliberately independent of the daily quota window.
        manager.mark_failure(lease, 401, "invalid api key")
        data = _json.loads(manager._state_path.read_text(encoding="utf-8"))
        data["providers"]["mistral"]["date"] = "2020-01-01"
        manager._state_path.write_text(_json.dumps(data), encoding="utf-8")
        status_final = manager.provider_status("mistral")
        self.assertEqual(
            status_final["lockedKeys"],
            1,
            "an auth lock (lockedUntil, hour-based) must survive a date rollover -- it is not a daily-quota lock",
        )

    def test_expired_provider_wide_auth_lock_does_not_mask_a_fresh_unrelated_key_lock(self) -> None:
        # Reproduces a real incident, in the exact sequence it actually happened this project:
        # under limit_scope="provider" (the old default), every key got 403'd (a leaked-key report
        # from the provider), which correctly escalates to a provider-WIDE "auth_locked" lock --
        # only the "provider" scope's escalation path sets provider_state["status"] at all. Days
        # later that lock has long expired, the deployment has since switched to
        # limit_scope="key" (this project's current config), and a completely different,
        # genuinely-current problem hits every key again (a 429 "quota reduced to 0" today).
        # provider_status() must report TODAY's real condition, not the stale provider-wide
        # auth-lock label inherited from the resolved incident.
        manager = self.build_manager(
            {
                "GEMINI_API_KEYS": "key-a,key-b",
                "GEMINI_DAILY_TOKEN_LIMIT": "1000",
                "GEMINI_DAILY_REQUEST_LIMIT": "20",
                "GEMINI_LIMIT_SCOPE": "provider",
            }
        )
        for index in (1, 2):
            lease = manager.reserve_key_index("gemini", index, 10)
            manager.mark_failure(lease, 403, "Your API key was reported as leaked. Please use another API key.")
        status = manager.provider_status("gemini")
        self.assertEqual(status["health"], "auth_locked", "setup: provider-wide lock escalated correctly")

        import json as _json

        data = _json.loads(manager._state_path.read_text(encoding="utf-8"))
        self.assertTrue(
            data["providers"]["gemini"].get("status") and data["providers"]["gemini"].get("lockedUntil"),
            "setup: the provider-level fields must actually be set for this test to reproduce the real bug",
        )

        # Backdate the provider-wide lock (and each key's own lock) so it reads as long expired --
        # simulating "this incident is over, days have passed" without touching the current date.
        provider_entry = data["providers"]["gemini"]
        provider_entry["lockedUntil"] = "2020-01-01T00:00:00+00:00"
        for key_entry in provider_entry["keys"].values():
            key_entry["lockedUntil"] = "2020-01-01T00:00:00+00:00"
            key_entry["status"] = "healthy"
            key_entry["softLocked"] = False
        manager._state_path.write_text(_json.dumps(data), encoding="utf-8")

        status_after_expiry = manager.provider_status("gemini")
        self.assertEqual(
            status_after_expiry["health"],
            "healthy",
            "an expired provider-wide auth lock must actually clear, not just stop blocking reservations",
        )

        # The deployment has since moved to limit_scope="key" (this project's current config,
        # flipped after the old incident) -- a fresh, unrelated, genuinely-current failure now
        # hits both keys individually: a 429 quota error, classified as "exhausted", never
        # "auth_locked". Under scope="key" this never touches provider_state at all, so the bug
        # can ONLY be caught if the stale provider-level fields were actually cleared above.
        os.environ["GEMINI_LIMIT_SCOPE"] = "key"
        for index in (1, 2):
            lease = manager.reserve_key_index("gemini", index, 10)
            manager.mark_failure(
                lease, 429, "Quota exceeded for metric: generate_content_free_tier_requests, limit: 0"
            )
        status_today = manager.provider_status("gemini")
        self.assertEqual(
            status_today["health"],
            "exhausted",
            "today's real quota-exhaustion must be reported as such, not inherit the resolved auth-lock label",
        )
        self.assertEqual(status_today["keys"][0]["status"], "locked")
        with self.assertRaises(ApiQuotaExhausted):
            manager.reserve_key_index("gemini", 1, 10)

    def test_counters_never_go_negative_or_nan(self) -> None:
        manager = self.build_manager(
            {
                "GEMINI_API_KEYS": "key-a",
                "GEMINI_DAILY_TOKEN_LIMIT": "1000",
                "GEMINI_DAILY_REQUEST_LIMIT": "20",
            }
        )
        lease = manager.reserve_key("gemini", 100)
        # A provider reporting fewer actual tokens used than was reserved must not push the
        # tracked usage below zero (mark_success only adds a delta when actual > reserved).
        manager.mark_success(lease, {"usage": {"total_tokens": 1}})
        status = manager.provider_status("gemini")
        self.assertGreaterEqual(status["remainingTokens"], 0)
        self.assertGreaterEqual(status["remainingRequests"], 0)
        self.assertFalse(_is_nan(status["remainingPercent"]))
        self.assertGreaterEqual(status["remainingPercent"], 0)

    def test_unconfigured_provider_is_zero_quota_and_never_reserved(self) -> None:
        # No env keys set for any provider at all -- the zero-quota / offline mode the
        # extension's own validation runners rely on.
        manager = self.build_manager({})
        for provider in manager.providers:
            status = manager.provider_status(provider)
            self.assertFalse(status["configured"], f"{provider} must report unconfigured with no keys set")
            self.assertEqual(status["health"], "unconfigured")
            with self.assertRaises(ApiProviderUnavailable):
                manager.reserve_key(provider, 100)


def _is_nan(value: float) -> bool:
    return value != value


class ApiManagerKeyRoleTests(unittest.TestCase):
    """Per-key role assignment (the user's own idea: e.g. 3 NIM keys from 3 separate accounts,
    one dedicated to translation, one to vision_ocr/other steps, one held back as backup)."""

    def build_manager(self, env: dict[str, str]) -> ApiManager:
        self.env_patch = patch.dict(os.environ, env, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.load_patch = patch.object(ApiManager, "_load_env", lambda self: None)
        self.load_patch.start()
        self.addCleanup(self.load_patch.stop)
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return ApiManager(Path(temp_dir.name) / "quota_state.json")

    def test_no_roles_configured_behaves_exactly_as_before(self) -> None:
        manager = self.build_manager(
            {
                "NVIDIA_NIM_API_KEYS": "key-a,key-b,key-c",
                "NVIDIA_DAILY_TOKEN_LIMIT": "1000",
                "NVIDIA_DAILY_REQUEST_LIMIT": "20",
            }
        )
        lease = manager.reserve_key("nvidia", 100, capability="translation")
        self.assertEqual(lease.key_index, 1, "no KEY_ROLES set: plain rotation order, unchanged")
        lease2 = manager.reserve_key("nvidia", 100, capability="vision_ocr")
        self.assertEqual(lease2.key_index, 2, "any key serves any capability with no roles configured")

    def test_translation_traffic_never_touches_the_vision_dedicated_key(self) -> None:
        manager = self.build_manager(
            {
                "NVIDIA_NIM_API_KEYS": "key-translate,key-vision,key-backup",
                "NVIDIA_KEY_ROLES": "translation,vision_ocr,backup",
                "NVIDIA_DAILY_TOKEN_LIMIT": "1000",
                "NVIDIA_DAILY_REQUEST_LIMIT": "20",
                "NVIDIA_LIMIT_SCOPE": "key",
            }
        )
        # Push the translation-role key to just under its own safe threshold (750/800), then a
        # second translation request tips it over -- if roles are respected, that second request
        # must fall through to the backup key, NEVER to the vision-dedicated key.
        first = manager.reserve_key("nvidia", 750, capability="translation")
        self.assertEqual(first.key_index, 1)
        lease = manager.reserve_key("nvidia", 100, capability="translation")
        self.assertEqual(
            lease.key_index,
            3,
            "translation traffic falls through to the backup key once its own dedicated key is capped, "
            "skipping straight past the vision-dedicated key",
        )

    def test_vision_traffic_never_touches_the_translation_dedicated_key(self) -> None:
        manager = self.build_manager(
            {
                "NVIDIA_NIM_API_KEYS": "key-translate,key-vision,key-backup",
                "NVIDIA_KEY_ROLES": "translation,vision_ocr,backup",
                "NVIDIA_DAILY_TOKEN_LIMIT": "1000",
                "NVIDIA_DAILY_REQUEST_LIMIT": "20",
                "NVIDIA_LIMIT_SCOPE": "key",
            }
        )
        lease = manager.reserve_key("nvidia", 100, capability="vision_ocr")
        self.assertEqual(lease.key_index, 2, "vision_ocr traffic goes straight to the vision-dedicated key")
        status = manager.provider_status("nvidia")
        translate_key = next(k for k in status["keys"] if k["index"] == 1)
        self.assertEqual(translate_key["status"], "healthy", "the translation-dedicated key is untouched by vision traffic")

    def test_backup_key_only_activates_once_its_sibling_role_is_exhausted(self) -> None:
        manager = self.build_manager(
            {
                "NVIDIA_NIM_API_KEYS": "key-translate,key-backup",
                "NVIDIA_KEY_ROLES": "translation,backup",
                "NVIDIA_DAILY_TOKEN_LIMIT": "1000",
                "NVIDIA_DAILY_REQUEST_LIMIT": "20",
                "NVIDIA_LIMIT_SCOPE": "key",
            }
        )
        lease = manager.reserve_key("nvidia", 100, capability="translation")
        self.assertEqual(lease.key_index, 1, "the backup key is not touched while its sibling role still has headroom")
        status = manager.provider_status("nvidia")
        backup_key = next(k for k in status["keys"] if k["index"] == 2)
        self.assertEqual(backup_key["status"], "healthy", "backup key reservation state is untouched")

    def test_role_locked_key_falls_through_to_backup(self) -> None:
        manager = self.build_manager(
            {
                "NVIDIA_NIM_API_KEYS": "key-translate,key-backup",
                "NVIDIA_KEY_ROLES": "translation,backup",
                "NVIDIA_DAILY_TOKEN_LIMIT": "1000",
                "NVIDIA_DAILY_REQUEST_LIMIT": "20",
            }
        )
        lease = manager.reserve_key("nvidia", 100, capability="translation")
        manager.mark_failure(lease, 401, "invalid api key")
        second = manager.reserve_key("nvidia", 100, capability="translation")
        self.assertEqual(second.key_index, 2, "an auth-locked dedicated key falls through to backup, same as normal rotation")

    def test_a_role_dedicated_to_a_different_capability_never_serves_as_fallback(self) -> None:
        # Only 2 keys, neither is "backup" or "*" -- if the translation key is unusable, there
        # must be NO fallback (the vision-only key must never silently serve translation traffic).
        manager = self.build_manager(
            {
                "NVIDIA_NIM_API_KEYS": "key-translate,key-vision",
                "NVIDIA_KEY_ROLES": "translation,vision_ocr",
                "NVIDIA_DAILY_TOKEN_LIMIT": "1000",
                "NVIDIA_DAILY_REQUEST_LIMIT": "20",
            }
        )
        lease = manager.reserve_key("nvidia", 100, capability="translation")
        manager.mark_failure(lease, 401, "invalid api key")
        with self.assertRaises((ApiQuotaExhausted, ApiProviderAuthLocked, ApiProviderUnavailable)):
            manager.reserve_key("nvidia", 100, capability="translation")

    def test_mismatched_role_count_is_ignored_not_misrouted(self) -> None:
        # A misconfiguration (roles list doesn't match key count) must fail open to unrestricted
        # behavior, not silently misroute traffic based on a malformed/truncated role list.
        manager = self.build_manager(
            {
                "NVIDIA_NIM_API_KEYS": "key-a,key-b,key-c",
                "NVIDIA_KEY_ROLES": "translation,vision_ocr",  # only 2 roles for 3 keys
                "NVIDIA_DAILY_TOKEN_LIMIT": "1000",
                "NVIDIA_DAILY_REQUEST_LIMIT": "20",
            }
        )
        lease = manager.reserve_key("nvidia", 100, capability="translation")
        self.assertEqual(lease.key_index, 1, "mismatched role count falls back to plain unrestricted rotation")


if __name__ == "__main__":
    unittest.main()
