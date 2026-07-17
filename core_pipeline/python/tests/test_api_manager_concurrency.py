from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_ROOT = PROJECT_ROOT / "python" / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from api_manager import ApiManager  # noqa: E402


class ApiManagerConcurrencyTests(unittest.TestCase):
    """ApiManager reads its state file once at __init__ and writes the full in-memory dict back
    on every mutation with no cross-process lock and no re-read-before-write. Two ApiManager
    instances in two different processes (the always-running backend plus a separate CLI/batch
    script, both pointed at the same runtime_samples/api_quota_state.json) each hold their own
    stale copy -- whichever writes last silently discards the other's concurrent reservations or
    key-lock updates. A single in-process threading.Lock (which IS present) cannot prevent this,
    since it only serializes threads sharing one Python process, not two separate processes.

    These tests use multiple ApiManager INSTANCES sharing one state file as the faithful minimal
    simulation of that (matches the technique used for the extension's SW-restart tests this same
    session) -- not threads, which would share the singleton lock and never reproduce the race.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.state_path = Path(self.temp_dir.name) / "quota_state.json"
        self.env = {
            "GEMINI_API_KEYS": "key-a,key-b",
            "GEMINI_DAILY_TOKEN_LIMIT": "10000",
            "GEMINI_DAILY_REQUEST_LIMIT": "200",
            "API_MANAGER_SOFT_CAP_RATIO": "0.8",
            # This project's real .env now sets LIMIT_SCOPE=key for every provider (each key gets
            # its own safe budget instead of sharing one pooled provider-wide budget) -- these
            # cross-process races must be proven safe under the config actually running in
            # production, not just the "provider" default these tests used to run under.
            "GEMINI_LIMIT_SCOPE": "key",
        }
        self.env_patch = patch.dict(os.environ, self.env, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.load_patch = patch.object(ApiManager, "_load_env", lambda self: None)
        self.load_patch.start()
        self.addCleanup(self.load_patch.stop)

    def new_manager(self) -> ApiManager:
        # A fresh ApiManager() call re-reads the state file at construction time, exactly like a
        # new process (or a fresh backend restart) would -- this is the "process boundary".
        return ApiManager(self.state_path)

    def read_key1_tokens_used(self) -> int:
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        provider_state = data.get("providers", {}).get("gemini", {})
        for key_state in provider_state.get("keys", {}).values():
            if key_state.get("keyIndex") == 1:
                return int(key_state.get("tokensUsed", 0))
        return 0

    def test_concurrent_processes_do_not_lose_each_others_reservations(self) -> None:
        # Process A and process B both start from the same (empty) on-disk state.
        manager_a = self.new_manager()
        manager_b = self.new_manager()

        lease_a = manager_a.reserve_key_index("gemini", 1, 300)
        manager_a.mark_success(lease_a)

        # Process B's in-memory state still reflects the empty file it read at construction --
        # this reservation must not silently overwrite A's already-persisted 300 tokens.
        lease_b = manager_b.reserve_key_index("gemini", 1, 500)
        manager_b.mark_success(lease_b)

        # A fresh, third instance reading the file after both writes is the ground truth: it must
        # see BOTH reservations (300 + 500 = 800), not just whichever process wrote last.
        manager_c = self.new_manager()
        status = manager_c.provider_status("gemini")
        key1 = next(k for k in status["keys"] if k["index"] == 1)
        used_tokens = int(status["safeTokenLimit"]) - int(key1["remainingTokens"]) if status["limitScope"] == "key" else None
        on_disk_tokens = self.read_key1_tokens_used()
        self.assertEqual(
            on_disk_tokens,
            800,
            f"process B's write must not silently discard process A's already-persisted reservation "
            f"(got {on_disk_tokens} tokens tracked on disk, expected 300 + 500 = 800)",
        )

    def test_concurrent_key_lock_from_one_process_is_not_erased_by_the_other(self) -> None:
        manager_a = self.new_manager()
        manager_b = self.new_manager()

        # Process A learns key 1 is dead (401) and locks it.
        lease_a = manager_a.reserve_key_index("gemini", 1, 50)
        manager_a.mark_failure(lease_a, 401, "invalid api key")

        # Process B, still holding its stale pre-lock view, reserves and succeeds on key 2 --
        # its write must not un-lock key 1 by omission (a full-dict overwrite from B's state,
        # which never knew about A's lock, would silently erase it).
        lease_b = manager_b.reserve_key_index("gemini", 2, 50)
        manager_b.mark_success(lease_b)

        manager_c = self.new_manager()
        status = manager_c.provider_status("gemini")
        key1 = next(k for k in status["keys"] if k["index"] == 1)
        self.assertEqual(
            key1["status"],
            "locked",
            "process B's unrelated write must not silently erase process A's key-1 auth lock",
        )

    def test_many_concurrent_processes_never_reserve_more_than_the_safe_limit(self) -> None:
        # 20 "processes" (fresh ApiManager instances, real OS threads so they genuinely race on
        # the same file handle rather than running sequentially) each try to reserve 100 tokens
        # against a 1000-token limit with an 0.8 safe ratio (safe budget = 800, i.e. at most 8
        # should succeed). This is the actual safety property the whole quota system exists for:
        # no combination of concurrent callers should ever push real usage past the safe cap.
        results: list[bool] = []
        results_lock = threading.Lock()

        def worker() -> None:
            manager = self.new_manager()
            try:
                lease = manager.reserve_key_index("gemini", 1, 100)
                manager.mark_success(lease)
                with results_lock:
                    results.append(True)
            except Exception:
                with results_lock:
                    results.append(False)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successes = sum(1 for r in results if r)
        on_disk_tokens = self.read_key1_tokens_used()
        self.assertLessEqual(
            on_disk_tokens,
            800,
            f"tracked usage ({on_disk_tokens}) must never exceed the safe cap (800) no matter how many "
            f"processes race to reserve concurrently -- a lost update here means real overspend risk",
        )
        self.assertLessEqual(successes, 8, f"at most 8 reservations of 100 tokens should fit in an 800-token safe budget, got {successes}")
        self.assertEqual(on_disk_tokens, successes * 100, "tracked usage must exactly match the number of reservations that actually succeeded")


if __name__ == "__main__":
    unittest.main()
