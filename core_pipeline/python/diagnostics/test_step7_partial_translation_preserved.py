from __future__ import annotations

import sys
from pathlib import Path

_BOOTSTRAP_FILE = Path(__file__).resolve()
for _candidate in _BOOTSTRAP_FILE.parents:
    if (_candidate / "samples").exists() and (_candidate / "python").exists():
        _PROJECT_ROOT_FOR_IMPORTS = _candidate
        break
else:
    _PROJECT_ROOT_FOR_IMPORTS = _BOOTSTRAP_FILE.parents[2]
for _rel in ("python/common", "python/steps"):
    _path = str(_PROJECT_ROOT_FOR_IMPORTS / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)
del _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path

import run_step7_translate as step7
from api_manager import ApiQuotaExhausted


def test_partial_translations_survive_quota_exhaustion_instead_of_being_discarded() -> None:
    """Regression test for Tier A #2: the provider waterfall's tail `raise
    ApiQuotaExhausted` used to fire even after earlier providers in the same
    call had already produced accepted translations for some ids, discarding
    that already-paid-for work entirely (it would have to be re-translated,
    burning API budget again, on retry). The fix returns the accepted partial
    results instead of raising, as long as at least one id was translated."""
    items = [
        {"id": 1, "text": "こんにちは"},
        {"id": 2, "text": "さようなら"},
    ]

    original_provider_calls = dict(step7.PROVIDER_CALLS)
    original_disabled = dict(step7.API_PROVIDER_DISABLED)
    original_provider_order = step7._provider_order
    original_local_fallback = step7._local_fallback_enabled
    original_api_enabled = step7._api_translation_enabled
    original_valid = step7._valid_api_translation
    original_exhausted = step7.API_MANAGER.all_configured_providers_exhausted

    try:
        step7.API_PROVIDER_DISABLED.clear()
        step7._provider_order = lambda: ["provider_a", "provider_b"]
        step7._local_fallback_enabled = lambda: False
        step7._api_translation_enabled = lambda: True
        step7._valid_api_translation = lambda source, translated: bool(translated)

        def fake_provider_a(prompt):
            return {1: "hello"}, {"provider": "provider_a"}

        def fake_provider_b(prompt):
            raise ApiQuotaExhausted("provider_b: daily quota reached")

        step7.PROVIDER_CALLS.clear()
        step7.PROVIDER_CALLS.update({"provider_a": fake_provider_a, "provider_b": fake_provider_b})

        # First call (used by the early pre-loop check) says not-yet-exhausted,
        # matches real behavior where the exhaustion is only discovered once
        # provider_b's call actually raises inside the loop.
        _calls = {"n": 0}

        def fake_all_exhausted(providers):
            _calls["n"] += 1
            return _calls["n"] > 1

        step7.API_MANAGER.all_configured_providers_exhausted = fake_all_exhausted

        translations, report = step7._api_translate_items(items, "test_sample", "")

        assert translations == {1: "hello"}, (
            f"expected id 1's already-accepted translation to survive the later exhaustion, got {translations}"
        )
        assert 2 not in translations, "id 2 was never translated -- it should be 'missing', not fabricated"
        statuses = [a.get("status") for a in report.get("attempts", [])]
        assert "api_quota_exhausted_partial" in statuses, (
            f"expected an 'api_quota_exhausted_partial' status attempt recording the partial result, got {statuses}"
        )
    finally:
        step7.PROVIDER_CALLS.clear()
        step7.PROVIDER_CALLS.update(original_provider_calls)
        step7.API_PROVIDER_DISABLED.clear()
        step7.API_PROVIDER_DISABLED.update(original_disabled)
        step7._provider_order = original_provider_order
        step7._local_fallback_enabled = original_local_fallback
        step7._api_translation_enabled = original_api_enabled
        step7._valid_api_translation = original_valid
        step7.API_MANAGER.all_configured_providers_exhausted = original_exhausted


def test_fully_failed_page_still_raises() -> None:
    """When NOTHING was accepted before exhaustion, raising is still correct
    (nothing to lose, and callers rely on the exception for the fully-failed
    case) -- guards against an overly broad fix that never raises at all."""
    items = [{"id": 1, "text": "こんにちは"}]

    original_provider_calls = dict(step7.PROVIDER_CALLS)
    original_disabled = dict(step7.API_PROVIDER_DISABLED)
    original_provider_order = step7._provider_order
    original_local_fallback = step7._local_fallback_enabled
    original_api_enabled = step7._api_translation_enabled
    original_exhausted = step7.API_MANAGER.all_configured_providers_exhausted

    try:
        step7.API_PROVIDER_DISABLED.clear()
        step7._provider_order = lambda: ["provider_a"]
        step7._local_fallback_enabled = lambda: False
        step7._api_translation_enabled = lambda: True

        def fake_provider_a(prompt):
            raise ApiQuotaExhausted("provider_a: daily quota reached")

        step7.PROVIDER_CALLS.clear()
        step7.PROVIDER_CALLS.update({"provider_a": fake_provider_a})

        _calls = {"n": 0}

        def fake_all_exhausted(providers):
            _calls["n"] += 1
            return _calls["n"] > 1

        step7.API_MANAGER.all_configured_providers_exhausted = fake_all_exhausted

        raised = False
        try:
            step7._api_translate_items(items, "test_sample", "")
        except ApiQuotaExhausted:
            raised = True
        assert raised, "a page with zero accepted translations should still raise ApiQuotaExhausted"
    finally:
        step7.PROVIDER_CALLS.clear()
        step7.PROVIDER_CALLS.update(original_provider_calls)
        step7.API_PROVIDER_DISABLED.clear()
        step7.API_PROVIDER_DISABLED.update(original_disabled)
        step7._provider_order = original_provider_order
        step7._local_fallback_enabled = original_local_fallback
        step7._api_translation_enabled = original_api_enabled
        step7.API_MANAGER.all_configured_providers_exhausted = original_exhausted


def main() -> int:
    test_partial_translations_survive_quota_exhaustion_instead_of_being_discarded()
    test_fully_failed_page_still_raises()
    print("step7_partial_translation_preserved=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
