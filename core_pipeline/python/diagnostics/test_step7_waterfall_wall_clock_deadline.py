from __future__ import annotations

import os
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


def test_waterfall_gives_up_after_wall_clock_budget_exceeded() -> None:
    """Regression test for Tier B #10: the provider-fallback loop had no global
    wall-clock deadline. Each provider is individually bounded, but the worst
    case across every configured provider could still add up to tens of
    minutes on a single page while holding a GPU pipeline slot open. The fix
    adds a time.monotonic() deadline checked at the top of each loop
    iteration; this test forces the deadline to already be in the past (via
    the env override) so the very first iteration must bail out instead of
    calling any provider at all."""
    items = [{"id": 1, "text": "こんにちは"}]

    original_provider_calls = dict(step7.PROVIDER_CALLS)
    original_disabled = dict(step7.API_PROVIDER_DISABLED)
    original_provider_order = step7._provider_order
    original_local_fallback = step7._local_fallback_enabled
    original_api_enabled = step7._api_translation_enabled
    original_exhausted = step7.API_MANAGER.all_configured_providers_exhausted
    original_env = os.environ.get("API_TRANSLATION_WATERFALL_DEADLINE_SECONDS")

    calls_made = {"n": 0}

    try:
        # 60 is the minimum floor enforced by max(60, ...) in the implementation
        # -- use a fake "now" comparison instead by forcing all_configured_
        # providers_exhausted to False (so we reach the loop) and instead
        # verify the loop's own deadline math using a tiny sleep-free approach:
        # patch time.monotonic itself is fragile across modules, so instead
        # verify indirectly: the deadline is `time.monotonic() + budget`, and
        # budget floors at 60s -- we can't easily force it to 0 without
        # patching time.monotonic. Patch it directly on the `time` module the
        # step7 module imported.
        os.environ["API_TRANSLATION_WATERFALL_DEADLINE_SECONDS"] = "60"

        step7.API_PROVIDER_DISABLED.clear()
        step7._provider_order = lambda: ["provider_a"]
        step7._local_fallback_enabled = lambda: False
        step7._api_translation_enabled = lambda: True
        step7.API_MANAGER.all_configured_providers_exhausted = lambda providers: False

        def fake_provider_a(prompt):
            calls_made["n"] += 1
            return {1: "hello"}, {"provider": "provider_a"}

        step7.PROVIDER_CALLS.clear()
        step7.PROVIDER_CALLS.update({"provider_a": fake_provider_a})

        original_monotonic = step7.time.monotonic
        # First call (computing the deadline) returns a real, small value;
        # every call after that (the in-loop check) returns something far in
        # the future, simulating the budget having already elapsed.
        call_count = {"n": 0}

        def fake_monotonic():
            call_count["n"] += 1
            if call_count["n"] == 1:
                return 1000.0
            return 1000.0 + 999999.0

        step7.time.monotonic = fake_monotonic
        try:
            translations, report = step7._api_translate_items(items, "test_sample", "")
        finally:
            step7.time.monotonic = original_monotonic

        assert calls_made["n"] == 0, (
            f"expected the loop to bail out at the deadline check before calling any provider, "
            f"but the provider was called {calls_made['n']} time(s)"
        )
        statuses = [a.get("status") for a in report.get("attempts", [])]
        assert "waterfall_deadline_exceeded" in statuses, (
            f"expected a 'waterfall_deadline_exceeded' status attempt, got {statuses}"
        )
    finally:
        step7.PROVIDER_CALLS.clear()
        step7.PROVIDER_CALLS.update(original_provider_calls)
        step7.API_PROVIDER_DISABLED.clear()
        step7.API_PROVIDER_DISABLED.update(original_disabled)
        step7._provider_order = original_provider_order
        step7._local_fallback_enabled = original_local_fallback
        step7._api_translation_enabled = original_api_enabled
        step7.API_MANAGER.all_configured_providers_exhausted = original_exhausted
        if original_env is None:
            os.environ.pop("API_TRANSLATION_WATERFALL_DEADLINE_SECONDS", None)
        else:
            os.environ["API_TRANSLATION_WATERFALL_DEADLINE_SECONDS"] = original_env


def main() -> int:
    test_waterfall_gives_up_after_wall_clock_budget_exceeded()
    print("step7_waterfall_wall_clock_deadline=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
