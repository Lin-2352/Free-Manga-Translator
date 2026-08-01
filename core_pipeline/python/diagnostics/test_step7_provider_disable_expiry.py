from __future__ import annotations

import sys
import time
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


def test_disabled_provider_expires_instead_of_staying_disabled_forever() -> None:
    """Regression test for Tier A #1: API_PROVIDER_DISABLED used to be a plain
    dict[str, str] -- once a provider was marked disabled (e.g. one transient
    ApiRateLimited exception), it stayed disabled for the rest of the process
    lifetime since step7 runs inside the long-lived backend server process.
    The fix stores (reason, expires_at) and _provider_disabled_reason() must
    treat a stale entry as no-longer-disabled and clear it."""
    original = dict(step7.API_PROVIDER_DISABLED)
    try:
        step7.API_PROVIDER_DISABLED.clear()

        # Mark disabled with a TTL that has already elapsed (simulates a
        # rate-limit that self-cleared on the provider's side after 60s).
        step7.API_PROVIDER_DISABLED["gemini"] = ("rate limited", time.monotonic() - 1)
        reason = step7._provider_disabled_reason("gemini")
        assert reason is None, (
            f"expected an expired disable entry to be treated as no-longer-disabled, got {reason!r} -- "
            "this is the old permanent-disable bug if it fails"
        )
        assert "gemini" not in step7.API_PROVIDER_DISABLED, (
            "expired entry should be cleared out of API_PROVIDER_DISABLED once observed as stale"
        )

        # Still within its window -> still disabled.
        step7._mark_provider_disabled("gemini", "rate limited", "rate_limited")
        reason = step7._provider_disabled_reason("gemini")
        assert reason == "rate limited", f"expected the fresh disable to still be active, got {reason!r}"
        assert "gemini" in step7.API_PROVIDER_DISABLED
    finally:
        step7.API_PROVIDER_DISABLED.clear()
        step7.API_PROVIDER_DISABLED.update(original)


def main() -> int:
    test_disabled_provider_expires_instead_of_staying_disabled_forever()
    print("step7_provider_disable_expiry=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
