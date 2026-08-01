from __future__ import annotations

import os
import sys
import tempfile
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

import api_manager

# Any real GEMINI_API_KEY_1.._N (or GEMINI_API_KEYS) already loaded from a real .env
# on this machine would give reserve_key() multiple real keys to rotate across --
# lease_a and lease_b could then land on two DIFFERENT keys, so checking lease_b's
# key_hash for lease_a's charge would spuriously read 0 (correctly -- lease_a never
# touched that key) and look like a refund bug that isn't one. Both tests below need
# a single-key environment to deterministically force lease_a and lease_b onto the
# SAME key, so this isolates against every numbered/plural Gemini key variant.
_GEMINI_ENV_VARS_TO_ISOLATE = ["GEMINI_API_KEY", "GEMINI_API_KEYS"] + [
    f"GEMINI_API_KEY_{i}" for i in range(1, 51)
]


def _isolate_gemini_env() -> dict[str, str | None]:
    saved = {name: os.environ.pop(name, None) for name in _GEMINI_ENV_VARS_TO_ISOLATE}
    return saved


def _restore_gemini_env(saved: dict[str, str | None]) -> None:
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_failed_call_refunds_its_reserved_tokens() -> None:
    """Regression test for Tier A #4: reserve_key() pre-charges the daily
    token/request budget with an estimate before the call is even made. If the
    call ultimately fails, the reservation used to never be refunded --
    burning daily budget for zero delivered translation, and enough failures
    alone could exhaust the daily quota with nothing translated. The fix
    refunds the reservation in mark_failure()."""
    saved_env = _isolate_gemini_env()
    try:
        os.environ["GEMINI_API_KEY"] = "test_refund_probe_key"
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "api_quota_state.json"
            manager = api_manager.ApiManager(state_path=state_path)

            lease = manager.reserve_key("gemini", estimated_tokens=5000, capability="translation")

            provider_state = manager._state["providers"]["gemini"]
            key_state = provider_state["keys"][lease.key_hash]
            assert key_state["tokensUsed"] == 5000, (
                f"expected the reservation to pre-charge 5000 tokens, got {key_state['tokensUsed']}"
            )
            assert key_state["requestsUsed"] == 1

            # A generic transient failure (no status_code) -- not terminal, not
            # an auth block -- must still refund the reservation.
            manager.mark_failure(lease, status_code=None, error="transient network error")

            provider_state = manager._state["providers"]["gemini"]
            key_state = provider_state["keys"][lease.key_hash]
            assert key_state["tokensUsed"] == 0, (
                f"expected the failed call's reserved tokens to be refunded back to 0, "
                f"got {key_state['tokensUsed']} -- this is the bug if nonzero"
            )
            assert key_state["requestsUsed"] == 0, (
                f"expected the failed call's reserved request slot to be refunded, got {key_state['requestsUsed']}"
            )
    finally:
        _restore_gemini_env(saved_env)


def test_refund_does_not_go_negative_across_multiple_reservations() -> None:
    """A second, independent reservation's tokens must not be clobbered by
    refunding more than what this particular lease reserved."""
    saved_env = _isolate_gemini_env()
    try:
        os.environ["GEMINI_API_KEY"] = "test_refund_probe_key_2"
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "api_quota_state.json"
            manager = api_manager.ApiManager(state_path=state_path)

            lease_a = manager.reserve_key("gemini", estimated_tokens=3000, capability="translation")
            manager.mark_success(lease_a, response_payload=None)

            lease_b = manager.reserve_key("gemini", estimated_tokens=2000, capability="translation")
            manager.mark_failure(lease_b, status_code=None, error="transient")

            provider_state = manager._state["providers"]["gemini"]
            key_state = provider_state["keys"][lease_b.key_hash]
            assert key_state["tokensUsed"] == 3000, (
                f"expected lease_a's successful 3000 tokens to remain charged after lease_b's "
                f"failed 2000 tokens were refunded, got {key_state['tokensUsed']}"
            )
    finally:
        _restore_gemini_env(saved_env)


def main() -> int:
    test_failed_call_refunds_its_reserved_tokens()
    test_refund_does_not_go_negative_across_multiple_reservations()
    print("api_manager_token_refund_on_failure=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
