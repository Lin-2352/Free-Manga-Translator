from __future__ import annotations

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


def test_corrupt_state_file_returns_none_not_empty_dict() -> None:
    """Regression test for Tier B #9: _read_state() used to be
    `except Exception: return {}` -- a transient file-lock or corrupt/partial
    write (antivirus scanner, file indexer, crash mid-write) would wipe EVERY
    provider's quota/usage counters and PERSIST that wipe. The fix must
    distinguish FileNotFoundError (legitimate cold start, {} is correct) from
    other read/parse errors (must return None: 'couldn't read, don't touch
    existing state')."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "api_quota_state.json"
        # Not FileNotFoundError -- the file exists but is not valid JSON.
        state_path.write_text("{not valid json at all", encoding="utf-8")

        manager = api_manager.ApiManager(state_path=state_path)
        result = manager._read_state()
        assert result is None, (
            f"a corrupt-but-present state file must yield None ('don't touch existing state'), got {result!r}"
        )

        missing_path = Path(tmp) / "does_not_exist.json"
        manager2 = api_manager.ApiManager(state_path=missing_path)
        result2 = manager2._read_state()
        assert result2 == {}, f"a genuinely missing state file (cold start) must yield {{}}, got {result2!r}"


def test_synced_preserves_in_memory_state_on_transient_read_failure() -> None:
    """The _synced() context manager refreshes self._state from disk before
    every state-touching operation. If that refresh transiently fails, it must
    NOT clobber the previously-loaded in-memory state with {}."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "api_quota_state.json"
        manager = api_manager.ApiManager(state_path=state_path)

        # Seed some in-memory state as if a previous successful cycle wrote it.
        manager._state = {"providers": {"gemini": {"keys": {"probe": {"tokensUsed": 12345}}}}}

        original_read_state = manager._read_state
        manager._read_state = lambda: None  # simulate a transient read failure
        try:
            with manager._synced():
                pass
        finally:
            manager._read_state = original_read_state

        assert manager._state.get("providers", {}).get("gemini", {}).get("keys", {}).get("probe", {}).get(
            "tokensUsed"
        ) == 12345, (
            f"a transient read failure during _synced() must preserve the prior in-memory state, "
            f"got: {manager._state!r}"
        )


def main() -> int:
    test_corrupt_state_file_returns_none_not_empty_dict()
    test_synced_preserves_in_memory_state_on_transient_read_failure()
    print("api_manager_read_state_transient_error_preserves_state=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
