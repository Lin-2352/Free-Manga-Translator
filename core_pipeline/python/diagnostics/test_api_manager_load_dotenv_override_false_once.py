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


def test_load_env_calls_load_dotenv_with_override_false_and_only_once_per_process() -> None:
    """Regression test for Tier B #11: _load_env() used to call
    load_dotenv(env_path, override=True) on every _csv_env() call (the actual
    key-lookup hot path calls _load_env() every time). override=True made a
    real operator/CI-set environment variable get stomped back to the .env
    file's value on every single call, and there was no once-per-process
    guard, so os.environ was mutated repeatedly with no lock around it. The
    fix: override=False, plus a once-per-process class-level guard flag."""
    original_load_dotenv = api_manager.load_dotenv
    original_attempted = api_manager.ApiManager._DOTENV_LOADED_ONCE
    calls: list[dict] = []

    def fake_load_dotenv(path, override=None):
        calls.append({"path": path, "override": override})

    try:
        api_manager.load_dotenv = fake_load_dotenv
        api_manager.ApiManager._DOTENV_LOADED_ONCE = False

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("PROBE_VAR=from_dotenv\n", encoding="utf-8")

            manager = api_manager.ApiManager(state_path=Path(tmp) / "state.json")
            manager._env_path = None
            import os as _os

            _os.environ["FMT_ENV_FILE"] = str(env_path)
            try:
                manager._load_env()
                manager._load_env()
                manager._load_env()
            finally:
                _os.environ.pop("FMT_ENV_FILE", None)

        assert len(calls) == 1, (
            f"expected load_dotenv() to be called exactly once per process across 3 _load_env() "
            f"calls, got {len(calls)} calls: {calls}"
        )
        assert calls[0]["override"] is False, (
            f"expected override=False (operator/CI env vars must win over the .env file), "
            f"got override={calls[0]['override']!r}"
        )
    finally:
        api_manager.load_dotenv = original_load_dotenv
        api_manager.ApiManager._DOTENV_LOADED_ONCE = original_attempted


def main() -> int:
    test_load_env_calls_load_dotenv_with_override_false_and_only_once_per_process()
    print("api_manager_load_dotenv_override_false_once=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
