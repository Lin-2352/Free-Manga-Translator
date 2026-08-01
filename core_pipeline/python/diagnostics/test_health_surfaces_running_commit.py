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
for _rel in ("python/common", "python/steps", "python/runtime", "backend_api/app"):
    _path = str(_PROJECT_ROOT_FOR_IMPORTS / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)
del _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend_api"))

import subprocess


def test_health_response_includes_a_real_git_commit() -> None:
    """Regression test for the stale-backend trap: a running server's /v1/health must
    surface which commit it is actually running, so a source edit with no restart is
    visible instead of silently inferred (see LOCAL_USER_MANUAL.md)."""
    from app.main import RUNNING_COMMIT

    expected = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()

    assert RUNNING_COMMIT, "RUNNING_COMMIT resolved to None/empty in a real git checkout"
    assert RUNNING_COMMIT == expected, (
        f"RUNNING_COMMIT ({RUNNING_COMMIT!r}) does not match `git rev-parse --short HEAD` "
        f"({expected!r}) -- health's commit field would mislead about what's actually running"
    )


def test_health_endpoint_response_model_carries_commit_field() -> None:
    from app.schemas import HealthResponse

    response = HealthResponse(ok=True, service="x", mode="y", version="z", commit="abc1234")
    assert response.commit == "abc1234"
    # Optional/nullable: a non-git deployment (e.g. a zipped copy with no .git) must not crash.
    response_no_commit = HealthResponse(ok=True, service="x", mode="y", version="z")
    assert response_no_commit.commit is None


def main() -> int:
    test_health_response_includes_a_real_git_commit()
    test_health_endpoint_response_model_carries_commit_field()
    print("health_surfaces_running_commit=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
