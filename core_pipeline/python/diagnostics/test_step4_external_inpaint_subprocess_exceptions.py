from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

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

import run_step4_inpaint


def _make_crop_args():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    return image, mask, 32, 32, 0, 0, 32, 32


def _run_with(monkey_subprocess_run, env_value: str = "fake_command {input} {output} {mask}"):
    original_env = os.environ.get("MANGA_INPAINT_COMMAND")
    original_run = subprocess.run
    try:
        os.environ["MANGA_INPAINT_COMMAND"] = env_value
        subprocess.run = monkey_subprocess_run
        return run_step4_inpaint._external_inpaint_command_local_crop(*_make_crop_args())
    finally:
        subprocess.run = original_run
        if original_env is None:
            os.environ.pop("MANGA_INPAINT_COMMAND", None)
        else:
            os.environ["MANGA_INPAINT_COMMAND"] = original_env


def test_timeout_expired_returns_false_not_raised() -> None:
    """Regression test for Tier B #6: this function's signature returns bool
    and every other failure path returns False -- but subprocess.TimeoutExpired
    used to propagate uncaught, killing the request mid-inpaint. Config-gated
    behind MANGA_INPAINT_COMMAND (unset by default), but still a real bug."""

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="fake_command", timeout=1.0)

    result = _run_with(fake_run)
    assert result is False, f"expected False on subprocess.TimeoutExpired, got {result!r}"


def test_os_error_returns_false_not_raised() -> None:
    """Same as above but for OSError (e.g. the external command binary doesn't
    exist / isn't executable)."""

    def fake_run(*args, **kwargs):
        raise OSError("fake: executable not found")

    result = _run_with(fake_run)
    assert result is False, f"expected False on OSError, got {result!r}"


def main() -> int:
    test_timeout_expired_returns_false_not_raised()
    test_os_error_returns_false_not_raised()
    print("step4_external_inpaint_subprocess_exceptions=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
