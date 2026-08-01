"""Regression test: diagnostic_logger's write_diagnostic_event must rotate the day's log
file once it exceeds a size threshold, instead of growing it unbounded forever.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_BOOTSTRAP_FILE = Path(__file__).resolve()
for _candidate in _BOOTSTRAP_FILE.parents:
    if (_candidate / "samples").exists() and (_candidate / "python").exists():
        _PROJECT_ROOT_FOR_IMPORTS = _candidate
        break
else:
    _PROJECT_ROOT_FOR_IMPORTS = _BOOTSTRAP_FILE.parents[2]
for _rel in ("python/common",):
    _path = str(_PROJECT_ROOT_FOR_IMPORTS / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)
del _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path

import diagnostic_logger  # noqa: E402


def test_diagnostics_log_rotation() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="fmt_diag_rotation_"))
    old_dir_env = os.environ.get("FMT_DIAGNOSTICS_DIR")
    old_bytes_env = os.environ.get("FMT_DIAGNOSTICS_MAX_BYTES")
    os.environ["FMT_DIAGNOSTICS_DIR"] = str(tmp_dir)
    os.environ["FMT_DIAGNOSTICS_MAX_BYTES"] = "2000"  # tiny threshold so a few events rotate it
    try:
        # write_diagnostic_event stamps the file by UTC "now", so resolve the same day here
        # (matters near local-midnight/UTC-day boundaries, otherwise unrelated to rotation).
        path = diagnostic_logger.diagnostics_log_path(datetime.now(timezone.utc))
        for i in range(200):
            diagnostic_logger.write_diagnostic_event(
                "test.rotation", {"i": i, "padding": "x" * 50}, source="test"
            )
        assert path.exists(), "current log file should exist after writes"
        assert path.stat().st_size < 2000 + 500, (
            f"log file grew past the rotation threshold without rotating: {path.stat().st_size} bytes"
        )
        backups = list(tmp_dir.glob(f"{path.name}.*"))
        assert backups, "expected at least one rotated backup file, found none"
        print("diagnostics_log_rotation=pass")
    finally:
        if old_dir_env is None:
            os.environ.pop("FMT_DIAGNOSTICS_DIR", None)
        else:
            os.environ["FMT_DIAGNOSTICS_DIR"] = old_dir_env
        if old_bytes_env is None:
            os.environ.pop("FMT_DIAGNOSTICS_MAX_BYTES", None)
        else:
            os.environ["FMT_DIAGNOSTICS_MAX_BYTES"] = old_bytes_env
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_diagnostics_log_rotation()
