"""Regression test: when the pipeline scheduler's bounded acquire() wait times out
(SchedulerBusyError), the /translate route must surface HTTP 503 -- not be silently
folded into the generic 500 PIPELINE_ERROR path.
"""
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
for _rel in ("", "python/common", "python/runtime"):
    _path = str(_PROJECT_ROOT_FOR_IMPORTS / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)
del _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path

from fastapi import HTTPException

from backend_api.app import main
from backend_api.app.gpu_scheduler import SchedulerBusyError


ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB"
    "/6X4sZ8AAAAASUVORK5CYII="
)


def _raise_scheduler_busy(payload):
    raise SchedulerBusyError("Timed out after 60.0s waiting for a pipeline slot")


def test_scheduler_busy_returns_503() -> None:
    original = main.run_pipeline_payload
    main.run_pipeline_payload = _raise_scheduler_busy
    try:
        request = main.TranslateRequest(
            imageData=f"data:image/png;base64,{ONE_PIXEL_PNG}",
            sourceLanguage="ja",
            targetLanguage="en",
            qualityProfile="strict",
        )
        try:
            main.translate_image(request)
            raise AssertionError("expected HTTPException(503), request succeeded instead")
        except HTTPException as error:
            assert error.status_code == 503, f"expected 503, got {error.status_code}"
            assert error.detail.get("code") == "SCHEDULER_BUSY", error.detail
        print("scheduler_busy_returns_503=pass")
    finally:
        main.run_pipeline_payload = original


if __name__ == "__main__":
    test_scheduler_busy_returns_503()
