"""Regression test: /v1/batch must dispatch through a shared, bounded, module-level
executor rather than creating a fresh ThreadPoolExecutor per request (previously
`with ThreadPoolExecutor(...) as pool` inside the route handler -- a fresh pool every call).
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
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

from backend_api.app import main


ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB"
    "/6X4sZ8AAAAASUVORK5CYII="
)


def test_batch_shared_executor() -> None:
    assert isinstance(main._BATCH_EXECUTOR, ThreadPoolExecutor), "expected a module-level shared executor"

    original = main.run_pipeline_payload

    def _fake_pipeline_payload(payload):
        return {
            "sampleName": "batch_test_sample",
            "language": payload["sourceLanguage"],
            "translatedImageDataUrl": f"data:image/png;base64,{ONE_PIXEL_PNG}",
            "report": {"pipeline": "local-8-stage"},
            "artifacts": {},
        }

    main.run_pipeline_payload = _fake_pipeline_payload
    try:
        executor_before = main._BATCH_EXECUTOR
        request = main.BatchRequest(
            images=[
                main.TranslateRequest(imageData=f"data:image/png;base64,{ONE_PIXEL_PNG}")
                for _ in range(3)
            ]
        )
        response = main.translate_batch(request)
        assert response.status == "pass"
        assert len(response.results) == 3
        # The handler must not have replaced/shut down the shared executor.
        assert main._BATCH_EXECUTOR is executor_before
        assert not main._BATCH_EXECUTOR._shutdown

        # A second call must reuse the same executor instance (no per-request recreation).
        response2 = main.translate_batch(request)
        assert len(response2.results) == 3
        assert main._BATCH_EXECUTOR is executor_before

        print("batch_shared_executor=pass")
    finally:
        main.run_pipeline_payload = original


if __name__ == "__main__":
    test_batch_shared_executor()
