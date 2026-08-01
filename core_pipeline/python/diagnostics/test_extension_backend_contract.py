from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any

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


def _fake_pipeline_payload(payload: dict[str, Any]) -> dict[str, Any]:
    assert payload["imageData"].startswith("data:image/png;base64,")
    assert payload["sourceLanguage"] in {"ja", "ko", "zh"}
    assert payload["targetLanguage"] == "en"
    assert payload["qualityProfile"] == "strict"
    return {
        "sampleName": "contract_runtime_sample",
        "language": payload["sourceLanguage"],
        "translatedImageDataUrl": f"data:image/png;base64,{ONE_PIXEL_PNG}",
        "report": {
            "pipeline": "local-8-stage",
            "stageSequence": [{"step": step} for step in range(1, 9)],
            "layoutConstraints": 1,
            "translations": 1,
            "renderedRegions": 1,
            "typesetStatuses": ["fit"],
        },
        "artifacts": {"step8_output": "runtime_samples/extension/contract/step_8_typeset/final_output.png"},
    }


def main_test() -> int:
    original = main.run_pipeline_payload
    main.run_pipeline_payload = _fake_pipeline_payload
    try:
        health = main.health()
        assert health.ok is True

        payload = {
            "imageData": f"data:image/png;base64,{ONE_PIXEL_PNG}",
            "sourceLanguage": "ja",
            "targetLanguage": "en",
            "qualityProfile": "strict",
            "requestedOutput": "translatedImageDataUrl",
            "metadata": {"source": "extension-contract-test"},
        }
        request = main.TranslateRequest(**payload)
        response = main.translate_image(request)
        data = response.model_dump() if hasattr(response, "model_dump") else response.dict()
        assert data["status"] == "pass"
        assert data["translatedImageDataUrl"].startswith("data:image/png;base64,")
        assert data["imageDataUrl"] == data["translatedImageDataUrl"]
        assert data["report"]["pipeline"] == "local-8-stage"
        assert len(data["report"]["stageSequence"]) == 8

        print("extension_backend_contract=pass")
        return 0
    finally:
        main.run_pipeline_payload = original


if __name__ == "__main__":
    raise SystemExit(main_test())
