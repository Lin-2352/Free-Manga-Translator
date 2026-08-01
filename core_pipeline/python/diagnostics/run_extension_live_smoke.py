from __future__ import annotations

import argparse
import base64
import json
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

from backend_api.app.main import TranslateRequest, translate_image
from pipeline_paths import PROJECT_ROOT


def _data_url_for_image(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _write_data_url(path: Path, data_url: str) -> None:
    if "," not in data_url:
        raise ValueError("Response image is not a data URL")
    encoded = data_url.split(",", 1)[1]
    path.write_bytes(base64.b64decode(encoded))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one live extension/API/pipeline smoke test.")
    parser.add_argument(
        "--image",
        default=str(PROJECT_ROOT / "samples" / "sample1" / "sample.jpg"),
        help="Input manga image path.",
    )
    parser.add_argument("--source-language", default="ja", choices=["ja", "ko", "zh"])
    parser.add_argument("--endpoint", default="/v1/translate-image")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if args.endpoint not in {"/translate", "/v1/translate-image", "/v1/translate-snapshot"}:
        raise ValueError(f"Unsupported local endpoint alias for direct smoke test: {args.endpoint}")

    response = translate_image(
        TranslateRequest(
            imageData=_data_url_for_image(image_path),
            sourceLanguage=args.source_language,
            targetLanguage="en",
            qualityProfile="strict",
            requestedOutput="translatedImageDataUrl",
            metadata={"source": "diagnostic-live-smoke", "inputPath": str(image_path)},
        )
    )

    payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()
    report = payload.get("report") or {}
    stage_sequence = report.get("stageSequence") or []
    if payload.get("status") != "pass":
        raise RuntimeError(f"Unexpected status: {payload.get('status')}")
    if len(stage_sequence) != 8:
        raise RuntimeError(f"Expected 8 pipeline stages, got {len(stage_sequence)}")
    if not str(payload.get("translatedImageDataUrl", "")).startswith("data:image/"):
        raise RuntimeError("Missing translated image data URL")

    out_dir = PROJECT_ROOT / "quality_reports" / "extension_runtime"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "live_smoke_report.json"
    output_path = out_dir / "live_smoke_output.png"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_data_url(output_path, payload["translatedImageDataUrl"])

    print("extension_live_smoke=pass")
    print(f"input={image_path}")
    print(f"report={report_path}")
    print(f"output={output_path}")
    print(f"layoutConstraints={report.get('layoutConstraints')}")
    print(f"translations={report.get('translations')}")
    print(f"renderedRegions={report.get('renderedRegions')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
