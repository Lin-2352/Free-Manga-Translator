"""
Local HTTP bridge for the Chrome extension.

The browser extension cannot run ONNX/PyTorch/LaMa models inside Chrome. This
server accepts a base64 image from the extension, runs the existing local
pipeline on a runtime sample folder, and returns the Step 8 output as a data URL.
"""
from __future__ import annotations

                                   
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_BOOTSTRAP_FILE = _BootstrapPath(__file__).resolve()
for _candidate in _BOOTSTRAP_FILE.parents:
    if (_candidate / "samples").exists() and (_candidate / "python").exists():
        _PROJECT_ROOT_FOR_IMPORTS = _candidate
        break
else:
    _PROJECT_ROOT_FOR_IMPORTS = _BOOTSTRAP_FILE.parents[2]
for _rel in (
    "python/common",
    "python/steps",
    "python/validation",
    "python/runtime",
    "python/downloaders",
    "python/reference",
    "python/diagnostics",
):
    _path = str(_PROJECT_ROOT_FOR_IMPORTS / _rel)
    if _path not in _bootstrap_sys.path:
        _bootstrap_sys.path.insert(0, _path)
del _BootstrapPath, _bootstrap_sys, _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path
                                       

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline_paths import EXTENSION_RUNTIME_ROOT, PROJECT_ROOT


SAMPLES_ROOT = EXTENSION_RUNTIME_ROOT
RUNTIME_MANIFEST_ROOT = PROJECT_ROOT / "quality_reports" / "extension_runtime"
RUNTIME_CACHE_VERSION = "extension-runtime-cache-v9-manga-cleaner-device-overlay"
RUNTIME_CACHE_META = "runtime_cache_meta.json"
OUTPUT_FOLDERS = [
    "step_1_detect",
    "step_4_final",
    "step_5_ocr",
    "step_6_layout",
    "step_7_translate",
    "step_8_typeset",
]
STAGE_SEQUENCE = [
    {"step": 1, "name": "text detection", "artifact": "step_1_detect/detections.json"},
    {"step": 2, "name": "bubble and semantic routing", "artifact": "step_1_detect/semantic_detections.json"},
    {"step": 3, "name": "text grouping and cleanup mask planning", "artifact": "step_5_ocr/ocr_results.json"},
    {"step": 4, "name": "context-aware inpainting", "artifact": "step_4_final/inpainted_result.jpg"},
    {"step": 5, "name": "OCR consolidation", "artifact": "step_5_ocr/ocr_results.json"},
    {"step": 6, "name": "layout constraints", "artifact": "step_6_layout/layout_constraints.json"},
    {"step": 7, "name": "LLM translation", "artifact": "step_7_translate/translation_results.json"},
    {"step": 8, "name": "typeset final image", "artifact": "step_8_typeset/final_output.png"},
]
PIPELINE_LOCK = threading.Lock()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


def _decode_image_data(payload: dict[str, Any]) -> bytes:
    image_data = payload.get("imageData") or payload.get("base64Data")
    if not isinstance(image_data, str) or not image_data:
        raise ValueError("Missing imageData/base64Data")
    if "," in image_data and image_data.startswith("data:"):
        image_data = image_data.split(",", 1)[1]
    return base64.b64decode(image_data, validate=False)


def _normalize_language_hint(value: Any) -> str:
    normalized = str(value or "ja").strip().lower()
    if normalized in {"zh", "ch", "chi", "chinese", "cn", "tw"}:
        return "zh"
    if normalized in {"ko", "kor", "korean", "kr"}:
        return "ko"
    return "ja"


def _clear_runtime_outputs(sample_path: Path) -> None:
    for folder in OUTPUT_FOLDERS:
        target = sample_path / folder
        if target.exists():
            shutil.rmtree(target)
    meta_path = sample_path / RUNTIME_CACHE_META
    if meta_path.exists():
        meta_path.unlink()


def _runtime_cache_meta_path(sample_name: str) -> Path:
    return SAMPLES_ROOT / sample_name / RUNTIME_CACHE_META


def _has_reusable_runtime_output(sample_name: str, language: str) -> bool:
    sample_path = SAMPLES_ROOT / sample_name
    meta_path = _runtime_cache_meta_path(sample_name)
    output_path = sample_path / "step_8_typeset" / "final_output.png"
    required = [
        sample_path / "step_6_layout" / "layout_constraints.json",
        sample_path / "step_7_translate" / "translation_results.json",
        sample_path / "step_8_typeset" / "typeset_report.json",
        output_path,
        meta_path,
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        meta.get("cacheVersion") == RUNTIME_CACHE_VERSION
        and meta.get("language") == _normalize_language_hint(language)
        and meta.get("status") == "pass"
    )


def _write_runtime_cache_meta(sample_name: str, language: str, report: dict[str, Any]) -> None:
    _runtime_cache_meta_path(sample_name).write_text(
        json.dumps(
            {
                "cacheVersion": RUNTIME_CACHE_VERSION,
                "language": _normalize_language_hint(language),
                "status": "pass",
                "layoutConstraints": report.get("layoutConstraints"),
                "translations": report.get("translations"),
                "renderedRegions": report.get("renderedRegions"),
                "totalSeconds": report.get("totalSeconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_runtime_sample(image_bytes: bytes, language: str, preserve_reusable_output: bool = True) -> tuple[str, Path]:
    digest = hashlib.sha1(image_bytes).hexdigest()[:14]
    language = _normalize_language_hint(language)
    sample_name = f"runtime_{language}_{digest}"
    sample_path = SAMPLES_ROOT / sample_name
    sample_path.mkdir(parents=True, exist_ok=True)

    input_path = sample_path / "input.jpg"
    source_path = sample_path / "source_upload.bin"
    source_path.write_bytes(image_bytes)
    with Image.open(source_path) as image:
        image.convert("RGB").save(input_path, quality=97)

    if not preserve_reusable_output or not _has_reusable_runtime_output(sample_name, language):
        _clear_runtime_outputs(sample_path)
    return sample_name, input_path


def _write_runtime_manifest(sample_name: str, language: str) -> Path:
    RUNTIME_MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = RUNTIME_MANIFEST_ROOT / f"{sample_name}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_for": "extension runtime safe local pipeline",
                "samples": [
                    {
                        "sample_name": sample_name,
                        "input_file": "input.jpg",
                        "language": _normalize_language_hint(language),
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _patch_sample_maps(sample_map: dict[str, str]) -> None:
    import ml_region_lib
    import run_step4_inpaint
    import run_step5_ocr
    import run_step6_layout
    import run_step7_translate
    import run_step8_typeset

    for module in [
        ml_region_lib,
        run_step4_inpaint,
        run_step5_ocr,
        run_step6_layout,
        run_step7_translate,
        run_step8_typeset,
    ]:
        module.SAMPLE_MAP = sample_map


def _collect_runtime_report(
    sample_name: str,
    language: str,
    stage_timings: list[dict[str, Any]] | None = None,
    total_seconds: float | None = None,
    reused_output: bool = False,
) -> dict[str, Any]:
    sample_path = SAMPLES_ROOT / sample_name
    stage_artifacts = {
        f"step_{stage['step']}": str(sample_path / stage["artifact"])
        for stage in STAGE_SEQUENCE
    }
    missing_artifacts = [
        f"step_{stage['step']}"
        for stage in STAGE_SEQUENCE
        if not (sample_path / stage["artifact"]).exists()
    ]
    layout_path = sample_path / "step_6_layout" / "layout_constraints.json"
    rejected_layout_path = sample_path / "step_6_layout" / "rejected_layout_items.json"
    translation_path = sample_path / "step_7_translate" / "translation_results.json"
    typeset_report_path = sample_path / "step_8_typeset" / "typeset_report.json"

    layout = json.loads(layout_path.read_text(encoding="utf-8")) if layout_path.exists() else []
    rejected_layout = json.loads(rejected_layout_path.read_text(encoding="utf-8")) if rejected_layout_path.exists() else []
    translations = json.loads(translation_path.read_text(encoding="utf-8")) if translation_path.exists() else []
    typeset_report = json.loads(typeset_report_path.read_text(encoding="utf-8")) if typeset_report_path.exists() else []
    layout_ids = {
        int(item["id"])
        for item in layout
        if isinstance(item, dict) and "id" in item
    }
    renderable_translations = [
        item for item in translations
        if int(item.get("id", -1)) in layout_ids
    ]
    placeholder_count = sum(
        1 for item in renderable_translations
        if str(item.get("en_text", "")).strip().startswith("[TL:")
    )
    return {
        "sampleName": sample_name,
        "sourceLanguage": _normalize_language_hint(language),
        "pipeline": "local-8-stage",
        "stageSequence": STAGE_SEQUENCE,
        "stageArtifacts": stage_artifacts,
        "missingStageArtifacts": missing_artifacts,
        "layoutConstraints": len(layout),
        "rejectedLayoutItems": len(rejected_layout),
        "rawTranslations": len(translations),
        "translations": len(renderable_translations),
        "placeholderTranslations": placeholder_count,
        "renderedRegions": len(typeset_report),
        "typesetStatuses": [item.get("status", "unknown") for item in typeset_report],
        "stageTimings": stage_timings or [],
        "totalSeconds": round(total_seconds or 0, 3),
        "runtimeOutputCache": "hit" if reused_output else "miss",
    }


def _runtime_report_warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    review_statuses = {"fallback_clipped", "emergency", "clipped", "missing", "empty"}
    observed_review_statuses = sorted(
        {
            str(status)
            for status in report.get("typesetStatuses", [])
            if str(status) in review_statuses
        }
    )
    if observed_review_statuses:
        warnings.append(f"typeset_review_statuses={','.join(observed_review_statuses)}")
    if int(report.get("translations", 0)) < int(report.get("layoutConstraints", 0)):
        warnings.append(
            f"translations_below_layout_constraints="
            f"{report.get('translations', 0)}/{report.get('layoutConstraints', 0)}"
        )
    if int(report.get("renderedRegions", 0)) < int(report.get("translations", 0)):
        warnings.append(
            f"rendered_regions_below_translations="
            f"{report.get('renderedRegions', 0)}/{report.get('translations', 0)}"
        )
    return warnings


def _annotate_runtime_report_safety(report: dict[str, Any]) -> dict[str, Any]:
    warnings = _runtime_report_warnings(report)
    report["reviewWarnings"] = warnings
    report["outputSafety"] = "review" if warnings else "pass"
    return report


def _assert_runtime_report_safe(report: dict[str, Any]) -> None:
    critical_errors: list[str] = []
    if report.get("missingStageArtifacts"):
        critical_errors.append(f"missing_stage_artifacts={report['missingStageArtifacts']}")
    if int(report.get("placeholderTranslations", 0)):
        critical_errors.append(f"placeholder_translations={report['placeholderTranslations']}")
    if not int(report.get("renderedRegions", 0)):
        critical_errors.append("no_rendered_regions")

    _annotate_runtime_report_safety(report)
    if critical_errors:
        raise RuntimeError(f"Local pipeline failed before producing a usable output: {critical_errors}; report={report}")


def _run_runtime_pipeline(sample_name: str, language: str) -> dict[str, Any]:
    total_started = time.perf_counter()
    os.environ["PIPELINE_SAMPLES_ROOT"] = str(SAMPLES_ROOT)
    os.environ["LOCAL_NLLB_TRANSLATION"] = "1"
    import run_step4_inpaint
    import run_step5_ocr
    import run_step6_layout
    import run_step7_translate
    import run_step8_typeset

    sample_map = {sample_name: "input.jpg"}
    _patch_sample_maps(sample_map)

    _write_runtime_manifest(sample_name, language)

    stage_timings: list[dict[str, Any]] = []

    def run_stage(label: str, callback: Any) -> None:
        started = time.perf_counter()
        print(f"[runtime] {sample_name} {label}: start", flush=True)
        callback()
        elapsed = time.perf_counter() - started
        stage_timings.append({"stage": label, "seconds": round(elapsed, 3)})
        print(f"[runtime] {sample_name} {label}: done in {elapsed:.2f}s", flush=True)

    run_stage("step5_ocr", run_step5_ocr.run_step5_ocr)
    run_stage("step6_layout", run_step6_layout.run_step6_layout)
    run_stage("step7_translate", run_step7_translate.run_step7_translate)
    run_stage("step4_inpaint", run_step4_inpaint.run_step4_inpaint)
    run_stage("step8_typeset", run_step8_typeset.run_step8_typeset)

    report = _collect_runtime_report(
        sample_name,
        language,
        stage_timings=stage_timings,
        total_seconds=time.perf_counter() - total_started,
    )
    _assert_runtime_report_safe(report)
    _write_runtime_cache_meta(sample_name, language, report)
    print(f"[runtime] {sample_name} total: {report['totalSeconds']:.2f}s", flush=True)
    return report


def _read_output_data_url(sample_name: str) -> str:
    output_path = SAMPLES_ROOT / sample_name / "step_8_typeset" / "final_output.png"
    if not output_path.exists():
        raise FileNotFoundError(f"Missing Step 8 output: {output_path}")
    mime = "image/png"
    data = base64.b64encode(output_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


class PipelineRequestHandler(BaseHTTPRequestHandler):
    server_version = "MangaPipelineBridge/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("%s - %s\n" % (self.address_string(), format % args))

    def do_OPTIONS(self) -> None:
        _json_response(self, 200, {"ok": True})

    def do_GET(self) -> None:
        if self.path.rstrip("/") in {"", "/health"}:
            _json_response(self, 200, {"ok": True, "service": "local 8-step manga pipeline"})
        else:
            _json_response(self, 404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/translate":
            _json_response(self, 404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("Empty request body")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            image_bytes = _decode_image_data(payload)
            language = _normalize_language_hint(
                payload.get("sourceLanguage")
                or payload.get("language")
                or payload.get("lang")
            )
            with PIPELINE_LOCK:
                sample_name, _ = _write_runtime_sample(image_bytes, language)
                if _has_reusable_runtime_output(sample_name, language):
                    report = _collect_runtime_report(
                        sample_name,
                        language,
                        stage_timings=[{"stage": "runtime_output_cache", "seconds": 0}],
                        total_seconds=0,
                        reused_output=True,
                    )
                    _assert_runtime_report_safe(report)
                    print(f"[runtime] {sample_name} output cache hit", flush=True)
                else:
                    report = _run_runtime_pipeline(sample_name, language)
                translated_image = _read_output_data_url(sample_name)
            _json_response(
                self,
                200,
                {
                    "translatedImageDataUrl": translated_image,
                    "report": report,
                    "translations": [],
                },
            )
        except Exception as error:
            _json_response(self, 500, {"error": str(error)})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local pipeline server for the Chrome extension.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    server = ThreadingHTTPServer((args.host, args.port), PipelineRequestHandler)
    print(f"Local manga pipeline server listening on http://{args.host}:{args.port}/translate")
    print("Use Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
