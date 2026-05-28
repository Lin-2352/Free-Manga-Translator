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


from pathlib import Path
import time
from typing import Any

import run_extension_pipeline_server as legacy_bridge


class PipelineRunError(RuntimeError):
    """Raised when the local pipeline cannot produce a safe final output."""


def _artifact_paths(sample_name: str) -> dict[str, str]:
    sample_path = legacy_bridge.SAMPLES_ROOT / sample_name
    paths = {
        "input": sample_path / "input.jpg",
        "step1_detect": sample_path / "step_1_detect" / "detections.json",
        "step4_inpaint": sample_path / "step_4_final" / "inpainted_result.jpg",
        "step6_layout": sample_path / "step_6_layout" / "layout_constraints.json",
        "step6_debug": sample_path / "step_6_layout" / "debug_layout_boxes.jpg",
        "step6_rejected": sample_path / "step_6_layout" / "rejected_layout_items.json",
        "step7_translate": sample_path / "step_7_translate" / "translation_results.json",
        "step7_provider_report": sample_path / "step_7_translate" / "translation_provider_report.json",
        "step8_output": sample_path / "step_8_typeset" / "final_output.png",
        "step8_report": sample_path / "step_8_typeset" / "typeset_report.json",
    }
    return {
        key: str(path)
        for key, path in paths.items()
        if isinstance(path, Path) and path.exists()
    }


def run_pipeline_payload(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    image_bytes = legacy_bridge._decode_image_data(payload)
    language = legacy_bridge._normalize_language_hint(
        payload.get("sourceLanguage")
        or payload.get("source_language")
        or payload.get("language")
        or payload.get("lang")
    )
    target_language = str(payload.get("targetLanguage") or payload.get("target_language") or "en").lower()
    quality_profile = str(payload.get("qualityProfile") or payload.get("quality_profile") or "strict").lower()

    if target_language != "en":
        raise PipelineRunError("Only English target output is supported by the current validated pipeline.")
    if quality_profile not in {"strict", "manual-review", "fast-preview"}:
        raise PipelineRunError(f"Unsupported quality profile: {quality_profile}")
    if quality_profile != "strict":
        raise PipelineRunError("Only strict mode is enabled for consumer-safe output.")

    with legacy_bridge.PIPELINE_LOCK:
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        cache_id = metadata.get("cacheId") or metadata.get("cacheKey") or "no-cache-id"
        print(
            f"[api] pipeline request start language={language} bytes={len(image_bytes)} cache={str(cache_id)[:80]}",
            flush=True,
        )
        sample_name, _ = legacy_bridge._write_runtime_sample(image_bytes, language)
        if legacy_bridge._has_reusable_runtime_output(sample_name, language):
            report = legacy_bridge._collect_runtime_report(
                sample_name,
                language,
                stage_timings=[{"stage": "runtime_output_cache", "seconds": 0}],
                total_seconds=0,
                reused_output=True,
            )
            legacy_bridge._assert_runtime_report_safe(report)
            print(f"[api] runtime output cache hit sample={sample_name}", flush=True)
        else:
            report = legacy_bridge._run_runtime_pipeline(sample_name, language)
        translated_image = legacy_bridge._read_output_data_url(sample_name)
        print(
            f"[api] pipeline request done sample={sample_name} total={time.perf_counter() - started:.2f}s",
            flush=True,
        )

    return {
        "sampleName": sample_name,
        "language": language,
        "translatedImageDataUrl": translated_image,
        "report": report,
        "artifacts": _artifact_paths(sample_name),
    }
