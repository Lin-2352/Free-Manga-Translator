from __future__ import annotations

# --- Clean-copy path bootstrap ---
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
# --- End clean-copy path bootstrap ---

from pathlib import Path
import contextlib
import hashlib
import threading
import time
from typing import Any, Iterator

import run_extension_pipeline_server as legacy_bridge

from .gpu_scheduler import PIPELINE_SCHEDULER


_SAMPLE_LOCKS: dict[str, threading.Lock] = {}
_SAMPLE_LOCK_REFCOUNTS: dict[str, int] = {}
_SAMPLE_LOCKS_GUARD = threading.Lock()


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


def _failure_diagnostics(sample_name: str, language: str) -> dict[str, Any]:
    artifacts = _artifact_paths(sample_name)
    try:
        report = legacy_bridge._collect_runtime_report(sample_name, language)
        legacy_bridge._annotate_runtime_report_safety(report)
    except Exception as error:
        report = {"diagnosticError": str(error)}
    return {
        "sampleName": sample_name,
        "language": language,
        "artifacts": artifacts,
        "report": report,
    }


def _runtime_sample_name(image_bytes: bytes, language: str) -> str:
    digest = hashlib.sha1(image_bytes).hexdigest()[:14]
    normalized = legacy_bridge._normalize_language_hint(language)
    return f"runtime_{normalized}_{digest}"


@contextlib.contextmanager
def _sample_lock(sample_name: str) -> Iterator[None]:
    # A plain dict of one Lock per unique image, never evicted, grows without bound across
    # the server's lifetime (every distinct image ever translated leaks a Lock object
    # forever). Refcount each lock's outstanding holders/waiters and drop it from the dict
    # once nobody needs it -- a later request for the same image just creates a fresh Lock,
    # which is equivalent for mutual exclusion since there is no concurrent user left to race.
    with _SAMPLE_LOCKS_GUARD:
        lock = _SAMPLE_LOCKS.get(sample_name)
        if lock is None:
            lock = threading.Lock()
            _SAMPLE_LOCKS[sample_name] = lock
        _SAMPLE_LOCK_REFCOUNTS[sample_name] = _SAMPLE_LOCK_REFCOUNTS.get(sample_name, 0) + 1
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        with _SAMPLE_LOCKS_GUARD:
            remaining = _SAMPLE_LOCK_REFCOUNTS.get(sample_name, 1) - 1
            if remaining <= 0:
                _SAMPLE_LOCK_REFCOUNTS.pop(sample_name, None)
                _SAMPLE_LOCKS.pop(sample_name, None)
            else:
                _SAMPLE_LOCK_REFCOUNTS[sample_name] = remaining


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

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    cache_id = metadata.get("cacheId") or metadata.get("cacheKey") or "no-cache-id"
    trace_id = metadata.get("traceId") or payload.get("clientRequestId") or "no-trace"
    source = metadata.get("source") or "unknown"
    sample_name = _runtime_sample_name(image_bytes, language)
    stop_generation = legacy_bridge.current_stop_generation()
    with _sample_lock(sample_name):
        # A cache hit (and the write/reusability-check preceding it) never went through
        # _begin_runtime_job/_end_runtime_job, which is what registers a sample in
        # ACTIVE_RUNTIME_SAMPLES -- so a concurrent /v1/cache/clear could rmtree this
        # sample's output folders mid-read. Register for the whole span (the miss path's
        # _run_runtime_pipeline call also registers internally; that's fine, both are now
        # refcounted so nested registration can't cause an early release).
        legacy_bridge._mark_sample_active(sample_name)
        try:
            with PIPELINE_SCHEDULER.acquire(str(cache_id)[:80]) as scheduler_slot:
                print(
                    (
                        f"[api] pipeline request start trace={trace_id} sample={sample_name} "
                        f"language={language} source={source} bytes={len(image_bytes)} cache={str(cache_id)[:80]}"
                    ),
                    flush=True,
                )
                sample_name, _ = legacy_bridge._write_runtime_sample(image_bytes, language)
                try:
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
                        report = legacy_bridge._run_runtime_pipeline(
                            sample_name, language, stop_generation=stop_generation
                        )
                    report["scheduler"] = scheduler_slot.as_report()
                    translated_image = legacy_bridge._read_output_data_url(sample_name)
                except Exception as error:
                    diagnostics = _failure_diagnostics(sample_name, language)
                    print(
                        (
                            f"[api] pipeline request failed trace={trace_id} sample={sample_name} "
                            f"language={language} error={error} diagnostics={diagnostics}"
                        ),
                        flush=True,
                    )
                    raise PipelineRunError(
                        f"Pipeline failed for {sample_name}: {error}; diagnostics={diagnostics}"
                    ) from error
                step8 = report.get("step8") if isinstance(report.get("step8"), dict) else {}
                print(
                    (
                        f"[api] pipeline request done trace={trace_id} sample={sample_name} "
                        f"outputSafety={report.get('outputSafety')} rendered={step8.get('regions')} "
                        f"total={time.perf_counter() - started:.2f}s"
                    ),
                    flush=True,
                )
        finally:
            legacy_bridge._mark_sample_inactive(sample_name)

    return {
        "sampleName": sample_name,
        "language": language,
        "translatedImageDataUrl": translated_image,
        "report": report,
        "artifacts": _artifact_paths(sample_name),
    }
