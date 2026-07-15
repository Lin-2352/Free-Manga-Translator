"""
Local HTTP bridge for the Chrome extension.

The browser extension cannot run ONNX/PyTorch/LaMa models inside Chrome. This
server accepts a base64 image from the extension, runs the existing local
pipeline on a runtime sample folder, and returns the Step 8 output as a data URL.
"""
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

import argparse
import base64
import gc
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

from pipeline_paths import EXTENSION_RUNTIME_ROOT, PROJECT_ROOT


SAMPLES_ROOT = EXTENSION_RUNTIME_ROOT
RUNTIME_MANIFEST_ROOT = PROJECT_ROOT / "quality_reports" / "extension_runtime"
RUNTIME_CACHE_VERSION = "extension-runtime-cache-v10-ja-vision-rescue"
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
WARMUP_LOCK = threading.Lock()
WARMUP_THREAD: threading.Thread | None = None
IDLE_MONITOR_LOCK = threading.Lock()
IDLE_MONITOR_THREAD: threading.Thread | None = None
RUNTIME_ACTIVITY_LOCK = threading.Lock()
RUNTIME_STOP_EVENT = threading.Event()
# Bumped on every request_runtime_stop() call. A job captures the current generation when
# it is SUBMITTED (before it may have to wait on the scheduler) and is expected to abort if
# the generation has moved on by the time it actually runs a stage -- this makes stop
# semantics point-in-time instead of a single sticky flag that a later job's own start can
# silently clear out from under an earlier, still-running/queued job (see _begin_runtime_job).
RUNTIME_STOP_GENERATION = 0
RUNTIME_PENDING_RELEASE = False
RUNTIME_ACTIVE_JOBS = 0
# sample_name -> refcount. A plain set here would let two overlapping registrations for the
# same sample (e.g. pipeline_bridge's outer active-marking span plus this module's own
# _begin_runtime_job/_end_runtime_job around the miss-path pipeline run) have the INNER one's
# release wipe out the OUTER one's registration early, reopening the exact race this guards
# against. Refcounting makes nested/overlapping registrations for the same sample safe.
ACTIVE_RUNTIME_SAMPLES: dict[str, int] = {}
# Rolling per-site translation context so names/tone stay consistent across a
# few pages read in sequence -- keyed by page domain (not full URL, so every
# page/chapter of the same manga site shares one ring), each entry is the list
# of accepted English lines from one page. Process-lifetime only (an in-memory
# ring, not persisted) -- a service restart just starts a fresh ring, which is
# fine since this is a soft quality nicety, not correctness-critical state.
PAGE_CONTEXT_LOCK = threading.Lock()
PAGE_CONTEXT_RINGS: dict[str, deque] = {}
PAGE_CONTEXT_RING_MAXLEN = 2
PAGE_CONTEXT_MAX_LINES_PER_PAGE = 8
RUNTIME_LAST_ACTIVITY_MONOTONIC = time.monotonic()
RUNTIME_LAST_ACTIVITY_AT = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
WARMUP_STATE: dict[str, Any] = {
    "status": "idle",
    "models": {},
    "startedAt": None,
    "finishedAt": None,
    "seconds": None,
}


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def _idle_unload_seconds() -> int:
    return max(0, _env_int("FMT_GPU_IDLE_UNLOAD_SECONDS", 0))


def _mark_runtime_activity() -> None:
    global RUNTIME_LAST_ACTIVITY_MONOTONIC, RUNTIME_LAST_ACTIVITY_AT
    with RUNTIME_ACTIVITY_LOCK:
        RUNTIME_LAST_ACTIVITY_MONOTONIC = time.monotonic()
        RUNTIME_LAST_ACTIVITY_AT = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _add_active_sample_locked(sample_name: str) -> None:
    ACTIVE_RUNTIME_SAMPLES[sample_name] = ACTIVE_RUNTIME_SAMPLES.get(sample_name, 0) + 1


def _discard_active_sample_locked(sample_name: str) -> None:
    remaining = ACTIVE_RUNTIME_SAMPLES.get(sample_name, 1) - 1
    if remaining <= 0:
        ACTIVE_RUNTIME_SAMPLES.pop(sample_name, None)
    else:
        ACTIVE_RUNTIME_SAMPLES[sample_name] = remaining


def _mark_sample_active(sample_name: str | None) -> None:
    if not sample_name:
        return
    with RUNTIME_ACTIVITY_LOCK:
        _add_active_sample_locked(sample_name)


def _mark_sample_inactive(sample_name: str | None) -> None:
    if not sample_name:
        return
    with RUNTIME_ACTIVITY_LOCK:
        _discard_active_sample_locked(sample_name)


def current_stop_generation() -> int:
    with RUNTIME_ACTIVITY_LOCK:
        return RUNTIME_STOP_GENERATION


def _is_stop_current(captured_generation: int) -> bool:
    """True if a stop has been requested since captured_generation was captured."""
    with RUNTIME_ACTIVITY_LOCK:
        return RUNTIME_STOP_GENERATION != captured_generation


def _begin_runtime_job(sample_name: str | None = None, stop_generation: int | None = None) -> None:
    global RUNTIME_ACTIVE_JOBS
    with RUNTIME_ACTIVITY_LOCK:
        # Only acknowledge (clear) the stop-requested flag for a job that was submitted in
        # the CURRENT generation -- i.e. one unaffected by any stop that has happened since
        # it was requested. A job whose captured generation is stale is about to be aborted
        # by the generation check in run_stage anyway, so it must not clear the flag out from
        # under a more recent stop that other jobs still need to see.
        if stop_generation is not None and stop_generation == RUNTIME_STOP_GENERATION:
            RUNTIME_STOP_EVENT.clear()
        RUNTIME_ACTIVE_JOBS += 1
        if sample_name:
            _add_active_sample_locked(sample_name)
    _mark_runtime_activity()


def _end_runtime_job(sample_name: str | None = None) -> None:
    global RUNTIME_ACTIVE_JOBS
    release_after_job = False
    with RUNTIME_ACTIVITY_LOCK:
        RUNTIME_ACTIVE_JOBS = max(0, RUNTIME_ACTIVE_JOBS - 1)
        if sample_name:
            _discard_active_sample_locked(sample_name)
        release_after_job = RUNTIME_ACTIVE_JOBS == 0 and RUNTIME_PENDING_RELEASE
    _mark_runtime_activity()
    if release_after_job:
        release_runtime_models(reason="deferred_hard_stop")


def get_warmup_state() -> dict[str, Any]:
    state = dict(WARMUP_STATE)
    with RUNTIME_ACTIVITY_LOCK:
        state["activeJobs"] = RUNTIME_ACTIVE_JOBS
        state["activeSamples"] = sorted(ACTIVE_RUNTIME_SAMPLES)
        state["stopRequested"] = RUNTIME_STOP_EVENT.is_set()
        state["pendingRelease"] = RUNTIME_PENDING_RELEASE
        state["lastActivityAt"] = RUNTIME_LAST_ACTIVITY_AT
    state["idleUnloadSeconds"] = _idle_unload_seconds()
    return state


def _warmup_thread_running() -> bool:
    thread = WARMUP_THREAD
    return WARMUP_STATE.get("status") == "running" and thread is not None and thread.is_alive()


def release_runtime_models(reason: str = "manual_release") -> dict[str, Any]:
    global WARMUP_STATE, WARMUP_THREAD, RUNTIME_PENDING_RELEASE
    with RUNTIME_ACTIVITY_LOCK:
        active_jobs = RUNTIME_ACTIVE_JOBS
        warmup_running = _warmup_thread_running()
        if active_jobs > 0 or warmup_running:
            RUNTIME_PENDING_RELEASE = True
            return {
                "success": False,
                "status": "deferred",
                "reason": reason,
                "activeJobs": active_jobs,
                "warmupRunning": warmup_running,
                "message": "GPU release deferred until active pipeline work or warmup finishes.",
            }

    released: list[str] = []
    with PIPELINE_LOCK:
        modules = sys.modules
        step5 = modules.get("run_step5_ocr")
        if step5 is not None:
            for name in (
                "_MANGA_OCR_MODEL",
                "_TEXT_HANDLE",
                "_BUBBLE_MODEL",
                "_BUBBLE_DEVICE",
                "_SEMANTIC_HANDLE",
            ):
                if getattr(step5, name, None) is not None:
                    setattr(step5, name, None)
                    released.append(f"run_step5_ocr.{name}")
            if getattr(step5, "_EASYOCR_READERS", None):
                step5._EASYOCR_READERS.clear()
                released.append("run_step5_ocr._EASYOCR_READERS")
            if getattr(step5, "_PADDLEOCR_READERS", None):
                step5._PADDLEOCR_READERS.clear()
                released.append("run_step5_ocr._PADDLEOCR_READERS")

        step4 = modules.get("run_step4_inpaint")
        if step4 is not None:
            for name in ("_LAMA_SESSION", "_ANIME_LAMA_MODEL", "_ANIME_LAMA_DEVICE", "_MANGA_CLEANER_MODELS"):
                if getattr(step4, name, None) is not None:
                    setattr(step4, name, None)
                    released.append(f"run_step4_inpaint.{name}")
            if hasattr(step4, "_ANIME_LAMA_LOAD_ATTEMPTED"):
                step4._ANIME_LAMA_LOAD_ATTEMPTED = False
            if hasattr(step4, "_MANGA_CLEANER_LOAD_ATTEMPTED"):
                step4._MANGA_CLEANER_LOAD_ATTEMPTED = False

        step8 = modules.get("run_step8_typeset")
        if step8 is not None and hasattr(step8, "_load_font"):
            try:
                step8._load_font.cache_clear()
                released.append("run_step8_typeset._load_font")
            except Exception:
                pass

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                released.append("torch.cuda.cache")
        except Exception:
            pass

    with RUNTIME_ACTIVITY_LOCK:
        RUNTIME_PENDING_RELEASE = False
    WARMUP_THREAD = None
    WARMUP_STATE = {
        "status": "released",
        "models": {},
        "startedAt": None,
        "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": None,
        "reason": reason,
        "released": released,
    }
    _mark_runtime_activity()
    print(f"[runtime] GPU/runtime models released reason={reason} objects={len(released)}", flush=True)
    return {
        "success": True,
        "status": "released",
        "reason": reason,
        "released": released,
        "warmup": get_warmup_state(),
    }


def request_runtime_stop(release_gpu: bool = False, reason: str = "manual_stop") -> dict[str, Any]:
    global RUNTIME_PENDING_RELEASE, RUNTIME_STOP_GENERATION
    RUNTIME_STOP_EVENT.set()
    with RUNTIME_ACTIVITY_LOCK:
        RUNTIME_STOP_GENERATION += 1
        active_jobs = RUNTIME_ACTIVE_JOBS
        if release_gpu:
            RUNTIME_PENDING_RELEASE = True
    release_payload = release_runtime_models(reason=reason) if release_gpu and active_jobs == 0 else None
    return {
        "success": True,
        "status": "stopping" if active_jobs else "stopped",
        "reason": reason,
        "releaseGpu": release_gpu,
        "activeJobs": active_jobs,
        "release": release_payload,
        "warmup": get_warmup_state(),
    }


def start_idle_unload_monitor() -> dict[str, Any]:
    global IDLE_MONITOR_THREAD
    seconds = _idle_unload_seconds()
    if seconds <= 0:
        return {"enabled": False, "idleUnloadSeconds": 0}
    with IDLE_MONITOR_LOCK:
        if IDLE_MONITOR_THREAD and IDLE_MONITOR_THREAD.is_alive():
            return {"enabled": True, "idleUnloadSeconds": seconds, "status": "running"}

        def idle_loop() -> None:
            while True:
                current_seconds = _idle_unload_seconds()
                if current_seconds <= 0:
                    time.sleep(30)
                    continue
                time.sleep(max(10, min(60, current_seconds // 4 or 10)))
                with RUNTIME_ACTIVITY_LOCK:
                    active = RUNTIME_ACTIVE_JOBS
                    idle_for = time.monotonic() - RUNTIME_LAST_ACTIVITY_MONOTONIC
                    warm_status = WARMUP_STATE.get("status")
                if active == 0 and warm_status == "pass" and idle_for >= current_seconds:
                    release_runtime_models(reason=f"idle_{current_seconds}s")

        IDLE_MONITOR_THREAD = threading.Thread(target=idle_loop, name="fmt-idle-gpu-release", daemon=True)
        IDLE_MONITOR_THREAD.start()
    return {"enabled": True, "idleUnloadSeconds": seconds, "status": "started"}


def _warm_runtime_models_worker(force: bool = False) -> dict[str, Any]:
    global WARMUP_STATE
    with WARMUP_LOCK:
        if WARMUP_STATE.get("status") == "pass" and not force:
            return get_warmup_state()

        started = time.perf_counter()
        WARMUP_STATE = {
            "status": "running",
            "models": {},
            "startedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "finishedAt": None,
            "seconds": None,
        }
        print("[warmup] local pipeline model warmup started", flush=True)

        models: dict[str, str] = {}
        try:
            with PIPELINE_LOCK:
                os.environ.setdefault("LOCAL_NLLB_TRANSLATION", "1")
                from ml_region_lib import MLConfig, load_bubble_model, load_ocr_model, load_semantic_model, load_text_model
                import run_step4_inpaint
                import run_step5_ocr
                import run_step8_typeset

                cfg = MLConfig()

                if getattr(run_step5_ocr, "_MANGA_OCR_MODEL", None) is None:
                    run_step5_ocr._MANGA_OCR_MODEL = load_ocr_model(force_cpu=False)
                models["manga_ocr"] = "ready"

                if getattr(run_step5_ocr, "_TEXT_HANDLE", None) is None:
                    run_step5_ocr._TEXT_HANDLE = load_text_model(cfg.text_model_path)
                models["text_detector"] = "ready"

                if getattr(run_step5_ocr, "_BUBBLE_MODEL", None) is None:
                    bubble_model, bubble_device = load_bubble_model(cfg.bubble_model_path)
                    run_step5_ocr._BUBBLE_MODEL = bubble_model
                    run_step5_ocr._BUBBLE_DEVICE = bubble_device
                models["bubble_segmentor"] = "ready"

                if getattr(run_step5_ocr, "_SEMANTIC_HANDLE", None) is None:
                    run_step5_ocr._SEMANTIC_HANDLE = load_semantic_model("magi")
                models["semantic_detector"] = "ready"

                run_step4_inpaint._get_lama_session(cfg.lama_model_path)
                models["lama_onnx"] = "ready"

                anime_model, _ = run_step4_inpaint._get_anime_lama_model()
                models["anime_lama"] = "ready" if anime_model is not None else "unavailable"

                manga_cleaner = run_step4_inpaint._load_manga_cleaner_models()
                models["manga_cleaner"] = "ready" if manga_cleaner is not None else "disabled_or_unavailable"

                run_step8_typeset._load_font(24)
                run_step8_typeset._load_font(32)
                models["typeset_fonts"] = "ready"

            WARMUP_STATE = {
                "status": "pass",
                "models": models,
                "startedAt": WARMUP_STATE.get("startedAt"),
                "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seconds": round(time.perf_counter() - started, 3),
            }
        except Exception as error:
            models["error"] = str(error)
            WARMUP_STATE = {
                "status": "fail",
                "models": models,
                "startedAt": WARMUP_STATE.get("startedAt"),
                "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seconds": round(time.perf_counter() - started, 3),
                "error": str(error),
            }
            print(f"[warmup] local pipeline model warmup failed: {error}", flush=True)
            with RUNTIME_ACTIVITY_LOCK:
                release_after_warmup = RUNTIME_PENDING_RELEASE
            if release_after_warmup:
                return release_runtime_models(reason="deferred_failed_warmup_release")
            return get_warmup_state()

        print(f"[warmup] local pipeline model warmup done in {WARMUP_STATE['seconds']:.2f}s", flush=True)
        with RUNTIME_ACTIVITY_LOCK:
            release_after_warmup = RUNTIME_PENDING_RELEASE
        if release_after_warmup:
            return release_runtime_models(reason="deferred_warmup_release")
        return get_warmup_state()


def warm_runtime_models(force: bool = False, background: bool = True) -> dict[str, Any]:
    global WARMUP_STATE, WARMUP_THREAD
    start_idle_unload_monitor()
    if not _env_enabled("FMT_STARTUP_WARMUP", False) and not force:
        return {"status": "disabled", "models": {}, "seconds": None}

    if not background:
        return _warm_runtime_models_worker(force=force)

    if WARMUP_STATE.get("status") == "running":
        return get_warmup_state()
    if WARMUP_STATE.get("status") == "pass" and not force:
        return get_warmup_state()

    WARMUP_STATE = {
        "status": "running",
        "models": dict(WARMUP_STATE.get("models") or {}),
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finishedAt": None,
        "seconds": None,
    }
    WARMUP_THREAD = threading.Thread(
        target=_warm_runtime_models_worker,
        kwargs={"force": force},
        name="fmt-model-warmup",
        daemon=True,
    )
    WARMUP_THREAD.start()
    return get_warmup_state()


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


def clear_runtime_output_cache() -> dict[str, Any]:
    cleared_samples = 0
    removed_folders = 0
    removed_meta = 0
    skipped_active: list[str] = []
    with PIPELINE_LOCK:
        with RUNTIME_ACTIVITY_LOCK:
            active_samples = set(ACTIVE_RUNTIME_SAMPLES)
        if not SAMPLES_ROOT.exists():
            return {
                "success": True,
                "status": "cleared",
                "clearedSamples": 0,
                "removedFolders": 0,
                "removedMeta": 0,
                "skippedActiveSamples": [],
            }
        for sample_path in SAMPLES_ROOT.glob("runtime_*"):
            if not sample_path.is_dir():
                continue
            if sample_path.name in active_samples:
                skipped_active.append(sample_path.name)
                continue
            touched = False
            for folder in OUTPUT_FOLDERS:
                target = sample_path / folder
                if target.exists():
                    shutil.rmtree(target)
                    removed_folders += 1
                    touched = True
            meta_path = sample_path / RUNTIME_CACHE_META
            if meta_path.exists():
                meta_path.unlink()
                removed_meta += 1
                touched = True
            if touched:
                cleared_samples += 1
    print(
        (
            "[runtime] output cache cleared "
            f"samples={cleared_samples} folders={removed_folders} meta={removed_meta} "
            f"skippedActive={len(skipped_active)}"
        ),
        flush=True,
    )
    if skipped_active:
        print(f"[runtime] output cache clear skipped active samples={skipped_active}", flush=True)
    return {
        "success": True,
        "status": "cleared",
        "clearedSamples": cleared_samples,
        "removedFolders": removed_folders,
        "removedMeta": removed_meta,
        "skippedActiveSamples": skipped_active,
    }


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
    # A blank/no-text outcome must NEVER be reused from cache: _has_reusable_
    # runtime_output already requires status=="pass" above, so writing
    # anything else here is sufficient -- the next request for the same
    # image bytes re-runs the full pipeline (picking up the vision rescue,
    # rotated keys, or any other fix) instead of being served the same blank
    # page forever.
    status = "no_renderable_text" if report.get("noRenderableText") else "pass"
    _runtime_cache_meta_path(sample_name).write_text(
        json.dumps(
            {
                "cacheVersion": RUNTIME_CACHE_VERSION,
                "language": _normalize_language_hint(language),
                "status": status,
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


def _vision_rescue_report_fields(meta_path: Path) -> dict[str, Any]:
    # run_step5_ocr.py writes this file whenever the zero-usable-text
    # condition was checked, whether or not a rescue actually fired --
    # letting the extension distinguish "genuinely no readable text (vision
    # also tried and found none)" from "the pipeline never got that far" and
    # from "there IS text, just not renderable for some other reason", which
    # a bare noRenderableText:true collapses into one undifferentiated case.
    defaults = {
        "rescueAttempted": False,
        "rescueMode": None,
        "rescueSucceeded": False,
        "rescueRegions": 0,
    }
    if not meta_path.exists():
        return defaults
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(meta, dict):
        return defaults
    return {
        "rescueAttempted": bool(meta.get("attempted")),
        "rescueMode": meta.get("mode"),
        "rescueSucceeded": bool(meta.get("succeeded")),
        "rescueRegions": int(meta.get("regions") or 0),
    }


def _page_context_domain(page_key: str) -> str:
    if not page_key:
        return ""
    try:
        parsed = urlparse(page_key)
    except ValueError:
        return ""
    return (parsed.netloc or "").lower()


def _write_page_context_file(sample_path: Path, domain: str) -> None:
    # Called BEFORE step 7 runs, so the ring only ever contains PRIOR pages'
    # accepted lines -- this page's own translations are recorded afterward.
    context_path = sample_path / "page_context.json"
    if not domain:
        context_path.unlink(missing_ok=True)
        return
    with PAGE_CONTEXT_LOCK:
        ring = PAGE_CONTEXT_RINGS.get(domain)
        lines = [line for page_lines in (ring or ()) for line in page_lines]
    if not lines:
        context_path.unlink(missing_ok=True)
        return
    context_path.write_text(
        json.dumps({"domain": domain, "lines": lines}, ensure_ascii=False),
        encoding="utf-8",
    )


def _accepted_translation_lines(sample_path: Path) -> list[str]:
    results_path = sample_path / "step_7_translate" / "translation_results.json"
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = data if isinstance(data, list) else []
    # Excludes "fallback" (local NLLB, no page context, lower quality) so a
    # poor fragment-by-fragment guess never seeds a bad name spelling for the
    # NEXT page's prompt -- only API-sourced or vision-pretranslated text
    # feeds the rolling context forward.
    return [
        str(entry.get("en_text", "")).strip()
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("translation_source") != "fallback"
        and str(entry.get("en_text", "")).strip()
    ]


def _record_page_context(domain: str, translated_lines: list[str]) -> None:
    if not domain:
        return
    trimmed = [str(line).strip() for line in translated_lines if str(line).strip()]
    if not trimmed:
        return
    with PAGE_CONTEXT_LOCK:
        ring = PAGE_CONTEXT_RINGS.setdefault(domain, deque(maxlen=PAGE_CONTEXT_RING_MAXLEN))
        ring.append(trimmed[:PAGE_CONTEXT_MAX_LINES_PER_PAGE])


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
    ocr_path = sample_path / "step_5_ocr" / "ocr_results.json"
    translation_path = sample_path / "step_7_translate" / "translation_results.json"
    provider_report_path = sample_path / "step_7_translate" / "translation_provider_report.json"
    typeset_report_path = sample_path / "step_8_typeset" / "typeset_report.json"
    vision_rescue_meta_path = sample_path / "step_5_ocr" / "vision_rescue_meta.json"

    ocr_items = json.loads(ocr_path.read_text(encoding="utf-8")) if ocr_path.exists() else []
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
    translation_source_counts: dict[str, int] = {}
    serving_providers: list[str] = []
    if provider_report_path.exists():
        try:
            provider_report = json.loads(provider_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            provider_report = {}
        translation_source_counts = provider_report.get("translation_source_counts") or {}
        for attempt in (provider_report.get("api_report") or {}).get("attempts", []):
            if isinstance(attempt, dict) and attempt.get("status") == "pass" and attempt.get("accepted"):
                label = str(attempt.get("provider") or "")
                model = str(attempt.get("model") or "")
                serving_providers.append(f"{label}:{model}" if model else label)
    return {
        "sampleName": sample_name,
        "sourceLanguage": _normalize_language_hint(language),
        "pipeline": "local-8-stage",
        "stageSequence": STAGE_SEQUENCE,
        "stageArtifacts": stage_artifacts,
        "missingStageArtifacts": missing_artifacts,
        "ocrItems": len(ocr_items),
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
        # Which provider(s)/model(s) actually served this request's translations,
        # and the api/vision/fallback/skipped breakdown -- lets a console log
        # attribute "translation quality is bad" to e.g. the local NLLB fallback
        # engaging instead of an API provider, without opening the sample's own
        # translation_provider_report.json by hand.
        "translationServingProviders": sorted(set(serving_providers)),
        "translationSourceCounts": translation_source_counts,
        **_vision_rescue_report_fields(vision_rescue_meta_path),
    }


def _runtime_report_warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    # "skipped_unsafe_floating_cleanup" is Step 8's fail-closed skip when
    # Step 4 couldn't safely clean a region (see unsafe_art_preserved):
    # the source text is left untranslated on the page. This used to be
    # recorded only in typeset_report.json with no signal reaching the
    # API response, so a human reviewer had no way to know a region was
    # silently left untranslated without opening that file by hand.
    review_statuses = {
        "fallback_clipped", "emergency", "clipped", "missing", "empty",
        "skipped_unsafe_floating_cleanup",
    }
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
    if int(report.get("ocrItems", 0)) and not int(report.get("layoutConstraints", 0)):
        warnings.append(f"ocr_without_layout_constraints={report.get('ocrItems', 0)}")
    return warnings


def _annotate_runtime_report_safety(report: dict[str, Any]) -> dict[str, Any]:
    warnings = _runtime_report_warnings(report)
    report["reviewWarnings"] = warnings
    if _runtime_report_has_no_renderable_text(report):
        report["outputSafety"] = "no_renderable_text"
        report["noRenderableText"] = True
    else:
        report["outputSafety"] = "review" if warnings else "pass"
        report["noRenderableText"] = False
    return report


def _runtime_report_has_no_renderable_text(report: dict[str, Any]) -> bool:
    missing = set(report.get("missingStageArtifacts") or [])
    essential_ready = not ({"step_5", "step_6", "step_7", "step_8"} & missing)
    return (
        essential_ready
        and not int(report.get("layoutConstraints", 0))
        and not int(report.get("translations", 0))
        and not int(report.get("renderedRegions", 0))
    )


def _runtime_failure_summary(report: dict[str, Any], critical_errors: list[str]) -> dict[str, Any]:
    return {
        "sampleName": report.get("sampleName"),
        "sourceLanguage": report.get("sourceLanguage"),
        "criticalErrors": critical_errors,
        "missingStageArtifacts": report.get("missingStageArtifacts", []),
        "ocrItems": report.get("ocrItems", 0),
        "layoutConstraints": report.get("layoutConstraints", 0),
        "rejectedLayoutItems": report.get("rejectedLayoutItems", 0),
        "rawTranslations": report.get("rawTranslations", 0),
        "translations": report.get("translations", 0),
        "renderedRegions": report.get("renderedRegions", 0),
        "reviewWarnings": report.get("reviewWarnings", []),
        "runtimeOutputCache": report.get("runtimeOutputCache"),
    }


def _maybe_export_training_case(
    report: dict[str, Any],
    critical_errors: list[str] | None = None,
) -> dict[str, Any] | None:
    try:
        from training_data_export import export_runtime_training_case

        exported = export_runtime_training_case(
            SAMPLES_ROOT,
            str(report.get("sampleName", "")),
            str(report.get("sourceLanguage", "")),
            report,
            critical_errors=critical_errors,
        )
        if exported:
            report["trainingDataExport"] = exported
            print(
                f"[runtime] training data exported case={exported['caseId']} regions={exported['regions']} path={exported['path']}",
                flush=True,
            )
        return exported
    except Exception as error:
        print(f"[runtime] training data export skipped: {str(error)[:180]}", flush=True)
        return None


def _assert_runtime_report_safe(report: dict[str, Any]) -> None:
    critical_errors: list[str] = []
    _annotate_runtime_report_safety(report)
    missing_artifacts = list(report.get("missingStageArtifacts") or [])
    if _runtime_report_has_no_renderable_text(report):
        missing_artifacts = [
            item
            for item in missing_artifacts
            if item not in {"step_1", "step_2", "step_4"}
        ]
    if missing_artifacts:
        critical_errors.append(f"missing_stage_artifacts={missing_artifacts}")
    if int(report.get("placeholderTranslations", 0)):
        critical_errors.append(f"placeholder_translations={report['placeholderTranslations']}")
    if (
        not int(report.get("renderedRegions", 0))
        and (
            int(report.get("layoutConstraints", 0))
            or int(report.get("translations", 0))
        )
    ):
        critical_errors.append("no_rendered_regions")

    if critical_errors:
        exported = _maybe_export_training_case(report, critical_errors=critical_errors)
        summary = _runtime_failure_summary(report, critical_errors)
        if exported:
            summary["trainingDataExport"] = exported
        print(f"[runtime] pipeline safety failure: {json.dumps(summary, ensure_ascii=False)}", flush=True)
        raise RuntimeError(f"Local pipeline failed before producing a usable output: {summary}")


def _run_runtime_pipeline(
    sample_name: str, language: str, stop_generation: int | None = None, page_key: str = ""
) -> dict[str, Any]:
    total_started = time.perf_counter()
    page_domain = _page_context_domain(page_key)
    os.environ["LOCAL_NLLB_TRANSLATION"] = "1"
    import run_step4_inpaint
    import run_step5_ocr
    import run_step6_layout
    import run_step7_translate
    import run_step8_typeset

    sample_map = {sample_name: "input.jpg"}

    _write_runtime_manifest(sample_name, language)

    stage_timings: list[dict[str, Any]] = []
    # Captured once, at (or before) the point this specific job actually started running --
    # callers that submit work before it runs (e.g. while waiting on the GPU scheduler) pass
    # their own earlier-captured generation down so a stop requested during that wait still
    # cancels them. Falls back to "now" for direct callers (e.g. the legacy standalone HTTP
    # handler) that never captured one.
    job_generation = stop_generation if stop_generation is not None else current_stop_generation()

    def run_stage(label: str, callback: Any) -> None:
        if _is_stop_current(job_generation):
            raise RuntimeError("Translation stopped by user")
        started = time.perf_counter()
        print(f"[runtime] {sample_name} {label}: start", flush=True)
        callback()
        if _is_stop_current(job_generation):
            raise RuntimeError("Translation stopped by user")
        elapsed = time.perf_counter() - started
        stage_timings.append({"stage": label, "seconds": round(elapsed, 3)})
        print(f"[runtime] {sample_name} {label}: done in {elapsed:.2f}s", flush=True)

    _begin_runtime_job(sample_name, stop_generation=job_generation)
    try:
        run_stage("step5_ocr", lambda: run_step5_ocr.run_step5_ocr(sample_map=sample_map, samples_dir=SAMPLES_ROOT))
        run_stage("step6_layout", lambda: run_step6_layout.run_step6_layout(sample_map=sample_map, samples_dir=SAMPLES_ROOT))
        _write_page_context_file(SAMPLES_ROOT / sample_name, page_domain)
        run_stage("step7_translate", lambda: run_step7_translate.run_step7_translate(sample_map=sample_map, samples_dir=SAMPLES_ROOT))
        _record_page_context(page_domain, _accepted_translation_lines(SAMPLES_ROOT / sample_name))
        run_stage("step4_inpaint", lambda: run_step4_inpaint.run_step4_inpaint(sample_map=sample_map, samples_dir=SAMPLES_ROOT))
        run_stage("step8_typeset", lambda: run_step8_typeset.run_step8_typeset(sample_map=sample_map, samples_dir=SAMPLES_ROOT))

        report = _collect_runtime_report(
            sample_name,
            language,
            stage_timings=stage_timings,
            total_seconds=time.perf_counter() - total_started,
        )
        _assert_runtime_report_safe(report)
        _maybe_export_training_case(report)
        _write_runtime_cache_meta(sample_name, language, report)
        print(f"[runtime] {sample_name} total: {report['totalSeconds']:.2f}s", flush=True)
        return report
    finally:
        _end_runtime_job(sample_name)


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
