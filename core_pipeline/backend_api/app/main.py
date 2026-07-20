from __future__ import annotations

import hmac
import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .pipeline_bridge import PipelineRunError, run_pipeline_payload
from .gpu_scheduler import PIPELINE_SCHEDULER
from .schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    JobStatusResponse,
    TranslateRequest,
    TranslateResponse,
)
from api_manager import API_MANAGER, DAILY_LIMIT_MESSAGE, ApiQuotaExhausted
from diagnostic_logger import diagnostics_enabled, diagnostics_log_path, write_diagnostic_event
import run_extension_pipeline_server as legacy_bridge


API_VERSION = "0.1.4-local-8stage"

app = FastAPI(
    title="Free Manga Translator Local Pipeline API",
    version=API_VERSION,
    description="Local companion API that wraps the validated 8-stage manga translation pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://.*|http://127\.0\.0\.1(:\d+)?|http://localhost(:\d+)?)$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Fmt-Client", "X-Fmt-Auth", "ngrok-skip-browser-warning"],
)

# Every route except the bare health check requires this header. It buys two things cheaply:
# (1) a non-empty custom header forces the browser to CORS-preflight the request, closing the
# unpreflighted "simple request" CSRF gap against this unauthenticated loopback API; (2) it's a
# minimum bar against a *different* installed extension calling in opportunistically. It is not a
# secret/token (the value is public in this repo) and a full auth scheme is out of proportion for
# a single-user local companion app -- this is deliberately a shallow, cheap gate, not real auth.
EXTENSION_CLIENT_HEADER = "X-Fmt-Client"
EXTENSION_CLIENT_VALUE = "free-manga-translator-extension"

# Optional real auth, layered on top of the client-header gate above rather than replacing it.
# Unset (the default for every local/loopback deployment) => byte-identical to today's behavior,
# nothing below ever runs. Set (e.g. when this server is reachable over a public tunnel, such as
# the Kaggle-GPU deployment path) => every gated route also requires this exact token in
# X-Fmt-Auth. FMT_AUTH_TOKEN is read once at import time like the rest of this module's
# environment-derived constants; a running server picking up a changed token needs a restart,
# same as every other env-derived setting here.
FMT_AUTH_TOKEN = os.environ.get("FMT_AUTH_TOKEN", "").strip()
AUTH_HEADER = "X-Fmt-Auth"


def _require_extension_client(
    x_fmt_client: str | None = Header(default=None, alias=EXTENSION_CLIENT_HEADER),
    x_fmt_auth: str | None = Header(default=None, alias=AUTH_HEADER),
) -> None:
    if x_fmt_client != EXTENSION_CLIENT_VALUE:
        raise HTTPException(status_code=403, detail="Missing or invalid client header")
    if FMT_AUTH_TOKEN and not hmac.compare_digest(x_fmt_auth or "", FMT_AUTH_TOKEN):
        raise HTTPException(status_code=403, detail="Missing or invalid auth token")


_GATED = [Depends(_require_extension_client)]

JOBS: dict[str, dict[str, Any]] = {}
_JOBS_GUARD = threading.Lock()
_JOBS_MAX_ENTRIES = 200


def _record_job(job_id: str, record: dict[str, Any]) -> None:
    with _JOBS_GUARD:
        JOBS[job_id] = record
        if len(JOBS) > _JOBS_MAX_ENTRIES:
            for stale_id in list(JOBS.keys())[: len(JOBS) - _JOBS_MAX_ENTRIES]:
                JOBS.pop(stale_id, None)


def _diagnostic_json(payload: dict[str, Any]) -> str:
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if key.lower() in {"imagedata", "base64data", "translatedimagedataurl", "imagedataurl"}:
            safe[key] = "<redacted>"
        elif isinstance(value, str) and len(value) > 260:
            safe[key] = value[:260] + "..."
        elif isinstance(value, dict):
            safe[key] = json.loads(_diagnostic_json(value))
        else:
            safe[key] = value
    return json.dumps(safe, ensure_ascii=False, sort_keys=True)


@app.on_event("startup")
def startup_warmup() -> None:
    legacy_bridge.start_idle_unload_monitor()
    legacy_bridge.warm_runtime_models(background=True)


def _request_payload(request: TranslateRequest) -> dict[str, Any]:
    image_data = request.imageData or request.base64Data
    if not image_data:
        raise HTTPException(status_code=400, detail="Missing imageData/base64Data")
    return {
        "imageData": image_data,
        "sourceLanguage": request.sourceLanguage,
        "targetLanguage": request.targetLanguage,
        "qualityProfile": request.qualityProfile,
        "requestedOutput": request.requestedOutput,
        "clientRequestId": request.clientRequestId,
        "metadata": request.metadata,
    }


@app.post("/v1/diagnostics/log", dependencies=_GATED)
def extension_diagnostic_log(payload: dict[str, Any]) -> dict[str, Any]:
    trace_id = str(payload.get("traceId") or payload.get("trace_id") or "no-trace")
    event = str(payload.get("event") or "diagnostic")
    safe_payload = json.loads(_diagnostic_json(payload))
    write_diagnostic_event(event, safe_payload, trace_id=trace_id, source="extension", level="info")
    print(f"[extdiag] trace={trace_id} event={event} payload={json.dumps(safe_payload, ensure_ascii=False, sort_keys=True)}", flush=True)
    return {"ok": True, "traceId": trace_id}


@app.get("/v1/diagnostics/status", dependencies=_GATED)
def diagnostics_status() -> dict[str, Any]:
    return {
        "ok": True,
        "enabled": diagnostics_enabled(),
        "path": str(diagnostics_log_path()),
        "checkedAt": datetime.now(timezone.utc).isoformat(),
    }


def _request_trace_id(request: TranslateRequest, fallback: str) -> str:
    metadata = request.metadata if isinstance(request.metadata, dict) else {}
    return str(
        request.clientRequestId
        or metadata.get("traceId")
        or metadata.get("trace_id")
        or fallback
    )


def _execute_translate(request: TranslateRequest) -> TranslateResponse:
    job_id = str(uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    trace_id = _request_trace_id(request, job_id)
    _record_job(job_id, {
        "jobId": job_id,
        "status": "running",
        "report": {"startedAt": started_at},
        "artifacts": {},
    })
    write_diagnostic_event(
        "pipeline.request.start",
        {
            "jobId": job_id,
            "sourceLanguage": request.sourceLanguage,
            "targetLanguage": request.targetLanguage,
            "qualityProfile": request.qualityProfile,
            "requestedOutput": request.requestedOutput,
            "metadata": request.metadata,
        },
        trace_id=trace_id,
        source="backend",
        level="info",
    )

    try:
        result = run_pipeline_payload(_request_payload(request))
        response = TranslateResponse(
            jobId=job_id,
            status="pass",
            translatedImageDataUrl=result["translatedImageDataUrl"],
            imageDataUrl=result["translatedImageDataUrl"],
            translations=[],
            report=result["report"],
            artifacts=result["artifacts"],
        )
        _record_job(job_id, response.model_dump() if hasattr(response, "model_dump") else response.dict())
        report = response.report if isinstance(response.report, dict) else {}
        write_diagnostic_event(
            "pipeline.request.done",
            {
                "jobId": job_id,
                "status": response.status,
                "sample": report.get("sample") or report.get("sample_name"),
                "reportStatus": report.get("status") or report.get("outputSafety"),
                "renderedRegions": report.get("renderedRegions"),
                "translationCount": report.get("translationCount"),
            },
            trace_id=trace_id,
            source="backend",
            level="info",
        )
        return response
    except HTTPException:
        # job_id's JOBS entry can be evicted by _record_job's FIFO cap (concurrent churn from
        # other requests) between it being recorded above and this handler running -- a bare
        # subscript mutation would then KeyError here, masking the real HTTPException with an
        # unrelated crash during its own error handling.
        with _JOBS_GUARD:
            if job_id in JOBS:
                JOBS[job_id]["status"] = "fail"
        write_diagnostic_event(
            "pipeline.request.http_error",
            {"jobId": job_id, "status": "fail"},
            trace_id=trace_id,
            source="backend",
            level="error",
        )
        raise
    except ApiQuotaExhausted as error:
        _record_job(job_id, {
            "jobId": job_id,
            "status": "quota_exhausted",
            "report": JOBS.get(job_id, {}).get("report", {}),
            "artifacts": JOBS.get(job_id, {}).get("artifacts", {}),
            "error": DAILY_LIMIT_MESSAGE,
        })
        write_diagnostic_event(
            "pipeline.request.quota_exhausted",
            {"jobId": job_id, "error": str(error), "message": DAILY_LIMIT_MESSAGE},
            trace_id=trace_id,
            source="backend",
            level="error",
        )
        raise HTTPException(status_code=429, detail={"code": "DAILY_LIMIT_REACHED", "message": DAILY_LIMIT_MESSAGE}) from error
    except (PipelineRunError, Exception) as error:
        # Diagnostics get the full exception text (including local paths); the HTTP response
        # only ever gets an opaque code + traceId so callers on the local API surface can't use
        # error bodies to enumerate this machine's filesystem layout.
        safe_message = "Translation failed. See server diagnostics for details."
        _record_job(job_id, {
            "jobId": job_id,
            "status": "fail",
            "report": JOBS.get(job_id, {}).get("report", {}),
            "artifacts": JOBS.get(job_id, {}).get("artifacts", {}),
            "error": safe_message,
        })
        write_diagnostic_event(
            "pipeline.request.error",
            {"jobId": job_id, "error": str(error), "errorType": type(error).__name__},
            trace_id=trace_id,
            source="backend",
            level="error",
        )
        raise HTTPException(
            status_code=500,
            detail={"code": "PIPELINE_ERROR", "message": safe_message, "traceId": trace_id},
        ) from error


@app.get("/health", response_model=HealthResponse)
@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        service="free-manga-translator-local-pipeline-api",
        mode="local-only-strict",
        version=API_VERSION,
        warmup=legacy_bridge.get_warmup_state(),
        scheduler=PIPELINE_SCHEDULER.status(),
    )


@app.get("/v1/warmup", dependencies=_GATED)
@app.post("/v1/warmup", dependencies=_GATED)
def warmup() -> dict[str, Any]:
    return legacy_bridge.warm_runtime_models(force=True, background=True)


@app.post("/api/runtime/soft-stop", dependencies=_GATED)
@app.post("/v1/runtime/soft-stop", dependencies=_GATED)
def runtime_soft_stop() -> dict[str, Any]:
    return legacy_bridge.request_runtime_stop(release_gpu=False, reason="soft_stop")


@app.post("/api/runtime/hard-stop", dependencies=_GATED)
@app.post("/v1/runtime/hard-stop", dependencies=_GATED)
def runtime_hard_stop() -> dict[str, Any]:
    return legacy_bridge.request_runtime_stop(release_gpu=True, reason="hard_stop")


@app.post("/api/gpu/release", dependencies=_GATED)
@app.post("/v1/gpu/release", dependencies=_GATED)
def release_gpu() -> dict[str, Any]:
    return legacy_bridge.release_runtime_models(reason="manual_release")


@app.post("/api/cache/clear", dependencies=_GATED)
@app.post("/v1/cache/clear", dependencies=_GATED)
def clear_runtime_cache() -> dict[str, Any]:
    return legacy_bridge.clear_runtime_output_cache()


@app.get("/api/quota-status", dependencies=_GATED)
@app.get("/v1/quota-status", dependencies=_GATED)
def quota_status() -> dict[str, Any]:
    return API_MANAGER.quota_status()


def _parse_nvidia_smi_rows(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            total = int(parts[2])
            used = int(parts[3])
            free = int(parts[4])
            utilization = int(parts[5])
        except ValueError:
            continue
        gpus.append(
            {
                "index": int(parts[0]) if parts[0].isdigit() else parts[0],
                "name": parts[1],
                "memoryTotalMiB": total,
                "memoryUsedMiB": used,
                "memoryFreeMiB": free,
                "usagePercent": round((used / total) * 100, 1) if total else 0.0,
                "utilizationGpuPercent": utilization,
            }
        )
    return gpus


def _scheduler_vram_status_payload(reason: str) -> dict[str, Any]:
    status = PIPELINE_SCHEDULER.status()
    gpu = status.get("gpu") if isinstance(status, dict) else None
    if not isinstance(gpu, dict):
        return {
            "ok": False,
            "available": False,
            "reason": reason,
            "gpus": [],
            "checkedAt": datetime.now(timezone.utc).isoformat(),
        }
    total = int(gpu.get("total_mb") or 0)
    used = int(gpu.get("used_mb") or 0)
    free = int(gpu.get("free_mb") or max(0, total - used))
    source = str(gpu.get("source") or "scheduler")
    name = "CUDA GPU"
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {
        "ok": total > 0,
        "available": total > 0,
        "source": source,
        "fallbackReason": reason,
        "gpus": [
            {
                "index": 0,
                "name": name,
                "memoryTotalMiB": total,
                "memoryUsedMiB": used,
                "memoryFreeMiB": free,
                "usagePercent": round((used / total) * 100, 1) if total else 0.0,
                "utilizationGpuPercent": 0,
            }
        ] if total > 0 else [],
        "checkedAt": datetime.now(timezone.utc).isoformat(),
    }


def _vram_status_payload() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return _scheduler_vram_status_payload("nvidia-smi not found")
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as error:
        return _scheduler_vram_status_payload(str(error))
    if result.returncode != 0:
        return _scheduler_vram_status_payload((result.stderr or result.stdout or "nvidia-smi failed").strip())
    gpus = _parse_nvidia_smi_rows(result.stdout)
    return {
        "ok": bool(gpus),
        "available": bool(gpus),
        "source": "nvidia-smi",
        "gpus": gpus,
        "checkedAt": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/vram-status", dependencies=_GATED)
@app.get("/v1/vram-status", dependencies=_GATED)
def vram_status() -> dict[str, Any]:
    return _vram_status_payload()


def _log_window_command(log_type: str) -> str:
    if log_type == "quota":
        endpoint = "http://127.0.0.1:8766/v1/quota-status"
        title = "Manga Translator Quota Monitor"
        interval = 5
    elif log_type == "vram":
        endpoint = "http://127.0.0.1:8766/v1/vram-status"
        title = "Manga Translator VRAM Monitor"
        interval = 2
    else:
        raise HTTPException(status_code=400, detail="Unsupported log window type")
    return (
        f"$Host.UI.RawUI.WindowTitle='{title}'; "
        f"$fmtHeaders = @{{'{EXTENSION_CLIENT_HEADER}'='{EXTENSION_CLIENT_VALUE}'}}; "
        f"while ($true) {{ Clear-Host; "
        f"Write-Host '{title}' -ForegroundColor Cyan; "
        f"Write-Host 'Endpoint: {endpoint}'; "
        f"Write-Host ''; "
        f"try {{ Invoke-RestMethod '{endpoint}' -Headers $fmtHeaders | ConvertTo-Json -Depth 8 }} "
        f"catch {{ Write-Host $_.Exception.Message -ForegroundColor Red }}; "
        f"Start-Sleep -Seconds {interval}; }}"
    )


@app.post("/api/open-log-window/{log_type}", dependencies=_GATED)
@app.post("/v1/open-log-window/{log_type}", dependencies=_GATED)
def open_log_window(log_type: str) -> dict[str, Any]:
    command = _log_window_command(log_type)
    try:
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NoExit", "-Command", command],
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )
    except Exception as error:
        write_diagnostic_event(
            "backend.open_log_window.error",
            {"logType": log_type, "error": str(error)},
            source="backend",
            level="error",
        )
        raise HTTPException(
            status_code=500,
            detail={"code": "LOG_WINDOW_ERROR", "message": "Failed to open log window."},
        ) from error
    return {"success": True, "logType": log_type}


@app.post("/translate", response_model=TranslateResponse, dependencies=_GATED)
@app.post("/v1/translate-image", response_model=TranslateResponse, dependencies=_GATED)
@app.post("/v1/translate-snapshot", response_model=TranslateResponse, dependencies=_GATED)
def translate_image(request: TranslateRequest) -> TranslateResponse:
    return _execute_translate(request)


@app.post("/v1/batch", response_model=BatchResponse, dependencies=_GATED)
def translate_batch(request: BatchRequest) -> BatchResponse:
    # Each _execute_translate call already blocks on PIPELINE_SCHEDULER.acquire(), the real
    # VRAM-adaptive concurrency cap (see gpu_scheduler.py). Dispatching the batch across a small
    # thread pool -- rather than one item fully finishing before the next starts -- lets a batch
    # inherit that same cap instead of getting strictly sequential (1x) throughput regardless of
    # how much scheduler capacity is actually free. ThreadPoolExecutor.map preserves input order
    # and re-raises the first failing item's exception at that item's position when iterated,
    # matching the previous sequential comprehension's fail-fast behavior on error.
    worker_count = max(1, min(len(request.images), 8))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="fmt-batch") as pool:
        results = list(pool.map(_execute_translate, request.images))
    return BatchResponse(status="pass", results=results)


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse, dependencies=_GATED)
def job_status(job_id: str) -> JobStatusResponse:
    record = JOBS.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        jobId=job_id,
        status=record.get("status", "unknown"),
        report=record.get("report", {}),
        artifacts=record.get("artifacts", {}),
        error=record.get("error"),
    )
