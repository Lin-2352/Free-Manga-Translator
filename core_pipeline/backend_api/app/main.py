from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .pipeline_bridge import PipelineRunError, run_pipeline_payload
from .schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    JobStatusResponse,
    TranslateRequest,
    TranslateResponse,
)


API_VERSION = "0.1.0-local-8stage"

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
    allow_headers=["Content-Type", "Authorization"],
)

JOBS: dict[str, dict[str, Any]] = {}


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


def _execute_translate(request: TranslateRequest) -> TranslateResponse:
    job_id = str(uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    JOBS[job_id] = {
        "jobId": job_id,
        "status": "running",
        "report": {"startedAt": started_at},
        "artifacts": {},
    }

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
        JOBS[job_id] = response.model_dump() if hasattr(response, "model_dump") else response.dict()
        return response
    except HTTPException:
        JOBS[job_id]["status"] = "fail"
        raise
    except (PipelineRunError, Exception) as error:
        JOBS[job_id] = {
            "jobId": job_id,
            "status": "fail",
            "report": JOBS[job_id].get("report", {}),
            "artifacts": JOBS[job_id].get("artifacts", {}),
            "error": str(error),
        }
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/health", response_model=HealthResponse)
@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        service="free-manga-translator-local-pipeline-api",
        mode="local-only-strict",
        version=API_VERSION,
    )


@app.post("/translate", response_model=TranslateResponse)
@app.post("/v1/translate-image", response_model=TranslateResponse)
@app.post("/v1/translate-snapshot", response_model=TranslateResponse)
def translate_image(request: TranslateRequest) -> TranslateResponse:
    return _execute_translate(request)


@app.post("/v1/batch", response_model=BatchResponse)
def translate_batch(request: BatchRequest) -> BatchResponse:
    results = [_execute_translate(item) for item in request.images]
    return BatchResponse(status="pass", results=results)


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
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
