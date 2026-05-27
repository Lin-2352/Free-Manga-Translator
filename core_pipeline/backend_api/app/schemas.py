from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    imageData: str | None = None
    base64Data: str | None = None
    sourceLanguage: str = "ja"
    targetLanguage: str = "en"
    qualityProfile: str = "strict"
    requestedOutput: str = "translatedImageDataUrl"
    clientRequestId: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranslateResponse(BaseModel):
    jobId: str
    status: str
    translatedImageDataUrl: str | None = None
    imageDataUrl: str | None = None
    translations: list[dict[str, Any]] = Field(default_factory=list)
    report: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class BatchRequest(BaseModel):
    images: list[TranslateRequest]


class BatchResponse(BaseModel):
    status: str
    results: list[TranslateResponse]


class JobStatusResponse(BaseModel):
    jobId: str
    status: str
    report: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class HealthResponse(BaseModel):
    ok: bool
    service: str
    mode: str
    version: str
