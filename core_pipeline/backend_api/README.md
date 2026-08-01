# Backend API

The backend API is the local companion service used by the browser extension. It wraps the validated 8-stage Python pipeline and returns final Step 8 translated images.

## Start

From `core_pipeline/`:

```powershell
$env:PYTHONIOENCODING='utf-8'
$py='<path-to-your-python.exe>'
& $py -m uvicorn backend_api.app.main:app --host 127.0.0.1 --port 8766
```

Health check:

```powershell
Invoke-RestMethod "http://127.0.0.1:8766/v1/health"
```

The companion starts a background model warmup automatically unless `FMT_STARTUP_WARMUP=0` is set. Warmup loads the OCR, detection, LaMa, and typesetting resources before the first real browser request when possible.

Runtime memory controls:

```powershell
$env:FMT_STARTUP_WARMUP='0'            # Lazy-load models only when a request needs them
$env:FMT_GPU_IDLE_UNLOAD_SECONDS='600' # Auto-release idle GPU models after 10 minutes
```

Manual warmup:

```powershell
Invoke-RestMethod "http://127.0.0.1:8766/v1/warmup" -Method Post
```

## Endpoints

- `GET /v1/health`
- `GET|POST /v1/warmup`
- `POST /v1/translate-image`
- `POST /v1/translate-snapshot`
- `POST /v1/batch`
- `GET /v1/jobs/{job_id}`
- `POST /v1/runtime/soft-stop`
- `POST /v1/runtime/hard-stop`
- `POST /v1/gpu/release`
- `POST /translate` compatibility alias

## Request

```json
{
  "imageData": "data:image/png;base64,...",
  "sourceLanguage": "ja",
  "targetLanguage": "en",
  "qualityProfile": "strict",
  "requestedOutput": "translatedImageDataUrl"
}
```

`sourceLanguage` accepts `ja`, `ko`, or `zh`. The validated target language is currently English.

## Runtime Behavior

The API writes each request as a runtime sample, runs the local pipeline, checks output safety, and returns the final Step 8 image. Pages with no renderable text return a successful no-text report instead of a `500`, while real missing-artifact and placeholder-translation failures still block the response.

The backend process reuses loaded model handles where supported by the pipeline modules. Startup warmup reduces first-request latency, and exact repeated image bytes can reuse the existing validated runtime output.

## Adaptive GPU Scheduling

The API can run more than one browser page pipeline at the same time when VRAM headroom is available. The scheduler probes GPU memory with `nvidia-smi` first, then `torch.cuda.mem_get_info()`, and falls back to sequential mode if memory cannot be measured safely.

Defaults:

- `FMT_PIPELINE_MAX_PARALLEL=2`
- `FMT_GPU_RESERVED_FREE_MB=2048`
- `FMT_PIPELINE_JOB_VRAM_MB=3072`

With the defaults, an 8 GB card will stop admitting extra concurrent jobs once free VRAM drops below roughly one estimated job plus the 2 GB safety reserve. Health responses include the current scheduler mode, active jobs, capacity, and GPU memory snapshot.

Telemetry endpoints:

- `GET /v1/quota-status` returns local API-manager provider state without calling external providers.
- `GET /v1/vram-status` returns local GPU memory usage through `nvidia-smi` when available.
- `POST /v1/runtime/soft-stop` requests active pipeline cancellation while keeping loaded models resident.
- `POST /v1/runtime/hard-stop` requests active pipeline cancellation and defers GPU release until active stages finish.
- `POST /v1/gpu/release` unloads cached OCR/detection/inpainting/font handles and clears CUDA cache where possible.
- `POST /v1/open-log-window/quota` and `POST /v1/open-log-window/vram` open separate local PowerShell monitor windows for quota or VRAM polling.

Generated runtime files are written under:

```text
core_pipeline/runtime_samples/
```

These files are ignored by Git.
