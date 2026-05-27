# Free Manga Translator

Offline-first manga/manhwa/manhua page translator with an 8-stage local pipeline and a Chrome/Brave extension.

## What This Repo Contains

- Local backend API (`FastAPI`) that runs the full 8-step image pipeline.
- Browser extension that sends detected manga panels to the local API and overlays translated output.
- Required pipeline code for OCR, layouting, translation, inpainting, and typesetting.
- Required local model assets tracked with Git LFS.
- Minimal docs for architecture and usage.

## Repository Structure

- `core_pipeline/backend_api`: local API service.
- `core_pipeline/extension`: browser extension.
- `core_pipeline/python/common`: shared pipeline primitives.
- `core_pipeline/python/runtime`: runtime orchestrator.
- `core_pipeline/python/steps`: step 4-8 pipeline stages.
- `core_pipeline/models`: required model files.
- `examples`: sample input/output screenshots.

## Requirements

- Windows 10/11 with CUDA-capable GPU recommended.
- Python 3.11+.
- Git LFS installed (`git lfs install`).
- Browser: Chrome or Brave.

## Setup

1. Clone and pull LFS assets.

```powershell
git clone https://github.com/Lin-2352/Free-Manga-Translator.git
cd "Free Manga Translator"
git lfs pull
```

2. Create virtual environment and install dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Configure environment.

```powershell
copy core_pipeline\.env.example core_pipeline\.env
```

Fill provider keys in `core_pipeline/.env`.

## Run Local API

```powershell
$env:PYTHONIOENCODING='utf-8'
$py="${PWD}\.venv\Scripts\python.exe"
cd core_pipeline
& $py -m uvicorn backend_api.app.main:app --host 127.0.0.1 --port 8766
```

Health check:

```powershell
curl http://127.0.0.1:8766/v1/health
```

## Load Extension

1. Open `chrome://extensions` (or `brave://extensions`).
2. Enable Developer mode.
3. Click Load unpacked.
4. Select `core_pipeline/extension`.
5. Open a manga page and click Translate.

## Sample Outputs

### Sample 4

Input:

![Sample 4 Input](examples/sample4/input.jpg)

Step 8 Output:

![Sample 4 Output](examples/sample4/output_step8.jpg)

### Sample 5

Input:

![Sample 5 Input](examples/sample5/input.jpg)

Step 8 Output:

![Sample 5 Output](examples/sample5/output_step8.jpg)

### Sample 6

Input:

![Sample 6 Input](examples/sample6/input.jpg)

Step 8 Output:

![Sample 6 Output](examples/sample6/output_step8.jpg)

## Notes

- This repository intentionally excludes QA reports, validation datasets, checkpoints, diagnostics, and runtime caches.
- Do not commit `core_pipeline/.env`.
