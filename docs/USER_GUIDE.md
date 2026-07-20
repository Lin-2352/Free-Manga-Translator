# User Guide

No GPU, or want to skip keeping a local server open? See `docs/KAGGLE_USER_MANUAL.md`
for running the backend on Kaggle's free GPU instead.

## Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy core_pipeline\.env.example core_pipeline\.env
```

## Run API

```powershell
$env:PYTHONIOENCODING='utf-8'
$py="${PWD}\.venv\Scripts\python.exe"
cd core_pipeline
& $py -m uvicorn backend_api.app.main:app --host 127.0.0.1 --port 8766
```

## Load Extension

- Open browser extension page.
- Enable Developer mode.
- Load unpacked folder: `core_pipeline/extension`.

## Translate

- Open a manga page.
- Click extension popup.
- Click Translate.
- Use Auto Translate if needed.

## Stop

- Close terminal or press `Ctrl+C` in API terminal.
