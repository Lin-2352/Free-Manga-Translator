# Browser Extension

This directory contains the Chrome/Brave extension for Manga Translator.

The extension does not run OCR, inpainting, model inference, or provider API calls in the browser. It captures manga images, sends them to the local FastAPI companion, receives the final Step 8 translated image, and displays it in the page.

## Default Endpoint

```text
http://127.0.0.1:8766/v1/translate-image
```

The local companion must be running before the extension can translate images.

## Load Unpacked

1. Open `chrome://extensions` or `brave://extensions`.
2. Enable **Developer mode**.
3. Click **Load unpacked**.
4. Select this `extension/` directory.
5. Open the extension popup.
6. Set the local pipeline URL to `http://127.0.0.1:8766/v1/translate-image`.
7. Select source language: `ja`, `ko`, or `zh`.

Full user guide: `../../docs/USER_GUIDE.md`.

## Extension Flow

```text
content.js
  Finds eligible images or captures a selected region.

background.js
  Queues local pipeline requests, dispatches bounded parallel jobs, and sends strict metadata.

local FastAPI companion
  Runs the 8-stage Python pipeline and returns a translated image.

content.js / translationPanel.js
  Replaces the image or displays the translated crop.
```

## Runtime Controls

The popup includes:

- **Start Engine** to verify the local companion and request `/v1/warmup` when the backend is reachable. If it is not reachable, the popup shows the manual `start_backend.ps1` steps because Chrome/Brave cannot launch Python directly without a native messaging host.
- **Translate Page** to manually scan the active page without enabling auto-translate.
- **Soft Stop** to stop active/queued extension requests while keeping backend models resident for a faster resume.
- **Hard Stop** to stop active/queued extension requests and ask the backend to release GPU model memory. It does not kill the Python server process.
- **Resume** to unpause the extension; it performs a one-shot scan unless Auto Translate is enabled.
- **Auto Translate** for explicit continuous scanning after the user turns it on.
- **Queue ahead** to cap how many pending pages/images Auto Translate can keep waiting behind the active local pipeline job.
- **Parallel jobs** to allow two or three active local requests while the backend scheduler enforces VRAM safety.
- **Provider Quota** as a collapsed telemetry panel. It reads backend quota-manager state only; it does not call providers or spend API credits.
- **GPU VRAM** as a collapsed telemetry panel. It queries the local companion and reports `nvidia-smi` memory usage when available. Its **Release GPU** button unloads backend model handles without closing the Python server.
- **Open Log** buttons for quota and VRAM monitors. These ask the local companion to open a separate PowerShell monitor window so translation logs stay focused on pipeline errors and timings.
- **Keep in cache** to choose how many recent translated images are retained for the browser session.
- **Clear Cache** to remove browser cached translations and ask the backend to clear reusable runtime Step 8 outputs.

Cached results must be keyed from the original source image identity, not from the currently displayed image element. This prevents a translated Step 8 output from being sent back through OCR as a new manga source after back/forward navigation.

Cached results are temporary browser-session data. Active background jobs should continue even if the user leaves the page, then apply from cache when the user returns.

The popup shows the cache count inside the **Recent Translations** section. Cache hits are restored without a server health round-trip so back/forward browsing can display translated images immediately.

The background worker uses a full deterministic cache hash and bumps the cache version when pipeline behavior changes. Health checks also trigger the companion warmup endpoint, so the first translation after opening the popup is less likely to pay full model-load cost. Clear Cache and Re-translate now also call `/v1/cache/clear`, which removes backend reusable `runtime_*` output artifacts so fixed pipeline logic is not hidden by stale Step 8 output.

The extension can dispatch multiple active jobs, but the backend remains authoritative. If available VRAM is below the configured safety threshold, the backend queues internally and effectively runs sequentially. The queue-ahead limit is intentionally bounded to prevent Chrome/Brave from collecting dozens of pages and exhausting GPU memory, API quota, or browser service-worker lifetime. Setting it to **Current only** disables waiting jobs while still allowing the current visible page to translate.

The queue-ahead and parallel-job controls include hover info icons. Use them as operating guidance: `Current only` and sequential mode are safest for debugging, `5` queued pages and `2` parallel jobs are the normal balanced setting, and higher values are intentionally marked heavy because they can keep GPU and provider work queued for longer.

Translation Control buttons also expose hover/focus help inside the popup. The help is rendered as a floating list so it does not permanently take popup space.

Queue and cache state are keyed by the original image URL plus original dimensions. A translated data URL is never allowed to become the source cache key, which prevents back/forward navigation from re-queuing the same three reader pages repeatedly.

## Activation Reliability

The popup does not assume `content.js` is already present in the active tab. Translate, resume, re-translate, clear, and context-menu actions route through the background worker, which injects `content.js` with `chrome.scripting.executeScript` when needed before sending content-script commands.

Manual translation and auto-translation are intentionally separate:

- `Translate Page` sends `translatePageOnce`.
- The Auto toggle is off by default and is the only control that persists `translationEnabled=true`.
- Passive cache restoration can still display existing cached translations while Auto is off.

Captured browser images are sent at up to `2000px` on the longest edge with higher JPEG quality. This is a deliberate accuracy/speed balance: it keeps local requests manageable while reducing compression damage around small CJK text and line art.

## Request Contract

The background worker sends:

```json
{
  "imageData": "data:image/png;base64,...",
  "sourceLanguage": "ja",
  "targetLanguage": "en",
  "qualityProfile": "strict",
  "requestedOutput": "translatedImageDataUrl",
  "metadata": {
    "cacheKey": "original-image-url|widthxheight",
    "originalImageUrl": "https://...",
    "pageCacheKey": "https://example.com/chapter/1"
  }
}
```

The companion returns:

```json
{
  "status": "pass",
  "translatedImageDataUrl": "data:image/png;base64,...",
  "report": {
    "pipeline": "local-8-stage",
    "layoutConstraints": 5,
    "translations": 5,
    "renderedRegions": 5
  }
}
```

## Diagnostics

From `core_pipeline/`:

```powershell
node "extension\tests\background_contract_test.mjs"
node "extension\tests\popup_contract_test.mjs"
node "extension\tests\content_contract_test.mjs"
```

Expected:

```text
extension_background_contract=pass
```

Run the backend and live-pipeline checks:

```powershell
$py='<path-to-your-python.exe>'
& $py "python\diagnostics\test_extension_backend_contract.py"
& $py "python\diagnostics\run_extension_live_smoke.py" --image "samples\sample1\sample.jpg" --source-language ja
```

## Security

- Do not place provider keys in extension files.
- Keep keys in `core_pipeline/.env`.
- The extension sends requests only to the configured local pipeline URL.
- CORS/canvas failures are handled by delegating image fetches to the background service worker.
