# Running the pipeline on Kaggle's free GPU (no NVIDIA GPU required locally)

This document explains, in full detail, how to run the heavy 8-step manga translation
pipeline on a **Kaggle notebook** (free NVIDIA T4 ×2 GPU, 30 hours/week quota) while
your Chrome extension keeps running locally in your normal browser, exactly as it does
today. The extension talks to the Kaggle-hosted backend over an HTTPS tunnel instead of
`127.0.0.1`. Nothing about how you use the extension changes — you paste one URL into
the popup once, and everything else (queue, cache, auto-translate, quota dashboard, the
offline circuit breaker) works unmodified.

If you only want the short version: skip to [Quick start](#quick-start). Everything
after that is the "extreme detail" the rest of this document promises — read it before
you put real API keys anywhere.

---

## Table of contents

1. [Architecture](#1-architecture)
2. [Prerequisites](#2-prerequisites)
3. [Getting the private code onto Kaggle](#3-getting-the-private-code-onto-kaggle)
4. [Kaggle Secrets setup](#4-kaggle-secrets-setup)
5. [The notebook, cell by cell](#5-the-notebook-cell-by-cell)
6. [Pointing the extension at Kaggle](#6-pointing-the-extension-at-kaggle)
7. [Operational reality — what to actually expect](#7-operational-reality--what-to-actually-expect)
8. [Security & leak analysis](#8-security--leak-analysis)
9. [Cloudflare Tunnel — the zero-account fallback](#9-cloudflare-tunnel--the-zero-account-fallback)
10. [Troubleshooting](#10-troubleshooting)
11. [Quick start](#quick-start)

---

## 1. Architecture

```
┌─────────────────────────┐        HTTPS (public internet)        ┌──────────────────────────────────────┐
│   Your computer          │                                       │   Kaggle notebook VM (ephemeral)       │
│                          │                                       │                                        │
│  Chrome extension        │  POST https://you.ngrok-free.app/     │  ngrok client (or cloudflared)          │
│  (background.js,         │──────────/v1/translate-image────────▶│  forwards → 127.0.0.1:8766              │
│   content.js, popup)     │                                       │        │                               │
│                          │                                       │        ▼                               │
│  Local Pipeline URL =    │                                       │  uvicorn (backend_api.app.main:app)     │
│  https://you.ngrok       │                                       │        │                               │
│  -free.app/v1/           │                                       │        ▼                               │
│  translate-image         │                                       │  8-step pipeline (steps 1-8)            │
│                          │                                       │  detect → OCR → translate → inpaint     │
│  Everything else         │                                       │  → typeset, running on the GPU(s)       │
│  unchanged: queue,       │                                       │        │                               │
│  cache, auto-translate,  │                                       │        ▼                               │
│  offline breaker, quota  │                                       │  NVIDIA T4 (×2, only GPU 0 used          │
│  dashboard               │                                       │  by default — see §7)                   │
└─────────────────────────┘                                       └──────────────────────────────────────┘
```

The extension already has **zero-code-change support** for this — verified by reading
the actual extension source, not assumed:

- `manifest.json` declares `"host_permissions": ["<all_urls>"]`, so the extension's
  background service worker is permitted to fetch **any** HTTPS origin, not just
  `127.0.0.1`. A tunnel URL introduces no new permission problem.
- Every backend call the extension makes — translate, health, warmup, quota, VRAM,
  diagnostics, soft/hard-stop, cache-clear — is derived from **one single setting**,
  `Local Pipeline URL`, by swapping only the path (`background.js`'s
  `healthUrlForPipeline`, `warmupUrlForPipeline`, `backendUrlForPipeline`, etc., all
  read the origin from that one value via `new URL(...)`). Change that one field and
  every feature follows it automatically.
- The backend's CORS policy (`backend_api/app/main.py`) matches on `Origin:
  chrome-extension://...`, not on the destination host — so a tunnel target passes
  CORS preflight exactly like `127.0.0.1` does today. No server-side CORS change is
  needed for this deployment.
- The extension already has an **optional auth-token field** (`Backend auth token` in
  the popup, under Local Pipeline settings) that layers a shared-secret header
  (`X-Fmt-Auth`) on top of the existing `X-Fmt-Client` gate. This is exactly what you
  want once your backend is reachable from the public internet, and it already exists
  in the codebase — this document tells you how to turn it on for this deployment (§4,
  §6, §8), not how to build it.

**What does NOT work remotely, by design, and why that's fine:**

- `/v1/open-log-window/*` (the popup's "Open Quota Log" / "Open VRAM Log" buttons)
  spawns a PowerShell console **on the machine running the backend**. On a headless
  Kaggle VM this has no meaningful effect. The popup already catches the failure
  gracefully (you'll see a status message, not a broken UI) — you just won't get a
  pop-up terminal window, because there's no desktop on the other end to show it on.

---

## 2. Prerequisites

1. **A phone-verified Kaggle account.** Kaggle requires phone verification before it
   will grant GPU access at all (Settings → Phone Verification). Without this, the GPU
   accelerator option is greyed out.
2. **A free ngrok account** (recommended path — see §9 for the zero-account
   alternative):
   - Sign up at ngrok.com, free tier.
   - Copy your **authtoken** from the ngrok dashboard (Your Authtoken page). You'll
     store this as a Kaggle Secret, never in notebook code.
   - Claim your **one free static domain** (ngrok dashboard → Domains → "Create
     Domain"). This gives you a permanent hostname like
     `yourname-something.ngrok-free.app` that never changes between sessions — you
     paste it into the extension **once, ever**, instead of every time you start the
     notebook. This is the single biggest quality-of-life win in this whole setup.
3. **The code, bundled for Kaggle.** Your repository is **private** (verified via `gh
   repo view` before writing this document). §3 below walks through the recommended
   way to get it onto Kaggle without ever putting a long-lived GitHub credential there.
4. **Your `.env` file**, the same one your local backend already uses, with your real
   API provider keys. You are not creating new keys — you're reusing the ones you
   already have, delivered to Kaggle via Kaggle Secrets (§4), never committed or
   uploaded anywhere in plaintext.

---

## 3. Getting the private code onto Kaggle

Your repo is private, and it uses **Git LFS** for the vendored model weights
(`core_pipeline/models/`, ~731MB: text/bubble detectors, inpainting models). Two ways
to get it onto Kaggle; **Option A is recommended** and is what the rest of this
document assumes.

### Option A (recommended): private Kaggle Dataset

Build a zip of the code **on your own machine**, where Git LFS is already resolved
(you already have the real files, not LFS pointers), then upload it as a **private**
Kaggle Dataset.

```powershell
# Run this on your own machine, inside the repo.
cd "D:\Desktop\translator D\app\Manga Translator"

# Build a clean copy for upload -- excludes secrets and every local-only dev/test
# artifact directory (quality_reports, runtime_samples, validation_logs, training_data,
# runtime_logs, extension) so the upload carries only what's actually needed to run the
# backend (code + vendored model weights: ~682MB zipped, verified by actually running
# this command), not the ~1GB of local test output that would otherwise get swept in
# alongside it. The extension itself never runs on Kaggle -- it stays local in your
# browser -- so it's excluded too.
robocopy core_pipeline kaggle_upload\core_pipeline /E /XD .git __pycache__ runtime_samples `
  .venv quality_reports validation_logs training_data runtime_logs extension `
  /XF .env "*.pyc"

Compress-Archive -Path kaggle_upload\core_pipeline -DestinationPath fmt_core_pipeline.zip -Force
```

This creates `fmt_core_pipeline.zip` in the directory you `cd`'d into above — i.e.
`D:\Desktop\translator D\app\Manga Translator\fmt_core_pipeline.zip` — right alongside
the `kaggle_upload\` staging folder `robocopy` builds it from. That's the file you
upload as the Kaggle Dataset in the next step.

Then on kaggle.com:

1. **Create → New Dataset**.
2. Upload `fmt_core_pipeline.zip`, give it a name (e.g. `fmt-core-pipeline`).
3. Set visibility to **Private** (this is the default for a new dataset — leave it
   alone, do not click "Make Public").
4. Click **Create**.

**Why this beats a GitHub token:** no credential of any kind is stored on Kaggle's
infrastructure at all — not even a scoped one. Updating the code later is just
uploading a **new dataset version** (same Dataset page → "New Version"), which is
simpler than juggling Git LFS pulls inside a notebook that has no persistent disk
between sessions anyway.

### Option B: fine-grained GitHub PAT (if you'd rather clone directly)

If you prefer live `git clone` inside the notebook instead of a dataset:

1. GitHub → Settings → Developer settings → **Fine-grained personal access tokens** →
   Generate new token.
2. Scope it to **this one repository only**, permission **Contents: Read-only**, and
   set an expiration (30-90 days — you'll rotate it, which is the point of a
   fine-grained token over a classic PAT with broad scope).
3. Store it as a Kaggle Secret named `GH_PAT` (§4).
4. In the notebook: `git clone https://<token>@github.com/Lin-2352/Manga-Translator.git`
   then `git lfs pull` (Kaggle's base image includes `git-lfs`; if a specific notebook
   image doesn't, `apt-get install git-lfs -y` first).

This works, but carries a real (if minimized) credential-exposure surface that Option A
avoids entirely: the token lives in Kaggle Secrets (encrypted at rest, same as any
other secret — see §4), but every notebook run touches it, and if you ever
accidentally `!echo $GH_PAT` or the clone command errors and echoes the URL to output,
it leaks into the notebook's saved output. **Prefer Option A unless you have a specific
reason to want live git history on Kaggle.**

---

## 4. Kaggle Secrets setup

Kaggle Secrets (Notebook editor → **Add-ons → Secrets**) are the correct place for your
API keys. Directly answering the question this document exists to answer: **yes, this
is safe to do**, with the caveats below.

**Why it's safe:** Kaggle Secrets are encrypted at rest, are not visible in the
notebook's source (they're referenced by name via `kaggle_secrets.UserSecretsClient`,
never pasted as literal text), are **not** included when you share or fork a notebook
(a person who copies your notebook does not get your secrets — they'd need to add
their own), and are scoped per-notebook (you explicitly attach a secret to a given
notebook before it's readable there).

**The real leak vectors — a "never do" list, because the mechanism being safe doesn't
protect you from your own notebook code:**

- **Never `print()` a secret value**, not even partially, not even for "debugging" —
  Kaggle's output cells are saved with the notebook version.
- **Never write a secret into a file that ends up in notebook Output** (Kaggle
  auto-saves everything under `/kaggle/working` as output when you "Save Version" with
  outputs). Write the decoded `.env` to a path you're confident isn't captured, or
  delete it in your shutdown cell.
- **Never make the notebook public** while a Secret is attached — public notebooks
  with attached secrets are exactly the pattern Kaggle's own docs warn about; the code
  itself would be fine to share, the *live run* with your keys attached would not.
- **Never dump environment variables wholesale** (`!env`, `os.environ` printed in
  full) once you've loaded the `.env` into the process — filter to just the keys you
  need to see, or better, don't print environment state at all in a saved cell.

**Secrets to create** (Add-ons → Secrets → Add a new secret, one at a time):

| Secret name | Value | Used for |
|---|---|---|
| `FMT_ENV_B64` | Base64 of your entire local `.env` file (command below) | All your provider API keys, in one shot — no per-key drift between your local `.env` and Kaggle |
| `NGROK_AUTHTOKEN` | Your ngrok authtoken | Authenticates the tunnel client |
| `FMT_AUTH_TOKEN` | A password you make up (e.g. output of `openssl rand -hex 24`) | The shared secret the extension and backend use to authenticate each other over the public tunnel (§6, §8) |
| `GH_PAT` (only if using Option B in §3) | Your fine-grained PAT | Cloning the private repo |

To produce the `FMT_ENV_B64` value, run **on your own machine**:

```powershell
# PowerShell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("D:\Desktop\translator D\app\Manga Translator\core_pipeline\.env")) | Set-Clipboard
```

This copies the base64 text to your clipboard — paste it directly as the Secret value.
Nothing is printed to any terminal or log.

**Why one base64 blob instead of one Kaggle Secret per API key:** your `.env` already
has ~9 providers × up to 4 keys each, plus model overrides, quota settings, and now
`FMT_AUTH_TOKEN`. One secret that's decoded straight back into a real `.env` file on
the Kaggle side means the notebook, `api_manager.py`, and your local machine are all
reading literally the same file format — no risk of a Kaggle-side key list silently
drifting out of sync with your local one as you rotate keys over time.

---

## 5. The notebook, cell by cell

This mirrors `deploy/kaggle/fmt_kaggle_backend.ipynb`, committed in this repository —
you can copy that file directly into a new Kaggle notebook, or type these cells in
yourself. Each cell is explained here; expected timings are in §7.

**Before running anything:** in the notebook's right-side panel, set **Accelerator →
GPU T4 x2**, and toggle **Internet → On** (required for pip installs, HF model
downloads, and the tunnel itself). Attach your `fmt-core-pipeline` dataset (Add-ons →
Data → your dataset). Attach the four secrets from §4 (Add-ons → Secrets → toggle each
one on for this notebook).

**Cell 1 — environment sanity check.** Confirms GPU is actually visible, the dataset is
attached, and you didn't forget a toggle before burning notebook quota on nothing:
```python
import subprocess, os
assert os.path.isdir("/kaggle/input/fmt-core-pipeline"), \
    "Dataset not attached -- Add-ons -> Data -> attach your fmt-core-pipeline dataset"
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                      capture_output=True, text=True).stdout)
```
Expect two lines naming `Tesla T4, 15360 MiB` (or similar). If this cell errors or
shows no GPUs, the accelerator setting wasn't applied — fix it in the side panel and
restart the session before continuing.

**Cell 2 — copy code + install dependencies.** Kaggle's disk under `/kaggle/input` is
read-only; copy to `/kaggle/working` first. Kaggle's base image already ships a recent
torch build — check it before reinstalling the pinned CUDA wheel, since a fresh
`pip install torch==2.6.0 --index-url .../cu124` costs several minutes you can usually
skip:
```python
import shutil, subprocess, sys, torch

shutil.copytree("/kaggle/input/fmt-core-pipeline/core_pipeline",
                 "/kaggle/working/core_pipeline", dirs_exist_ok=True)
%cd /kaggle/working/core_pipeline

need_torch_reinstall = not (torch.__version__.startswith("2.6") and torch.cuda.is_available())
if need_torch_reinstall:
    subprocess.run([sys.executable, "-m", "pip", "install",
                     "torch==2.6.0", "torchvision==0.21.0",
                     "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-r", "python/requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "backend_api/requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "pyngrok"], check=True)
```
`python/requirements.txt` and `backend_api/requirements.txt` are two separate files in
this repo (the pipeline's ML deps vs. the FastAPI layer) — both are needed; the
repo's own `backend_api/Dockerfile` only installs the second one and is not a complete
reference for this deployment.

**Cell 3 — decode secrets into environment.** Reads `FMT_ENV_B64` from Kaggle Secrets,
decodes it back into a real `.env` file, and points the backend at it via
`FMT_ENV_FILE` — the exact override mechanism `api_manager.py`'s `_load_env` already
supports, read at startup before every other candidate path. `FMT_AUTH_TOKEN` is set
directly as a process environment variable (the backend reads it once at import time),
and — critically — **nothing here is printed**:
```python
import base64, os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
env_bytes = base64.b64decode(secrets.get_secret("FMT_ENV_B64"))
env_path = "/kaggle/working/fmt.env"
with open(env_path, "wb") as f:
    f.write(env_bytes)
os.chmod(env_path, 0o600)

os.environ["FMT_ENV_FILE"] = env_path
os.environ["FMT_AUTH_TOKEN"] = secrets.get_secret("FMT_AUTH_TOKEN")
print("Secrets loaded (values not shown).")
```

**Cell 4 — launch the backend.** Runs uvicorn exactly the way `start_backend.ps1` does
locally (same working directory, same module path), as a background subprocess so the
notebook cell returns immediately and later cells can run:
```python
import subprocess, sys

backend_log = open("/kaggle/working/backend.log", "w")
backend_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend_api.app.main:app",
     "--host", "127.0.0.1", "--port", "8766"],
    cwd="/kaggle/working/core_pipeline",
    stdout=backend_log, stderr=subprocess.STDOUT,
)
print(f"Backend starting, pid={backend_proc.pid}. Logs: /kaggle/working/backend.log")
```
The bind stays on **127.0.0.1**, not `0.0.0.0` — the tunnel client (cell 6) runs on
this same VM and reaches the backend over loopback, so there is never a reason to bind
a wider interface here; doing so would only expand the Kaggle VM's own attack surface
for no benefit.

**Cell 5 — wait for health + force warmup.** Polls `/v1/health` until the process
actually accepts connections, then forces `/v1/warmup` (loads manga_ocr, the text/
bubble detectors, magi, the inpainting models, fonts) and polls until it reports
`pass`:
```python
import time, urllib.request, json

def get_json(url, method="GET"):
    req = urllib.request.Request(url, method=method,
                                  headers={"X-Fmt-Client": "free-manga-translator-extension"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)

for _ in range(30):
    try:
        get_json("http://127.0.0.1:8766/v1/health")
        break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError("Backend did not come up -- check /kaggle/working/backend.log")

get_json("http://127.0.0.1:8766/v1/warmup", method="POST")
for _ in range(60):
    status = get_json("http://127.0.0.1:8766/v1/health")["warmup"]["status"]
    print("warmup:", status)
    if status in ("pass", "fail"):
        break
    time.sleep(5)
```
See §7 for realistic timing (first-ever run vs. a warm cache).

**Cell 6 — open the tunnel.** Uses `pyngrok` with your static domain, so the public URL
is the same every session:
```python
from pyngrok import ngrok, conf
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
conf.get_default().auth_token = secrets.get_secret("NGROK_AUTHTOKEN")

# Replace with the exact static domain you claimed in the ngrok dashboard (§2).
STATIC_DOMAIN = "yourname-something.ngrok-free.app"
tunnel = ngrok.connect(8766, domain=STATIC_DOMAIN)
print("Public URL:", tunnel.public_url)
print("Paste this into the extension popup's Local Pipeline URL field:")
print(f"  {tunnel.public_url}/v1/translate-image")
```

**Cell 7 — keep the session alive.** Kaggle interactive sessions can idle-disconnect;
this cell just keeps the notebook actively running and periodically confirms the
backend is still healthy, so you have a live signal in the notebook output if
something crashes:
```python
import time
while True:
    try:
        status = get_json("http://127.0.0.1:8766/v1/health")
        print(time.strftime("%H:%M:%S"), "ok, warmup:", status["warmup"]["status"],
              "active jobs:", status["scheduler"]["active"])
    except Exception as e:
        print(time.strftime("%H:%M:%S"), "backend check failed:", e)
    time.sleep(60)
```
Interrupt this cell (■ Stop) when you're done reading and want to move to the shutdown
cell — it's an intentional infinite loop, not a bug.

**Cell 8 — clean shutdown.** Closes the tunnel, stops the backend, and removes the
decoded `.env` from disk so it doesn't linger in `/kaggle/working` if you save outputs:
```python
import os
ngrok.disconnect(tunnel.public_url)
backend_proc.terminate()
if os.path.exists("/kaggle/working/fmt.env"):
    os.remove("/kaggle/working/fmt.env")
print("Shut down cleanly.")
```

---

## 6. Pointing the extension at Kaggle

Once cell 6 has printed your public URL:

1. Open the extension popup → **Local Pipeline URL** field.
2. Paste the **full endpoint path**, not just the domain:
   `https://yourname-something.ngrok-free.app/v1/translate-image`
   (the field has **no validation** — a bare domain without `/v1/translate-image` will
   save without error but silently fail on the first real translate request, so get
   this exact).
3. Paste your `FMT_AUTH_TOKEN` value into the **Backend auth token** field (same popup
   panel, just below the URL field).
4. Click **Save**. This automatically re-runs the health check (`checkServerHealth()`)
   and the engine status badge should flip to **Reachable** within a second or two.

Because you claimed a **static** ngrok domain, this is a **one-time setup** — the URL
does not change between Kaggle sessions. Every future session, you only need to re-run
the notebook (cells 1-6); the extension side needs nothing further, unless you rotate
your `FMT_AUTH_TOKEN`.

**What works remotely vs. what doesn't**, so you're not surprised:

| Popup feature | Works remotely? |
|---|---|
| Manual translate / auto-translate / queue-ahead | Yes — unchanged |
| Cache (view size, clear) | Yes |
| Pause / soft-stop / hard-stop | Yes |
| Quota dashboard (`/v1/quota-status`) | Yes — arguably more useful here, since it's the Kaggle-side key usage |
| VRAM dashboard (`/v1/vram-status`) | Yes — shows the actual T4's memory, via `nvidia-smi` on the Kaggle VM |
| Offline "will resume automatically" badge | Yes — see §7 for tunnel-blip behavior |
| "Open Quota/VRAM Log" buttons | **No** — spawns a PowerShell window on the backend host, meaningless on headless Kaggle; fails gracefully with a status message, no crash |
| "Start it from PowerShell" hint text | Not applicable — you started it from the notebook, not PowerShell |

---

## 7. Operational reality — what to actually expect

**Kaggle GPU quota:** 30 GPU-hours per week (resets weekly), and a **~12-hour maximum
continuous session** even within quota. Plan reading sessions accordingly; the notebook
does not auto-restart itself when a session ends — you'll need to re-run it.

**Idle behavior:** Kaggle can disconnect an interactive session that appears idle in
the browser tab. Keep the notebook tab open and occasionally check it; cell 7's
keep-alive loop helps but is not a guarantee against Kaggle's own idle policy.

**Cold-start timing** (first run on a fresh session, nothing cached):
- pip installs (cell 2): **5-10 minutes**, most of it Paddle/PaddleOCR and (if
  needed) the pinned CUDA torch wheel.
- Model downloads on first warmup (magi, manga-ocr, EasyOCR/PaddleOCR packs from
  HuggingFace/package caches): **5-15 minutes**, network-dependent.
- Warmup itself once weights are on disk: **2-5 minutes**, this session's own
  local verification measured **~100 seconds** with weights already cached on disk —
  budget for the higher end (or more) on a truly fresh Kaggle session downloading
  everything for the first time.

**Speeding up subsequent sessions:** after your first successful run, you can bake the
downloaded HuggingFace/EasyOCR/PaddleOCR caches into a **new version of your Kaggle
dataset** (zip `~/.cache/huggingface`, `~/.EasyOCR`, `~/.paddleocr` from
`/kaggle/working` alongside the code, re-upload as dataset v2). Every session after
that skips the 5-15 minute download step entirely — this is the single biggest
speed-up available and worth doing once you're past initial setup.

**Per-page latency:** pipeline processing time (same as your local machine, likely
faster — a T4 is a real datacenter GPU) **plus** the round-trip over the tunnel for the
image upload and translated-image download. For typical manga page sizes this tunnel
overhead is small compared to pipeline time, but it is not zero — expect slightly
higher latency than a purely local setup, especially on a slower home connection.

**Tunnel blips and the offline circuit breaker:** the extension's circuit breaker
(`background.js`) trips on a genuine network-level failure (`TypeError` from a failed
`fetch()` — the tunnel dropped, ngrok restarted, your notebook session ended) and
silently short-circuits new requests for up to **12 seconds** before it re-probes.
This is expected, self-recovering behavior, not a bug: if your Kaggle session is
genuinely still running and the tunnel reconnects, translation resumes on its own
within one probe cycle, with no spinner or error badge shown on the page (by design —
see the extension's offline-handling behavior). If the Kaggle session actually ended,
the breaker will just keep silently retrying every 12 seconds until you restart the
notebook and get a new health-check success.

**Both T4s — the honest truth:** the pipeline's GPU-scheduling code
(`gpu_scheduler.py`) is **single-GPU admission control**; every model load in this
codebase is hardcoded to `cuda:0`. On Kaggle's 2×T4 offering, **only the first GPU is
used** by the setup in §5 — the second one sits idle. This is not a bug you need to
work around for normal use; `FMT_PIPELINE_MAX_PARALLEL` (default 2) already lets a
single T4 (16GB) run 2 pipeline jobs concurrently, which is plenty for one person's
reading pace.

*Advanced/untested appendix — using the second GPU:* it's possible in principle to
launch a **second** backend process with `CUDA_VISIBLE_DEVICES=1` bound to a different
local port, and open a **second** ngrok tunnel/domain to it, effectively giving
yourself two independent backend instances the extension could round-robin between.
This repository does not implement or test that split — it would need its own
notebook cells and a second popup URL mechanism the extension doesn't currently have
(today's extension talks to exactly one backend URL). Treat this as a documented idea
for a future enhancement, not a supported path.

---

## 8. Security & leak analysis

**Is your pipeline code exposed?** No, under the setup this document describes:
- The Kaggle **Dataset** holding your code is private (§3) — only your account can see
  or attach it.
- The Kaggle **Notebook** itself should also stay private (don't click "Make Public",
  don't submit it to a competition, don't add collaborators you don't trust with the
  code). A private notebook with a private dataset means nobody but you can see any of
  it.
- Caveat, stated plainly: like any cloud platform, Kaggle's own infrastructure/staff
  have the technical ability to access data on their systems, governed by their own
  ToS and privacy policy — this is a "trust the platform" tradeoff inherent to using
  any hosted notebook service, not something specific to this pipeline. If that's
  unacceptable for your threat model, this whole Kaggle path isn't the right fit and
  you should stick to local-only.

**Is your running backend exposed?** It's reachable at a **public URL** (that's the
entire point — your extension needs to reach it from wherever you're browsing), so
treat the tunnel URL itself as the real perimeter:

- Every route except the bare `/v1/health` check already requires the
  `X-Fmt-Client: free-manga-translator-extension` header (`main.py`'s
  `_require_extension_client` dependency) — this alone stops a random internet scanner
  from getting anywhere, since the header value has to be known and sent exactly.
- **With `FMT_AUTH_TOKEN` set** (which this document has you do — §4, §6), every gated
  route *additionally* requires a matching `X-Fmt-Auth` header, checked with
  `hmac.compare_digest` (constant-time comparison, not vulnerable to a timing side
  channel). **Set this token for this deployment.** A static ngrok domain is
  permanent and guessable-by-pattern in a way a random `*.trycloudflare.com` URL from
  §9 is not — the auth token is what actually keeps a stranger who stumbles on or
  guesses your domain from doing anything with it.
- Even in the worst case — someone gets both headers right and can issue real
  translate requests against your backend — the damage is bounded by your existing
  **API quota soft-caps** (`api_manager.py`): every provider key already has a daily
  token/request ceiling with an 80-90% soft-cap reservation margin, so a leaked tunnel
  cannot run your keys into a surprise negative balance or an unbounded bill; it would,
  at worst, burn through your existing daily quota faster than you'd like.
- `/v1/open-log-window/*` executing `powershell.exe` is a **Windows-only, local-only**
  code path in this codebase; on a Linux Kaggle VM this either fails outright or is a
  no-op — it is not a remote-code-execution surface introduced by this deployment
  (there's no PowerShell interpreter on the box to spawn in the first place).

**Kaggle Terms of Service note:** this document describes running your own personal
pipeline for your own interactive reading use, bounded by Kaggle's own quota and
session limits — not standing up a 24/7 public service on Kaggle's free tier. Review
Kaggle's current ToS yourself before relying on this for anything beyond that; this
document is not legal advice.

---

## 9. Cloudflare Tunnel — the zero-account fallback

If you don't want to create an ngrok account, `cloudflared` needs **no account and no
token** for a "quick tunnel" — but the tradeoff is the URL is **random and changes
every time you start it**, so you re-paste it into the extension popup every session
instead of once, ever.

Add this instead of cell 6 in §5:
```python
import subprocess, re, threading

subprocess.run(["wget", "-q",
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    "-O", "/usr/local/bin/cloudflared"], check=True)
subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"], check=True)

cf_log = open("/kaggle/working/cloudflared.log", "w")
cf_proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8766"],
    stdout=cf_log, stderr=subprocess.STDOUT,
)

def print_url_when_ready():
    for line in open("/kaggle/working/cloudflared.log"):
        m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if m:
            print("Public URL:", m.group(0))
            print(f"Paste into the extension: {m.group(0)}/v1/translate-image")
            return

threading.Timer(8, print_url_when_ready).start()
```
Everything else in this document (Secrets, `FMT_AUTH_TOKEN`, warmup, extension setup)
is identical — only the tunnel mechanism and the "re-paste every session" cost differ.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/v1/health` never responds (cell 5 loop exhausts) | Backend crashed on startup — usually a missing dependency or bad `.env` | Check `/kaggle/working/backend.log`; a torch/CUDA mismatch shows up here as an import error |
| Warmup status stays `fail` | A model failed to load (check `models` field in the health JSON for which one) | Re-run `/v1/warmup`; if it keeps failing, check disk space (`!df -h`) — HF downloads can fill `/kaggle/working` |
| Popup shows "Offline — will resume automatically" and never recovers | Kaggle session actually ended, or the tunnel process died | Check the notebook is still running; re-open the tunnel cell if needed |
| ngrok error `ERR_NGROK_...` on connect | Authtoken wrong/expired, or the static domain doesn't match what you claimed | Re-check the `NGROK_AUTHTOKEN` secret value and the `STATIC_DOMAIN` string exactly |
| Extension badge stuck on "Offline" right after Save | Pasted a bare domain instead of the full `/v1/translate-image` path (§6) | Fix the URL field, click Save again |
| `403` on every request from the extension | `FMT_AUTH_TOKEN` set on the backend but not matching (or not set) in the popup's auth token field | Make sure both sides have the exact same token value, no trailing whitespace |
| `paddlepaddle`/`paddleocr` install hangs or fails | Occasionally flaky on Kaggle's network | Re-run the pip install cell; these packages are large and occasionally need a retry |
| Out-of-memory (OOM) errors during translation | Too many concurrent jobs for one T4 | Lower `FMT_PIPELINE_MAX_PARALLEL` in your `.env` (and re-encode `FMT_ENV_B64`) — default 2 should normally fit a T4's 16GB comfortably |
| Dataset not visible in notebook | Forgot to attach it, or uploaded under a different name than referenced in cell 1/2 | Add-ons → Data → attach; match the exact `/kaggle/input/<name>` path in the notebook |

---

## Quick start

For when you've already read the above once and just need the checklist:

1. Kaggle: phone-verified, GPU T4×2 + Internet on, dataset attached, 4 secrets
   attached.
2. Run notebook cells 1-6 (`deploy/kaggle/fmt_kaggle_backend.ipynb`).
3. Copy the printed public URL + `/v1/translate-image`.
4. Extension popup → Local Pipeline URL = that value, Backend auth token = your
   `FMT_AUTH_TOKEN`, Save.
5. Translate normally. Cell 7 keeps the session alive; cell 8 shuts down cleanly when
   you're done.
