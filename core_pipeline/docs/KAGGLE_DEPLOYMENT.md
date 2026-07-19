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
4. **Your provider API keys**, the same ones your local backend already uses, delivered
   to Kaggle as individual Kaggle Secrets (§4) — never as a `.env` file, never committed
   or uploaded anywhere in plaintext. If any of your existing keys have ever been
   pasted into a chat, a public gist, a screenshot, or anywhere else outside your own
   `.env` file and provider dashboards, **rotate that key on the provider's dashboard
   first** and use the freshly rotated value when you create its Kaggle Secret in §4 —
   an old, possibly-exposed key has no business getting a second life on a
   publicly-tunneled backend.

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

**Backend-code changes need a new dataset version to take effect.** Cell 2 of the
notebook (§5) copies straight from this dataset every run — editing `main.py` (or
anything else) in your local repo has zero effect on Kaggle until you re-zip and
upload a new dataset version. One narrow exception: Cell 2 hot-patches the *copied*
`main.py`'s CORS config on every run (adds the ngrok bypass header, §4) specifically
so that particular fix doesn't require a re-upload while it's still new — but that's
a deliberate, temporary, single-purpose patch, not a general mechanism. Don't rely on
it for anything else; re-upload the dataset for any other backend change.

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

### One small secret per credential — not one blob

An earlier version of this document used a single `FMT_ENV_B64` secret holding your
entire `.env` file, base64-encoded, as one ~15KB blob. **That design is retired.** While
testing this deployment, that single large secret would not save — the Kaggle Secrets
panel accepted the label and value, the Save button was clickable, but the panel
reverted to "No secrets added" immediately after, with no error shown. Deep research
across Kaggle's own documentation, the `kaggle_secrets.py` client source, and every
public bug report and forum thread found **no documented size or count limit** for
Kaggle Secrets, and no report matching this exact symptom — so the precise cause was
never confirmed. What *is* confirmed is that the only pattern Kaggle's own community
actually uses for many credentials is **one small secret per credential**, not a single
blob. This document now uses that pattern. It sidesteps whatever the blob's problem
was, whether that was payload size or something else, and it's the better-supported
approach regardless.

The committed notebook (`deploy/kaggle/fmt_kaggle_backend.ipynb`) reads exactly
**12 secret names** — verified by grepping the notebook's own cells, not assumed. Every
label below is spelled **identically** to the environment variable it becomes — no
translation table to keep in your head. **You do not need a `GH_PAT` secret** — that's
only for the git-clone path (§3 Option B), and this document's primary path (Option A,
the dataset zip) doesn't touch GitHub credentials at all.

For each one: in the Kaggle notebook editor, open **Add-ons → Secrets**, type the
**exact label** shown, paste the **value**, click **Save**, then make sure the toggle
next to it is **ON** for this notebook (Kaggle requires you to explicitly attach each
secret per-notebook — easy to create a secret and forget this step, which silently
makes it invisible to `UserSecretsClient.get_secret()` at runtime, not a helpful
message).

**A provider you don't use can simply be skipped** — leave its secret uncreated. The
notebook's Cell 3 treats a missing provider secret exactly like an unfilled line in a
local `.env`: that provider is left unconfigured, and the pipeline already treats an
unconfigured provider as zero-quota and never reserved. You don't need all 12 to get a
working deployment; you need the ones matching the providers your local `.env`
currently has real keys for.

---

### Step 1 — the go/no-go check: create `NGROK_AUTHTOKEN` first

Before touching any provider key, create **just this one secret** and confirm it
actually appears in the Secrets list afterward:

- Label: `NGROK_AUTHTOKEN`
- Value: from your ngrok dashboard (dashboard.ngrok.com) → **Your Authtoken** (left
  sidebar, under Getting Started/Setup & Installation — the exact menu wording has
  moved around over time, but it's always on the page showing a long token starting
  `2` or similar, with a copy-icon button next to it).

Click **Save**, then look at the Secrets list.

- **If it appears and stays there** — the account/browser can save secrets normally.
  Continue to Step 2 below with confidence; the rest of the secrets will behave the
  same way.
- **If it also reverts to "No secrets added"** — this isn't about payload size (this
  value is a short token, nowhere near 15KB), so don't create the other 11 secrets yet;
  they'll very likely hit the same wall. Instead try, in order: a hard refresh
  (Ctrl+Shift+R) before retrying, an incognito window or a different browser, clicking
  **Save Version** on the notebook first (an unsaved draft notebook has been reported
  to behave oddly with Secrets) and then retrying, or simply waiting a few minutes and
  retrying (transient Kaggle-side hiccups have been reported and self-resolve). Once
  this one secret saves and sticks, come back and continue below.

This is not a separate testing phase tacked onto setup — it *is* the first real step,
just ordered so a systemic problem shows up after 30 seconds instead of after
re-entering 12 credentials.

**13th secret — `NGROK_STATIC_DOMAIN` (recommended, required for a true "Run All").**
Cell 6 normally reads your claimed domain from a `STATIC_DOMAIN` constant you edit
directly in the notebook. Create a secret labeled `NGROK_STATIC_DOMAIN` with your
domain as the value instead, and Cell 6 picks it up automatically — no manual cell
editing needed at all. This is what makes **Run All** actually work start to finish;
without it, Cell 6 stops with an assertion telling you to either create this secret or
edit the placeholder by hand.

---

### Step 2 — the provider keys

For every provider you actually use (skip the rest), create a secret with the label
below, using the **same comma-separated key format** your local `.env` already uses for
that variable. **If any of these keys were ever exposed outside your own `.env` file
and provider dashboard — including having been pasted into any chat — rotate it on the
provider's dashboard first and paste the freshly rotated value here, not the old one.**

| Secret label (== exact env var name) | Get the key from |
|---|---|
| `GEMINI_API_KEYS` | Google AI Studio (aistudio.google.com) → **Get API key** |
| `GITHUB_API_KEYS` | GitHub → Settings → Developer settings → **Personal access tokens** (fine-grained) |
| `GROQ_API_KEYS` | console.groq.com → **API Keys** |
| `MISTRAL_API_KEYS` | console.mistral.ai → **API Keys** |
| `OPENROUTER_API_KEYS` | openrouter.ai → **Keys** |
| `CEREBRAS_API_KEYS` | cloud.cerebras.ai → **API Keys** |
| `FIREWORKS_API_KEYS` | fireworks.ai → **API Keys** |
| `CLOUDFLARE_WORKERS_API_KEYS` | Cloudflare dashboard → My Profile → **API Tokens** |
| `CLOUDFLARE_ACCOUNT_IDS` | Cloudflare dashboard → Workers & Pages overview page (right sidebar shows your Account ID) |
| `NVIDIA_NIM_API_KEYS` | build.nvidia.com (NIM/API Catalog) → **Get API Key** on any model page |

Same routine as Step 1 for each: Add-ons → Secrets → type the label exactly → paste the
value → Save → confirm it's toggled **ON** for this notebook.

---

### Step 3 — `FMT_AUTH_TOKEN`

A password only you know, shared between your Kaggle-hosted backend and your local
extension, so a stranger who stumbles on your public tunnel URL can't send it real
requests (§8). Generate one yourself — don't reuse a password from anywhere else:

```powershell
-join ((48..57)+(65..90)+(97..122)|Get-Random -Count 40|ForEach-Object{[char]$_})
```

This prints a random 40-character token to the PowerShell window (verified working:
produces exactly 40 alphanumeric characters each run). **Select and copy that printed
value** and paste it as the Value for a secret labeled `FMT_AUTH_TOKEN`.

**Write this value down somewhere** (a local password manager, a text file that isn't
ever committed) — you'll need to paste this *exact same string* into the extension
popup's "Backend auth token" field in §6. If the two don't match character-for-character,
every request will get a `403` and the popup will just show "Offline" with no more
specific explanation, which is confusing to debug blind — matching values on both ends
is the whole point.

---

### Before you run the notebook: a 60-second checklist

This is what "no error in the first build" actually comes down to — confirm all of this
*before* clicking run on cell 1:

- [ ] Add-ons → Secrets shows `NGROK_AUTHTOKEN` and it's stayed in the list since Step 1
      (your go/no-go check already passed).
- [ ] Add-ons → Secrets shows a secret for every provider your `.env` currently has a
      real key for (Step 2) — skipped providers are fine, just double-check you didn't
      skip one you actually meant to use.
- [ ] Add-ons → Secrets shows `FMT_AUTH_TOKEN` (Step 3).
- [ ] Every secret above is toggled **ON** for this notebook — creating a secret does
      not automatically attach it.
- [ ] Add-ons → Data shows your `fmt-core-pipeline` dataset attached (§3) — if you
      haven't uploaded `fmt_core_pipeline.zip` as a dataset yet, do that first; cell 1
      asserts on this path and fails fast, on purpose, rather than limping through a
      confusing later error.
- [ ] Side panel → Accelerator = **GPU T4 x2**, Internet = **On**.
- [ ] You've written down your `FMT_AUTH_TOKEN` value somewhere durable — you'll need
      it again in §6, after the notebook is already running.
- [ ] Either Cell 6's `STATIC_DOMAIN` variable is edited to your actual claimed ngrok
      domain (§2), or you created the optional `NGROK_STATIC_DOMAIN` secret — Cell 6
      raises immediately, before opening a tunnel, if neither is a real domain.

---

## 5. The notebook, cell by cell

This mirrors `deploy/kaggle/fmt_kaggle_backend.ipynb`, committed in this repository —
you can copy that file directly into a new Kaggle notebook, or type these cells in
yourself. Expected timings are in §7. **Every cell below is idempotent** — safe to
re-run on its own if something fails partway, and if the whole session dies, the
recovery is simply "re-run Cell 1 through Cell 6 top to bottom," no manual cleanup.

**Before running anything:** in the notebook's right-side panel, set **Accelerator →
GPU T4 x2**, and toggle **Internet → On** (required for pip installs, HF model
downloads, and the tunnel itself). Attach your `fmt-core-pipeline` dataset (Add-ons →
Data → your dataset). Attach the secrets from §4 (Add-ons → Secrets → toggle each one
on for this notebook) — `NGROK_AUTHTOKEN` and `FMT_AUTH_TOKEN` always, plus one per
provider you actually use.

**Cell 1 — environment sanity check.** Confirms GPU is actually visible, the dataset is
attached, and prints every preinstalled library version Cell 2 is about to touch
(torch/numpy/cv2/onnxruntime) via a **fresh subprocess** — never a kernel import,
since Cell 2 uninstalls/reinstalls several of these and a kernel that already imported
one would hold a stale copy. Kaggle mounts an attached dataset at `/kaggle/input/<slug>`
in most sessions, but interactive/draft sessions have been observed mounting it one
level deeper instead, at `/kaggle/input/datasets/<your-username>/<slug>` — confirmed by
running `os.listdir("/kaggle/input")` during this document's own testing and seeing
`['datasets']` instead of the dataset slug directly. This cell checks both locations so
it works either way, and prints which one it found. See the notebook's own Cell 1 for
the full code (`fmt-cell1-v2`).

Expect a `Dataset found at: ...` line, GPU lines naming `Tesla T4, 15360 MiB` (or
similar), and a version line per preinstalled library. If this cell raises the
`AssertionError`, the printed `checked:` list shows every path it looked at — compare
against `os.listdir("/kaggle/input")` (and, if present,
`os.listdir("/kaggle/input/datasets")`) to see what Kaggle actually named your mount.
If GPU output is empty instead, the accelerator setting wasn't applied — fix it in the
side panel and restart the session before continuing.

**Cell 2 — copy code, fonts, CORS + tokenizer-routing hot-patches, smart dependency
install.** The biggest cell in the notebook, doing eleven things in a load-bearing
order (`fmt-cell2-v7` in the notebook):

1. **Copy** `DATASET_DIR/core_pipeline` → `/kaggle/working/core_pipeline` (`/kaggle/input`
   is read-only).
2. **Hot-patch the copy's CORS config** to allow ngrok's interstitial-bypass header
   (temporary — bakes into the next dataset version).
3. **Hot-patch the copy's `ml_region_lib.py`** with a tokenizer-routing shim — the
   actual, source-confirmed fix for the tokenizer failure below (also temporary,
   same reason). Explained in full in §10; short version: `transformers` ≥5.13.0
   registers TrOCR's `model_type` (`"vision-encoder-decoder"`, loaded internally by
   magi via `TrOCRProcessor`) in `TOKENIZER_MAPPING_NAMES` pointing at the generic
   `TokenizersBackend` class, which can only build from a `tokenizer.json` —
   and `microsoft/trocr-base-printed` has never shipped one (confirmed: only
   `vocab.json`+`merges.txt`, plus a negative-cache 404 marker for `tokenizer.json`
   in the local HF cache). `AutoTokenizer.from_pretrained` then raises a misleading
   `"need sentencepiece or tiktoken"` `ValueError` that has nothing to do with either
   package. The shim re-registers that one mapping entry to `RobertaTokenizer`
   (which reads `vocab.json`+`merges.txt` directly, exactly what this repo needs) —
   **verified empirically, in both directions, on this exact failing call**, before
   ever going near Kaggle: fails without it on `transformers==5.14.1`, succeeds with
   it; harmless no-op on `transformers==5.12.0`, which never even populates that
   mapping entry (falls through to the hub's own `tokenizer_config.json` class,
   already `RobertaTokenizer` there).
4. **Fonts.** The dataset ships no fonts (verified: zero files matching `font` in the
   zip) and the vendored `ComicNeue-{Bold,Regular}.ttf` live outside `core_pipeline`
   in this repo, so they never make it into the dataset at all. Step 8's typesetting
   font-resolution chain (`run_step8_typeset.py`) tries Windows paths first, then a
   bundled-fonts directory that — on Kaggle's `cwd=/kaggle/working/core_pipeline` —
   resolves to `/kaggle/working/fonts`, then finally `/usr/share/fonts/truetype/dejavu/`.
   This cell downloads both ComicNeue fonts from Google's official Fonts repo into
   exactly that path, **validates each one actually loads** with `PIL.ImageFont`
   (a truncated download would otherwise sit there looking valid and render garbage
   text later), and deletes anything that fails validation. It only raises if
   ComicNeue failed **and** the DejaVu fallback is also absent — otherwise you get a
   working font, just possibly not the intended comic-style one.
5. **Torch**: keeps the preinstalled build if it's already CUDA-capable ≥ 2.6, otherwise
   installs the pinned `cu124` wheels. The version check compares `(major, minor)` as
   integers, not a string prefix — Kaggle's image drifts over time (observed
   torch 2.6.0+cu124 one session, 2.10.0+cu128 the next on a fresh container), and an
   earlier string-`.startswith()` version of this check silently broke on "2.10" and
   triggered an unwanted downgrade to 2.6.0, which is what caused a `torchaudio`
   mismatch in an earlier round of this notebook.
6. **Uninstall-first**: removes any preinstalled `onnxruntime` (CPU build), and every
   `opencv-*` variant, before installing anything — a CPU `onnxruntime` package can
   **silently shadow** `onnxruntime-gpu` (the pipeline's ONNX text detector and LaMa
   inpainter create a CUDA-only `InferenceSession` with no CPU fallback — this is a
   hard crash on every request, not a slowdown), and Kaggle images commonly preinstall
   `opencv-python-headless`, which conflicts with this repo's pinned non-headless
   `opencv-python` (both packages own the same `cv2/` namespace — having both installed
   corrupts the import).
7. **Filtered install**: writes a copy of `python/requirements.txt` with the
   torch/torchvision lines dropped (when keeping the preinstalled build) and
   `opencv-python` swapped for `opencv-python-headless`, then installs it, then
   `backend_api/requirements.txt`, then `pyngrok`.
8. **`sentencepiece` + `protobuf`**: needed by the NLLB local-translator tokenizer
   (`facebook/nllb-200-distilled-600M`) — unrelated to the TrOCR/magi tokenizer issue
   above, which the routing shim (step 3) fixes, not these packages. (An earlier round
   of this notebook incorrectly attributed the TrOCR error to a sentencepiece/protobuf
   mismatch; that theory didn't survive contact with the real traceback — see §10.)
9. **Post-install cleanup**: `ultralytics` and `paddleocr`/`paddlex`
   transitively pull non-headless `opencv-python` back in even after the filtered
   install — this cell uninstalls it again and force-reinstalls
   `opencv-python-headless` last, with `--no-deps` so nothing re-drags it back a
   second time.
10. **`onnxruntime-gpu` CUDA pin** — found via a live run: the requirements'
   `onnxruntime-gpu>=1.26.0` resolves to 1.27.0+, which switched its default build
   from CUDA 12 to CUDA 13 (`libcudart.so.13`, absent on any Kaggle CUDA-12.x box) —
   import fails outright. Force-reinstalls the pinned `onnxruntime-gpu==1.26.0`, the
   last version still defaulting to CUDA 12. Also **`torchaudio`**: a transitive
   dependency (transformers/easyocr audio extras) can pull in a `torchaudio` build
   mismatched with the active torch, breaking with an `undefined symbol` error the
   first time anything imports it — this cell queries the *actual* active torch
   version via a fresh subprocess and force-reinstalls a matching `torchaudio` build.
   Finally, **`transformers==5.12.0`** is pinned explicitly — belt-and-braces with the
   step-3 shim, matching the exact version the full pipeline (magi + TrOCR +
   manga-ocr + NLLB) is proven on daily on the maintainer's own machine.
11. **Smoke test**: in a **fresh subprocess** (this kernel may still hold stale
   imports of packages just uninstalled), scans installed distributions (exactly
   one opencv variant, `onnxruntime-gpu` present and plain `onnxruntime` absent) and
   creates a **real** `onnxruntime.InferenceSession` on the actual text-detector model
   with `providers=['CUDAExecutionProvider']`, asserting CUDA is the active provider —
   `get_available_providers()` alone would only prove the wheel *compiled with* CUDA
   support, not that it can actually *initialize* it on this VM's driver/cuDNN stack.
   It also runs the literal `TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")`
   call magi makes — the exact call that was failing during warmup — rather than just
   checking that a package imports, which turned out not to be sufficient evidence in
   an earlier round (see §10). A real failure here surfaces in Cell 2, in seconds,
   with the true traceback, instead of 5+ minutes into warmup behind a generic error.

**Cell 3 — load secrets into environment.** Reads each provider's key straight from its
own small Kaggle Secret into `os.environ` — **no `.env` file is ever written to disk**.
A provider whose secret was never created (§4 Step 2) is simply left unconfigured,
matching how an unfilled line in a local `.env` already behaves. `FMT_AUTH_TOKEN` is
read with `.strip()` and the cell **raises if it's missing or empty after stripping** —
a token that silently collapsed to `""` would leave every gated route armed only by
the `X-Fmt-Client` header, which is public (it's in this repo's own source), so this
cell refuses to start with auth silently disabled on a public tunnel rather than fail
open. Non-secret pipeline tuning is hardcoded in the same cell since none of it is
sensitive — this mirrors the values in this repo's own `.env.example`; edit them
directly if you want Kaggle to run with different settings than your local machine.
**Nothing here is ever printed.**

The printed summary line (`Configured providers (N/10): ...`) is your first real
verification checkpoint — it lists which providers loaded (never their values) before
you've even reached Cell 4. If a provider you expected is missing from it, its secret
either wasn't created or wasn't toggled ON for this notebook (§4).

Why no `.env` file at all: `api_manager.py`'s `_load_env` uses
`load_dotenv(path, override=True)` — if a `.env` file were found anywhere on its
search path, *its* values would silently win over anything already set in `os.environ`
by this cell. Since the dataset zip never includes a `.env` file (§3 excludes it) and
this cell doesn't write one, `_load_env` finds nothing on Kaggle and logs a purely
cosmetic "no .env file found" warning — the directly-set values above are used cleanly,
with no override risk.

**Cell 4 — launch the backend.** Runs uvicorn exactly the way `start_backend.ps1` does
locally (same working directory, same module path), as a background subprocess so the
notebook cell returns immediately and later cells can run. Re-run safe: terminates any
backend this notebook already started (and, as a belt-and-braces fallback, `pkill`s any
stray uvicorn process matching this app) before launching a new one — otherwise a
re-run leaks the old process holding port 8766 and the new one fails to bind. The bind
stays on **127.0.0.1**, not `0.0.0.0` — the tunnel client (Cell 6) runs on this same VM
and reaches the backend over loopback, so there is never a reason to bind a wider
interface here; doing so would only expand the Kaggle VM's own attack surface for no
benefit.

**Cell 5 — wait for health + force warmup.** Polls `/v1/health` (up to 5 minutes),
checking after every poll whether the backend process has actually **died** —
if it has, this cell prints the last 60 lines of `backend.log` and raises immediately,
instead of waiting out the rest of the timeout for nothing. Once healthy, it forces
`/v1/warmup` (loads manga_ocr, the text/bubble detectors, magi, the inpainting models)
and polls for up to **20 minutes**, since a first-ever run downloads several GB
(magi, manga-ocr, EasyOCR, the ~2.4GB NLLB translator — see §7). Requests now send
**`X-Fmt-Auth` alongside `X-Fmt-Client`**, because `/v1/warmup` is one of the routes
gated by `FMT_AUTH_TOKEN` (Cell 3 always sets it) — sending only `X-Fmt-Client`, as an
earlier version of this cell did, gets a `403` here and crashes the cell.

**Cell 5b — loopback end-to-end translate (new).** The single highest-value check in
the whole notebook. Before any tunnel exists, this cell POSTs a real sample manga page
(`samples/sample1/sample.jpg`, bundled in the dataset) to
`127.0.0.1:8766/v1/translate-image` with both headers, asserts the response status is
`pass`, prints a report summary, and displays the translated image inline. This single
request exercises the CUDA ONNX text detector, OpenCV, PaddleOCR, EasyOCR, the LaMa
inpainter, the fonts installed in Cell 2, and whichever translation provider keys you
configured — all at once. **If this cell passes, the pipeline itself is proven working
end to end**, and any problem you hit afterward is tunnel- or extension-side, not
pipeline-side — a genuinely useful thing to know before debugging blind.

**Cell 6 — open the tunnel.** Uses `pyngrok` with your static domain, so the public URL
is the same every session. Re-run safe: calls `ngrok.kill()` first, since the free
tier only allows one active agent session and a bare re-run of `ngrok.connect` on the
same domain fails with `ERR_NGROK_334`/`108`. The domain comes from the
`NGROK_STATIC_DOMAIN` secret if you created one (§4, recommended — this is what lets
Run All work unattended), otherwise from the `STATIC_DOMAIN` constant — edit that
constant to your claimed domain if you didn't create the secret; the cell raises
immediately if neither is a real domain, rather than opening a tunnel to a
placeholder. **Self-healing fallback**: if ngrok rejects the explicit domain as a
paid-only "custom subdomain" — observed live with newer `.ngrok-free.dev` dev-domains,
even when the domain is genuinely reserved to the account (§10) — this cell catches
that specific error and automatically retries with no domain argument at all, letting
ngrok auto-assign the account's dev domain instead of hard-failing.

**Cell 7 — keep the session alive.** Kaggle interactive sessions can idle-disconnect;
this cell just keeps the notebook actively running and periodically confirms the
backend is still healthy, so you have a live signal in the notebook output if
something crashes. Interrupt it (■ Stop) when you're done reading and want to move to
the shutdown cell — it's an intentional infinite loop, not a bug. **This loop does NOT
defeat Kaggle's idle watchdog** (§7) — a running cell isn't counted as user
interaction, only clicks/keystrokes in the tab are. Don't start this cell during
initial setup; only start it at handoff, once Cells 1-6 have all passed.

**Cell 8 — clean shutdown.** Uses `ngrok.kill()` rather than
`ngrok.disconnect(tunnel.public_url)`, since the latter raises a `NameError` if the
kernel was ever restarted and `tunnel` no longer exists in memory — `ngrok.kill()`
works regardless of kernel state. Nothing is written to disk by Cell 3 in this design,
so there's no leftover secrets file to clean up.

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

**You don't need to do anything about ngrok's browser-warning interstitial.** The
extension's `fmtHeaders()` (`background.js`) already sends the documented bypass
header (`ngrok-skip-browser-warning`) on every backend request, unconditionally — it's
harmless against a plain local backend, which just ignores the extra header. This is
built into the extension itself, not something this deployment adds.

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

**Kaggle GPU quota:** 30 GPU-hours per week (resets weekly), and a **9-hour maximum
continuous session** even within quota — there is no way to extend a single session
past that. Plan reading sessions accordingly; the notebook does not auto-restart
itself when a session ends — you'll need to re-run Cells 1-6 (all idempotent, §5).

**Idle watchdog — read this, it's not what Cell 7 makes it look like.** Kaggle kills
an interactive session after roughly **60 minutes with no interaction in the browser
tab** — and critically, **a running cell does not count as interaction**. Cell 7's
keep-alive loop keeps the *kernel* busy, which is necessary but not sufficient; Kaggle
still expects UI interaction (a click, a keystroke) roughly hourly and will show an
"are you still there?" prompt. Keep the notebook tab open and actually interact with
it occasionally, even while Cell 7 is running — don't rely on the loop alone to keep
an unattended session alive for hours.

**Only interactive sessions work for this deployment.** A "Save Version"/batch commit
run executes headless to completion and its tunnel URL isn't reachable the way you
need — always run this as a live interactive session, not a scheduled/committed one.

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

**ngrok free-tier egress cap: 1GB/month.** Every translated page comes back as a
base64 data URL (≈1.33× the raw image bytes), so realistically budget **~400-700
translated pages per month** on the free tier before you hit the cap (depends heavily
on page resolution). For light personal reading this is unlikely to matter; for heavy
use, switch to the `cloudflared` fallback in §9, which has no such bandwidth cap
(trading away the static-domain convenience — see §9 for that trade-off).

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

**If you've ever shared a real key outside your own `.env` and provider dashboards** —
including pasting one into a chat, a screenshot, or a gist while troubleshooting this
setup — treat it as compromised and rotate it on the provider's dashboard before this
deployment goes live. §4 Step 2 already has you re-enter each provider's key
individually, so rotating first costs nothing extra; it's the same paste, just with a
fresh value.

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
| Out-of-memory (OOM) errors during translation | Too many concurrent jobs for one T4 | Lower `FMT_PIPELINE_MAX_PARALLEL` directly in cell 3's `os.environ.update({...})` block — default 2 should normally fit a T4's 16GB comfortably |
| Cell 1's `AssertionError` fires even though Add-ons → Data shows the dataset attached | Kaggle mounted it one level deeper than expected (`/kaggle/input/datasets/<username>/<slug>`, seen on interactive/draft sessions) — the error's printed `checked:` list shows exactly what was tried | Run `os.listdir("/kaggle/input")` and, if it shows `['datasets']`, also `os.listdir("/kaggle/input/datasets")` to see the real path; Cell 1 already checks both layouts, so this should only happen if the dataset name itself doesn't match `fmt-core-pipeline` |
| Dataset not visible in notebook at all | Forgot to attach it, or uploaded under a different name | Add-ons → Data → attach; the name must match `DATASET_SLUG` in cell 1 |
| A secret you created won't stay in the Secrets list (reverts to "No secrets added" right after Save) | Not confirmed — no documented Kaggle limit matches this; possibly a transient session/browser issue | Follow the go/no-go branch in §4 Step 1: hard refresh, try incognito/a different browser, save the notebook first, or wait a few minutes and retry |
| A provider you expected doesn't show up in cell 3's "Configured providers" printout | Its secret wasn't created, or was created but not toggled ON for this notebook | Add-ons → Secrets → check the label matches exactly and the toggle is ON, then re-run cell 3 |
| Cell 2's smoke test fails with `session.get_providers()[0] != 'CUDAExecutionProvider'` (or an ORT provider/cuDNN error) | A CPU `onnxruntime` package shadowed `onnxruntime-gpu`, or the CUDA/cuDNN stack didn't initialize | Re-run cell 2 (idempotent — it uninstalls onnxruntime variants before reinstalling); if it persists, restart the session and re-run cells 1-2 fresh |
| `onnxruntime` import fails with `libcudart.so.13: cannot open shared object file` | `onnxruntime-gpu` resolved to 1.27.0+, which switched its default CUDA build from 12 to 13 — Kaggle's box only has CUDA 12.4 | Cell 2 already pins `onnxruntime-gpu==1.26.0` (step 6b) specifically for this; if a future Kaggle image ships a newer CUDA, that pin may need bumping — check `torch.version.cuda` in Cell 1's preflight output first |
| Warmup fails with `libtorchaudio.so: undefined symbol: aoti_torch_abi_version` | A transitive dependency (transformers/easyocr audio extras) pulled in a `torchaudio` build that doesn't match the active torch | Cell 2 already force-reinstalls a matching `torchaudio` build from the same CUDA index as the active torch (step 10) — determined dynamically, not hardcoded, specifically for this |
| Cell 2's smoke test fails loading `microsoft/trocr-base-printed`'s tokenizer, or warmup fails with `Couldn't instantiate the backend tokenizer... You need to have sentencepiece or tiktoken installed` | A `sentencepiece`/`protobuf` version mismatch — this is magi's bundled TrOCR/RobertaTokenizer dependency, not NLLB, traced directly from magi's own HF Hub source (step 7) | Cell 2 already force-reinstalls `sentencepiece` and `protobuf` together (not `--no-deps`, so pip picks a mutually compatible pair); if it still fails, the smoke test's traceback now shows the real underlying exception instead of this generic message — paste that, not this row |
| Cell 6 gets `ERR_NGROK_313` / "Only paid plans may create endpoints with custom subdomains" even though the domain is reserved to your account | A known quirk with newer `.ngrok-free.dev` dev-domains and explicit `domain=` requests | Cell 6 already catches this specific error and retries with no domain argument, letting ngrok auto-assign your account's dev domain instead of failing |
| Cell 2's smoke test fails with a stray-opencv-dist assertion, or `cv2` import errors mentioning `cv2.dnn`/missing attributes | Two opencv variants installed simultaneously (dual-`cv2` corruption) — usually `ultralytics` or `paddleocr` re-pulling non-headless `opencv-python` | Re-run cell 2 — its post-install cleanup step force-reinstalls `opencv-python-headless` last with `--no-deps` specifically to fix this |
| Any import error mentioning `_ARRAY_API not found` or "compiled using NumPy 1.x" | numpy ABI mismatch — something upgraded numpy without recompiling against it | Restart the Kaggle session and re-run cells 1-2 fresh; don't `pip install --upgrade numpy` manually mid-session |
| Cell 5/5b requests return the ngrok interstitial HTML instead of JSON (only after the tunnel is up — see §9 for pre-tunnel testing) | The `ngrok-skip-browser-warning` header wasn't sent, or is being stripped somewhere upstream | The extension already sends this header on every request (§6); if you're testing with `curl` directly, add `-H "ngrok-skip-browser-warning: 1"` |
| `ERR_NGROK_334` or `ERR_NGROK_108` when opening the tunnel | A previous tunnel session on this domain/account is still open (common after a notebook re-run) | Cell 6 already calls `ngrok.kill()` before connecting — if you're still seeing this, the *previous* session may be running in a different notebook or browser tab; close it first |
| Cell 4 fails with "address already in use" on port 8766 | A previous backend process from this notebook is still holding the port | Cell 4 already kills any prior backend it started before launching a new one — if you're still seeing this, another notebook/session is bound to 8766; restart the Kaggle session |

---

## 11. Cold-start timing

As of `fmt-cell2-v8`/`fmt-cell5b-v3`, every cell from Cell 1 through Cell 6 prints
`[TIMING]` lines: Cell 2 breaks its own runtime down by step (repo copy, hot-patches,
fonts, torch check, each pip install/reinstall, the smoke test), and Cells 4/5/5b/6
each print elapsed time since Cell 1 started, so `Run All` on a genuinely fresh
Kaggle session (not a resumed one — resumed sessions reuse a warm pip/HF cache and
give misleadingly fast numbers) tells you exactly where the time goes instead of
guessing. This is purely additive instrumentation — no installs, pins, or hot-patches
changed.

What to do with the numbers: the pipeline bundles most of its models directly in the
dataset (`core_pipeline/models/`) — only `manga-ocr` (~450MB) and `magi` (~1-1.2GB)
download from Hugging Face during warmup on a cold cache, and `facebook/nllb-200-
distilled-600M` (~2.4GB) downloads separately, and only if Cell 5b's translate falls
back to the local translator because no configured API provider succeeded (Cell 5b's
`[TIMING]` line now also prints which provider was actually used, so you know whether
that download happened on this run). If Cell 2's step-by-step total dominates the
overall time, the pip install/dependency-resolution path is the bottleneck; if
warmup/Cell 5b dominates instead, it's model download or GPU load. Whichever it is,
that's the number to optimize next — don't guess.

For when you've already read the above once and just need the checklist:

1. Kaggle: phone-verified, GPU T4×2 + Internet on, dataset attached, `NGROK_AUTHTOKEN`
   + `FMT_AUTH_TOKEN` + `NGROK_STATIC_DOMAIN` + one secret per provider you use
   attached (§4). Including `NGROK_STATIC_DOMAIN` means the next step needs zero
   manual editing.
2. **Run → Run All** (`deploy/kaggle/fmt_kaggle_backend.ipynb`). Cells 1-6 run
   unattended — Cell 5b's loopback translate passing means the pipeline itself is
   proven working before the tunnel or extension are ever touched. It will then
   appear to hang on Cell 7 — that's correct, not a bug (see the notebook's intro
   cell).
3. Copy Cell 6's printed public URL + `/v1/translate-image`.
4. Extension popup → Local Pipeline URL = that value, Backend auth token = your
   `FMT_AUTH_TOKEN`, Save.
5. Translate normally. Cell 7 keeps the session alive (but see §7 — you still need to
   interact with the tab roughly hourly); when done, ■ Stop Cell 7, then run Cell 8
   by hand to shut down cleanly.
