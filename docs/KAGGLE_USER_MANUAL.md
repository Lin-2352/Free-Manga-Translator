# Kaggle User Manual — Manga Translator

This guide walks through running the backend on Kaggle's free NVIDIA T4 GPU instead of
your own machine, and pointing the same Chrome/Brave extension at it. Use this if you
don't have a GPU, or don't want to keep a local PowerShell window open while reading.

For the deep technical reference behind every step here (why each cell does what it
does, security analysis, full troubleshooting table) see
`core_pipeline/docs/KAGGLE_DEPLOYMENT.md`. This manual is the walkthrough; that
document is the "why."

## 0. Kaggle vs Local — which one do you want?

Use **local** (`docs/USER_GUIDE.md` or `LOCAL_USER_MANUAL.md` if you have it) if you
have an NVIDIA GPU on your own machine and don't mind keeping a terminal open.

Use **Kaggle** (this guide) if:

- You don't have a GPU, or yours doesn't have enough VRAM.
- You'd rather not keep your own machine running while you read.
- You're fine with the trade-offs in section 8 below (a Kaggle session eventually
  ends, there's a small monthly bandwidth cap on the free tunnel, and setup takes
  about 20-30 minutes the first time).

Kaggle is not "better" than local — it's a different set of trade-offs. Read section 8
before committing to it as your daily driver.

## 1. Kaggle Account Prerequisites

1. Create a Kaggle account if you don't have one, at kaggle.com.
2. Go to **Settings → Phone Verification** and verify a phone number. Kaggle will not
   grant GPU access without this — the GPU accelerator option stays greyed out.
3. Create a free ngrok account at ngrok.com. This is what exposes your Kaggle
   notebook's backend to the public internet so your local extension can reach it.
4. On the ngrok dashboard, go to **Your Authtoken** and copy it. You'll paste this
   into a Kaggle Secret in section 3, never into notebook code.
5. On the ngrok dashboard, go to **Domains → Create Domain** and claim your one free
   static domain (something like `yourname-something.ngrok-free.app`). This is a
   permanent hostname that never changes between sessions — you configure the
   extension with it **once, ever**, instead of every time you start the notebook.
   This is the single biggest convenience in this whole setup; don't skip it.

## 2. Build and Upload the Dataset Zip

Kaggle notebooks can't read your private GitHub repo directly without extra
credential-handling. The simplest path is to zip the code on your own machine (where
Git LFS has already resolved the vendored model weights to real files) and upload that
zip as a private Kaggle Dataset.

On your own machine, in the repo root:

```powershell
.\core_pipeline\deploy\kaggle\build_kaggle_dataset.ps1
```

This script excludes secrets, dev/test artifact folders, and the extension itself (it
never runs on Kaggle — it stays local in your browser). Expected result: a zip a few
hundred MB to under 1GB, depending on your local test data — the script prints the exact
size and warns if it looks like it accidentally picked up an excluded folder.

Then on kaggle.com:

1. **Create → New Dataset**.
2. Upload `fmt_core_pipeline.zip` (built at the repo root by the script above).
3. Give it a name — this manual assumes `fmt-core-pipeline`.
4. Leave visibility as **Private** (the default). Do not click "Make Public."
5. Click **Create**.

**Updating the code later:** re-run the script above, then on your dataset's page click
**New Version** and upload the fresh zip. The notebook always copies straight from
whatever dataset version is currently attached — a code change on your machine has zero
effect on Kaggle until you do this.

## 3. Create the 13 Kaggle Secrets

Kaggle Secrets (notebook editor → **Add-ons → Secrets**) hold your API keys
encrypted at rest, never visible in the notebook's saved source. **Type each value
directly into Kaggle's Secrets field — never paste a real key value into a chat
message, a document, or anywhere else outside your own `.env` file and the provider's
own dashboard.**

### Step 1 — the go/no-go check

Create just one secret first and confirm it actually sticks:

- Label: `NGROK_AUTHTOKEN`
- Value: the authtoken you copied in section 1.

Click **Save**, then look at the Secrets list. If it's still there, continue below. If
it reverts to "No secrets added" with no error, don't create the other 12 yet — try a
hard refresh, an incognito window, saving the notebook first, or waiting a few minutes
and retrying. This is a known but unconfirmed Kaggle-side quirk, not something wrong
with your values.

### Step 2 — the provider keys

Create a secret for every translation provider your local `.env` already has a real
key for (skip the rest — an unconfigured provider behaves exactly like an unfilled
line in `.env`, zero-quota, never reserved). Use the exact label shown; it must match
the environment variable name character-for-character:

```text
GEMINI_API_KEYS
GITHUB_API_KEYS
GROQ_API_KEYS
MISTRAL_API_KEYS
OPENROUTER_API_KEYS
CEREBRAS_API_KEYS
FIREWORKS_API_KEYS
CLOUDFLARE_WORKERS_API_KEYS
CLOUDFLARE_ACCOUNT_IDS
NVIDIA_NIM_API_KEYS
```

Same routine for each: **Add-ons → Secrets → type the label exactly → paste the value
→ Save → confirm the toggle is ON** for this notebook. Creating a secret does not
automatically attach it to the current notebook — that toggle step is easy to miss.

### Step 3 — `FMT_AUTH_TOKEN`

This is a password only you know, shared between your Kaggle-hosted backend and your
local extension, so a stranger who finds your public tunnel URL can't send it real
requests. Generate a random one yourself — don't reuse a password from anywhere else:

```powershell
-join ((48..57)+(65..90)+(97..122)|Get-Random -Count 40|ForEach-Object{[char]$_})
```

This prints a random 40-character string. Copy it and save it as a secret labeled
`FMT_AUTH_TOKEN`. **Write this value down somewhere durable** — you'll paste this
exact same string into the extension popup's "Backend auth token" field in section 6.
If the two don't match exactly, every request gets a `403` and the popup just shows
"Offline" with no more specific explanation.

### Step 4 — `NGROK_STATIC_DOMAIN`

Create one more secret, labeled `NGROK_STATIC_DOMAIN`, with the static domain you
claimed in section 1 as the value (e.g. `yourname-something.ngrok-free.app`, no
`https://` prefix). This is what lets **Run All** work start to finish with zero
manual cell editing — without it, the tunnel cell stops and asks you to edit a
placeholder by hand every time.

### Before running the notebook — checklist

- [ ] `NGROK_AUTHTOKEN` is in the Secrets list and stayed there.
- [ ] A secret exists for every provider you actually use.
- [ ] `FMT_AUTH_TOKEN` is set, and you've written the value down somewhere durable.
- [ ] `NGROK_STATIC_DOMAIN` is set to your claimed domain.
- [ ] Every secret above is toggled **ON** for this notebook.
- [ ] Add-ons → Data shows your dataset attached.
- [ ] Side panel → Accelerator = **GPU T4 x2**, Internet = **On**.

## 4. Import the Notebook and Run All

1. On kaggle.com, **Create → New Notebook**, then **File → Import Notebook** and
   select `core_pipeline/deploy/kaggle/fmt_kaggle_backend.ipynb` from your machine.
2. Confirm the checklist from section 3 is complete.
3. Click **Run → Run All**.

Cells 1 through 6 run unattended. Expected output, roughly in order:

```text
Cell 1: dataset path, GPU name (Tesla T4), and preinstalled library versions
Cell 2: a series of [TIMING] lines, ending with SMOKE TEST PASSED
Cell 3: Configured providers (N/10): <the providers you set up>
Cell 4: Backend starting, pid=<some number>
Cell 5: Health check passed, then warmup: running, then warmup: pass
Cell 5b: status: pass, a summary line, and Loopback translate PASSED
Cell 6: Public URL: https://<your-domain>.ngrok-free.dev
```

Cell 5b is the single most important line to watch: it runs one real translation
entirely inside the Kaggle VM, before any tunnel exists. If it passes, the pipeline
itself works end to end and any remaining problem is tunnel or extension
configuration, not the pipeline.

**Stale-backend trap (Kaggle version):** on your own machine, `/v1/health`'s `commit`
field (see `LOCAL_USER_MANUAL.md`) tells you the exact git commit the running backend
loaded. **On Kaggle this doesn't work** — the dataset zip only ever contains
`core_pipeline/`, never the repo's `.git` directory, so `commit` reads `null` for every
Kaggle session; there's no git history on the VM to hash. Comparing it against
`git rev-parse --short HEAD` will never match anything and isn't a useful check here.

The real freshness signal on Kaggle is each code cell's own version-tag comment — the
first line of Cell 1 through Cell 6 (e.g. `# fmt-cell2-v11`). Check these against the
list in `core_pipeline/docs/KAGGLE_DEPLOYMENT.md` (search that file for "fmt-cell"): if
your imported notebook's tags are older than what's documented there, re-download/
re-import `core_pipeline/deploy/kaggle/fmt_kaggle_backend.ipynb` before trusting a test
result. Kaggle sessions are long-lived, so it's easy to keep re-running an old imported
notebook after the source repo has moved on without noticing.

**Between Cell 6 and Cell 7 sits one more cell you don't have to run yet** — an
on-demand "print recent backend.log lines" cell. It's not part of the unattended Run
All sequence and does nothing on its own; it's there for later, so you can re-run it
any time (mid-session, next to Cell 7's loop, whenever something looks wrong) to see
what the backend has actually logged recently, without restarting anything. See
section 11 below for how to use it.

**Then it will look like it's hanging on Cell 7.** This is correct, not a bug. Cell 7
is a deliberate infinite loop that keeps the session's kernel busy while you read.
Leave it running for as long as you're using the extension.

**First run takes about 6-8 minutes** end to end (mostly Cell 2's dependency
installs). Every run after that, on the same session, is much faster since packages
and models stay cached in that session's memory — but a brand new session (tomorrow,
or after a restart) starts cold again, at roughly the same 6-8 minutes.

If any cell fails, its output now contains the real error at the exact point it
failed — see section 10 below, or paste it into a chat with an AI assistant if you're
stuck; the error itself is safe to share, just never a secret value.

## 5. Get the Public URL

Cell 6's output prints two lines:

```text
Public URL: https://<your-domain>.ngrok-free.dev
Paste this into the extension popup's Local Pipeline URL field:
  https://<your-domain>.ngrok-free.dev/v1/translate-image
```

```for extension location: 
D:\Desktop\translator D\app\Manga Translator\core_pipeline\extension

for fmt auth token:
Wnak75CQVg13ltjfDLXZr0icqxASpsGbYByPHTdN
```

Copy the **second** line — the one ending in `/v1/translate-image`, not the bare
domain. This matters: the extension's Local Pipeline URL field is used exactly as
typed for translation requests, but every other check (health, warmup) is derived
from it by rewriting the path. A bare domain with no path will still show a green
"Reachable" dot in the popup while every actual translation silently fails — this is
the single most common way to misconfigure this setup.

If you claimed a static domain (section 1) and created the `NGROK_STATIC_DOMAIN`
secret (section 3), this URL is the **same every time you start a new session** — you
only need to do the next section once, not every day.

## 6. Point the Extension at Kaggle

If you haven't loaded the extension yet, follow `docs/USER_GUIDE.md` sections 4-5
first (Load the Extension in Chrome/Brave) — that part is identical whether the
backend is local or on Kaggle.

1. Click the Manga Translator extension icon.
2. In the **Local Pipeline URL** field, paste the full URL from section 5 (ending in
   `/v1/translate-image`).
3. In the **Backend auth token** field, paste the exact `FMT_AUTH_TOKEN` value you
   wrote down in section 3, step 3.
4. Choose your source language.
5. Click **Save**.

The header badge next to the extension title switches from `LOCAL` to `REMOTE` when
it detects the saved URL isn't your own machine — that's confirmation the setting
took.

**A green "Reachable" status does not by itself prove the auth token is correct.**
The health check that drives that dot is intentionally the one ungated backend route
— it doesn't require the token. A wrong or missing token only shows up as a `403` on
the first real **Translate Page** click. If that happens, re-check that the token in
the popup matches `FMT_AUTH_TOKEN` exactly, with no extra spaces.

From here, translating a page works exactly like the local setup — see
`docs/USER_GUIDE.md` sections 7 onward for Translate Page, Auto-translate, cache
behavior, and the selection panel. None of that changes based on where the backend
runs.

## 7. Daily Routine — Starting a Session

Once section 1-6 are done once, starting a new day's reading session is:

1. Open your Kaggle notebook.
2. Click **Run → Run All** (or just re-run cells 1 through 6 if you'd rather watch
   each one).
3. Wait for Cell 6 to print the public URL — if you claimed a static domain, it's the
   same URL as last time, so you don't need to touch the extension popup at all.
4. Once Cell 7 is looping, open your browser and start translating.
5. When you're done, click the ■ Stop button on Cell 7, then run Cell 8 to shut down
   cleanly (closes the tunnel, stops the backend).

You do not need to repeat sections 1-6 (accounts, secrets, extension config) on
future days — only this section, and only step 5 if you want a clean shutdown rather
than just closing the tab (Kaggle will end the session on its own via the idle
watchdog either way, described next).

## 8. What to Expect (Read This Once)

**Kaggle GPU quota:** 30 GPU-hours per week, resetting weekly, and a 9-hour maximum
for any single continuous session even within quota. Plan reading sessions
accordingly — the notebook does not auto-restart when a session ends.

**Idle watchdog:** Kaggle ends an interactive session after roughly 60 minutes with
**no interaction in the browser tab** — and a running cell does not count as
interaction. Cell 7's loop keeps the kernel busy, which is necessary but not
sufficient; keep the tab open and click or scroll in it occasionally, even while Cell
7 runs, or the session ends anyway.

**Only interactive sessions work.** A "Save Version" / batch-commit run executes
headless and its tunnel isn't reachable the way you need — always run this live, not
scheduled.

**ngrok free tier:** 1 GB of bandwidth and 20,000 HTTP requests per month, shared
across everything you tunnel. Every translated page returned as a data URL is
roughly 1.3x the raw image bytes, so budget somewhere around 400-700 translated pages
per month before hitting the cap, depending on page resolution. There is **no time
limit on the tunnel itself** — it can stay open as long as your Kaggle session does.
For light personal reading this bandwidth cap is unlikely to matter.

**Per-page latency:** pipeline processing time (a T4 is a real datacenter GPU, likely
comparable to or faster than a local consumer GPU) plus the round-trip over the
tunnel. For typical page sizes this overhead is small but not zero.

**Both T4s aren't used.** Kaggle's 2xT4 offering only uses the first GPU — the second
sits idle. This is not something you need to fix; one T4 comfortably runs 2
translations at once by default, which is plenty for one person's reading pace.

## 9. When the Session Ends

If you come back and the extension shows "Offline — will resume automatically" and it
doesn't recover within a few seconds, the Kaggle session most likely ended (idle
timeout, 9-hour cap, or you closed the tab). Nothing on the extension side is broken —
go back to section 7 and start a new session; if you claimed a static domain, the
extension needs no changes once the new session's Cell 6 finishes.

## 10. Troubleshooting

### The popup shows a green dot but nothing ever translates

Almost always the bare-origin trap from section 5: the saved URL is missing
`/v1/translate-image`. Open the popup, check the Local Pipeline URL field, and
re-save it with the full path. As of this manual, saving a bare origin
auto-appends the correct path for you — if you're on an older extension build, add it
by hand.

### Every translation returns a 403 ("Missing or invalid auth token")

The value the extension is sending doesn't match what the backend actually has in
memory right now. This one error covers two different real causes — both worth
checking, since the error text can't tell you which:

1. **The backend hasn't picked up a token you changed.** `FMT_AUTH_TOKEN` is read
   once, when the backend process starts (Cell 4) — editing the Kaggle Secret
   afterward does nothing to a process that's already running. If you ever created or
   changed this secret after Cell 4 had already launched once this session, **re-run
   Cell 4** (or Run All) so the new value actually takes effect, then re-save the same
   value in the popup.
2. **You don't have the exact value on both ends.** Kaggle Secrets are write-only
   after you save them — there is no way to open the Secrets panel later and see what
   you typed. If the token wasn't written down somewhere durable *before* it went into
   the Kaggle secret, there's no way to confirm the popup has the same string. The
   only reliable fix is to generate a brand new token, write it down first, set it as
   the `FMT_AUTH_TOKEN` secret, restart the backend (point 1 above), and paste that
   same fresh value into the popup.

Copy-paste both ends rather than retyping either — a single wrong character produces
this exact same generic failure with no more specific hint.

### The response looks like an HTML page instead of a translation

This is ngrok's free-tier browser-warning interstitial. The extension and backend
already send the header that bypasses it (`ngrok-skip-browser-warning`) — if you still
see this, you're likely running an older build of either the extension or the
notebook; re-import the latest notebook and reload the extension.

### The public URL changed since last time

This only happens if you didn't claim a static domain (section 1) or didn't create
the `NGROK_STATIC_DOMAIN` secret (section 3) — without those, ngrok assigns a new
random URL every session. Go back and do both; afterward the URL stays fixed.

### The Kaggle session ended mid-read

See section 9. Start a new session; nothing to fix on the extension side.

### Cell 2's smoke test fails

The notebook's own error output at that point already contains the real traceback —
that's specifically designed so you don't have to guess. See
`core_pipeline/docs/KAGGLE_DEPLOYMENT.md` section 10 for the full troubleshooting
table covering every cell.

## 11. Debugging — What to Check and What to Paste Back

If something's wrong and section 10 above didn't cover it, this section gives you the
actual commands to run and exactly what to paste into a chat when asking for help. Run
these roughly in this order — each one narrows down where the problem is.

### 1. The log-tail cell (fastest, start here)

Between Cell 6 and Cell 7 in the notebook is a cell that prints the last 150 lines of
`/kaggle/working/backend.log` — the backend's own stdout/stderr, redirected there by
Cell 4. Run it any time; it's read-only and doesn't touch the running backend or
tunnel. **This is usually the single most useful thing to paste back.** It requires
Cell 5 to have already run once in this kernel (it reuses a helper function Cell 5
defines) — if you get a `NameError`, run Cell 5 first, then retry.

If you'd rather look at the whole file instead of just the tail, add a new cell
anywhere after Cell 4 and run:

```python
with open("/kaggle/working/backend.log") as f:
    print(f.read())
```

### 2. Health and warmup state — `/v1/health`

Tells you whether the backend process is up, what it thinks its own warmup state is,
and current GPU/scheduler load — without needing the extension popup open. Run from a
new notebook cell (loopback, no auth header games) or from your own machine against the
tunnel URL (needs both headers plus the ngrok bypass header).

From a Kaggle cell (loopback):
```python
get_json("http://127.0.0.1:8766/v1/health")
```
(`get_json` is defined in Cell 5 — run it first if you haven't this session.)

From your own machine, against the public tunnel URL — PowerShell:
```powershell
Invoke-RestMethod "https://<your-domain>.ngrok-free.dev/v1/health" -Headers @{
  "X-Fmt-Client" = "free-manga-translator-extension"
  "X-Fmt-Auth"   = "<your FMT_AUTH_TOKEN>"
  "ngrok-skip-browser-warning" = "1"
}
```
or curl:
```bash
curl -s "https://<your-domain>.ngrok-free.dev/v1/health" \
  -H "X-Fmt-Client: free-manga-translator-extension" \
  -H "X-Fmt-Auth: <your FMT_AUTH_TOKEN>" \
  -H "ngrok-skip-browser-warning: 1"
```

Fields worth checking in the response:
- `commit` — will be `null` on Kaggle (see section 4 above); not a bug.
- `warmup.status` — one of `idle` (never warmed up), `running`, `pass` (ready),
  `fail`, `released` (GPU freed after idle), or `disabled` (only before Cell 5's
  forced warmup POST has run this session — Kaggle sets `FMT_STARTUP_WARMUP=0`, so
  this is expected to say `disabled` if you check *before* Cell 5 finishes).
- `scheduler.active` — how many translation jobs are running right now.
- `scheduler.gpu` — `total_mb`/`used_mb`/`free_mb` VRAM, straight from `nvidia-smi`.

The `ngrok-skip-browser-warning` header matters over the public tunnel — without it
you can get a 200 response that's an HTML interstitial page instead of JSON (see the
next item for how to tell these apart automatically).

### 3. VRAM and provider quota — `/v1/vram-status`, `/v1/quota-status`

Same request pattern as above, different paths. `/v1/vram-status` gives the same GPU
numbers as the popup's VRAM panel; `/v1/quota-status` gives today's per-provider usage
counts (which providers are actually configured and how much of each is used) — useful
if translations are unexpectedly falling back to the slower local NLLB translator.

### 4. Reproduce the exact extension request — `run_remote_backend_smoke.py`

This is the one tool that reproduces tunnel-specific failures the loopback Cell 5b
can't see (since Cell 5b runs before any tunnel exists). It sends the same request
shape the extension sends, straight at your public tunnel URL, and tells you which of
three failure modes you're hitting: a real HTTP error from the backend, the tunnel
host being unreachable entirely, or a 200 response that's actually an ngrok
interstitial HTML page instead of JSON (the same trap `/v1/health` can hit, but this
script checks for it explicitly instead of you having to notice).

Run from your own machine, inside `core_pipeline/`, with your local Python environment
(see `LOCAL_USER_MANUAL.md` for the interpreter path):

```powershell
& $py "python\diagnostics\run_remote_backend_smoke.py" --url "https://<your-domain>.ngrok-free.dev/v1/translate-image" --auth-token "<your FMT_AUTH_TOKEN>"
```

(the full path matters here too — same bare-origin trap as section 5; the script itself
will tell you plainly if you get it wrong)

### 5. GPU sanity check — `nvidia-smi`

Cell 1 already prints this once at the very start of a session. If you want a fresh
reading later (e.g. VRAM looks wrong in `/v1/vram-status`), add a new cell anywhere and
run:

```python
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

### What to actually paste into chat

In order of usefulness, if you're reporting a problem:

1. The log-tail cell's output (item 1 above) — almost always paste this first.
2. The `/v1/health` response (item 2) — one line, tells you warmup/scheduler state.
3. Which cell (if any) actually failed, and its full output — Kaggle keeps this
   visible in the notebook even after the run stops.
4. If it's specifically "translations don't work but the notebook looks fine," the
   `run_remote_backend_smoke.py` output (item 4) — it's built to answer exactly that.

None of the above ever prints a secret value — it's all safe to paste as-is. The one
exception is your `FMT_AUTH_TOKEN` itself if you type it directly into a command as
shown above; don't paste the command line with the real token filled in, only its
output.

## 12. Security Notes

- Kaggle Secrets are encrypted at rest and are never included if you share or fork
  the notebook — a copy of your notebook does not carry your keys with it.
- Never `print()` a secret value in a notebook cell, even partially — Kaggle saves
  output cells with the notebook version.
- Never make the notebook public while secrets are attached.
- The `FMT_AUTH_TOKEN` is the only thing standing between your public tunnel URL and
  a stranger who finds it — treat it like a password, not like the provider API keys
  (which are never exposed to the tunnel at all, only used server-side).
- If any of your provider keys have ever been pasted into a chat, a screenshot, or
  anywhere outside your own `.env` file and the provider's dashboard, rotate that key
  on the provider's dashboard before using it here.

## 13. Daily Copy-Paste Checklist

For when you've read all of the above once and just need the routine:

1. Kaggle notebook open → **Run → Run All**.
2. Wait for Cell 6's public URL (same URL every time, if you claimed a static
   domain).
3. Extension already configured from section 6 — no changes needed unless the URL
   changed.
4. Read, translating pages as normal.
5. Done reading: Cell 7 → ■ Stop → run Cell 8.
