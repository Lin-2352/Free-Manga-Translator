// Free Manga Translator - Local-only background service worker.
// All image translation goes through the local Python 8-step pipeline bridge.

const DEFAULT_LOCAL_PIPELINE_URL = 'http://127.0.0.1:8766/v1/translate-image';
const APP_VERSION = '1.1.15';
// Every backend route except /health now requires this header (backend_api/app/main.py). It is
// not a secret -- it forces the browser to CORS-preflight requests to the local pipeline server,
// closing an unpreflighted "simple request" CSRF gap against a companion API with no other auth.
const FMT_CLIENT_HEADER = 'X-Fmt-Client';
const FMT_CLIENT_VALUE = 'free-manga-translator-extension';
// Optional second header, only sent when the user has set a token in the popup (e.g. the backend
// is reachable over a public tunnel rather than loopback -- see
// core_pipeline/docs/KAGGLE_DEPLOYMENT.md and docs/KAGGLE_USER_MANUAL.md). Empty or unset locally
// means this never gets added, so every existing local deployment is unaffected.
const FMT_AUTH_HEADER = 'X-Fmt-Auth';
// Bypasses ngrok's free-tier browser-warning interstitial (ERR_NGROK_6024), which can otherwise
// return an HTML page instead of the real JSON response on a free *.ngrok-free.app tunnel. Sent
// unconditionally -- harmless against a plain loopback backend, which just ignores an extra header.
const NGROK_SKIP_WARNING_HEADER = 'ngrok-skip-browser-warning';

function fmtHeaders(settings, extra = {}) {
  const headers = {
    ...extra,
    [FMT_CLIENT_HEADER]: FMT_CLIENT_VALUE,
    [NGROK_SKIP_WARNING_HEADER]: '1',
  };
  if (settings?.localPipelineAuthToken) headers[FMT_AUTH_HEADER] = settings.localPipelineAuthToken;
  return headers;
}
const CACHE_VERSION = `local-8-step-v13-quality-performance-hardening-v${APP_VERSION}`;
// Each cache entry is its own chrome.storage.local item under this prefix (fmtCacheV1:<cacheId>),
// not one giant blob under a single key. cacheId already embeds CACHE_VERSION (buildCacheId), so
// a stale-version key self-identifies without needing a separate prefix per version.
const CACHE_ENTRY_PREFIX = 'fmtCacheV1:';
const LEGACY_SESSION_CACHE_KEY = 'translationCacheEntries';
const DEFAULT_PARALLEL_LIMIT = 2;
const MAX_PARALLEL_LIMIT = 3;
// Must stay >= content.js's DEFAULT_AUTO_QUEUE_LIMIT (20): navigating across
// several image-heavy pages queues up to that many translations ahead, and a
// cache smaller than the queue-ahead depth LRU-evicts the page you started
// on before you've even finished reading it -- the exact "sometimes it's
// there when I come back, sometimes it isn't" inconsistency this fixes.
const DEFAULT_CACHE_LIMIT = 24;
const MAX_CACHE_LIMIT = 40;
const DEFAULT_QUEUE_LIMIT = 20;
const MAX_QUEUE_LIMIT = 50;
// A cold-GPU pipeline run has been observed taking 1-3+ minutes; 360s leaves real margin
// while still guaranteeing a hung request can't hold a parallel slot for an entire session.
const DEFAULT_FETCH_TIMEOUT_MS = 360_000;
const MIN_FETCH_TIMEOUT_MS = 30_000;
const MAX_FETCH_TIMEOUT_MS = 900_000;
// A busy backend already waited out its own gpu_scheduler.py max_wait_seconds (60s by
// default) before returning PIPELINE_BUSY -- that request lost a race that wasn't its fault,
// so it gets a bounded number of background-owned retries instead of being handed back to
// content.js, whose own node-anchored retry silently no-ops on a single-<img> page viewer
// once the node has moved on to a later page (see requeueIfBusy()).
const MAX_BUSY_REQUEUE_ATTEMPTS = 3;
const BUSY_REQUEUE_BASE_DELAY_MS = 2000; // 2s, 4s, 6s across the 3 attempts
// How long a learned backend capacity reading stays trusted before falling back to the
// user's configured parallelLimit -- long enough to survive normal polling gaps, short
// enough that a backend restart (capacity can change with free VRAM) is noticed quickly.
const BACKEND_CAPACITY_TTL_MS = 120_000;

const outgoingRequests = new Map();
// Parallel to outgoingRequests, keyed the same way: outgoingRequests only ever stored the raw
// promise, with no way to tell a caller (the popup's live queue list) WHICH page/image is
// actually dispatched right now. Set/deleted alongside outgoingRequests itself, never read from
// admission-decision code paths so it can't affect dispatch/ordering, only what's displayed.
const outgoingRequestMeta = new Map();
const activeControllers = new Map();
const queuedRequests = new Map();
const requestQueue = [];
const translationCache = new Map();

// MV3 service workers can be torn down by Chrome's own idle/lifetime limits WHILE a translate
// request is still awaiting the local pipeline fetch (observed: requests still pending past
// ~5-10 minutes under a deep queue backlog). A held chrome.runtime.onMessage sendResponse from
// before the teardown is unrecoverable once that happens -- the caller sees "the message channel
// closed before a response was received" and the work has to be retried from content.js. This
// alarm-based heartbeat does not prevent that outright (Chrome can still terminate a worker for
// reasons unrelated to idle time), but a periodic alarm event is itself an activity signal that
// measurably extends how long Chrome keeps a worker alive with in-flight work, and unlike a bare
// fetch() it is not dependent on any particular Chrome version's fetch-keepalive behavior. It only
// runs while there is real outstanding work (in-flight or queued), and is cleared the moment there
// is none, so it costs nothing at idle.
const KEEP_ALIVE_ALARM_NAME = 'fmt-translation-keep-alive';
const KEEP_ALIVE_PERIOD_MINUTES = 0.4; // ~24s -- under Chrome's ~30s SW idle-kill window
let keepAliveAlarmActive = false;

function ensureKeepAliveAlarm() {
  if (keepAliveAlarmActive || !chrome.alarms?.create) return;
  keepAliveAlarmActive = true;
  chrome.alarms.create(KEEP_ALIVE_ALARM_NAME, { periodInMinutes: KEEP_ALIVE_PERIOD_MINUTES });
}

function maybeClearKeepAliveAlarm() {
  if (!keepAliveAlarmActive) return;
  if (outgoingRequests.size > 0 || requestQueue.length > 0) return;
  keepAliveAlarmActive = false;
  chrome.alarms.clear(KEEP_ALIVE_ALARM_NAME).catch(() => {});
}

chrome.alarms?.onAlarm?.addListener((alarm) => {
  if (alarm.name !== KEEP_ALIVE_ALARM_NAME) return;
  // The alarm firing is the actual keep-alive mechanism; touching storage here is only to give
  // the handler a real async op instead of an empty no-op, in case that matters to any given
  // Chrome version's activity accounting.
  chrome.storage.session?.get?.([]).catch(() => {});
  if (outgoingRequests.size === 0 && requestQueue.length === 0) {
    maybeClearKeepAliveAlarm();
  }
});

// outgoingRequests.size only reflects a dispatch once processTranslation's async body has
// run past several awaits (getSettings, cache lookups, ...). Two admission checks that read
// outgoingRequests.size back-to-back within that window both see the pre-dispatch count, so
// a burst of arrivals (or a queue drain) can admit more than parallelLimit at once. This
// counter is incremented synchronously at the exact moment a dispatch is decided (before any
// await), so every admission check sees in-flight-but-not-yet-registered dispatches too.
let reservedSlots = 0;

// The backend's real GPU-slot capacity, learned opportunistically -- never queried for on
// its own. Two write sites: checkPipelineHealth() (scheduler.capacity, from an already-made
// /v1/health fetch) and processTranslation()'s success path (report.scheduler.capacityAtAcquire,
// from an already-made translate response). Latest-write-wins by design: capacityAtAcquire
// reads low during a burst of admissions (gpu_scheduler.py's admission grace window
// deliberately discounts recent admissions), so latching a running minimum would get
// permanently stuck low after any burst. A stale/unknown reading (backendCapacity === 0, or
// older than BACKEND_CAPACITY_TTL_MS) falls back to the user's configured parallelLimit --
// i.e. today's exact behavior -- so this can only ever narrow concurrency, never widen it
// past what the user asked for.
let backendCapacity = 0;
let backendCapacityAt = 0;

function noteBackendCapacity(capacity) {
  const value = Number(capacity);
  if (!Number.isFinite(value) || value < 1) return;
  backendCapacity = Math.floor(value);
  backendCapacityAt = nowMs();
}

function effectiveParallelLimit(settings) {
  const configured = settings?.parallelLimit ?? DEFAULT_PARALLEL_LIMIT;
  if (backendCapacity < 1) return configured;
  if (nowMs() - backendCapacityAt > BACKEND_CAPACITY_TTL_MS) return configured;
  return Math.max(1, Math.min(configured, backendCapacity));
}

// Tripped only by a fetch()-level rejection inside callLocalPipeline's own await -- the ONLY
// place callLocalPipeline throws a TypeError; every other throw there (!response.ok branch,
// LOCAL_PIPELINE_EMPTY_RESPONSE) is a plain Error, proving the server DID respond. So
// error?.name === 'TypeError' is the precise, narrow signal for "server unreachable" -- distinct
// from an ordinary PIPELINE_TIMEOUT (one slow request, server may still be alive) and distinct
// from an HTTP error response (server IS reachable). Once tripped, new/retried dispatches
// short-circuit instead of each independently re-waiting out settings.fetchTimeoutMs.
let breakerOpen = false;
let breakerOpenedAt = 0;
let breakerNextProbeAt = 0;
const BREAKER_PROBE_INTERVAL_MS = 12_000;

function tripBreaker() {
  if (breakerOpen) return;
  breakerOpen = true;
  breakerOpenedAt = nowMs();
  breakerNextProbeAt = breakerOpenedAt + BREAKER_PROBE_INTERVAL_MS;
  console.warn('[FMT] local pipeline appears unreachable; short-circuiting new requests until it recovers');
}

function resetBreaker() {
  if (!breakerOpen) return;
  breakerOpen = false;
  breakerOpenedAt = 0;
  breakerNextProbeAt = 0;
}

// false => short-circuit this dispatch with PIPELINE_OFFLINE instead of touching the network.
function shouldAttemptDispatch() {
  if (!breakerOpen) return true;
  const now = nowMs();
  if (now < breakerNextProbeAt) return false;
  breakerNextProbeAt = now + BREAKER_PROBE_INTERVAL_MS;
  return true;
}

let cacheLoaded = false;
// Set only after loadCacheFromStorage()'s awaits fully resolve (see ensureCacheLoaded) --
// memoizes the in-flight LOAD, not just a "loaded" flag, so a second caller arriving while the
// first is still awaiting chrome.storage.local.get() joins the same promise instead of racing
// ahead against a still-empty translationCache.
let cacheLoadPromise = null;
let isPaused = false;
// Mirrors queueActivitySinceBoot below: a real pause/resume action (via setTranslationPaused,
// stopTranslationRuntime, or a storage.onChanged event reflecting one from another context) can
// land BEFORE this boot-time storage.local.get resolves. Without this flag, that still-pending
// read finishing late clobbers the just-applied real state back to whatever was persisted before
// boot -- observed as the extension silently re-pausing (or un-pausing) itself moments after the
// user acted, for no visible reason. Any real pause-state write, from any source, sets this so the
// stale boot read can no longer overwrite isPaused.
let pauseStateSetSinceBoot = false;

chrome.storage.local.get(['translationPaused']).then((result) => {
  if (pauseStateSetSinceBoot) return;
  isPaused = result.translationPaused === true;
}).catch(() => {});

// requestQueue is in-memory only -- a service-worker idle-kill wipes it silently, leaving the
// popup polling a queue that no longer exists. We don't attempt to recover the lost work (that
// would mean re-fetching pages the user may have already left, spending real backend/API quota
// on their behalf with no way to know if they still want it); we just persist enough to tell
// them truthfully that it happened, once, on the next SW instantiation.
let restartLost = 0;
// A real translate request can wake a freshly-instantiated SW before the async
// chrome.storage.session.get below resolves -- without this flag, queueTranslation's own
// restartLost = 0 (the "user already re-ran Translate Page" clear) can run and then be
// overwritten moments later when that read resolves and sets restartLost again, resurrecting a
// warning the user already acted on.
let queueActivitySinceBoot = false;
const QUEUE_DESCRIPTOR_KEY = 'fmtQueueDescriptorsV1';

function persistQueueDescriptors() {
  if (!chrome.storage.session?.set) return;
  const cacheIds = requestQueue.map((item) => item.cacheId).filter(Boolean);
  if (cacheIds.length === 0) {
    chrome.storage.session.remove?.([QUEUE_DESCRIPTOR_KEY]).catch(() => {});
    return;
  }
  chrome.storage.session.set({ [QUEUE_DESCRIPTOR_KEY]: cacheIds }).catch(() => {});
}

chrome.storage.session?.get?.([QUEUE_DESCRIPTOR_KEY]).then((result) => {
  const lost = result?.[QUEUE_DESCRIPTOR_KEY];
  // If real queue activity already happened this boot, that activity already reset restartLost
  // (queueTranslation) as the honest "re-run happened" signal -- this stale read must not
  // override it. The descriptor key is still always removed either way so it can't resurface
  // on a later boot.
  if (Array.isArray(lost) && lost.length > 0 && !queueActivitySinceBoot) {
    restartLost = lost.length;
    console.warn(`[FMT] ${lost.length} queued translation(s) lost in a service worker restart`);
  }
  return chrome.storage.session?.remove?.([QUEUE_DESCRIPTOR_KEY]);
}).catch(() => {});

function fastHash(str) {
  if (!str) return '';
  const len = str.length;
  let hash = 2166136261 >>> 0;
  for (let i = 0; i < len; i += 1) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  hash ^= len;
  return Math.imul(hash, 16777619).toString(36);
}

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

async function getSettings() {
  const result = await chrome.storage.local.get([
    'localPipelineUrl',
    'localPipelineLanguage',
    'localPipelineAuthToken',
    'translationCachePages',
    'translationQueuePages',
    'translationParallelPages',
    'translationFetchTimeoutMs',
  ]);
  const cacheLimit = Number.parseInt(result.translationCachePages, 10);
  const queueLimit = Number.parseInt(result.translationQueuePages, 10);
  const parallelLimit = Number.parseInt(result.translationParallelPages, 10);
  const fetchTimeoutMs = Number.parseInt(result.translationFetchTimeoutMs, 10);
  return {
    localPipelineUrl: String(result.localPipelineUrl || DEFAULT_LOCAL_PIPELINE_URL).trim() || DEFAULT_LOCAL_PIPELINE_URL,
    localPipelineLanguage: String(result.localPipelineLanguage || 'ja').trim() || 'ja',
    localPipelineAuthToken: String(result.localPipelineAuthToken || '').trim(),
    cacheLimit: Number.isFinite(cacheLimit)
      ? Math.max(0, Math.min(MAX_CACHE_LIMIT, cacheLimit))
      : DEFAULT_CACHE_LIMIT,
    queueLimit: Number.isFinite(queueLimit)
      ? Math.max(0, Math.min(MAX_QUEUE_LIMIT, queueLimit))
      : DEFAULT_QUEUE_LIMIT,
    parallelLimit: Number.isFinite(parallelLimit)
      ? Math.max(1, Math.min(MAX_PARALLEL_LIMIT, parallelLimit))
      : DEFAULT_PARALLEL_LIMIT,
    fetchTimeoutMs: Number.isFinite(fetchTimeoutMs)
      ? Math.max(MIN_FETCH_TIMEOUT_MS, Math.min(MAX_FETCH_TIMEOUT_MS, fetchTimeoutMs))
      : DEFAULT_FETCH_TIMEOUT_MS,
  };
}

function buildQueueStats(settings) {
  const parallelLimit = settings?.parallelLimit ?? DEFAULT_PARALLEL_LIMIT;
  const queueLimit = settings?.queueLimit ?? DEFAULT_QUEUE_LIMIT;
  const capacity = Math.max(1, parallelLimit + queueLimit);
  const queuedCount = requestQueue.length;
  // reservedSlots covers the exact same dispatch-in-flight-but-not-yet-
  // registered window documented at dispatchTranslation()/processQueue()'s
  // own admission checks -- outgoingRequests.size alone under-reports
  // Active during that window (see reservedSlots' own comment for why
  // adding both here can't double-count: the reservation releases the
  // moment the request actually registers in outgoingRequests).
  const activeCount = outgoingRequests.size + reservedSlots;
  // Per-item breakdown for the popup's live queue list: active items first (whichever page/image
  // actually holds a GPU slot right now), then queued items in the SAME order they sit in
  // requestQueue -- push() at the tail, shift() from the head, so this position is exactly the
  // real FIFO dispatch order, not a re-derived guess.
  const items = [
    ...Array.from(outgoingRequestMeta.entries()).map(([cacheId, meta]) => ({
      cacheId,
      pageHost: pageHostFromUrl(meta.pageUrl),
      originalImageUrl: meta.originalImageUrl,
      status: 'active',
    })),
    ...requestQueue.map((entry, index) => ({
      cacheId: entry.cacheId || '',
      pageHost: pageHostFromUrl(entry.message?.pageUrl),
      originalImageUrl: entry.message?.originalImageUrl || '',
      status: 'queued',
      position: index + 1,
    })),
  ];
  return {
    cacheSize: translationCache.size,
    cacheLimit: settings?.cacheLimit ?? DEFAULT_CACHE_LIMIT,
    cachePersistFailures,
    activeRequests: activeCount,
    queueLength: queuedCount,
    queueLimit,
    parallelLimit,
    // Read-only: the learned backend capacity and the resulting effective dispatch limit.
    // Deliberately NOT folded into parallelLimit itself -- popup.js echoes parallelLimit back
    // after a Save, and the queue stress-test suite asserts activeRequests === parallelLimit;
    // both would spuriously disagree with a live poll if this clamp were applied upstream.
    backendCapacity: backendCapacity > 0 && (nowMs() - backendCapacityAt <= BACKEND_CAPACITY_TTL_MS)
      ? backendCapacity
      : null,
    effectiveParallelLimit: effectiveParallelLimit(settings),
    queuedUnique: queuedRequests.size,
    items,
    isPaused,
    restartLost,
    pipelineBreakerOpen: breakerOpen,
    pressurePercent: Math.min(100, Math.round(((activeCount + queuedCount) / capacity) * 100)),
  };
}

function healthUrlForPipeline(pipelineUrl) {
  try {
    const url = new URL(pipelineUrl);
    if (url.pathname.endsWith('/translate-image') || url.pathname.endsWith('/translate-snapshot')) {
      url.pathname = url.pathname.replace(/\/v1\/translate-(image|snapshot)$/, '/v1/health');
    } else if (url.pathname.endsWith('/translate')) {
      url.pathname = '/v1/health';
    } else {
      url.pathname = '/v1/health';
    }
    url.search = '';
    url.hash = '';
    return url.toString();
  } catch {
    return 'http://127.0.0.1:8766/v1/health';
  }
}

function warmupUrlForPipeline(pipelineUrl) {
  try {
    const url = new URL(pipelineUrl);
    url.pathname = '/v1/warmup';
    url.search = '';
    url.hash = '';
    return url.toString();
  } catch {
    return 'http://127.0.0.1:8766/v1/warmup';
  }
}

function quotaUrlForPipeline(pipelineUrl) {
  try {
    const url = new URL(pipelineUrl);
    url.pathname = '/v1/quota-status';
    url.search = '';
    url.hash = '';
    return url.toString();
  } catch {
    return 'http://127.0.0.1:8766/v1/quota-status';
  }
}

function vramUrlForPipeline(pipelineUrl) {
  try {
    const url = new URL(pipelineUrl);
    url.pathname = '/v1/vram-status';
    url.search = '';
    url.hash = '';
    return url.toString();
  } catch {
    return 'http://127.0.0.1:8766/v1/vram-status';
  }
}

function backendUrlForPipeline(pipelineUrl, pathname) {
  try {
    const url = new URL(pipelineUrl);
    url.pathname = pathname;
    url.search = '';
    url.hash = '';
    return url.toString();
  } catch {
    return `http://127.0.0.1:8766${pathname}`;
  }
}

function diagnosticLogUrlForPipeline(pipelineUrl) {
  return backendUrlForPipeline(pipelineUrl, '/v1/diagnostics/log');
}

function makeTraceId(prefix = 'bg') {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function compactForDiagnostics(value) {
  if (value === null || value === undefined) return value;
  if (typeof value === 'string') {
    if (value.startsWith('data:image/')) return `<data-url:${value.length}>`;
    return value.length > 260 ? `${value.slice(0, 260)}...` : value;
  }
  if (Array.isArray(value)) return value.slice(0, 12).map(compactForDiagnostics);
  if (typeof value === 'object') {
    const result = {};
    for (const [key, item] of Object.entries(value)) {
      if (/base64|imageData|translatedImageDataUrl|imageDataUrl/i.test(key)) {
        result[key] = '<redacted>';
      } else {
        result[key] = compactForDiagnostics(item);
      }
    }
    return result;
  }
  return value;
}

async function sendDiagnosticLog(event, details = {}, settings = null) {
  try {
    const resolvedSettings = settings || await getSettings();
    const payload = {
      event,
      version: APP_VERSION,
      traceId: details.traceId || makeTraceId('diag'),
      timestamp: new Date().toISOString(),
      details: compactForDiagnostics(details),
    };
    await fetch(diagnosticLogUrlForPipeline(resolvedSettings.localPipelineUrl), {
      method: 'POST',
      cache: 'no-store',
      headers: fmtHeaders(resolvedSettings, { 'Content-Type': 'application/json' }),
      body: JSON.stringify(payload),
    });
  } catch {
  }
}

function logWindowUrlForPipeline(pipelineUrl, logType) {
  try {
    const url = new URL(pipelineUrl);
    url.pathname = `/v1/open-log-window/${encodeURIComponent(logType)}`;
    url.search = '';
    url.hash = '';
    return url.toString();
  } catch {
    return `http://127.0.0.1:8766/v1/open-log-window/${encodeURIComponent(logType)}`;
  }
}

async function postPipelineControl(settings, pathname) {
  try {
    const response = await fetch(backendUrlForPipeline(settings.localPipelineUrl, pathname), {
      method: 'POST',
      cache: 'no-store',
      headers: fmtHeaders(settings),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(responseErrorDetail(payload) || `CONTROL_${response.status}`);
    return payload;
  } catch (error) {
    return { success: false, error: error.message };
  }
}

const warmupRequestedFor = new Set();

async function requestPipelineWarmup(settings, force = false) {
  const key = `${settings.localPipelineUrl}:${settings.localPipelineLanguage}`;
  if (!force && warmupRequestedFor.has(key)) return { ok: true, skipped: true };
  warmupRequestedFor.add(key);
  try {
    const response = await fetch(warmupUrlForPipeline(settings.localPipelineUrl), {
      method: 'POST',
      cache: 'no-store',
      headers: fmtHeaders(settings),
    });
    if (!response.ok) warmupRequestedFor.delete(key);
    const payload = await response.json().catch(() => ({}));
    return {
      ok: response.ok,
      skipped: false,
      status: response.status,
      payload,
    };
  } catch (error) {
    warmupRequestedFor.delete(key);
    return { ok: false, skipped: false, error: error.message };
  }
}

async function checkPipelineHealth(settings, options = {}) {
  try {
    const response = await fetch(healthUrlForPipeline(settings.localPipelineUrl), {
      method: 'GET',
      cache: 'no-store',
      // Was missing: /v1/health is the ONE request that decides "is the backend up" for
      // the badge/Start Engine, and it was the only fmtHeaders() caller sending no headers
      // at all -- so it never carried the ngrok interstitial-bypass header, meaning a free
      // ngrok tunnel could hand back its HTML warning page here (and fail this check) while
      // every other request correctly bypassed it.
      headers: fmtHeaders(settings),
    });
    if (!response.ok) throw new Error(`HEALTH_${response.status}`);
    // /v1/health already carries the scheduler's live capacity (gpu_scheduler.py's VRAM-based
    // admission estimate) -- read it here rather than adding a dedicated fetch elsewhere.
    const payload = await response.json().catch(() => null);
    noteBackendCapacity(payload?.scheduler?.capacity);
    requestPipelineWarmup(settings).catch(() => {});
    // resetBreaker() was previously called ONLY from the translate dispatch path -- so a
    // breaker opened by one transient failure (a preflight revalidation landing badly, an
    // ngrok tunnel blip) stayed open until the next translate attempt happened to land in a
    // shouldAttemptDispatch() probe window, or forever if nothing was actively translating.
    // A successful health check is definitionally proof the backend is reachable right now,
    // so any earlier "give up" latch is stale -- clearing it here means opening the popup (or
    // Save, or engine start, the three existing callers) is its own independent recovery path,
    // and fixes the "green for a moment, then flips red" popup bug: without this, the health
    // check itself could succeed while the breaker (checked separately by refreshStats) stayed
    // open from an unrelated earlier failure.
    resetBreaker();
    return true;
  } catch {
    if (options.clearCacheOnFailure) await clearTranslationCache();
    return false;
  }
}

// A configured pipeline URL that isn't loopback is a remote backend (Kaggle+ngrok being the
// only supported case today) -- used to swap the "start a local server" guidance for tunnel
// guidance, since telling a Kaggle user to open PowerShell and run uvicorn is actively wrong.
function isLoopbackPipelineUrl(pipelineUrl) {
  try {
    const host = new URL(pipelineUrl).hostname;
    return host === '127.0.0.1' || host === 'localhost' || host === '[::1]';
  } catch {
    return true; // unparseable -- assume local, the historical default, rather than guess remote
  }
}

function manualStartStepsForEngine(pipelineUrl) {
  if (!isLoopbackPipelineUrl(pipelineUrl)) {
    return [
      'This URL points at a remote backend (Kaggle + ngrok), not a local server -- Chrome cannot start that for you.',
      'Confirm the Kaggle notebook session is still running (Kaggle sessions end after ~60 min idle or a 9-hour max).',
      'If the session ended, re-run the notebook (Run All) and wait for Cell 6 to print a public URL.',
      'If the session is running but this still fails, re-run Cell 6 -- the ngrok URL may have changed if the tunnel restarted.',
      'Confirm the URL saved here still ends in /v1/translate-image and the auth token matches the notebook\'s FMT_AUTH_TOKEN secret.',
      'See docs/KAGGLE_USER_MANUAL.md for the full walkthrough.',
    ];
  }
  return [
    'Chrome and Brave extensions cannot directly launch Python for every user without a native messaging host.',
    'Open PowerShell in the app core_pipeline folder.',
    'Run .\\start_backend.ps1, or run python -m uvicorn backend_api.app.main:app --host 127.0.0.1 --port 8766.',
    'After the terminal says Uvicorn is running, press Start Engine again.',
  ];
}

async function startPipelineEngine() {
  const settings = await getSettings();
  const healthy = await checkPipelineHealth(settings, { clearCacheOnFailure: false });
  if (!healthy) {
    return {
      ok: false,
      available: false,
      needsManualStart: true,
      message: isLoopbackPipelineUrl(settings.localPipelineUrl)
        ? 'Local backend is not running. Start it from PowerShell, then retry.'
        : 'Remote backend is not reachable. Check the Kaggle session and tunnel, then retry.',
      manualStartSteps: manualStartStepsForEngine(settings.localPipelineUrl),
    };
  }
  const warmup = await requestPipelineWarmup(settings, true);
  return {
    ok: warmup?.ok !== false,
    available: true,
    warmup,
    message: warmup?.ok === false ? (warmup.error || 'Warmup request failed') : 'Local backend is reachable.',
  };
}

async function getPipelineQuotaStatus(settings) {
  try {
    const response = await fetch(quotaUrlForPipeline(settings.localPipelineUrl), {
      method: 'GET',
      cache: 'no-store',
      headers: fmtHeaders(settings),
    });
    if (!response.ok) throw new Error(`QUOTA_${response.status}`);
    return await response.json();
  } catch (error) {
    return { ok: false, error: error.message, providers: [] };
  }
}

async function getPipelineVramStatus(settings) {
  try {
    const response = await fetch(vramUrlForPipeline(settings.localPipelineUrl), {
      method: 'GET',
      cache: 'no-store',
      headers: fmtHeaders(settings),
    });
    if (!response.ok) throw new Error(`VRAM_${response.status}`);
    return await response.json();
  } catch (error) {
    return { ok: false, available: false, error: error.message, gpus: [] };
  }
}

async function openPipelineLogWindow(settings, logType) {
  try {
    const response = await fetch(logWindowUrlForPipeline(settings.localPipelineUrl, logType), {
      method: 'POST',
      cache: 'no-store',
      headers: fmtHeaders(settings),
    });
    if (!response.ok) throw new Error(`LOG_WINDOW_${response.status}`);
    return await response.json();
  } catch (error) {
    return { success: false, error: error.message, logType };
  }
}

async function stopPipelineRuntime(settings, mode) {
  const pathname = mode === 'hard' ? '/v1/runtime/hard-stop' : '/v1/runtime/soft-stop';
  return postPipelineControl(settings, pathname);
}

async function releasePipelineGpu(settings) {
  return postPipelineControl(settings, '/v1/gpu/release');
}

async function clearPipelineRuntimeCache(settings) {
  return postPipelineControl(settings, '/v1/cache/clear');
}

function responseErrorDetail(payload) {
  if (!payload || typeof payload !== 'object') return '';
  if (payload.error) return String(payload.error);
  if (payload.detail) {
    if (typeof payload.detail === 'string') return payload.detail;
    if (payload.detail.message) return String(payload.detail.message);
    try {
      return JSON.stringify(payload.detail);
    } catch {
      return String(payload.detail);
    }
  }
  return '';
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// Errors thrown here are tagged isImageSourceError so the catch block in processTranslation can
// tell "the target image URL itself is broken/unreachable" apart from "the local pipeline backend
// is unreachable" -- both can surface as a fetch()-level TypeError, but only the latter is real
// evidence the backend is down. Without this tag a single broken/dead image src on a page (a
// common, harmless occurrence) would trip the same circuit breaker as an actual backend outage,
// short-circuiting every OTHER image on the page with a false PIPELINE_OFFLINE.
function taggedImageSourceError(message) {
  const error = new Error(message);
  error.isImageSourceError = true;
  return error;
}

async function fetchImageAsDataUrl(imageUrl) {
  if (!imageUrl || !/^https?:|^file:|^data:/i.test(imageUrl)) {
    throw taggedImageSourceError('IMAGE_URL_UNSUPPORTED');
  }
  if (imageUrl.startsWith('data:')) return imageUrl;
  let response;
  try {
    response = await fetch(imageUrl, { credentials: 'omit', cache: 'force-cache' });
  } catch (fetchError) {
    // A network-level failure fetching the IMAGE (DNS/connection-refused/CORS) throws the exact
    // same TypeError shape as a failed backend fetch -- rethrow as a plain, tagged Error so it
    // can never be mistaken for "the backend pipeline itself is unreachable".
    throw taggedImageSourceError(`IMAGE_FETCH_NETWORK_ERROR: ${fetchError?.message || fetchError}`);
  }
  if (!response.ok) throw taggedImageSourceError(`IMAGE_FETCH_${response.status}`);
  const blob = await response.blob();
  return blobToDataUrl(blob);
}

// Failures here were previously invisible (a bare catch {} on the old persistCache) -- surfaced
// in buildQueueStats so the popup/diagnostics can show a real signal instead of a silently
// diverging count.
let cachePersistFailures = 0;

function ensureCacheLoaded() {
  if (cacheLoaded) return Promise.resolve();
  if (!cacheLoadPromise) cacheLoadPromise = loadCacheFromStorage();
  return cacheLoadPromise;
}

async function loadCacheFromStorage() {
  if (!chrome.storage.local?.get) {
    cacheLoaded = true;
    return;
  }
  try {
    // chrome.storage.local.get(null) returns EVERY key in local storage, including unrelated
    // settings (localPipelineUrl, translationCachePages, ...) -- the CACHE_ENTRY_PREFIX filter
    // below is load-bearing, not cosmetic.
    const all = await chrome.storage.local.get(null);
    const staleKeys = [];
    const liveEntries = [];
    for (const key of Object.keys(all)) {
      if (!key.startsWith(CACHE_ENTRY_PREFIX)) continue;
      const entry = all[key];
      const cacheId = key.slice(CACHE_ENTRY_PREFIX.length);
      // Mirrors putCachedResult's own storage criteria (hasImage OR non-empty translations).
      // The CACHE_VERSION prefix check discards entries from a previous extension version so a
      // version bump can't inflate the count with entries nothing can actually serve
      // (buildCacheId always embeds CACHE_VERSION, so a stale-version key could never be looked
      // up again anyway) -- and since chrome.storage.local is durable (unlike the old session
      // storage, which self-wiped on browser close), those dead entries must be explicitly
      // removed here or they leak forever.
      const hasImage = Boolean(entry?.result?.translatedImageDataUrl);
      const hasTranslations = Array.isArray(entry?.result?.translations) && entry.result.translations.length > 0;
      if ((hasImage || hasTranslations) && cacheId.startsWith(`${CACHE_VERSION}:`)) {
        liveEntries.push([cacheId, entry]);
      } else {
        staleKeys.push(key);
      }
    }
    // Sort ascending by lastUsed so Map insertion order below IS LRU order (trimCache relies on
    // this: it evicts translationCache.keys().next().value as "the oldest").
    liveEntries.sort((a, b) => (a[1].lastUsed || 0) - (b[1].lastUsed || 0));
    liveEntries.forEach(([cacheId, entry]) => translationCache.set(cacheId, entry));
    if (staleKeys.length) chrome.storage.local.remove?.(staleKeys).catch(() => {});

    // Best-effort migration from the old single-key chrome.storage.session store (pre-Fix-B).
    // Session storage is memory-backed, so this only ever matters once per browser session --
    // after a successful migration the legacy key is gone and this branch is a no-op forever.
    const legacy = await chrome.storage.session?.get?.([LEGACY_SESSION_CACHE_KEY]).catch(() => null);
    const legacyEntries = legacy?.[LEGACY_SESSION_CACHE_KEY];
    if (legacyEntries && typeof legacyEntries === 'object') {
      for (const [cacheId, entry] of Object.entries(legacyEntries)) {
        const hasImage = Boolean(entry?.result?.translatedImageDataUrl);
        const hasTranslations = Array.isArray(entry?.result?.translations) && entry.result.translations.length > 0;
        if (!(hasImage || hasTranslations) || !cacheId.startsWith(`${CACHE_VERSION}:`)) continue;
        if (!translationCache.has(cacheId)) translationCache.set(cacheId, entry);
        await writeCacheEntry(cacheId, translationCache.get(cacheId));
      }
      await chrome.storage.session.remove([LEGACY_SESSION_CACHE_KEY]).catch(() => {});
    }

    cacheLoaded = true;
  } catch (error) {
    // Deliberately NOT clearing translationCache and NOT setting cacheLoaded here -- a
    // transient storage failure must be retried on the next access, not permanently treated as
    // "loaded empty" for the rest of this service-worker's lifetime.
    console.warn('[FMT] cache load failed; will retry on next access:', error?.message || error);
    cacheLoadPromise = null;
  }
}

async function writeCacheEntry(cacheId, entry) {
  if (!chrome.storage.local?.set) return;
  try {
    await chrome.storage.local.set({ [CACHE_ENTRY_PREFIX + cacheId]: entry });
  } catch (error) {
    cachePersistFailures += 1;
    console.warn('[FMT] cache entry persist failed', cacheId, error?.message || error);
  }
}

async function removeCacheEntries(cacheIds) {
  if (!cacheIds.length || !chrome.storage.local?.remove) return;
  try {
    await chrome.storage.local.remove(cacheIds.map((id) => CACHE_ENTRY_PREFIX + id));
  } catch (error) {
    cachePersistFailures += 1;
    console.warn('[FMT] cache entry removal failed', error?.message || error);
  }
}

async function trimCache(limit) {
  const safeLimit = Math.max(0, Math.min(MAX_CACHE_LIMIT, Number(limit) || 0));
  const evicted = [];
  if (safeLimit === 0) {
    evicted.push(...translationCache.keys());
    translationCache.clear();
  } else {
    while (translationCache.size > safeLimit) {
      const oldestKey = translationCache.keys().next().value;
      if (!oldestKey) break;
      translationCache.delete(oldestKey);
      evicted.push(oldestKey);
    }
  }
  await removeCacheEntries(evicted);
}

async function clearTranslationCache() {
  const allKeys = Array.from(translationCache.keys());
  translationCache.clear();
  await removeCacheEntries(allKeys);
  if (chrome.storage.session?.remove) {
    try {
      await chrome.storage.session.remove([LEGACY_SESSION_CACHE_KEY]);
    } catch {
      // no-op
    }
  }
}

function buildCacheId(message, settings) {
  const pageIdentity = message.pageCacheKey || message.pageUrl || 'no-page';
  const sourceIdentity = (
    message.cacheKey
    || message.originalImageUrl
    || message.imageUrl
    || message.base64Data
    || ''
  );
  return [
    CACHE_VERSION,
    settings.localPipelineUrl,
    settings.localPipelineLanguage,
    fastHash(String(pageIdentity)),
    fastHash(String(sourceIdentity)),
  ].join(':');
}

// A cache HIT bumping lastUsed used to persist the ENTIRE cache (await persistCache()) on every
// single hit, purely to save one timestamp -- multi-megabyte re-serialization on the hottest
// read path, and (before Fix B) the trigger for the quota-truncation loop running on almost
// every request. lastUsed is only persisted past this threshold, so gallery ordering
// (getRecentTranslations sorts on it) survives a restart with a small bounded drift instead of
// costing a full entry write every time.
const LAST_USED_PERSIST_INTERVAL_MS = 60_000;

async function getCachedResult(cacheId, settings) {
  await ensureCacheLoaded();
  if (settings.cacheLimit <= 0 || !translationCache.has(cacheId)) return null;
  const entry = translationCache.get(cacheId);
  translationCache.delete(cacheId);
  const previousLastUsed = entry.lastUsed || 0;
  entry.lastUsed = Date.now();
  translationCache.set(cacheId, entry);
  if (entry.lastUsed - previousLastUsed > LAST_USED_PERSIST_INTERVAL_MS) {
    writeCacheEntry(cacheId, entry).catch(() => {});
  }
  return entry.result;
}

function pageHostFromUrl(pageUrl) {
  try {
    return new URL(String(pageUrl || '')).hostname || '';
  } catch {
    return '';
  }
}

// The popup's recent-translations gallery previously reused the FULL-SIZE translated PNG as its
// own "thumbnail" field, and polled up to 40 of them every couple seconds -- multi-megabyte data
// URLs decoded and repainted on a timer for a ~100px preview. Generating a small thumbnail ONCE
// here, at cache-write time, means the (potentially large) full image is decoded to pixels exactly
// once per translation, not once per gallery poll.
const THUMBNAIL_MAX_DIMENSION = 112;

async function generateThumbnailDataUrl(fullDataUrl) {
  // A service-worker context always has OffscreenCanvas/createImageBitmap in real Chrome; this
  // guard only matters for older test/sandbox environments -- falling back to the full image there
  // keeps the gallery working (just not shrunk) instead of throwing.
  if (!fullDataUrl || typeof OffscreenCanvas === 'undefined' || typeof createImageBitmap !== 'function') {
    return fullDataUrl || null;
  }
  let bitmap;
  try {
    const blob = await (await fetch(fullDataUrl)).blob();
    bitmap = await createImageBitmap(blob);
    const scale = Math.min(1, THUMBNAIL_MAX_DIMENSION / Math.max(bitmap.width || 1, bitmap.height || 1));
    const width = Math.max(1, Math.round((bitmap.width || THUMBNAIL_MAX_DIMENSION) * scale));
    const height = Math.max(1, Math.round((bitmap.height || THUMBNAIL_MAX_DIMENSION) * scale));
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    if (!ctx) return fullDataUrl;
    ctx.drawImage(bitmap, 0, 0, width, height);
    const thumbBlob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.72 });
    return await blobToDataUrl(thumbBlob);
  } catch {
    // Thumbnail generation failing (corrupt image, canvas unavailable, quota, ...) must never
    // block caching the real translation -- just fall back to the full image for the gallery.
    return fullDataUrl;
  } finally {
    bitmap?.close?.();
  }
}

async function putCachedResult(cacheId, result, settings, pageUrl = '') {
  await ensureCacheLoaded();
  const hasImage = Boolean(result?.translatedImageDataUrl);
  const hasTranslations = Array.isArray(result?.translations) && result.translations.length > 0;
  if (settings.cacheLimit <= 0 || !(hasImage || hasTranslations)) return;
  const thumbnail = hasImage ? await generateThumbnailDataUrl(result.translatedImageDataUrl) : null;
  const entry = {
    result,
    thumbnail,
    lastUsed: Date.now(),
    pageHost: pageHostFromUrl(pageUrl),
  };
  translationCache.set(cacheId, entry);
  await writeCacheEntry(cacheId, entry);
  await trimCache(settings.cacheLimit);
}

async function lookupCachedTranslation(message) {
  const settings = await getSettings();
  const cacheId = buildCacheId(message, settings);
  const cached = await getCachedResult(cacheId, settings);
  if (cached) {
    console.log(`[FMT] cache hit ${cacheId}`);
    return { ...cached, hit: true, fromCache: true, cacheId };
  }
  if (outgoingRequests.has(cacheId)) {
    console.log(`[FMT] cache miss; request still in-flight ${cacheId}`);
    return { hit: false, inFlight: true, cacheId };
  }
  return { hit: false, inFlight: false, cacheId };
}

async function callLocalPipeline(base64Data, width, height, settings, metadata = {}, signal = undefined) {
  const traceId = metadata.traceId || makeTraceId('pipe');
  const response = await fetch(settings.localPipelineUrl, {
    method: 'POST',
    headers: fmtHeaders(settings, { 'Content-Type': 'application/json' }),
    signal,
    body: JSON.stringify({
      imageData: base64Data,
      width: width || 0,
      height: height || 0,
      sourceLanguage: settings.localPipelineLanguage,
      targetLanguage: 'en',
      qualityProfile: 'strict',
      requestedOutput: 'translatedImageDataUrl',
      clientRequestId: traceId,
      metadata,
    }),
  });

  if (!response.ok) {
    let detail = '';
    let errorCode = '';
    try {
      const payload = await response.json();
      detail = responseErrorDetail(payload);
      errorCode = typeof payload?.detail?.code === 'string' ? payload.detail.code : '';
    } catch {
      detail = await response.text().catch(() => '');
    }
    await sendDiagnosticLog('background.pipeline.http_error', {
      traceId,
      status: response.status,
      detail,
      metadata,
    }, settings);
    // The backend tags a saturated-scheduler 503 with a machine-readable code so the caller can
    // tell "try again, a slot will free up" apart from a genuine failure -- normalize it to a
    // stable PIPELINE_BUSY code (same pattern as AbortError -> PIPELINE_TIMEOUT below) so
    // content.js can route it to its patient retry path instead of the human message, which
    // matches no entry in content.js's retryable-error allowlist and was being treated as terminal.
    if (response.status === 503 && errorCode === 'SCHEDULER_BUSY') {
      throw new Error('PIPELINE_BUSY');
    }
    throw new Error(detail || `LOCAL_PIPELINE_${response.status}`);
  }

  const payload = await response.json();
  const pipelineTranslationCount = Number(
    payload.report?.translations
    ?? payload.report?.rawTranslations
    ?? payload.report?.renderedRegions
    ?? 0
  ) || 0;
  await sendDiagnosticLog('background.pipeline.response', {
    traceId,
    status: response.status,
    hasImage: Boolean(payload.translatedImageDataUrl || payload.imageDataUrl),
    translationCount: Array.isArray(payload.translations) && payload.translations.length
      ? payload.translations.length
      : pipelineTranslationCount,
    renderedRegions: Number(payload.report?.renderedRegions ?? 0) || 0,
    reportStatus: payload.report?.outputSafety || payload.report?.status || null,
    sample: payload.report?.sample || payload.report?.sampleName || null,
  }, settings);
  if (payload.translatedImageDataUrl || payload.imageDataUrl) {
    return {
      translatedImageDataUrl: payload.translatedImageDataUrl || payload.imageDataUrl,
      translations: payload.translations || [],
      pipelineReport: payload.report || null,
    };
  }
  if (Array.isArray(payload.translations)) {
    return { translations: payload.translations };
  }
  throw new Error('LOCAL_PIPELINE_EMPTY_RESPONSE');
}

async function processTranslation(message, options = {}) {
  if (isPaused) return { error: 'TranslationPaused' };

  const width = message.width || 0;
  const height = message.height || 0;
  const settings = await getSettings();
  const cacheId = buildCacheId(message, settings);
  const traceId = message.traceId || makeTraceId('live');

  const cached = await getCachedResult(cacheId, settings);
  if (cached) {
    console.log(`[FMT] translateImage served from cache ${cacheId}`);
    sendDiagnosticLog('background.translate.cache_hit', { traceId, cacheId }, settings).catch(() => {});
    return { ...cached, fromCache: true };
  }

  if (outgoingRequests.has(cacheId)) {
    console.log(`[FMT] joining in-flight translation ${cacheId}`);
    sendDiagnosticLog('background.translate.join_in_flight', { traceId, cacheId }, settings).catch(() => {});
    try {
      const result = await outgoingRequests.get(cacheId);
      return { ...result, fromInFlight: true };
    } catch (error) {
      return { error: error.message };
    }
  }
  if (!options.bypassQueueCheck && outgoingRequests.size >= effectiveParallelLimit(settings)) {
    sendDiagnosticLog('background.translate.deferred_to_queue', {
      traceId,
      cacheId,
      activeRequests: outgoingRequests.size,
      parallelLimit: settings.parallelLimit,
      effectiveParallelLimit: effectiveParallelLimit(settings),
    }, settings).catch(() => {});
    return queueTranslation(message);
  }

  // Placed after the parallel-limit admission check (so queued items still queue exactly as
  // today) and before any AbortController/timeout/outgoingRequests bookkeeping below, so a
  // short-circuit here never reserves a slot and never starts a timer. It DOES still need to
  // recheck the keep-alive alarm: a message reaching here via the queue-drain path may have had
  // ensureKeepAliveAlarm() called back when it was originally enqueued, and this return skips
  // processTranslation's own finally block (the alarm's normal recheck point) entirely -- without
  // this call, a burst that drains purely through short-circuits would leave the alarm dangling
  // even though outgoingRequests/requestQueue both correctly settle back to empty.
  if (!shouldAttemptDispatch()) {
    sendDiagnosticLog('background.translate.breaker_open', { traceId, cacheId }, settings).catch(() => {});
    maybeClearKeepAliveAlarm();
    return { error: 'PIPELINE_OFFLINE' };
  }

  const controller = new AbortController();
  activeControllers.set(cacheId, controller);
  options.onDispatched?.();

  // Without a timeout, a hung backend request holds this slot for the rest of the browser
  // session -- with only 1-3 parallel slots total, one stuck request can quietly stall the
  // whole queue behind it. timedOut is separate from a pause-abort so the catch block below
  // can tell the two apart and report a distinct, retryable error instead of reusing
  // 'TranslationPaused' for a failure that has nothing to do with the user pausing.
  let timedOut = false;
  const timeoutTimer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, settings.fetchTimeoutMs);

  const promise = (async () => {
    const startedAt = nowMs();
    try {
      if (isPaused) throw new Error('TranslationPaused');
      const base64Data = message.base64Data || await fetchImageAsDataUrl(message.imageUrl);
      if (isPaused) throw new Error('TranslationPaused');
      console.log(`[FMT] local pipeline start trace=${traceId} ${cacheId} ${width}x${height}`);
      await sendDiagnosticLog('background.translate.start', {
        traceId,
        cacheId,
        width,
        height,
        source: message.kind === 'snapshot' ? 'selection-crop' : (message.imageUrl ? 'image-url' : 'canvas'),
        pageUrl: message.pageUrl || '',
        originalImageUrl: message.originalImageUrl || '',
        language: settings.localPipelineLanguage,
      }, settings);
      const result = await callLocalPipeline(base64Data, width, height, settings, {
        traceId,
        extensionVersion: APP_VERSION,
        source: message.kind === 'snapshot'
          ? 'extension-selection-crop'
          : (message.imageUrl ? 'extension-image-url' : 'extension-canvas'),
        pageUrl: message.pageUrl || '',
        pageCacheKey: message.pageCacheKey || '',
        cacheKey: message.cacheKey || '',
        originalImageUrl: message.originalImageUrl || '',
        cacheId,
      }, controller.signal);
      resetBreaker(); // a real attempt reached the server and got a real answer -- clearly back up
      noteBackendCapacity(result.pipelineReport?.scheduler?.capacityAtAcquire);
      if (isPaused) throw new Error('TranslationPaused');
      await putCachedResult(cacheId, result, settings, message.pageUrl);
      console.log(`[FMT] local pipeline done trace=${traceId} ${cacheId} in ${Math.round(nowMs() - startedAt)}ms`);
      const servingProviders = result.pipelineReport?.translationServingProviders;
      const sourceCounts = result.pipelineReport?.translationSourceCounts;
      if ((servingProviders && servingProviders.length) || (sourceCounts && Object.keys(sourceCounts).length)) {
        console.log(`[FMT] translation served by [${(servingProviders || []).join(', ')}] sources=${JSON.stringify(sourceCounts || {})}`);
      }
      await sendDiagnosticLog('background.translate.done', {
        traceId,
        cacheId,
        elapsedMs: Math.round(nowMs() - startedAt),
        hasImage: Boolean(result?.translatedImageDataUrl),
        translationCount: Array.isArray(result?.translations) && result.translations.length
          ? result.translations.length
          : (Number(result?.pipelineReport?.translations ?? result?.pipelineReport?.rawTranslations ?? 0) || 0),
        renderedRegions: Number(result?.pipelineReport?.renderedRegions ?? 0) || 0,
        outputSafety: result?.pipelineReport?.outputSafety || null,
      }, settings);
      return result;
    } catch (error) {
      if (error?.isImageSourceError) {
        // The IMAGE URL itself is broken/unreachable, not the backend -- leave breaker state
        // completely untouched (neither trip nor reset) so a single dead image src on a page
        // can't falsely short-circuit every OTHER image, nor mask a genuinely open breaker.
      } else if (error?.name === 'TypeError') {
        // fetch() itself could not reach the server -- connection refused / offline / DNS fail.
        tripBreaker();
      } else if (error?.name !== 'AbortError') {
        // Any other exception (HTTP error status, malformed response) proves the server IS
        // reachable -- a stale "open" breaker must not keep short-circuiting on fresh evidence.
        resetBreaker();
      }
      // AbortError (our own fetchTimeoutMs timeout) intentionally leaves breaker state untouched
      // -- one slow request is not proof the whole backend is down.
      const messageText = error?.isImageSourceError
        ? error.message
        : (error?.name === 'TypeError'
          // Normalize so the request that TRIPS the breaker reports the same code as every
          // subsequent short-circuited request -- otherwise the first offline image would show a
          // raw "Failed to fetch" while every later one shows PIPELINE_OFFLINE.
          ? 'PIPELINE_OFFLINE'
          : (error.name === 'AbortError' ? (timedOut ? 'PIPELINE_TIMEOUT' : 'TranslationPaused') : error.message));
      if (messageText !== 'TranslationPaused') console.error('[FMT] Local pipeline error:', messageText);
      await sendDiagnosticLog('background.translate.error', {
        traceId,
        cacheId,
        error: messageText,
        elapsedMs: Math.round(nowMs() - startedAt),
      }, settings);
      return { error: messageText };
    } finally {
      clearTimeout(timeoutTimer);
      outgoingRequests.delete(cacheId);
      outgoingRequestMeta.delete(cacheId);
      activeControllers.delete(cacheId);
      maybeClearKeepAliveAlarm();
      scheduleQueueDrain();
    }
  })();

  outgoingRequests.set(cacheId, promise);
  outgoingRequestMeta.set(cacheId, {
    pageUrl: message.pageUrl || '',
    originalImageUrl: message.originalImageUrl || '',
    startedAt: nowMs(),
  });
  ensureKeepAliveAlarm();
  return promise;
}

// processQueue() is the only thing that promotes a queued item. Every path that frees up
// admission room (a real dispatch finishing, an early-return in processTranslation settling
// its reservation, a fresh item being queued while slots are free) must be able to trigger a
// drain, or queued items past that point can stall forever with free slots sitting idle. This
// coalesces every trigger into at most one processQueue() per tick: a flag prevents piling up
// redundant passes, and the setTimeout(0) defers past the current synchronous release chain so
// a drain never re-enters processQueue() while another one is still walking the queue.
let drainScheduled = false;
function scheduleQueueDrain() {
  if (drainScheduled) return;
  drainScheduled = true;
  setTimeout(() => {
    drainScheduled = false;
    processQueue();
  }, 0);
}

// The single entry point admission checks must dispatch through: reserves a slot
// synchronously (before processTranslation's first await), and releases it either the
// moment processTranslation actually registers in outgoingRequests (onDispatched -- the
// reservation has become a real entry, so counting both would double-count) or, as a
// safety net, whenever the returned promise settles (covers every early-return path in
// processTranslation -- paused/cached/in-flight-dedup -- where no slot was ever consumed).
// Every one of those early-return settles here too, which is why release() must itself
// trigger a drain: a queued item dispatched into a cache-hit or in-flight-join releases its
// slot without ever entering processTranslation's own finally, and without this the rest of
// the queue would stall behind it even though the slot is free again.
function dispatchTranslation(message, cacheId) {
  reservedSlots += 1;
  let released = false;
  const release = () => {
    if (released) return;
    released = true;
    reservedSlots = Math.max(0, reservedSlots - 1);
    scheduleQueueDrain();
  };
  const result = processTranslation(message, { bypassQueueCheck: true, onDispatched: release });
  result.then(release, release);
  return result.then((value) => requeueIfBusy(message, value, cacheId));
}

// A PIPELINE_BUSY response means the backend's own gpu_scheduler.py already waited out its
// full max_wait_seconds (60s by default) for a GPU slot and gave up -- the request lost a
// race that had nothing to do with it. Retrying it HERE (background.js), rather than leaving
// it to content.js's per-image retry, matters for one concrete reason: putCachedResult() is
// awaited (see processTranslation) before this promise ever resolves, so a retry that
// eventually succeeds is durably cached even if the page has since navigated away and would
// have discarded the result for display. content.js's own retry is node-anchored and silently
// no-ops once a page's <img> node has moved on (single-<img> viewers reuse one DOM node for
// every page) -- that dead-end is exactly what left pages permanently untranslated.
function requeueIfBusy(message, value, cacheId) {
  if (value?.error !== 'PIPELINE_BUSY') return value;
  const attempt = (message.busyAttempts || 0) + 1;
  if (attempt > MAX_BUSY_REQUEUE_ATTEMPTS) return value;
  message.busyAttempts = attempt;
  // During the backoff below this item holds no reservedSlots and is not sitting in
  // requestQueue, so maybeClearKeepAliveAlarm() would see both empty and let Chrome idle-kill
  // the service worker mid-backoff -- silently evaporating the very retry this exists for, on
  // exactly the slow/contended backend where it's needed most.
  ensureKeepAliveAlarm();
  console.warn(`[FMT] busy requeue attempt ${attempt}/${MAX_BUSY_REQUEUE_ATTEMPTS} ${cacheId || ''}`);
  sendDiagnosticLog('background.translate.busy_requeue', { cacheId, attempt }).catch(() => {});
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(queueTranslation(message, { front: true, bypassQueueLimit: true }));
    }, BUSY_REQUEUE_BASE_DELAY_MS * attempt);
  });
}

function processQueue() {
  if (isPaused) return;
  getSettings().then((settings) => {
    let shifted = false;
    while (requestQueue.length > 0 && (outgoingRequests.size + reservedSlots) < effectiveParallelLimit(settings)) {
      const { message, resolve, cacheId } = requestQueue.shift();
      if (cacheId) queuedRequests.delete(cacheId);
      // processTranslation() normally resolves with an {error} object rather than rejecting,
      // but it CAN reject before its own try block (e.g. the awaited getSettings()/
      // getCachedResult() calls, or the queueTranslation re-entry). Without this .catch, that
      // rejection left `resolve` never called -- the promise queueTranslation handed back to
      // the content script's sendResponse would never settle, keeping that image's message
      // port (and its spinner) open forever.
      dispatchTranslation(message, cacheId).then(resolve, (error) => resolve({ error: error?.message || 'DISPATCH_FAILED' }));
      shifted = true;
    }
    if (shifted) persistQueueDescriptors();
  }).catch(() => {});
}

function clearQueuedTranslations(reason = 'QueueCleared') {
  const dropped = requestQueue.splice(0);
  dropped.forEach(({ resolve, cacheId }) => {
    if (cacheId) queuedRequests.delete(cacheId);
    resolve?.({ error: reason, queueLength: 0 });
  });
  if (dropped.length > 0) persistQueueDescriptors();
  restartLost = 0;
  queueActivitySinceBoot = true;
  maybeClearKeepAliveAlarm();
  return dropped.length;
}

async function queueTranslation(message, options = {}) {
  if (isPaused) return { error: 'TranslationPaused' };
  // The popup's restart-lost warning explicitly tells the user to "re-run Translate Page" --
  // this is that re-run actually happening, so the stale warning must clear here too, not only
  // on an explicit Clear Queue. Otherwise the message keeps showing a truthful-when-written but
  // now-stale claim even after the user did exactly what it asked, which is its own honesty bug.
  restartLost = 0;
  queueActivitySinceBoot = true;

  const settings = await getSettings();
  const cacheId = buildCacheId(message, settings);
  const cached = await getCachedResult(cacheId, settings);
  if (cached) {
    console.log(`[FMT] queued request served from cache ${cacheId}`);
    return { ...cached, fromCache: true };
  }

  if (outgoingRequests.has(cacheId)) {
    console.log(`[FMT] queued request joining active translation ${cacheId}`);
    try {
      const result = await outgoingRequests.get(cacheId);
      return { ...result, fromInFlight: true };
    } catch (error) {
      return { error: error.message };
    }
  }

  if (queuedRequests.has(cacheId)) {
    console.log(`[FMT] queued request joining queued translation ${cacheId}`);
    return queuedRequests.get(cacheId);
  }

  if ((outgoingRequests.size + reservedSlots) < effectiveParallelLimit(settings)) {
    return dispatchTranslation(message, cacheId);
  }

  // bypassQueueLimit is set only by requeueIfBusy(): this item already consumed one full
  // admission cycle (it dispatched, ran, and lost a real backend race) -- it is not new work
  // competing for a queue slot, so a queue that happens to be full at retry time must not
  // convert it into a QueueFull error and hand it back to content.js's dead-end retry.
  if (!options.bypassQueueLimit && requestQueue.length >= settings.queueLimit) {
    console.warn(`[FMT] queue full ${requestQueue.length}/${settings.queueLimit}`);
    return {
      error: 'QueueFull',
      queueLength: requestQueue.length,
      queueLimit: settings.queueLimit,
    };
  }

  const queuedPromise = new Promise((resolve) => {
    // front is set only by requeueIfBusy(): this item already waited out the backend's full
    // gpu_scheduler.py wait window once through no fault of its own -- placing it at the tail
    // of a deep queue (at capacity 1 with a 20-deep queue, ~26 minutes) penalizes it twice.
    if (options.front) requestQueue.unshift({ message, resolve, cacheId });
    else requestQueue.push({ message, resolve, cacheId });
  });
  queuedRequests.set(cacheId, queuedPromise);
  queuedPromise.finally(() => queuedRequests.delete(cacheId));
  ensureKeepAliveAlarm();
  persistQueueDescriptors();
  console.log(`[FMT] queued translation ${requestQueue.length}/${settings.queueLimit} ${cacheId}`);
  // A slot may have freed up during the awaits above (getSettings/getCachedResult/
  // in-flight join) between this item being deemed queue-worthy and actually landing in
  // requestQueue -- without this, that freed slot has nothing to wake it back up until
  // some unrelated request happens to settle later.
  scheduleQueueDrain();
  return queuedPromise;
}

async function setTranslationPaused(paused) {
  pauseStateSetSinceBoot = true;
  isPaused = paused === true;
  await chrome.storage.local.set({ translationPaused: isPaused });

  if (isPaused) {
    clearQueuedTranslations('TranslationPaused');
    for (const controller of activeControllers.values()) controller.abort();
  } else {
    processQueue();
  }

  return { success: true, ...buildQueueStats(await getSettings()) };
}

async function stopTranslationRuntime(mode, tabId) {
  pauseStateSetSinceBoot = true;
  isPaused = true;
  await chrome.storage.local.set({ translationPaused: true });
  const dropped = clearQueuedTranslations(mode === 'hard' ? 'HardStopped' : 'SoftStopped');
  for (const controller of activeControllers.values()) controller.abort();
  if (tabId) {
    try {
      await sendContentMessage(tabId, { kind: 'setTranslationPaused', paused: true, mode });
    } catch {
    }
  }
  const settings = await getSettings();
  const backend = await stopPipelineRuntime(settings, mode);
  return {
    success: backend.success !== false,
    mode,
    dropped,
    backend,
    ...buildQueueStats(settings),
  };
}

async function ensureContentScript(tabId) {
  if (!tabId) throw new Error('TAB_ID_MISSING');
  try {
    await chrome.tabs.sendMessage(tabId, { kind: 'pingContentScript' });
    return { injected: false };
  } catch {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ['content.js'],
    });
    await chrome.tabs.sendMessage(tabId, { kind: 'pingContentScript' });
    return { injected: true };
  }
}

async function sendContentMessage(tabId, message) {
  await ensureContentScript(tabId);
  return chrome.tabs.sendMessage(tabId, message);
}

async function activatePageTranslation(tabId, options = {}) {
  const persistAuto = options.persistAuto === true;
  await setTranslationPaused(false);
  const storageUpdate = { translationPaused: false };
  if (persistAuto) storageUpdate.translationEnabled = true;
  await chrome.storage.local.set(storageUpdate);
  await ensureContentScript(tabId);
  await chrome.tabs.sendMessage(tabId, { kind: 'setTranslationPaused', paused: false });
  if (persistAuto) {
    await chrome.tabs.sendMessage(tabId, { kind: 'toggleTranslation', enabled: true });
  } else {
    await chrome.tabs.sendMessage(tabId, { kind: 'translatePageOnce' });
  }
  return { success: true, autoEnabled: persistAuto };
}

async function pausePageTranslation(tabId) {
  const state = await setTranslationPaused(true);
  if (tabId) {
    try {
      await sendContentMessage(tabId, { kind: 'setTranslationPaused', paused: true });
    } catch {
      // A page without a content script still has background queue state paused.
    }
  }
  return state;
}

async function cropVisibleTab(tabId, dimensions) {
  if (!dimensions) throw new Error('SNAPSHOT_DIMENSIONS_MISSING');
  const dataUrl = await new Promise((resolve) => {
    chrome.tabs.captureVisibleTab(null, { format: 'png' }, resolve);
  });
  if (!dataUrl) throw new Error('CAPTURE_FAILED');

  const zoomFactor = await chrome.tabs.getZoom(tabId).catch(() => 1);
  const devicePixelRatio = dimensions.devicePixelRatio || 1;
  const scale = zoomFactor * devicePixelRatio;
  const cropLeft = Math.max(0, Math.round((dimensions.left || 0) * scale));
  const cropTop = Math.max(0, Math.round((dimensions.top || 0) * scale));
  const cropWidth = Math.max(1, Math.round((dimensions.width || 0) * scale));
  const cropHeight = Math.max(1, Math.round((dimensions.height || 0) * scale));

  if (!Number.isFinite(cropWidth) || !Number.isFinite(cropHeight) || cropWidth < 1 || cropHeight < 1) {
    throw new Error('SNAPSHOT_DIMENSIONS_INVALID');
  }

  const screenshotBlob = await (await fetch(dataUrl)).blob();
  const croppedBitmap = await createImageBitmap(screenshotBlob, cropLeft, cropTop, cropWidth, cropHeight);
  const canvas = new OffscreenCanvas(cropWidth, cropHeight);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('SNAPSHOT_CANVAS_UNAVAILABLE');
  ctx.drawImage(croppedBitmap, 0, 0);
  croppedBitmap.close();
  const croppedBlob = await canvas.convertToBlob({ type: 'image/png' });
  return {
    dataUrl: await blobToDataUrl(croppedBlob),
    width: cropWidth,
    height: cropHeight,
    zoomFactor,
    devicePixelRatio,
  };
}

// Selection-area (snapshot) translate used to call callLocalPipeline directly: no cache
// read/write, no queue/parallel accounting, and no AbortController, so pause/stop couldn't
// cancel it and repeating an identical selection always recomputed. The screen capture
// itself still happens immediately here (a queued capture would risk shooting the wrong
// content if the page scrolled/navigated in the meantime), but the pipeline call is routed
// through the same queueTranslation used by every other translation kind, so it shares
// caching, dedupe, the parallel budget, and cancellation. The cache identity is the
// captured pixels themselves (buildCacheId's base64Data fallback), not just the selection's
// geometry, so a re-selected rect whose underlying content actually changed still misses.
async function translateSnapshot(tabId, pageUrl, dimensions) {
  if (isPaused) return { error: 'TranslationPaused' };
  let cropped;
  try {
    cropped = await cropVisibleTab(tabId, dimensions);
  } catch (error) {
    return { error: error.message };
  }
  const result = await queueTranslation({
    kind: 'snapshot',
    base64Data: cropped.dataUrl,
    width: cropped.width,
    height: cropped.height,
    pageUrl: pageUrl || '',
    pageCacheKey: pageUrl || '',
  });
  return {
    ...result,
    zoomFactor: cropped.zoomFactor,
    devicePixelRatio: cropped.devicePixelRatio,
    imageWidth: cropped.width,
    imageHeight: cropped.height,
  };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.kind === 'translateImage') {
    queueTranslation(message).then(sendResponse);
    return true;
  }

  if (message.kind === 'diagnosticLog') {
    sendDiagnosticLog(message.event || 'content.diagnostic', {
      ...(message.details || {}),
      traceId: message.traceId,
      pageUrl: message.pageUrl,
    }).then(sendResponse).catch((error) => sendResponse({ ok: false, error: error.message }));
    return true;
  }
  if (message.kind === 'lookupCachedTranslation') {
    lookupCachedTranslation(message)
      .then(sendResponse)
      .catch((error) => sendResponse({ hit: false, inFlight: false, error: error.message }));
    return true;
  }
  if (message.kind === 'translateSnapshot') {
    translateSnapshot(sender.tab?.id || message.tabId, message.pageUrl, message.dimensions).then(sendResponse);
    return true;
  }
  if (message.kind === 'getTranslationStats') {
    ensureCacheLoaded().then(async () => {
      const settings = await getSettings();
      sendResponse(buildQueueStats(settings));
    }).catch((error) => {
      // Without this, a storage failure here left the popup's stats poll waiting on a
      // message port that would never receive a response -- the Active/Queued/Cache
      // numbers silently froze at their last values with no indication anything failed.
      sendResponse({ error: error?.message || String(error) });
    });
    return true;
  }
  if (message.kind === 'getRecentTranslations') {
    ensureCacheLoaded().then(() => {
      // Upper bound matches the popup's own max selectable cache size (see
      // popup.html's translationCachePages options, up to 40) -- the previous hardcoded
      // cap of 12 silently truncated the gallery even when the popup correctly asked for
      // more (e.g. passing the real configured cacheLimit of 15-40), so a user who chose
      // to keep 20+ images cached still only ever saw 12 of them.
      const limit = Math.max(1, Math.min(40, Number(message.limit) || 6));
      const entries = Array.from(translationCache.values())
        .filter((entry) => entry?.result?.translatedImageDataUrl)
        .sort((a, b) => (b.lastUsed || 0) - (a.lastUsed || 0))
        .slice(0, limit)
        .map((entry) => ({
          // entry.thumbnail is the small (~112px) preview generated once at cache-write time;
          // entries rehydrated from a pre-thumbnail session or an environment where generation
          // failed fall back to the full-size image rather than showing nothing.
          thumbnail: entry.thumbnail || entry.result.translatedImageDataUrl,
          pageHost: entry.pageHost || '',
          lastUsed: entry.lastUsed || 0,
        }));
      sendResponse({ entries });
    }).catch((error) => {
      sendResponse({ error: error?.message || String(error), entries: [] });
    });
    return true;
  }
  if (message.kind === 'checkPipelineHealth') {
    getSettings().then(async (settings) => {
      // A transiently unreachable backend does not mean cached results are
      // stale -- clearing here silently wiped the cache on every popup open
      // that happened to race a slow health check. Only the explicit Clear
      // Cache action (and a genuine pipeline URL/language change) clears it.
      const ok = await checkPipelineHealth(settings, { clearCacheOnFailure: false });
      sendResponse({ ok, cacheSize: translationCache.size });
    }).catch((error) => {
      sendResponse({ ok: false, error: error?.message || String(error), cacheSize: translationCache.size });
    });
    return true;
  }
  if (message.kind === 'startEngine') {
    startPipelineEngine()
      .then(sendResponse)
      .catch((error) => sendResponse({
        ok: false,
        available: false,
        error: error.message,
        message: 'Local backend could not be started from the extension.',
        manualStartSteps: manualStartStepsForEngine(),
      }));
    return true;
  }
  if (message.kind === 'getQuotaStatus') {
    getSettings()
      .then(getPipelineQuotaStatus)
      .then(sendResponse)
      .catch((error) => sendResponse({ ok: false, error: error.message, providers: [] }));
    return true;
  }
  if (message.kind === 'getVramStatus') {
    getSettings()
      .then(getPipelineVramStatus)
      .then(sendResponse)
      .catch((error) => sendResponse({ ok: false, available: false, error: error.message, gpus: [] }));
    return true;
  }
  if (message.kind === 'releaseGpu') {
    getSettings()
      .then(releasePipelineGpu)
      .then(sendResponse)
      .catch((error) => sendResponse({ success: false, error: error.message }));
    return true;
  }
  if (message.kind === 'openLogWindow') {
    getSettings()
      .then((settings) => openPipelineLogWindow(settings, message.logType || 'quota'))
      .then(sendResponse)
      .catch((error) => sendResponse({ success: false, error: error.message, logType: message.logType || 'quota' }));
    return true;
  }
  if (message.kind === 'clearCache') {
    getSettings()
      .then(async (settings) => {
        await clearTranslationCache();
        const backend = await clearPipelineRuntimeCache(settings);
        sendResponse({
          success: true,
          cacheSize: 0,
          backendCleared: backend.success !== false,
          backend,
        });
      })
      .catch((error) => sendResponse({ success: false, error: error.message, cacheSize: 0 }));
    return true;
  }
  if (message.kind === 'clearQueue') {
    getSettings()
      .then((settings) => {
        const dropped = clearQueuedTranslations('QueueCleared');
        sendResponse({ success: true, dropped, ...buildQueueStats(settings) });
      })
      .catch((error) => sendResponse({ success: false, error: error.message }));
    return true;
  }
  if (message.kind === 'setCacheLimit') {
    getSettings()
      .then((settings) => trimCache(message.limit ?? settings.cacheLimit))
      .then(() => sendResponse({ success: true, cacheSize: translationCache.size }))
      .catch((error) => sendResponse({ success: false, error: error.message }));
    return true;
  }
  if (message.kind === 'setQueueLimit') {
    getSettings()
      .then(async (settings) => {
        const limit = Math.max(0, Math.min(MAX_QUEUE_LIMIT, Number(message.limit ?? settings.queueLimit) || 0));
        await chrome.storage.local.set({ translationQueuePages: limit });
        let shed = false;
        while (requestQueue.length > limit) {
          const dropped = requestQueue.pop();
          if (dropped?.cacheId) queuedRequests.delete(dropped.cacheId);
          dropped?.resolve?.({ error: 'QueueFull', queueLength: requestQueue.length, queueLimit: limit });
          shed = true;
        }
        // Every other requestQueue mutation (push/shift/splice) persists its descriptors --
        // this shed loop must too, or a later SW restart reports leftover pre-shed cacheIds as
        // "lost" even though the user's own dropdown change is what removed them, not a crash.
        // This is a capacity-limit rejection, not a queue-clearing acknowledgement, so it must
        // NOT touch restartLost.
        if (shed) persistQueueDescriptors();
        maybeClearKeepAliveAlarm();
        sendResponse({ success: true, ...buildQueueStats({ ...settings, queueLimit: limit }) });
      })
      .catch((error) => sendResponse({ success: false, error: error.message }));
    return true;
  }
  if (message.kind === 'setParallelLimit') {
    getSettings()
      .then(async (settings) => {
        const limit = Math.max(1, Math.min(MAX_PARALLEL_LIMIT, Number(message.limit ?? settings.parallelLimit) || DEFAULT_PARALLEL_LIMIT));
        await chrome.storage.local.set({ translationParallelPages: limit });
        processQueue();
        sendResponse({ success: true, ...buildQueueStats({ ...settings, parallelLimit: limit }) });
      })
      .catch((error) => sendResponse({ success: false, error: error.message }));
    return true;
  }
  if (message.kind === 'setTranslationPaused') {
    setTranslationPaused(message.paused).then(sendResponse);
    return true;
  }
  if (message.kind === 'stopTranslations') {
    stopTranslationRuntime(message.mode === 'hard' ? 'hard' : 'soft', message.tabId || sender?.tab?.id)
      .then(sendResponse)
      .catch((error) => sendResponse({ success: false, error: error.message, mode: message.mode || 'soft' }));
    return true;
  }
  if (message.kind === 'activatePageTranslation') {
    activatePageTranslation(message.tabId, { persistAuto: message.persistAuto === true })
      .then(sendResponse)
      .catch((error) => sendResponse({ success: false, error: error.message }));
    return true;
  }
  if (message.kind === 'pausePageTranslation') {
    pausePageTranslation(message.tabId)
      .then(sendResponse)
      .catch((error) => sendResponse({ success: false, error: error.message }));
    return true;
  }
  if (message.kind === 'sendContentCommand') {
    sendContentMessage(message.tabId, message.command)
      .then((response) => sendResponse({ success: true, response }))
      .catch((error) => sendResponse({ success: false, error: error.message }));
    return true;
  }
  if (message.kind === 'startTranslationPanel') {
    ensureContentScript(message.tabId)
      .then(() => chrome.scripting.executeScript({ target: { tabId: message.tabId }, files: ['translationPanel.js'] }))
      .then(() => sendResponse({ success: true }))
      .catch((error) => sendResponse({ success: false, error: error.message }));
    return true;
  }
  if (message.kind === 'testProviderKey') {
    sendResponse({ success: false, error: 'Cloud providers are disabled; use the local pipeline server.' });
    return true;
  }
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'translateMangaImage',
    title: 'Translate this manga panel locally',
    contexts: ['image'],
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'translateMangaImage' && tab?.id) {
    sendContentMessage(tab.id, { kind: 'translateSpecificImage', imageUrl: info.srcUrl });
  }
});

function updateIcon() {
  chrome.action.setIcon({
    path: {
      '16': 'icons/16x16.png',
      '48': 'icons/48x48.png',
      '128': 'icons/128x128.png',
    },
  });
}

chrome.storage.onChanged.addListener((changes) => {
  if (changes.localPipelineUrl) {
    clearTranslationCache();
    updateIcon();
  }
  if (changes.localPipelineLanguage) clearTranslationCache();
  if (changes.translationPaused) {
    pauseStateSetSinceBoot = true;
    isPaused = changes.translationPaused.newValue === true;
  }
});

updateIcon();

