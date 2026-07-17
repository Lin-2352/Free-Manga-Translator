// Exhaustive queue-ahead matrix + edge-set + seeded fuzz for background.js.
//
// Scope: the user's complaint was "queue 5 pages, only 2 translate" -- queue_cache_stress_test's
// scenario 11 already proves the drain fix for that ONE depth. This suite proves it for every
// queue-ahead value the popup dropdown actually offers (0/3/5/10/20/35/50), every parallel-jobs
// setting (1/2/3), both burst and paced arrival, PLUS the cache-hit re-dispatch pattern that was
// the actual root cause, PLUS the two defects (A: setQueueLimit shed not persisting descriptors,
// B: restartLost wake-by-message race) found and fixed alongside this suite.
//
// Composition over combinatorics: the full exact-fit matrix (42 cells) proves every queue-ahead
// value drains cleanly on its own -- that is the user's actual ask. Overflow/QueueFull behavior,
// mid-drain limit changes, and SW-restart are then sampled on a representative subset (not all
// 42 x N) since those mechanisms don't interact with queue depth in a way that varies per-cell;
// re-proving them at every depth would burn suite runtime for no added signal.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const backgroundPath = path.join(extensionRoot, 'background.js');
const source = fs.readFileSync(backgroundPath, 'utf8');

const QUEUE_AHEAD_VALUES = [0, 3, 5, 10, 20, 35, 50]; // exact popup dropdown options
const PARALLEL_VALUES = [1, 2, 3];
const DEFAULT_CACHE = 40; // >= max queue-ahead so LRU eviction never confounds the main matrix
const DEFAULT_FETCH_TIMEOUT = 360000;
const QUEUE_DESCRIPTOR_KEY = 'fmtQueueDescriptorsV1';

let messageListener;
let queueLimitSetting = 10;
let parallelLimitSetting = 2;
let cacheLimitSetting = DEFAULT_CACHE;
let fetchTimeoutSetting = DEFAULT_FETCH_TIMEOUT;
let sessionStore = {};
const heldResolvers = new Map();

function respondFor() {
  return {
    translatedImageDataUrl: 'data:image/png;base64,ZmFrZQ==',
    report: { pipeline: 'local-8-stage', stageSequence: [] },
    translations: [],
  };
}

async function mockFetch(url, options = {}) {
  const urlStr = String(url);
  if (urlStr.includes('/v1/diagnostics/log')) return { ok: true, json: async () => ({ ok: true }) };
  if (urlStr.includes('/v1/cache/clear')) return { ok: true, json: async () => ({ success: true, status: 'cleared', clearedSamples: 0 }) };
  if (urlStr.includes('/v1/runtime/soft-stop')) return { ok: true, json: async () => ({ success: true, status: 'stopped', releaseGpu: false }) };
  if (urlStr.includes('/v1/runtime/hard-stop')) return { ok: true, json: async () => ({ success: true, status: 'stopped', releaseGpu: true }) };
  if (urlStr.includes('/v1/health')) return { ok: true, json: async () => ({ ok: true }) };
  if (urlStr.includes('/v1/warmup')) return { ok: true, json: async () => ({ ok: true, status: 'warming' }) };

  const body = options.body ? JSON.parse(options.body) : null;
  const cacheKey = body?.metadata?.cacheKey || '';
  if (cacheKey.includes('hold')) {
    return new Promise((resolve, reject) => {
      const settle = () => heldResolvers.delete(cacheKey);
      heldResolvers.set(cacheKey, () => {
        settle();
        resolve({ ok: true, json: async () => respondFor() });
      });
      options.signal?.addEventListener('abort', () => {
        settle();
        const err = new Error('Aborted');
        err.name = 'AbortError';
        reject(err);
      });
    });
  }
  return { ok: true, json: async () => respondFor() };
}

const sandbox = {
  console,
  setTimeout,
  clearTimeout,
  AbortController,
  chrome: {
    storage: {
      local: {
        get: async () => ({
          localPipelineUrl: 'http://127.0.0.1:8766/v1/translate-image',
          localPipelineLanguage: 'ja',
          translationCachePages: cacheLimitSetting,
          translationQueuePages: queueLimitSetting,
          translationParallelPages: parallelLimitSetting,
          translationFetchTimeoutMs: fetchTimeoutSetting,
        }),
        // Real chrome.storage.local.set persists, and background.js's own setQueueLimit/
        // setParallelLimit handlers call it before re-reading settings via getSettings() ->
        // storage.local.get(). A no-op mock here would make those re-reads see stale values,
        // masking whether a live limit change actually takes effect for admission decisions.
        set: async (entries) => {
          if ('translationQueuePages' in entries) queueLimitSetting = entries.translationQueuePages;
          if ('translationParallelPages' in entries) parallelLimitSetting = entries.translationParallelPages;
          if ('translationCachePages' in entries) cacheLimitSetting = entries.translationCachePages;
          if ('translationFetchTimeoutMs' in entries) fetchTimeoutSetting = entries.translationFetchTimeoutMs;
        },
      },
      session: {
        get: async (keys) => {
          const key = Array.isArray(keys) ? keys[0] : keys;
          return Object.prototype.hasOwnProperty.call(sessionStore, key) ? { [key]: sessionStore[key] } : {};
        },
        set: async (entries) => { Object.assign(sessionStore, entries); },
        remove: async (keys) => {
          for (const key of (Array.isArray(keys) ? keys : [keys])) delete sessionStore[key];
        },
      },
      onChanged: { addListener: () => {} },
    },
    runtime: {
      onMessage: { addListener: (listener) => { messageListener = listener; } },
      onInstalled: { addListener: () => {} },
    },
    contextMenus: { create: () => {}, onClicked: { addListener: () => {} } },
    tabs: {
      sendMessage: async () => ({ ok: true }),
      captureVisibleTab: (windowId, options, callback) => callback('data:image/png;base64,c25hcHNob3Q='),
      getZoom: async () => 1,
    },
    scripting: { executeScript: async () => [] },
    action: { setIcon: () => {} },
  },
  fetch: mockFetch,
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: backgroundPath });
assert.equal(typeof messageListener, 'function', 'background message listener registered');

function send(message, tabId = 1) {
  return new Promise((resolve) => {
    messageListener(message, { tab: { id: tabId } }, resolve);
  });
}

async function flush(times = 20) {
  for (let i = 0; i < times; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

async function stats() {
  return send({ kind: 'getTranslationStats' });
}

function pageMessage(tag, n, { hold = true } = {}) {
  const cacheKey = `${hold ? 'hold-' : ''}${tag}-page-${n}|100x100`;
  return {
    kind: 'translateImage',
    base64Data: 'data:image/png;base64,ZmFrZQ==',
    cacheKey,
    originalImageUrl: `https://example.test/${tag}/${n}.jpg`,
    pageCacheKey: `https://example.test/${tag}/manga`,
    pageUrl: `https://example.test/${tag}/manga`,
    width: 100,
    height: 100,
  };
}

// Resolves whatever is currently dispatched, flushes (which drains the queue and dispatches
// the next wave), and repeats until nothing is active or queued. Needed because items still
// sitting in requestQueue have no fetch in flight yet -- they only appear in heldResolvers once
// a drain actually dispatches them.
async function resolveAllAndDrain(tag, maxWaves = 200) {
  for (let wave = 0; wave < maxWaves; wave += 1) {
    const current = await stats();
    if (current.activeRequests === 0 && current.queueLength === 0) return;
    const resolvers = Array.from(heldResolvers.values());
    resolvers.forEach((resolve) => resolve());
    await flush();
  }
  throw new Error(`${tag}: did not reach quiescence within ${maxWaves} drain waves`);
}

async function resetState() {
  await send({ kind: 'stopTranslations', mode: 'hard' });
  await send({ kind: 'setTranslationPaused', paused: false });
  await send({ kind: 'clearQueue' });
  await send({ kind: 'clearCache' });
  heldResolvers.clear();
  sessionStore = {};
  queueLimitSetting = 10;
  parallelLimitSetting = 2;
  cacheLimitSetting = DEFAULT_CACHE;
  fetchTimeoutSetting = DEFAULT_FETCH_TIMEOUT;
  await flush();
}

const results = [];
async function run(name, fn) {
  await resetState();
  try {
    await fn();
    results.push([name, 'pass', null]);
  } catch (error) {
    results.push([name, 'fail', error]);
    console.log(`[queue_matrix] ${name}: FAIL -- ${error.message}`);
  }
}

// =========================================================================================
// Section 1 -- exact-fit matrix: every queue-ahead value, every parallel setting, both arrival
// patterns, sending exactly (parallel + queueAhead) requests (the maximum that should ALL
// succeed with zero QueueFull). This is the direct answer to "does queue-ahead N actually work".
// =========================================================================================
async function runExactFitCell(tag, queueAhead, parallel, arrival) {
  queueLimitSetting = queueAhead;
  parallelLimitSetting = parallel;
  const n = parallel + queueAhead;
  const promises = [];
  for (let i = 0; i < n; i += 1) {
    promises.push(send(pageMessage(tag, i)));
    if (arrival === 'paced') await flush(3);
  }
  await flush();

  const mid = await stats();
  assert.ok(mid.activeRequests <= parallel, `${tag}: active (${mid.activeRequests}) <= parallel (${parallel})`);
  assert.ok(mid.queueLength <= queueAhead, `${tag}: queued (${mid.queueLength}) <= queueAhead (${queueAhead})`);
  assert.equal(mid.activeRequests + mid.queueLength, n, `${tag}: all ${n} requests accounted for (active+queued), none silently dropped`);
  const persisted = sessionStore[QUEUE_DESCRIPTOR_KEY] || [];
  assert.equal(persisted.length, mid.queueLength, `${tag}: persisted descriptors (${persisted.length}) match live queue length (${mid.queueLength})`);

  await resolveAllAndDrain(tag);

  const final = await stats();
  assert.equal(final.activeRequests, 0, `${tag}: quiescent active=0`);
  assert.equal(final.queueLength, 0, `${tag}: quiescent queue=0`);
  assert.ok(!sessionStore[QUEUE_DESCRIPTOR_KEY]?.length, `${tag}: descriptors cleared at quiescence`);

  const outcomes = await Promise.all(promises);
  for (const outcome of outcomes) {
    assert.ok(
      outcome.translatedImageDataUrl || outcome.fromCache,
      `${tag}: every one of ${n} requests resolved with a real result, none stuck/dropped -- got ${JSON.stringify(outcome)}`,
    );
  }
}

for (const queueAhead of QUEUE_AHEAD_VALUES) {
  for (const parallel of PARALLEL_VALUES) {
    for (const arrival of ['burst', 'paced']) {
      const tag = `m-${queueAhead}-${parallel}-${arrival}`;
      await run(`exact-fit qa=${queueAhead} par=${parallel} ${arrival}`, () => runExactFitCell(tag, queueAhead, parallel, arrival));
    }
  }
}

// =========================================================================================
// Section 2 -- cache-hit re-dispatch regression (the ACTUAL root cause of "5 queued, 2
// translate"): re-run the queue_cache_stress_test scenario-11 pattern at a representative
// spread of queue-ahead depths and parallel settings, not just the one depth already covered
// there.
// =========================================================================================
async function runCacheHitRedispatchCell(tag, queueAhead, parallel) {
  queueLimitSetting = Math.max(queueAhead, 3);
  parallelLimitSetting = parallel;

  // 1. Get page X genuinely cached.
  const xMsg = pageMessage(tag, 'x', { hold: false });
  const first = await send(xMsg);
  assert.ok(first.translatedImageDataUrl, `${tag}: X cached on first real dispatch`);

  // 2. Occupy every parallel slot with held pages.
  const heldSlotPromises = [];
  for (let i = 0; i < parallel; i += 1) {
    heldSlotPromises.push(send(pageMessage(tag, `slot-${i}`)));
    await flush(4);
  }
  let s = await stats();
  assert.equal(s.activeRequests, parallel, `${tag}: all ${parallel} slots occupied`);

  // 3. Force a genuine cache miss for X's lookup (translationCache itself untouched), then
  // re-request it -- with every slot occupied this must genuinely queue.
  const savedCacheLimit = cacheLimitSetting;
  cacheLimitSetting = 0;
  const xRequeue = send(xMsg);
  await flush(4);
  cacheLimitSetting = savedCacheLimit;

  // 4. Queue extra pages behind X.
  const extraPromises = [];
  for (let i = 0; i < 2; i += 1) {
    extraPromises.push(send(pageMessage(tag, `extra-${i}`)));
    await flush(4);
  }
  s = await stats();
  assert.equal(s.activeRequests, parallel, `${tag}: slots still occupied before any release`);
  assert.equal(s.queueLength, 3, `${tag}: X + 2 extras genuinely queued`);

  // 5. Free exactly one slot -- X (FIFO) resolves via the cache-hit early return. The bug this
  // regression-tests: that early-return settle must still drain the extras behind it.
  const firstHeldKey = Array.from(heldResolvers.keys())[0];
  heldResolvers.get(firstHeldKey)();
  await flush();

  const xResult = await xRequeue;
  assert.equal(xResult.fromCache, true, `${tag}: X resolved via cache-hit, not a real dispatch`);

  s = await stats();
  assert.equal(s.activeRequests, parallel, `${tag}: the freed slot was immediately re-filled from the queue, not left idle`);
  assert.equal(s.queueLength, 1, `${tag}: exactly one extra remains queued behind the promoted one`);

  await resolveAllAndDrain(tag);
  await Promise.all(heldSlotPromises);
  await Promise.all(extraPromises);

  s = await stats();
  assert.equal(s.activeRequests, 0);
  assert.equal(s.queueLength, 0);
}

for (const queueAhead of [0, 3, 5, 20, 50]) {
  for (const parallel of [1, 2, 3]) {
    const tag = `ch-${queueAhead}-${parallel}`;
    await run(`cache-hit-redispatch qa=${queueAhead} par=${parallel}`, () => runCacheHitRedispatchCell(tag, queueAhead, parallel));
  }
}

// =========================================================================================
// Section 3 -- edge set
// =========================================================================================

// Overflow: sends exactly 2 more than capacity must produce exactly 2 QueueFull, no more/less.
async function runOverflowCell(tag, queueAhead, parallel) {
  queueLimitSetting = queueAhead;
  parallelLimitSetting = parallel;
  const capacity = parallel + queueAhead;
  const n = capacity + 2;
  const promises = [];
  for (let i = 0; i < n; i += 1) promises.push(send(pageMessage(tag, i)));
  await flush();
  const outcomes = await Promise.all(
    promises.map((p) => Promise.race([p, new Promise((resolve) => setTimeout(() => resolve({ __pending: true }), 0))])),
  );
  const fullCount = outcomes.filter((o) => o.error === 'QueueFull').length;
  assert.equal(fullCount, 2, `${tag}: exactly 2 of ${n} requests over capacity ${capacity} were rejected`);
  await resolveAllAndDrain(tag);
}
for (const [queueAhead, parallel] of [[0, 1], [3, 2], [5, 1], [20, 2], [50, 3]]) {
  await run(`overflow qa=${queueAhead} par=${parallel}`, () => runOverflowCell(`ov-${queueAhead}-${parallel}`, queueAhead, parallel));
}

// Pause is a hard stop, not a freeze: setTranslationPaused(true) aborts every active controller
// AND clears the queue (background.js:935-940) -- this is existing, intentional behavior (the
// popup's own Pause help text says "Cancels queued browser work"). Resume does not resurrect
// anything; it just means new work can proceed normally again.
await run('pause aborts active and clears queued; resume starts clean', async () => {
  queueLimitSetting = 10;
  parallelLimitSetting = 1;
  const tag = 'pr';
  const p0 = send(pageMessage(tag, 0));
  await flush(4);
  const p1 = send(pageMessage(tag, 1));
  await flush(4);
  let s = await stats();
  assert.equal(s.activeRequests, 1);
  assert.equal(s.queueLength, 1);

  await send({ kind: 'setTranslationPaused', paused: true });
  await flush();
  s = await stats();
  assert.equal(s.isPaused, true);
  assert.equal(s.activeRequests, 0, 'pause aborts the in-flight request rather than leaving it running');
  assert.equal(s.queueLength, 0, 'pause clears the queue rather than freezing it');
  const r0 = await p0;
  const r1 = await p1;
  assert.equal(r0.error, 'TranslationPaused');
  assert.equal(r1.error, 'TranslationPaused');

  await send({ kind: 'setTranslationPaused', paused: false });
  s = await stats();
  assert.equal(s.isPaused, false);

  const fresh = send(pageMessage(tag, 'fresh'));
  await flush(4);
  s = await stats();
  assert.equal(s.activeRequests, 1, 'a fresh request after resume dispatches normally');
  await resolveAllAndDrain(tag);
  const freshResult = await fresh;
  assert.ok(freshResult.translatedImageDataUrl, 'post-resume work completes normally');
});

// clearQueue mid-drain, then re-enqueue -- the cleared items must not resurface, and fresh
// enqueues afterward must work normally.
await run('clearQueue mid-drain then re-enqueue', async () => {
  queueLimitSetting = 5;
  parallelLimitSetting = 1;
  const tag = 'cq';
  send(pageMessage(tag, 'active'));
  await flush(4);
  const cleared = send(pageMessage(tag, 'queued'));
  await flush(4);
  let s = await stats();
  assert.equal(s.queueLength, 1);

  const clearResponse = await send({ kind: 'clearQueue' });
  assert.equal(clearResponse.dropped, 1);
  const clearedResult = await cleared;
  assert.equal(clearedResult.error, 'QueueCleared');
  s = await stats();
  assert.equal(s.queueLength, 0);
  assert.ok(!sessionStore[QUEUE_DESCRIPTOR_KEY]?.length, 'descriptors cleared along with the queue');

  const fresh = send(pageMessage(tag, 'fresh'));
  await flush(4);
  s = await stats();
  assert.equal(s.queueLength, 1, 'a fresh enqueue after clearQueue still queues normally');
  await resolveAllAndDrain(tag);
  const freshResult = await fresh;
  assert.ok(freshResult.translatedImageDataUrl);
});

// setQueueLimit shrink mid-flight -- defect-A canary: shed items must be reflected in the
// persisted descriptors, or a later restart would falsely report them as crash-lost.
await run('setQueueLimit shrink mid-flight (defect-A canary)', async () => {
  queueLimitSetting = 10;
  parallelLimitSetting = 1;
  const tag = 'sql';
  send(pageMessage(tag, 'active'));
  await flush(4);
  const queued = [];
  for (let i = 0; i < 4; i += 1) {
    queued.push(send(pageMessage(tag, `q${i}`)));
    await flush(4);
  }
  let s = await stats();
  assert.equal(s.queueLength, 4);
  assert.equal((sessionStore[QUEUE_DESCRIPTOR_KEY] || []).length, 4, 'descriptors match before shrink');

  const shrinkResponse = await send({ kind: 'setQueueLimit', limit: 2 });
  assert.equal(shrinkResponse.success, true);
  s = await stats();
  assert.equal(s.queueLength, 2, 'shrink sheds the newest items down to the new limit');
  assert.equal(
    (sessionStore[QUEUE_DESCRIPTOR_KEY] || []).length,
    2,
    'THE FIX: persisted descriptors reflect the shed queue, not the stale pre-shed list',
  );

  await resolveAllAndDrain(tag);
});

// setParallelLimit up/down mid-drain.
await run('setParallelLimit down does not shed in-flight, blocks new admission until below', async () => {
  queueLimitSetting = 10;
  parallelLimitSetting = 3;
  const tag = 'spd';
  const active = [];
  for (let i = 0; i < 3; i += 1) {
    active.push(send(pageMessage(tag, `a${i}`)));
    await flush(4);
  }
  let s = await stats();
  assert.equal(s.activeRequests, 3);

  await send({ kind: 'setParallelLimit', limit: 1 });
  s = await stats();
  assert.equal(s.activeRequests, 3, 'lowering the parallel cap does not abort already-active requests');

  const blocked = send(pageMessage(tag, 'blocked'));
  await flush(4);
  s = await stats();
  assert.equal(s.queueLength, 1, 'new requests queue instead of exceeding the new lower cap');

  await resolveAllAndDrain(tag);
  await Promise.all(active);
  const blockedResult = await blocked;
  assert.ok(blockedResult.translatedImageDataUrl);
});

await run('setParallelLimit up admits queued work immediately', async () => {
  queueLimitSetting = 10;
  parallelLimitSetting = 1;
  const tag = 'spu';
  send(pageMessage(tag, 'a0'));
  await flush(4);
  const queued = [send(pageMessage(tag, 'q0')), send(pageMessage(tag, 'q1'))];
  await flush(4);
  let s = await stats();
  assert.equal(s.activeRequests, 1);
  assert.equal(s.queueLength, 2);

  await send({ kind: 'setParallelLimit', limit: 3 });
  await flush();
  s = await stats();
  assert.equal(s.activeRequests, 3, 'raising the parallel cap immediately admits queued work, not just future arrivals');
  assert.equal(s.queueLength, 0);

  await resolveAllAndDrain(tag);
  await Promise.all(queued);
});

// cacheLimit < queueDepth LRU characterization at the largest queue-ahead value -- documented
// behavior, not a bug: early pages evict and a revisit re-translates.
await run('cacheLimit(24) < queueAhead(50) LRU eviction characterized', async () => {
  queueLimitSetting = 50;
  parallelLimitSetting = 3;
  cacheLimitSetting = 24;
  const tag = 'lru';
  const promises = [];
  for (let i = 0; i < 53; i += 1) promises.push(send(pageMessage(tag, i)));
  await flush();
  await resolveAllAndDrain(tag);
  await Promise.all(promises);

  let s = await stats();
  assert.equal(s.cacheSize, 24, 'cache holds exactly its configured limit, not the full 53 translated');

  // Revisiting the FIRST page (evicted first under LRU) must be a genuine miss (real dispatch),
  // not silently wrong -- this is the documented cache<queue-depth tradeoff, not a defect.
  const revisit = await send(pageMessage(tag, 0, { hold: false }));
  assert.notEqual(revisit.fromCache, true, 'the earliest page was LRU-evicted and re-translates on revisit, as documented');
});

// =========================================================================================
// Section 4 -- SW-restart edge cases, incl. defect-B canary (wake-by-message race). Needs a
// fresh vm context sharing the same session-store object -- the single-sandbox pattern above
// can't simulate a real service-worker death (in-memory state wiped, storage.session survives).
// =========================================================================================
function makeRestartContext(store, queueLimit = 50) {
  let listener;
  const held = new Map();
  async function fetchImpl(url, options = {}) {
    const urlStr = String(url);
    if (urlStr.includes('/v1/health') || urlStr.includes('/v1/warmup') || urlStr.includes('/v1/diagnostics/log')) {
      return { ok: true, json: async () => ({ ok: true }) };
    }
    const body = options.body ? JSON.parse(options.body) : null;
    const cacheKey = body?.metadata?.cacheKey || '';
    if (cacheKey.includes('hold')) {
      return new Promise((resolve) => {
        held.set(cacheKey, () => resolve({ ok: true, json: async () => respondFor() }));
      });
    }
    return { ok: true, json: async () => respondFor() };
  }
  const ctxSandbox = {
    console,
    setTimeout,
    clearTimeout,
    AbortController,
    chrome: {
      storage: {
        local: {
          get: async () => ({
            localPipelineUrl: 'http://127.0.0.1:8766/v1/translate-image',
            localPipelineLanguage: 'ja',
            translationCachePages: 50,
            translationQueuePages: queueLimit,
            translationParallelPages: 1,
            translationFetchTimeoutMs: 360000,
          }),
          set: async () => {},
        },
        session: {
          get: async (keys) => {
            const key = Array.isArray(keys) ? keys[0] : keys;
            return Object.prototype.hasOwnProperty.call(store, key) ? { [key]: store[key] } : {};
          },
          set: async (entries) => { Object.assign(store, entries); },
          remove: async (keys) => {
            for (const key of (Array.isArray(keys) ? keys : [keys])) delete store[key];
          },
        },
        onChanged: { addListener: () => {} },
      },
      runtime: {
        onMessage: { addListener: (l) => { listener = l; } },
        onInstalled: { addListener: () => {} },
      },
      contextMenus: { create: () => {}, onClicked: { addListener: () => {} } },
      tabs: {
        sendMessage: async () => ({ ok: true }),
        captureVisibleTab: (w, o, cb) => cb('data:image/png;base64,c25hcHNob3Q='),
        getZoom: async () => 1,
      },
      scripting: { executeScript: async () => [] },
      action: { setIcon: () => {} },
    },
    fetch: fetchImpl,
  };
  vm.createContext(ctxSandbox);
  vm.runInContext(source, ctxSandbox, { filename: backgroundPath });
  const sendMsg = (message) => new Promise((resolve) => listener(message, { tab: { id: 1 } }, resolve));
  const flushCtx = async (times = 20) => { for (let i = 0; i < times; i += 1) await new Promise((r) => setTimeout(r, 0)); };
  const pageMsg = (n, holdIt = true) => ({
    kind: 'translateImage',
    base64Data: 'data:image/png;base64,ZmFrZQ==',
    cacheKey: `${holdIt ? 'hold-' : ''}restart-${n}|100x100`,
    originalImageUrl: `https://example.test/restart/${n}.jpg`,
    pageCacheKey: 'https://example.test/restart/manga',
    pageUrl: 'https://example.test/restart/manga',
    width: 100,
    height: 100,
  });
  return { send: sendMsg, flush: flushCtx, pageMessage: pageMsg, held };
}

await run('SW-restart at depth 0 reports nothing lost', async () => {
  const store = {};
  const ctx1 = makeRestartContext(store);
  await ctx1.flush();
  const ctx2 = makeRestartContext(store);
  const s = await ctx2.send({ kind: 'getTranslationStats' });
  assert.equal(s.restartLost, 0);
});

for (const depth of [1, 5, 20]) {
  await run(`SW-restart at depth ${depth} reports exactly ${depth} lost`, async () => {
    const store = {};
    const ctx1 = makeRestartContext(store);
    ctx1.send(ctx1.pageMessage('active')); // occupies the one slot, held forever
    await ctx1.flush();
    for (let i = 0; i < depth; i += 1) {
      ctx1.send(ctx1.pageMessage(`q${i}`));
      await ctx1.flush(4);
    }
    const before = await ctx1.send({ kind: 'getTranslationStats' });
    assert.equal(before.queueLength, depth);

    const ctx2 = makeRestartContext(store);
    await ctx2.flush();
    const after = await ctx2.send({ kind: 'getTranslationStats' });
    assert.equal(after.restartLost, depth, `depth ${depth}: exactly ${depth} reported lost, not more or fewer`);

    await ctx1.send({ kind: 'stopTranslations', mode: 'hard' });
    await ctx1.flush();
  });
}

// Defect-B canary: the SW is woken BY a real translate request (not an idle module load) --
// queueTranslation's synchronous restartLost=0 must not be overwritten by the async session
// read resolving after it.
await run('SW wake-by-message does not resurrect a just-cleared restartLost (defect-B canary)', async () => {
  const store = { [QUEUE_DESCRIPTOR_KEY]: ['stale-a', 'stale-b'] };
  const ctx = makeRestartContext(store);
  // Fire the real translate request in the same tick the context is created, before the
  // module-top session read has had any chance to resolve -- this is the actual race window.
  const p = ctx.send(ctx.pageMessage('wake', false));
  await ctx.flush(30);
  await p;
  const s = await ctx.send({ kind: 'getTranslationStats' });
  assert.equal(s.restartLost, 0, 'a real translate request racing the restart-loss read must win -- the warning must not resurface');
});

// =========================================================================================
// Section 5 -- seeded invariant fuzz. mulberry32 PRNG for reproducibility; a failing seed is
// printed so it can be replayed deterministically.
// =========================================================================================
function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const FUZZ_SEEDS = 12;
const FUZZ_OPS = 150;

async function runFuzzSeed(seed) {
  const rng = mulberry32(seed);
  queueLimitSetting = 10;
  parallelLimitSetting = 2;
  const tag = `fz${seed}`;
  const inFlight = []; // { promise, cacheKey }
  let nextId = 0;

  const weightedOps = [
    ['enqueue', 5],
    ['resolveOldest', 4],
    ['revisitCompleted', 2],
    ['pauseResume', 1],
    ['clearQueue', 1],
    ['setQueueLimit', 1],
    ['setParallelLimit', 1],
  ];
  const opTable = weightedOps.flatMap(([op, weight]) => Array(weight).fill(op));
  const completedKeys = [];
  // Lowering parallelLimit mid-flight intentionally does NOT abort already-active requests
  // (verified by the dedicated "setParallelLimit down" edge test) -- so activeRequests can
  // legitimately exceed the new lower limit transiently until those requests finish. The real
  // invariant is narrower: while over capacity, the count must never INCREASE (no new admission
  // happens), only shrink as in-flight requests complete.
  let previousActive = 0;

  for (let step = 0; step < FUZZ_OPS; step += 1) {
    const op = opTable[Math.floor(rng() * opTable.length)];
    if (op === 'enqueue') {
      const cacheKey = `hold-${tag}-${nextId}|100x100`;
      const msg = pageMessage(tag, nextId, { hold: true });
      nextId += 1;
      const promise = send(msg).then((result) => {
        completedKeys.push(cacheKey);
        return result;
      });
      inFlight.push({ promise, cacheKey });
    } else if (op === 'resolveOldest') {
      const key = Array.from(heldResolvers.keys())[0];
      if (key) heldResolvers.get(key)();
    } else if (op === 'revisitCompleted' && completedKeys.length > 0) {
      const key = completedKeys[Math.floor(rng() * completedKeys.length)];
      const parts = key.replace(/^hold-/, '').split('-');
      const n = parts[parts.length - 1].split('|')[0];
      send(pageMessage(tag, n, { hold: false })).catch(() => {});
    } else if (op === 'pauseResume') {
      await send({ kind: 'setTranslationPaused', paused: true });
      await flush(2);
      await send({ kind: 'setTranslationPaused', paused: false });
    } else if (op === 'clearQueue') {
      await send({ kind: 'clearQueue' });
    } else if (op === 'setQueueLimit') {
      queueLimitSetting = [0, 1, 3, 5, 10, 20][Math.floor(rng() * 6)];
      await send({ kind: 'setQueueLimit', limit: queueLimitSetting });
    } else if (op === 'setParallelLimit') {
      parallelLimitSetting = [1, 2, 3][Math.floor(rng() * 3)];
      await send({ kind: 'setParallelLimit', limit: parallelLimitSetting });
    }
    await flush(3);

    const s = await stats();
    if (previousActive > s.parallelLimit) {
      assert.ok(
        s.activeRequests <= previousActive,
        `seed ${seed} step ${step}: while over capacity (prev ${previousActive} > limit ${s.parallelLimit}), active must only shrink, not grow to ${s.activeRequests}`,
      );
    } else {
      assert.ok(s.activeRequests <= s.parallelLimit, `seed ${seed} step ${step}: active (${s.activeRequests}) <= parallel (${s.parallelLimit})`);
    }
    assert.ok(s.queueLength <= s.queueLimit, `seed ${seed} step ${step}: queued (${s.queueLength}) <= queueLimit (${s.queueLimit})`);
    assert.ok(s.activeRequests >= 0 && s.queueLength >= 0, `seed ${seed} step ${step}: no negative counts`);
    const persisted = sessionStore[QUEUE_DESCRIPTOR_KEY] || [];
    assert.equal(persisted.length, s.queueLength, `seed ${seed} step ${step}: persisted descriptors (${persisted.length}) match live queue (${s.queueLength})`);
    previousActive = s.activeRequests;
  }

  // Full drain: raise limits generously so nothing is permanently starved, then resolve
  // everything and confirm every fired request eventually settles.
  queueLimitSetting = 999;
  await send({ kind: 'setQueueLimit', limit: 999 });
  parallelLimitSetting = 3;
  await send({ kind: 'setParallelLimit', limit: 3 });
  await resolveAllAndDrain(tag, 500);
  await Promise.all(inFlight.map((entry) => entry.promise));

  const finalStats = await stats();
  assert.equal(finalStats.activeRequests, 0, `seed ${seed}: quiescent active=0`);
  assert.equal(finalStats.queueLength, 0, `seed ${seed}: quiescent queue=0`);
}

for (let seed = 1; seed <= FUZZ_SEEDS; seed += 1) {
  await run(`fuzz seed=${seed} (${FUZZ_OPS} ops)`, () => runFuzzSeed(seed));
}

// ---------------------------------------------------------------------------------------
const failed = results.filter(([, status]) => status === 'fail');
console.log(`\nqueue_matrix_stress_test summary: ${results.length - failed.length}/${results.length} passed`);
if (failed.length > 0) {
  console.log('Failed:', failed.map(([name, , error]) => `${name} -- ${error.message}`).join('\n  '));
  process.exitCode = 1;
} else {
  console.log('queue_matrix_stress_test=pass');
}
