// Stress suite for the queue / parallel-cap / cache stack in background.js.
// Unlike background_contract_test.mjs (one linear happy-path script), this file runs
// independent scenarios and reports pass/fail per scenario so a single invocation can
// demonstrate multiple known defects (E1/E2/E3) at once, pre-fix, and a clean pass post-fix.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const backgroundPath = path.join(extensionRoot, 'background.js');
const source = fs.readFileSync(backgroundPath, 'utf8');

const DEFAULT_QUEUE = 10;
const DEFAULT_PARALLEL = 2;
const DEFAULT_CACHE = 12;
const DEFAULT_FETCH_TIMEOUT = 360000;

let messageListener;
let fetchCount = 0;
let queueLimitSetting = DEFAULT_QUEUE;
let parallelLimitSetting = DEFAULT_PARALLEL;
let cacheLimitSetting = DEFAULT_CACHE;
let localPipelineLanguageSetting = 'ja';
let fetchTimeoutSetting = DEFAULT_FETCH_TIMEOUT;
const contentMessages = [];
const executedScripts = [];
let contentInjected = false;

// cacheKey (the raw metadata field, not the internal hashed cacheId) -> resolver.
// Requests only land here once the mock fetch() is actually invoked, i.e. once the
// extension has truly dispatched them to the network -- queued items never appear here.
const heldResolvers = new Map();
const translationsOnlyKeys = new Set();

// translateSnapshot's fix derives cache identity from the captured pixel content
// (base64Data), not just selection geometry -- so distinct selections must produce
// distinguishable mock output, or every snapshot in this suite would collide on one
// cacheId. This chain threads the crop rect through createImageBitmap -> canvas ->
// blob -> FileReader so the final "captured" data URL actually varies per selection,
// mirroring how a real screenshot crop varies with what's on screen.
class MockFileReader {
  readAsDataURL(blob) {
    const descriptor = blob?.__descriptor || 'unknown';
    this.result = `data:image/png;base64,${Buffer.from(`mock-crop:${descriptor}`).toString('base64')}`;
    this.onloadend();
  }
}

class MockOffscreenCanvas {
  constructor(width, height) {
    this.width = width;
    this.height = height;
    this._descriptor = `${width}x${height}`;
  }
  getContext() {
    return {
      drawImage: (bitmap) => {
        if (bitmap?.__descriptor) this._descriptor = bitmap.__descriptor;
      },
    };
  }
  convertToBlob() {
    return Promise.resolve({ __mockBlob: true, __descriptor: this._descriptor });
  }
}

function respondFor(cacheKey) {
  if (translationsOnlyKeys.has(cacheKey)) {
    return { translations: [{ text: 'translated-overlay' }], report: { pipeline: 'local-8-stage' } };
  }
  return {
    translatedImageDataUrl: 'data:image/png;base64,ZmFrZQ==',
    report: { pipeline: 'local-8-stage', stageSequence: [] },
    translations: [],
  };
}

async function mockFetch(url, options = {}) {
  const urlStr = String(url);
  if (urlStr.startsWith('data:')) {
    // The selection-snapshot crop path fetches its own captured data URL to get a Blob.
    return { ok: true, blob: async () => ({ __mockBlob: true }) };
  }
  if (urlStr.includes('/v1/diagnostics/log')) {
    return { ok: true, json: async () => ({ ok: true }) };
  }
  if (urlStr.includes('/v1/cache/clear')) {
    return { ok: true, json: async () => ({ success: true, status: 'cleared', clearedSamples: 0 }) };
  }
  if (urlStr.includes('/v1/runtime/soft-stop')) {
    return { ok: true, json: async () => ({ success: true, status: 'stopped', releaseGpu: false }) };
  }
  if (urlStr.includes('/v1/runtime/hard-stop')) {
    return { ok: true, json: async () => ({ success: true, status: 'stopped', releaseGpu: true }) };
  }
  if (urlStr.includes('/v1/gpu/release')) {
    return { ok: true, json: async () => ({ success: true, status: 'released' }) };
  }
  if (urlStr.includes('/v1/health')) {
    return { ok: true, json: async () => ({ ok: true }) };
  }
  if (urlStr.includes('/v1/warmup')) {
    return { ok: true, json: async () => ({ ok: true, status: 'warming' }) };
  }

  fetchCount += 1;
  const body = options.body ? JSON.parse(options.body) : null;
  const cacheKey = body?.metadata?.cacheKey || '';
  // translateSnapshot never sets metadata.cacheKey (its cache identity is the captured
  // pixel content, not a caller-supplied key) -- fall back to pageUrl so snapshot requests
  // can still be individually held/resolved by name in tests.
  const holdKey = cacheKey || String(body?.metadata?.pageUrl || '');

  if (holdKey.includes('hold')) {
    return new Promise((resolve, reject) => {
      const settle = () => heldResolvers.delete(holdKey);
      heldResolvers.set(holdKey, () => {
        settle();
        resolve({ ok: true, json: async () => respondFor(cacheKey) });
      });
      if (options.signal) {
        if (options.signal.aborted) {
          settle();
          const err = new Error('Aborted');
          err.name = 'AbortError';
          reject(err);
          return;
        }
        options.signal.addEventListener('abort', () => {
          settle();
          const err = new Error('Aborted');
          err.name = 'AbortError';
          reject(err);
        });
      }
    });
  }

  return { ok: true, json: async () => respondFor(cacheKey) };
}

const sandbox = {
  console,
  URL,
  setTimeout,
  clearTimeout,
  AbortController,
  FileReader: MockFileReader,
  OffscreenCanvas: MockOffscreenCanvas,
  createImageBitmap: async (blob, sx, sy, sw, sh) => ({ close: () => {}, __descriptor: `${sx},${sy},${sw}x${sh}` }),
  chrome: {
    storage: {
      local: {
        get: async () => ({
          localPipelineUrl: 'http://127.0.0.1:8766/v1/translate-image',
          localPipelineLanguage: localPipelineLanguageSetting,
          translationCachePages: cacheLimitSetting,
          translationQueuePages: queueLimitSetting,
          translationParallelPages: parallelLimitSetting,
          translationFetchTimeoutMs: fetchTimeoutSetting,
        }),
        set: async () => {},
      },
      session: {
        get: async () => ({}),
        set: async () => {},
        remove: async () => {},
      },
      onChanged: { addListener: () => {} },
    },
    runtime: {
      onMessage: { addListener: (listener) => { messageListener = listener; } },
      onInstalled: { addListener: () => {} },
    },
    contextMenus: { create: () => {}, onClicked: { addListener: () => {} } },
    tabs: {
      sendMessage: async (tabId, message) => {
        contentMessages.push({ tabId, message });
        if (message.kind === 'pingContentScript' && !contentInjected) {
          throw new Error('Receiving end does not exist.');
        }
        return { ok: true };
      },
      captureVisibleTab: (windowId, options, callback) => callback('data:image/png;base64,c25hcHNob3Q='),
      getZoom: async () => 1,
    },
    scripting: {
      executeScript: async (details) => {
        executedScripts.push(details);
        if (details.files?.includes('content.js')) contentInjected = true;
        return [];
      },
    },
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

function pageMessage(scenarioTag, n, { hold = true, translationsOnly = false } = {}) {
  const cacheKey = `${hold ? 'hold-' : ''}${scenarioTag}-page-${n}|100x100`;
  if (translationsOnly) translationsOnlyKeys.add(cacheKey);
  return {
    kind: 'translateImage',
    base64Data: 'data:image/png;base64,ZmFrZQ==',
    cacheKey,
    originalImageUrl: `https://example.test/${scenarioTag}/${n}.jpg`,
    pageCacheKey: `https://example.test/${scenarioTag}/manga`,
    pageUrl: `https://example.test/${scenarioTag}/manga`,
    width: 100,
    height: 100,
  };
}

function resolvePage(cacheKey) {
  const resolver = heldResolvers.get(cacheKey);
  assert.ok(resolver, `expected an in-flight fetch to resolve for ${cacheKey}`);
  resolver();
}

async function resetState() {
  await send({ kind: 'stopTranslations', mode: 'hard' });
  await send({ kind: 'setTranslationPaused', paused: false });
  await send({ kind: 'clearQueue' });
  await send({ kind: 'clearCache' });
  heldResolvers.clear();
  translationsOnlyKeys.clear();
  fetchCount = 0;
  queueLimitSetting = DEFAULT_QUEUE;
  parallelLimitSetting = DEFAULT_PARALLEL;
  cacheLimitSetting = DEFAULT_CACHE;
  fetchTimeoutSetting = DEFAULT_FETCH_TIMEOUT;
  await flush();
}

// ---------------------------------------------------------------------------------------
// Scenario 1 -- the user's exact end-to-end scenario, paced arrival (one page after another,
// as real page navigation would produce), not a synchronous burst.
// ---------------------------------------------------------------------------------------
async function scenarioUserWalkthrough() {
  queueLimitSetting = 10;
  parallelLimitSetting = 2;
  const tag = 's1';
  const pending = [];
  for (let n = 1; n <= 10; n += 1) {
    pending.push(send(pageMessage(tag, n)));
    await flush(4);
  }

  let s = await stats();
  assert.equal(s.activeRequests, 2, 'two pages in flight at the parallel cap');
  assert.equal(s.queueLength, 8, 'the remaining eight pages are queued, not dropped or fetched early');

  const key1 = `hold-${tag}-page-1|100x100`;
  resolvePage(key1);
  await flush();

  s = await stats();
  assert.equal(s.activeRequests, 2, 'a queued page is promoted to fill the freed slot');
  assert.equal(s.queueLength, 7, 'the queue shrinks by exactly one as the freed slot is filled');

  const cachedLookup = await send({
    kind: 'lookupCachedTranslation',
    cacheKey: key1,
    originalImageUrl: `https://example.test/${tag}/1.jpg`,
    pageCacheKey: `https://example.test/${tag}/manga`,
    pageUrl: `https://example.test/${tag}/manga`,
  });
  assert.equal(cachedLookup.hit, true, 'page 1 is already cached after completing');

  const sAfterLookup = await stats();
  assert.deepEqual(
    { active: sAfterLookup.activeRequests, queued: sAfterLookup.queueLength },
    { active: s.activeRequests, queued: s.queueLength },
    'a read-only cache lookup must not consume or perturb queue/parallel state',
  );

  // Note: revisits below deliberately reuse pageMessage(tag, n) (the same default hold:true
  // shape used for the original dispatch), NOT { hold: false } -- the "hold-" marker only
  // affects the mock fetch's behaviour, and a cache hit never reaches the mock fetch at all,
  // so it's inert here. Passing hold:false would instead build a DIFFERENT cacheKey string
  // (no "hold-" prefix), which is a different cache identity entirely and would wrongly miss.
  const fetchCountBeforeRevisit = fetchCount;
  const revisit1 = await send(pageMessage(tag, 1));
  assert.equal(revisit1.fromCache, true, 're-requesting a cached page returns fromCache');
  assert.equal(fetchCount, fetchCountBeforeRevisit, 'no new network fetch for a cached revisit');

  for (let n = 2; n <= 10; n += 1) {
    const key = `hold-${tag}-page-${n}|100x100`;
    await flush(4);
    resolvePage(key);
  }
  await flush();

  const drained = await stats();
  assert.equal(drained.activeRequests, 0, 'queue fully drains');
  assert.equal(drained.queueLength, 0, 'queue fully drains');

  const fetchCountBeforeFinalRevisit = fetchCount;
  for (let n = 1; n <= 10; n += 1) {
    const result = await send(pageMessage(tag, n));
    assert.equal(result.fromCache, true, `page ${n} is cached after the walkthrough`);
  }
  assert.equal(fetchCount, fetchCountBeforeFinalRevisit, 'revisiting ten already-cached pages issues zero fetches');

  for (const promise of pending) await promise;
}

// ---------------------------------------------------------------------------------------
// Scenario 2 -- parallel enforcement under a true synchronous burst (Bug E1: TOCTOU on the
// admission check lets more than parallelLimit requests reach the network at once).
// ---------------------------------------------------------------------------------------
async function scenarioParallelBurst() {
  queueLimitSetting = 10;
  parallelLimitSetting = 2;
  const tag = 's2';
  const promises = [];
  for (let n = 1; n <= 6; n += 1) {
    promises.push(send(pageMessage(tag, n)));
  }
  await flush();

  assert.ok(
    heldResolvers.size <= parallelLimitSetting,
    `parallel cap violated: ${heldResolvers.size} concurrent network fetches with parallelLimit=${parallelLimitSetting}`,
  );

  for (const key of Array.from(heldResolvers.keys())) resolvePage(key);
  await flush();
  for (let round = 0; round < 6 && heldResolvers.size > 0; round += 1) {
    for (const key of Array.from(heldResolvers.keys())) resolvePage(key);
    await flush();
  }
  for (const promise of promises) await promise;
}

// ---------------------------------------------------------------------------------------
// Scenario 3 -- drain over-dispatch: when one active slot frees, the queue must promote
// only as many items as slots actually freed, not the whole backlog at once.
// ---------------------------------------------------------------------------------------
async function scenarioDrainOverDispatch() {
  queueLimitSetting = 10;
  parallelLimitSetting = 2;
  const tag = 's3';
  const promises = [];
  for (let n = 1; n <= 8; n += 1) {
    promises.push(send(pageMessage(tag, n)));
    await flush(4);
  }

  let s = await stats();
  assert.equal(s.activeRequests, 2);
  assert.equal(s.queueLength, 6);

  resolvePage(`hold-${tag}-page-1|100x100`);
  await flush();

  s = await stats();
  assert.equal(s.activeRequests, 2, 'freeing exactly one slot promotes exactly one queued item, not the whole backlog');
  assert.equal(s.queueLength, 5);

  for (let n = 2; n <= 8; n += 1) {
    const key = `hold-${tag}-page-${n}|100x100`;
    await flush(4);
    resolvePage(key);
  }
  await flush();
  for (const promise of promises) await promise;
}

// ---------------------------------------------------------------------------------------
// Scenario 4 -- QueueFull rejection at the bounded queue limit.
// ---------------------------------------------------------------------------------------
async function scenarioQueueFull() {
  queueLimitSetting = 2;
  parallelLimitSetting = 1;
  const tag = 's4';
  const held = [];
  for (let n = 1; n <= 3; n += 1) {
    held.push(send(pageMessage(tag, n)));
    await flush(4);
  }

  const rejected = await send(pageMessage(tag, 4));
  assert.equal(rejected.error, 'QueueFull');
  assert.equal(rejected.queueLimit, 2);

  for (let n = 1; n <= 3; n += 1) {
    const key = `hold-${tag}-page-${n}|100x100`;
    await flush(4);
    resolvePage(key);
  }
  await flush();
  for (const promise of held) await promise;
}

// ---------------------------------------------------------------------------------------
// Scenario 5 -- in-flight dedupe: identical page requested twice while the first is still
// running must not trigger a second network fetch.
// ---------------------------------------------------------------------------------------
async function scenarioInFlightDedupe() {
  queueLimitSetting = 10;
  parallelLimitSetting = 2;
  const tag = 's5';
  const msg = pageMessage(tag, 1);
  const first = send(msg);
  await flush(4);
  const fetchCountAfterFirst = fetchCount;
  const second = send({ ...msg });
  await flush(4);

  assert.equal(fetchCount, fetchCountAfterFirst, 'a duplicate in-flight request must not fetch again');

  resolvePage(`hold-${tag}-page-1|100x100`);
  const [firstResult, secondResult] = await Promise.all([first, second]);
  assert.equal(firstResult.translatedImageDataUrl, 'data:image/png;base64,ZmFrZQ==');
  assert.equal(secondResult.fromInFlight, true, 'the duplicate joins the in-flight promise');
}

// ---------------------------------------------------------------------------------------
// Scenario 6 -- LRU cache retention: eviction of the oldest entry, and recency bump on
// access keeping a frequently-revisited page alive under pressure.
// ---------------------------------------------------------------------------------------
async function scenarioLruRetention() {
  queueLimitSetting = 10;
  parallelLimitSetting = 4;
  cacheLimitSetting = 3;
  const tag = 's6';

  for (let n = 1; n <= 3; n += 1) {
    await send(pageMessage(tag, n, { hold: false }));
  }
  let s = await stats();
  assert.equal(s.cacheSize, 3);

  const fetchCountBeforeFourth = fetchCount;
  await send(pageMessage(tag, 4, { hold: false }));
  s = await stats();
  assert.equal(s.cacheSize, 3, 'cache stays at its configured limit');
  assert.ok(fetchCount > fetchCountBeforeFourth, 'page 4 was a genuine fetch');

  const revisit1 = await send(pageMessage(tag, 1, { hold: false }));
  assert.equal(revisit1.fromCache, undefined, 'the oldest page (1) was evicted and must be recomputed');

  const revisit3 = await send(pageMessage(tag, 3, { hold: false }));
  assert.equal(revisit3.fromCache, true, 'a page inside the retention window is still cached');

  cacheLimitSetting = 2;
  await send({ kind: 'setCacheLimit', limit: 2 });
  await send(pageMessage(tag, 5, { hold: false }));
  await send({ kind: 'getTranslationStats' });

  const accessOrderTag = `${tag}b`;
  cacheLimitSetting = 3;
  await send({ kind: 'setCacheLimit', limit: 3 });
  await send(pageMessage(accessOrderTag, 1, { hold: false }));
  await send(pageMessage(accessOrderTag, 2, { hold: false }));
  await send({
    kind: 'lookupCachedTranslation',
    ...pageMessage(accessOrderTag, 1, { hold: false }),
  });
  await send(pageMessage(accessOrderTag, 3, { hold: false }));
  await send(pageMessage(accessOrderTag, 4, { hold: false }));
  const page1AfterBump = await send(pageMessage(accessOrderTag, 1, { hold: false }));
  assert.equal(page1AfterBump.fromCache, true, 'accessing page 1 bumped its recency, so it survived two later insertions');

  cacheLimitSetting = 0;
  await send({ kind: 'setCacheLimit', limit: 0 });
  const zeroTag = `${tag}c`;
  await send(pageMessage(zeroTag, 1, { hold: false }));
  const zeroRevisit = await send(pageMessage(zeroTag, 1, { hold: false }));
  assert.equal(zeroRevisit.fromCache, undefined, 'cacheLimit=0 disables caching entirely');
}

// ---------------------------------------------------------------------------------------
// Scenario 7 -- pause/stop under load: queued items reject, actives abort, cache survives,
// and a subsequent request for an already-completed page is still a cache hit.
// ---------------------------------------------------------------------------------------
async function scenarioPauseStopUnderLoad() {
  queueLimitSetting = 10;
  parallelLimitSetting = 2;
  const tag = 's7';

  const completed = await send(pageMessage(tag, 0, { hold: false }));
  assert.ok(completed.translatedImageDataUrl);

  const promises = [];
  for (let n = 1; n <= 5; n += 1) {
    promises.push(send(pageMessage(tag, n)));
    await flush(4);
  }
  let s = await stats();
  assert.equal(s.activeRequests, 2);
  assert.equal(s.queueLength, 3);

  const pauseResult = await send({ kind: 'setTranslationPaused', paused: true });
  assert.equal(pauseResult.success, true);
  await flush();

  const results = await Promise.all(promises);
  const errors = results.map((r) => r.error);
  assert.ok(errors.every((e) => e === 'TranslationPaused'), `all in-flight and queued pages reject on pause: ${JSON.stringify(errors)}`);

  s = await stats();
  assert.equal(s.activeRequests, 0, 'pause aborts active jobs');
  assert.equal(s.queueLength, 0, 'pause drops queued jobs');
  assert.equal(s.isPaused, true);

  await send({ kind: 'setTranslationPaused', paused: false });
  const revisit0 = await send(pageMessage(tag, 0, { hold: false }));
  assert.equal(revisit0.fromCache, true, 'the cache survives a pause/resume cycle');

  const promises2 = [];
  for (let n = 10; n <= 11; n += 1) {
    promises2.push(send(pageMessage(tag, n)));
    await flush(4);
  }
  const hardStop = await send({ kind: 'stopTranslations', mode: 'hard', tabId: 1 });
  assert.equal(hardStop.backend.releaseGpu, true);
  await flush();
  const results2 = await Promise.all(promises2);
  assert.ok(results2.every((r) => r.error === 'HardStopped' || r.error === 'TranslationPaused'));

  await send({ kind: 'setTranslationPaused', paused: false });
}

// ---------------------------------------------------------------------------------------
// Scenario 8 -- limit changes mid-flight: shrinking the queue sheds the tail (documented
// LIFO behaviour), raising the parallel limit immediately drains more of the backlog.
// ---------------------------------------------------------------------------------------
async function scenarioLimitChangesMidFlight() {
  queueLimitSetting = 10;
  parallelLimitSetting = 1;
  const tag = 's8';
  const promises = [];
  for (let n = 1; n <= 5; n += 1) {
    promises.push(send(pageMessage(tag, n)));
    await flush(4);
  }
  let s = await stats();
  assert.equal(s.activeRequests, 1);
  assert.equal(s.queueLength, 4);

  const shrink = await send({ kind: 'setQueueLimit', limit: 1 });
  assert.equal(shrink.success, true);
  assert.equal(shrink.queueLength, 1, 'shrinking the queue sheds the tail down to the new limit');

  const droppedResults = await Promise.all(promises.slice(2));
  assert.ok(droppedResults.slice(1).every((r) => r.error === 'QueueFull'), 'shed tail entries resolve with QueueFull');

  parallelLimitSetting = 3;
  const raise = await send({ kind: 'setParallelLimit', limit: 3 });
  assert.equal(raise.success, true);
  await flush();
  s = await stats();
  assert.equal(s.activeRequests <= 3, true);

  resolvePage(`hold-${tag}-page-1|100x100`);
  await flush();
  const remainingHeld = Array.from(heldResolvers.keys()).filter((k) => k.startsWith(`hold-${tag}-`));
  for (const key of remainingHeld) resolvePage(key);
  await flush();
  await Promise.all(promises);
}

// ---------------------------------------------------------------------------------------
// Scenario 9 -- selection-area (snapshot) translate must behave like any other job: cached
// on repeat identical selections, cancellable by stop, and counted against the parallel
// budget. (Bug E2: today translateSnapshot bypasses all of this.)
// ---------------------------------------------------------------------------------------
async function scenarioSelectionSnapshot() {
  queueLimitSetting = 10;
  parallelLimitSetting = 1;
  const tag = 's9';
  // translateSnapshot has no caller-supplied cache key: its identity is the captured pixel
  // content (see the mock createImageBitmap/OffscreenCanvas/FileReader chain), scoped by
  // pageUrl. An identical selection (same rect) on the same page must therefore hit cache.
  const snapshotMessage = {
    kind: 'translateSnapshot',
    pageUrl: `https://example.test/${tag}/manga`,
    dimensions: { left: 10, top: 20, width: 100, height: 80, devicePixelRatio: 2 },
  };

  const fetchCountBeforeFirst = fetchCount;
  const first = await send(snapshotMessage, 1);
  assert.ok(!first.error, `first snapshot should succeed: ${JSON.stringify(first)}`);
  assert.ok(fetchCount > fetchCountBeforeFirst, 'first snapshot performs a real translation');

  const fetchCountBeforeSecond = fetchCount;
  const second = await send({ ...snapshotMessage }, 1);
  assert.equal(second.fromCache, true, 'an identical repeat selection must be served from cache, not recomputed');
  assert.equal(fetchCount, fetchCountBeforeSecond, 'a cached snapshot revisit issues zero new fetches');

  // A distinct (non-cached) selection on a page URL containing "hold" -- the mock fetch
  // falls back to metadata.pageUrl to key held responses when there is no cacheKey (snapshots
  // never set one). This proves an in-flight snapshot occupies a real parallel slot rather
  // than running out-of-band.
  const heldPageUrl = `https://example.test/${tag}-hold/manga`;
  const heldSnapshotMessage = {
    kind: 'translateSnapshot',
    pageUrl: heldPageUrl,
    dimensions: { left: 999, top: 999, width: 5, height: 5, devicePixelRatio: 1 },
  };
  const heldSnapshot = send(heldSnapshotMessage, 1);
  await flush(6);
  const pageDuringSnapshot = send(pageMessage(tag, 1), 1);
  await flush(4);
  const s = await stats();
  assert.equal(s.activeRequests, 1, 'an in-flight snapshot occupies a parallel slot, blocking a page translation into the queue');
  assert.equal(s.queueLength, 1);

  if (heldResolvers.has(heldPageUrl)) resolvePage(heldPageUrl);
  await flush();
  const pageKey = `hold-${tag}-page-1|100x100`;
  if (heldResolvers.has(pageKey)) resolvePage(pageKey);
  await flush();
  await Promise.all([heldSnapshot, pageDuringSnapshot]);
}

// ---------------------------------------------------------------------------------------
// Scenario 10 -- overlay-only responses (translations array, no re-rendered image) must
// still be cached so a revisit is instant. (Bug E3: putCachedResult currently requires
// translatedImageDataUrl and silently drops translations-only results.)
// ---------------------------------------------------------------------------------------
async function scenarioOverlayOnlyCaching() {
  queueLimitSetting = 10;
  parallelLimitSetting = 2;
  const tag = 's10';
  const msg = pageMessage(tag, 1, { hold: false, translationsOnly: true });
  const first = await send(msg);
  assert.ok(!first.translatedImageDataUrl, 'this backend response intentionally carries no re-rendered image');
  assert.ok(Array.isArray(first.translations) && first.translations.length > 0);

  const fetchCountBeforeRevisit = fetchCount;
  const revisit = await send(msg);
  assert.equal(revisit.fromCache, true, 'an overlay-only translation must still be cached');
  assert.equal(fetchCount, fetchCountBeforeRevisit, 'revisiting an overlay-only cached page issues zero fetches');
}

// ---------------------------------------------------------------------------------------
// Scenario 11 -- regression for the drain-stall defect (the user's "queue 5 pages, only 2
// translate" report): a queued item that resolves via processTranslation's cache-hit early
// return -- not a real dispatch -- must still drain the queue behind it. dispatchTranslation
// releases its reserved slot on every settle path, but pre-fix only processTranslation's own
// finally (which a cache hit never reaches) triggered processQueue(), so items queued behind
// a cache-hit dispatch stalled forever with a free slot sitting idle.
//
// Constructed by toggling cacheLimitSetting to 0 right before X is re-requested (forcing
// getCachedResult to report a miss even though X's entry is still sitting in
// translationCache untouched) so it is genuinely pushed into requestQueue, then restoring the
// setting before a slot frees up -- so when processQueue dispatches X, processTranslation's
// fresh getSettings()/getCachedResult() call now sees a real hit.
// ---------------------------------------------------------------------------------------
async function scenarioCacheHitDispatchDrain() {
  queueLimitSetting = 10;
  parallelLimitSetting = 1;
  const tag = 's11';

  // 1. Get page X fully cached under normal conditions.
  const xMsg = pageMessage(tag, 'x', { hold: false });
  const first = await send(xMsg);
  assert.equal(first.fromCache, undefined, 'first run for X is a real dispatch');
  assert.ok(first.translatedImageDataUrl);

  // 2. Occupy the single parallel slot with a held page A.
  const heldA = send(pageMessage(tag, 'a'));
  await flush(4);
  let s = await stats();
  assert.equal(s.activeRequests, 1);

  // 3. Force a genuine miss for X's cache lookup at enqueue time (translationCache itself is
  //    untouched), then re-request X. With slot A occupied, this must land in requestQueue.
  cacheLimitSetting = 0;
  const xRequeue = send(xMsg);
  await flush(4);
  cacheLimitSetting = DEFAULT_CACHE;

  // 4. Queue two more pages behind X while the slot is still occupied.
  const heldC = send(pageMessage(tag, 'c'));
  await flush(4);
  const heldD = send(pageMessage(tag, 'd'));
  await flush(4);

  s = await stats();
  assert.equal(s.activeRequests, 1, 'A still occupies the only slot');
  assert.equal(s.queueLength, 3, 'X, C, D are all genuinely queued');

  // 5. Free the only slot. processQueue dispatches X first (FIFO); with cacheLimitSetting
  //    restored, processTranslation now serves it from the cache it never actually left --
  //    this early-return settle must still drain C and D behind it.
  resolvePage(`hold-${tag}-page-a|100x100`);
  await flush();

  const xResult = await xRequeue;
  assert.equal(xResult.fromCache, true, 'X resolves via the cache-hit early return, not a real dispatch');

  s = await stats();
  assert.equal(s.activeRequests, 1, 'C was promoted into the freed slot');
  assert.equal(s.queueLength, 1, 'D is queued behind C, not stalled with a free slot sitting idle');

  resolvePage(`hold-${tag}-page-c|100x100`);
  await flush();
  resolvePage(`hold-${tag}-page-d|100x100`);
  await flush();
  await heldA;
  await heldC;
  await heldD;

  s = await stats();
  assert.equal(s.activeRequests, 0);
  assert.equal(s.queueLength, 0);
}

// ---------------------------------------------------------------------------------------
// Scenario 12 -- a hung backend request must not hold its parallel slot forever. The mock
// fetch's held-request promise only settles when resolvePage() is called by name, so a
// genuinely stuck request is never resolved here -- instead the real per-request abort timer
// (fetchTimeoutSetting, set well below the real default so the test doesn't actually wait
// minutes) must fire on its own and free the slot for the queued item behind it.
// ---------------------------------------------------------------------------------------
async function scenarioFetchTimeoutSlotRelease() {
  queueLimitSetting = 10;
  parallelLimitSetting = 1;
  fetchTimeoutSetting = 60;
  const tag = 's12';

  const stuckPromise = send(pageMessage(tag, 'stuck'));
  await flush(4);
  let s = await stats();
  assert.equal(s.activeRequests, 1, 'the stuck request occupies the only slot');

  const queuedPromise = send(pageMessage(tag, 'b', { hold: false }));
  await flush(4);
  s = await stats();
  assert.equal(s.activeRequests, 1);
  assert.equal(s.queueLength, 1, 'B queues behind the still-pending stuck request');

  // The abort timer is a real setTimeout, not something flush()'s fake-tick loop can
  // fast-forward -- wait past it for real.
  await new Promise((resolve) => setTimeout(resolve, 150));
  await flush();

  const stuckResult = await stuckPromise;
  assert.equal(stuckResult.error, 'PIPELINE_TIMEOUT', 'the hung request reports a distinct timeout error, not a generic failure');

  const bResult = await queuedPromise;
  assert.ok(bResult.translatedImageDataUrl, 'B was freed and dispatched once the stuck slot released');

  s = await stats();
  assert.equal(s.activeRequests, 0);
  assert.equal(s.queueLength, 0);
}

// ---------------------------------------------------------------------------------------
// Scenario 13 -- an unset cache-limit setting (fresh profile, never touched the dropdown)
// must fall back to the same default background.js and popup.js both agree on. A prior
// mismatch (background 24, popup 12, and 24 wasn't even a selectable dropdown option) meant
// a fresh profile's status line and dropdown permanently disagreed with each other.
// ---------------------------------------------------------------------------------------
async function scenarioDefaultCacheLimit() {
  cacheLimitSetting = undefined;
  const s = await stats();
  assert.equal(s.cacheLimit, 24, 'an unset cache-limit setting falls back to 24, matching popup.js\'s own default');
}

// ---------------------------------------------------------------------------------------
const scenarios = [
  ['1_user_walkthrough', scenarioUserWalkthrough],
  ['2_parallel_burst_E1', scenarioParallelBurst],
  ['3_drain_over_dispatch', scenarioDrainOverDispatch],
  ['4_queue_full', scenarioQueueFull],
  ['5_in_flight_dedupe', scenarioInFlightDedupe],
  ['6_lru_retention', scenarioLruRetention],
  ['7_pause_stop_under_load', scenarioPauseStopUnderLoad],
  ['8_limit_changes_mid_flight', scenarioLimitChangesMidFlight],
  ['9_selection_snapshot_E2', scenarioSelectionSnapshot],
  ['10_overlay_only_caching_E3', scenarioOverlayOnlyCaching],
  ['11_cache_hit_dispatch_drain', scenarioCacheHitDispatchDrain],
  ['12_fetch_timeout_slot_release', scenarioFetchTimeoutSlotRelease],
  ['13_default_cache_limit', scenarioDefaultCacheLimit],
];

const results = [];
for (const [name, fn] of scenarios) {
  await resetState();
  try {
    await fn();
    results.push([name, 'pass', null]);
    console.log(`[queue_cache_stress] ${name}: pass`);
  } catch (error) {
    results.push([name, 'fail', error]);
    console.log(`[queue_cache_stress] ${name}: FAIL -- ${error.message}`);
  }
}

const failed = results.filter(([, status]) => status === 'fail');
console.log(`\nqueue_cache_stress_test summary: ${results.length - failed.length}/${results.length} passed`);
if (failed.length > 0) {
  console.log('Failed scenarios:', failed.map(([name]) => name).join(', '));
  process.exitCode = 1;
} else {
  console.log('queue_cache_stress_test=pass');
}
