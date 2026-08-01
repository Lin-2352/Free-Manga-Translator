// Regression test for the intermittent wrong cached-count/thumbnail bug.
//
// Before Fix B: the cache persisted to chrome.storage.session under ONE key holding every
// entry. Large (~2.2MB measured) translated pages made that single write exceed the ~10MB
// session quota; persistCache() handled quota failures by halving the entry set until a write
// succeeded (10 -> 5 -> 2), silently leaving the PERSISTED store far smaller than the real
// in-memory cache -- and if it halved all the way to an empty set, the empty set itself
// persisted successfully, wiping the store outright. ensureCacheLoaded() also set its "loaded"
// flag BEFORE awaiting the load, so a second caller arriving mid-load read an empty Map. All of
// this made the popup's cached-entry count "sometimes accurate, sometimes not" depending purely
// on service-worker restart timing.
//
// Fix B moves to chrome.storage.local with the unlimitedStorage permission, one entry per
// storage key, and memoizes the in-flight load as a real promise.
//
// This test drives the real message-handling code path, mirroring cache_gallery_limit_test.mjs.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const backgroundPath = path.join(extensionRoot, 'background.js');
const source = fs.readFileSync(backgroundPath, 'utf8');

const CACHE_ENTRY_PREFIX = 'fmtCacheV1:';
const BIG_PAYLOAD = 'x'.repeat(2_200_000); // ~2.2MB, matching the real measured entry size

function makeStatefulLocalMock({ quotaBudgetBytes = Infinity, getDelayMs = 0 } = {}) {
  const store = new Map();
  let getCallCount = 0;
  const approxByteSize = (value) => Buffer.byteLength(JSON.stringify(value), 'utf8');
  const mock = {
    get: async (keys) => {
      getCallCount += 1;
      if (getDelayMs > 0) await new Promise((resolve) => setTimeout(resolve, getDelayMs));
      const out = {};
      if (keys === null || keys === undefined) {
        for (const [k, v] of store.entries()) out[k] = v;
        return out;
      }
      for (const k of Array.isArray(keys) ? keys : [keys]) if (store.has(k)) out[k] = store.get(k);
      return out;
    },
    set: async (obj) => {
      for (const [k, v] of Object.entries(obj)) {
        const size = approxByteSize({ [k]: v });
        const currentTotal = Array.from(store.entries())
          .filter(([existingKey]) => existingKey !== k)
          .reduce((sum, [, existingVal]) => sum + approxByteSize(existingVal), 0);
        if (currentTotal + size > quotaBudgetBytes) {
          const error = new Error('QUOTA_BYTES exceeded');
          error.name = 'QuotaExceededError';
          throw error;
        }
        store.set(k, v);
      }
    },
    remove: async (keys) => {
      for (const k of Array.isArray(keys) ? keys : [keys]) store.delete(k);
    },
  };
  return { mock, store, getGetCallCount: () => getCallCount };
}

function buildSandbox({ localMock, sessionSeed = null }) {
  let messageListener;
  const sandbox = {
    console,
    URL,
    setTimeout,
    clearTimeout,
    AbortController,
    chrome: {
      storage: {
        local: {
          get: localMock.get,
          set: localMock.set,
          remove: localMock.remove,
        },
        session: {
          get: async (keys) => {
            if (!sessionSeed) return {};
            const out = {};
            for (const k of Array.isArray(keys) ? keys : [keys]) if (k in sessionSeed) out[k] = sessionSeed[k];
            return out;
          },
          set: async () => {},
          remove: async (keys) => {
            if (!sessionSeed) return;
            for (const k of Array.isArray(keys) ? keys : [keys]) delete sessionSeed[k];
          },
        },
        onChanged: { addListener: () => {} },
      },
      runtime: {
        onMessage: { addListener: (listener) => { messageListener = listener; } },
        onInstalled: { addListener: () => {} },
      },
      contextMenus: { create: () => {}, onClicked: { addListener: () => {} } },
      tabs: { sendMessage: async () => ({ ok: true }), getZoom: async () => 1 },
      scripting: { executeScript: async () => [] },
      action: { setIcon: () => {} },
      alarms: { create: () => {}, clear: () => Promise.resolve(true), onAlarm: { addListener: () => {} } },
    },
    fetch: async (url, options = {}) => {
      const urlStr = String(url);
      if (urlStr.includes('/v1/diagnostics/log')) return { ok: true, json: async () => ({ ok: true }) };
      if (urlStr.includes('/v1/health')) return { ok: true, json: async () => ({ ok: true }) };
      if (urlStr.includes('/v1/warmup')) return { ok: true, json: async () => ({ ok: true, status: 'warming' }) };
      const body = options.body ? JSON.parse(options.body) : null;
      const cacheKey = body?.metadata?.cacheKey || '';
      return {
        ok: true,
        json: async () => ({
          translatedImageDataUrl: `data:image/png;base64,${BIG_PAYLOAD}`,
          report: { pipeline: 'local-8-stage', stageSequence: [] },
          translations: [],
        }),
      };
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename: backgroundPath });
  assert.equal(typeof messageListener, 'function', 'background message listener registered');
  return {
    send: (message) => new Promise((resolve) => { messageListener(message, { tab: { id: 1 } }, resolve); }),
  };
}

async function flush(times = 10) {
  for (let i = 0; i < times; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
}

// ===== Test 1: no truncation -- 10 large entries persist as 10 keys, not silently reduced =====
{
  const { mock, store } = makeStatefulLocalMock(); // Infinity budget: unlimitedStorage in effect
  const env = buildSandbox({ localMock: mock });
  for (let n = 0; n < 10; n += 1) {
    await env.send({
      kind: 'translateImage',
      base64Data: 'data:image/png;base64,ZmFrZQ==',
      cacheKey: `https://example.test/persist-page-${n}.jpg|100x120`,
      originalImageUrl: `https://example.test/persist-page-${n}.jpg`,
      pageCacheKey: 'https://example.test/manga',
      pageUrl: 'https://example.test/manga',
      width: 100,
      height: 120,
    });
  }
  await flush();
  const cacheKeys = Array.from(store.keys()).filter((k) => k.startsWith(CACHE_ENTRY_PREFIX));
  assert.equal(cacheKeys.length, 10, `expected all 10 entries persisted as separate keys, got ${cacheKeys.length}`);
  const stats = await env.send({ kind: 'getTranslationStats' });
  assert.equal(stats.cacheSize, 10, 'in-memory cache size must also be 10');
  console.log('[cache_persistence] test1 (no truncation, 10/10 persisted) PASS');
}

// ===== Test 2: terminal empty-set bug -- a write that exceeds quota must NOT wipe previously =====
// ===== persisted entries down to nothing. =====
{
  // In this sandbox (no OffscreenCanvas) generateThumbnailDataUrl falls back to the full image,
  // so a stored entry duplicates the ~2.2MB payload into both result.translatedImageDataUrl AND
  // thumbnail (~4.4MB/entry). Budget comfortably fits one such entry but not two.
  const { mock, store } = makeStatefulLocalMock({ quotaBudgetBytes: 6_000_000 });
  const env = buildSandbox({ localMock: mock });
  await env.send({
    kind: 'translateImage',
    base64Data: 'data:image/png;base64,ZmFrZQ==',
    cacheKey: 'https://example.test/first-page.jpg|100x120',
    originalImageUrl: 'https://example.test/first-page.jpg',
    pageCacheKey: 'https://example.test/manga',
    pageUrl: 'https://example.test/manga',
    width: 100,
    height: 120,
  });
  await flush();
  const afterFirst = Array.from(store.keys()).filter((k) => k.startsWith(CACHE_ENTRY_PREFIX));
  assert.equal(afterFirst.length, 1, 'the first entry (within budget) must persist');

  // A second entry that cannot fit must fail to persist WITHOUT deleting the first.
  await env.send({
    kind: 'translateImage',
    base64Data: 'data:image/png;base64,ZmFrZQ==',
    cacheKey: 'https://example.test/second-page.jpg|100x120',
    originalImageUrl: 'https://example.test/second-page.jpg',
    pageCacheKey: 'https://example.test/manga',
    pageUrl: 'https://example.test/manga',
    width: 100,
    height: 120,
  });
  await flush();
  const afterSecond = Array.from(store.keys()).filter((k) => k.startsWith(CACHE_ENTRY_PREFIX));
  assert.equal(
    afterSecond.length,
    1,
    `a persist failure for entry 2 must not touch entry 1's already-persisted key, got ${afterSecond.length} keys`,
  );
  console.log('[cache_persistence] test2 (quota failure does not wipe existing entries) PASS');
}

// ===== Test 3: in-flight load memoization -- two concurrent first-callers against a slow =====
// ===== storage.local.get must both see the fully-loaded cache, not a race where the second =====
// ===== reads an empty Map. =====
{
  const { mock, store } = makeStatefulLocalMock({ getDelayMs: 30 });
  // Pre-seed the store directly (bypassing the extension) to simulate a prior session's cache.
  const seededResult = { translatedImageDataUrl: 'data:image/png;base64,c2VlZGVk', translations: [] };
  const seededCacheId = `local-8-step-v13-quality-performance-hardening-v1.1.15:seeded-key`;
  store.set(CACHE_ENTRY_PREFIX + seededCacheId, {
    result: seededResult,
    thumbnail: null,
    lastUsed: Date.now(),
    pageHost: 'example.test',
  });
  const env = buildSandbox({ localMock: mock });

  // Fire two different first-touch messages concurrently -- both trigger ensureCacheLoaded()
  // before either's storage.local.get(null) has resolved.
  const [statsResponse, galleryResponse] = await Promise.all([
    env.send({ kind: 'getTranslationStats' }),
    env.send({ kind: 'getRecentTranslations', limit: 10 }),
  ]);
  assert.equal(statsResponse.cacheSize, 1, `getTranslationStats must see the seeded entry, got cacheSize=${statsResponse.cacheSize}`);
  assert.equal(
    galleryResponse.entries.length,
    1,
    `getRecentTranslations must see the seeded entry, got ${galleryResponse.entries.length} entries`,
  );
  console.log('[cache_persistence] test3 (concurrent first-callers both see the loaded cache) PASS');
}

// ===== Test 4: a cache HIT must not re-serialize the whole entry on every access =====
{
  const { mock, store } = makeStatefulLocalMock();
  const env = buildSandbox({ localMock: mock });
  await env.send({
    kind: 'translateImage',
    base64Data: 'data:image/png;base64,ZmFrZQ==',
    cacheKey: 'https://example.test/hit-page.jpg|100x120',
    originalImageUrl: 'https://example.test/hit-page.jpg',
    pageCacheKey: 'https://example.test/manga',
    pageUrl: 'https://example.test/manga',
    width: 100,
    height: 120,
  });
  await flush();

  let setCallsBeforeHit = 0;
  const originalSet = mock.set;
  mock.set = async (obj) => { setCallsBeforeHit += 1; return originalSet(obj); };

  // Second request for the SAME image (cache hit).
  await env.send({
    kind: 'translateImage',
    base64Data: 'data:image/png;base64,ZmFrZQ==',
    cacheKey: 'https://example.test/hit-page.jpg|100x120',
    originalImageUrl: 'https://example.test/hit-page.jpg',
    pageCacheKey: 'https://example.test/manga',
    pageUrl: 'https://example.test/manga',
    width: 100,
    height: 120,
  });
  await flush();
  assert.equal(setCallsBeforeHit, 0, `a fresh cache hit (lastUsed bump under the 60s threshold) must not write to storage, got ${setCallsBeforeHit} set() calls`);
  console.log('[cache_persistence] test4 (cache hit does not re-persist) PASS');
}

// ===== Test 5: migration from the legacy chrome.storage.session single-key store =====
{
  const { mock, store } = makeStatefulLocalMock();
  const legacyCacheId = 'local-8-step-v13-quality-performance-hardening-v1.1.15:legacy-key';
  const sessionSeed = {
    translationCacheEntries: {
      [legacyCacheId]: {
        result: { translatedImageDataUrl: 'data:image/png;base64,bGVnYWN5', translations: [] },
        thumbnail: null,
        lastUsed: Date.now(),
        pageHost: 'example.test',
      },
    },
  };
  const env = buildSandbox({ localMock: mock, sessionSeed });
  const stats = await env.send({ kind: 'getTranslationStats' });
  assert.equal(stats.cacheSize, 1, `legacy session entry must be migrated and visible, got cacheSize=${stats.cacheSize}`);
  const migratedKeys = Array.from(store.keys()).filter((k) => k.startsWith(CACHE_ENTRY_PREFIX));
  assert.equal(migratedKeys.length, 1, 'the migrated entry must land under the new per-entry local key');
  assert.equal(sessionSeed.translationCacheEntries, undefined, 'the legacy session key must be removed after a successful migration');
  console.log('[cache_persistence] test5 (session -> local migration) PASS');
}

console.log('cache_persistence_local=pass');
