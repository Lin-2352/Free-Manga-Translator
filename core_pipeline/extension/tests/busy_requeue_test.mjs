// Regression test for the dropped-page bug: a page that received PIPELINE_BUSY (the backend's
// GPU-slot scheduler timed out waiting for a slot) was previously handed straight back to
// content.js, whose own node-anchored retry silently no-ops once a single-<img> page viewer's
// node has moved on to a later, already-translated page -- the page was never translated at
// all. background.js now retries PIPELINE_BUSY internally (requeueIfBusy(), re-entering its own
// src-keyed queue at the head) before ever returning the error, and clamps its own dispatch
// concurrency to the backend's advertised capacity so the deterministic-503 condition (2
// concurrent requests against a capacity-1 backend, given a page takes longer than the
// backend's own wait timeout) is avoided in the first place.
//
// This test drives the real message-handling code path, mirroring cache_gallery_limit_test.mjs's
// pattern. Real setTimeout delays are used (not faked) to keep this a genuine end-to-end proof
// of the production backoff timing, not a proof of a mocked substitute -- expect this file to
// take several real seconds to run.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const backgroundPath = path.join(extensionRoot, 'background.js');
const source = fs.readFileSync(backgroundPath, 'utf8');

function buildSandbox({ parallelLimitSetting = 2 } = {}) {
  let messageListener;
  let fetchCount = 0;
  const busyUntilAttempt = new Map(); // cacheKey substring -> attempt number it starts succeeding on
  let healthScheduler = null; // set per-test to control /v1/health's scheduler.capacity
  const heldFetchResolvers = []; // for holdNeedle: fetches that must stay open until released

  const sandbox = {
    console,
    URL,
    setTimeout,
    clearTimeout,
    AbortController,
    chrome: {
      storage: {
        local: {
          get: async () => ({
            localPipelineUrl: 'http://127.0.0.1:8766/v1/translate-image',
            localPipelineLanguage: 'ja',
            translationCachePages: 20,
            translationQueuePages: 40,
            translationParallelPages: parallelLimitSetting,
          }),
          set: async () => {},
        },
        session: { get: async () => ({}), set: async () => {}, remove: async () => {} },
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
      if (urlStr.includes('/v1/health')) {
        return {
          ok: true,
          json: async () => ({ ok: true, scheduler: healthScheduler }),
        };
      }
      if (urlStr.includes('/v1/warmup')) return { ok: true, json: async () => ({ ok: true, status: 'warming' }) };
      fetchCount += 1;
      const body = options.body ? JSON.parse(options.body) : null;
      const cacheKey = body?.metadata?.cacheKey || '';
      if (cacheKey.includes('holdopen')) {
        return new Promise((resolve) => {
          heldFetchResolvers.push(() => resolve({
            ok: true,
            json: async () => ({
              translatedImageDataUrl: `data:image/png;base64,${Buffer.from(cacheKey).toString('base64')}`,
              report: { pipeline: 'local-8-stage', stageSequence: [] },
              translations: [],
            }),
          }));
        });
      }
      for (const [needle, startsSucceedingOnAttempt] of busyUntilAttempt.entries()) {
        if (!cacheKey.includes(needle)) continue;
        const attemptForThisKey = (sandbox.__attemptCounts.get(needle) || 0) + 1;
        sandbox.__attemptCounts.set(needle, attemptForThisKey);
        if (attemptForThisKey < startsSucceedingOnAttempt) {
          return {
            ok: false,
            status: 503,
            json: async () => ({ detail: { code: 'SCHEDULER_BUSY', message: 'busy', traceId: 'x' } }),
          };
        }
        break;
      }
      return {
        ok: true,
        json: async () => ({
          translatedImageDataUrl: `data:image/png;base64,${Buffer.from(cacheKey).toString('base64')}`,
          report: { pipeline: 'local-8-stage', stageSequence: [], scheduler: { capacityAtAcquire: 1 } },
          translations: [],
        }),
      };
    },
  };
  sandbox.__attemptCounts = new Map();

  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename: backgroundPath });
  assert.equal(typeof messageListener, 'function', 'background message listener registered');

  function send(message) {
    return new Promise((resolve) => { messageListener(message, { tab: { id: 1 } }, resolve); });
  }

  return {
    send,
    setBusyUntilAttempt: (needle, attempt) => busyUntilAttempt.set(needle, attempt),
    setHealthScheduler: (scheduler) => { healthScheduler = scheduler; },
    getFetchCount: () => fetchCount,
    releaseOneHeld: () => heldFetchResolvers.shift()?.(),
    heldCount: () => heldFetchResolvers.length,
  };
}

// ===== Test 1: succeeds on the 2nd attempt -> final response is a SUCCESS, not PIPELINE_BUSY =====
{
  const env = buildSandbox();
  env.setBusyUntilAttempt('recovers', 2); // 1st call busy, 2nd call (the requeue) succeeds
  const response = await env.send({
    kind: 'translateImage',
    base64Data: 'data:image/png;base64,ZmFrZQ==',
    cacheKey: 'https://example.test/recovers.jpg|100x120',
    originalImageUrl: 'https://example.test/recovers.jpg',
    pageCacheKey: 'https://example.test/manga',
    pageUrl: 'https://example.test/manga',
    width: 100,
    height: 120,
  });
  assert.equal(response.error, undefined, `expected a successful response, got error=${response.error}`);
  assert.equal(
    typeof response.translatedImageDataUrl,
    'string',
    'a page that recovers after one busy response must end up translated, not silently dropped',
  );
  console.log('[busy_requeue] test1 (recovers on 2nd attempt) PASS');
}

// ===== Test 2: always busy -> exhausts at MAX_BUSY_REQUEUE_ATTEMPTS + 1 total fetches, =====
// ===== terminal response is PIPELINE_BUSY (so content.js still badges a truly wedged backend) =====
{
  const env = buildSandbox();
  env.setBusyUntilAttempt('neverrecovers', Infinity);
  const before = env.getFetchCount();
  const response = await env.send({
    kind: 'translateImage',
    base64Data: 'data:image/png;base64,ZmFrZQ==',
    cacheKey: 'https://example.test/neverrecovers.jpg|100x120',
    originalImageUrl: 'https://example.test/neverrecovers.jpg',
    pageCacheKey: 'https://example.test/manga',
    pageUrl: 'https://example.test/manga',
    width: 100,
    height: 120,
  });
  const attempts = env.getFetchCount() - before;
  assert.equal(response.error, 'PIPELINE_BUSY', 'a backend that never recovers must still terminate as PIPELINE_BUSY, not hang forever');
  assert.equal(attempts, 4, `expected exactly 1 initial + 3 requeue attempts (4 total fetches), got ${attempts}`);
  console.log('[busy_requeue] test2 (never recovers, bounded to 4 fetches) PASS');
}

// ===== Test 3a: known capacity=1 clamps 3 concurrent requests down to 1 in-flight fetch at a time =====
{
  const env = buildSandbox({ parallelLimitSetting: 3 });
  env.setHealthScheduler({ capacity: 1 });
  await env.send({ kind: 'checkPipelineHealth' });

  // Each fetch is held open (never resolves) until explicitly released, so fetchCount at a
  // given instant is a true "currently in flight" count, not an artifact of fast synchronous
  // resolution letting queued items cycle through within the observation window.
  const sends = [1, 2, 3].map((n) => env.send({
    kind: 'translateImage',
    base64Data: 'data:image/png;base64,ZmFrZQ==',
    cacheKey: `https://example.test/holdopen-capacitytest-${n}.jpg|100x120`,
    originalImageUrl: `https://example.test/holdopen-capacitytest-${n}.jpg`,
    pageCacheKey: 'https://example.test/manga',
    pageUrl: 'https://example.test/manga',
    width: 100,
    height: 120,
  }));
  await new Promise((resolve) => setTimeout(resolve, 20));
  const inFlightFetches = env.getFetchCount();
  assert.equal(inFlightFetches, 1, `capacity=1 must clamp dispatch to 1 concurrent fetch even though parallelLimit=3, got ${inFlightFetches}`);
  console.log('[busy_requeue] test3a (capacity=1 clamps 3->1 concurrent) PASS');
  // Drain the held requests so this test doesn't leak pending state into the next one.
  while (env.heldCount() > 0) {
    env.releaseOneHeld();
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  await Promise.all(sends);
}

// ===== Test 3b: unknown capacity (plain {ok:true} health, no scheduler field) falls back to =====
// ===== today's exact behavior -- parallelLimit alone governs concurrency =====
{
  const env = buildSandbox({ parallelLimitSetting: 3 });
  // No setHealthScheduler call -- healthScheduler stays null/undefined, matching every
  // existing test's health mock shape ({ok:true} with no scheduler key).
  await env.send({ kind: 'checkPipelineHealth' });
  const sends = [1, 2, 3].map((n) => env.send({
    kind: 'translateImage',
    base64Data: 'data:image/png;base64,ZmFrZQ==',
    cacheKey: `https://example.test/holdopen-fallbacktest-${n}.jpg|100x120`,
    originalImageUrl: `https://example.test/holdopen-fallbacktest-${n}.jpg`,
    pageCacheKey: 'https://example.test/manga',
    pageUrl: 'https://example.test/manga',
    width: 100,
    height: 120,
  }));
  await new Promise((resolve) => setTimeout(resolve, 20));
  const inFlightFetches = env.getFetchCount();
  assert.equal(inFlightFetches, 3, `unknown backend capacity must fall back to the configured parallelLimit (3), got ${inFlightFetches}`);
  console.log('[busy_requeue] test3b (unknown capacity falls back to configured parallelLimit) PASS');
  while (env.heldCount() > 0) {
    env.releaseOneHeld();
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  await Promise.all(sends);
}

console.log('busy_requeue=pass');
