// Proves background.js's circuit breaker for an unreachable local pipeline backend: a
// fetch()-level network failure (TypeError -- connection refused / offline / DNS fail) trips the
// breaker so subsequent dispatches short-circuit instead of each re-waiting out
// settings.fetchTimeoutMs; a periodic probe lets exactly one real attempt through to detect
// recovery; and none of this bypasses the queue-persistence/keep-alive-alarm invariants already
// hardened elsewhere this session.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const backgroundPath = path.join(extensionRoot, 'background.js');
const source = fs.readFileSync(backgroundPath, 'utf8');

let messageListener;
let mockNow = 1_000_000;
let backendUp = false;
let translateFetchCount = 0;
const sessionSetCalls = [];
const alarmCalls = { create: 0, clear: 0 };
let alarmActive = false;

class MockDate {
  static now() {
    return mockNow;
  }
}

const sandbox = {
  console,
  URL,
  Date: MockDate,
  setTimeout,
  clearTimeout,
  AbortController,
  chrome: {
    storage: {
      local: {
        get: async () => ({
          localPipelineUrl: 'http://127.0.0.1:8766/v1/translate-image',
          localPipelineLanguage: 'ja',
          translationCachePages: 12,
          translationQueuePages: 10,
          translationParallelPages: 1,
        }),
        set: async () => {},
      },
      session: {
        get: async () => ({}),
        set: async (entries) => {
          sessionSetCalls.push(entries);
        },
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
      sendMessage: async () => ({ ok: true }),
      captureVisibleTab: () => {},
      getZoom: async () => 1,
    },
    scripting: { executeScript: async () => [] },
    action: { setIcon: () => {} },
    alarms: {
      create: () => { alarmCalls.create += 1; alarmActive = true; },
      clear: () => { alarmCalls.clear += 1; alarmActive = false; return Promise.resolve(true); },
      onAlarm: { addListener: () => {} },
    },
  },
  fetch: async (url, options = {}) => {
    const urlStr = String(url);
    if (urlStr.includes('/v1/diagnostics/log')) return { ok: true, json: async () => ({ ok: true }) };
    if (urlStr.includes('/v1/health')) return { ok: true, json: async () => ({ ok: true }) };
    if (urlStr.includes('/v1/warmup')) return { ok: true, json: async () => ({ ok: true, status: 'warming' }) };
    // The real translate-image POST -- the only call site whose rejection the breaker classifies.
    translateFetchCount += 1;
    if (!backendUp) {
      throw new TypeError('Failed to fetch');
    }
    return {
      ok: true,
      json: async () => ({
        translatedImageDataUrl: 'data:image/png;base64,ZmFrZQ==',
        report: { pipeline: 'local-8-stage', stageSequence: [] },
        translations: [],
      }),
    };
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: backgroundPath });
assert.equal(typeof messageListener, 'function', 'background message listener registered');

function send(message, tabId = 1) {
  return new Promise((resolve) => {
    messageListener(message, { tab: { id: tabId } }, resolve);
  });
}

function pageMessage(n) {
  return {
    kind: 'translateImage',
    base64Data: 'data:image/png;base64,ZmFrZQ==',
    cacheKey: `page-${n}|100x100`,
    originalImageUrl: `https://example.test/page-${n}.jpg`,
    pageCacheKey: 'https://example.test/manga',
    pageUrl: 'https://example.test/manga',
    width: 100,
    height: 100,
  };
}

// (a) First attempt while down: real fetch happens, response is the normalized PIPELINE_OFFLINE
// code (not a raw "Failed to fetch"), breaker is now open.
const first = await send(pageMessage(1));
assert.equal(translateFetchCount, 1, 'first offline attempt actually touches the network');
assert.equal(first.error, 'PIPELINE_OFFLINE', 'tripping request reports the normalized offline code');

// (b) Second attempt immediately after: short-circuits, no network call.
const second = await send(pageMessage(2));
assert.equal(translateFetchCount, 1, 'second attempt while breaker is open does not touch the network');
assert.equal(second.error, 'PIPELINE_OFFLINE');

// (c) Still within the probe interval: still short-circuits.
mockNow += 5_000; // BREAKER_PROBE_INTERVAL_MS is 12_000
const third = await send(pageMessage(3));
assert.equal(translateFetchCount, 1, 'attempt inside the probe interval still short-circuits');
assert.equal(third.error, 'PIPELINE_OFFLINE');

// (d) Past the probe interval, backend still down: exactly one real probe attempt is let through.
mockNow += 8_000; // now 13s past the trip -- past the 12s interval
const probe = await send(pageMessage(4));
assert.equal(translateFetchCount, 2, 'probe attempt after the interval touches the network exactly once');
assert.equal(probe.error, 'PIPELINE_OFFLINE', 'failed probe reports offline again');

const probeFollowup = await send(pageMessage(5));
assert.equal(translateFetchCount, 2, 'immediately after a failed probe, short-circuit resumes');
assert.equal(probeFollowup.error, 'PIPELINE_OFFLINE');

// (e) Backend recovers; once the next probe window opens, a real attempt succeeds and the
// breaker closes.
backendUp = true;
mockNow += 12_000;
const recovered = await send(pageMessage(6));
assert.equal(translateFetchCount, 3, 'recovery probe touches the network');
assert.equal(recovered.translatedImageDataUrl, 'data:image/png;base64,ZmFrZQ==', 'recovery probe succeeds');

// (f) Breaker is closed now -- the very next dispatch goes straight to the network without
// needing to wait for another probe window.
const afterRecovery = await send(pageMessage(7));
assert.equal(translateFetchCount, 4, 'closed breaker dispatches normally, no further probe wait');
assert.equal(afterRecovery.translatedImageDataUrl, 'data:image/png;base64,ZmFrZQ==');

// (g) Queue-bookkeeping check: with the breaker open and parallelLimit=1, a backlog of
// short-circuited items must still drain cleanly through the real queue plumbing -- persisted
// descriptors written on every real mutation, keep-alive alarm cleared once fully drained.
backendUp = false;
mockNow += 20_000; // safely past another probe window so the burst below opens the breaker again
sessionSetCalls.length = 0;
alarmCalls.create = 0;
alarmCalls.clear = 0;

const burst = [pageMessage(8), pageMessage(9), pageMessage(10)].map((message) => send(message));
await Promise.all(burst);

const statsAfterBurst = await send({ kind: 'getTranslationStats' });
assert.equal(statsAfterBurst.activeRequests, 0, 'no active requests left stuck after a short-circuited burst');
assert.equal(statsAfterBurst.queueLength, 0, 'no queued requests left stuck after a short-circuited burst');
assert.equal(statsAfterBurst.pipelineBreakerOpen, true, 'stats expose the breaker state for the popup');
assert.equal(sessionSetCalls.length > 0, true, 'queue descriptor persistence still fires on real queue mutations while the breaker is open');
assert.equal(alarmActive, false, 'keep-alive alarm is not left dangling once a short-circuited backlog fully drains');

// (h) A successful health check alone -- no translate dispatch involved -- must also close the
// breaker. Before this, resetBreaker() was only ever called from the translate path, so a
// breaker opened by one transient failure stayed open (per stats) even after /v1/health
// confirmed the backend was reachable again, until a translate happened to land in a probe
// window or came along at all. This is the exact mechanism behind the popup's "green for a
// moment, then flips red" bug: checkPipelineHealth() succeeds, but a stale open breaker (read
// separately by refreshStats) still reported true.
backendUp = true; // health fetch always returns ok in this sandbox regardless of this flag,
// but flip it anyway so a subsequent dispatch (below) exercises the real network path too.
const healthCheck = await send({ kind: 'checkPipelineHealth' });
assert.equal(healthCheck.ok, true, 'health check itself succeeds');
const statsAfterHealthCheck = await send({ kind: 'getTranslationStats' });
assert.equal(
  statsAfterHealthCheck.pipelineBreakerOpen,
  false,
  'a successful health check closes the breaker on its own, with no translate dispatch required',
);

console.log('extension_pipeline_breaker=pass');
