// Regression test for the queue-loss-detection feature (Commit 4): a service-worker restart
// wipes requestQueue in memory, so the popup's next poll would otherwise show a lying "ready"
// even though pages the user queued never got translated. This does NOT test auto-recovery --
// there isn't any, by design (re-fetching pages the user may have already left would spend real
// backend/API quota on their behalf with no way to know if they still want it). It only tests
// that the loss is detected and reported truthfully exactly once, and that in-flight (already
// dispatched) requests are correctly excluded since those get content.js's own connection-error
// retry/badge treatment instead.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const backgroundPath = path.join(extensionRoot, 'background.js');
const source = fs.readFileSync(backgroundPath, 'utf8');

function respondFor() {
  return {
    translatedImageDataUrl: 'data:image/png;base64,ZmFrZQ==',
    report: { pipeline: 'local-8-stage', stageSequence: [] },
    translations: [],
  };
}

// A real chrome.storage.session survives a service-worker restart (that's the whole premise);
// two separate contexts sharing this one plain object is the minimal faithful simulation of that.
function makeContext(sessionStore) {
  let messageListener;
  const heldResolvers = new Map();

  async function mockFetch(url, options = {}) {
    const urlStr = String(url);
    if (urlStr.includes('/v1/health') || urlStr.includes('/v1/warmup') || urlStr.includes('/v1/diagnostics/log')) {
      return { ok: true, json: async () => ({ ok: true }) };
    }
    const body = options.body ? JSON.parse(options.body) : null;
    const cacheKey = body?.metadata?.cacheKey || '';
    if (cacheKey.includes('hold')) {
      return new Promise((resolve, reject) => {
        heldResolvers.set(cacheKey, () => resolve({ ok: true, json: async () => respondFor() }));
        options.signal?.addEventListener('abort', () => {
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
            translationCachePages: 24,
            translationQueuePages: 10,
            translationParallelPages: 1,
            translationFetchTimeoutMs: 360000,
          }),
          set: async () => {},
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

  function send(message) {
    return new Promise((resolve) => messageListener(message, { tab: { id: 1 } }, resolve));
  }
  async function flush(times = 20) {
    for (let i = 0; i < times; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
  }
  function pageMessage(n, { hold = true } = {}) {
    const cacheKey = `${hold ? 'hold-' : ''}restart-page-${n}|100x100`;
    return {
      kind: 'translateImage',
      base64Data: 'data:image/png;base64,ZmFrZQ==',
      cacheKey,
      originalImageUrl: `https://example.test/restart/${n}.jpg`,
      pageCacheKey: 'https://example.test/restart/manga',
      pageUrl: 'https://example.test/restart/manga',
      width: 100,
      height: 100,
    };
  }

  return { send, flush, pageMessage, heldResolvers };
}

async function main() {
  const sessionStore = {};

  // --- "Session 1": queue 3 pages with parallelLimit=1. Page 1 dispatches immediately (held,
  // simulating an in-flight backend call); pages 2 and 3 sit in requestQueue, never dispatched.
  const ctx1 = makeContext(sessionStore);
  ctx1.send(ctx1.pageMessage(1)); // dispatches, holds forever -- simulates in-flight work
  await ctx1.flush();
  ctx1.send(ctx1.pageMessage(2)); // queues
  ctx1.send(ctx1.pageMessage(3)); // queues
  await ctx1.flush();

  const statsBeforeRestart = await ctx1.send({ kind: 'getTranslationStats' });
  assert.equal(statsBeforeRestart.activeRequests, 1, 'one page actually dispatched (in-flight)');
  assert.equal(statsBeforeRestart.queueLength, 2, 'two pages sitting in requestQueue, never dispatched');
  assert.ok(
    Array.isArray(sessionStore.fmtQueueDescriptorsV1) && sessionStore.fmtQueueDescriptorsV1.length === 2,
    'only the genuinely-queued (never-dispatched) items are persisted -- the in-flight one is excluded',
  );

  // --- Simulate the service worker dying: ctx1's entire in-memory state (requestQueue,
  // outgoingRequests, everything) is gone. sessionStore is the one thing that survives, exactly
  // like real chrome.storage.session across a real SW restart.
  const ctx2 = makeContext(sessionStore);
  await ctx2.flush();

  const statsAfterRestart = await ctx2.send({ kind: 'getTranslationStats' });
  assert.equal(statsAfterRestart.restartLost, 2, 'the 2 never-dispatched pages are honestly reported as lost');
  assert.equal(statsAfterRestart.activeRequests, 0, 'the new instance starts with a genuinely empty queue');
  assert.equal(statsAfterRestart.queueLength, 0);
  assert.deepEqual(sessionStore, {}, 'the descriptor key is cleared after being read once, so it is not double-counted');

  // A third instantiation with nothing new queued must NOT re-report the same loss.
  const ctx3 = makeContext(sessionStore);
  const statsCtx3 = await ctx3.send({ kind: 'getTranslationStats' });
  assert.equal(statsCtx3.restartLost, 0, 'restartLost does not resurface once already consumed and cleared');

  // The popup's restart-lost message explicitly instructs "re-run Translate Page" -- that re-run
  // must clear the stale warning on its own, not only via the separate Clear Queue button.
  const sessionStoreForRerun = { fmtQueueDescriptorsV1: ['lost-a', 'lost-b'] };
  const ctx4 = makeContext(sessionStoreForRerun);
  const statsBeforeRerun = await ctx4.send({ kind: 'getTranslationStats' });
  assert.equal(statsBeforeRerun.restartLost, 2, 'rerun-test setup: loss detected on this fresh instance');
  ctx4.send(ctx4.pageMessage(6, { hold: false })); // resolves immediately, no cleanup needed
  await ctx4.flush();
  const statsAfterRerun = await ctx4.send({ kind: 'getTranslationStats' });
  assert.equal(
    statsAfterRerun.restartLost,
    0,
    'simply re-running Translate Page clears the stale restart-lost warning, exactly as the popup message instructs',
  );

  // Explicit user acknowledgement (Clear Queue) resets the counter within a still-running instance.
  ctx2.send(ctx2.pageMessage(4)); // dispatches, holds -- occupies the single parallel slot
  await ctx2.flush();
  ctx2.send(ctx2.pageMessage(5)); // queues behind it
  await ctx2.flush();
  await ctx2.send({ kind: 'clearQueue' });
  const statsAfterClear = await ctx2.send({ kind: 'getTranslationStats' });
  assert.equal(statsAfterClear.restartLost, 0, 'Clear Queue resets restartLost as an explicit acknowledgement');

  // Held ("in-flight forever") mock fetches are never resolved above -- their real fetch-timeout
  // setTimeout would otherwise keep the Node process alive for the full 360s. Hard-stop aborts
  // any still-active controllers so those timers clear before the process exits.
  await ctx1.send({ kind: 'stopTranslations', mode: 'hard' });
  await ctx2.send({ kind: 'stopTranslations', mode: 'hard' });
  await ctx1.flush();
  await ctx2.flush();

  console.log('[queue_persistence] all assertions passed');
  console.log('extension_queue_persistence=pass');
}

main().catch((error) => {
  console.error('[queue_persistence] FAIL --', error.message);
  process.exitCode = 1;
});
