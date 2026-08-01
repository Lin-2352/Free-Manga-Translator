// Regression test for the popup "recent translations" gallery undercounting bug.
//
// Before the fix: popup.js's refreshRecentTranslations() always requested exactly 6
// entries (chrome.runtime.sendMessage({ kind: 'getRecentTranslations', limit: 6 })),
// regardless of the user's configured cache size -- so translating e.g. 15 pages with
// cache kept at 15+ only ever showed the 6 most-recently-visited in the gallery strip.
// Separately, background.js's own handler clamped the limit to a hardcoded max of 12,
// so even a caller that correctly asked for more (as the fixed popup.js now does) would
// still be truncated for any cache size above 12 (the popup offers up to 40).
//
// This test drives the real message-handling code path (translateImage -> cache ->
// getRecentTranslations), not internals-poking, mirroring queue_cache_stress_test.mjs's
// pattern.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const backgroundPath = path.join(extensionRoot, 'background.js');
const source = fs.readFileSync(backgroundPath, 'utf8');

let messageListener;
const cacheLimitSetting = 40; // max selectable in popup.html's translationCachePages

async function mockFetch(url, options = {}) {
  const urlStr = String(url);
  if (urlStr.includes('/v1/diagnostics/log')) return { ok: true, json: async () => ({ ok: true }) };
  if (urlStr.includes('/v1/health')) return { ok: true, json: async () => ({ ok: true }) };
  if (urlStr.includes('/v1/warmup')) return { ok: true, json: async () => ({ ok: true, status: 'warming' }) };
  const body = options.body ? JSON.parse(options.body) : null;
  const cacheKey = body?.metadata?.cacheKey || '';
  return {
    ok: true,
    json: async () => ({
      // Distinct per-request payload so each cached entry is genuinely distinguishable,
      // not 15 copies of the same bytes collapsing into one cache entry.
      translatedImageDataUrl: `data:image/png;base64,${Buffer.from(`page:${cacheKey}`).toString('base64')}`,
      report: { pipeline: 'local-8-stage', stageSequence: [] },
      translations: [],
    }),
  };
}

const contentMessages = [];
let contentInjected = false;

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
          translationCachePages: cacheLimitSetting,
          translationQueuePages: 40,
          translationParallelPages: 4,
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
    tabs: {
      sendMessage: async (tabId, message) => {
        contentMessages.push({ tabId, message });
        if (message.kind === 'pingContentScript' && !contentInjected) {
          throw new Error('Receiving end does not exist.');
        }
        return { ok: true };
      },
      getZoom: async () => 1,
    },
    scripting: {
      executeScript: async (details) => {
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

async function flush(times = 10) {
  for (let i = 0; i < times; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
}

const TOTAL_PAGES = 15; // the user's real reported scenario

for (let n = 0; n < TOTAL_PAGES; n += 1) {
  await send({
    kind: 'translateImage',
    base64Data: 'data:image/png;base64,ZmFrZQ==',
    cacheKey: `gallery-page-${n}`,
    originalImageUrl: `https://example.test/manga/${n}.jpg`,
    pageCacheKey: 'https://example.test/manga',
    pageUrl: 'https://example.test/manga',
    width: 100,
    height: 100,
  });
}
await flush();

const statsResponse = await send({ kind: 'getTranslationStats' });
assert.equal(statsResponse.cacheSize, TOTAL_PAGES, `all ${TOTAL_PAGES} translations should be cached (cache limit is ${cacheLimitSetting})`);

// This mirrors the FIXED popup.js: it now requests the real configured cache limit
// instead of a hardcoded 6.
const galleryResponse = await send({ kind: 'getRecentTranslations', limit: statsResponse.cacheLimit });
assert.equal(
  galleryResponse.entries.length,
  TOTAL_PAGES,
  `gallery should show all ${TOTAL_PAGES} cached translations, got ${galleryResponse.entries.length} -- ` +
    'this is the regression: background.js previously hard-capped this at 12 regardless of what was requested',
);

// A caller asking for more than what's actually cached must not error or pad -- just
// return everything that exists.
const overAskResponse = await send({ kind: 'getRecentTranslations', limit: 999 });
assert.equal(overAskResponse.entries.length, TOTAL_PAGES, 'requesting more than cached entries returns exactly what exists');

// The old hardcoded-6 default must no longer be the ceiling when a caller omits limit
// entirely -- but a caller that truly doesn't pass one should still get the documented
// small default (6), not everything -- this documents the fallback contract explicitly
// so it isn't silently widened again by accident.
const noLimitResponse = await send({ kind: 'getRecentTranslations' });
assert.equal(noLimitResponse.entries.length, 6, 'omitting limit entirely still uses the documented default of 6');

console.log('cache_gallery_limit=pass');
