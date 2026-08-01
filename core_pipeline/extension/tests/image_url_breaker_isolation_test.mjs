// Proves background.js distinguishes "the target image URL itself is broken/unreachable" (a
// message with imageUrl, requiring background.js to fetch it via fetchImageAsDataUrl) from "the
// local pipeline backend is unreachable" -- both can surface as a fetch()-level TypeError, but
// only the latter may legitimately trip the shared circuit breaker. A single dead image src on a
// page must fail just that one image without tripping the breaker and short-circuiting every
// OTHER (perfectly fine) image on the page with a false PIPELINE_OFFLINE.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const backgroundPath = path.join(extensionRoot, 'background.js');
const source = fs.readFileSync(backgroundPath, 'utf8');

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
        get: async () => ({
          localPipelineUrl: 'http://127.0.0.1:8766/v1/translate-image',
          localPipelineLanguage: 'ja',
          translationCachePages: 12,
          translationQueuePages: 10,
          translationParallelPages: 2,
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
      sendMessage: async () => ({ ok: true }),
      captureVisibleTab: () => {},
      getZoom: async () => 1,
    },
    scripting: { executeScript: async () => [] },
    action: { setIcon: () => {} },
    alarms: {
      create: () => {},
      clear: () => Promise.resolve(true),
      onAlarm: { addListener: () => {} },
    },
  },
  fetch: async (url) => {
    const urlStr = String(url);
    if (urlStr.includes('/v1/diagnostics/log')) return { ok: true, json: async () => ({ ok: true }) };
    if (urlStr.includes('translate-image')) {
      // The backend endpoint itself is healthy in this test -- only the IMAGE fetch fails.
      return {
        ok: true,
        json: async () => ({
          translatedImageDataUrl: 'data:image/png;base64,ZmFrZQ==',
          report: { pipeline: 'local-8-stage', stageSequence: [] },
          translations: [],
        }),
      };
    }
    // Any other URL is treated as the target image -- simulate a broken/unreachable image src
    // (DNS failure, connection refused, ...): a real fetch()-level TypeError.
    throw new TypeError('Failed to fetch');
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

function brokenImageMessage(n) {
  return {
    kind: 'translateImage',
    imageUrl: `https://broken.example.test/page-${n}.jpg`,
    cacheKey: `broken-page-${n}|100x100`,
    originalImageUrl: `https://broken.example.test/page-${n}.jpg`,
    pageCacheKey: 'https://example.test/manga',
    pageUrl: 'https://example.test/manga',
    width: 100,
    height: 100,
  };
}

// (a) A broken image URL fails on its own -- the error is NOT normalized to PIPELINE_OFFLINE
// (that would falsely tell content.js/the popup the whole backend is down).
const brokenResult = await send(brokenImageMessage(1));
assert.ok(brokenResult.error, 'broken image URL surfaces an error');
assert.notEqual(brokenResult.error, 'PIPELINE_OFFLINE', 'a broken image URL is not misreported as a backend outage');

// (b) The breaker must NOT have tripped -- stats still report it closed, and a real translateImage
// request for a DIFFERENT, working image is dispatched normally (not short-circuited).
const statsAfterBrokenImage = await send({ kind: 'getTranslationStats' });
assert.equal(statsAfterBrokenImage.pipelineBreakerOpen, false, 'a broken image URL never trips the shared circuit breaker');

const workingResult = await send({
  kind: 'translateImage',
  base64Data: 'data:image/png;base64,ZmFrZQ==',
  cacheKey: 'working-page|100x100',
  originalImageUrl: 'https://example.test/working-page.jpg',
  pageCacheKey: 'https://example.test/manga',
  pageUrl: 'https://example.test/manga',
  width: 100,
  height: 100,
});
assert.equal(
  workingResult.translatedImageDataUrl,
  'data:image/png;base64,ZmFrZQ==',
  'a different, healthy image translates normally right after the broken-image-URL failure -- proving the breaker was never falsely tripped',
);

console.log('extension_image_url_breaker_isolation=pass');
