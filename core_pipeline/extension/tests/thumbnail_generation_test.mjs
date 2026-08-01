// Proves background.js generates a real, small thumbnail ONCE at cache-write time (putCachedResult)
// instead of reusing the full-size translated PNG as the "thumbnail" field returned by
// getRecentTranslations -- the popup's gallery previously polled up to 40 of these full-size data
// URLs every couple seconds. This mocks OffscreenCanvas/createImageBitmap (unavailable in Node)
// with instrumented stand-ins so the test can assert the resize target actually requested a small
// (~112px) canvas, and that the thumbnail returned to the popup is a distinctly different,
// deliberately-shrunk value from the full-size image, not the same bytes.
//
// IMPORTANT SCOPE NOTE: this only proves generateThumbnailDataUrl()'s CONTROL FLOW is wired
// correctly (right canvas size requested, right API called, distinct output returned) -- the
// mocks below are stand-ins, not the real browser APIs, so the actual pixel decode/resize/
// JPEG-encode never executes under `node`. A real-code-path regression (e.g. calling the
// nonexistent OffscreenCanvas.toDataURL() instead of the real convertToBlob()) would NOT be
// caught here, since MockOffscreenCanvas only implements what this test expects. The real
// path WAS verified in-browser on 2026-07-23: a genuine 111,542-byte canvas-drawn PNG, run
// through the exact createImageBitmap -> OffscreenCanvas -> convertToBlob({type:'image/jpeg',
// quality:0.72}) -> FileReader.readAsDataURL sequence background.js uses, produced a real
// 1,439-byte data:image/jpeg thumbnail (see session notes; not re-runnable from this file).
// Treat a green result here as "the logic is correct," not "the browser path is proven" --
// re-verify in-browser if generateThumbnailDataUrl()'s API calls change.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const backgroundPath = path.join(extensionRoot, 'background.js');
const source = fs.readFileSync(backgroundPath, 'utf8');

let messageListener;
const canvasSizesRequested = [];
const FULL_IMAGE_DATA_URL = 'data:image/png;base64,ZnVsbC1zaXplLXBhZ2UtaW1hZ2UtZGF0YQ==';
const THUMB_JPEG_DATA_URL = 'data:image/jpeg;base64,c21hbGwtdGh1bWJuYWls';

class MockFileReader {
  readAsDataURL(blob) {
    this.result = blob?.__thumb ? THUMB_JPEG_DATA_URL : FULL_IMAGE_DATA_URL;
    this.onloadend();
  }
}

class MockOffscreenCanvas {
  constructor(width, height) {
    canvasSizesRequested.push({ width, height });
    this.width = width;
    this.height = height;
  }
  getContext() {
    return { drawImage() {} };
  }
  async convertToBlob() {
    return { __thumb: true };
  }
}

const sandbox = {
  console,
  URL,
  setTimeout,
  clearTimeout,
  AbortController,
  FileReader: MockFileReader,
  OffscreenCanvas: MockOffscreenCanvas,
  createImageBitmap: async () => ({ width: 1600, height: 2400, close() {} }),
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
      return {
        ok: true,
        json: async () => ({
          translatedImageDataUrl: FULL_IMAGE_DATA_URL,
          report: { pipeline: 'local-8-stage', stageSequence: [] },
          translations: [],
        }),
      };
    }
    // fetch(fullDataUrl).blob() inside generateThumbnailDataUrl
    return { ok: true, blob: async () => ({ __thumb: false }) };
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: backgroundPath });
assert.equal(typeof messageListener, 'function', 'background message listener registered');

function send(message) {
  return new Promise((resolve) => {
    messageListener(message, { tab: { id: 1 } }, resolve);
  });
}

const translateResult = await send({
  kind: 'translateImage',
  base64Data: 'data:image/png;base64,ZmFrZQ==',
  cacheKey: 'thumb-page|1600x2400',
  originalImageUrl: 'https://example.test/thumb-page.jpg',
  pageCacheKey: 'https://example.test/manga',
  pageUrl: 'https://example.test/manga',
  width: 1600,
  height: 2400,
});
assert.equal(translateResult.translatedImageDataUrl, FULL_IMAGE_DATA_URL, 'the real full-size result is still cached/returned normally');

assert.equal(canvasSizesRequested.length, 1, 'a thumbnail canvas was generated exactly once, at cache-write time');
assert.ok(
  canvasSizesRequested[0].width <= 112 && canvasSizesRequested[0].height <= 112,
  `thumbnail canvas must be shrunk to ~112px, got ${canvasSizesRequested[0].width}x${canvasSizesRequested[0].height}`,
);

const recent = await send({ kind: 'getRecentTranslations', limit: 10 });
assert.equal(recent.entries.length, 1, 'the cached entry is returned to the popup gallery');
assert.equal(
  recent.entries[0].thumbnail,
  THUMB_JPEG_DATA_URL,
  'the gallery receives the small generated thumbnail, not the full-size translated image',
);
assert.notEqual(
  recent.entries[0].thumbnail,
  FULL_IMAGE_DATA_URL,
  'the thumbnail field must never be byte-identical to the full-size image',
);

// A second poll for the SAME cached entry must not regenerate the thumbnail -- it was computed
// once at cache-write time, not on every gallery poll.
canvasSizesRequested.length = 0;
const recentAgain = await send({ kind: 'getRecentTranslations', limit: 10 });
assert.equal(recentAgain.entries[0].thumbnail, THUMB_JPEG_DATA_URL);
assert.equal(canvasSizesRequested.length, 0, 'polling the gallery again does not regenerate the thumbnail');

console.log('extension_thumbnail_generation=pass');
