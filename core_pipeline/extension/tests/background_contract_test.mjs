import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const backgroundPath = path.join(extensionRoot, 'background.js');
const source = fs.readFileSync(backgroundPath, 'utf8');

let messageListener;
let fetchedUrl = '';
let fetchedBody = null;
let fetchCount = 0;
let contentInjected = false;
const contentMessages = [];
const executedScripts = [];
let queueLimitSetting = 12;
let parallelLimitSetting = 2;
let localPipelineLanguageSetting = 'ja';
const heldFetchResolvers = [];

class MockFileReader {
  readAsDataURL() {
    this.result = 'data:image/png;base64,ZmFrZQ==';
    this.onloadend();
  }
}

const sandbox = {
  console,
  setTimeout,
  clearTimeout,
  AbortController,
  FileReader: MockFileReader,
  chrome: {
    storage: {
      local: {
        get: async () => ({
          localPipelineUrl: 'http://127.0.0.1:8766/v1/translate-image',
          localPipelineLanguage: localPipelineLanguageSetting,
          translationCachePages: 12,
          translationQueuePages: queueLimitSetting,
          translationParallelPages: parallelLimitSetting,
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
      onMessage: {
        addListener: (listener) => {
          messageListener = listener;
        },
      },
      onInstalled: { addListener: () => {} },
    },
    contextMenus: {
      create: () => {},
      onClicked: { addListener: () => {} },
    },
    tabs: {
      sendMessage: async (tabId, message) => {
        contentMessages.push({ tabId, message });
        if (message.kind === 'pingContentScript' && !contentInjected) {
          throw new Error('Receiving end does not exist.');
        }
        return { ok: true };
      },
      captureVisibleTab: () => {},
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
  fetch: async (url, options = {}) => {
    if (String(url).includes('/v1/diagnostics/log')) {
      return {
        ok: true,
        json: async () => ({ ok: true }),
      };
    }
    fetchCount += 1;
    fetchedUrl = url;
    fetchedBody = options.body ? JSON.parse(options.body) : null;
    if (String(url).includes('/v1/quota-status')) {
      return {
        ok: true,
        json: async () => ({ ok: true, providers: [] }),
      };
    }
    if (String(url).includes('/v1/vram-status')) {
      return {
        ok: true,
        json: async () => ({
          ok: true,
          available: true,
          gpus: [
            {
              index: 0,
              name: 'RTX Test GPU',
              memoryTotalMiB: 8192,
              memoryUsedMiB: 4096,
              memoryFreeMiB: 4096,
              usagePercent: 50,
              utilizationGpuPercent: 35,
            },
          ],
        }),
      };
    }
    if (String(url).includes('/v1/open-log-window/')) {
      return {
        ok: true,
        json: async () => ({ success: true, logType: String(url).split('/').pop() }),
      };
    }
    if (String(url).includes('/v1/health')) {
      return {
        ok: true,
        json: async () => ({ ok: true }),
      };
    }
    if (String(url).includes('/v1/warmup')) {
      return {
        ok: true,
        json: async () => ({ ok: true, status: 'warming' }),
      };
    }
    if (String(url).includes('/v1/gpu/release')) {
      return {
        ok: true,
        json: async () => ({ success: true, status: 'released' }),
      };
    }
    if (String(url).includes('/v1/cache/clear')) {
      return {
        ok: true,
        json: async () => ({ success: true, status: 'cleared', clearedSamples: 2 }),
      };
    }
    if (String(url).includes('/v1/runtime/soft-stop')) {
      return {
        ok: true,
        json: async () => ({ success: true, status: 'stopped', releaseGpu: false }),
      };
    }
    if (String(url).includes('/v1/runtime/hard-stop')) {
      return {
        ok: true,
        json: async () => ({ success: true, status: 'stopped', releaseGpu: true }),
      };
    }
    const responsePayload = {
      translatedImageDataUrl: 'data:image/png;base64,ZmFrZQ==',
      report: {
        pipeline: 'local-8-stage',
        stageSequence: Array.from({ length: 8 }, (_, index) => ({ step: index + 1 })),
      },
      translations: [],
    };
    if (fetchedBody?.metadata?.cacheKey?.includes('hold')) {
      return new Promise((resolve) => {
        heldFetchResolvers.push(() => resolve({
          ok: true,
          json: async () => responsePayload,
        }));
      });
    }
    return {
      ok: true,
      json: async () => responsePayload,
    };
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: backgroundPath });

assert.equal(typeof messageListener, 'function', 'background message listener registered');

const response = await new Promise((resolve) => {
  const keepAlive = messageListener(
    {
      kind: 'translateImage',
      base64Data: 'data:image/png;base64,ZmFrZQ==',
      cacheKey: 'https://example.test/page-1.jpg|100x120',
      originalImageUrl: 'https://example.test/page-1.jpg',
      pageCacheKey: 'https://example.test/manga',
      width: 100,
      height: 120,
      pageUrl: 'https://example.test/manga',
    },
    { tab: { id: 1 } },
    resolve,
  );
  assert.equal(keepAlive, true, 'async sendResponse kept alive');
});

assert.equal(fetchedUrl, 'http://127.0.0.1:8766/v1/translate-image');
assert.equal(fetchedBody.sourceLanguage, 'ja');
assert.equal(fetchedBody.targetLanguage, 'en');
assert.equal(fetchedBody.qualityProfile, 'strict');
assert.equal(fetchedBody.requestedOutput, 'translatedImageDataUrl');
assert.equal(fetchedBody.metadata.source, 'extension-canvas');
assert.equal(fetchedBody.metadata.cacheKey, 'https://example.test/page-1.jpg|100x120');
assert.equal(fetchedBody.metadata.originalImageUrl, 'https://example.test/page-1.jpg');
assert.equal(response.translatedImageDataUrl, 'data:image/png;base64,ZmFrZQ==');
assert.equal(response.pipelineReport.pipeline, 'local-8-stage');
assert.equal(fetchCount, 1);

localPipelineLanguageSetting = 'zh';
await new Promise((resolve) => {
  messageListener(
    {
      kind: 'translateImage',
      base64Data: 'data:image/png;base64,emg=',
      cacheKey: 'https://example.test/chinese-page.jpg|100x120',
      originalImageUrl: 'https://example.test/chinese-page.jpg',
      pageCacheKey: 'https://example.test/manhua',
      width: 100,
      height: 120,
      pageUrl: 'https://example.test/manhua',
    },
    { tab: { id: 1 } },
    resolve,
  );
});
assert.equal(fetchedBody.sourceLanguage, 'zh', 'selected Chinese source language reaches backend payload');
localPipelineLanguageSetting = 'ja';

const cachedLookup = await new Promise((resolve) => {
  const keepAlive = messageListener(
    {
      kind: 'lookupCachedTranslation',
      cacheKey: 'https://example.test/page-1.jpg|100x120',
      originalImageUrl: 'https://example.test/page-1.jpg',
      pageCacheKey: 'https://example.test/manga',
      pageUrl: 'https://example.test/manga',
    },
    { tab: { id: 1 } },
    resolve,
  );
  assert.equal(keepAlive, true, 'cache lookup response kept alive');
});

assert.equal(cachedLookup.hit, true, 'stable original-source key restores from cache');
assert.equal(cachedLookup.fromCache, true);
assert.equal(cachedLookup.translatedImageDataUrl, 'data:image/png;base64,ZmFrZQ==');

const clearCacheResponse = await new Promise((resolve) => {
  messageListener({ kind: 'clearCache' }, { tab: { id: 1 } }, resolve);
});
assert.equal(clearCacheResponse.success, true, 'clear cache succeeds even when it also talks to backend runtime cache');
assert.equal(clearCacheResponse.backend.status, 'cleared', 'clear cache purges backend runtime output cache');

const activationResponse = await new Promise((resolve) => {
  const keepAlive = messageListener(
    {
      kind: 'activatePageTranslation',
      tabId: 9,
      persistAuto: false,
    },
    { tab: { id: 9 } },
    resolve,
  );
  assert.equal(keepAlive, true, 'activation response kept alive');
});

assert.equal(activationResponse.success, true);
assert.equal(executedScripts.some((item) => item.target?.tabId === 9 && item.files?.includes('content.js')), true);
assert.equal(contentMessages.some((item) => item.tabId === 9 && item.message.kind === 'translatePageOnce'), true);
assert.equal(contentMessages.some((item) => item.tabId === 9 && item.message.kind === 'setTranslationPaused' && item.message.paused === false), true);

const autoActivationResponse = await new Promise((resolve) => {
  const keepAlive = messageListener(
    {
      kind: 'activatePageTranslation',
      tabId: 9,
      persistAuto: true,
    },
    { tab: { id: 9 } },
    resolve,
  );
  assert.equal(keepAlive, true, 'auto activation response kept alive');
});

assert.equal(autoActivationResponse.success, true);
assert.equal(autoActivationResponse.autoEnabled, true);
assert.equal(contentMessages.some((item) => item.tabId === 9 && item.message.kind === 'toggleTranslation' && item.message.enabled === true), true);

const statsResponse = await new Promise((resolve) => {
  messageListener({ kind: 'getTranslationStats' }, { tab: { id: 1 } }, resolve);
});
assert.equal(statsResponse.parallelLimit, 2, 'stats expose adaptive parallel limit');

const vramResponse = await new Promise((resolve) => {
  messageListener({ kind: 'getVramStatus' }, { tab: { id: 1 } }, resolve);
});
assert.equal(vramResponse.ok, true, 'background exposes backend VRAM status');
assert.equal(vramResponse.gpus[0].memoryUsedMiB, 4096);

const logWindowResponse = await new Promise((resolve) => {
  messageListener({ kind: 'openLogWindow', logType: 'vram' }, { tab: { id: 1 } }, resolve);
});
assert.equal(logWindowResponse.success, true, 'background can request a separate telemetry log window');

const startEngineResponse = await new Promise((resolve) => {
  messageListener({ kind: 'startEngine' }, { tab: { id: 1 } }, resolve);
});
assert.equal(startEngineResponse.ok, true, 'background can health-check and warm up the local engine');
assert.equal(startEngineResponse.warmup.ok, true, 'start engine requests backend warmup');

const releaseGpuResponse = await new Promise((resolve) => {
  messageListener({ kind: 'releaseGpu' }, { tab: { id: 1 } }, resolve);
});
assert.equal(releaseGpuResponse.status, 'released', 'background can request backend GPU release');

const softStopResponse = await new Promise((resolve) => {
  messageListener({ kind: 'stopTranslations', mode: 'soft', tabId: 9 }, { tab: { id: 1 } }, resolve);
});
assert.equal(softStopResponse.backend.releaseGpu, false, 'soft stop does not request GPU release');
assert.equal(
  contentMessages.some((item) => item.tabId === 9 && item.message.kind === 'setTranslationPaused' && item.message.paused === true),
  true,
  'soft stop tells the active content script to clear loaders',
);

const hardStopResponse = await new Promise((resolve) => {
  messageListener({ kind: 'stopTranslations', mode: 'hard', tabId: 9 }, { tab: { id: 1 } }, resolve);
});
assert.equal(hardStopResponse.backend.releaseGpu, true, 'hard stop requests backend GPU release');
await new Promise((resolve) => {
  messageListener({ kind: 'setTranslationPaused', paused: false }, { tab: { id: 1 } }, resolve);
});

queueLimitSetting = 1;
parallelLimitSetting = 1;
const activeHeld = new Promise((resolve) => {
  messageListener(
    {
      kind: 'translateImage',
      base64Data: 'data:image/png;base64,YWN0aXZl',
      cacheKey: 'hold-active|100x120',
      originalImageUrl: 'https://example.test/hold-active.jpg',
      pageCacheKey: 'https://example.test/queue',
      width: 100,
      height: 120,
      pageUrl: 'https://example.test/queue',
    },
    { tab: { id: 1 } },
    resolve,
  );
});

await new Promise((resolve) => setTimeout(resolve, 0));

const queuedHeld = new Promise((resolve) => {
  messageListener(
    {
      kind: 'translateImage',
      base64Data: 'data:image/png;base64,cXVldWVk',
      cacheKey: 'hold-queued|100x120',
      originalImageUrl: 'https://example.test/hold-queued.jpg',
      pageCacheKey: 'https://example.test/queue',
      width: 100,
      height: 120,
      pageUrl: 'https://example.test/queue',
    },
    { tab: { id: 1 } },
    resolve,
  );
});

await new Promise((resolve) => setTimeout(resolve, 0));

const queueFull = await new Promise((resolve) => {
  messageListener(
    {
      kind: 'translateImage',
      base64Data: 'data:image/png;base64,ZnVsbA==',
      cacheKey: 'hold-full|100x120',
      originalImageUrl: 'https://example.test/hold-full.jpg',
      pageCacheKey: 'https://example.test/queue',
      width: 100,
      height: 120,
      pageUrl: 'https://example.test/queue',
    },
    { tab: { id: 1 } },
    resolve,
  );
});

assert.equal(queueFull.error, 'QueueFull', 'bounded queue rejects over-limit jobs');
assert.equal(queueFull.queueLimit, 1);
heldFetchResolvers.shift()?.();
assert.equal((await activeHeld).translatedImageDataUrl, 'data:image/png;base64,ZmFrZQ==');
await new Promise((resolve) => setTimeout(resolve, 0));
heldFetchResolvers.shift()?.();
assert.equal((await queuedHeld).translatedImageDataUrl, 'data:image/png;base64,ZmFrZQ==');

console.log('extension_background_contract=pass');
