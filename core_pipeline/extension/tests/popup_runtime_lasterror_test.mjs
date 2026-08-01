// Proves popup.js's runtimeMessage() helper checks chrome.runtime.lastError after a
// sendMessage callback fires. Previously a failed sendMessage (background script not ready, or
// the message port closed before a real response arrived) resolved silently to `{}` -- every
// caller's `response.success === false` check then saw a plain, false-successful-looking `{}` and
// reported success even though the message never actually reached background.js (e.g. Clear Cache
// claiming "caches cleared" when nothing was cleared).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const popupPath = path.join(extensionRoot, 'popup.js');
const source = fs.readFileSync(popupPath, 'utf8');

const elementIds = [
  'translationToggle', 'startEngineBtn', 'translatePageBtn', 'translationPanelBtn', 'pickPanelBtn',
  'pauseBtn', 'softStopBtn', 'hardStopBtn', 'resumeBtn', 'clearBtn', 'clearCacheBtn', 'retranslateBtn',
  'fontSelect', 'fontColorInput', 'statusText', 'statsText', 'cacheStatusText', 'localPipelineUrl',
  'localPipelineLanguage', 'translationCachePages', 'translationQueuePages', 'translationParallelPages',
  'localPipelineAuthToken', 'saveLocalPipelineBtn', 'apiStatus', 'quotaCards', 'quotaAlert',
  'quotaSummaryText', 'refreshQuotaBtn', 'openQuotaLogBtn', 'vramCards', 'vramAlert', 'vramSummaryText',
  'refreshVramBtn', 'releaseGpuBtn', 'openVramLogBtn', 'activeJobsText', 'queuedJobsText',
  'parallelJobsText', 'queueMeterFill', 'queueMeter', 'queueItemsList', 'clearQueueBtn',
  'engineStatusText', 'hoverHelp', 'versionBadge', 'pipelineModeBadge', 'themeToggleBtn',
];

const elements = new Map();
for (const id of elementIds) {
  elements.set(id, {
    id,
    value: '',
    checked: false,
    textContent: '',
    innerHTML: '',
    hidden: false,
    style: {},
    listeners: {},
    classList: {
      toggle() {}, add() {}, remove() {}, contains: () => false,
    },
    addEventListener(type, listener) { this.listeners[type] = listener; },
    append() {},
    attributes: {},
    setAttribute(name, value) { this.attributes[name] = String(value); },
    getAttribute(name) { return Object.prototype.hasOwnProperty.call(this.attributes, name) ? this.attributes[name] : null; },
  });
}

let domReadyListener;
const documentElementAttrs = {};
const documentElement = {
  setAttribute(name, value) { documentElementAttrs[name] = String(value); },
  removeAttribute(name) { delete documentElementAttrs[name]; },
  getAttribute(name) { return Object.prototype.hasOwnProperty.call(documentElementAttrs, name) ? documentElementAttrs[name] : null; },
};

// The test flips this before triggering a sendMessage call to simulate the callback firing with
// chrome.runtime.lastError set (background script not ready / message port closed early).
let simulateLastError = null;

const sandbox = {
  console,
  URL,
  setTimeout: (callback) => { callback(); return 1; },
  clearTimeout: () => {},
  setInterval: () => 0,
  window: { close: () => {}, matchMedia: () => ({ matches: false }) },
  document: {
    addEventListener: (type, listener) => { if (type === 'DOMContentLoaded') domReadyListener = listener; },
    getElementById: (id) => elements.get(id),
    querySelectorAll: () => [],
    createElement: () => ({ tagName: 'DIV', textContent: '', children: [], appendChild() {} }),
    documentElement,
  },
  chrome: {
    storage: {
      local: {
        get: async () => ({
          translationEnabled: false,
          translationPaused: false,
          translationCachePages: 12,
          translationQueuePages: 5,
          translationParallelPages: 2,
          localPipelineUrl: 'http://127.0.0.1:8766/v1/translate-image',
          localPipelineLanguage: 'ja',
        }),
        set: async () => {},
      },
    },
    runtime: {
      getManifest: () => ({ version: '1.1.15' }),
      get lastError() { return simulateLastError; },
      sendMessage: (message, callback) => {
        let response = { success: true };
        if (message.kind === 'getTranslationStats') {
          response = {
            cacheSize: 0, cacheLimit: 12, activeRequests: 0, queueLength: 0, queueLimit: 5,
            parallelLimit: 2, pressurePercent: 0, isPaused: false, items: [], pipelineBreakerOpen: false,
          };
        }
        if (message.kind === 'checkPipelineHealth') response = { ok: true, cacheSize: 0 };
        if (message.kind === 'getQuotaStatus') response = { ok: true, globalLimitReached: false, providers: [] };
        if (message.kind === 'clearCache') {
          // Simulate the message port failing (background not ready / channel closed) -- the
          // callback still fires (as Chrome does), but with no real payload and lastError set.
          if (simulateLastError) {
            callback?.(undefined);
            return;
          }
          response = { success: true, cacheSize: 0, backendCleared: true, backend: { success: true } };
        }
        callback?.(response);
      },
    },
    tabs: { query: (query, callback) => callback([{ id: 123, url: 'https://example.test/manga' }]) },
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: popupPath });
assert.equal(typeof domReadyListener, 'function', 'popup registered DOMContentLoaded');

await domReadyListener();

// (a) Normal case: sendMessage succeeds, no lastError -- Clear Cache reports real success.
simulateLastError = null;
await elements.get('clearCacheBtn').listeners.click();
assert.equal(
  elements.get('statusText').textContent.includes('cleared'),
  true,
  'a real, successful clearCache response reports success normally',
);

// (b) Failure case: the message port fails and chrome.runtime.lastError is set on the callback.
// This must surface as a real, visible error -- NOT a false "cleared" success message.
simulateLastError = { message: 'Could not establish connection. Receiving end does not exist.' };
elements.get('statusText').textContent = '';
await elements.get('clearCacheBtn').listeners.click();
assert.notEqual(
  elements.get('statusText').textContent.includes('cleared'),
  true,
  'a failed sendMessage (chrome.runtime.lastError set) must not be reported as a successful cache clear',
);
assert.equal(
  elements.get('statusText').textContent.length > 0,
  true,
  'a failed sendMessage still surfaces SOME visible error text to the user, not silence',
);

console.log('extension_popup_runtime_lasterror=pass');
