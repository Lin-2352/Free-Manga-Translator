// Regression test for the one unescaped interpolation in popup.js's recent-translations
// gallery. entry.thumbnail is backend-controlled (round-tripped through translationCache from
// the pipeline server's translatedImageDataUrl response) and, before this fix, was written
// directly into `src="${entry.thumbnail}"` with no escaping and no validation that it is
// actually an image data URL -- a crafted value could break out of the src="" attribute into
// this privileged extension page's HTML. Every other interpolated field in this gallery item
// (title, alt, time label) was already escaped; this was the exception.
//
// Drives the REAL popup.js render path (refreshRecentTranslations -> DOMContentLoaded), not
// internals-poking, mirroring cache_gallery_limit_test.mjs's technique.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const popupPath = path.join(extensionRoot, 'popup.js');
const source = fs.readFileSync(popupPath, 'utf8');

const MALICIOUS_THUMBNAIL = '"><img src=x onerror=alert(1)><span x="';

const elementIds = [
  'recentTranslations', 'statsText', 'cacheStatusText', 'apiStatus', 'engineStatusText',
  'versionBadge', 'pipelineModeBadge', 'hoverHelp', 'themeToggleBtn', 'translationToggle',
  'startEngineBtn', 'translatePageBtn', 'translationPanelBtn', 'pickPanelBtn', 'pauseBtn',
  'softStopBtn', 'hardStopBtn', 'resumeBtn', 'clearBtn', 'clearCacheBtn', 'retranslateBtn',
  'fontSelect', 'fontColorInput', 'statusText', 'localPipelineUrl', 'localPipelineLanguage',
  'translationCachePages', 'translationQueuePages', 'translationParallelPages',
  'localPipelineAuthToken', 'saveLocalPipelineBtn', 'quotaCards', 'quotaAlert',
  'quotaSummaryText', 'refreshQuotaBtn', 'openQuotaLogBtn', 'vramCards', 'vramAlert',
  'vramSummaryText', 'refreshVramBtn', 'releaseGpuBtn', 'openVramLogBtn', 'activeJobsText',
  'queuedJobsText', 'parallelJobsText', 'queueMeterFill', 'queueItemsList', 'clearQueueBtn',
];
const elements = new Map();
for (const id of elementIds) {
  elements.set(id, {
    id, value: '', checked: false, textContent: '', innerHTML: '', hidden: false, style: {},
    listeners: {}, attributes: {},
    classList: { toggles: [], toggle() {}, add() {}, remove() {} },
    addEventListener(type, fn) { this.listeners[type] = fn; },
    setAttribute(n, v) { this.attributes[n] = String(v); },
    getAttribute(n) { return this.attributes[n] ?? null; },
  });
}

let domReadyListener;
let capturedMessage;
const documentElement = { setAttribute() {}, removeAttribute() {}, getAttribute: () => null };
const sandbox = {
  console, URL, setTimeout, clearTimeout, setInterval: () => 0,
  window: { close() {}, matchMedia: () => ({ matches: false }) },
  document: {
    addEventListener: (type, fn) => { if (type === 'DOMContentLoaded') domReadyListener = fn; },
    getElementById: (id) => elements.get(id) || { addEventListener() {}, classList: { toggle() {}, add() {}, remove() {} }, style: {} },
    querySelectorAll: () => [],
    createElement: (tag) => ({ tagName: tag, children: [], appendChild() {} }),
    documentElement,
  },
  chrome: {
    storage: { local: { get: async () => ({ translationCachePages: 20 }), set: async () => {} } },
    runtime: {
      getManifest: () => ({ version: '1.1.15' }),
      sendMessage: (message, callback) => {
        capturedMessage = message;
        if (message.kind === 'getRecentTranslations') {
          callback({
            entries: [
              { thumbnail: MALICIOUS_THUMBNAIL, pageHost: 'example.test', lastUsed: Date.now() },
              { thumbnail: 'data:image/png;base64,aGVsbG8=', pageHost: 'legit.test', lastUsed: Date.now() },
            ],
          });
        } else if (callback) {
          callback({});
        }
      },
    },
    tabs: { query: (q, cb) => cb([{ id: 1, url: 'https://example.test' }]) },
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: popupPath });
assert.equal(typeof domReadyListener, 'function', 'popup registered DOMContentLoaded');

await domReadyListener();

const html = elements.get('recentTranslations').innerHTML;

assert.equal(
  html.includes(MALICIOUS_THUMBNAIL),
  false,
  'a non-data:image thumbnail value must never appear verbatim in the rendered gallery HTML',
);
assert.equal(
  html.includes('<img src=x onerror=alert(1)>'),
  false,
  'a crafted thumbnail must not be able to inject a second unescaped <img> element',
);
assert.equal(
  html.includes('data:image/png;base64,aGVsbG8='),
  true,
  'a genuine data:image thumbnail must still render normally',
);

console.log('recent_translations_thumbnail_xss=pass');
