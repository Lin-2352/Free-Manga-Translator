import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const popupPath = path.join(extensionRoot, 'popup.js');
const source = fs.readFileSync(popupPath, 'utf8');

const elementIds = [
  'translationToggle',
  'startEngineBtn',
  'translatePageBtn',
  'translationPanelBtn',
  'pickPanelBtn',
  'pauseBtn',
  'softStopBtn',
  'hardStopBtn',
  'resumeBtn',
  'clearBtn',
  'clearCacheBtn',
  'retranslateBtn',
  'fontSelect',
  'fontColorInput',
  'statusText',
  'statsText',
  'cacheStatusText',
  'localPipelineUrl',
  'localPipelineLanguage',
  'translationCachePages',
  'translationQueuePages',
  'translationParallelPages',
  'localPipelineAuthToken',
  'saveLocalPipelineBtn',
  'apiStatus',
  'quotaCards',
  'quotaAlert',
  'quotaSummaryText',
  'refreshQuotaBtn',
  'openQuotaLogBtn',
  'vramCards',
  'vramAlert',
  'vramSummaryText',
  'refreshVramBtn',
  'releaseGpuBtn',
  'openVramLogBtn',
  'activeJobsText',
  'queuedJobsText',
  'parallelJobsText',
  'queueMeterFill',
  'queueItemsList',
  'clearQueueBtn',
  'engineStatusText',
  'hoverHelp',
  'versionBadge',
  'pipelineModeBadge',
  'themeToggleBtn',
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
      toggles: [],
      added: [],
      removed: [],
      toggle(name, enabled) {
        this.toggles.push({ name, enabled });
      },
      add(name) {
        this.added.push(name);
      },
      remove(name) {
        this.removed.push(name);
      },
    },
    addEventListener(type, listener) {
      this.listeners[type] = listener;
    },
    append(...nodes) {
      this.appendedChildren = (this.appendedChildren || []).concat(nodes);
    },
    attributes: {},
    setAttribute(name, value) {
      this.attributes[name] = String(value);
    },
    getAttribute(name) {
      return Object.prototype.hasOwnProperty.call(this.attributes, name) ? this.attributes[name] : null;
    },
  });
}

let domReadyListener;
let closed = false;
let systemPrefersDark = false;
const runtimeMessages = [];
let pickerShouldStart = true;
let statsItems = [];
let statsPipelineBreakerOpen = false;
const storageWrites = [];
const documentElementAttrs = {};
const documentElement = {
  setAttribute(name, value) {
    documentElementAttrs[name] = String(value);
  },
  removeAttribute(name) {
    delete documentElementAttrs[name];
  },
  getAttribute(name) {
    return Object.prototype.hasOwnProperty.call(documentElementAttrs, name) ? documentElementAttrs[name] : null;
  },
};

// A real, controllable fake-timer harness (replacing an always-fire-synchronously stub) --
// needed to test the tooltip onset delay (Commit 7): whether show() actually waits for the
// delay, and whether a quick mouseleave cancels it via clearTimeout before it fires. Audited
// every other setTimeout call site in popup.js (flashSaved/flashAction/Start Engine label
// reset) before this swap: none of them have an existing assertion depending on the old
// fire-immediately behavior, so this is a safe global replacement.
let nextTimerId = 1;
const pendingTimers = new Map();
function fireTimer(id) {
  const callback = pendingTimers.get(id);
  pendingTimers.delete(id);
  callback?.();
}
function fireAllTimers() {
  const callbacks = Array.from(pendingTimers.values());
  pendingTimers.clear();
  callbacks.forEach((callback) => callback());
}

const helpPanelListeners = {};
const helpPanelTarget = {
  dataset: { help: 'First detail|Second detail', helpTitle: 'Test Action' },
  textContent: 'Test Action',
  getBoundingClientRect: () => ({ left: 10, top: 10, bottom: 30, right: 100 }),
  addEventListener(type, listener) {
    helpPanelListeners[type] = listener;
  },
};

const sandbox = {
  console,
  URL,
  setTimeout: (callback) => {
    const id = nextTimerId++;
    pendingTimers.set(id, callback);
    return id;
  },
  clearTimeout: (id) => {
    pendingTimers.delete(id);
  },
  setInterval: () => 0,
  window: {
    close: () => {
      closed = true;
    },
    matchMedia: () => ({ matches: systemPrefersDark }),
  },
  document: {
    addEventListener: (type, listener) => {
      if (type === 'DOMContentLoaded') domReadyListener = listener;
    },
    getElementById: (id) => elements.get(id),
    querySelectorAll: (selector) => (selector === '[data-help-panel]' ? [helpPanelTarget] : []),
    createElement: (tag) => {
      const node = {
        tagName: String(tag).toUpperCase(),
        textContent: '',
        children: [],
        appendChild(child) {
          this.children.push(child);
        },
      };
      Object.defineProperty(node, 'childElementCount', { get() { return this.children.length; } });
      return node;
    },
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
        set: async (payload) => {
          storageWrites.push(payload);
        },
      },
    },
    runtime: {
      getManifest: () => ({ version: '1.1.13' }),
      sendMessage: (message, callback) => {
        runtimeMessages.push(message);
        let response = { success: true };
        if (message.kind === 'getTranslationStats') {
          response = {
            cacheSize: 0,
            cacheLimit: 12,
            activeRequests: statsItems.filter((item) => item.status === 'active').length,
            queueLength: statsItems.filter((item) => item.status !== 'active').length,
            queueLimit: 5,
            parallelLimit: 2,
            pressurePercent: 0,
            isPaused: false,
            items: statsItems,
            pipelineBreakerOpen: statsPipelineBreakerOpen,
          };
        }
        if (message.kind === 'checkPipelineHealth') {
          response = { ok: true, cacheSize: 0 };
        }
        if (message.kind === 'getQuotaStatus') {
          response = {
            ok: true,
            globalLimitReached: false,
            providers: [
              {
                provider: 'mistral',
                configured: true,
                health: 'healthy',
                remainingPercent: 91,
                activeKeys: 2,
                keyCount: 2,
              },
              {
                provider: 'openrouter',
                configured: true,
                health: 'rate_limited',
                remainingPercent: 87,
                activeKeys: 3,
                keyCount: 3,
                rateLimitedUntil: '2026-06-05T12:35:00+00:00',
              },
              {
                provider: 'groq',
                configured: true,
                health: 'auth_locked',
                reason: 'provider access denied or network-blocked (HTTP 403)',
                remainingPercent: 80,
                activeKeys: 0,
                keyCount: 1,
              },
            ],
          };
        }
        if (message.kind === 'getVramStatus') {
          response = {
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
          };
        }
        if (message.kind === 'openLogWindow') {
          response = { success: true, logType: message.logType };
        }
        if (message.kind === 'releaseGpu') {
          response = { success: true, status: 'released' };
        }
        if (message.kind === 'startEngine') {
          response = { ok: true, available: true, warmup: { ok: true, skipped: false, payload: { status: 'running' } } };
        }
        if (message.kind === 'clearCache') {
          response = { success: true, cacheSize: 0, backendCleared: true, backend: { success: true } };
        }
        if (message.kind === 'clearQueue') {
          response = { success: true, dropped: 2, activeRequests: 0, queueLength: 0, queueLimit: 5, parallelLimit: 2 };
        }
        if (message.kind === 'stopTranslations') {
          response = { success: true, status: 'stopped', mode: message.mode, activeRequests: 0, queueLength: 0, queueLimit: 5, parallelLimit: 2, cacheSize: 0, cacheLimit: 12 };
        }
        if (message.kind === 'sendContentCommand' && message.command?.kind === 'togglePanelPicker') {
          response = pickerShouldStart
            ? { success: true, response: { success: true, active: true } }
            : { success: true, response: { success: false, active: false, error: 'SelectionPanelActive' } };
        }
        if (callback) callback(response);
      },
    },
    tabs: {
      query: (query, callback) => {
        callback([{ id: 123, url: 'https://example.test/manga' }]);
      },
    },
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: popupPath });
assert.equal(typeof domReadyListener, 'function', 'popup registered DOMContentLoaded');

await domReadyListener();
assert.equal(elements.get('localPipelineAuthToken').value, '', 'auth token field is empty by default (no local deployment sends one)');
elements.get('localPipelineAuthToken').value = '  my-tunnel-token  ';
await elements.get('saveLocalPipelineBtn').listeners.click();
assert.equal(
  storageWrites.some((payload) => payload.localPipelineAuthToken === 'my-tunnel-token'),
  true,
  'Save trims and persists the auth token alongside the pipeline URL/language',
);
elements.get('localPipelineAuthToken').value = '';
await elements.get('saveLocalPipelineBtn').listeners.click();
assert.equal(
  storageWrites.some((payload) => payload.localPipelineAuthToken === ''),
  true,
  'clearing the field and saving again persists an empty token, not the stale previous value',
);

// ===== URL normalization (the bare-origin trap): a bare origin with no path passed health
// checks (which rewrite to /v1/health) while every real translate request 404'd, since the
// translate POST uses the saved URL verbatim. Saving must append /v1/translate-image when
// the path is empty, and must never touch a URL that already has a real path. =====
elements.get('localPipelineUrl').value = 'https://example-test.ngrok-free.app';
await elements.get('saveLocalPipelineBtn').listeners.click();
assert.equal(
  storageWrites.at(-1).localPipelineUrl,
  'https://example-test.ngrok-free.app/v1/translate-image',
  'a bare origin with no path gets /v1/translate-image appended on save',
);
assert.equal(
  elements.get('localPipelineUrl').value,
  'https://example-test.ngrok-free.app/v1/translate-image',
  'the visible field is updated to show what was actually stored',
);
assert.equal(
  elements.get('pipelineModeBadge').textContent,
  'REMOTE',
  'the header badge flips to REMOTE when the saved backend is not loopback',
);

elements.get('localPipelineUrl').value = 'https://example-test.ngrok-free.app/v1/translate-image';
await elements.get('saveLocalPipelineBtn').listeners.click();
assert.equal(
  storageWrites.at(-1).localPipelineUrl,
  'https://example-test.ngrok-free.app/v1/translate-image',
  'a URL that already has the correct path is left untouched',
);

elements.get('localPipelineUrl').value = 'not a valid url at all';
await elements.get('saveLocalPipelineBtn').listeners.click();
assert.equal(
  storageWrites.at(-1).localPipelineUrl,
  'not a valid url at all',
  'an unparseable value is saved exactly as typed, never mangled or guessed at',
);

elements.get('localPipelineUrl').value = 'http://127.0.0.1:8766/v1/translate-image';
await elements.get('saveLocalPipelineBtn').listeners.click();
assert.equal(
  elements.get('pipelineModeBadge').textContent,
  'LOCAL',
  'the header badge flips back to LOCAL when the saved backend is loopback again',
);

await elements.get('startEngineBtn').listeners.click();
assert.equal(
  runtimeMessages.some((message) => message.kind === 'startEngine'),
  true,
  'Start Engine requests backend health and warmup through the background service worker',
);
await elements.get('translatePageBtn').listeners.click();

assert.equal(
  runtimeMessages.some((message) => message.kind === 'activatePageTranslation' && message.tabId === 123 && message.persistAuto === false),
  true,
  'Translate Page uses one-shot background activation with content-script injection',
);
assert.equal(
  storageWrites.some((payload) => payload.translationEnabled === true),
  false,
  'Translate Page does not enable auto-translate',
);
assert.equal(elements.get('translationToggle').checked, false, 'auto-translate remains off by default');
assert.equal(elements.get('translationQueuePages').value, '5', 'queue-ahead defaults to five pending jobs');
assert.equal(elements.get('translationParallelPages').value, '2', 'adaptive parallelism defaults to two active jobs');
assert.equal(elements.get('engineStatusText').textContent, 'Warming', 'Start Engine gives visible engine-state feedback');
assert.equal(elements.get('cacheStatusText').textContent, '0/12 entries cached');
assert.equal(elements.get('activeJobsText').textContent, '0', 'queue dashboard renders active job count');
assert.equal(elements.get('queuedJobsText').textContent, '0/5', 'queue dashboard renders bounded queue count');
assert.equal(elements.get('parallelJobsText').textContent, '2', 'queue dashboard renders parallel job count');
assert.equal(elements.get('queueMeterFill').style.width, '0%', 'queue pressure meter renders');
assert.equal(elements.get('quotaCards').innerHTML.includes('Mistral'), true, 'quota dashboard renders provider status');
assert.equal(elements.get('quotaSummaryText').textContent, '1/3 healthy', 'quota status collapses to a summary');
assert.equal(elements.get('quotaCards').innerHTML.includes('rate-limited until'), true, 'quota dashboard renders per-minute throttling state');
assert.equal(elements.get('quotaCards').innerHTML.includes('access blocked'), true, 'quota dashboard renders auth/network lock state');
assert.equal(elements.get('versionBadge').textContent, 'v1.1.13', 'popup renders the extension version badge');
await elements.get('refreshVramBtn').listeners.click();
assert.equal(elements.get('vramSummaryText').textContent, '50% used', 'VRAM dashboard renders usage summary');
assert.equal(elements.get('vramCards').innerHTML.includes('RTX Test GPU'), true, 'VRAM dashboard renders GPU card');
await elements.get('releaseGpuBtn').listeners.click();
assert.equal(
  runtimeMessages.some((message) => message.kind === 'releaseGpu'),
  true,
  'Release GPU button calls the background GPU release command',
);
await elements.get('pauseBtn').listeners.click();
assert.equal(
  runtimeMessages.some((message) => message.kind === 'pausePageTranslation' && message.tabId === 123),
  true,
  'Pause button calls the background pause command for the active tab',
);
await elements.get('softStopBtn').listeners.click();
assert.equal(
  runtimeMessages.some((message) => message.kind === 'stopTranslations' && message.mode === 'soft'),
  true,
  'Soft Stop calls the background soft-stop command',
);
assert.equal(
  runtimeMessages.some((message) => message.kind === 'sendContentCommand' && message.command?.kind === 'setTranslationPaused' && message.command?.paused === true),
  true,
  'Soft Stop also tells the content script to clear page loaders',
);
assert.equal(
  storageWrites.some((payload) => payload.translationEnabled === false),
  true,
  'Soft Stop disables auto-translate in stored popup state',
);
await elements.get('hardStopBtn').listeners.click();
assert.equal(
  runtimeMessages.some((message) => message.kind === 'stopTranslations' && message.mode === 'hard'),
  true,
  'Hard Stop calls the background hard-stop command',
);
statsItems = [
  { status: 'active', pageHost: 'example.test', originalImageUrl: 'https://example.test/manga/page-1.jpg' },
  { status: 'queued', position: 1, pageHost: 'example.test', originalImageUrl: 'https://example.test/manga/page-2.jpg' },
];
await elements.get('clearQueueBtn').listeners.click();
assert.equal(
  runtimeMessages.some((message) => message.kind === 'clearQueue'),
  true,
  'Clear Queue calls the bounded queue cleanup command',
);
assert.equal(elements.get('queueItemsList').hidden, false, 'per-item queue list becomes visible once items are reported');
const queueItemsHtml = elements.get('queueItemsList').innerHTML;
assert.equal(queueItemsHtml.includes('page-1.jpg'), true, 'active item renders its image label');
assert.equal(queueItemsHtml.includes('page-2.jpg'), true, 'queued item renders its image label');
assert.equal(queueItemsHtml.includes('status-active'), true, 'active item gets the active status class');
assert.equal(queueItemsHtml.includes('status-queued'), true, 'queued item gets the queued status class');
statsItems = [];

// The circuit breaker in background.js (pipelineBreakerOpen) is a live signal, distinct from
// checkPipelineHealth -- it must be able to push the badge to Offline even though the health
// check mock above already returned { ok: true }, since the breaker can trip well after the
// popup's one-time health check ran.
assert.equal(
  elements.get('apiStatus').classList.toggles.some((t) => t.name === 'active' && t.enabled === true),
  true,
  'sanity: apiStatus was previously marked active by checkServerHealth',
);
statsPipelineBreakerOpen = true;
await elements.get('clearQueueBtn').listeners.click();
assert.equal(
  elements.get('apiStatus').classList.removed.slice(-1)[0],
  'active',
  'a live breaker-open signal removes the active class even after an earlier healthy check',
);
assert.equal(elements.get('apiStatus').classList.added.slice(-1)[0], 'error', 'and marks the badge as error');
assert.equal(elements.get('engineStatusText').textContent, 'Offline — will resume automatically', 'engine status text explains the outage is self-recovering');
statsPipelineBreakerOpen = false;
let messagesBefore = runtimeMessages.length;
await elements.get('clearBtn').listeners.click();
let clearPageMessages = runtimeMessages.slice(messagesBefore);
assert.equal(
  clearPageMessages.some((message) => message.kind === 'sendContentCommand' && message.command?.kind === 'clearTranslations'),
  true,
  'Clear Page sends the visual-reset content command',
);
assert.equal(
  clearPageMessages.some((message) => message.kind === 'clearCache'),
  false,
  'Clear Page must NOT clear the browser/backend cache -- only the dedicated Clear Cache button does',
);

messagesBefore = runtimeMessages.length;
await elements.get('retranslateBtn').listeners.click();
const retranslateMessages = runtimeMessages.slice(messagesBefore);
assert.equal(
  retranslateMessages.some((message) => message.kind === 'sendContentCommand' && message.command?.kind === 'retranslateAll'),
  true,
  'Re-translate sends the page re-scan content command',
);
assert.equal(
  retranslateMessages.some((message) => message.kind === 'clearCache'),
  false,
  'Re-translate must NOT clear the cache -- already-cached pages should restore instantly, not recompute',
);

messagesBefore = runtimeMessages.length;
await elements.get('clearCacheBtn').listeners.click();
assert.equal(
  runtimeMessages.slice(messagesBefore).some((message) => message.kind === 'clearCache'),
  true,
  'Clear Cache calls browser/backend cache cleanup',
);
await elements.get('resumeBtn').listeners.click();
assert.equal(
  runtimeMessages.some((message) => message.kind === 'activatePageTranslation' && message.persistAuto === false),
  true,
  'Resume restarts translation without forcing auto mode when the toggle is off',
);
assert.equal(closed, true, 'popup closes after successful activation');

const themeToggleBtn = elements.get('themeToggleBtn');
assert.equal(
  documentElement.getAttribute('data-theme'),
  null,
  'no stored theme preference means no data-theme override, so CSS falls back to the OS setting',
);
assert.equal(
  themeToggleBtn.classList.toggles.at(-1)?.enabled,
  false,
  'theme toggle button does not render as active/dark when following the light system default',
);

await themeToggleBtn.listeners.click();
assert.equal(documentElement.getAttribute('data-theme'), 'dark', 'clicking the theme toggle forces dark mode via data-theme');
assert.equal(
  storageWrites.some((payload) => payload.uiTheme === 'dark'),
  true,
  'theme choice persists to chrome.storage.local so it survives popup reopen',
);
assert.equal(themeToggleBtn.classList.toggles.at(-1)?.enabled, true, 'theme toggle button reflects the active dark state');
assert.equal(themeToggleBtn.getAttribute('aria-pressed'), 'true', 'theme toggle exposes its state to assistive tech');

await themeToggleBtn.listeners.click();
assert.equal(documentElement.getAttribute('data-theme'), 'light', 'clicking again forces light mode');
assert.equal(
  storageWrites.some((payload) => payload.uiTheme === 'light'),
  true,
  'switching back to light also persists',
);
assert.equal(themeToggleBtn.classList.toggles.at(-1)?.enabled, false, 'theme toggle button reflects the active light state');

// ===== Pick Panel =====
closed = false;
messagesBefore = runtimeMessages.length;
pickerShouldStart = true;
await elements.get('pickPanelBtn').listeners.click();
const pickPanelMessages = runtimeMessages.slice(messagesBefore);
assert.equal(
  pickPanelMessages.some((message) => message.kind === 'sendContentCommand' && message.command?.kind === 'togglePanelPicker'),
  true,
  'Pick Panel sends the picker-toggle content command',
);
assert.equal(closed, true, 'popup closes after the picker starts successfully, same as Selection Panel');

// The Selection Panel already being open is the one real failure mode content.js can report
// back (mutual exclusion) -- the popup must surface it instead of closing as if nothing happened.
closed = false;
const statusTextEl = elements.get('statusText');
statusTextEl.textContent = '';
pickerShouldStart = false;
await elements.get('pickPanelBtn').listeners.click();
assert.equal(closed, false, 'popup stays open when the picker refuses to start');
assert.equal(
  statusTextEl.textContent,
  'SelectionPanelActive',
  'the picker-blocked error from content.js is surfaced to the user, not swallowed',
);

// ===== Tooltip onset delay =====
const hoverHelpEl = elements.get('hoverHelp');

helpPanelListeners.mouseenter();
assert.equal(
  hoverHelpEl.classList.toggles.length,
  0,
  'hovering does not show the tooltip instantly',
);
fireAllTimers();
assert.equal(
  hoverHelpEl.classList.toggles.at(-1)?.enabled,
  true,
  'the tooltip appears after the hover delay elapses',
);
helpPanelListeners.mouseleave();

// A quick hover-and-leave before the delay elapses must cancel the pending show via
// clearTimeout, not show it late once the timer is advanced.
const togglesBeforeQuickHover = hoverHelpEl.classList.toggles.length;
helpPanelListeners.mouseenter();
helpPanelListeners.mouseleave();
fireAllTimers();
assert.equal(
  hoverHelpEl.classList.toggles.length,
  togglesBeforeQuickHover,
  'leaving before the delay elapses cancels the pending tooltip instead of showing it late',
);

// Keyboard focus and click stay instant (accessibility) -- no delay, no timer needed.
helpPanelListeners.focus();
assert.equal(
  hoverHelpEl.classList.toggles.at(-1)?.enabled,
  true,
  'keyboard focus shows the tooltip immediately, without waiting for the hover delay',
);
helpPanelListeners.blur();

console.log('extension_popup_contract=pass');

