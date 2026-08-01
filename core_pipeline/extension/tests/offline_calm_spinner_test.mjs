// Proves content.js's bounded offline-spinner behavior (item 2 of the Phase 4 plan): the
// PIPELINE_OFFLINE retry loop itself stays unbounded/deliberate (no cap, no error badge -- a long
// backend outage should not spam per-image failure state), but after
// OFFLINE_SPINNER_CALM_THRESHOLD consecutive offline responses for the same image, the visible
// spinner switches to a calm/static "waiting" state instead of continuing to animate as if
// progress were happening. The retry loop must keep running underneath regardless, and normal
// spinning must resume automatically the moment a later retry gets ANY non-PIPELINE_OFFLINE
// response (background.js's circuit breaker only ever returns PIPELINE_OFFLINE while tripped, so
// that is the real "backend is back online" signal for this per-image loop).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const contentPath = path.join(extensionRoot, 'content.js');
const source = fs.readFileSync(contentPath, 'utf8');

let contentListener;
const appendedNodes = [];
const translatedAttrs = new Map();
// Each 'translateImage' send() pushes a resolver here instead of resolving immediately -- the
// test drives the retry loop one round-trip at a time, so it can inspect spinner state between
// every response instead of racing a fully-synchronous flood of instant responses.
const pendingSendResolvers = [];

const fakeImage = {
  nodeName: 'IMG',
  complete: true,
  naturalWidth: 900,
  naturalHeight: 1300,
  width: 900,
  height: 1300,
  src: 'https://example.test/offline-calm-page.jpg',
  currentSrc: 'https://example.test/offline-calm-page.jpg',
  srcset: '',
  isConnected: true,
  hasAttribute(name) { return translatedAttrs.has(name); },
  getAttribute(name) { return translatedAttrs.has(name) ? translatedAttrs.get(name) : null; },
  setAttribute(name, value) { translatedAttrs.set(name, String(value)); },
  removeAttribute(name) { translatedAttrs.delete(name); },
  addEventListener() {},
  getBoundingClientRect() {
    return { left: 20, top: 30, right: 920, bottom: 1330, width: 900, height: 1300 };
  },
};

function createElement(tag) {
  if (tag === 'canvas') {
    return {
      width: 0,
      height: 0,
      getContext: () => ({
        fillStyle: '#fff',
        fillRect() {},
        drawImage() {},
        save() {},
        restore() {},
      }),
      toDataURL: () => 'data:image/jpeg;base64,ZmFrZS1jYW52YXM=',
    };
  }
  const listeners = new Map();
  const classSet = new Set();
  return {
    tagName: tag.toUpperCase(),
    className: '',
    style: {},
    textContent: '',
    title: '',
    id: '',
    classList: {
      add: (...names) => names.forEach((name) => classSet.add(name)),
      remove: (...names) => names.forEach((name) => classSet.delete(name)),
      contains: (name) => classSet.has(name),
    },
    addEventListener(type, handler) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(handler);
    },
    remove() { this.removed = true; },
  };
}

const sandbox = {
  console,
  // Mirrors content_contract_test.mjs's own documented sandbox pattern: only short (<=20ms)
  // timers fire for real, so unrelated long-delay reschedule loops elsewhere in content.js (e.g.
  // the passive-cache-restore recheck) stay inert instead of spinning immediately and flooding the
  // page with extra translateImage() attempts for the same image. The one exception is the exact
  // PIPELINE_OFFLINE_RETRY_DELAY_MS (12000ms) constant under test here -- that one fires promptly
  // (with a tiny real delay) so this test can actually drive the offline retry loop forward.
  setTimeout: (callback, delay = 0) => {
    if (delay === 12000) return setTimeout(callback, 0);
    if (delay <= 20) return setTimeout(callback, delay);
    return 0;
  },
  clearTimeout,
  requestAnimationFrame: () => 1,
  cancelAnimationFrame: () => {},
  window: {
    __mangaTranslatorInjected: false,
    innerWidth: 1200,
    innerHeight: 1600,
    location: { href: 'https://example.test/chapter/1' },
    addEventListener() {},
  },
  document: {
    readyState: 'complete',
    contentType: 'text/html',
    body: {
      get children() { return [fakeImage]; },
      appendChild(node) { appendedNodes.push(node); },
    },
    documentElement: {
      appendChild(node) { appendedNodes.push(node); },
      classList: {
        _set: new Set(),
        add(cls) { this._set.add(cls); },
        remove(cls) { this._set.delete(cls); },
        contains(cls) { return this._set.has(cls); },
      },
    },
    head: { appendChild(node) { appendedNodes.push(node); } },
    getElementById: () => null,
    elementsFromPoint: () => [],
    createElement,
    querySelectorAll(selector) {
      if (selector === 'img') return [fakeImage];
      if (selector === '[data-fmt-processing]') return translatedAttrs.get('data-fmt-processing') ? [fakeImage] : [];
      if (selector === '[data-fmt-translated]') return translatedAttrs.get('data-fmt-translated') ? [fakeImage] : [];
      if (selector === '[data-fmt-manually-cleared]') return [];
      return [];
    },
    querySelector(selector) { return selector === 'img' ? fakeImage : null; },
    addEventListener() {},
    removeEventListener() {},
  },
  MutationObserver: class { constructor() {} observe() {} },
  IntersectionObserver: class { observe() {} },
  Image: class {},
  chrome: {
    runtime: {
      getURL: (asset) => `chrome-extension://test/${asset}`,
      onMessage: { addListener: (listener) => { contentListener = listener; } },
      sendMessage: async (message) => {
        if (message.kind === 'lookupCachedTranslation') return { hit: false, inFlight: false };
        if (message.kind === 'translateImage') {
          return new Promise((resolve) => { pendingSendResolvers.push(resolve); });
        }
        return {};
      },
    },
    storage: {
      local: {
        get: (keys, callback) => callback({ translationEnabled: false, translationPaused: false, translationQueuePages: 1 }),
      },
      onChanged: { addListener: () => {} },
    },
  },
};

sandbox.window.window = sandbox.window;

vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: contentPath });
assert.equal(typeof contentListener, 'function', 'content message listener registered');

function currentSpinner() {
  return appendedNodes.filter((node) => node.className === 'fmt-img-spinner' && !node.removed).at(-1);
}

async function releaseOneOffline() {
  await new Promise((resolve) => setTimeout(resolve, 5)); // let the pending send() land
  const resolver = pendingSendResolvers.shift();
  assert.ok(resolver, 'a translateImage request was in flight to release');
  resolver({ error: 'PIPELINE_OFFLINE' });
  await new Promise((resolve) => setTimeout(resolve, 15)); // let the response + retry scheduling settle
}

// translateSpecificImage (the right-click "Translate this manga panel" path) dispatches
// translateImage exactly once for the matched image -- unlike translatePageOnce/toggleTranslation,
// which independently run both scanForImages() and handleStandaloneImage() and can each pick up
// the same image, this keeps the retry loop under test to a single, unambiguous in-flight request.
contentListener({ kind: 'translateSpecificImage', imageUrl: fakeImage.src }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));

// Drive 8 consecutive PIPELINE_OFFLINE round-trips -- comfortably past
// OFFLINE_SPINNER_CALM_THRESHOLD (6) -- for the same image.
for (let i = 0; i < 8; i += 1) {
  await releaseOneOffline();
}

const spinner = currentSpinner();
assert.ok(spinner, 'spinner is still shown through the offline retry loop');
assert.ok(
  spinner.classList.contains('fmt-img-spinner--calm'),
  'spinner enters the calm/waiting state after enough consecutive offline responses',
);
assert.equal(
  translatedAttrs.get('data-fmt-processing'),
  'true',
  'the retry loop is still actively running (image still marked processing) even though the spinner went calm',
);

// Now the backend recovers: the next retry gets a real, successful response.
await new Promise((resolve) => setTimeout(resolve, 5));
const finalResolver = pendingSendResolvers.shift();
assert.ok(finalResolver, 'the retry loop kept running underneath and issued another request');
finalResolver({ translatedImageDataUrl: 'data:image/png;base64,cmVjb3ZlcmVk' });
await new Promise((resolve) => setTimeout(resolve, 15));

assert.equal(
  translatedAttrs.get('data-fmt-translated'),
  'true',
  'the image recovers and translates normally once a response stops being PIPELINE_OFFLINE',
);
const spinnerAfterRecovery = currentSpinner();
assert.ok(
  !spinnerAfterRecovery || spinnerAfterRecovery.removed || !spinnerAfterRecovery.classList.contains('fmt-img-spinner--calm'),
  'the spinner is no longer stuck in the calm state once translation succeeds',
);

console.log('extension_offline_calm_spinner=pass');
