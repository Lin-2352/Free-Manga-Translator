import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const contentPath = path.join(extensionRoot, 'content.js');
const source = fs.readFileSync(contentPath, 'utf8');

let contentListener;
let mutationCallback;
const windowListeners = {};
const translatedAttrs = new Map();
const removedAttrs = [];
const appendedNodes = [];
const translateRequests = [];
const lookupRequests = [];
let lookupHit = false;
let cachedDataUrl = 'data:image/png;base64,Y2FjaGVk';
let fakeRect = { left: 20, top: 30, right: 920, bottom: 1330, width: 900, height: 1300 };
let holdPage4Translation = false;
let resolvePage4Translation = null;
let holdUrl = null;
let resolveHeldUrl = null;
let noTextRescuedUrl = null;
let errorTriggerUrl = null;
const documentListeners = new Map();
let fakePanelOverlayPresent = false;
let elementsFromPointStack = [];

const fakeImage = {
  nodeName: 'IMG',
  complete: true,
  naturalWidth: 900,
  naturalHeight: 1300,
  width: 900,
  height: 1300,
  src: 'https://example.test/page-1.jpg',
  currentSrc: 'https://example.test/page-1.jpg',
  srcset: '',
  isConnected: true,
  getAttribute(name) {
    return translatedAttrs.get(name) || '';
  },
  setAttribute(name, value) {
    translatedAttrs.set(name, String(value));
  },
  removeAttribute(name) {
    removedAttrs.push(name);
    translatedAttrs.delete(name);
  },
  addEventListener() {},
  getBoundingClientRect() {
    return fakeRect;
  },
};
let fakeImages = [fakeImage];

function makeQueueImage(url, rect, width = 900, height = 1300) {
  const attrs = new Map();
  return {
    nodeName: 'IMG',
    complete: true,
    naturalWidth: width,
    naturalHeight: height,
    width,
    height,
    src: url,
    currentSrc: url,
    srcset: '',
    isConnected: true,
    getAttribute(name) {
      return attrs.get(name) || '';
    },
    setAttribute(name, value) {
      attrs.set(name, String(value));
    },
    removeAttribute(name) {
      attrs.delete(name);
    },
    addEventListener() {},
    getBoundingClientRect() {
      return rect;
    },
  };
}

function createElement(tag) {
  if (tag === 'canvas') {
    return {
      width: 0,
      height: 0,
      getContext: () => ({
        fillStyle: '#fff',
        fillRect() {},
        drawImage() {},
        measureText(text) {
          return { width: String(text).length * 10 };
        },
        save() {},
        restore() {},
        beginPath() {},
        rect() {},
        clip() {},
        strokeText() {},
        fillText() {},
      }),
      toDataURL: () => 'data:image/jpeg;base64,ZmFrZS1pbWFnZQ==',
    };
  }
  const listeners = new Map();
  return {
    tagName: tag.toUpperCase(),
    className: '',
    style: {},
    textContent: '',
    title: '',
    id: '',
    listeners,
    addEventListener(type, handler) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(handler);
    },
    remove() {
      this.removed = true;
    },
  };
}

const sandbox = {
  console,
  setTimeout: (callback, delay = 0) => {
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
    addEventListener(type, listener) {
      (windowListeners[type] ||= []).push(listener);
    },
  },
  document: {
    readyState: 'complete',
    contentType: 'text/html',
    body: {
      get children() {
        return fakeImages;
      },
      appendChild(node) {
        appendedNodes.push(node);
      },
    },
    documentElement: {
      appendChild(node) {
        appendedNodes.push(node);
      },
      classList: {
        _set: new Set(),
        add(cls) { this._set.add(cls); },
        remove(cls) { this._set.delete(cls); },
        contains(cls) { return this._set.has(cls); },
      },
    },
    head: {
      appendChild(node) {
        appendedNodes.push(node);
      },
    },
    getElementById: (id) => (id === 'fmt-panel-overlay' && fakePanelOverlayPresent ? {} : null),
    elementsFromPoint: (x, y) => elementsFromPointStack,
    createElement,
    querySelectorAll(selector) {
      if (selector === 'img') return fakeImages;
      if (selector === '[data-fmt-processing]') {
        return fakeImages.filter((image) => image.getAttribute?.('data-fmt-processing'));
      }
      if (selector === '[data-fmt-translated]') {
        return fakeImages.filter((image) => image.getAttribute?.('data-fmt-translated'));
      }
      if (selector === '[data-fmt-manually-cleared]') {
        return fakeImages.filter((image) => image.getAttribute?.('data-fmt-manually-cleared'));
      }
      return [];
    },
    querySelector(selector) {
      return selector === 'img' ? fakeImages[0] || null : null;
    },
    addEventListener(type, handler) {
      documentListeners.set(type, handler);
    },
    removeEventListener(type, handler) {
      if (documentListeners.get(type) === handler) documentListeners.delete(type);
    },
  },
  MutationObserver: class {
    constructor(callback) {
      mutationCallback = callback;
    }
    observe() {}
  },
  IntersectionObserver: class {
    observe() {}
  },
  Image: class {},
  chrome: {
    runtime: {
      getURL: (asset) => `chrome-extension://test/${asset}`,
      onMessage: {
        addListener(listener) {
          contentListener = listener;
        },
      },
      sendMessage: async (message) => {
        if (message.kind === 'lookupCachedTranslation') {
          lookupRequests.push(message);
          if (lookupHit) {
            return {
              hit: true,
              fromCache: true,
              translatedImageDataUrl: cachedDataUrl,
            };
          }
          return { hit: false, inFlight: false };
        }
        if (message.kind === 'translateImage') translateRequests.push(message);
        if (message.kind === 'translateImage' && message.originalImageUrl === errorTriggerUrl) {
          return { error: 'LOCAL_PIPELINE_500' };
        }
        if (message.kind === 'translateImage' && message.originalImageUrl === noTextRescuedUrl) {
          return {
            translatedImageDataUrl: 'data:image/png;base64,dW50cmFuc2xhdGVkLW9yaWdpbmFs',
            pipelineReport: { noRenderableText: true, rescueAttempted: true, rescueSucceeded: false },
          };
        }
        if (message.kind === 'translateImage' && holdPage4Translation && message.originalImageUrl === 'https://example.test/page-4.jpg') {
          return new Promise((resolve) => {
            resolvePage4Translation = () => resolve({ translatedImageDataUrl: 'data:image/png;base64,cGFnZS00' });
          });
        }
        if (message.kind === 'translateImage' && holdUrl && message.originalImageUrl === holdUrl) {
          return new Promise((resolve) => {
            resolveHeldUrl = () => resolve({ translatedImageDataUrl: 'data:image/png;base64,c3RhbGU=' });
          });
        }
        return { translatedImageDataUrl: 'data:image/png;base64,dHJhbnNsYXRlZA==' };
      },
    },
    storage: {
      local: {
        get: (keys, callback) => callback({
          translationEnabled: false,
          translationPaused: false,
          translationQueuePages: 1,
        }),
      },
      onChanged: { addListener: () => {} },
    },
  },
};

sandbox.window.window = sandbox.window;

vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: contentPath });

assert.equal(typeof contentListener, 'function', 'content message listener registered');

function documentElement_isPicking() {
  return sandbox.document.documentElement.classList.contains('fmt-picking');
}

contentListener({ kind: 'toggleTranslation', enabled: true }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));

assert.equal(translateRequests.length, 1, 'content script sent one translateImage request');
assert.equal(translateRequests[0].pageUrl, 'https://example.test/chapter/1');
assert.equal(translateRequests[0].originalImageUrl, 'https://example.test/page-1.jpg');
assert.equal(translateRequests[0].cacheKey, 'https://example.test/page-1.jpg|900x1300');
assert.equal(translateRequests[0].pageCacheKey, 'https://example.test/chapter/1');
assert.equal(translateRequests[0].width, 900);
assert.equal(translateRequests[0].height, 1300);
assert.equal(fakeImage.src, 'data:image/png;base64,dHJhbnNsYXRlZA==');
assert.equal(translatedAttrs.get('data-fmt-translated'), 'true');
assert.equal(translatedAttrs.get('data-fmt-original-src'), 'https://example.test/page-1.jpg');
assert.equal(translatedAttrs.get('data-fmt-cache-key'), 'https://example.test/page-1.jpg|900x1300');
assert.equal(
  appendedNodes.some((node) => node.className === 'fmt-img-spinner'),
  true,
  'content script displayed the per-image loader',
);
assert.equal(
  appendedNodes.find((node) => node.className === 'fmt-img-spinner')?.removed,
  true,
  'content script removed the per-image loader after applying the translated image',
);
assert.equal(
  translatedAttrs.has('data-fmt-processing'),
  false,
  'content script clears processing state after applying the translated image',
);

const translatedRequestCount = translateRequests.length;
fakeImage.src = 'data:image/png;base64,dHJhbnNsYXRlZA==';
fakeImage.currentSrc = 'data:image/png;base64,dHJhbnNsYXRlZA==';
fakeImage.naturalWidth = 1200;
fakeImage.naturalHeight = 1800;
translatedAttrs.delete('data-fmt-translated');
translatedAttrs.delete('data-fmt-translated-src');
lookupHit = true;

contentListener({ kind: 'translatePageOnce' }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));

assert.equal(lookupRequests.length >= 2, true, 'content script looked up cached translations');
assert.equal(lookupRequests.at(-1).cacheKey, 'https://example.test/page-1.jpg|900x1300', 'cache lookup keeps original dimensions after translated replacement');
assert.equal(translateRequests.length, translatedRequestCount, 'cached translated data URL was not sent back to the pipeline');
assert.equal(fakeImage.src, 'data:image/png;base64,Y2FjaGVk');

lookupHit = false;
fakeImage.naturalWidth = 900;
fakeImage.naturalHeight = 1300;
fakeImage.src = 'https://example.test/page-2.jpg';
fakeImage.currentSrc = 'https://example.test/page-2.jpg';
contentListener({ kind: 'translatePageOnce' }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));

assert.equal(
  translateRequests.at(-1).originalImageUrl,
  'https://example.test/page-2.jpg',
  'reused image node gets a fresh original source',
);
assert.equal(translatedAttrs.get('data-fmt-original-src'), 'https://example.test/page-2.jpg');

const beforePage3Requests = translateRequests.length;
cachedDataUrl = 'data:image/png;base64,cGFnZS0zLWNhY2hlZA==';
lookupHit = true;
fakeImage.src = 'https://example.test/page-3.jpg';
fakeImage.currentSrc = 'https://example.test/page-3.jpg';
contentListener({ kind: 'translatePageOnce' }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));

assert.equal(translateRequests.length, beforePage3Requests, 'page 3 cache restored without a new backend request');
assert.equal(translatedAttrs.get('data-fmt-original-src'), 'https://example.test/page-3.jpg');
assert.equal(fakeImage.src, 'data:image/png;base64,cGFnZS0zLWNhY2hlZA==');

const beforeForceRetranslate = translateRequests.length;
lookupHit = false;
translatedAttrs.set('data-fmt-translated', 'no-text');
contentListener({ kind: 'translatePageOnce' }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));

assert.equal(
  translateRequests.length,
  beforeForceRetranslate + 1,
  'force translate resets stale translated/no-text state and sends backend work',
);
assert.equal(
  translateRequests.at(-1).originalImageUrl,
  'https://example.test/page-3.jpg',
  'force translate keeps the stored original source after resetting translated state',
);

const removedBeforePause = removedAttrs.length;
translatedAttrs.set('data-fmt-processing', 'true');
let pauseResponse = null;
contentListener({ kind: 'setTranslationPaused', paused: true }, {}, (response) => {
  pauseResponse = response;
});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(pauseResponse.success, true, 'pause command responds to background');
assert.equal(
  removedAttrs.slice(removedBeforePause).includes('data-fmt-processing'),
  true,
  'pause command clears active processing state so loaders stop',
);
contentListener({ kind: 'setTranslationPaused', paused: false }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));

const removedBeforeClearTranslations = removedAttrs.length;
translatedAttrs.set('data-fmt-processing', 'true');
let clearTranslationsResponse = null;
contentListener({ kind: 'clearTranslations' }, {}, (response) => {
  clearTranslationsResponse = response;
});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(clearTranslationsResponse.success, true, 'clear translations responds to popup');
assert.equal(
  removedAttrs.slice(removedBeforeClearTranslations).includes('data-fmt-processing'),
  true,
  'clear translations also clears active processing loaders',
);

lookupHit = false;
holdPage4Translation = true;
fakeRect = { left: 20, top: 30, right: 21, bottom: 31, width: 1, height: 1 };
fakeImage.src = 'https://example.test/page-4.jpg';
fakeImage.currentSrc = 'https://example.test/page-4.jpg';
translatedAttrs.delete('data-fmt-translated');
translatedAttrs.delete('data-fmt-translated-src');

const spinnerCountBeforeHiddenPage4 = appendedNodes.filter((node) => node.className === 'fmt-img-spinner').length;
contentListener({ kind: 'translatePageOnce' }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(
  appendedNodes.filter((node) => node.className === 'fmt-img-spinner').length,
  spinnerCountBeforeHiddenPage4,
  'hidden pending image did not create a loader before it became visible',
);

fakeRect = { left: 20, top: 30, right: 920, bottom: 1330, width: 900, height: 1300 };
contentListener({ kind: 'translatePageOnce' }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(
  appendedNodes.filter((node) => node.className === 'fmt-img-spinner').length > spinnerCountBeforeHiddenPage4,
  true,
  'pending visible image restores the per-image loader',
);
resolvePage4Translation?.();
await new Promise((resolve) => setTimeout(resolve, 10));

const beforeBoundedAutoRequests = translateRequests.length;
lookupHit = false;
holdPage4Translation = false;
fakeImages = [
  makeQueueImage('https://example.test/near-page.jpg', { left: 10, top: 10, right: 910, bottom: 1310, width: 900, height: 1300 }),
  makeQueueImage('https://example.test/far-page.jpg', { left: 10, top: 5000, right: 910, bottom: 6300, width: 900, height: 1300 }),
];
contentListener({ kind: 'toggleTranslation', enabled: true }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));

assert.equal(
  translateRequests.length,
  beforeBoundedAutoRequests + 1,
  'auto mode respects queue-ahead limit before sending backend work',
);
assert.equal(
  translateRequests.at(-1).originalImageUrl,
  'https://example.test/near-page.jpg',
  'auto mode prioritizes nearest visible image for queue-ahead',
);

const beforeThumbnailFilterRequests = translateRequests.length;
fakeImages = [
  makeQueueImage(
    'https://naverwebtoon-phinf.pstatic.net/thumb.JPEG?type=p100',
    { left: 10, top: 10, right: 242, bottom: 234, width: 232, height: 224 },
    232,
    224,
  ),
  makeQueueImage(
    'https://image-comic.pstatic.net/mobilewebimg/page.jpg',
    { left: 10, top: 20, right: 700, bottom: 860, width: 690, height: 840 },
    690,
    840,
  ),
];
contentListener({ kind: 'translatePageOnce' }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(
  translateRequests.length,
  beforeThumbnailFilterRequests + 1,
  'force page translation skips low-resolution reader thumbnails',
);
assert.equal(
  translateRequests.at(-1).originalImageUrl,
  'https://image-comic.pstatic.net/mobilewebimg/page.jpg',
  'thumbnail filtering leaves the actual manga page image',
);

// ===== Small real manga page (358x520, from the localhost:3000 test site) is auto-detected =====
// This dimension previously missed MIN_PAGE_IMAGE_SIZE (360) by 2px on the min-side gate alone;
// the lowered 340/170000 thresholds must catch it while the 232x224 thumbnail above still fails.
const beforeSmallPageRequests = translateRequests.length;
fakeImages = [
  makeQueueImage('https://example.test/small-manga-page.jpg', { left: 10, top: 10, right: 368, bottom: 530, width: 358, height: 520 }, 358, 520),
];
contentListener({ kind: 'toggleTranslation', enabled: true }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(
  translateRequests.length,
  beforeSmallPageRequests + 1,
  'a 358x520 manga page is auto-detected and translated, not filtered out as too small',
);
assert.equal(
  translateRequests.at(-1).originalImageUrl,
  'https://example.test/small-manga-page.jpg',
);
contentListener({ kind: 'toggleTranslation', enabled: false }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));

// ===== bfcache restore recovers a stuck PROCESSING marker =====
// A back/forward navigation freezes the page (with any PROCESSING_ATTR/
// pendingSrcs state) mid-request; the message port that request was awaiting
// dies with the frozen page, so its promise never settles on its own. Without
// recovery the image is stuck spinning forever and is never rescanned.
// Uses a URL never touched by any earlier block in this file -- reusing one
// would risk a stale (but harmless in real usage; setTimeout(...,1500) never
// fires in this sandbox's mock) cacheMissSrcs negative-cache entry leaking in
// and masking whether the cache hit below is real.
fakeImages = [fakeImage];
fakeRect = { left: 20, top: 30, right: 920, bottom: 1330, width: 900, height: 1300 };
lookupHit = false;
holdUrl = 'https://example.test/page-9-bfcache.jpg';
fakeImage.src = holdUrl;
fakeImage.currentSrc = holdUrl;
translatedAttrs.delete('data-fmt-translated');
translatedAttrs.delete('data-fmt-translated-src');
translatedAttrs.delete('data-fmt-processing');

contentListener({ kind: 'translatePageOnce' }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(
  translatedAttrs.has('data-fmt-processing'),
  true,
  'a held translation request leaves the image marked as processing',
);

assert.equal(
  Array.isArray(windowListeners.pageshow) && windowListeners.pageshow.length > 0,
  true,
  'content script registered a pageshow listener',
);
windowListeners.pageshow.forEach((listener) => listener({ persisted: true }));
await new Promise((resolve) => setTimeout(resolve, 10));

assert.equal(
  translatedAttrs.has('data-fmt-processing'),
  false,
  'a bfcache restore (pageshow persisted:true) clears the stale processing marker',
);

// After recovery, a normal (non-forced) rescan -- as the real pageshow
// handler's scheduleNavigationScan eventually triggers -- must be able to
// pick the image back up on its own, proving it is not permanently stuck.
// (Not testing this via a cache hit: this sandbox's setTimeout mock silently
// drops delays over 20ms, so lookupCachedTranslation's own 1500ms
// cacheMissSrcs negative-cache entry -- recorded by the very first lookup
// above, before the hold was armed -- never naturally expires here. A fresh
// dispatch + successful apply is an equally meaningful, sandbox-safe signal
// that recovery worked.)
const recoveringUrl = holdUrl;
holdUrl = null; // the recovery scan's own new request must resolve normally, not get held too
const requestsBeforeRecoveryScan = translateRequests.length;
contentListener({ kind: 'toggleTranslation', enabled: true }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(
  translateRequests.slice(requestsBeforeRecoveryScan).some((message) => message.originalImageUrl === recoveringUrl),
  true,
  'the recovered image is rescanned and re-dispatched, proving it is not permanently stuck',
);
assert.notEqual(
  fakeImage.src,
  recoveringUrl,
  'the recovered rescan applied a real translation result to the image',
);
const recoveredSrc = fakeImage.src;

// The original held request's dead-port promise resolving late (Chrome does
// not actually deliver this in practice -- the port is invalidated by the
// freeze -- but the generation guard must hold even if it somehow did) must
// not clobber the now-correctly-recovered image with its stale response.
resolveHeldUrl?.();
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(
  fakeImage.src,
  recoveredSrc,
  'a late response from the pre-bfcache request cannot overwrite the recovered image',
);
holdUrl = null;

// ===== MutationObserver self-heals a phantom translatedSrcs entry =====
// Reproduces: user navigates through several pages on a site that swaps
// `src` on a single reused <img> node (no real navigation event). If a
// translate response lands after the node has already moved on to a
// different src, applyTranslatedImage's staleness check discards it --
// but translateImage() already added the cache key to translatedSrcs
// unconditionally before that check runs, leaving a "phantom" entry that
// blocks a later retry when the user navigates back to the original src.
// shouldTranslate()/processImage() already self-heal this; the observer's
// own attribute-change branch did not.
fakeImages = [fakeImage];
fakeRect = { left: 20, top: 30, right: 920, bottom: 1330, width: 900, height: 1300 };
lookupHit = false;
translatedAttrs.clear();
const observerHealUrl = 'https://example.test/observer-heal.jpg';
holdUrl = observerHealUrl;
fakeImage.src = observerHealUrl;
fakeImage.currentSrc = observerHealUrl;

contentListener({ kind: 'translatePageOnce' }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(
  translatedAttrs.has('data-fmt-processing'),
  true,
  'observer-heal setup: the initial translate request is in flight',
);

// Simulate the node having already been reused for a different page by the
// time the held response lands -- exactly what getOriginalSrc()'s "reused
// image node detected" path does in real usage.
fakeImage.setAttribute('data-fmt-original-src', 'https://example.test/observer-heal-other.jpg');
fakeImage.src = 'https://example.test/observer-heal-other.jpg';
fakeImage.currentSrc = 'https://example.test/observer-heal-other.jpg';

resolveHeldUrl?.();
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(
  translatedAttrs.has('data-fmt-translated'),
  false,
  'observer-heal setup: the late response was discarded as stale, leaving translatedSrcs holding a phantom entry',
);

// User navigates back to the original src on the same reused node, with
// auto-translate off (matches the reported "AFK, some pages never finish"
// scenario) so the observer takes its synchronous restoreOnly path instead
// of a setTimeout-scheduled scan (this sandbox's setTimeout mock drops any
// delay over 20ms, same limitation the bfcache test above documents).
// lookupCachedTranslation's own negative-cache (cacheMissSrcs, real 1500ms
// TTL) was already populated for this exact cacheKey by the miss above and
// can never expire here either, so this step also sets img.src to a data:
// URL with ORIGINAL_SRC_ATTR already pointing at observerHealUrl -- the one
// condition (needsTranslatedDataRestore) that legitimately bypasses that
// negative cache in the real code -- purely so the restore attempt is
// observable in this sandbox. The precondition under test (the phantom
// translatedSrcs entry with no TRANSLATED_ATTR) was already established
// realistically above; only this final probe is sandbox-shaped.
contentListener({ kind: 'toggleTranslation', enabled: false }, {}, () => {});
lookupHit = true;
fakeImage.setAttribute('data-fmt-original-src', observerHealUrl);
fakeImage.src = cachedDataUrl;
fakeImage.currentSrc = cachedDataUrl;
mutationCallback([{ type: 'attributes', target: fakeImage }]);
await new Promise((resolve) => setTimeout(resolve, 10));

assert.equal(
  translatedAttrs.has('data-fmt-translated'),
  true,
  'the observer self-heals the phantom translatedSrcs entry and restores the cached translation on revisit',
);
holdUrl = null;
lookupHit = false;

// ===== Clear Page does not silently auto-restore the translation =====
// Reproduces: with auto-translate off, Clear Page should leave the image
// showing its untranslated original until the user explicitly re-translates
// it. restoreOriginalImage()'s own `img.src = originalSrc` write fires the
// MutationObserver, and passive cache-restore is deliberately built to
// bypass the auto-translate toggle (so revisiting an untouched, previously
// translated page still restores) -- that collided with an unhandled case:
// nothing distinguished "revisited" from "just explicitly cleared".
fakeImages = [fakeImage];
translatedAttrs.clear();
const clearPageUrl = 'https://example.test/clear-page-no-autorestore.jpg';
fakeImage.src = clearPageUrl;
fakeImage.currentSrc = clearPageUrl;
lookupHit = false;

contentListener({ kind: 'translatePageOnce' }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(
  translatedAttrs.has('data-fmt-translated'),
  true,
  'clear-page setup: the image is genuinely translated and cached first',
);

lookupHit = true; // Clear Page intentionally leaves the cache entry intact
contentListener({ kind: 'toggleTranslation', enabled: false }, {}, () => {});
contentListener({ kind: 'clearTranslations' }, {}, () => {});
assert.equal(
  translatedAttrs.has('data-fmt-translated'),
  false,
  'Clear Page visually reverts the image immediately',
);
assert.equal(
  translatedAttrs.get('data-fmt-manually-cleared'),
  '1',
  'Clear Page marks the image as manually cleared',
);

// The real browser fires the observer automatically when restoreOriginalImage
// writes img.src back to the original; the sandbox's MutationObserver stub
// does not observe, so simulate that callback firing here.
mutationCallback([{ type: 'attributes', target: fakeImage }]);
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(
  translatedAttrs.has('data-fmt-translated'),
  false,
  'Clear Page must NOT auto-restore the translation on its own -- only an explicit user action may',
);

// An explicit re-translate afterward must still work normally.
contentListener({ kind: 'translatePageOnce' }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(
  translatedAttrs.has('data-fmt-translated'),
  true,
  'an explicit Translate Page click after Clear Page still restores the translation',
);
lookupHit = false;
contentListener({ kind: 'toggleTranslation', enabled: false }, {}, () => {});

// ===== Clear Page must not auto-restore even with auto-translate ON =====
// The MANUAL_CLEAR_ATTR gate originally only fired in restoreOnly mode. With
// auto-translate on, ordinary non-forced rescans (the intersection observer,
// setTranslationPaused resuming, navigation scans) call processImage with
// restoreOnly:false and could bypass the gate entirely, re-applying a cached
// translation to an image the user just explicitly cleared.
fakeImages = [fakeImage];
translatedAttrs.clear();
const clearPageAutoOnUrl = 'https://example.test/clear-page-auto-on.jpg';
fakeImage.src = clearPageAutoOnUrl;
fakeImage.currentSrc = clearPageAutoOnUrl;
lookupHit = true; // simulates an image already cached from an earlier session

contentListener({ kind: 'toggleTranslation', enabled: true }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(
  translatedAttrs.has('data-fmt-translated'),
  true,
  'auto-on setup: the image restores from cache once auto-translate is enabled',
);

contentListener({ kind: 'clearTranslations' }, {}, () => {});
assert.equal(
  translatedAttrs.has('data-fmt-translated'),
  false,
  'Clear Page reverts the image even while auto-translate stays on',
);

// A legitimate non-forced rescan while still enabled and paused/resumed --
// the exact shape of trigger an intersection-observer or navigation rescan
// takes -- must not resurrect the cleared image.
contentListener({ kind: 'setTranslationPaused', paused: true }, {}, () => {});
contentListener({ kind: 'setTranslationPaused', paused: false }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(
  translatedAttrs.has('data-fmt-translated'),
  false,
  'Clear Page must NOT auto-restore on a later non-forced rescan even with auto-translate on',
);

// Re-enabling auto-translate (even though it is already on here, this is
// the same code path a real off-then-on toggle takes) is itself explicit
// user intent and must lift the suppression.
contentListener({ kind: 'translatePageOnce' }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(
  translatedAttrs.has('data-fmt-translated'),
  true,
  'an explicit Translate Page click still restores the translation after the auto-on clear',
);
lookupHit = false;

// ===== Clear Page's suppression marker must not survive a src swap on a reused node =====
// SPA readers (the localhost:3000 test site included) reuse a single <img> element and just
// swap its src between "pages" -- getOriginalSrc() detects that as a different logical image
// and calls clearImageRuntimeState(), which must also drop MANUAL_CLEAR_ATTR. Without that, one
// Clear Page silently suppresses auto-translation of every image the site swaps in afterward.
contentListener({ kind: 'clearTranslations' }, {}, () => {});
assert.equal(
  translatedAttrs.get('data-fmt-manually-cleared'),
  '1',
  'reused-node setup: Clear Page marks the current image as manually cleared',
);
const reusedNodeNewUrl = 'https://example.test/spa-reused-node-next-page.jpg';
fakeImage.src = reusedNodeNewUrl;
fakeImage.currentSrc = reusedNodeNewUrl;
lookupHit = true; // simulates this "new" image already being cached from an earlier session
// Deliberately NOT translatePageOnce/force -- force already bypasses the MANUAL_CLEAR_ATTR gate
// on its own, which would prove nothing about whether the marker itself got cleared. This is the
// same non-forced-rescan shape (pause/resume) used above to prove the gate actually fires.
contentListener({ kind: 'setTranslationPaused', paused: true }, {}, () => {});
contentListener({ kind: 'setTranslationPaused', paused: false }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.ok(
  !translatedAttrs.get('data-fmt-manually-cleared'),
  'a src swap onto a different logical image clears the stale manually-cleared marker',
);
assert.equal(
  translatedAttrs.has('data-fmt-translated'),
  true,
  'the new image auto-translates normally, unsuppressed by the previous image\'s Clear Page',
);
lookupHit = false;

// ===== A vision-rescue-confirmed blank page must be marked no-text
// immediately, without ever applying the backend's "unchanged original
// image" response as if it were a real translation. =====
fakeImages = [fakeImage];
translatedAttrs.clear();
const noTextUrl = 'https://example.test/no-text-page.jpg';
fakeImage.src = noTextUrl;
fakeImage.currentSrc = noTextUrl;
noTextRescuedUrl = noTextUrl;
lookupHit = false;
const requestsBeforeNoText = translateRequests.length;

contentListener({ kind: 'translatePageOnce' }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));

assert.equal(
  translatedAttrs.get('data-fmt-translated'),
  'no-text',
  'a vision-rescue-confirmed blank page is marked no-text, not "true" as if translated',
);
assert.notEqual(
  fakeImage.src,
  'data:image/png;base64,dW50cmFuc2xhdGVkLW9yaWdpbmFs',
  'the backend\'s unchanged-original response is never applied to the image as a fake translation',
);
assert.equal(
  translateRequests.length,
  requestsBeforeNoText + 1,
  'a rescue-confirmed blank page is not retried -- exactly one request was sent',
);
noTextRescuedUrl = null;

// ===== A terminally-failed translation (a non-retryable backend error) must show a visible
// error badge instead of silently vanishing, and clicking it must retry the translation. =====
fakeImages = [fakeImage];
translatedAttrs.clear();
removedAttrs.length = 0;
const errorUrl = 'https://example.test/error-page.jpg';
fakeImage.src = errorUrl;
fakeImage.currentSrc = errorUrl;
errorTriggerUrl = errorUrl;
const appendedBeforeError = appendedNodes.length;
const requestsBeforeError = translateRequests.length;

contentListener({ kind: 'translatePageOnce' }, {}, () => {});
await new Promise((resolve) => setTimeout(resolve, 10));

assert.equal(translateRequests.length, requestsBeforeError + 1, 'the terminal error was not retried (not in the retryable set)');
assert.equal(translatedAttrs.has('data-fmt-processing'), false, 'the processing marker is cleared after a terminal failure');

const badgeNode = appendedNodes.slice(appendedBeforeError).find((node) => node.className === 'fmt-img-error-badge');
assert.ok(badgeNode, 'a visible error badge was appended for the failed image');
assert.ok(String(badgeNode.title).includes('LOCAL_PIPELINE_500'), 'the badge title carries the error text');

// Clicking the badge must retry via the same force+manualSpecific path right-click uses.
errorTriggerUrl = null; // the retry should succeed this time
const clickHandlers = badgeNode.listeners.get('click') || [];
assert.equal(clickHandlers.length, 1, 'badge has exactly one click handler');
clickHandlers[0]({ preventDefault() {}, stopPropagation() {} });
await new Promise((resolve) => setTimeout(resolve, 10));

assert.equal(translateRequests.length, requestsBeforeError + 2, 'clicking the badge issued a fresh translate request');
assert.equal(badgeNode.removed, true, 'the badge is removed as soon as it is clicked');
assert.equal(translatedAttrs.get('data-fmt-translated'), 'true', 'the retried translation succeeded and applied normally');

// ===== Panel picker (manual "inspect element"-style target selection) =====
const pickerCandidate = makeQueueImage(
  'https://example.test/picker-candidate.jpg',
  { left: 100, top: 100, right: 400, bottom: 500, width: 300, height: 400 },
  300,
  400,
);
elementsFromPointStack = [pickerCandidate];
fakePanelOverlayPresent = false;

let toggleResponse;
contentListener({ kind: 'togglePanelPicker' }, {}, (response) => { toggleResponse = response; });
assert.equal(toggleResponse?.success, true, 'the picker starts when nothing else is using page-level clicks');
assert.equal(toggleResponse?.active, true);
assert.equal(
  documentElement_isPicking(),
  true,
  'starting the picker adds the crosshair-cursor class to the document root',
);
const highlightNode = appendedNodes.filter((node) => node.className === 'fmt-picker-highlight').at(-1);
const hintNode = appendedNodes.filter((node) => node.className === 'fmt-picker-hint').at(-1);
assert.ok(highlightNode, 'a highlight element was appended when the picker started');
assert.ok(hintNode, 'a hint pill was appended when the picker started');

// Hovering over the candidate positions the highlight to match its bounding rect.
documentListeners.get('mousemove')?.({ clientX: 150, clientY: 150 });
assert.equal(highlightNode.style.display, 'block');
assert.equal(highlightNode.style.left, '100px');
assert.equal(highlightNode.style.top, '100px');
assert.equal(highlightNode.style.width, '300px');
assert.equal(highlightNode.style.height, '400px');

// Clicking the candidate force-translates it directly and exits the picker (single-shot).
const requestsBeforePickerClick = translateRequests.length;
let pickerClickDefaultPrevented = false;
documentListeners.get('click')?.({
  clientX: 150,
  clientY: 150,
  preventDefault: () => { pickerClickDefaultPrevented = true; },
  stopPropagation: () => {},
});
await new Promise((resolve) => setTimeout(resolve, 10));
assert.equal(pickerClickDefaultPrevented, true, 'the picker click intercepts the page\'s own click handling on the image');
assert.equal(translateRequests.length, requestsBeforePickerClick + 1, 'the picked image was force-translated directly, no URL round-trip needed');
assert.equal(translateRequests.at(-1).originalImageUrl, 'https://example.test/picker-candidate.jpg');
assert.equal(
  documentElement_isPicking(),
  false,
  'the picker exits itself immediately after a successful pick',
);
assert.equal(documentListeners.has('click'), false, 'picker listeners are removed on exit');
assert.equal(documentListeners.has('mousemove'), false);
assert.equal(documentListeners.has('keydown'), false);

// Esc exits without translating anything.
elementsFromPointStack = [pickerCandidate];
contentListener({ kind: 'togglePanelPicker' }, {}, () => {});
assert.equal(documentElement_isPicking(), true, 'picker re-started for the Esc test');
const requestsBeforeEsc = translateRequests.length;
documentListeners.get('keydown')?.({ key: 'Escape' });
assert.equal(documentElement_isPicking(), false, 'Esc exits the picker');
assert.equal(translateRequests.length, requestsBeforeEsc, 'Esc does not translate anything');

// Toggling again while active turns it off (not a second start).
contentListener({ kind: 'togglePanelPicker' }, {}, () => {});
assert.equal(documentElement_isPicking(), true);
let secondToggleResponse;
contentListener({ kind: 'togglePanelPicker' }, {}, (response) => { secondToggleResponse = response; });
assert.equal(secondToggleResponse?.active, false, 'toggling while active turns the picker off');
assert.equal(documentElement_isPicking(), false);

// Mutual exclusion: the Selection Panel already being open must refuse the picker, not fight it
// for the same clicks.
fakePanelOverlayPresent = true;
let blockedResponse;
contentListener({ kind: 'togglePanelPicker' }, {}, (response) => { blockedResponse = response; });
assert.equal(blockedResponse?.success, false, 'the picker refuses to start while the Selection Panel is open');
assert.equal(blockedResponse?.error, 'SelectionPanelActive');
assert.equal(documentElement_isPicking(), false);
fakePanelOverlayPresent = false;

// cancelPageWork (triggered here via pausing) must also stop an active picker -- pausing
// translation is itself a strong signal the user no longer wants to be mid-pick.
contentListener({ kind: 'togglePanelPicker' }, {}, () => {});
assert.equal(documentElement_isPicking(), true, 'picker active before pausing');
contentListener({ kind: 'setTranslationPaused', paused: true }, {}, () => {});
assert.equal(documentElement_isPicking(), false, 'pausing translation stops an active picker');
contentListener({ kind: 'setTranslationPaused', paused: false }, {}, () => {});

console.log('extension_content_contract=pass');
