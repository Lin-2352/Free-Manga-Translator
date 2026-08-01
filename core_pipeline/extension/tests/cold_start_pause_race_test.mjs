// Proves background.js's cold-start pause race is fixed: on a fresh service-worker boot, the
// boot-time chrome.storage.local.get(['translationPaused']) read is async. If a REAL pause-state
// action (setTranslationPaused, stopTranslations, or a storage.onChanged echo of one from another
// context) lands and completes BEFORE that boot read resolves, the boot read finishing late with
// its now-stale value must NOT clobber the real, just-applied state -- previously it silently did,
// which looked like the extension re-pausing (or un-pausing) itself moments after the user acted,
// for no visible reason.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const backgroundPath = path.join(extensionRoot, 'background.js');
const source = fs.readFileSync(backgroundPath, 'utf8');

let messageListener;
let resolveBootStorageGet;
const localSetCalls = [];

const sandbox = {
  console,
  URL,
  setTimeout,
  clearTimeout,
  AbortController,
  chrome: {
    storage: {
      local: {
        get: (keys) => {
          // The FIRST call is the module's own boot-time read of ['translationPaused']. Hold it
          // open (return a promise this test controls) to simulate it resolving late, after real
          // pause-state activity has already happened this boot. Every LATER get() (used by
          // getSettings()) resolves immediately with normal settings so the rest of the extension
          // functions normally.
          if (Array.isArray(keys) && keys.length === 1 && keys[0] === 'translationPaused') {
            return new Promise((resolve) => { resolveBootStorageGet = resolve; });
          }
          return Promise.resolve({
            localPipelineUrl: 'http://127.0.0.1:8766/v1/translate-image',
            localPipelineLanguage: 'ja',
            translationCachePages: 12,
            translationQueuePages: 10,
            translationParallelPages: 2,
          });
        },
        set: async (values) => { localSetCalls.push(values); },
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
    if (String(url).includes('/v1/diagnostics/log')) return { ok: true, json: async () => ({ ok: true }) };
    return { ok: true, json: async () => ({ ok: true }) };
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: backgroundPath });
assert.equal(typeof messageListener, 'function', 'background message listener registered');
assert.equal(typeof resolveBootStorageGet, 'function', 'boot-time translationPaused read captured and held open');

function send(message) {
  return new Promise((resolve) => {
    messageListener(message, { tab: { id: 1 } }, resolve);
  });
}

// The user resumes translation (e.g. clicking Resume in the popup) while the boot-time storage
// read is still pending -- a real, current pause-state action landing before rehydration finishes.
const resumeResult = await send({ kind: 'setTranslationPaused', paused: false });
assert.equal(resumeResult.success, true, 'setTranslationPaused(false) completes normally');
assert.equal(resumeResult.isPaused, false, 'immediately after resuming, stats report unpaused');

// The boot-time read NOW resolves, carrying the STALE persisted value from before this boot (the
// user had it paused last session) -- this must not silently override the real, just-applied
// unpaused state.
resolveBootStorageGet({ translationPaused: true });
await new Promise((resolve) => setTimeout(resolve, 10));

const statsAfterStaleBootRead = await send({ kind: 'getTranslationStats' });
assert.equal(
  statsAfterStaleBootRead.isPaused,
  false,
  'a late-resolving boot-time storage read must not clobber a real pause-state action that already happened this boot',
);

// A real translateImage request must actually be allowed to dispatch (not silently treated as
// paused) after this.
const translateResult = await send({
  kind: 'translateImage',
  base64Data: 'data:image/png;base64,ZmFrZQ==',
  cacheKey: 'cold-start-page|100x100',
  originalImageUrl: 'https://example.test/cold-start-page.jpg',
  pageCacheKey: 'https://example.test/manga',
  pageUrl: 'https://example.test/manga',
  width: 100,
  height: 100,
});
assert.notEqual(
  translateResult.error,
  'TranslationPaused',
  'translation actually dispatches after the real resume action, undisturbed by the stale boot read',
);

console.log('extension_cold_start_pause_race=pass');
