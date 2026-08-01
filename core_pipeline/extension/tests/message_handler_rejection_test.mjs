// Regression test: three message handlers (getTranslationStats, getRecentTranslations,
// checkPipelineHealth) called sendResponse only inside a .then() with no .catch(). If the
// awaited chrome.storage/getSettings call rejected, sendResponse was never called while
// `return true` kept the message port open -- the caller's promise (and whatever UI element
// was waiting on it) hung forever, with Chrome only logging an "Unchecked runtime.lastError"
// nobody sees. A fourth case, processQueue's dispatchTranslation(message).then(resolve) with
// no rejection handler, could orphan a queued translation's own response promise the same way.
//
// Drives the real message-handling code path (same technique as cache_gallery_limit_test.mjs),
// injecting a genuine storage rejection rather than asserting on source text.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const backgroundPath = path.join(extensionRoot, 'background.js');
const source = fs.readFileSync(backgroundPath, 'utf8');

let messageListener;
let storageShouldFail = false;

async function mockFetch(url) {
  const urlStr = String(url);
  if (urlStr.includes('/v1/health')) return { ok: true, json: async () => ({ ok: true }) };
  return { ok: true, json: async () => ({ ok: true }) };
}

const sandbox = {
  console,
  URL,
  setTimeout,
  clearTimeout,
  AbortController,
  chrome: {
    storage: {
      local: {
        get: async () => {
          if (storageShouldFail) throw new Error('simulated chrome.storage.local failure');
          return {
            localPipelineUrl: 'http://127.0.0.1:8766/v1/translate-image',
            localPipelineLanguage: 'ja',
            translationCachePages: 24,
            translationQueuePages: 20,
            translationParallelPages: 2,
          };
        },
        set: async () => {},
      },
      session: {
        get: async () => {
          if (storageShouldFail) throw new Error('simulated chrome.storage.session failure');
          return {};
        },
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
    tabs: { sendMessage: async () => ({ ok: true }), getZoom: async () => 1 },
    scripting: { executeScript: async () => [] },
    action: { setIcon: () => {} },
  },
  fetch: mockFetch,
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: backgroundPath });
assert.equal(typeof messageListener, 'function', 'background message listener registered');

function send(message, timeoutMs = 2000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`sendResponse was never called for ${message.kind} within ${timeoutMs}ms -- the message port hung`)),
      timeoutMs,
    );
    messageListener(message, { tab: { id: 1 } }, (response) => {
      clearTimeout(timer);
      resolve(response);
    });
  });
}

// ---- Baseline: all three handlers respond normally when storage succeeds ----
storageShouldFail = false;
for (const kind of ['getTranslationStats', 'getRecentTranslations', 'checkPipelineHealth']) {
  const response = await send({ kind });
  assert.ok(response, `${kind} should respond normally when storage succeeds`);
}

// ---- The actual regression: sendResponse must still fire when getSettings() rejects ----
// getTranslationStats and checkPipelineHealth both call getSettings(), which reads
// chrome.storage.local directly with no internal try/catch of its own -- a rejection there
// used to propagate straight past their missing .catch(), leaving sendResponse never called.
// (getRecentTranslations goes through ensureCacheLoaded() instead, which already has its own
// catch-all around chrome.storage.session -- its .catch() here is defensive-in-depth for any
// other failure in that handler, not reachable via a storage rejection specifically, so it is
// not re-tested via this injection point.)
storageShouldFail = true;
for (const kind of ['getTranslationStats', 'checkPipelineHealth']) {
  const response = await send({ kind });
  assert.ok(
    response && typeof response === 'object',
    `${kind} must call sendResponse even when getSettings() rejects, not leave the message port hanging`,
  );
  assert.ok(
    'error' in response,
    `${kind}'s response should surface the failure (an "error" field), got: ${JSON.stringify(response)}`,
  );
}
storageShouldFail = false;

// getRecentTranslations must still respond normally (ensureCacheLoaded's own catch-all
// degrades to an empty cache rather than rejecting) -- confirms the added .catch() here
// doesn't change normal-path behavior.
{
  const response = await send({ kind: 'getRecentTranslations' });
  assert.ok(Array.isArray(response?.entries), 'getRecentTranslations must still respond normally');
}

console.log('message_handler_rejection=pass');
