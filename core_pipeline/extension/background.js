


const DEFAULT_LOCAL_PIPELINE_URL = 'http://127.0.0.1:8766/v1/translate-image';
const CACHE_VERSION = 'local-8-step-v12-manga-cleaner-device-overlay';
const MAX_CONCURRENT = 1;
const DEFAULT_CACHE_LIMIT = 12;
const MAX_CACHE_LIMIT = 40;

const outgoingRequests = new Map();
const activeControllers = new Map();
const requestQueue = [];
const translationCache = new Map();

let cacheLoaded = false;
let isPaused = false;

chrome.storage.local.get(['translationPaused']).then((result) => {
  isPaused = result.translationPaused === true;
}).catch(() => {});

function fastHash(str) {
  if (!str) return '';
  const len = str.length;
  let hash = 2166136261 >>> 0;
  const step = Math.max(1, Math.floor(len / 2000));
  for (let i = 0; i < len; i += step) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  hash ^= len;
  return Math.imul(hash, 16777619).toString(36);
}

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

async function getSettings() {
  const result = await chrome.storage.local.get([
    'localPipelineUrl',
    'localPipelineLanguage',
    'translationCachePages',
  ]);
  const cacheLimit = Number.parseInt(result.translationCachePages, 10);
  return {
    localPipelineUrl: String(result.localPipelineUrl || DEFAULT_LOCAL_PIPELINE_URL).trim() || DEFAULT_LOCAL_PIPELINE_URL,
    localPipelineLanguage: String(result.localPipelineLanguage || 'ja').trim() || 'ja',
    cacheLimit: Number.isFinite(cacheLimit)
      ? Math.max(0, Math.min(MAX_CACHE_LIMIT, cacheLimit))
      : DEFAULT_CACHE_LIMIT,
  };
}

function healthUrlForPipeline(pipelineUrl) {
  try {
    const url = new URL(pipelineUrl);
    if (url.pathname.endsWith('/translate-image') || url.pathname.endsWith('/translate-snapshot')) {
      url.pathname = url.pathname.replace(/\/v1\/translate-(image|snapshot)$/, '/v1/health');
    } else if (url.pathname.endsWith('/translate')) {
      url.pathname = '/v1/health';
    } else {
      url.pathname = '/v1/health';
    }
    url.search = '';
    url.hash = '';
    return url.toString();
  } catch {
    return 'http://127.0.0.1:8766/v1/health';
  }
}

async function checkPipelineHealth(settings, options = {}) {
  try {
    const response = await fetch(healthUrlForPipeline(settings.localPipelineUrl), {
      method: 'GET',
      cache: 'no-store',
    });
    if (!response.ok) throw new Error(`HEALTH_${response.status}`);
    return true;
  } catch {
    if (options.clearCacheOnFailure) await clearTranslationCache();
    return false;
  }
}

function responseErrorDetail(payload) {
  if (!payload || typeof payload !== 'object') return '';
  if (payload.error) return String(payload.error);
  if (payload.detail) {
    if (typeof payload.detail === 'string') return payload.detail;
    try {
      return JSON.stringify(payload.detail);
    } catch {
      return String(payload.detail);
    }
  }
  return '';
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function fetchImageAsDataUrl(imageUrl) {
  if (!imageUrl || !/^https?:|^file:|^data:/i.test(imageUrl)) {
    throw new Error('IMAGE_URL_UNSUPPORTED');
  }
  if (imageUrl.startsWith('data:')) return imageUrl;
  const response = await fetch(imageUrl, { credentials: 'omit', cache: 'force-cache' });
  if (!response.ok) throw new Error(`IMAGE_FETCH_${response.status}`);
  const blob = await response.blob();
  return blobToDataUrl(blob);
}

async function ensureCacheLoaded() {
  if (cacheLoaded) return;
  cacheLoaded = true;
  if (!chrome.storage.session?.get) return;
  try {
    const result = await chrome.storage.session.get(['translationCacheEntries']);
    const entries = result.translationCacheEntries || {};
    Object.keys(entries)
      .sort((a, b) => (entries[a].lastUsed || 0) - (entries[b].lastUsed || 0))
      .forEach((key) => {
        const entry = entries[key];
        if (entry?.result?.translatedImageDataUrl) translationCache.set(key, entry);
      });
  } catch {
    translationCache.clear();
  }
}

async function persistCache() {
  if (!chrome.storage.session?.set) return;
  const entries = {};
  for (const [key, entry] of translationCache.entries()) entries[key] = entry;
  try {
    await chrome.storage.session.set({ translationCacheEntries: entries });
  } catch {
    
  }
}

async function trimCache(limit) {
  const safeLimit = Math.max(0, Math.min(MAX_CACHE_LIMIT, Number(limit) || 0));
  if (safeLimit === 0) translationCache.clear();
  while (translationCache.size > safeLimit) {
    const oldestKey = translationCache.keys().next().value;
    if (!oldestKey) break;
    translationCache.delete(oldestKey);
  }
  await persistCache();
}

async function clearTranslationCache() {
  translationCache.clear();
  if (chrome.storage.session?.remove) {
    try {
      await chrome.storage.session.remove(['translationCacheEntries']);
    } catch {
      
    }
  }
}

function buildCacheId(message, settings) {
  const pageIdentity = message.pageCacheKey || message.pageUrl || 'no-page';
  const sourceIdentity = (
    message.cacheKey
    || message.originalImageUrl
    || message.imageUrl
    || message.base64Data
    || ''
  );
  return [
    CACHE_VERSION,
    settings.localPipelineUrl,
    settings.localPipelineLanguage,
    fastHash(String(pageIdentity)),
    fastHash(String(sourceIdentity)),
  ].join(':');
}

async function getCachedResult(cacheId, settings) {
  await ensureCacheLoaded();
  if (settings.cacheLimit <= 0 || !translationCache.has(cacheId)) return null;
  const entry = translationCache.get(cacheId);
  translationCache.delete(cacheId);
  entry.lastUsed = Date.now();
  translationCache.set(cacheId, entry);
  await persistCache();
  return entry.result;
}

async function putCachedResult(cacheId, result, settings) {
  await ensureCacheLoaded();
  if (settings.cacheLimit <= 0 || !result?.translatedImageDataUrl) return;
  translationCache.set(cacheId, {
    result,
    lastUsed: Date.now(),
  });
  await trimCache(settings.cacheLimit);
}

async function lookupCachedTranslation(message) {
  const settings = await getSettings();
  const cacheId = buildCacheId(message, settings);
  const cached = await getCachedResult(cacheId, settings);
  if (cached) {
    console.log(`[FMT] cache hit ${cacheId}`);
    return { ...cached, hit: true, fromCache: true, cacheId };
  }
  if (outgoingRequests.has(cacheId)) {
    console.log(`[FMT] cache miss; request still in-flight ${cacheId}`);
    return { hit: false, inFlight: true, cacheId };
  }
  return { hit: false, inFlight: false, cacheId };
}

async function callLocalPipeline(base64Data, width, height, settings, metadata = {}, signal = undefined) {
  const response = await fetch(settings.localPipelineUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify({
      imageData: base64Data,
      width: width || 0,
      height: height || 0,
      sourceLanguage: settings.localPipelineLanguage,
      targetLanguage: 'en',
      qualityProfile: 'strict',
      requestedOutput: 'translatedImageDataUrl',
      metadata,
    }),
  });

  if (!response.ok) {
    let detail = '';
    try {
      const payload = await response.json();
      detail = responseErrorDetail(payload);
    } catch {
      detail = await response.text().catch(() => '');
    }
    throw new Error(detail || `LOCAL_PIPELINE_${response.status}`);
  }

  const payload = await response.json();
  if (payload.translatedImageDataUrl || payload.imageDataUrl) {
    return {
      translatedImageDataUrl: payload.translatedImageDataUrl || payload.imageDataUrl,
      translations: payload.translations || [],
      pipelineReport: payload.report || null,
    };
  }
  if (Array.isArray(payload.translations)) {
    return { translations: payload.translations };
  }
  throw new Error('LOCAL_PIPELINE_EMPTY_RESPONSE');
}

async function processTranslation(message) {
  if (isPaused) return { error: 'TranslationPaused' };

  const width = message.width || 0;
  const height = message.height || 0;
  const settings = await getSettings();
  const cacheId = buildCacheId(message, settings);

  const cached = await getCachedResult(cacheId, settings);
  if (cached) {
    console.log(`[FMT] translateImage served from cache ${cacheId}`);
    return { ...cached, fromCache: true };
  }

  if (outgoingRequests.has(cacheId)) {
    console.log(`[FMT] joining in-flight translation ${cacheId}`);
    try {
      const result = await outgoingRequests.get(cacheId);
      return { ...result, fromInFlight: true };
    } catch (error) {
      return { error: error.message };
    }
  }
  if (outgoingRequests.size >= MAX_CONCURRENT) return { error: 'FullQueue' };

  const controller = new AbortController();
  activeControllers.set(cacheId, controller);

  const promise = (async () => {
    const startedAt = nowMs();
    try {
      if (isPaused) throw new Error('TranslationPaused');
      const base64Data = message.base64Data || await fetchImageAsDataUrl(message.imageUrl);
      if (isPaused) throw new Error('TranslationPaused');
      console.log(`[FMT] local pipeline start ${cacheId} ${width}x${height}`);
      const result = await callLocalPipeline(base64Data, width, height, settings, {
        source: message.imageUrl ? 'extension-image-url' : 'extension-canvas',
        pageUrl: message.pageUrl || '',
        pageCacheKey: message.pageCacheKey || '',
        cacheKey: message.cacheKey || '',
        originalImageUrl: message.originalImageUrl || '',
        cacheId,
      }, controller.signal);
      if (isPaused) throw new Error('TranslationPaused');
      await putCachedResult(cacheId, result, settings);
      console.log(`[FMT] local pipeline done ${cacheId} in ${Math.round(nowMs() - startedAt)}ms`);
      return result;
    } catch (error) {
      const messageText = error.name === 'AbortError' ? 'TranslationPaused' : error.message;
      if (messageText !== 'TranslationPaused') console.error('[FMT] Local pipeline error:', messageText);
      return { error: messageText };
    } finally {
      outgoingRequests.delete(cacheId);
      activeControllers.delete(cacheId);
      processQueue();
    }
  })();

  outgoingRequests.set(cacheId, promise);
  return promise;
}

function processQueue() {
  if (isPaused) return;
  while (requestQueue.length > 0 && outgoingRequests.size < MAX_CONCURRENT) {
    const { message, resolve } = requestQueue.shift();
    processTranslation(message).then(resolve);
  }
}

function queueTranslation(message) {
  return new Promise((resolve) => {
    if (isPaused) {
      resolve({ error: 'TranslationPaused' });
      return;
    }
    if (outgoingRequests.size < MAX_CONCURRENT) {
      processTranslation(message).then(resolve);
    } else {
      requestQueue.push({ message, resolve });
    }
  });
}

async function setTranslationPaused(paused) {
  isPaused = paused === true;
  await chrome.storage.local.set({ translationPaused: isPaused });

  if (isPaused) {
    const queued = requestQueue.splice(0);
    queued.forEach(({ resolve }) => resolve({ error: 'TranslationPaused' }));
    for (const controller of activeControllers.values()) controller.abort();
  } else {
    processQueue();
  }

  return {
    success: true,
    isPaused,
    cacheSize: translationCache.size,
    activeRequests: outgoingRequests.size,
    queueLength: requestQueue.length,
  };
}

async function ensureContentScript(tabId) {
  if (!tabId) throw new Error('TAB_ID_MISSING');
  try {
    await chrome.tabs.sendMessage(tabId, { kind: 'pingContentScript' });
    return { injected: false };
  } catch {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ['content.js'],
    });
    await chrome.tabs.sendMessage(tabId, { kind: 'pingContentScript' });
    return { injected: true };
  }
}

async function sendContentMessage(tabId, message) {
  await ensureContentScript(tabId);
  return chrome.tabs.sendMessage(tabId, message);
}

async function activatePageTranslation(tabId, options = {}) {
  const persistAuto = options.persistAuto === true;
  await setTranslationPaused(false);
  const storageUpdate = { translationPaused: false };
  if (persistAuto) storageUpdate.translationEnabled = true;
  await chrome.storage.local.set(storageUpdate);
  await ensureContentScript(tabId);
  await chrome.tabs.sendMessage(tabId, { kind: 'setTranslationPaused', paused: false });
  if (persistAuto) {
    await chrome.tabs.sendMessage(tabId, { kind: 'toggleTranslation', enabled: true });
  } else {
    await chrome.tabs.sendMessage(tabId, { kind: 'translatePageOnce' });
  }
  return { success: true, autoEnabled: persistAuto };
}

async function pausePageTranslation(tabId) {
  const state = await setTranslationPaused(true);
  if (tabId) {
    try {
      await sendContentMessage(tabId, { kind: 'setTranslationPaused', paused: true });
    } catch {
      
    }
  }
  return state;
}

async function cropVisibleTab(tabId, dimensions) {
  if (!dimensions) throw new Error('SNAPSHOT_DIMENSIONS_MISSING');
  const dataUrl = await new Promise((resolve) => {
    chrome.tabs.captureVisibleTab(null, { format: 'png' }, resolve);
  });
  if (!dataUrl) throw new Error('CAPTURE_FAILED');

  const zoomFactor = await chrome.tabs.getZoom(tabId).catch(() => 1);
  const devicePixelRatio = dimensions.devicePixelRatio || 1;
  const scale = zoomFactor * devicePixelRatio;
  const cropLeft = Math.max(0, Math.round((dimensions.left || 0) * scale));
  const cropTop = Math.max(0, Math.round((dimensions.top || 0) * scale));
  const cropWidth = Math.max(1, Math.round((dimensions.width || 0) * scale));
  const cropHeight = Math.max(1, Math.round((dimensions.height || 0) * scale));

  if (!Number.isFinite(cropWidth) || !Number.isFinite(cropHeight) || cropWidth < 1 || cropHeight < 1) {
    throw new Error('SNAPSHOT_DIMENSIONS_INVALID');
  }

  const screenshotBlob = await (await fetch(dataUrl)).blob();
  const croppedBitmap = await createImageBitmap(screenshotBlob, cropLeft, cropTop, cropWidth, cropHeight);
  const canvas = new OffscreenCanvas(cropWidth, cropHeight);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('SNAPSHOT_CANVAS_UNAVAILABLE');
  ctx.drawImage(croppedBitmap, 0, 0);
  croppedBitmap.close();
  const croppedBlob = await canvas.convertToBlob({ type: 'image/png' });
  return {
    dataUrl: await blobToDataUrl(croppedBlob),
    width: cropWidth,
    height: cropHeight,
    zoomFactor,
    devicePixelRatio,
  };
}

async function captureAndTranslate(tabId, dimensions) {
  if (isPaused) return { error: 'TranslationPaused' };
  try {
    const cropped = await cropVisibleTab(tabId, dimensions);
    const settings = await getSettings();
    const result = await callLocalPipeline(cropped.dataUrl, cropped.width, cropped.height, settings, {
      source: 'extension-selection-crop',
    });
    return {
      ...result,
      zoomFactor: cropped.zoomFactor,
      devicePixelRatio: cropped.devicePixelRatio,
      imageWidth: cropped.width,
      imageHeight: cropped.height,
    };
  } catch (error) {
    return { error: error.message };
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.kind === 'translateImage') {
    queueTranslation(message).then(sendResponse);
    return true;
  }
  if (message.kind === 'lookupCachedTranslation') {
    lookupCachedTranslation(message)
      .then(sendResponse)
      .catch((error) => sendResponse({ hit: false, inFlight: false, error: error.message }));
    return true;
  }
  if (message.kind === 'translateSnapshot') {
    captureAndTranslate(sender.tab?.id || message.tabId, message.dimensions).then(sendResponse);
    return true;
  }
  if (message.kind === 'getTranslationStats') {
    ensureCacheLoaded().then(async () => {
      const settings = await getSettings();
      sendResponse({
        cacheSize: translationCache.size,
        cacheLimit: settings.cacheLimit,
        activeRequests: outgoingRequests.size,
        queueLength: requestQueue.length,
        isPaused,
      });
    });
    return true;
  }
  if (message.kind === 'checkPipelineHealth') {
    getSettings().then(async (settings) => {
      const ok = await checkPipelineHealth(settings, { clearCacheOnFailure: true });
      sendResponse({ ok, cacheSize: translationCache.size });
    });
    return true;
  }
  if (message.kind === 'clearCache') {
    clearTranslationCache().then(() => sendResponse({ success: true, cacheSize: 0 }));
    return true;
  }
  if (message.kind === 'setCacheLimit') {
    getSettings()
      .then((settings) => trimCache(message.limit ?? settings.cacheLimit))
      .then(() => sendResponse({ success: true, cacheSize: translationCache.size }))
      .catch((error) => sendResponse({ success: false, error: error.message }));
    return true;
  }
  if (message.kind === 'setTranslationPaused') {
    setTranslationPaused(message.paused).then(sendResponse);
    return true;
  }
  if (message.kind === 'activatePageTranslation') {
    activatePageTranslation(message.tabId, { persistAuto: message.persistAuto === true })
      .then(sendResponse)
      .catch((error) => sendResponse({ success: false, error: error.message }));
    return true;
  }
  if (message.kind === 'pausePageTranslation') {
    pausePageTranslation(message.tabId)
      .then(sendResponse)
      .catch((error) => sendResponse({ success: false, error: error.message }));
    return true;
  }
  if (message.kind === 'sendContentCommand') {
    sendContentMessage(message.tabId, message.command)
      .then((response) => sendResponse({ success: true, response }))
      .catch((error) => sendResponse({ success: false, error: error.message }));
    return true;
  }
  if (message.kind === 'startTranslationPanel') {
    ensureContentScript(message.tabId)
      .then(() => chrome.scripting.executeScript({ target: { tabId: message.tabId }, files: ['translationPanel.js'] }))
      .then(() => sendResponse({ success: true }))
      .catch((error) => sendResponse({ success: false, error: error.message }));
    return true;
  }
  if (message.kind === 'testProviderKey') {
    sendResponse({ success: false, error: 'Cloud providers are disabled; use the local pipeline server.' });
    return true;
  }
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'translateMangaImage',
    title: 'Translate this manga panel locally',
    contexts: ['image'],
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'translateMangaImage' && tab?.id) {
    sendContentMessage(tab.id, { kind: 'translateSpecificImage', imageUrl: info.srcUrl });
  }
});

function updateIcon() {
  chrome.action.setIcon({
    path: {
      '16': 'icons/16x16.png',
      '48': 'icons/48x48.png',
      '128': 'icons/128x128.png',
    },
  });
}

chrome.storage.onChanged.addListener((changes) => {
  if (changes.localPipelineUrl) {
    clearTranslationCache();
    updateIcon();
  }
  if (changes.localPipelineLanguage) clearTranslationCache();
  if (changes.translationPaused) isPaused = changes.translationPaused.newValue === true;
});

updateIcon();
