// Free Manga Translator - Content Script
// Detects manga images on pages and overlays translations

(function () {
  'use strict';

  if (window.__mangaTranslatorInjected) return;
  window.__mangaTranslatorInjected = true;

  // ===== Constants =====
  const MIN_IMAGE_SIZE = 200;
  const MIN_PAGE_IMAGE_SIZE = 360;
  const MIN_PAGE_IMAGE_AREA = 220000;
  const MAX_DIMENSION = 4096;
  const CAPTURE_IMAGE_TYPE = 'image/jpeg';
  const CAPTURE_IMAGE_QUALITY = 0.92;
  const TRANSLATED_ATTR = 'data-fmt-translated';
  const MANUAL_CLEAR_ATTR = 'data-fmt-manually-cleared';
  const PROCESSING_ATTR = 'data-fmt-processing';
  const PROCESSING_SINCE_ATTR = 'data-fmt-processing-since';
  const STALE_PROCESSING_MS = 5 * 60 * 1000; // comfortably longer than any real pipeline run
  const ORIGINAL_SRC_ATTR = 'data-fmt-original-src';
  const ORIGINAL_SRCSET_ATTR = 'data-fmt-original-srcset';
  const ORIGINAL_WIDTH_ATTR = 'data-fmt-original-width';
  const ORIGINAL_HEIGHT_ATTR = 'data-fmt-original-height';
  const TRANSLATED_SRC_ATTR = 'data-fmt-translated-src';
  const CACHE_KEY_ATTR = 'data-fmt-cache-key';
  const MAX_RETRIES = 3;
  const BASE_RETRY_DELAY = 3000;
  const DEFAULT_AUTO_QUEUE_LIMIT = 20;
  const MAX_AUTO_QUEUE_LIMIT = 50;
  const EXTENSION_VERSION = '1.1.14';

  // ===== State =====
  let isEnabled = false;
  let isPaused = false;
  let fontFamily = 'CC Wild Words';
  let fontColor = '#000000';
  let autoQueueLimit = DEFAULT_AUTO_QUEUE_LIMIT;
  // Bumped by cancelPageWork(). A request captures the current generation
  // before awaiting its response; if the generation has moved on by the time
  // it resolves, the request was superseded (e.g. a bfcache-restore recovery
  // ran while it was in flight) and its result must be discarded rather than
  // applied -- a newer request may already own the same image's pending/
  // processing state, and isCurrentImageSource() alone can't detect this
  // (the image's src hasn't necessarily changed, just which request "owns" it).
  let pageWorkGeneration = 0;

  const translatedSrcs = new Set();
  const pendingSrcs = new Set();
  const cacheMissSrcs = new Set();
  const retryCountMap = new Map();
  const observedImages = new WeakSet();
  const spinnerMap = new Map();
  let spinnerFrame = null;
  let scheduledScanTimer = null;
  let lastNavigationKey = '';
  let autoWatchdogTimer = null;

  function makeTraceId(prefix = 'live') {
    return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  }

  function compactDiagnosticValue(value) {
    if (value === null || value === undefined) return value;
    if (typeof value === 'string') {
      if (value.startsWith('data:image/')) return `<data-url:${value.length}>`;
      return value.length > 220 ? `${value.slice(0, 220)}...` : value;
    }
    if (Array.isArray(value)) return value.slice(0, 12).map(compactDiagnosticValue);
    if (typeof value === 'object') {
      const result = {};
      for (const [key, item] of Object.entries(value)) {
        result[key] = /base64|imageData|translatedImageDataUrl|imageDataUrl/i.test(key)
          ? '<redacted>'
          : compactDiagnosticValue(item);
      }
      return result;
    }
    return value;
  }

  function emitDiagnostic(event, traceId, details = {}) {
    const payload = {
      kind: 'diagnosticLog',
      event,
      traceId,
      pageUrl: window.location.href,
      details: {
        version: EXTENSION_VERSION,
        ...compactDiagnosticValue(details),
      },
    };
    console.log(`[MangaTranslator][trace=${traceId}] ${event}`, payload.details);
    chrome.runtime.sendMessage(payload).catch(() => {});
  }

  // ===== Load Settings (auto-translate OFF by default) =====
  function normalizeAutoQueueLimit(value) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed)) return DEFAULT_AUTO_QUEUE_LIMIT;
    return Math.max(0, Math.min(MAX_AUTO_QUEUE_LIMIT, parsed));
  }

  chrome.storage.local.get(['translationEnabled', 'translationPaused', 'mangaFontStyle', 'mangaFontColor', 'translationQueuePages'], (result) => {
    isEnabled = result.translationEnabled === true; // OFF by default
    isPaused = result.translationPaused === true;
    fontFamily = result.mangaFontStyle || 'CC Wild Words';
    fontColor = result.mangaFontColor || '#000000';
    autoQueueLimit = normalizeAutoQueueLimit(result.translationQueuePages);
    if (isEnabled && !isPaused) scheduleInitialScan();
    if (!isPaused) schedulePassiveCacheRestore();
  });

  chrome.storage.onChanged.addListener((changes) => {
    if (changes.translationEnabled) {
      isEnabled = changes.translationEnabled.newValue;
      if (isEnabled) {
        clearManualClearMarkers();
        if (!isPaused) scanForImages();
      }
    }
    if (changes.translationPaused) {
      isPaused = changes.translationPaused.newValue === true;
      if (isPaused) {
        cancelPageWork({ pause: true });
      } else if (isEnabled) {
        scanForImages();
      } else {
        schedulePassiveCacheRestore();
      }
    }
    if (changes.mangaFontStyle) fontFamily = changes.mangaFontStyle.newValue;
    if (changes.mangaFontColor) fontColor = changes.mangaFontColor.newValue;
    if (changes.translationQueuePages) {
      autoQueueLimit = normalizeAutoQueueLimit(changes.translationQueuePages.newValue);
      if (isEnabled && !isPaused) scanForImages();
    }
  });

  // ===== Font Injection =====
  function injectFonts() {
    if (document.getElementById('fmt-fonts')) return;
    const style = document.createElement('style');
    style.id = 'fmt-fonts';
    style.textContent = `
      @font-face {
        font-family: 'CC Wild Words';
        src: url('${chrome.runtime.getURL('fonts/CCWildWords-Regular.otf')}') format('opentype');
        font-display: swap;
      }
      @font-face {
        font-family: 'Bangers';
        src: url('${chrome.runtime.getURL('fonts/Bangers-Regular.ttf')}') format('truetype');
        font-display: swap;
      }
      @font-face {
        font-family: 'Patrick Hand';
        src: url('${chrome.runtime.getURL('fonts/PatrickHand-Regular.ttf')}') format('truetype');
        font-display: swap;
      }
    `;
    (document.head || document.documentElement).appendChild(style);
  }

  // ===== Spinner Styles & Helpers (prominent dark badge with white spinner) =====
  function injectSpinnerStyles() {
    if (document.getElementById('fmt-spinner-styles')) return;
    const style = document.createElement('style');
    style.id = 'fmt-spinner-styles';
    style.textContent = `
      @keyframes fmt-img-spin {
        to { transform: rotate(360deg); }
      }
      .fmt-img-spinner {
        position: fixed;
        width: 40px;
        height: 40px;
        background: rgba(0, 0, 0, 0.7);
        border-radius: 50%;
        z-index: 2147483640;
        pointer-events: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.5);
      }
      .fmt-img-spinner::after {
        content: '';
        position: absolute;
        top: 8px;
        left: 8px;
        width: 24px;
        height: 24px;
        border: 3px solid rgba(255,255,255,0.3);
        border-top-color: #ffffff;
        border-radius: 50%;
        animation: fmt-img-spin 0.8s linear infinite;
        box-sizing: border-box;
      }
    `;
    (document.head || document.documentElement).appendChild(style);
  }

  function showSpinner(img) {
    if (spinnerMap.has(img)) {
      positionSpinner(img, spinnerMap.get(img));
      ensureSpinnerLoop();
      return;
    }
    injectSpinnerStyles();
    const rect = img.getBoundingClientRect();
    if (rect.width < 10 || rect.height < 10) return; // not visible
    const spinner = document.createElement('div');
    spinner.className = 'fmt-img-spinner';
    document.body.appendChild(spinner);
    spinnerMap.set(img, spinner);
    positionSpinner(img, spinner);
    ensureSpinnerLoop();
    console.log('[MangaTranslator] Spinner shown for image');
  }

  function hideSpinner(img) {
    const spinner = spinnerMap.get(img);
    if (spinner) {
      spinner.remove();
      spinnerMap.delete(img);
    }
    stopSpinnerLoopIfIdle();
  }

  function imageHasAttr(img, attr) {
    return typeof img.hasAttribute === 'function'
      ? img.hasAttribute(attr)
      : img.getAttribute(attr) !== null && img.getAttribute(attr) !== undefined;
  }

  function pruneStaleSpinners() {
    for (const img of Array.from(spinnerMap.keys())) {
      if (
        !img.isConnected
        || !imageHasAttr(img, PROCESSING_ATTR)
        || imageHasAttr(img, TRANSLATED_ATTR)
        || isPaused
      ) {
        hideSpinner(img);
      }
    }
  }

  function positionSpinner(img, spinner) {
    if (
      !img.isConnected
      || !imageHasAttr(img, PROCESSING_ATTR)
      || imageHasAttr(img, TRANSLATED_ATTR)
      || isPaused
    ) {
      hideSpinner(img);
      return;
    }
    const rect = img.getBoundingClientRect();
    if (rect.width < 10 || rect.height < 10 || rect.bottom < 0 || rect.right < 0 ||
        rect.top > window.innerHeight || rect.left > window.innerWidth) {
      spinner.style.display = 'none';
      return;
    }
    spinner.style.display = 'block';
    spinner.style.left = `${Math.round(rect.left + 8)}px`;
    spinner.style.top = `${Math.round(rect.top + 8)}px`;
  }

  function updateSpinners() {
    pruneStaleSpinners();
    for (const [img, spinner] of Array.from(spinnerMap.entries())) {
      positionSpinner(img, spinner);
    }
  }

  function ensureSpinnerLoop() {
    if (spinnerFrame !== null) return;
    const tick = () => {
      updateSpinners();
      spinnerFrame = spinnerMap.size > 0 ? requestAnimationFrame(tick) : null;
    };
    spinnerFrame = requestAnimationFrame(tick);
  }

  function stopSpinnerLoopIfIdle() {
    if (spinnerMap.size === 0 && spinnerFrame !== null) {
      cancelAnimationFrame(spinnerFrame);
      spinnerFrame = null;
    }
  }

  function refreshProcessingSpinners() {
    document.querySelectorAll(`img[${PROCESSING_ATTR}]`).forEach((img) => {
      if (!img.isConnected || img.getAttribute(TRANSLATED_ATTR)) return;
      showSpinner(img);
    });
  }

  // A PROCESSING marker's underlying request can die silently without ever
  // clearing it: the background message port breaks (MV3 service worker
  // restart) with no navigation event to react to, or bfcache freezes the
  // page mid-request and the promise never settles on restore. Either way the
  // image is stuck showing a spinner forever and never gets rescanned. Called
  // from the watchdog on every tick (regardless of tab visibility) so a
  // zombie marker is recovered even on a backgrounded tab the user isn't
  // currently viewing.
  function recoverStaleProcessingMarkers() {
    const now = Date.now();
    document.querySelectorAll(`img[${PROCESSING_ATTR}]`).forEach((img) => {
      const originalSrc = getOriginalSrc(img);
      const cacheKey = originalSrc ? buildImageCacheKey(img, originalSrc) : null;
      // No live pendingSrcs entry at all means nothing is actually tracking
      // this request anymore, regardless of age.
      const orphaned = !cacheKey || !pendingSrcs.has(cacheKey);
      const since = Number(img.getAttribute(PROCESSING_SINCE_ATTR)) || 0;
      const tooOld = since > 0 && (now - since) > STALE_PROCESSING_MS;
      if (!orphaned && !tooOld) return;
      if (cacheKey) pendingSrcs.delete(cacheKey);
      img.removeAttribute(PROCESSING_ATTR);
      img.removeAttribute(PROCESSING_SINCE_ATTR);
      hideSpinner(img);
    });
  }

  function cancelPageWork(options = {}) {
    if (options.pause === true) isPaused = true;
    pageWorkGeneration += 1;
    pendingSrcs.clear();
    retryCountMap.clear();
    document.querySelectorAll(`[${PROCESSING_ATTR}]`).forEach((img) => {
      img.removeAttribute(PROCESSING_ATTR);
      img.removeAttribute(PROCESSING_SINCE_ATTR);
      hideSpinner(img);
    });
    for (const img of Array.from(spinnerMap.keys())) hideSpinner(img);
  }

  // ===== Image Size Calculation =====
  function calculateResizedDimensions(width, height) {
    if (width <= MAX_DIMENSION && height <= MAX_DIMENSION) return { width, height };
    const ratio = Math.min(MAX_DIMENSION / width, MAX_DIMENSION / height);
    return { width: Math.round(width * ratio), height: Math.round(height * ratio) };
  }

  // ===== Get effective image src (handles lazy-load patterns) =====
  function getEffectiveSrc(img) {
    if (img.currentSrc && img.currentSrc !== '') return img.currentSrc;
    if (img.src && img.src !== '' && !img.src.endsWith('/')) return img.src;
    for (const attr of ['data-src', 'data-lazy-src', 'data-original', 'data-lazy', 'data-url']) {
      const val = img.getAttribute(attr);
      if (val && val.startsWith('http')) return val;
    }
    return img.src || '';
  }

  function isDataImage(src) {
    return typeof src === 'string' && src.startsWith('data:image/');
  }

  function isTranslatedReplacement(img, src) {
    const translatedSrc = img.getAttribute(TRANSLATED_SRC_ATTR);
    return !!translatedSrc && src === translatedSrc;
  }

  function clearImageRuntimeState(img, previousCacheKey) {
    if (previousCacheKey) {
      pendingSrcs.delete(previousCacheKey);
      retryCountMap.delete(previousCacheKey);
    }
    img.removeAttribute(TRANSLATED_ATTR);
    img.removeAttribute(TRANSLATED_SRC_ATTR);
    img.removeAttribute(CACHE_KEY_ATTR);
    img.removeAttribute(PROCESSING_ATTR);
    img.removeAttribute(ORIGINAL_WIDTH_ATTR);
    img.removeAttribute(ORIGINAL_HEIGHT_ATTR);
    hideSpinner(img);
  }

  // Turning auto-translate on is itself an explicit user request to translate the
  // page, so it lifts any earlier Clear Page suppression -- otherwise a page
  // cleared while auto-translate was off would stay stuck untranslated forever
  // once the user re-enables it.
  function clearManualClearMarkers() {
    document.querySelectorAll(`[${MANUAL_CLEAR_ATTR}]`).forEach((img) => {
      img.removeAttribute(MANUAL_CLEAR_ATTR);
    });
  }

  function getOriginalSrc(img) {
    const effectiveSrc = getEffectiveSrc(img);
    const stored = img.getAttribute(ORIGINAL_SRC_ATTR);
    if (stored) {
      if (effectiveSrc && effectiveSrc !== stored && !isTranslatedReplacement(img, effectiveSrc) && !isDataImage(effectiveSrc)) {
        const previousCacheKey = img.getAttribute(CACHE_KEY_ATTR) || buildImageCacheKey(img, stored);
        clearImageRuntimeState(img, previousCacheKey);
        img.setAttribute(ORIGINAL_SRC_ATTR, effectiveSrc);
        if (img.srcset) {
          img.setAttribute(ORIGINAL_SRCSET_ATTR, img.srcset);
        } else {
          img.removeAttribute(ORIGINAL_SRCSET_ATTR);
        }
        console.log('[MangaTranslator] Reused image node detected; source updated:', effectiveSrc.substring(0, 80));
        return effectiveSrc;
      }
      return stored;
    }
    for (const attr of ['data-src', 'data-lazy-src', 'data-original', 'data-lazy', 'data-url']) {
      const val = img.getAttribute(attr);
      if (val && (val.startsWith('http') || val.startsWith('data:image/') || val.startsWith('file:'))) return val;
    }
    return effectiveSrc;
  }

  function getPageCacheKey() {
    const loc = window.location || {};
    return `${loc.origin || ''}${loc.pathname || ''}${loc.search || ''}` || String(loc.href || '');
  }

  function getNavigationKey() {
    return `${getPageCacheKey()}#${(window.location && window.location.hash) || ''}`;
  }

  function buildImageCacheKey(img, originalSrc) {
    const storedWidth = Number.parseInt(img.getAttribute(ORIGINAL_WIDTH_ATTR) || '', 10);
    const storedHeight = Number.parseInt(img.getAttribute(ORIGINAL_HEIGHT_ATTR) || '', 10);
    const width = Number.isFinite(storedWidth) && storedWidth > 0 ? storedWidth : (img.naturalWidth || img.width || 0);
    const height = Number.isFinite(storedHeight) && storedHeight > 0 ? storedHeight : (img.naturalHeight || img.height || 0);
    return `${originalSrc || ''}|${width}x${height}`;
  }

  function imageDimensionsReady(img) {
    const width = img.naturalWidth || img.width || 0;
    const height = img.naturalHeight || img.height || 0;
    return width >= MIN_IMAGE_SIZE && height >= MIN_IMAGE_SIZE;
  }

  function isLikelyPageImage(img) {
    if (isStandaloneImagePage()) return true;
    const width = img.naturalWidth || img.width || 0;
    const height = img.naturalHeight || img.height || 0;
    if (width < MIN_PAGE_IMAGE_SIZE || height < MIN_PAGE_IMAGE_SIZE) return false;
    if (width * height < MIN_PAGE_IMAGE_AREA) return false;

    const src = getOriginalSrc(img) || getEffectiveSrc(img) || '';
    if (/[?&]type=p100\b/i.test(src) && (width < 720 || height < 720)) return false;

    const rect = img.getBoundingClientRect?.();
    if (rect && rect.width > 0 && rect.height > 0) {
      if (rect.width < 180 || rect.height < 180) return false;
      if (rect.width * rect.height < 80000) return false;
    }
    return true;
  }

  function rememberOriginalImage(img, originalSrc) {
    if (originalSrc && !img.getAttribute(ORIGINAL_SRC_ATTR)) {
      img.setAttribute(ORIGINAL_SRC_ATTR, originalSrc);
    }
    const width = img.naturalWidth || img.width || 0;
    const height = img.naturalHeight || img.height || 0;
    if (width > 0 && !img.getAttribute(ORIGINAL_WIDTH_ATTR)) {
      img.setAttribute(ORIGINAL_WIDTH_ATTR, String(width));
    }
    if (height > 0 && !img.getAttribute(ORIGINAL_HEIGHT_ATTR)) {
      img.setAttribute(ORIGINAL_HEIGHT_ATTR, String(height));
    }
    if (!img.getAttribute(ORIGINAL_SRCSET_ATTR) && img.srcset) {
      img.setAttribute(ORIGINAL_SRCSET_ATTR, img.srcset);
    }
  }

  function applyTranslatedImage(img, translatedImageDataUrl, originalSrc, cacheKey) {
    if (!translatedImageDataUrl) return;
    if (!isCurrentImageSource(img, originalSrc)) {
      console.log('[MangaTranslator] Skipped stale translation result for:', String(originalSrc).substring(0, 80));
      cleanupProcessing(img, cacheKey || buildImageCacheKey(img, originalSrc));
      return;
    }
    rememberOriginalImage(img, originalSrc);
    img.setAttribute(TRANSLATED_ATTR, 'true');
    img.setAttribute(TRANSLATED_SRC_ATTR, translatedImageDataUrl);
    img.setAttribute(CACHE_KEY_ATTR, cacheKey || buildImageCacheKey(img, originalSrc));
    img.removeAttribute(PROCESSING_ATTR);
    hideSpinner(img);
    if (img.srcset) img.removeAttribute('srcset');
    img.src = translatedImageDataUrl;
    translatedSrcs.add(cacheKey || buildImageCacheKey(img, originalSrc));
  }

  function isCurrentImageSource(img, expectedOriginalSrc) {
    if (!img.isConnected) return false;
    const effectiveSrc = getEffectiveSrc(img);
    const storedOriginal = img.getAttribute(ORIGINAL_SRC_ATTR);
    if (storedOriginal && storedOriginal === expectedOriginalSrc) {
      return isTranslatedReplacement(img, effectiveSrc) || effectiveSrc === expectedOriginalSrc || isDataImage(effectiveSrc);
    }
    if (!storedOriginal && effectiveSrc === expectedOriginalSrc) return true;
    return false;
  }

  function restoreOriginalImage(img) {
    const originalSrc = img.getAttribute(ORIGINAL_SRC_ATTR);
    if (originalSrc) {
      img.src = originalSrc;
      const originalSrcset = img.getAttribute(ORIGINAL_SRCSET_ATTR);
      if (originalSrcset) img.srcset = originalSrcset;
    }
    img.removeAttribute(TRANSLATED_ATTR);
    img.removeAttribute(TRANSLATED_SRC_ATTR);
    img.removeAttribute(CACHE_KEY_ATTR);
    img.removeAttribute(PROCESSING_ATTR);
    hideSpinner(img);
  }

  function resetTranslatedStateForForce(img, cacheKey) {
    if (!img.getAttribute(TRANSLATED_ATTR)) return;
    if (cacheKey) {
      translatedSrcs.delete(cacheKey);
      pendingSrcs.delete(cacheKey);
      cacheMissSrcs.delete(cacheKey);
      retryCountMap.delete(cacheKey);
    }
    restoreOriginalImage(img);
  }

  async function lookupCachedTranslation(img, originalSrc, cacheKey) {
    if (!originalSrc || !cacheKey) return { hit: false, inFlight: false };
    const needsTranslatedDataRestore = isDataImage(getEffectiveSrc(img)) && !!img.getAttribute(ORIGINAL_SRC_ATTR);
    if (cacheMissSrcs.has(cacheKey) && !needsTranslatedDataRestore) return { hit: false, inFlight: false };
    try {
      const response = await chrome.runtime.sendMessage({
        kind: 'lookupCachedTranslation',
        cacheKey,
        originalImageUrl: originalSrc,
        pageUrl: window.location.href,
        pageCacheKey: getPageCacheKey(),
        width: img.naturalWidth || img.width || 0,
        height: img.naturalHeight || img.height || 0
      });
      if (response?.hit && response.translatedImageDataUrl) {
        applyTranslatedImage(img, response.translatedImageDataUrl, originalSrc, cacheKey);
        return { hit: true, inFlight: false };
      }
      if (response?.inFlight) return { hit: false, inFlight: true };
      cacheMissSrcs.add(cacheKey);
      setTimeout(() => cacheMissSrcs.delete(cacheKey), 1500);
    } catch (error) {
      console.warn('[MangaTranslator] Cache lookup failed:', error?.message || error);
    }
    return { hit: false, inFlight: false };
  }

  // ===== Detect standalone image page =====
  function isStandaloneImagePage() {
    const ct = document.contentType || '';
    if (ct.startsWith('image/')) return true;
    if (document.body && document.body.children.length === 1 &&
        document.body.children[0].nodeName === 'IMG') return true;
    return false;
  }

  // ===== Get Image as Base64 =====
  function getImageBase64(img) {
    return new Promise((resolve, reject) => {
      try {
        let width = img.naturalWidth || img.width;
        let height = img.naturalHeight || img.height;
        if (!width || !height) { reject(new Error('No dimensions')); return; }

        const resized = calculateResizedDimensions(width, height);
        const canvas = document.createElement('canvas');
        canvas.width = resized.width;
        canvas.height = resized.height;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, resized.width, resized.height);
        ctx.drawImage(img, 0, 0, resized.width, resized.height);
        const dataUrl = canvas.toDataURL(CAPTURE_IMAGE_TYPE, CAPTURE_IMAGE_QUALITY);
        resolve({
          dataUrl,
          width: resized.width,
          height: resized.height,
          originalWidth: width,
          originalHeight: height
        });
      } catch (e) {
        reject(e);
      }
    });
  }

  // ===== Fetch Image Cross-Origin =====
  async function fetchImageCrossOrigin(url) {
    try {
      const response = await fetch(url, { mode: 'cors' });
      if (!response.ok) throw new Error('Fetch failed');
      const blob = await response.blob();
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    } catch (e) {
      throw new Error('CORS_BLOCKED');
    }
  }

  // ===== Word Wrap (handles long words via character breaking) =====
  function wrapText(ctx, text, maxWidth) {
    const content = String(text || '');
    if (maxWidth <= 0) return [content];
    const words = content.split(/\s+/).filter(Boolean);
    if (words.length === 0) return [''];
    const lines = [];
    let currentLine = '';

    for (const word of words) {
      const testLine = currentLine ? currentLine + ' ' + word : word;
      if (ctx.measureText(testLine).width <= maxWidth) {
        currentLine = testLine;
      } else if (currentLine) {
        lines.push(currentLine);
        if (ctx.measureText(word).width > maxWidth) {
          let partial = '';
          for (const ch of word) {
            if (ctx.measureText(partial + ch).width > maxWidth && partial) {
              lines.push(partial);
              partial = ch;
            } else {
              partial += ch;
            }
          }
          currentLine = partial;
        } else {
          currentLine = word;
        }
      } else {
        let partial = '';
        for (const ch of word) {
          if (ctx.measureText(partial + ch).width > maxWidth && partial) {
            lines.push(partial);
            partial = ch;
          } else {
            partial += ch;
          }
        }
        currentLine = partial;
      }
    }
    if (currentLine) lines.push(currentLine);
    return lines.length > 0 ? lines : [''];
  }

  // ===== Fit Text to Box (strict: guarantees text fits within box) =====
  function fitText(ctx, text, boxWidth, boxHeight, fontFam) {
    const MIN_FONT_SIZE = 7;
    const PADDING = 8;
    const STROKE_MARGIN = 4;
    const availW = Math.max(1, boxWidth - PADDING * 2 - STROKE_MARGIN * 2);
    const availH = Math.max(1, boxHeight - PADDING * 2);

    let fontSize = Math.floor(Math.min(availH * 0.48, availW * 0.5, 56));
    if (!Number.isFinite(fontSize) || fontSize < MIN_FONT_SIZE) {
      fontSize = MIN_FONT_SIZE;
    }

    let fit = null;
    while (fontSize >= MIN_FONT_SIZE) {
      ctx.font = `bold ${fontSize}px "${fontFam}", "Comic Sans MS", cursive`;
      const lineHeight = Math.ceil(fontSize * 1.2);
      const wrapped = wrapText(ctx, text, availW);
      const totalH = wrapped.length * lineHeight;

      let maxLineW = 0;
      for (const line of wrapped) {
        maxLineW = Math.max(maxLineW, ctx.measureText(line).width);
      }

      if (maxLineW <= availW && totalH <= availH) {
        fit = { fontSize, lines: wrapped, lineHeight, padding: PADDING };
        break;
      }

      fontSize -= 1;
    }

    if (fit) return fit;

    ctx.font = `bold ${MIN_FONT_SIZE}px "${fontFam}", "Comic Sans MS", cursive`;
    return {
      fontSize: MIN_FONT_SIZE,
      lines: wrapText(ctx, text, availW),
      lineHeight: Math.ceil(MIN_FONT_SIZE * 1.2),
      padding: PADDING,
    };
  }

  // ===== Overlay Translations onto Image (strict masking + clipped text) =====
  function overlayTranslations(img, translations, imageData) {
    if (!translations || translations.length === 0) return;

    const canvas = document.createElement('canvas');
    const origW = imageData.originalWidth || imageData.width;
    const origH = imageData.originalHeight || imageData.height;
    canvas.width = origW;
    canvas.height = origH;
    const ctx = canvas.getContext('2d');

    ctx.drawImage(img, 0, 0, origW, origH);

    const scaleX = origW / imageData.width;
    const scaleY = origH / imageData.height;

    for (const t of translations) {
      const MASK_PADDING = 2;
      const minX = Math.max(0, Math.floor(t.minX * scaleX) - MASK_PADDING);
      const minY = Math.max(0, Math.floor(t.minY * scaleY) - MASK_PADDING);
      const maxX = Math.min(origW, Math.ceil(t.maxX * scaleX) + MASK_PADDING);
      const maxY = Math.min(origH, Math.ceil(t.maxY * scaleY) + MASK_PADDING);
      const boxW = maxX - minX;
      const boxH = maxY - minY;

      if (boxW < 8 || boxH < 8) continue;

      // STEP 1: MASK - solid white rectangle
      ctx.save();
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = 'rgba(255,255,255,1)';
      ctx.fillRect(minX, minY, boxW, boxH);

      // STEP 2: CLIP
      ctx.beginPath();
      ctx.rect(minX, minY, boxW, boxH);
      ctx.clip();

      // STEP 3: TEXT
      const fit = fitText(ctx, t.translatedText, boxW, boxH, fontFamily);
      ctx.font = `bold ${fit.fontSize}px "${fontFamily}", "Comic Sans MS", cursive`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';

      const innerH = Math.max(1, boxH - fit.padding * 2);
      const totalTextH = fit.lines.length * fit.lineHeight;
      const textStartY = minY + fit.padding + Math.max(0, (innerH - totalTextH) / 2);
      const textCenterX = minX + boxW / 2;

      for (let i = 0; i < fit.lines.length; i++) {
        const ly = textStartY + i * fit.lineHeight;

        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2.5;
        ctx.lineJoin = 'round';
        ctx.miterLimit = 2;
        ctx.strokeText(fit.lines[i], textCenterX, ly);

        ctx.fillStyle = fontColor;
        ctx.fillText(fit.lines[i], textCenterX, ly);
      }

      ctx.restore();
    }

    try {
      const newDataUrl = canvas.toDataURL('image/png');
      const originalSrc = getOriginalSrc(img);
      applyTranslatedImage(img, newDataUrl, originalSrc, buildImageCacheKey(img, originalSrc));
    } catch (e) {
      console.error('[MangaTranslator] Overlay failed:', e);
    }
  }

  // ===== Check if Element Qualifies for Translation =====
  function shouldTranslate(img, options = {}) {
    const allowManual = options.force === true || options.manualSpecific === true || options.restoreOnly === true;
    if (!allowManual && !isEnabled) return false;
    if (isPaused) return false;
    const originalSrc = getOriginalSrc(img);
    if (!originalSrc) return false;
    const cacheKey = buildImageCacheKey(img, originalSrc);
    if (options.force === true) resetTranslatedStateForForce(img, cacheKey);
    if (img.getAttribute(TRANSLATED_ATTR)) return false;
    if (img.getAttribute(PROCESSING_ATTR)) {
      showSpinner(img);
      return false;
    }
    if (translatedSrcs.has(cacheKey) && !img.getAttribute(TRANSLATED_ATTR)) translatedSrcs.delete(cacheKey);
    if (translatedSrcs.has(cacheKey)) return false;
    if (pendingSrcs.has(cacheKey)) return false;
    if (isDataImage(getEffectiveSrc(img)) && img.getAttribute(ORIGINAL_SRC_ATTR)) return false;
    if (!imageDimensionsReady(img)) return false;
    if (!options.manualSpecific && !isLikelyPageImage(img)) return false;

    if (!isStandaloneImagePage()) {
      const rect = img.getBoundingClientRect();
      if (rect.width < 100 || rect.height < 100) return false;
    }

    return true;
  }

  // ===== Translate a Single Image =====
  async function translateImage(img, options = {}) {
    if (!shouldTranslate(img, options)) return;
    img.removeAttribute(MANUAL_CLEAR_ATTR);

    const originalSrc = options.originalSrc || getOriginalSrc(img);
    if (!originalSrc) return;
    const cacheKey = options.cacheKey || buildImageCacheKey(img, originalSrc);
    const traceId = options.traceId || makeTraceId('img');

    if (!options.skipLookup) {
      const lookup = await lookupCachedTranslation(img, originalSrc, cacheKey);
      if (lookup.hit) return;
      if (lookup.inFlight) {
        await attachToInFlightTranslation(img, originalSrc, cacheKey);
        return;
      }
    }

    emitDiagnostic('content.translate.start', traceId, {
      cacheKey,
      src: originalSrc,
      width: img.naturalWidth || img.width || 0,
      height: img.naturalHeight || img.height || 0,
      force: options.force === true,
      skipLookup: options.skipLookup === true,
    });
    rememberOriginalImage(img, originalSrc);
    pendingSrcs.add(cacheKey);
    img.setAttribute(PROCESSING_ATTR, 'true');
    img.setAttribute(PROCESSING_SINCE_ATTR, String(Date.now()));
    showSpinner(img);
    const requestGeneration = pageWorkGeneration;

    try {
      let imageData;
      const naturalWidth = img.naturalWidth || img.width || 0;
      const naturalHeight = img.naturalHeight || img.height || 0;
      const preferBackgroundFetch = /^https?:/i.test(originalSrc)
        && Math.max(naturalWidth, naturalHeight) > MAX_DIMENSION;

      try {
        if (preferBackgroundFetch) {
          imageData = {
            dataUrl: null,
            width: naturalWidth,
            height: naturalHeight,
            originalWidth: naturalWidth,
            originalHeight: naturalHeight,
            useBackgroundFetch: true
          };
          emitDiagnostic('content.capture.background_fetch_preferred', traceId, {
            cacheKey,
            src: originalSrc,
            width: naturalWidth,
            height: naturalHeight,
            reason: 'preserve-full-resolution',
          });
        } else {
          imageData = await getImageBase64(img);
        }
      } catch (e) {
        if (originalSrc.startsWith('http://') || originalSrc.startsWith('https://') || originalSrc.startsWith('data:image/') || originalSrc.startsWith('file:')) {
          try {
            const dataUrl = originalSrc.startsWith('data:image/') ? originalSrc : await fetchImageCrossOrigin(originalSrc);
            const tempImg = new Image();
            tempImg.crossOrigin = 'anonymous';
            await new Promise((resolve, reject) => {
              tempImg.onload = resolve;
              tempImg.onerror = reject;
              tempImg.src = dataUrl;
            });
            imageData = await getImageBase64(tempImg);
          } catch (corsError) {
            console.warn('[MangaTranslator] Canvas/CORS blocked; delegating fetch to background:', originalSrc.substring(0, 80));
            imageData = {
              dataUrl: null,
              width: img.naturalWidth || img.width || 0,
              height: img.naturalHeight || img.height || 0,
              originalWidth: img.naturalWidth || img.width || 0,
              originalHeight: img.naturalHeight || img.height || 0,
              useBackgroundFetch: true
            };
            emitDiagnostic('content.capture.background_fetch_required', traceId, {
              cacheKey,
              src: originalSrc,
              error: corsError?.message || String(corsError || ''),
            });
          }
        } else {
          emitDiagnostic('content.capture.unsupported_source', traceId, { cacheKey, src: originalSrc });
          cleanupProcessing(img, cacheKey);
          return;
        }
      }

      emitDiagnostic('content.translate.send', traceId, {
        cacheKey,
        width: imageData.width,
        height: imageData.height,
        source: imageData.useBackgroundFetch ? 'background-fetch' : 'canvas',
      });
      const response = await chrome.runtime.sendMessage({
        kind: 'translateImage',
        traceId,
        base64Data: imageData.dataUrl || undefined,
        imageUrl: imageData.useBackgroundFetch ? originalSrc : undefined,
        originalImageUrl: originalSrc,
        cacheKey,
        pageUrl: window.location.href,
        pageCacheKey: getPageCacheKey(),
        width: imageData.width,
        height: imageData.height
      });

      if (requestGeneration !== pageWorkGeneration) {
        // Superseded by cancelPageWork() (e.g. a bfcache-restore recovery)
        // while this request was in flight. A newer request may already own
        // this image's pendingSrcs/PROCESSING_ATTR state, so this stale
        // response must not touch them or apply itself over fresher work.
        console.log('[MangaTranslator] Discarding superseded translation response for:', String(originalSrc).substring(0, 80));
        return;
      }

      if (response?.error) {
        console.warn('[MangaTranslator] API error:', response.error);
        emitDiagnostic('content.translate.error_response', traceId, {
          cacheKey,
          error: response.error,
          queueLength: response.queueLength,
          queueLimit: response.queueLimit,
        });
        if (response.error === 'TranslationPaused') {
          cleanupProcessing(img, cacheKey);
          return;
        }
        if (response.error === 'FullQueue' || response.error === 'QueueFull' || response.error === 'RATE_LIMITED') {
          const retryCount = (retryCountMap.get(cacheKey) || 0) + 1;
          retryCountMap.set(cacheKey, retryCount);

          if (retryCount <= MAX_RETRIES) {
            const delay = BASE_RETRY_DELAY * Math.pow(2, retryCount - 1) + Math.random() * 1000;
            console.log(`[MangaTranslator] Retry ${retryCount}/${MAX_RETRIES} in ${Math.round(delay)}ms`);
            setTimeout(() => {
              img.removeAttribute(PROCESSING_ATTR);
              pendingSrcs.delete(cacheKey);
              translateImage(img, { force: options.force === true, originalSrc, cacheKey });
            }, delay);
            return; // spinner stays during retry
          } else {
            retryCountMap.delete(cacheKey);
          }
        }
        cleanupProcessing(img, cacheKey);
        return;
      }

      retryCountMap.delete(cacheKey);
      translatedSrcs.add(cacheKey);
      cleanupProcessing(img, cacheKey);

      if (response?.translatedImageDataUrl) {
        console.log(response.fromCache ? '[MangaTranslator] Got cached translated image' : '[MangaTranslator] Got local pipeline image result');
        emitDiagnostic('content.translate.image_result', traceId, {
          cacheKey,
          fromCache: response.fromCache === true,
          fromInFlight: response.fromInFlight === true,
        });
        applyTranslatedImage(img, response.translatedImageDataUrl, originalSrc, cacheKey);
      } else if (response?.translations && response.translations.length > 0) {
        console.log('[MangaTranslator] Got', response.translations.length, 'translations');
        emitDiagnostic('content.translate.overlay_result', traceId, {
          cacheKey,
          translationCount: response.translations.length,
        });
        overlayTranslations(img, response.translations, imageData);
      } else {
        console.log('[MangaTranslator] No text found in image');
        emitDiagnostic('content.translate.no_text_result', traceId, { cacheKey });
        img.setAttribute(TRANSLATED_ATTR, 'no-text');
        img.removeAttribute(PROCESSING_ATTR);
      }
    } catch (error) {
      console.error('[MangaTranslator] Error:', error);
      emitDiagnostic('content.translate.exception', traceId, {
        cacheKey,
        error: error?.message || String(error || ''),
      });
      // A newer request may already own this cacheKey's pending/processing
      // state if this one was superseded before it failed; only this
      // request's own state may be cleaned up.
      if (requestGeneration === pageWorkGeneration) cleanupProcessing(img, cacheKey);
    }
  }

  async function attachToInFlightTranslation(img, originalSrc, cacheKey) {
    if (pendingSrcs.has(cacheKey)) return;
    rememberOriginalImage(img, originalSrc);
    pendingSrcs.add(cacheKey);
    img.setAttribute(PROCESSING_ATTR, 'true');
    img.setAttribute(PROCESSING_SINCE_ATTR, String(Date.now()));
    showSpinner(img);
    const requestGeneration = pageWorkGeneration;
    try {
      const response = await chrome.runtime.sendMessage({
        kind: 'translateImage',
        imageUrl: originalSrc,
        originalImageUrl: originalSrc,
        cacheKey,
        pageUrl: window.location.href,
        pageCacheKey: getPageCacheKey(),
        width: img.naturalWidth || img.width || 0,
        height: img.naturalHeight || img.height || 0
      });
      // Superseded by cancelPageWork() while this request was in flight -- a
      // newer request may already own this image's state; see translateImage.
      if (requestGeneration !== pageWorkGeneration) return;
      if (response?.translatedImageDataUrl) {
        applyTranslatedImage(img, response.translatedImageDataUrl, originalSrc, cacheKey);
      } else if (response?.error && response.error !== 'TranslationPaused') {
        console.warn('[MangaTranslator] In-flight translation failed:', response.error);
      }
    } catch (error) {
      console.warn('[MangaTranslator] In-flight attach failed:', error?.message || error);
    } finally {
      if (requestGeneration === pageWorkGeneration) {
        pendingSrcs.delete(cacheKey);
        img.removeAttribute(PROCESSING_ATTR);
        hideSpinner(img);
      }
    }
  }

  function cleanupProcessing(img, src) {
    img.removeAttribute(PROCESSING_ATTR);
    pendingSrcs.delete(src);
    hideSpinner(img);
  }

  // ===== Process an Image Element =====
  async function processImage(img, options = {}) {
    const allowTranslate = options.force === true || isEnabled;
    const restoreOnly = options.restoreOnly === true || !allowTranslate;
    if (isPaused) return;
    // A manual Clear Page must stick until the user explicitly asks to translate
    // again (Translate Page/Re-translate/right-click, or re-enabling auto-translate
    // -- all of which pass force:true or clear the marker themselves below). Every
    // OTHER path that reaches processImage (auto-translate scans, the intersection
    // observer, the watchdog) must not silently resurrect a page the user just
    // cleared, whether or not it happens to be in restoreOnly mode.
    if (img.getAttribute(MANUAL_CLEAR_ATTR) && options.force !== true) return;
    const originalSrc = getOriginalSrc(img);
    if (!originalSrc) return;
    const cacheKey = buildImageCacheKey(img, originalSrc);
    if (options.force === true) resetTranslatedStateForForce(img, cacheKey);
    if (img.getAttribute(TRANSLATED_ATTR)) return;
    if (img.getAttribute(PROCESSING_ATTR)) {
      showSpinner(img);
      return;
    }
    if (translatedSrcs.has(cacheKey) && !img.getAttribute(TRANSLATED_ATTR)) translatedSrcs.delete(cacheKey);
    if (translatedSrcs.has(cacheKey)) return;
    if (pendingSrcs.has(cacheKey)) {
      img.setAttribute(PROCESSING_ATTR, 'true');
      showSpinner(img);
      return;
    }

    if (img.complete && img.naturalWidth > 0) {
      const lookup = await lookupCachedTranslation(img, originalSrc, cacheKey);
      if (lookup.hit) return;
      if (lookup.inFlight) {
        await attachToInFlightTranslation(img, originalSrc, cacheKey);
        return;
      }
      if (!restoreOnly && allowTranslate) {
        translateImage(img, { force: options.force === true, skipLookup: true, originalSrc, cacheKey });
      }
    } else {
      img.addEventListener('load', () => processImage(img, options), { once: true });
    }
  }

  function collectCandidateImages() {
    const seen = new WeakSet();
    const images = [];
    const addImage = (img) => {
      if (!img || img.nodeName !== 'IMG' || seen.has(img)) return;
      seen.add(img);
      images.push(img);
    };
    document.querySelectorAll('img').forEach(addImage);
    document.querySelectorAll('picture img').forEach(addImage);
    document.querySelectorAll('img[data-src], img[data-lazy-src], img[data-original]').forEach(addImage);
    return images;
  }

  function imageViewportDistance(img) {
    try {
      const rect = img.getBoundingClientRect();
      if (!rect || rect.width <= 0 || rect.height <= 0) return Number.POSITIVE_INFINITY;
      const viewportCenterY = (window.innerHeight || 0) / 2;
      const viewportCenterX = (window.innerWidth || 0) / 2;
      const imageCenterY = rect.top + rect.height / 2;
      const imageCenterX = rect.left + rect.width / 2;
      const verticalGap = rect.bottom < 0
        ? Math.abs(rect.bottom)
        : (rect.top > (window.innerHeight || 0) ? rect.top - (window.innerHeight || 0) : 0);
      return verticalGap * 4 + Math.abs(imageCenterY - viewportCenterY) + Math.abs(imageCenterX - viewportCenterX) * 0.1;
    } catch {
      return Number.POSITIVE_INFINITY;
    }
  }

  function selectImagesForScan(images, options = {}) {
    const candidates = options.restoreOnly === true
      ? images
      : images.filter((img) => isLikelyPageImage(img));
    if (options.force === true || options.restoreOnly === true || !isEnabled) return candidates;
    const limit = autoQueueLimit <= 0 ? 1 : autoQueueLimit;
    return [...candidates]
      .sort((a, b) => imageViewportDistance(a) - imageViewportDistance(b))
      .slice(0, limit);
  }

  function scheduleAutoScanForNewImages(reason) {
    if (isEnabled) {
      scheduleNavigationScan(reason, { delay: 350 });
    } else {
      scheduleNavigationScan(reason, { delay: 450 });
    }
  }

  // ===== Scan Page for Images =====
  function scanForImages(options = {}) {
    const allowScan = options.force === true || options.restoreOnly === true || isEnabled;
    if (!allowScan) return;
    if (isPaused) return;
    injectFonts();
    refreshProcessingSpinners();
    console.log(options.restoreOnly ? '[MangaTranslator] Restoring cached images...' : '[MangaTranslator] Scanning for images...');

    const images = collectCandidateImages();
    images.forEach(observeWithIntersection);
    const selectedImages = selectImagesForScan(images, options);
    const selectedSet = new WeakSet(selectedImages);

    selectedImages.forEach((img) => {
      processImage(img, options);
    });

    if (isEnabled && options.force !== true && options.restoreOnly !== true) {
      images.forEach((img) => {
        if (!selectedSet.has(img)) processImage(img, { restoreOnly: true });
      });
    }

    console.log('[MangaTranslator] Found', images.length, 'img elements; scheduled', selectedImages.length);
  }

  // ===== Standalone Image Handler =====
  function handleStandaloneImage(options = {}) {
    if (!isStandaloneImagePage()) return;
    if (!options.force && !options.restoreOnly && !isEnabled) return;
    if (isPaused) return;
    injectFonts();

    const img = document.querySelector('img');
    if (!img) return;

    console.log('[MangaTranslator] Standalone image detected');
    if (img.complete && img.naturalWidth > 0) {
      processImage(img, options);
    } else {
      img.addEventListener('load', () => processImage(img, options), { once: true });
    }
  }

  // ===== Schedule Initial Scan =====
  function scheduleInitialScan() {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        setTimeout(scanForImages, 500);
        setTimeout(handleStandaloneImage, 600);
      });
    } else {
      setTimeout(scanForImages, 500);
      setTimeout(handleStandaloneImage, 600);
    }
    setTimeout(scanForImages, 4000);
  }

  function schedulePassiveCacheRestore() {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        setTimeout(() => scanForImages({ restoreOnly: true }), 300);
        setTimeout(() => handleStandaloneImage({ restoreOnly: true }), 400);
      });
    } else {
      setTimeout(() => scanForImages({ restoreOnly: true }), 300);
      setTimeout(() => handleStandaloneImage({ restoreOnly: true }), 400);
    }
    setTimeout(() => scanForImages({ restoreOnly: true }), 2500);
  }

  function scheduleNavigationScan(reason, options = {}) {
    if (scheduledScanTimer !== null) clearTimeout(scheduledScanTimer);
    scheduledScanTimer = setTimeout(() => {
      scheduledScanTimer = null;
      if (isPaused) return;
      const navigationKey = getNavigationKey();
      if (navigationKey !== lastNavigationKey) {
        lastNavigationKey = navigationKey;
        cacheMissSrcs.clear();
        console.log('[MangaTranslator] Navigation scan:', reason, navigationKey);
      }
      if (isEnabled || options.force) {
        scanForImages(options.force ? { force: true } : {});
        handleStandaloneImage(options.force ? { force: true } : {});
      } else {
        scanForImages({ restoreOnly: true });
        handleStandaloneImage({ restoreOnly: true });
      }
    }, options.delay ?? 250);
  }

  function startAutoWatchdog() {
    if (autoWatchdogTimer !== null) return;
    if (typeof setInterval !== 'function') return;
    autoWatchdogTimer = setInterval(() => {
      // Runs regardless of tab visibility/pause state -- a zombie marker on a
      // backgrounded or paused tab still needs to clear so the image isn't
      // stuck spinning forever whenever the user does return to it.
      recoverStaleProcessingMarkers();
      if (document.hidden || isPaused) return;
      if (isEnabled) {
        refreshProcessingSpinners();
        scheduleNavigationScan('auto-watchdog');
      } else {
        refreshProcessingSpinners();
        scheduleNavigationScan('cache-restore-watchdog', { delay: 400 });
      }
    }, 3000);
  }

  function patchHistoryNavigation() {
    if (window.__mangaTranslatorHistoryPatched || typeof history === 'undefined') return;
    window.__mangaTranslatorHistoryPatched = true;
    for (const method of ['pushState', 'replaceState']) {
      const original = history[method];
      if (typeof original !== 'function') continue;
      history[method] = function patchedHistoryMethod(...args) {
        const result = original.apply(this, args);
        scheduleNavigationScan(method, { delay: 350 });
        return result;
      };
    }
  }

  // ===== IntersectionObserver =====
  const intersectionObserver = new IntersectionObserver((entries) => {
    if (isPaused) return;
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const img = entry.target;
      if (img.nodeName !== 'IMG') continue;
      if (img.getAttribute(TRANSLATED_ATTR) || img.getAttribute(PROCESSING_ATTR)) continue;
      if (imageDimensionsReady(img)) {
        processImage(img, isEnabled ? {} : { restoreOnly: true });
      }
    }
  }, { rootMargin: '200px' });

  function observeWithIntersection(img) {
    if (observedImages.has(img)) return;
    observedImages.add(img);
    intersectionObserver.observe(img);
  }

  // ===== MutationObserver =====
  const mutationObserver = new MutationObserver((mutations) => {
    if (isPaused) return;
    for (const mutation of mutations) {
      if (mutation.type === 'childList') {
        for (const node of mutation.addedNodes) {
          if (node.nodeName === 'IMG') {
            observeWithIntersection(node);
            if (isEnabled) scheduleAutoScanForNewImages('mutation-image');
            else processImage(node, { restoreOnly: true });
          } else if (node.querySelectorAll) {
            node.querySelectorAll('img').forEach((img) => {
              observeWithIntersection(img);
              if (!isEnabled) processImage(img, { restoreOnly: true });
            });
            if (isEnabled) scheduleAutoScanForNewImages('mutation-images');
          }
        }
      }
      if (mutation.type === 'attributes' && mutation.target.nodeName === 'IMG') {
        const img = mutation.target;
        const newSrc = getEffectiveSrc(img);
        const translatedSrc = img.getAttribute(TRANSLATED_SRC_ATTR);
        if (translatedSrc && newSrc === translatedSrc) continue;
        const originalSrc = getOriginalSrc(img);
        const cacheKey = buildImageCacheKey(img, originalSrc);
        if (translatedSrcs.has(cacheKey) && !img.getAttribute(TRANSLATED_ATTR)) translatedSrcs.delete(cacheKey);
        if (newSrc && !translatedSrcs.has(cacheKey) && !pendingSrcs.has(cacheKey)) {
          if (img.getAttribute(TRANSLATED_ATTR)) {
            img.removeAttribute(TRANSLATED_ATTR);
            img.removeAttribute(TRANSLATED_SRC_ATTR);
          }
          if (!img.getAttribute(PROCESSING_ATTR)) {
            if (isEnabled) scheduleAutoScanForNewImages('image-source-change');
            else processImage(img, { restoreOnly: true });
          }
        }
      }
    }
  });

  // ===== Start Observers =====
  function startObservers() {
    const target = document.body || document.documentElement;
    if (!target) return;
    mutationObserver.observe(target, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['src', 'srcset', 'data-src', 'data-lazy-src', 'data-original']
    });
  }

  if (document.body) {
    startObservers();
  } else {
    document.addEventListener('DOMContentLoaded', startObservers);
  }

  // ===== Message Handler =====
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.kind === 'pingContentScript') {
      sendResponse({ ok: true });
      return true;
    }

    if (message.kind === 'translateSpecificImage') {
      const images = document.querySelectorAll('img');
      for (const img of images) {
        const originalSrc = getOriginalSrc(img);
        if (img.src === message.imageUrl || img.currentSrc === message.imageUrl || originalSrc === message.imageUrl) {
          const cacheKey = buildImageCacheKey(img, originalSrc);
          translatedSrcs.delete(cacheKey);
          pendingSrcs.delete(cacheKey);
          retryCountMap.delete(cacheKey);
          img.removeAttribute(TRANSLATED_ATTR);
          img.removeAttribute(TRANSLATED_SRC_ATTR);
          img.removeAttribute(PROCESSING_ATTR);
          translateImage(img, { force: true, manualSpecific: true, originalSrc, cacheKey });
          break;
        }
      }
    }

    if (message.kind === 'toggleTranslation') {
      isEnabled = message.enabled;
      console.log('[MangaTranslator] Translation', isEnabled ? 'enabled' : 'disabled');
      if (isEnabled) clearManualClearMarkers();
      if (isEnabled && !isPaused) {
        scanForImages();
        handleStandaloneImage();
      } else if (!isPaused) {
        schedulePassiveCacheRestore();
      }
    }

    if (message.kind === 'translatePageOnce') {
      if (!isPaused) {
        scanForImages({ force: true });
        handleStandaloneImage({ force: true });
      }
      sendResponse({ success: !isPaused });
      return true;
    }

    if (message.kind === 'setTranslationPaused') {
      isPaused = message.paused === true;
      if (isPaused) {
        cancelPageWork({ pause: true });
      } else if (isEnabled) {
        scanForImages();
        handleStandaloneImage();
      } else {
        schedulePassiveCacheRestore();
      }
      sendResponse({ success: true, paused: isPaused });
      return true;
    }

    if (message.kind === 'retranslateAll') {
      cancelPageWork();
      translatedSrcs.clear();
      pendingSrcs.clear();
      cacheMissSrcs.clear();
      retryCountMap.clear();
      document.querySelectorAll(`[${TRANSLATED_ATTR}]`).forEach(img => {
        restoreOriginalImage(img);
      });
      if (!isPaused) setTimeout(() => scanForImages({ force: true }), 500);
      sendResponse({ success: true });
      return true;
    }

    if (message.kind === 'clearTranslations') {
      cancelPageWork();
      translatedSrcs.clear();
      pendingSrcs.clear();
      cacheMissSrcs.clear();
      retryCountMap.clear();
      document.querySelectorAll(`[${TRANSLATED_ATTR}]`).forEach(img => {
        restoreOriginalImage(img);
        img.setAttribute(MANUAL_CLEAR_ATTR, '1');
      });
      sendResponse({ success: true });
      return true;
    }
  });

  // Initial scan
  lastNavigationKey = getNavigationKey();
  patchHistoryNavigation();
  startAutoWatchdog();

  window.addEventListener('resize', updateSpinners, { passive: true });
  window.addEventListener('scroll', updateSpinners, { passive: true, capture: true });
  window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
      // A bfcache restore resumes this exact JS heap, including any
      // PROCESSING markers/pendingSrcs entries from before navigation -- so
      // recoverStaleProcessingMarkers()'s "orphaned" check would NOT catch
      // them (pendingSrcs still has the entry; it was frozen, not cleared).
      // But the message port those in-flight requests were awaiting died
      // with the old page instance, so their promises are certain to never
      // settle. cancelPageWork() unconditionally clears every marker before
      // the scan below runs, so the cache lookup that scan triggers (not a
      // re-queue) is what applies the now-completed, background-cached
      // result instead of the image getting stuck on its old, dead marker.
      cancelPageWork();
    }
    scheduleNavigationScan('pageshow', { delay: 100 });
  }, { passive: true });
  window.addEventListener('popstate', () => scheduleNavigationScan('popstate', { delay: 150 }), { passive: true });
  window.addEventListener('hashchange', () => scheduleNavigationScan('hashchange', { delay: 150 }), { passive: true });
  window.addEventListener('focus', () => scheduleNavigationScan('focus', { delay: 250 }), { passive: true });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) scheduleNavigationScan('visibilitychange', { delay: 150 });
  }, { passive: true });
  document.addEventListener('keyup', (event) => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight' || event.key === 'PageUp' || event.key === 'PageDown') {
      scheduleNavigationScan(`key:${event.key}`, { delay: 500 });
    }
  }, { passive: true });
  document.addEventListener('click', () => scheduleNavigationScan('click', { delay: 650 }), { passive: true, capture: true });

  if (isEnabled && !isPaused) scheduleInitialScan();
  if (!isPaused) schedulePassiveCacheRestore();
})();

