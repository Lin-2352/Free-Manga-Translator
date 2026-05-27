


(function () {
  'use strict';

  if (window.__mangaTranslatorInjected) return;
  window.__mangaTranslatorInjected = true;

  
  const MIN_IMAGE_SIZE = 200;
  const MAX_DIMENSION = 1800;
  const TRANSLATED_ATTR = 'data-fmt-translated';
  const PROCESSING_ATTR = 'data-fmt-processing';
  const ORIGINAL_SRC_ATTR = 'data-fmt-original-src';
  const ORIGINAL_SRCSET_ATTR = 'data-fmt-original-srcset';
  const TRANSLATED_SRC_ATTR = 'data-fmt-translated-src';
  const CACHE_KEY_ATTR = 'data-fmt-cache-key';
  const MAX_RETRIES = 3;
  const BASE_RETRY_DELAY = 3000;

  
  let isEnabled = false;
  let isPaused = false;
  let fontFamily = 'CC Wild Words';
  let fontColor = '#000000';

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

  
  chrome.storage.local.get(['translationEnabled', 'translationPaused', 'mangaFontStyle', 'mangaFontColor'], (result) => {
    isEnabled = result.translationEnabled === true; 
    isPaused = result.translationPaused === true;
    fontFamily = result.mangaFontStyle || 'CC Wild Words';
    fontColor = result.mangaFontColor || '#000000';
    if (isEnabled && !isPaused) scheduleInitialScan();
    if (!isPaused) schedulePassiveCacheRestore();
  });

  chrome.storage.onChanged.addListener((changes) => {
    if (changes.translationEnabled) {
      isEnabled = changes.translationEnabled.newValue;
      if (isEnabled && !isPaused) scanForImages();
    }
    if (changes.translationPaused) {
      isPaused = changes.translationPaused.newValue === true;
      if (isPaused) {
        cancelPageWork();
      } else if (isEnabled) {
        scanForImages();
      } else {
        schedulePassiveCacheRestore();
      }
    }
    if (changes.mangaFontStyle) fontFamily = changes.mangaFontStyle.newValue;
    if (changes.mangaFontColor) fontColor = changes.mangaFontColor.newValue;
  });

  
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
    if (rect.width < 10 || rect.height < 10) return; 
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

  function positionSpinner(img, spinner) {
    if (!img.isConnected) {
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

  function cancelPageWork() {
    pendingSrcs.clear();
    retryCountMap.clear();
    document.querySelectorAll(`[${PROCESSING_ATTR}]`).forEach((img) => {
      img.removeAttribute(PROCESSING_ATTR);
      hideSpinner(img);
    });
    for (const img of Array.from(spinnerMap.keys())) hideSpinner(img);
  }

  
  function calculateResizedDimensions(width, height) {
    if (width <= MAX_DIMENSION && height <= MAX_DIMENSION) return { width, height };
    const ratio = Math.min(MAX_DIMENSION / width, MAX_DIMENSION / height);
    return { width: Math.round(width * ratio), height: Math.round(height * ratio) };
  }

  
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
    hideSpinner(img);
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
    const width = img.naturalWidth || img.width || 0;
    const height = img.naturalHeight || img.height || 0;
    return `${originalSrc || ''}|${width}x${height}`;
  }

  function imageDimensionsReady(img) {
    const width = img.naturalWidth || img.width || 0;
    const height = img.naturalHeight || img.height || 0;
    return width >= MIN_IMAGE_SIZE && height >= MIN_IMAGE_SIZE;
  }

  function rememberOriginalImage(img, originalSrc) {
    if (originalSrc && !img.getAttribute(ORIGINAL_SRC_ATTR)) {
      img.setAttribute(ORIGINAL_SRC_ATTR, originalSrc);
    }
    if (!img.getAttribute(ORIGINAL_SRCSET_ATTR) && img.srcset) {
      img.setAttribute(ORIGINAL_SRCSET_ATTR, img.srcset);
    }
  }

  function applyTranslatedImage(img, translatedImageDataUrl, originalSrc, cacheKey) {
    if (!translatedImageDataUrl) return;
    if (!isCurrentImageSource(img, originalSrc)) {
      console.log('[MangaTranslator] Skipped stale translation result for:', String(originalSrc).substring(0, 80));
      return;
    }
    rememberOriginalImage(img, originalSrc);
    img.setAttribute(TRANSLATED_ATTR, 'true');
    img.setAttribute(TRANSLATED_SRC_ATTR, translatedImageDataUrl);
    img.setAttribute(CACHE_KEY_ATTR, cacheKey || buildImageCacheKey(img, originalSrc));
    img.removeAttribute(PROCESSING_ATTR);
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

  
  function isStandaloneImagePage() {
    const ct = document.contentType || '';
    if (ct.startsWith('image/')) return true;
    if (document.body && document.body.children.length === 1 &&
        document.body.children[0].nodeName === 'IMG') return true;
    return false;
  }

  
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
        const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
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

      
      ctx.save();
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = 'rgba(255,255,255,1)';
      ctx.fillRect(minX, minY, boxW, boxH);

      
      ctx.beginPath();
      ctx.rect(minX, minY, boxW, boxH);
      ctx.clip();

      
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

  
  function shouldTranslate(img, options = {}) {
    const allowManual = options.force === true || options.restoreOnly === true;
    if (!allowManual && !isEnabled) return false;
    if (isPaused) return false;
    const originalSrc = getOriginalSrc(img);
    if (!originalSrc) return false;
    const cacheKey = buildImageCacheKey(img, originalSrc);
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

    if (!isStandaloneImagePage()) {
      const rect = img.getBoundingClientRect();
      if (rect.width < 100 || rect.height < 100) return false;
    }

    return true;
  }

  
  async function translateImage(img, options = {}) {
    if (!shouldTranslate(img, options)) return;

    const originalSrc = options.originalSrc || getOriginalSrc(img);
    if (!originalSrc) return;
    const cacheKey = options.cacheKey || buildImageCacheKey(img, originalSrc);

    if (!options.skipLookup) {
      const lookup = await lookupCachedTranslation(img, originalSrc, cacheKey);
      if (lookup.hit) return;
      if (lookup.inFlight) {
        await attachToInFlightTranslation(img, originalSrc, cacheKey);
        return;
      }
    }

    console.log('[MangaTranslator] Starting translation for:', originalSrc.substring(0, 80));
    rememberOriginalImage(img, originalSrc);
    pendingSrcs.add(cacheKey);
    img.setAttribute(PROCESSING_ATTR, 'true');
    showSpinner(img);

    try {
      let imageData;

      try {
        imageData = await getImageBase64(img);
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
          }
        } else {
          cleanupProcessing(img, cacheKey);
          return;
        }
      }

      console.log('[MangaTranslator] Sending to API:', imageData.width, 'x', imageData.height);
      const response = await chrome.runtime.sendMessage({
        kind: 'translateImage',
        base64Data: imageData.dataUrl || undefined,
        imageUrl: imageData.useBackgroundFetch ? originalSrc : undefined,
        originalImageUrl: originalSrc,
        cacheKey,
        pageUrl: window.location.href,
        pageCacheKey: getPageCacheKey(),
        width: imageData.width,
        height: imageData.height
      });

      if (response?.error) {
        console.warn('[MangaTranslator] API error:', response.error);
        if (response.error === 'TranslationPaused') {
          cleanupProcessing(img, cacheKey);
          return;
        }
        if (response.error === 'FullQueue' || response.error === 'RATE_LIMITED') {
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
            return; 
          } else {
            retryCountMap.delete(cacheKey);
          }
        }
        cleanupProcessing(img, cacheKey);
        return;
      }

      retryCountMap.delete(cacheKey);
      translatedSrcs.add(cacheKey);
      pendingSrcs.delete(cacheKey);
      hideSpinner(img);

      if (response?.translatedImageDataUrl) {
        console.log(response.fromCache ? '[MangaTranslator] Got cached translated image' : '[MangaTranslator] Got local pipeline image result');
        applyTranslatedImage(img, response.translatedImageDataUrl, originalSrc, cacheKey);
      } else if (response?.translations && response.translations.length > 0) {
        console.log('[MangaTranslator] Got', response.translations.length, 'translations');
        overlayTranslations(img, response.translations, imageData);
      } else {
        console.log('[MangaTranslator] No text found in image');
        img.setAttribute(TRANSLATED_ATTR, 'no-text');
        img.removeAttribute(PROCESSING_ATTR);
      }
    } catch (error) {
      console.error('[MangaTranslator] Error:', error);
      cleanupProcessing(img, cacheKey);
    }
  }

  async function attachToInFlightTranslation(img, originalSrc, cacheKey) {
    if (pendingSrcs.has(cacheKey)) return;
    rememberOriginalImage(img, originalSrc);
    pendingSrcs.add(cacheKey);
    img.setAttribute(PROCESSING_ATTR, 'true');
    showSpinner(img);
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
      if (response?.translatedImageDataUrl) {
        applyTranslatedImage(img, response.translatedImageDataUrl, originalSrc, cacheKey);
      } else if (response?.error && response.error !== 'TranslationPaused') {
        console.warn('[MangaTranslator] In-flight translation failed:', response.error);
      }
    } catch (error) {
      console.warn('[MangaTranslator] In-flight attach failed:', error?.message || error);
    } finally {
      pendingSrcs.delete(cacheKey);
      img.removeAttribute(PROCESSING_ATTR);
      hideSpinner(img);
    }
  }

  function cleanupProcessing(img, src) {
    img.removeAttribute(PROCESSING_ATTR);
    pendingSrcs.delete(src);
    hideSpinner(img);
  }

  
  async function processImage(img, options = {}) {
    const allowTranslate = options.force === true || isEnabled;
    const restoreOnly = options.restoreOnly === true || !allowTranslate;
    if (isPaused) return;
    const originalSrc = getOriginalSrc(img);
    if (!originalSrc) return;
    const cacheKey = buildImageCacheKey(img, originalSrc);
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

  
  function scanForImages(options = {}) {
    const allowScan = options.force === true || options.restoreOnly === true || isEnabled;
    if (!allowScan) return;
    if (isPaused) return;
    injectFonts();
    refreshProcessingSpinners();
    console.log(options.restoreOnly ? '[MangaTranslator] Restoring cached images...' : '[MangaTranslator] Scanning for images...');

    let count = 0;
    document.querySelectorAll('img').forEach((img) => {
      processImage(img, options);
      observeWithIntersection(img);
      count++;
    });

    document.querySelectorAll('picture img').forEach((img) => {
      processImage(img, options);
      observeWithIntersection(img);
    });

    document.querySelectorAll('img[data-src], img[data-lazy-src], img[data-original]').forEach((img) => {
      observeWithIntersection(img);
    });

    console.log('[MangaTranslator] Found', count, 'img elements');
  }

  
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

  
  const mutationObserver = new MutationObserver((mutations) => {
    if (isPaused) return;
    for (const mutation of mutations) {
      if (mutation.type === 'childList') {
        for (const node of mutation.addedNodes) {
          if (node.nodeName === 'IMG') {
            processImage(node, isEnabled ? {} : { restoreOnly: true });
            observeWithIntersection(node);
          } else if (node.querySelectorAll) {
            node.querySelectorAll('img').forEach((img) => {
              processImage(img, isEnabled ? {} : { restoreOnly: true });
              observeWithIntersection(img);
            });
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
        if (newSrc && !translatedSrcs.has(cacheKey) && !pendingSrcs.has(cacheKey)) {
          if (img.getAttribute(TRANSLATED_ATTR)) {
            img.removeAttribute(TRANSLATED_ATTR);
            img.removeAttribute(TRANSLATED_SRC_ATTR);
          }
          if (!img.getAttribute(PROCESSING_ATTR)) {
            processImage(img, isEnabled ? {} : { restoreOnly: true });
          }
        }
      }
    }
  });

  
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
          translateImage(img, { force: true, originalSrc, cacheKey });
          break;
        }
      }
    }

    if (message.kind === 'toggleTranslation') {
      isEnabled = message.enabled;
      console.log('[MangaTranslator] Translation', isEnabled ? 'enabled' : 'disabled');
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
        cancelPageWork();
      } else if (isEnabled) {
        scanForImages();
        handleStandaloneImage();
      } else {
        schedulePassiveCacheRestore();
      }
    }

    if (message.kind === 'retranslateAll') {
      translatedSrcs.clear();
      pendingSrcs.clear();
      cacheMissSrcs.clear();
      retryCountMap.clear();
      document.querySelectorAll(`[${TRANSLATED_ATTR}]`).forEach(img => {
        restoreOriginalImage(img);
      });
      if (!isPaused) setTimeout(() => scanForImages({ force: true }), 500);
    }

    if (message.kind === 'clearTranslations') {
      translatedSrcs.clear();
      pendingSrcs.clear();
      cacheMissSrcs.clear();
      retryCountMap.clear();
      document.querySelectorAll(`[${TRANSLATED_ATTR}]`).forEach(img => {
        restoreOriginalImage(img);
      });
    }
  });

  
  lastNavigationKey = getNavigationKey();
  patchHistoryNavigation();
  startAutoWatchdog();

  window.addEventListener('resize', updateSpinners, { passive: true });
  window.addEventListener('scroll', updateSpinners, { passive: true, capture: true });
  window.addEventListener('pageshow', () => scheduleNavigationScan('pageshow', { delay: 100 }), { passive: true });
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
