


(function () {
  'use strict';

  
  if (document.getElementById('fmt-panel-overlay')) {
    document.getElementById('fmt-panel-overlay').remove();
    return;
  }

  
  const styleId = 'fmt-panel-styles';
  if (!document.getElementById(styleId)) {
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = `
      .fmt-fader {
        position: fixed;
        inset: 0;
        z-index: 2147483646;
        cursor: crosshair;
        background: rgba(0, 0, 0, 0.3);
      }
      .fmt-selection {
        position: fixed;
        border: 2px dashed #5c6bc0;
        background: rgba(92, 107, 192, 0.1);
        border-radius: 4px;
        display: none;
        z-index: 2147483647;
        pointer-events: none;
      }
      .fmt-instructions {
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: #1a1a2e;
        color: white;
        padding: 12px 24px;
        border-radius: 10px;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 14px;
        z-index: 2147483647;
        pointer-events: none;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        white-space: nowrap;
      }
      .fmt-panel {
        all: initial;
        position: fixed;
        border: 2px dotted #5c6bc0;
        z-index: 2147483647;
        pointer-events: none;
      }
      .fmt-textbox {
        all: initial;
        position: absolute;
        display: flex;
        align-items: center;
        justify-content: center;
        box-sizing: border-box;
        border-radius: 0;
        background-color: #FFFFFF;
        pointer-events: auto;
        z-index: 2147483647;
        text-align: center;
        font-weight: bold;
        line-height: 1.25;
        overflow: hidden;
        word-break: break-word;
        overflow-wrap: anywhere;
        white-space: normal;
        padding: 8px;
        cursor: default;
      }
      .fmt-textbox-inner {
        all: initial;
        width: 100%;
        max-width: 100%;
        max-height: 100%;
        display: block;
        overflow: hidden;
        text-align: center;
        white-space: pre-line;
        word-break: break-word;
        overflow-wrap: anywhere;
        font-weight: 700;
      }
      .fmt-loading {
        position: fixed;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.9);
        z-index: 2147483647;
        border-radius: 8px;
        border: 2px solid #5c6bc0;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 14px;
        color: #5c6bc0;
        font-weight: 600;
      }
      .fmt-close-btn {
        position: absolute;
        top: -14px;
        right: -14px;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        border: 2px solid #5c6bc0;
        background: white;
        color: #5c6bc0;
        cursor: pointer;
        font-size: 14px;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
        pointer-events: auto;
        z-index: 2147483647;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      }
      .fmt-close-btn:hover {
        background: #5c6bc0;
        color: white;
      }
      @keyframes fmt-spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }
      .fmt-spinner {
        display: inline-block;
        animation: fmt-spin 1s linear infinite;
        margin-right: 8px;
      }
    `;
    document.head.appendChild(style);
  }

  
  const overlay = document.createElement('div');
  overlay.id = 'fmt-panel-overlay';

  const fader = document.createElement('div');
  fader.className = 'fmt-fader';

  const selection = document.createElement('div');
  selection.className = 'fmt-selection';

  const instructions = document.createElement('div');
  instructions.className = 'fmt-instructions';
  instructions.textContent = 'Click and drag to select a manga panel. Press ESC to cancel.';

  overlay.appendChild(fader);
  overlay.appendChild(selection);
  overlay.appendChild(instructions);
  document.body.appendChild(overlay);

  
  let isDragging = false;
  let startX = 0, startY = 0;

  
  fader.addEventListener('mousedown', (e) => {
    e.preventDefault();
    isDragging = true;
    startX = Math.round(e.clientX);
    startY = Math.round(e.clientY);
    selection.style.display = 'block';
    selection.style.left = startX + 'px';
    selection.style.top = startY + 'px';
    selection.style.width = '0px';
    selection.style.height = '0px';
    instructions.style.display = 'none';
  });

  fader.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const currentX = Math.round(e.clientX);
    const currentY = Math.round(e.clientY);
    const w = currentX - startX;
    const h = currentY - startY;

    
    selection.style.left = (w < 0 ? currentX : startX) + 'px';
    selection.style.top = (h < 0 ? currentY : startY) + 'px';
    selection.style.width = Math.abs(w) + 'px';
    selection.style.height = Math.abs(h) + 'px';
  });

  fader.addEventListener('mouseup', async (e) => {
    if (!isDragging) return;
    isDragging = false;

    const currentX = Math.round(e.clientX);
    const currentY = Math.round(e.clientY);
    const w = Math.abs(currentX - startX);
    const h = Math.abs(currentY - startY);
    const left = Math.min(startX, currentX);
    const top = Math.min(startY, currentY);

    
    if (w < 50 || h < 50) {
      cleanup();
      return;
    }

    
    fader.style.display = 'none';
    selection.style.display = 'none';

    
    await new Promise(r => setTimeout(r, 150));

    
    const loading = document.createElement('div');
    loading.className = 'fmt-loading';
    loading.style.cssText = `left:${left}px;top:${top}px;width:${w}px;height:${h}px;`;
    loading.innerHTML = '<span class="fmt-spinner">🔄</span> Translating...';
    document.body.appendChild(loading);

    try {
      const response = await chrome.runtime.sendMessage({
        kind: 'translateSnapshot',
        dimensions: {
          left, top, width: w, height: h,
          devicePixelRatio: window.devicePixelRatio || 1
        }
      });

      loading.remove();

      if (response?.error) {
        showError(response.error, left, top, w, h);
        return;
      }

      if (response?.translatedImageDataUrl) {
        showTranslatedImagePanel(response.translatedImageDataUrl, left, top, w, h);
      } else if (response?.translations && response.translations.length > 0) {
        showTranslationPanel(response.translations, left, top, w, h, response.zoomFactor || 1, window.devicePixelRatio || 1);
      } else {
        showError('No translatable text found in selection', left, top, w, h);
      }
    } catch (error) {
      loading.remove();
      showError(error.message, left, top, w, h);
    }
  });

  
  function precisionRound(num) {
    const m = Number((Math.abs(num) * 100).toPrecision(15));
    return (Math.round(m) / 100) * Math.sign(num);
  }

  
  function wrapTextForWidth(ctx, text, maxWidth) {
    const content = String(text || '');
    if (maxWidth <= 0) return [content];

    const paragraphs = content.split(/\r?\n/);
    const allLines = [];

    for (const paragraph of paragraphs) {
      const words = paragraph.trim().split(/\s+/).filter(Boolean);
      if (words.length === 0) {
        allLines.push('');
        continue;
      }

      let currentLine = '';
      for (const word of words) {
        const testLine = currentLine ? `${currentLine} ${word}` : word;

        if (ctx.measureText(testLine).width <= maxWidth) {
          currentLine = testLine;
          continue;
        }

        if (currentLine) {
          allLines.push(currentLine);
        }

        if (ctx.measureText(word).width <= maxWidth) {
          currentLine = word;
          continue;
        }

        let fragment = '';
        for (const ch of word) {
          const next = fragment + ch;
          if (ctx.measureText(next).width > maxWidth && fragment) {
            allLines.push(fragment);
            fragment = ch;
          } else {
            fragment = next;
          }
        }
        currentLine = fragment;
      }

      if (currentLine) {
        allLines.push(currentLine);
      }
    }

    return allLines.length > 0 ? allLines : [''];
  }

  
  function fitTextStrict(text, boxWidth, boxHeight, fontFamily) {
    const MIN_FONT_SIZE = 7;
    const PADDING = 8;
    const availW = Math.max(1, boxWidth - PADDING * 2);
    const availH = Math.max(1, boxHeight - PADDING * 2);

    const measureCanvas = document.createElement('canvas');
    const measureCtx = measureCanvas.getContext('2d');
    if (!measureCtx) {
      return {
        fontSize: MIN_FONT_SIZE,
        lineHeight: Math.ceil(MIN_FONT_SIZE * 1.2),
        lines: [String(text || '')],
        padding: PADDING,
      };
    }

    let fontSize = Math.floor(Math.min(availH * 0.48, availW * 0.42, 56));
    if (!Number.isFinite(fontSize) || fontSize < MIN_FONT_SIZE) {
      fontSize = MIN_FONT_SIZE;
    }

    let fit = null;
    while (fontSize >= MIN_FONT_SIZE) {
      measureCtx.font = `700 ${fontSize}px "${fontFamily}", "Comic Sans MS", cursive`;
      const lineHeight = Math.ceil(fontSize * 1.2);
      const lines = wrapTextForWidth(measureCtx, text, availW);
      const totalHeight = lines.length * lineHeight;

      let maxLineWidth = 0;
      for (const line of lines) {
        maxLineWidth = Math.max(maxLineWidth, measureCtx.measureText(line).width);
      }

      if (maxLineWidth <= availW && totalHeight <= availH) {
        fit = { fontSize, lineHeight, lines, padding: PADDING };
        break;
      }

      fontSize -= 1;
    }

    if (fit) return fit;

    measureCtx.font = `700 ${MIN_FONT_SIZE}px "${fontFamily}", "Comic Sans MS", cursive`;
    return {
      fontSize: MIN_FONT_SIZE,
      lineHeight: Math.ceil(MIN_FONT_SIZE * 1.2),
      lines: wrapTextForWidth(measureCtx, text, availW),
      padding: PADDING,
    };
  }

  
  function showTranslationPanel(translations, panelX, panelY, panelW, panelH, zoomFactor, dpr) {
    
    
    const scale = zoomFactor * dpr;
    const imgW = panelW * scale;
    const imgH = panelH * scale;

    const panel = document.createElement('div');
    panel.className = 'fmt-panel';
    panel.style.left = panelX + 'px';
    panel.style.top = panelY + 'px';
    panel.style.width = panelW + 'px';
    panel.style.height = panelH + 'px';

    
    const closeBtn = document.createElement('button');
    closeBtn.className = 'fmt-close-btn';
    closeBtn.textContent = '✕';
    closeBtn.addEventListener('click', () => {
      panel.remove();
      cleanup();
    });
    panel.appendChild(closeBtn);

    
    chrome.storage.local.get(['mangaFontStyle', 'mangaFontColor'], (result) => {
      const font = result.mangaFontStyle || 'CC Wild Words';
      const color = result.mangaFontColor || '#000000';

      const MASK_PADDING_PX = 2;

      for (const t of translations) {
        
        const minX = Math.max(0, Math.floor(t.minX - MASK_PADDING_PX));
        const minY = Math.max(0, Math.floor(t.minY - MASK_PADDING_PX));
        const maxX = Math.min(imgW, Math.ceil(t.maxX + MASK_PADDING_PX));
        const maxY = Math.min(imgH, Math.ceil(t.maxY + MASK_PADDING_PX));

        const boxWImgPx = maxX - minX;
        const boxHImgPx = maxY - minY;
        if (boxWImgPx < 8 || boxHImgPx < 8) continue;

        
        const boxLeftPx = minX / scale;
        const boxTopPx = minY / scale;
        const boxWPx = boxWImgPx / scale;
        const boxHPx = boxHImgPx / scale;

        if (boxWPx < 4 || boxHPx < 4) continue;

        const fit = fitTextStrict(t.translatedText, boxWPx, boxHPx, font);

        const textBox = document.createElement('div');
        textBox.className = 'fmt-textbox';
        textBox.style.left = precisionRound(boxLeftPx) + 'px';
        textBox.style.top = precisionRound(boxTopPx) + 'px';
        textBox.style.width = precisionRound(boxWPx) + 'px';
        textBox.style.height = precisionRound(boxHPx) + 'px';
        textBox.style.backgroundColor = 'rgba(255,255,255,1)';

        const textInner = document.createElement('div');
        textInner.className = 'fmt-textbox-inner';
        textInner.style.fontFamily = `"${font}", "Comic Sans MS", cursive`;
        textInner.style.color = color;
        textInner.style.fontSize = `${fit.fontSize}px`;
        textInner.style.lineHeight = `${fit.lineHeight}px`;
        textInner.textContent = fit.lines.join('\n');

        textBox.appendChild(textInner);

        panel.appendChild(textBox);

        
        requestAnimationFrame(() => {
          let strictSize = fit.fontSize;
          while (
            strictSize > 7 &&
            (textInner.scrollWidth > textInner.clientWidth || textInner.scrollHeight > textInner.clientHeight)
          ) {
            strictSize -= 1;
            textInner.style.fontSize = `${strictSize}px`;
            textInner.style.lineHeight = `${Math.ceil(strictSize * 1.2)}px`;
          }
        });
      }
    });

    document.body.appendChild(panel);
    overlay.remove();
  }

  
  function showTranslatedImagePanel(dataUrl, panelX, panelY, panelW, panelH) {
    const panel = document.createElement('div');
    panel.className = 'fmt-panel';
    panel.style.left = panelX + 'px';
    panel.style.top = panelY + 'px';
    panel.style.width = panelW + 'px';
    panel.style.height = panelH + 'px';
    panel.style.background = '#fff';
    panel.style.pointerEvents = 'auto';
    panel.style.overflow = 'hidden';

    const closeBtn = document.createElement('button');
    closeBtn.className = 'fmt-close-btn';
    closeBtn.textContent = '×';
    closeBtn.addEventListener('click', () => {
      panel.remove();
      cleanup();
    });

    const image = document.createElement('img');
    image.alt = 'Translated manga panel';
    image.src = dataUrl;
    image.style.cssText = `
      width: 100%;
      height: 100%;
      display: block;
      object-fit: contain;
      background: #fff;
    `;

    panel.appendChild(image);
    panel.appendChild(closeBtn);
    document.body.appendChild(panel);
    overlay.remove();
  }

  
  function showError(message, x, y, w, h) {
    const errorPanel = document.createElement('div');
    errorPanel.style.cssText = `
      position:fixed; left:${x}px; top:${y}px; width:${w}px; height:${h}px;
      background:rgba(255,255,255,0.95); display:flex; flex-direction:column;
      align-items:center; justify-content:center; z-index:2147483647;
      border-radius:8px; border:2px solid #f44336;
      font-family:-apple-system,BlinkMacSystemFont,sans-serif; gap:10px; padding:16px;
    `;

    const errorText = document.createElement('div');
    errorText.style.cssText = 'color:#f44336;font-size:13px;text-align:center;max-width:90%;';

    
    const friendlyMessages = {
      'NO_API_KEY': 'Please set your API key in the extension popup first',
      'INVALID_API_KEY': 'API key is invalid or expired. Please check your keys in the extension popup.',
      'RATE_LIMITED': 'Rate limited. Please wait a moment and try again.',
      'ALL_KEYS_FAILED': 'All API keys failed. Please check your keys are valid.',
      'FullQueue': 'Translation queue is full. Please wait and try again.',
      'CAPTURE_FAILED': 'Failed to capture the page. Please try again.',
    };
    
    let displayMsg = friendlyMessages[message];
    if (!displayMsg) {
      for (const [key, msg] of Object.entries(friendlyMessages)) {
        if (message && message.startsWith(key)) { displayMsg = msg; break; }
      }
    }
    errorText.textContent = displayMsg || (message && message.length > 150 ? message.substring(0, 150) + '...' : message);

    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Close';
    closeBtn.style.cssText = `
      padding:8px 20px;border:1px solid #ccc;border-radius:6px;
      background:white;cursor:pointer;font-size:13px;
    `;
    closeBtn.addEventListener('click', () => { errorPanel.remove(); cleanup(); });

    errorPanel.appendChild(errorText);
    errorPanel.appendChild(closeBtn);
    document.body.appendChild(errorPanel);
    overlay.remove();
  }

  
  function cleanup() {
    const el = document.getElementById('fmt-panel-overlay');
    if (el) el.remove();
  }

  
  function escHandler(e) {
    if (e.key === 'Escape') {
      cleanup();
      document.removeEventListener('keydown', escHandler);
    }
  }
  document.addEventListener('keydown', escHandler);
})();
