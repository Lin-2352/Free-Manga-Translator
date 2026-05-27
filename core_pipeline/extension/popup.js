

document.addEventListener('DOMContentLoaded', () => {
  const translationToggle = document.getElementById('translationToggle');
  const translatePageBtn = document.getElementById('translatePageBtn');
  const translationPanelBtn = document.getElementById('translationPanelBtn');
  const pauseBtn = document.getElementById('pauseBtn');
  const resumeBtn = document.getElementById('resumeBtn');
  const clearBtn = document.getElementById('clearBtn');
  const clearCacheBtn = document.getElementById('clearCacheBtn');
  const retranslateBtn = document.getElementById('retranslateBtn');
  const fontSelect = document.getElementById('fontSelect');
  const fontColorInput = document.getElementById('fontColorInput');
  const statusText = document.getElementById('statusText');
  const statsText = document.getElementById('statsText');
  const cacheStatusText = document.getElementById('cacheStatusText');
  const localPipelineUrl = document.getElementById('localPipelineUrl');
  const localPipelineLanguage = document.getElementById('localPipelineLanguage');
  const translationCachePages = document.getElementById('translationCachePages');
  const saveLocalPipelineBtn = document.getElementById('saveLocalPipelineBtn');
  const apiStatus = document.getElementById('apiStatus');

  const DEFAULT_LOCAL_PIPELINE_URL = 'http://127.0.0.1:8766/v1/translate-image';
  const DEFAULT_CACHE_LIMIT = 12;

  async function loadSettings() {
    const result = await chrome.storage.local.get([
      'translationEnabled',
      'translationPaused',
      'translationCachePages',
      'mangaFontStyle',
      'mangaFontColor',
      'localPipelineUrl',
      'localPipelineLanguage',
    ]);

    translationToggle.checked = result.translationEnabled === true;
    localPipelineUrl.value = result.localPipelineUrl || DEFAULT_LOCAL_PIPELINE_URL;
    localPipelineLanguage.value = result.localPipelineLanguage || 'ja';
    translationCachePages.value = String(result.translationCachePages ?? DEFAULT_CACHE_LIMIT);
    if (result.mangaFontStyle) fontSelect.value = result.mangaFontStyle;
    if (result.mangaFontColor) fontColorInput.value = result.mangaFontColor;
    statusText.textContent = result.translationPaused === true ? 'Translation paused' : 'Local pipeline mode';
  }

  function activeTab() {
    return new Promise((resolve, reject) => {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const tab = tabs[0];
        if (!tab?.id) {
          reject(new Error('No active tab found'));
          return;
        }
        resolve(tab);
      });
    });
  }

  function runtimeMessage(message) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage(message, (response) => {
        resolve(response || {});
      });
    });
  }

  async function sendActivePageCommand(command) {
    const tab = await activeTab();
    return runtimeMessage({
      kind: 'sendContentCommand',
      tabId: tab.id,
      command,
    });
  }

  function setBusy(button, busy) {
    if (!button) return;
    button.disabled = busy === true;
  }

  function flashSaved(button) {
    if (!button) return;
    button.classList.add('is-saved');
    setTimeout(() => button.classList.remove('is-saved'), 1100);
  }

  async function withButton(button, action) {
    setBusy(button, true);
    try {
      const result = await action();
      return result;
    } finally {
      setBusy(button, false);
    }
  }

  async function activateActiveTab(status, options = {}) {
    const tab = await activeTab();
    const persistAuto = options.persistAuto === true;
    const response = await runtimeMessage({
      kind: 'activatePageTranslation',
      tabId: tab.id,
      persistAuto,
    });
    if (response.success === false) throw new Error(response.error || 'Activation failed');
    if (persistAuto) translationToggle.checked = true;
    statusText.textContent = status;
    refreshStats();
    return response;
  }

  async function pauseActiveTab() {
    const tab = await activeTab();
    const response = await runtimeMessage({ kind: 'pausePageTranslation', tabId: tab.id });
    if (response.success === false) throw new Error(response.error || 'Pause failed');
    statusText.textContent = 'Translation paused';
    refreshStats();
    return response;
  }

  function refreshStats() {
    chrome.runtime.sendMessage({ kind: 'getTranslationStats' }, (response) => {
      if (!response) return;
      const cacheSize = response.cacheSize || 0;
      const cacheLimit = response.cacheLimit || 0;
      cacheStatusText.textContent = cacheLimit > 0
        ? `${cacheSize}/${cacheLimit} images cached`
        : 'Cache disabled';

      const parts = [];
      if (response.isPaused) parts.push('paused');
      if (response.activeRequests > 0) parts.push(`${response.activeRequests} active`);
      if (response.queueLength > 0) parts.push(`${response.queueLength} queued`);
      statsText.textContent = parts.length ? parts.join(' · ') : 'ready';
    });
  }

  function checkServerHealth() {
    chrome.runtime.sendMessage({ kind: 'checkPipelineHealth' }, (response) => {
      if (!response) return;
      apiStatus.classList.toggle('active', response.ok === true);
      apiStatus.classList.toggle('error', response.ok !== true);
      if (response.ok !== true) statusText.textContent = 'Local server unreachable; cache cleared';
      refreshStats();
    });
  }

  function reportPopupError(error) {
    console.error('[FMT popup]', error);
    statusText.textContent = error?.message || 'Extension command failed';
    refreshStats();
  }

  translationToggle.addEventListener('change', async () => {
    try {
      const enabled = translationToggle.checked;
      if (enabled) {
        await activateActiveTab('Auto-translate enabled', { persistAuto: true });
      } else {
        await chrome.storage.local.set({ translationEnabled: false });
        await sendActivePageCommand({ kind: 'toggleTranslation', enabled: false });
        statusText.textContent = 'Auto-translation disabled';
        refreshStats();
      }
    } catch (error) {
      translationToggle.checked = false;
      reportPopupError(error);
    }
  });

  translatePageBtn.addEventListener('click', async () => {
    try {
      await withButton(translatePageBtn, () => activateActiveTab('Translating current page...', { persistAuto: false }));
      window.close();
    } catch (error) {
      reportPopupError(error);
    }
  });

  translationPanelBtn.addEventListener('click', async () => {
    try {
      await withButton(translationPanelBtn, async () => {
        const tab = await activeTab();
        await runtimeMessage({ kind: 'setTranslationPaused', paused: false });
        const response = await runtimeMessage({ kind: 'startTranslationPanel', tabId: tab.id });
        if (response.success === false) throw new Error(response.error || 'Selection panel failed');
      });
      window.close();
    } catch (error) {
      reportPopupError(error);
    }
  });

  pauseBtn.addEventListener('click', async () => {
    try {
      await withButton(pauseBtn, pauseActiveTab);
    } catch (error) {
      reportPopupError(error);
    }
  });

  resumeBtn.addEventListener('click', async () => {
    try {
      await withButton(resumeBtn, () => activateActiveTab(
        translationToggle.checked ? 'Auto-translate resumed' : 'Translation resumed for current page',
        { persistAuto: translationToggle.checked },
      ));
    } catch (error) {
      reportPopupError(error);
    }
  });

  clearBtn.addEventListener('click', async () => {
    try {
      await withButton(clearBtn, async () => {
        await sendActivePageCommand({ kind: 'clearTranslations' });
        await runtimeMessage({ kind: 'clearCache' });
      });
      statusText.textContent = 'Translations cleared';
      refreshStats();
    } catch (error) {
      reportPopupError(error);
    }
  });

  clearCacheBtn.addEventListener('click', async () => {
    try {
      await withButton(clearCacheBtn, () => runtimeMessage({ kind: 'clearCache' }));
      statusText.textContent = 'Translation cache cleared';
      refreshStats();
    } catch (error) {
      reportPopupError(error);
    }
  });

  retranslateBtn.addEventListener('click', async () => {
    try {
      await withButton(retranslateBtn, async () => {
        await runtimeMessage({ kind: 'clearCache' });
        await runtimeMessage({ kind: 'setTranslationPaused', paused: false });
        await sendActivePageCommand({ kind: 'retranslateAll' });
      });
      statusText.textContent = 'Re-translating current page...';
      refreshStats();
      window.close();
    } catch (error) {
      reportPopupError(error);
    }
  });

  saveLocalPipelineBtn.addEventListener('click', async () => {
    try {
      await withButton(saveLocalPipelineBtn, async () => {
        const value = localPipelineUrl.value.trim() || DEFAULT_LOCAL_PIPELINE_URL;
        await chrome.storage.local.set({
          localPipelineUrl: value,
          localPipelineLanguage: localPipelineLanguage.value || 'ja',
        });
        await runtimeMessage({ kind: 'clearCache' });
      });
      flashSaved(saveLocalPipelineBtn);
      statusText.textContent = 'Local pipeline settings saved';
      checkServerHealth();
    } catch (error) {
      reportPopupError(error);
    }
  });

  localPipelineLanguage.addEventListener('change', async () => {
    await chrome.storage.local.set({ localPipelineLanguage: localPipelineLanguage.value || 'ja' });
    await runtimeMessage({ kind: 'clearCache' });
    statusText.textContent = 'Source language saved';
    refreshStats();
  });

  translationCachePages.addEventListener('change', async () => {
    const limit = Number.parseInt(translationCachePages.value, 10);
    await chrome.storage.local.set({ translationCachePages: limit });
    await runtimeMessage({ kind: 'setCacheLimit', limit });
    statusText.textContent = limit > 0 ? `Cache limit set to ${limit}` : 'Cache disabled';
    refreshStats();
  });

  fontSelect.addEventListener('change', () => {
    chrome.storage.local.set({ mangaFontStyle: fontSelect.value });
  });

  fontColorInput.addEventListener('input', () => {
    chrome.storage.local.set({ mangaFontColor: fontColorInput.value });
  });

  loadSettings();
  refreshStats();
  checkServerHealth();
});
