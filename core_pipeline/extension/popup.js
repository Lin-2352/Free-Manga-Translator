// Free Manga Translator - Local-only popup script.

document.addEventListener('DOMContentLoaded', () => {
  const translationToggle = document.getElementById('translationToggle');
  const startEngineBtn = document.getElementById('startEngineBtn');
  const translatePageBtn = document.getElementById('translatePageBtn');
  const translationPanelBtn = document.getElementById('translationPanelBtn');
  const pauseBtn = document.getElementById('pauseBtn');
  const softStopBtn = document.getElementById('softStopBtn');
  const hardStopBtn = document.getElementById('hardStopBtn');
  const resumeBtn = document.getElementById('resumeBtn');
  const clearBtn = document.getElementById('clearBtn');
  const clearCacheBtn = document.getElementById('clearCacheBtn');
  const retranslateBtn = document.getElementById('retranslateBtn');
  const fontSelect = document.getElementById('fontSelect');
  const fontColorInput = document.getElementById('fontColorInput');
  const statusText = document.getElementById('statusText');
  const statsText = document.getElementById('statsText');
  const cacheStatusText = document.getElementById('cacheStatusText');
  const recentTranslations = document.getElementById('recentTranslations');
  const localPipelineUrl = document.getElementById('localPipelineUrl');
  const localPipelineLanguage = document.getElementById('localPipelineLanguage');
  const translationCachePages = document.getElementById('translationCachePages');
  const translationQueuePages = document.getElementById('translationQueuePages');
  const translationParallelPages = document.getElementById('translationParallelPages');
  const saveLocalPipelineBtn = document.getElementById('saveLocalPipelineBtn');
  const apiStatus = document.getElementById('apiStatus');
  const quotaCards = document.getElementById('quotaCards');
  const quotaAlert = document.getElementById('quotaAlert');
  const quotaSummaryText = document.getElementById('quotaSummaryText');
  const refreshQuotaBtn = document.getElementById('refreshQuotaBtn');
  const openQuotaLogBtn = document.getElementById('openQuotaLogBtn');
  const vramCards = document.getElementById('vramCards');
  const vramAlert = document.getElementById('vramAlert');
  const vramSummaryText = document.getElementById('vramSummaryText');
  const refreshVramBtn = document.getElementById('refreshVramBtn');
  const releaseGpuBtn = document.getElementById('releaseGpuBtn');
  const openVramLogBtn = document.getElementById('openVramLogBtn');
  const activeJobsText = document.getElementById('activeJobsText');
  const queuedJobsText = document.getElementById('queuedJobsText');
  const parallelJobsText = document.getElementById('parallelJobsText');
  const queueMeterFill = document.getElementById('queueMeterFill');
  const clearQueueBtn = document.getElementById('clearQueueBtn');
  const engineStatusText = document.getElementById('engineStatusText');
  const hoverHelp = document.getElementById('hoverHelp');
  const versionBadge = document.getElementById('versionBadge');
  const themeToggleBtn = document.getElementById('themeToggleBtn');

  const DEFAULT_LOCAL_PIPELINE_URL = 'http://127.0.0.1:8766/v1/translate-image';
  const DEFAULT_CACHE_LIMIT = 12;
  const DEFAULT_QUEUE_LIMIT = 20;
  const DEFAULT_PARALLEL_LIMIT = 2;
  const THEME_STORAGE_KEY = 'uiTheme';

  const manifestVersion = chrome.runtime.getManifest?.().version || '1.1.14';
  if (versionBadge) versionBadge.textContent = `v${manifestVersion}`;

  function systemPrefersDark() {
    return Boolean(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
  }

  // forced is the persisted override ('light'/'dark') or null/undefined to follow the OS.
  function effectiveTheme(forced) {
    if (forced === 'light' || forced === 'dark') return forced;
    return systemPrefersDark() ? 'dark' : 'light';
  }

  function applyTheme(forced) {
    if (forced === 'light' || forced === 'dark') {
      document.documentElement.setAttribute('data-theme', forced);
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    const active = effectiveTheme(forced);
    if (themeToggleBtn) {
      themeToggleBtn.classList.toggle('is-dark', active === 'dark');
      themeToggleBtn.setAttribute('aria-pressed', active === 'dark' ? 'true' : 'false');
      themeToggleBtn.setAttribute('aria-label', active === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    }
  }

  async function initTheme() {
    const result = await chrome.storage.local.get([THEME_STORAGE_KEY]);
    applyTheme(result[THEME_STORAGE_KEY] || null);
  }

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', async () => {
      const next = effectiveTheme(document.documentElement.getAttribute('data-theme')) === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      await chrome.storage.local.set({ [THEME_STORAGE_KEY]: next });
    });
  }

  async function loadSettings() {
    const result = await chrome.storage.local.get([
      'translationEnabled',
      'translationPaused',
      'translationCachePages',
      'translationQueuePages',
      'translationParallelPages',
      'mangaFontStyle',
      'mangaFontColor',
      'localPipelineUrl',
      'localPipelineLanguage',
    ]);

    translationToggle.checked = result.translationEnabled === true;
    localPipelineUrl.value = result.localPipelineUrl || DEFAULT_LOCAL_PIPELINE_URL;
    localPipelineLanguage.value = result.localPipelineLanguage || 'ja';
    translationCachePages.value = String(result.translationCachePages ?? DEFAULT_CACHE_LIMIT);
    translationQueuePages.value = String(result.translationQueuePages ?? DEFAULT_QUEUE_LIMIT);
    translationParallelPages.value = String(result.translationParallelPages ?? DEFAULT_PARALLEL_LIMIT);
    if (result.mangaFontStyle) fontSelect.value = result.mangaFontStyle;
    if (result.mangaFontColor) fontColorInput.value = result.mangaFontColor;
    statusText.textContent = result.translationPaused === true ? 'Translation paused' : 'Local pipeline mode';
    setEngineStatus('Not checked');
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

  function flashAction(button) {
    if (!button) return;
    button.classList.add('is-saved');
    setTimeout(() => button.classList.remove('is-saved'), 1200);
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

  function setEngineStatus(text) {
    if (engineStatusText) engineStatusText.textContent = text;
  }

  function renderFloatingHelp(title, items) {
    if (!hoverHelp || !Array.isArray(items) || items.length === 0) return;
    hoverHelp.textContent = '';
    const strong = document.createElement('strong');
    strong.textContent = title || 'Details';
    const list = document.createElement('ul');
    items.forEach((item) => {
      const text = String(item || '').trim();
      if (!text) return;
      const row = document.createElement('li');
      row.textContent = text;
      list.appendChild(row);
    });
    hoverHelp.append(strong, list);
    hoverHelp.classList.toggle('is-visible', list.childElementCount > 0);
  }

  function positionFloatingHelp(element) {
    if (!hoverHelp || typeof element.getBoundingClientRect !== 'function') return;
    const rect = element.getBoundingClientRect();
    const width = Math.min(320, Math.max(220, document.documentElement?.clientWidth - 24 || 320));
    hoverHelp.style.width = `${width}px`;
    const viewportWidth = document.documentElement?.clientWidth || 372;
    const left = Math.max(8, Math.min(rect.left, viewportWidth - width - 8));
    const top = Math.max(8, rect.bottom + 8);
    hoverHelp.style.left = `${left}px`;
    hoverHelp.style.top = `${top}px`;
  }

  function setupHelpPanels() {
    if (typeof document.querySelectorAll !== 'function') return;
    // Visibility is now driven entirely by the .is-visible class (see renderFloatingHelp/hide
    // below) so the panel can fade in/out; the `hidden` attribute from the HTML default only
    // needs clearing once, up front.
    if (hoverHelp) hoverHelp.hidden = false;
    document.querySelectorAll('[data-help-panel]').forEach((element) => {
      const show = () => {
        const items = String(element.dataset.help || '')
          .split('|')
          .map((item) => item.trim())
          .filter(Boolean);
        renderFloatingHelp(element.dataset.helpTitle || element.textContent, items);
        positionFloatingHelp(element);
      };
      const hide = () => {
        if (hoverHelp) hoverHelp.classList.remove('is-visible');
      };
      element.addEventListener('mouseenter', show);
      element.addEventListener('focus', show);
      element.addEventListener('click', show);
      element.addEventListener('mouseleave', hide);
      element.addEventListener('blur', hide);
      // No native `title` attribute here on purpose: it used to duplicate this same
      // description as a second, unstyled browser tooltip stacking on top of this panel
      // ("double hover"). The panel already covers mouse (mouseenter), keyboard (focus),
      // and touch (click), so no native fallback is needed.
    });
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

  async function stopActiveTab(mode) {
    const tab = await activeTab();
    const response = await runtimeMessage({ kind: 'stopTranslations', mode, tabId: tab.id });
    await chrome.storage.local.set({ translationEnabled: false });
    translationToggle.checked = false;
    await runtimeMessage({
      kind: 'sendContentCommand',
      tabId: tab.id,
      command: { kind: 'setTranslationPaused', paused: true },
    });
    if (response.success === false) throw new Error(response.error || `${mode} stop failed`);
    statusText.textContent = mode === 'hard' ? 'Hard stopped; GPU release requested' : 'Soft stopped; models kept warm';
    refreshStats();
    if (mode === 'hard') refreshVramStatus().catch(() => {});
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
    refreshRecentTranslations();
    chrome.runtime.sendMessage({ kind: 'getTranslationStats' }, (response) => {
      if (!response) return;
      const cacheSize = response.cacheSize || 0;
      const cacheLimit = response.cacheLimit || 0;
      const activeRequests = Number(response.activeRequests || 0);
      const queueLength = Number(response.queueLength || 0);
      const queueLimit = Number.isFinite(response.queueLimit) ? response.queueLimit : DEFAULT_QUEUE_LIMIT;
      const parallelLimit = Number.isFinite(response.parallelLimit) ? response.parallelLimit : DEFAULT_PARALLEL_LIMIT;
      const pressurePercent = Number.isFinite(response.pressurePercent)
        ? Math.max(0, Math.min(100, response.pressurePercent))
        : Math.min(100, Math.round(((activeRequests + queueLength) / Math.max(1, parallelLimit + queueLimit)) * 100));
      // "entries", not "images" -- a cache entry is keyed per page+source, so
      // the same image translated on two different pages counts as 2 here.
      cacheStatusText.textContent = cacheLimit > 0
        ? `${cacheSize}/${cacheLimit} entries cached`
        : 'Cache disabled';
      if (activeJobsText) activeJobsText.textContent = String(activeRequests);
      if (queuedJobsText) queuedJobsText.textContent = `${queueLength}/${queueLimit}`;
      if (parallelJobsText) parallelJobsText.textContent = String(parallelLimit);
      if (queueMeterFill) queueMeterFill.style.width = `${pressurePercent}%`;

      const parts = [];
      if (response.isPaused) parts.push('paused');
      if (activeRequests > 0) parts.push(`${activeRequests} active`);
      if (parallelLimit > 1) parts.push(`parallel ${parallelLimit}`);
      if (queueLength > 0) parts.push(`${queueLength}/${queueLimit} queued`);
      statsText.textContent = parts.length ? parts.join(' Â· ') : 'ready';
    });
  }

  function timeAgoLabel(timestampMs) {
    const seconds = Math.max(0, Math.round((Date.now() - Number(timestampMs || 0)) / 1000));
    if (seconds < 60) return 'just now';
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.round(minutes / 60);
    return `${hours}h ago`;
  }

  function refreshRecentTranslations() {
    if (!recentTranslations) return;
    chrome.runtime.sendMessage({ kind: 'getRecentTranslations', limit: 6 }, (response) => {
      const entries = Array.isArray(response?.entries) ? response.entries : [];
      if (!entries.length) {
        recentTranslations.hidden = true;
        recentTranslations.innerHTML = '';
        return;
      }
      recentTranslations.hidden = false;
      recentTranslations.innerHTML = entries.map((entry) => {
        const title = `${entry.pageHost || 'unknown page'} - ${timeAgoLabel(entry.lastUsed)}`;
        return `
          <div class="recent-translation-thumb" title="${escapeHtml(title)}">
            <img src="${entry.thumbnail}" alt="${escapeHtml(entry.pageHost || 'Recent translation')}" loading="lazy" />
            <span class="recent-translation-time">${escapeHtml(timeAgoLabel(entry.lastUsed))}</span>
          </div>
        `;
      }).join('');
    });
  }

  const QUOTA_ROTATE_HINTS = {
    gemini: 'rotate at aistudio.google.com/apikey',
    github: 'rotate at github.com/settings/tokens',
    openrouter: 'rotate at openrouter.ai/keys',
    cloudflare: 'rotate at dash.cloudflare.com (API Tokens)',
    mistral: 'rotate at console.mistral.ai/api-keys',
    groq: 'rotate at console.groq.com/keys',
    cerebras: 'rotate at cloud.cerebras.ai',
    nvidia: 'rotate at build.nvidia.com',
    fireworks: 'rotate at fireworks.ai/account/api-keys',
  };

  function quotaHealthClass(provider) {
    if (!provider?.configured) return 'quota-muted';
    if (provider.health === 'exhausted' || provider.health === 'auth_locked') return 'quota-red';
    if (provider.health === 'rate_limited' || provider.health === 'warning' || Number(provider.remainingPercent) < 50) return 'quota-yellow';
    return 'quota-green';
  }

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function renderQuotaStatus(payload) {
    if (!quotaCards || !quotaAlert) return;
    const providers = Array.isArray(payload?.providers) ? payload.providers : [];
    const configuredProviders = providers.filter((provider) => provider.configured);
    const healthyProviders = configuredProviders.filter((provider) => (
      provider.health !== 'exhausted' && provider.health !== 'auth_locked' && provider.health !== 'rate_limited' && Number(provider.remainingPercent || 0) >= 50
    ));
    quotaAlert.hidden = true;
    quotaAlert.textContent = '';
    if (payload?.globalLimitReached) {
      quotaAlert.hidden = false;
      quotaAlert.textContent = payload.message || 'Daily translation limit reached to protect API quotas.';
      if (quotaSummaryText) quotaSummaryText.textContent = 'Limit reached';
    } else if (payload?.globalAuthLocked) {
      quotaAlert.hidden = false;
      quotaAlert.textContent = payload.message || 'All providers are access-blocked. Check API keys, account access, and network restrictions.';
      if (quotaSummaryText) quotaSummaryText.textContent = 'Access blocked';
    } else if (payload?.globalRateLimitReached) {
      quotaAlert.hidden = false;
      quotaAlert.textContent = payload.message || 'Provider per-minute rate limit reached. Translation will resume automatically after the next minute window.';
      if (quotaSummaryText) quotaSummaryText.textContent = 'Rate limited';
    } else if (payload?.ok === false) {
      quotaAlert.hidden = false;
      quotaAlert.textContent = 'Quota status unavailable. Start the local backend and refresh.';
      if (quotaSummaryText) quotaSummaryText.textContent = 'Unavailable';
    } else if (quotaSummaryText) {
      const envName = payload?.envFileLoaded ? String(payload.envFileLoaded).split(/[\\/]/).pop() : '';
      quotaSummaryText.textContent = configuredProviders.length
        ? `${healthyProviders.length}/${configuredProviders.length} healthy${envName ? ` - ${envName}` : ''}`
        : `No keys configured${envName ? ` - ${envName}` : ''}`;
    }
    if (!providers.length) {
      quotaCards.innerHTML = '<div class="quota-empty">No provider status returned.</div>';
      return;
    }
    quotaCards.innerHTML = providers.map((provider) => {
      const percent = Math.max(0, Math.min(100, Number(provider.remainingPercent || 0)));
      const label = String(provider.provider || 'provider').replace(/(^|-)([a-z])/g, (match) => match.toUpperCase());
      const healthClass = quotaHealthClass(provider);
      const active = Number(provider.activeKeys || 0);
      const total = Number(provider.keyCount || 0);
      const rotateHint = QUOTA_ROTATE_HINTS[String(provider.provider || '').toLowerCase()];
      const detail = provider.configured
        ? (
          provider.health === 'exhausted' && provider.reason
            ? `${active}/${total} keys active - locked: ${provider.reason}`
            : provider.health === 'auth_locked' && provider.reason
              ? `${active}/${total} keys active - access blocked: ${provider.reason}${rotateHint ? ` (${rotateHint})` : ''}`
              : provider.health === 'rate_limited'
                ? `${active}/${total} keys active - rate-limited until ${provider.rateLimitedUntil || 'next minute'}`
                : `${active}/${total} keys active - ${Math.round(percent)}% safe quota`
        )
        : (provider.reason || 'not configured');
      return `
        <div class="quota-card ${healthClass}">
          <div class="quota-card-top">
            <span class="quota-provider">${escapeHtml(label)}</span>
            <span class="quota-pill">${escapeHtml(provider.health || 'unknown')}</span>
          </div>
          <div class="quota-meter" aria-label="${escapeHtml(label)} remaining quota">
            <span style="width:${percent}%"></span>
          </div>
          <div class="quota-detail">${escapeHtml(detail)}</div>
        </div>
      `;
    }).join('');
  }

  async function refreshQuotaStatus() {
    const payload = await runtimeMessage({ kind: 'getQuotaStatus' });
    renderQuotaStatus(payload);
  }

  function vramHealthClass(percent) {
    if (percent >= 90) return 'quota-red';
    if (percent >= 70) return 'quota-yellow';
    return 'quota-green';
  }

  function renderVramStatus(payload) {
    if (!vramCards || !vramAlert) return;
    const gpus = Array.isArray(payload?.gpus) ? payload.gpus : [];
    vramAlert.hidden = true;
    vramAlert.textContent = '';
    if (payload?.ok === false || payload?.available === false) {
      vramAlert.hidden = false;
      vramAlert.textContent = payload.reason || 'VRAM status unavailable. nvidia-smi is required on NVIDIA systems.';
      if (vramSummaryText) vramSummaryText.textContent = 'Unavailable';
    }
    if (!gpus.length) {
      vramCards.innerHTML = '<div class="quota-empty">No GPU VRAM data returned.</div>';
      return;
    }
    const peakUsage = Math.max(...gpus.map((gpu) => Number(gpu.usagePercent || 0)));
    if (vramSummaryText) vramSummaryText.textContent = `${Math.round(peakUsage)}% used`;
    vramCards.innerHTML = gpus.map((gpu) => {
      const used = Number(gpu.memoryUsedMiB || 0);
      const total = Number(gpu.memoryTotalMiB || 0);
      const free = Number(gpu.memoryFreeMiB || 0);
      const percent = Math.max(0, Math.min(100, Number(gpu.usagePercent || 0)));
      const utilization = Number(gpu.utilizationGpuPercent || 0);
      const gpuIndex = escapeHtml(gpu.index ?? 0);
      const gpuName = escapeHtml(gpu.name || 'NVIDIA GPU');
      return `
        <div class="vram-card ${vramHealthClass(percent)}">
          <div class="quota-card-top">
            <span class="quota-provider">GPU ${gpuIndex}: ${gpuName}</span>
            <span class="quota-pill">${Math.round(percent)}% used</span>
          </div>
          <div class="quota-meter" aria-label="GPU ${gpuIndex} VRAM usage">
            <span style="width:${percent}%"></span>
          </div>
          <div class="quota-detail">${used}/${total} MiB used Â· ${free} MiB free Â· ${utilization}% GPU load</div>
        </div>
      `;
    }).join('');
  }

  async function refreshVramStatus() {
    const payload = await runtimeMessage({ kind: 'getVramStatus' });
    renderVramStatus(payload);
  }

  async function openLogWindow(logType) {
    const response = await runtimeMessage({ kind: 'openLogWindow', logType });
    if (response.success === false || response.ok === false) {
      throw new Error(response.error || `Could not open ${logType} log window`);
    }
    statusText.textContent = `${logType === 'vram' ? 'VRAM' : 'Quota'} log window opened`;
  }

  function checkServerHealth() {
    apiStatus.classList.add('checking');
    chrome.runtime.sendMessage({ kind: 'checkPipelineHealth' }, (response) => {
      apiStatus.classList.remove('checking');
      if (!response) return;
      apiStatus.classList.toggle('active', response.ok === true);
      apiStatus.classList.toggle('error', response.ok !== true);
      if (response.ok === true) {
        setEngineStatus('Reachable');
      } else {
        setEngineStatus('Offline');
        statusText.textContent = 'Local server unreachable';
      }
      refreshStats();
      refreshQuotaStatus().catch(() => {});
    });
  }

  function reportPopupError(error) {
    console.error('[FMT popup]', error);
    const message = error?.message || 'Extension command failed';
    statusText.textContent = message;
    if (message.includes('daily translation limit') && quotaAlert) {
      quotaAlert.hidden = false;
      quotaAlert.textContent = message;
      refreshQuotaStatus().catch(() => {});
    }
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

  startEngineBtn?.addEventListener('click', async () => {
    const originalLabel = startEngineBtn.textContent;
    try {
      startEngineBtn.textContent = 'Checking engine...';
      setEngineStatus('Checking...');
      statusText.textContent = 'Checking local backend...';
      const response = await withButton(startEngineBtn, () => runtimeMessage({ kind: 'startEngine' }));
      apiStatus.classList.toggle('active', response.ok === true);
      apiStatus.classList.toggle('error', response.ok !== true);
      if (response.ok === true) {
        const warmupStatus = response.warmup?.payload?.status || response.warmup?.payload?.warmup?.status || 'warming';
        setEngineStatus(warmupStatus === 'pass' ? 'Ready' : 'Warming');
        statusText.textContent = response.warmup?.skipped
          ? 'Engine already reachable'
          : 'Engine reachable; warmup requested';
        startEngineBtn.textContent = warmupStatus === 'pass' ? 'Engine Ready' : 'Warmup Requested';
        flashAction(startEngineBtn);
        refreshVramStatus().catch(() => {});
      } else {
        setEngineStatus('Offline');
        statusText.textContent = response.message || 'Start the local backend first';
        startEngineBtn.textContent = 'Backend Offline';
      }
      refreshStats();
    } catch (error) {
      setEngineStatus('Error');
      reportPopupError(error);
    } finally {
      setTimeout(() => {
        startEngineBtn.textContent = originalLabel || 'Start Engine';
      }, 1400);
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
      await withButton(pauseBtn, () => pauseActiveTab());
    } catch (error) {
      reportPopupError(error);
    }
  });

  softStopBtn.addEventListener('click', async () => {
    try {
      await withButton(softStopBtn, () => stopActiveTab('soft'));
    } catch (error) {
      reportPopupError(error);
    }
  });

  hardStopBtn.addEventListener('click', async () => {
    try {
      await withButton(hardStopBtn, () => stopActiveTab('hard'));
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
        // Visual reset only -- must NOT touch translationCache or the backend
        // runtime cache, or a later Translate Page recomputes everything from
        // scratch instead of restoring instantly from cache. Use the dedicated
        // Clear Cache button for that.
        const response = await sendActivePageCommand({ kind: 'clearTranslations' });
        if (response?.success === false) throw new Error(response.error || 'Clear page failed');
        statusText.textContent = 'Page cleared (cache kept)';
      });
      refreshStats();
    } catch (error) {
      reportPopupError(error);
    }
  });

  clearCacheBtn.addEventListener('click', async () => {
    try {
      const response = await withButton(clearCacheBtn, () => runtimeMessage({ kind: 'clearCache' }));
      if (response.success === false) throw new Error(response.error || 'Cache clear failed');
      statusText.textContent = response.backendCleared === false
        ? 'Browser cache cleared; backend cache unavailable'
        : 'Browser and backend runtime caches cleared';
      refreshStats();
    } catch (error) {
      reportPopupError(error);
    }
  });

  clearQueueBtn?.addEventListener('click', async () => {
    try {
      const response = await withButton(clearQueueBtn, () => runtimeMessage({ kind: 'clearQueue' }));
      if (response.success === false) throw new Error(response.error || 'Queue clear failed');
      statusText.textContent = response.dropped > 0 ? `Cleared ${response.dropped} queued jobs` : 'Queue already empty';
      refreshStats();
    } catch (error) {
      reportPopupError(error);
    }
  });

  retranslateBtn.addEventListener('click', async () => {
    try {
      await withButton(retranslateBtn, async () => {
        // Re-translate resets the page's visual state and re-scans; it does NOT
        // clear the cache -- already-cached pages restore instantly, only
        // genuinely uncached/changed images recompute. Use Clear Cache first if
        // a forced full recompute is actually wanted.
        await runtimeMessage({ kind: 'setTranslationPaused', paused: false });
        const response = await sendActivePageCommand({ kind: 'retranslateAll' });
        if (response?.success === false) throw new Error(response.error || 'Re-translate failed');
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

  refreshQuotaBtn?.addEventListener('click', async () => {
    try {
      await withButton(refreshQuotaBtn, refreshQuotaStatus);
      statusText.textContent = 'Provider quota refreshed';
    } catch (error) {
      reportPopupError(error);
    }
  });

  openQuotaLogBtn?.addEventListener('click', async () => {
    try {
      await withButton(openQuotaLogBtn, () => openLogWindow('quota'));
    } catch (error) {
      reportPopupError(error);
    }
  });

  refreshVramBtn?.addEventListener('click', async () => {
    try {
      await withButton(refreshVramBtn, refreshVramStatus);
      statusText.textContent = 'GPU VRAM refreshed';
    } catch (error) {
      reportPopupError(error);
    }
  });

  releaseGpuBtn?.addEventListener('click', async () => {
    try {
      const response = await withButton(releaseGpuBtn, () => runtimeMessage({ kind: 'releaseGpu' }));
      if (response.success === false && response.status !== 'deferred') {
        throw new Error(response.error || 'GPU release failed');
      }
      statusText.textContent = response.status === 'deferred'
        ? 'GPU release deferred until active work stops'
        : 'GPU models released';
      refreshVramStatus().catch(() => {});
      refreshStats();
    } catch (error) {
      reportPopupError(error);
    }
  });

  openVramLogBtn?.addEventListener('click', async () => {
    try {
      await withButton(openVramLogBtn, () => openLogWindow('vram'));
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

  translationQueuePages.addEventListener('change', async () => {
    try {
      const limit = Number.parseInt(translationQueuePages.value, 10);
      await chrome.storage.local.set({ translationQueuePages: limit });
      const response = await runtimeMessage({ kind: 'setQueueLimit', limit });
      if (response.success === false) throw new Error(response.error || 'Queue limit update failed');
      statusText.textContent = limit > 0 ? `Queue limit set to ${limit}` : 'Queue waits disabled';
      refreshStats();
    } catch (error) {
      reportPopupError(error);
    }
  });

  translationParallelPages.addEventListener('change', async () => {
    try {
      const limit = Number.parseInt(translationParallelPages.value, 10);
      await chrome.storage.local.set({ translationParallelPages: limit });
      const response = await runtimeMessage({ kind: 'setParallelLimit', limit });
      if (response.success === false) throw new Error(response.error || 'Parallel limit update failed');
      statusText.textContent = limit > 1 ? `Adaptive parallel limit set to ${limit}` : 'Sequential mode enabled';
      refreshStats();
    } catch (error) {
      reportPopupError(error);
    }
  });

  fontSelect.addEventListener('change', () => {
    chrome.storage.local.set({ mangaFontStyle: fontSelect.value });
  });

  fontColorInput.addEventListener('input', () => {
    chrome.storage.local.set({ mangaFontColor: fontColorInput.value });
  });

  initTheme();
  loadSettings();
  setupHelpPanels();
  refreshStats();
  refreshQuotaStatus().catch(() => {});
  checkServerHealth();

  // Keeps Active/Queued/Cache numbers current while the popup stays open --
  // there is no push channel from the background service worker, so this is
  // a simple poll. The interval dies with the popup's JS context on close,
  // no teardown needed.
  setInterval(refreshStats, 2000);
});

