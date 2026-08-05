// Free Manga Translator - Local-only popup script.

document.addEventListener('DOMContentLoaded', () => {
  const translationToggle = document.getElementById('translationToggle');
  const startEngineBtn = document.getElementById('startEngineBtn');
  const translatePageBtn = document.getElementById('translatePageBtn');
  const translationPanelBtn = document.getElementById('translationPanelBtn');
  const pickPanelBtn = document.getElementById('pickPanelBtn');
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
  const queueItemsList = document.getElementById('queueItemsList');
  const localPipelineUrl = document.getElementById('localPipelineUrl');
  const localPipelineAuthToken = document.getElementById('localPipelineAuthToken');
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
  const queueMeter = document.getElementById('queueMeter');
  const clearQueueBtn = document.getElementById('clearQueueBtn');
  const engineStatusText = document.getElementById('engineStatusText');
  const hoverHelp = document.getElementById('hoverHelp');
  const versionBadge = document.getElementById('versionBadge');
  const pipelineModeBadge = document.getElementById('pipelineModeBadge');
  const themeToggleBtn = document.getElementById('themeToggleBtn');

  const DEFAULT_LOCAL_PIPELINE_URL = 'http://127.0.0.1:8766/v1/translate-image';
  // The saved URL is used verbatim for the translate POST but every OTHER endpoint
  // (health, warmup, quota, vram) is derived from it by pathname rewriting -- so a
  // bare origin (e.g. "https://foo.ngrok-free.dev" with no path) passes the health
  // check (which rewrites to /v1/health) while every real translate request 404s.
  // This was the single most likely way to misconfigure a remote/Kaggle backend.
  // Conservative fix: only touch URLs that actually parse; anything that doesn't
  // parse is saved exactly as typed, same as before this existed.
  function normalizePipelineUrl(rawValue) {
    const trimmed = (rawValue || '').trim();
    if (!trimmed) return DEFAULT_LOCAL_PIPELINE_URL;
    let url;
    try {
      url = new URL(trimmed);
    } catch {
      return trimmed; // unparseable -- leave it alone, don't guess
    }
    if (url.pathname === '' || url.pathname === '/') {
      url.pathname = '/v1/translate-image';
    } else if (url.pathname.length > 1 && url.pathname.endsWith('/')) {
      url.pathname = url.pathname.replace(/\/+$/, '');
    }
    return url.toString();
  }

  // The one real local/remote signal in the popup -- was previously inlined only inside
  // updatePipelineModeBadge, so the badge could say REMOTE while every status/error message
  // elsewhere in the popup stayed hardcoded to local wording regardless. Factored out so every
  // caller reads the same signal the badge already computes.
  function isLoopbackUrl(pipelineUrl) {
    try {
      const host = new URL(pipelineUrl).hostname;
      return host === '127.0.0.1' || host === 'localhost' || host === '[::1]';
    } catch {
      return true;
    }
  }

  function currentPipelineIsLoopback() {
    return isLoopbackUrl(localPipelineUrl?.value || DEFAULT_LOCAL_PIPELINE_URL);
  }

  function updatePipelineModeBadge(pipelineUrl) {
    const loopback = isLoopbackUrl(pipelineUrl);
    if (pipelineModeBadge) pipelineModeBadge.textContent = loopback ? 'LOCAL' : 'REMOTE';
    // Start Engine's tooltip (popup.html's static data-help) describes only the local-launcher
    // flow -- "shows the local launcher command because Chrome cannot spawn Python directly" --
    // which is misleading/inapplicable once the backend is remote (Kaggle etc). Rewritten here
    // rather than in HTML since this is the one place the mode is actually known; the hover
    // handler already reads element.dataset.help live at hover time, so mutating the attribute
    // is enough -- no separate re-render call needed.
    if (startEngineBtn && startEngineBtn.dataset) {
      startEngineBtn.dataset.help = loopback
        ? 'Checks whether the local backend is reachable.|If reachable, requests model warmup through /v1/warmup.|If unreachable, shows the local launcher command because Chrome cannot spawn Python directly without a native host.'
        : 'Checks whether the remote backend (e.g. a Kaggle-hosted tunnel) is reachable.|If reachable, requests model warmup through /v1/warmup -- a fresh remote session\'s first warmup can take several minutes.|If unreachable, double-check the pipeline URL below is still current (tunnel URLs can change between sessions).';
    }
  }
  // Must match background.js's own DEFAULT_CACHE_LIMIT: a cache smaller than the
  // queue-ahead depth LRU-evicts the page you started reading before you're done
  // with it. A prior mismatch here (12 vs background's 24, and 24 wasn't even a
  // selectable dropdown option) meant a fresh profile showed "N/24 entries cached"
  // while this dropdown displayed 12 -- a real, user-visible false number.
  const DEFAULT_CACHE_LIMIT = 24;
  const DEFAULT_QUEUE_LIMIT = 20;
  const DEFAULT_PARALLEL_LIMIT = 2;
  const THEME_STORAGE_KEY = 'uiTheme';

  const manifestVersion = chrome.runtime.getManifest?.().version || '1.1.15';
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
      'localPipelineAuthToken',
    ]);

    translationToggle.checked = result.translationEnabled === true;
    localPipelineUrl.value = result.localPipelineUrl || DEFAULT_LOCAL_PIPELINE_URL;
    localPipelineLanguage.value = result.localPipelineLanguage || 'ja';
    if (localPipelineAuthToken) localPipelineAuthToken.value = result.localPipelineAuthToken || '';
    updatePipelineModeBadge(localPipelineUrl.value);
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
        // A failed sendMessage (background script not ready yet, or the message port closed
        // before a response arrived) previously resolved silently to {} here -- every caller's
        // `response.success === false` check then saw a plain {} and fell through as if nothing
        // had gone wrong, so e.g. Clear Cache could report "caches cleared" when the message
        // never actually reached background.js. chrome.runtime.lastError is Chrome's real signal
        // for that failure and must be surfaced as an explicit error, not swallowed into a
        // false-successful-looking empty object.
        if (chrome.runtime.lastError) {
          resolve({ success: false, error: chrome.runtime.lastError.message || 'Extension message failed' });
          return;
        }
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

  function setTextIfChanged(el, text) {
    if (!el || el.textContent === text) return;
    el.textContent = text;
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
    // 400ms mirrors the OS-default hover-tooltip delay -- an instant popup on every stray
    // mouseenter while scanning the toolbar felt like noise, not help. Keyboard focus and click
    // stay instant: those are deliberate, not incidental, so there's nothing to debounce.
    const HELP_HOVER_DELAY_MS = 400;
    document.querySelectorAll('[data-help-panel]').forEach((element) => {
      let showTimer = null;
      const clearShowTimer = () => {
        if (showTimer !== null) {
          clearTimeout(showTimer);
          showTimer = null;
        }
      };
      const show = () => {
        clearShowTimer();
        const items = String(element.dataset.help || '')
          .split('|')
          .map((item) => item.trim())
          .filter(Boolean);
        renderFloatingHelp(element.dataset.helpTitle || element.textContent, items);
        positionFloatingHelp(element);
      };
      const showAfterDelay = () => {
        clearShowTimer();
        showTimer = setTimeout(show, HELP_HOVER_DELAY_MS);
      };
      const hide = () => {
        clearShowTimer();
        if (hoverHelp) hoverHelp.classList.remove('is-visible');
      };
      element.addEventListener('mouseenter', showAfterDelay);
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

  // The gallery previously refreshed on every single stats poll (every 2s) -- each poll fetching
  // up to 40 cached thumbnails just to redraw a preview that essentially never changes between
  // two 2-second ticks. Decoupled to its own, much slower cadence: refreshed on the very first
  // poll (so the gallery isn't empty on open) and then only every GALLERY_REFRESH_EVERY_N_TICKS
  // stats polls after that.
  const GALLERY_REFRESH_EVERY_N_TICKS = 5; // ~10s at the 2s stats-poll cadence
  let statsPollCount = 0;

  function refreshStats() {
    chrome.runtime.sendMessage({ kind: 'getTranslationStats' }, (response) => {
      if (!response) return;
      const cacheSize = response.cacheSize || 0;
      const cacheLimit = response.cacheLimit || 0;
      statsPollCount += 1;
      if (statsPollCount === 1 || statsPollCount % GALLERY_REFRESH_EVERY_N_TICKS === 0) {
        // Show every cached entry, not a fixed slice -- the gallery previously always
        // requested the 6 most-recently-used entries regardless of how many were actually
        // cached, so translating e.g. 15 pages with a cache limit of 15+ only ever showed the
        // last 6 visited, reordered by recency. cacheLimit here is background.js's real
        // configured limit (chrome.storage-backed), not the possibly-unsaved DOM value in
        // the settings input, so it stays correct even before the user clicks Save.
        refreshRecentTranslations(cacheLimit > 0 ? cacheLimit : 6);
      }
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
      // .queue-status-card is an aria-live="polite" region -- writing to it every single 2s poll
      // (even when nothing actually changed) fires a DOM mutation on every tick, and several
      // screen readers announce on ANY live-region mutation regardless of whether the text
      // content actually differs. Only touching textContent when the value genuinely changed
      // keeps the announcements limited to real state changes instead of a 2s metronome.
      setTextIfChanged(activeJobsText, String(activeRequests));
      setTextIfChanged(queuedJobsText, `${queueLength}/${queueLimit}`);
      setTextIfChanged(parallelJobsText, String(parallelLimit));
      if (queueMeterFill) queueMeterFill.style.width = `${pressurePercent}%`;
      if (queueMeter?.setAttribute) queueMeter.setAttribute('aria-valuenow', String(pressurePercent));

      const restartLost = Number(response.restartLost || 0);
      const parts = [];
      if (response.isPaused) parts.push('paused');
      if (activeRequests > 0) parts.push(`${activeRequests} active`);
      if (parallelLimit > 1) parts.push(`parallel ${parallelLimit}`);
      if (queueLength > 0) parts.push(`${queueLength}/${queueLimit} queued`);
      if (restartLost > 0) parts.push(`${restartLost} lost in restart - re-run Translate Page`);
      statsText.textContent = parts.length ? parts.join(' · ') : 'ready';

      renderQueueItems(Array.isArray(response.items) ? response.items : []);

      // Live signal from background.js's own network-level failure detection (the circuit
      // breaker) -- flips the badge to Offline within one poll cycle of a real dispatch failing,
      // instead of waiting for the user to close/reopen the popup or hit Save (checkServerHealth's
      // only two call sites). Intentionally asymmetric: this only ever pushes the badge TOWARD
      // Offline, never away from it -- pipelineBreakerOpen === false just means no recent
      // network-level failure was observed, which is weaker than an actual health-check response,
      // so checkServerHealth() remains the sole source of truth for "Reachable".
      if (response.pipelineBreakerOpen === true) {
        apiStatus.classList.remove('active');
        apiStatus.classList.add('error');
        setEngineStatus('Offline — will resume automatically');
      }
    });
  }

  // Live per-item breakdown so the actual FIFO dispatch order (active items first, then queued
  // in the exact order they'll be dispatched) is visible, not just an aggregate count -- items
  // are keyed by cacheId, which is stable across an unchanged page revisit (see background.js's
  // pendingSrcs/translatedSrcs dedup), so this list only reflects genuinely new/changed work.
  function imageLabelFromUrl(url) {
    try {
      const path = new URL(String(url || '')).pathname;
      const name = path.split('/').filter(Boolean).pop();
      return name ? decodeURIComponent(name) : '';
    } catch {
      return '';
    }
  }

  function renderQueueItems(items) {
    if (!queueItemsList) return;
    if (!items.length) {
      queueItemsList.hidden = true;
      queueItemsList.innerHTML = '';
      return;
    }
    queueItemsList.hidden = false;
    queueItemsList.innerHTML = items.map((item) => {
      const isActive = item.status === 'active';
      const label = [item.pageHost, imageLabelFromUrl(item.originalImageUrl)].filter(Boolean).join(' - ') || 'image';
      const positionLabel = isActive ? '&bull;' : String(item.position ?? '');
      return `
        <div class="queue-item-row">
          <span class="queue-item-position">${positionLabel}</span>
          <span class="queue-item-label" title="${escapeHtml(label)}">${escapeHtml(label)}</span>
          <span class="queue-item-status status-${isActive ? 'active' : 'queued'}">${isActive ? 'active' : 'queued'}</span>
        </div>
      `;
    }).join('');
  }

  function timeAgoLabel(timestampMs) {
    const seconds = Math.max(0, Math.round((Date.now() - Number(timestampMs || 0)) / 1000));
    if (seconds < 60) return 'just now';
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.round(minutes / 60);
    return `${hours}h ago`;
  }

  function refreshRecentTranslations(limit = 6) {
    if (!recentTranslations) return;
    chrome.runtime.sendMessage({ kind: 'getRecentTranslations', limit }, (response) => {
      const entries = Array.isArray(response?.entries) ? response.entries : [];
      if (!entries.length) {
        recentTranslations.hidden = true;
        recentTranslations.innerHTML = '';
        return;
      }
      recentTranslations.hidden = false;
      recentTranslations.innerHTML = entries.map((entry) => {
        const title = `${entry.pageHost || 'unknown page'} - ${timeAgoLabel(entry.lastUsed)}`;
        // entry.thumbnail is backend-controlled (round-tripped through translationCache from
        // the pipeline server's response) and was the one interpolation in this file NOT run
        // through escapeHtml -- a crafted value could break out of the src="" attribute into
        // this privileged extension page's HTML. Every other field here is escaped; this one
        // additionally must be a real data:image/ URL, since nothing else is a legitimate
        // thumbnail source.
        const thumbnailSrc = /^data:image\//i.test(entry.thumbnail || '') ? entry.thumbnail : '';
        return `
          <div class="recent-translation-thumb" title="${escapeHtml(title)}">
            <img src="${escapeHtml(thumbnailSrc)}" alt="${escapeHtml(entry.pageHost || 'Recent translation')}" loading="lazy" />
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
          <div class="quota-detail">${used}/${total} MiB used · ${free} MiB free · ${utilization}% GPU load</div>
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
        statusText.textContent = currentPipelineIsLoopback()
          ? 'Local server unreachable'
          : 'Remote backend unreachable';
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
      const remoteMode = !currentPipelineIsLoopback();
      startEngineBtn.textContent = 'Checking engine...';
      setEngineStatus('Checking...');
      statusText.textContent = remoteMode ? 'Checking remote backend...' : 'Checking local backend...';
      const response = await withButton(startEngineBtn, () => runtimeMessage({ kind: 'startEngine' }));
      apiStatus.classList.toggle('active', response.ok === true);
      apiStatus.classList.toggle('error', response.ok !== true);
      if (response.ok === true) {
        const warmupStatus = response.warmup?.payload?.status || response.warmup?.payload?.warmup?.status || 'warming';
        setEngineStatus(warmupStatus === 'pass' ? 'Ready' : 'Warming');
        if (response.warmup?.skipped) {
          statusText.textContent = 'Engine already reachable';
        } else if (remoteMode && warmupStatus !== 'pass') {
          // A fresh Kaggle session's first warmup downloads several GB of model weights --
          // meaningfully different from a local backend's near-instant warmup, and nothing
          // told the user this before. Static local-only wording here read as a hang.
          statusText.textContent = 'Remote engine warming up -- first run can take several minutes (downloading models)';
        } else {
          statusText.textContent = 'Engine reachable; warmup requested';
        }
        startEngineBtn.textContent = warmupStatus === 'pass' ? 'Engine Ready' : 'Warmup Requested';
        flashAction(startEngineBtn);
        refreshVramStatus().catch(() => {});
      } else {
        setEngineStatus('Offline');
        statusText.textContent = response.message
          || (remoteMode ? 'Could not reach the remote backend' : 'Start the local backend first');
        startEngineBtn.textContent = remoteMode ? 'Remote Offline' : 'Backend Offline';
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

  pickPanelBtn.addEventListener('click', async () => {
    try {
      await withButton(pickPanelBtn, async () => {
        const response = await sendActivePageCommand({ kind: 'togglePanelPicker' });
        if (response?.success === false) throw new Error(response.error || 'Panel picker failed');
        // The outer wrapper only reports whether the message was delivered; content.js's own
        // reply (response.response) reports whether the picker actually started -- it refuses to
        // while the Selection Panel is already open, and that's the one failure mode worth
        // surfacing to the user instead of silently doing nothing.
        if (response?.response?.success === false) {
          throw new Error(response.response.error || 'Panel picker failed');
        }
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
      // An explicit Clear Cache is exactly the kind of real change the decoupled gallery
      // cadence must not delay -- force it to redraw (empty) right away instead of waiting for
      // the next scheduled tick.
      statsPollCount = 0;
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
      // Captured once, up front, before any `await` below -- loadSettings() runs
      // fire-and-forget from page init (never awaited by anything) and can still be
      // mid-flight the first time a user interacts with the popup; if it resolves
      // mid-save it would otherwise blow away whatever was just typed into these
      // fields between reads. Reading once here means the save is atomic with
      // respect to that race, regardless of how slow the background load is.
      const value = normalizePipelineUrl(localPipelineUrl.value);
      const authToken = localPipelineAuthToken ? localPipelineAuthToken.value.trim() : '';
      await withButton(saveLocalPipelineBtn, async () => {
        localPipelineUrl.value = value; // show what was actually stored, e.g. an appended /v1/translate-image
        updatePipelineModeBadge(value);
        await chrome.storage.local.set({
          localPipelineUrl: value,
          localPipelineLanguage: localPipelineLanguage.value || 'ja',
          localPipelineAuthToken: authToken,
        });
        await runtimeMessage({ kind: 'clearCache' });
      });
      flashSaved(saveLocalPipelineBtn);
      // Never show the token itself -- just enough to eyeball-confirm a paste actually
      // landed (a remote/Kaggle 403 is otherwise indistinguishable from "nothing saved"
      // vs. "saved the wrong value", see docs/KAGGLE_USER_MANUAL.md section 10).
      statusText.textContent = authToken
        ? `Local pipeline settings saved (auth token: ${authToken.length} chars)`
        : 'Local pipeline settings saved (no auth token set)';
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

