(() => {
  "use strict";

  const CATALOG_URL = "/assets/prerenders/catalog.json";
  const DEFAULT_MANIFEST_URL = "/assets/prerender/manifest.json";
  const LAST_MANIFEST_KEY = "atmos20-last-render-manifest-v5";
  const ACTIVE_JOB_KEY = "atmos20-active-render-job-v2";
  const SOURCE_MODE_KEY = "atmos20-result-source-v1";
  const ACTIVE_PROFILE_KEY = "atmos20-prerender-profile-v1";

  function exposeRenderer() {
    const OriginalRenderer = window.AtmosGlobeRenderer;
    if (!OriginalRenderer) return;

    if (typeof OriginalRenderer.prototype.seekFrame !== "function") {
      OriginalRenderer.prototype.seekFrame = async function seekFrame(frame) {
        if (this.fallback || this.timelineFrameCount <= 1) return false;
        const count = Math.max(1, Number(this.timelineFrameCount) || 1);
        const target = ((Math.floor(Number(frame) || 0) % count) + count) % count;
        this.invalidateFrameRequest();
        this.closePendingFrames();
        const ready = await this.requestFramePair(target, {
          force: true,
          sourceToken: this.sourceDecodeToken,
        });
        if (!ready) return false;
        this.commitPendingFrames();
        this.nextFrameAt = performance.now() + this.frameInterval;
        return true;
      };
    }

    class ExposedAtmosGlobeRenderer extends OriginalRenderer {
      constructor(...args) {
        super(...args);
        window.__atmos20Renderer = this;
        window.dispatchEvent(new CustomEvent("atmos-renderer-ready", { detail: this }));
      }
    }
    window.AtmosGlobeRenderer = ExposedAtmosGlobeRenderer;
  }

  exposeRenderer();

  const enhancementStyles = `
    .result-source-card {
      padding: 12px 15px 13px;
      border-bottom: 1px solid var(--line-soft);
      background: rgba(0, 0, 0, .13);
    }
    .source-tabs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px;
      padding: 3px;
      border: 1px solid rgba(255,255,255,.16);
      background: rgba(0,0,0,.18);
    }
    .source-tab {
      min-height: 34px;
      border: 0;
      background: transparent;
      color: rgba(255,255,255,.62);
      cursor: pointer;
      font: 500 9px/1 "IBM Plex Mono", monospace;
    }
    .source-tab.active {
      background: var(--signal);
      color: #172007;
    }
    .prerender-panel {
      margin-top: 11px;
    }
    .prerender-panel[hidden] { display: none; }
    .prerender-panel label > span {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 8px;
    }
    .prerender-panel select {
      width: 100%;
      height: 38px;
      padding: 0 9px;
      border: 1px solid rgba(255,255,255,.18);
      border-radius: 0;
      outline: 0;
      background: rgba(0,0,0,.2);
      color-scheme: dark;
      font: 500 9px/1 "IBM Plex Mono", monospace;
    }
    .prerender-panel select:focus { border-color: var(--signal); }
    .prerender-meta {
      min-height: 48px;
      margin: 9px 0 10px;
      color: var(--muted);
      font-size: 8px;
      line-height: 1.5;
    }
    .prerender-meta strong {
      display: block;
      margin-bottom: 4px;
      color: rgba(255,255,255,.88);
      font: 500 8px/1.3 "IBM Plex Mono", monospace;
    }
    .prerender-load {
      width: 100%;
      min-height: 40px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 11px;
      border: 1px solid var(--signal);
      background: transparent;
      color: var(--signal);
      cursor: pointer;
      font-size: 9px;
    }
    .prerender-load:hover:not(:disabled) {
      background: var(--signal);
      color: #172007;
    }
    .prerender-load:disabled {
      border-color: rgba(255,255,255,.18);
      color: rgba(255,255,255,.42);
      cursor: not-allowed;
    }
    #renderForm[data-source-mode="prerender"] > .job-panel,
    #renderForm[data-source-mode="prerender"] > .preset-strip,
    #renderForm[data-source-mode="prerender"] > .parameter-group:not([data-section="display"]),
    #renderForm[data-source-mode="prerender"] > .estimate-row,
    #renderForm[data-source-mode="prerender"] > .render-button {
      display: none !important;
    }
    .control-dock.atmos-enhanced {
      grid-template-columns: 46px 150px minmax(230px, 1fr) minmax(220px, .72fr) 145px;
      gap: 12px;
    }
    .timeline-scrubber {
      min-width: 0;
      padding-left: 13px;
      border-left: 1px solid rgba(255,255,255,.16);
    }
    .timeline-head,
    .timeline-labels {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .timeline-head small,
    .timeline-labels {
      color: var(--muted);
      font: 500 7px/1 "IBM Plex Mono", monospace;
      letter-spacing: .08em;
    }
    .timeline-head strong {
      overflow: hidden;
      color: rgba(255,255,255,.9);
      text-overflow: ellipsis;
      white-space: nowrap;
      font: 500 8px/1 "IBM Plex Mono", monospace;
    }
    .timeline-range {
      width: 100%;
      height: 18px;
      margin: 2px 0 0;
      appearance: none;
      background: transparent;
      cursor: ew-resize;
    }
    .timeline-range::-webkit-slider-runnable-track {
      height: 2px;
      background: linear-gradient(
        90deg,
        var(--signal) 0 var(--timeline-progress, 0%),
        rgba(255,255,255,.22) var(--timeline-progress, 0%) 100%
      );
    }
    .timeline-range::-webkit-slider-thumb {
      width: 12px;
      height: 12px;
      margin-top: -5px;
      appearance: none;
      border: 2px solid #102020;
      border-radius: 50%;
      background: var(--signal);
    }
    .timeline-range::-moz-range-track { height: 2px; background: rgba(255,255,255,.22); }
    .timeline-range::-moz-range-progress { height: 2px; background: var(--signal); }
    .timeline-range::-moz-range-thumb {
      width: 10px;
      height: 10px;
      border: 2px solid #102020;
      border-radius: 50%;
      background: var(--signal);
    }
    .timeline-range:disabled { cursor: not-allowed; opacity: .45; }
    .dynamic-badge {
      color: var(--signal);
    }
    @media (max-width: 1100px) and (min-width: 761px) {
      .control-dock.atmos-enhanced {
        grid-template-columns: 46px 110px minmax(210px, 1fr) minmax(190px, .8fr);
      }
      .control-dock.atmos-enhanced .media-note { display: none; }
    }
    @media (max-width: 760px) {
      .control-dock.atmos-enhanced {
        grid-template-columns: 44px minmax(0, 1fr);
        gap: 10px;
      }
      .control-dock.atmos-enhanced .level-control { display: none; }
      .timeline-scrubber {
        grid-column: 2;
        padding-left: 0;
        border-left: 0;
      }
      .timeline-head small { display: none; }
      .timeline-labels { font-size: 6px; }
      .source-tab, .prerender-load { min-height: 42px; }
      .prerender-panel select { height: 44px; }
    }
  `;

  function installStyles() {
    const style = document.createElement("style");
    style.id = "atmos20-enhancement-styles";
    style.textContent = enhancementStyles;
    document.head.append(style);
  }

  function localize(value, fallback = "") {
    if (value && typeof value === "object") {
      return value.zh || value.en || fallback;
    }
    return typeof value === "string" ? value : fallback;
  }

  function manifestPath(value) {
    try {
      return new URL(value || DEFAULT_MANIFEST_URL, window.location.href).pathname;
    } catch {
      return DEFAULT_MANIFEST_URL;
    }
  }

  function formatDays(days) {
    const numeric = Number(days);
    if (!Number.isFinite(numeric)) return "—";
    return `${numeric.toFixed(Number.isInteger(numeric) ? 0 : 1)} d`;
  }

  function formatModelHour(hours) {
    const numeric = Number(hours);
    if (!Number.isFinite(numeric)) return "—";
    if (Math.abs(numeric) >= 48) {
      const days = numeric / 24;
      return `D+${days.toFixed(Number.isInteger(days) ? 0 : 1)}`;
    }
    return `H+${numeric.toFixed(Number.isInteger(numeric) ? 0 : 1)}`;
  }

  function fallbackCatalog() {
    return {
      schemaVersion: 1,
      profiles: [{
        id: "circulation-default-10d",
        label: { zh: "标准三圈环流 · 10 天", en: "Standard circulation · 10 days" },
        description: {
          zh: "仓库自带的默认动态结果：5°、900 hPa、5 天建立期加 5 天分析窗。",
          en: "Bundled default dynamic result: 5°, 900 hPa, five-day spin-up plus five-day analysis.",
        },
        available: true,
        manifestUrl: DEFAULT_MANIFEST_URL,
        simulationDays: 10,
        settings: { scenario: "circulation", resolution: 5, level: 900 },
        frames: 48,
        fps: 12,
      }],
    };
  }

  async function fetchCatalog() {
    try {
      const response = await fetch(CATALOG_URL, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const catalog = await response.json();
      if (!Array.isArray(catalog.profiles)) throw new Error("catalog has no profiles");
      return catalog;
    } catch (error) {
      console.warn("Pre-render catalog unavailable; using the bundled default.", error);
      return fallbackCatalog();
    }
  }

  function installSourceSelector() {
    const form = document.querySelector("#renderForm");
    const drawerHead = form?.querySelector(".drawer-head");
    if (!form || !drawerHead) return null;

    const wrapper = document.createElement("section");
    wrapper.className = "result-source-card";
    wrapper.innerHTML = `
      <div class="source-tabs" role="tablist" aria-label="结果来源">
        <button class="source-tab" type="button" role="tab" data-source-mode="prerender">预渲染结果</button>
        <button class="source-tab" type="button" role="tab" data-source-mode="manual">手动计算</button>
      </div>
      <div id="prerenderPanel" class="prerender-panel">
        <label>
          <span>预渲染方案</span>
          <select id="prerenderSelect" aria-label="选择预渲染方案"></select>
        </label>
        <p id="prerenderMeta" class="prerender-meta">正在读取预渲染目录…</p>
        <button id="prerenderLoad" class="prerender-load" type="button" disabled>
          <span>加载预渲染结果</span><i aria-hidden="true">→</i>
        </button>
      </div>
    `;
    drawerHead.insertAdjacentElement("afterend", wrapper);

    const panel = wrapper.querySelector("#prerenderPanel");
    const tabs = [...wrapper.querySelectorAll(".source-tab")];
    const activeJob = localStorage.getItem(ACTIVE_JOB_KEY);
    const savedMode = localStorage.getItem(SOURCE_MODE_KEY);
    let mode = activeJob ? "manual" : (savedMode === "manual" ? "manual" : "prerender");

    const applyMode = (nextMode) => {
      mode = nextMode === "manual" ? "manual" : "prerender";
      form.dataset.sourceMode = mode;
      panel.hidden = mode !== "prerender";
      tabs.forEach((button) => {
        const active = button.dataset.sourceMode === mode;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", String(active));
      });
      localStorage.setItem(SOURCE_MODE_KEY, mode);
    };
    tabs.forEach((button) => button.addEventListener("click", () => applyMode(button.dataset.sourceMode)));
    applyMode(mode);

    return {
      form,
      wrapper,
      select: wrapper.querySelector("#prerenderSelect"),
      meta: wrapper.querySelector("#prerenderMeta"),
      load: wrapper.querySelector("#prerenderLoad"),
      applyMode,
    };
  }

  async function populateSourceSelector(controls) {
    if (!controls) return;
    const catalog = await fetchCatalog();
    const profiles = catalog.profiles;
    const currentPath = manifestPath(
      localStorage.getItem(LAST_MANIFEST_KEY) || DEFAULT_MANIFEST_URL
    );
    const savedProfile = localStorage.getItem(ACTIVE_PROFILE_KEY);
    controls.select.textContent = "";

    for (const profile of profiles) {
      const option = document.createElement("option");
      option.value = profile.id;
      option.disabled = !profile.available || !profile.manifestUrl;
      const availability = option.disabled ? " · 等待云端生成" : "";
      option.textContent = `${localize(profile.label, profile.id)}${availability}`;
      controls.select.append(option);
    }

    const currentProfile = profiles.find(
      (profile) => profile.available && manifestPath(profile.manifestUrl) === currentPath
    );
    const preferred = currentProfile
      || profiles.find((profile) => profile.id === savedProfile && profile.available)
      || profiles.find((profile) => profile.available);
    if (preferred) controls.select.value = preferred.id;

    const update = () => {
      const profile = profiles.find((item) => item.id === controls.select.value);
      if (!profile) {
        controls.meta.textContent = "没有可用的预渲染结果。";
        controls.load.disabled = true;
        return;
      }
      const settings = profile.settings || {};
      const scenario = settings.scenario === "baroclinic" ? "干斜压波" : "全球环流";
      const grid = Number.isFinite(Number(settings.resolution)) ? `${settings.resolution}°` : "—";
      const level = Number.isFinite(Number(settings.level)) ? `${settings.level} hPa` : "—";
      const duration = formatDays(profile.simulationDays);
      const frameText = `${profile.frames || settings.frames || "—"} f @ ${profile.fps || settings.fps || "—"} fps`;
      controls.meta.innerHTML = `
        <strong>${scenario} · ${grid} · ${level} · ${duration}</strong>
        ${localize(profile.description, "")}<br>${frameText}
      `;
      controls.load.disabled = !profile.available
        || !profile.manifestUrl
        || Boolean(localStorage.getItem(ACTIVE_JOB_KEY));
      controls.load.dataset.manifestUrl = profile.manifestUrl || "";
      controls.load.dataset.profileId = profile.id;
      controls.load.querySelector("span").textContent = profile.available
        ? "加载预渲染结果"
        : "该方案尚未生成";
    };

    controls.select.addEventListener("change", update);
    controls.load.addEventListener("click", () => {
      if (controls.load.disabled) return;
      const manifestUrl = controls.load.dataset.manifestUrl;
      if (!manifestUrl) return;
      localStorage.setItem(
        LAST_MANIFEST_KEY,
        new URL(manifestUrl, window.location.href).href
      );
      localStorage.setItem(ACTIVE_PROFILE_KEY, controls.load.dataset.profileId || "");
      localStorage.setItem(SOURCE_MODE_KEY, "prerender");
      controls.load.disabled = true;
      controls.load.querySelector("span").textContent = "正在切换…";
      window.location.reload();
    });
    update();
  }

  function installTimeline() {
    const dock = document.querySelector(".control-dock");
    const mediaNote = dock?.querySelector(".media-note");
    const canvas = document.querySelector("#globeCanvas");
    if (!dock || !mediaNote || !canvas) return null;

    dock.classList.add("atmos-enhanced");
    const timeline = document.createElement("div");
    timeline.className = "timeline-scrubber";
    timeline.innerHTML = `
      <div class="timeline-head">
        <small>MODEL TIME</small>
        <strong id="timelineCurrent">正在读取动态时次…</strong>
      </div>
      <input id="timelineRange" class="timeline-range" type="range" min="0" max="0" value="0" step="1" disabled aria-label="模拟时间轴" />
      <div class="timeline-labels">
        <span id="timelineStart">—</span>
        <span id="timelineMode" class="dynamic-badge">逐帧动态</span>
        <span id="timelineEnd">—</span>
      </div>
    `;
    mediaNote.insertAdjacentElement("beforebegin", timeline);

    const range = timeline.querySelector("#timelineRange");
    const current = timeline.querySelector("#timelineCurrent");
    const start = timeline.querySelector("#timelineStart");
    const end = timeline.querySelector("#timelineEnd");
    const mode = timeline.querySelector("#timelineMode");
    let manifest = null;
    let manifestUrl = "";
    let requestedFrame = null;
    let seekInFlight = false;

    const timelineItem = (frame) => {
      if (!Array.isArray(manifest?.timeline) || !manifest.timeline.length) return null;
      const index = Math.max(0, Math.min(manifest.timeline.length - 1, Number(frame) || 0));
      return manifest.timeline[index];
    };
    const itemHour = (item) => Number(item?.modelHour ?? item?.forecastHour);
    const refreshReadout = (frame) => {
      const count = Math.max(1, Number(canvas.dataset.count || manifest?.frames || 1));
      const safeFrame = Math.max(0, Math.min(count - 1, Number(frame) || 0));
      range.max = String(Math.max(0, count - 1));
      range.value = String(safeFrame);
      range.disabled = count <= 1 || Boolean(window.__atmos20Renderer?.fallback);
      const progress = count <= 1 ? 0 : safeFrame / (count - 1) * 100;
      range.style.setProperty("--timeline-progress", `${progress}%`);

      const item = timelineItem(safeFrame);
      const first = timelineItem(0);
      const last = timelineItem(count - 1);
      current.textContent = item
        ? `${formatModelHour(itemHour(item))} · ${safeFrame + 1}/${count}`
        : `FRAME ${safeFrame + 1}/${count}`;
      start.textContent = first ? formatModelHour(itemHour(first)) : "START";
      end.textContent = last ? formatModelHour(itemHour(last)) : "END";
      const fieldMode = canvas.dataset.fieldMode || "";
      const flowMode = canvas.dataset.flowMode || "";
      mode.textContent = fieldMode === "image-decoder" || flowMode === "image-decoder"
        ? "逐帧动态"
        : (canvas.dataset.renderer === "2d-fallback" ? "二维动画" : "静态层");
    };

    async function refreshManifest(force = false) {
      const nextUrl = localStorage.getItem(LAST_MANIFEST_KEY) || DEFAULT_MANIFEST_URL;
      const absolute = new URL(nextUrl, window.location.href).href;
      if (!force && absolute === manifestUrl) return;
      manifestUrl = absolute;
      try {
        const response = await fetch(absolute, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        manifest = await response.json();
      } catch (error) {
        manifest = null;
        console.warn("Timeline manifest could not be loaded.", error);
      }
      refreshReadout(Number(canvas.dataset.frame || 0));
    }

    async function pumpSeekQueue() {
      if (seekInFlight) return;
      seekInFlight = true;
      try {
        while (requestedFrame !== null) {
          const frame = requestedFrame;
          requestedFrame = null;
          const renderer = window.__atmos20Renderer;
          if (!renderer?.seekFrame) return;
          await renderer.seekFrame(frame);
        }
      } finally {
        seekInFlight = false;
      }
    }

    range.addEventListener("input", () => {
      const frame = Number(range.value);
      refreshReadout(frame);
      requestedFrame = frame;
      pumpSeekQueue();
    });
    canvas.addEventListener("atmos-frame", (event) => {
      refreshReadout(event.detail?.frame || 0);
    });
    window.addEventListener("atmos-renderer-ready", () => refreshReadout(0));
    window.setInterval(() => {
      refreshManifest();
      refreshReadout(Number(canvas.dataset.frame || range.value || 0));
    }, 1200);
    refreshManifest(true);
    return timeline;
  }

  function init() {
    installStyles();
    const sourceControls = installSourceSelector();
    populateSourceSelector(sourceControls);
    installTimeline();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
