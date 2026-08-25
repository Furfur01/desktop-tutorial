const INITIAL_MANIFEST_URL = "/assets/prerender/manifest.json";
const SETTINGS_KEY = "atmos20-render-settings-v1";
const ACTIVE_JOB_KEY = "atmos20-active-render-job-v1";
const LAST_MANIFEST_KEY = "atmos20-last-render-manifest-v2";
const palettes = {
  wind: ["#3d50a3", "#3374b5", "#269ba9", "#36b875", "#8dcc55", "#e2d653", "#e5974e", "#b64173"],
  temperature: ["#4545a5", "#347cc2", "#42b7b1", "#c3d779", "#f4c85f", "#e87545", "#b93455"],
  pressure: ["#513392", "#3d70ba", "#43a9a4", "#d4d18e", "#e39b4e", "#b84961"],
  omega: ["#67339b", "#397dcc", "#47c4c2", "#d6d6b1", "#e99952", "#bc3f62"],
  geopotential: ["#343d82", "#3769a6", "#399a9d", "#75b567", "#c7c65b", "#d48b54"],
  terrain: ["#284f85", "#367d75", "#66a762", "#aabb69", "#b9915e", "#8a684f", "#e1ddd0"],
};

const postFields = [
  "resolution", "level", "region", "season", "spinupHours", "frames", "fps", "particles",
  "flowSpeed", "trail", "tibetScale", "landHeatingScale", "oceanCurrentScale",
];
const numericFields = new Set(postFields.filter((name) => !["region", "season"].includes(name)));
const persistedFields = [...postFields, "flowOpacity", "fieldOpacity"];
const stageOrder = ["queued", "model", "particles", "layers", "encode"];
const stageAliases = {
  queue: "queued",
  pending: "queued",
  initializing: "model",
  spinup: "model",
  integrate: "model",
  integration: "model",
  simulation: "model",
  render: "layers",
  backgrounds: "layers",
  background: "layers",
  flow: "particles",
  particle: "particles",
  manifest: "encode",
  complete: "encode",
  webp: "encode",
  writing: "encode",
};
const stageLabels = {
  queued: "等待计算资源",
  model: "积分大气模型",
  layers: "绘制气象图层",
  particles: "生成流线帧",
  encode: "编码播放素材",
};

const map = document.querySelector(".weather-map");
const form = document.querySelector("#renderForm");
const fieldImage = document.querySelector("#fieldImage");
const flowImage = document.querySelector("#flowImage");
const flowFreeze = document.querySelector("#flowFreeze");
const loader = document.querySelector("#assetLoader");
const renderButton = document.querySelector("#renderButton");

const state = {
  config: null,
  manifest: null,
  manifestUrl: INITIAL_MANIFEST_URL,
  layer: localStorage.getItem("atmos20-prerender-layer") || "wind",
  level: Number(localStorage.getItem("atmos20-prerender-level") || 850),
  playing: true,
  rendering: false,
  jobId: null,
  statusUrl: null,
  pollToken: 0,
  pollFailures: 0,
  assetSwap: 0,
  presets: [],
};

function readSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
  } catch {
    return {};
  }
}

function saveSettings() {
  const values = {};
  for (const name of persistedFields) {
    const control = form.elements.namedItem(name);
    if (control) values[name] = control.value;
  }
  const openSections = [...document.querySelectorAll(".parameter-group")]
    .filter((section) => section.open)
    .map((section) => section.dataset.section);
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({
    values,
    openSections,
    drawerOpen: map.classList.contains("drawer-open"),
  }));
}

function restoreSettings() {
  const saved = readSettings();
  // Select options are supplied by /api/config. Applying saved select values
  // before those options exist can erase a valid 5°/region choice, so values
  // are restored later in applyConfig().
  if (Array.isArray(saved.openSections)) {
    document.querySelectorAll(".parameter-group").forEach((section) => {
      section.open = saved.openSections.includes(section.dataset.section);
    });
  }
  const defaultOpen = !window.matchMedia("(max-width: 760px)").matches;
  setDrawer(typeof saved.drawerOpen === "boolean" ? saved.drawerOpen : defaultOpen, false);
}

function setDrawer(open, persist = true) {
  map.classList.toggle("drawer-open", open);
  document.querySelector("#drawerToggle").setAttribute("aria-expanded", String(open));
  if (persist) saveSettings();
}

function numberLabel(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  const magnitude = Math.abs(numeric);
  if (magnitude >= 100) return numeric.toFixed(0);
  if (magnitude >= 10) return numeric.toFixed(1);
  return numeric.toFixed(2);
}

function setLoading(visible) {
  loader.classList.toggle("visible", visible);
}

function formatOutput(name, value) {
  const numeric = Number(value);
  if (name === "spinupHours") return `${Number.isInteger(numeric) ? numeric.toFixed(0) : numeric.toFixed(1)} h`;
  if (name === "trail" || name === "flowOpacity" || name === "fieldOpacity") return `${Math.round(numeric * 100)}%`;
  return `${numeric.toFixed(2)}×`;
}

function refreshFormReadout() {
  document.querySelectorAll("[data-output]").forEach((output) => {
    const control = form.elements.namedItem(output.dataset.output);
    if (control) output.value = formatOutput(output.dataset.output, control.value);
  });
  const seasonLabel = form.elements.season?.selectedOptions?.[0]?.textContent || "";
  document.querySelector("#modelSummary").textContent =
    `${form.elements.resolution.value}° · ${form.elements.level.value} hPa · ${seasonLabel}`;
  document.querySelector("#animationSummary").textContent =
    `${form.elements.frames.value} f · ${form.elements.fps.value} fps`;
  applyDisplaySettings();
  updateEstimate();
}

function applyDisplaySettings() {
  const flowOpacity = Number(form.elements.flowOpacity.value);
  const fieldOpacity = Number(form.elements.fieldOpacity.value);
  flowImage.style.opacity = String(flowOpacity);
  flowFreeze.style.opacity = String(flowOpacity);
  fieldImage.style.opacity = String(fieldOpacity);
}

function calculateEstimateSeconds() {
  const resolution = Number(form.elements.resolution.value);
  const frames = Number(form.elements.frames.value);
  const particles = Number(form.elements.particles.value);
  const spinupHours = Number(form.elements.spinupHours.value);
  const regionFactor = form.elements.region.value === "world" ? 1.65 : 1;
  const defaultsByResolution = resolution <= 1 ? 100 : resolution <= 2.5 ? 6 : 1;
  const configured = state.config?.estimate?.secondsAtDefaults || state.config?.estimateSecondsByResolution || {};
  const configuredBase = Number(configured[String(resolution)] ?? defaultsByResolution);
  const spinupFactor = 0.4 + 0.6 * Math.max(spinupHours, 0) / 3;
  const mediaFactor = 0.45 + 0.55 * (frames / 72) * (particles / 3600);
  return Math.max(1, configuredBase * regionFactor * (0.55 * spinupFactor + 0.45 * mediaFactor));
}

function durationLabel(seconds) {
  if (seconds < 8) {
    const lower = Math.max(1, Math.floor(seconds * 0.7));
    const upper = Math.max(lower + 1, Math.ceil(seconds * 1.5));
    return `约 ${lower}–${upper} 秒`;
  }
  const lower = Math.max(5, Math.round(seconds * 0.78 / 5) * 5);
  const upper = Math.max(lower + 5, Math.round(seconds * 1.25 / 5) * 5);
  if (upper < 60) return `约 ${lower}–${upper} 秒`;
  const lowerMinutes = Math.max(1, Math.round(lower / 60));
  const upperMinutes = Math.max(lowerMinutes + 1, Math.ceil(upper / 60));
  return `约 ${lowerMinutes}–${upperMinutes} 分钟`;
}

function validateForm(showMessage = true) {
  const message = document.querySelector("#validationMessage");
  if (!state.config) {
    if (showMessage) message.textContent = "计算配置尚未加载。";
    return false;
  }
  const invalid = [...form.querySelectorAll("input, select")].find((control) => !control.validity.valid);
  if (invalid) {
    const label = invalid.closest("label")?.querySelector(":scope > span, :scope > span b")?.textContent || invalid.name;
    if (showMessage) message.textContent = `${label}超出允许范围，请检查。`;
    return false;
  }
  const loopSeconds = Number(form.elements.frames.value) / Number(form.elements.fps.value);
  const loopMinimum = Number(state.config?.limits?.loopSeconds?.min ?? 1);
  const loopMaximum = Number(state.config?.limits?.loopSeconds?.max ?? 6);
  if (loopSeconds < loopMinimum || loopSeconds > loopMaximum) {
    if (showMessage) message.textContent = `帧数 ÷ FPS 需保持在 ${loopMinimum}–${loopMaximum} 秒。`;
    return false;
  }
  const budget = Number(state.config?.limits?.particleFrameBudget);
  const requestedWork = (Number(form.elements.frames.value) + 24) * Number(form.elements.particles.value);
  if (Number.isFinite(budget) && requestedWork > budget) {
    if (showMessage) message.textContent = "当前帧数与粒子数组合过大，请降低其中一项。";
    return false;
  }
  message.textContent = "";
  return true;
}

function updateEstimate() {
  document.querySelector("#estimateTime").textContent = durationLabel(calculateEstimateSeconds());
  validateForm(false);
}

function manifestAssetUrl(source) {
  const manifestUrl = new URL(state.manifestUrl, window.location.href);
  const assetBase = state.manifest?.assetBaseUrl || state.manifest?.assetBase;
  const base = assetBase
    ? new URL(assetBase, manifestUrl)
    : manifestUrl;
  return new URL(source, base).href;
}

function renderLevelLabel() {
  document.querySelector(".fixed-level strong").textContent = `${state.level} hPa`;
}

function updateMapText() {
  if (!state.manifest) return;
  const layer = state.manifest.layers[state.layer];
  const levelKey = String(state.level);
  const range = layer.ranges?.[levelKey] || [0, 1];
  document.querySelector("#layerEnglish").textContent = layer.english;
  document.querySelector("#layerName").textContent = layer.label;
  document.querySelector("#levelLabel").textContent = `${state.level} hPa`;
  document.querySelector("#loopLabel").textContent = `${Number(state.manifest.durationSeconds || state.manifest.frames / state.manifest.fps).toFixed(1)} s loop`;
  document.querySelector("#legendName").textContent = layer.english;
  document.querySelector("#legendUnit").textContent = layer.unit;
  document.querySelector("#legendMin").textContent = numberLabel(range[0]);
  document.querySelector("#legendMid").textContent = numberLabel((Number(range[0]) + Number(range[1])) / 2);
  document.querySelector("#legendMax").textContent = numberLabel(range[1]);
  document.querySelector("#legendBar").style.background = `linear-gradient(90deg,${(palettes[state.layer] || palettes.wind).join(",")})`;
  document.querySelector("#mediaNote").textContent = `${state.manifest.frames} FRAMES · READY`;
  if (!state.rendering) document.querySelector("#renderStatus").textContent = `READY · ${state.manifest.fps} FPS`;
  const viewportKey = state.manifest.viewport?.key || state.manifest.viewport?.name || state.manifest.region?.key || state.manifest.region;
  const viewportLabels = { east_asia_pacific: "EAST ASIA / PACIFIC", east_asia: "EAST ASIA / PACIFIC", asia: "ASIA", world: "WORLD", global: "WORLD" };
  document.querySelector("#regionLabel").textContent =
    state.manifest.regionLabel || state.manifest.region?.label || viewportLabels[viewportKey] || "EAST ASIA / PACIFIC";
  document.querySelectorAll(".layer").forEach((button) => {
    const available = Boolean(state.manifest.layers[button.dataset.layer]);
    button.disabled = !available;
    button.classList.toggle("active", button.dataset.layer === state.layer);
  });
  localStorage.setItem("atmos20-prerender-layer", state.layer);
  localStorage.setItem("atmos20-prerender-level", String(state.level));
  renderLevelLabel();
}

function loadImage(element, source) {
  return new Promise((resolve, reject) => {
    const generated = state.manifest.generated || state.jobId || Date.now();
    const url = new URL(manifestAssetUrl(source));
    url.searchParams.set("v", generated);
    element.onload = () => resolve();
    element.onerror = () => reject(new Error(`无法加载 ${url.pathname}`));
    element.src = url.href;
  });
}

function freezeCurrentFrame() {
  if (!flowImage.naturalWidth || !flowImage.naturalHeight) return false;
  flowFreeze.width = flowImage.naturalWidth;
  flowFreeze.height = flowImage.naturalHeight;
  const context = flowFreeze.getContext("2d");
  context.clearRect(0, 0, flowFreeze.width, flowFreeze.height);
  context.drawImage(flowImage, 0, 0, flowFreeze.width, flowFreeze.height);
  flowFreeze.classList.add("visible");
  flowImage.classList.add("paused");
  return true;
}

function setPlaying(playing) {
  state.playing = playing;
  const playButton = document.querySelector("#playButton");
  if (playing) {
    flowFreeze.classList.remove("visible");
    flowImage.classList.remove("paused");
  } else if (!freezeCurrentFrame()) {
    flowImage.classList.add("paused");
  }
  playButton.classList.toggle("paused", !playing);
  playButton.setAttribute("aria-label", playing ? "暂停动画" : "播放动画");
}

async function swapAssets(changeFlow) {
  if (!state.manifest) return;
  const layer = state.manifest.layers[state.layer];
  const levelKey = String(state.level);
  const fieldSource = layer?.assets?.[levelKey];
  const flowSource = state.manifest.particles?.[levelKey];
  if (!fieldSource || (changeFlow && !flowSource)) throw new Error("清单缺少当前层级的播放素材");
  const swap = ++state.assetSwap;
  setLoading(true);
  fieldImage.style.opacity = String(Number(form.elements.fieldOpacity.value) * 0.45);
  try {
    const loads = [loadImage(fieldImage, fieldSource)];
    if (changeFlow) loads.push(loadImage(flowImage, flowSource));
    await Promise.all(loads);
    if (swap !== state.assetSwap) return;
    applyDisplaySettings();
    setPlaying(state.playing);
    updateMapText();
  } finally {
    if (swap === state.assetSwap) setLoading(false);
  }
}

function prefetchAssets() {
  if (!state.manifest) return;
  const sources = [];
  for (const layer of Object.values(state.manifest.layers)) {
    for (const source of Object.values(layer.assets || {})) sources.push(source);
  }
  for (const source of Object.values(state.manifest.particles || {})) sources.push(source);
  sources.forEach((source) => {
    const image = new Image();
    image.src = manifestAssetUrl(source);
  });
}

async function activateManifest(manifestUrl, autoPlay = false) {
  const response = await fetch(manifestUrl, { cache: "no-store" });
  if (!response.ok) throw new Error(`清单请求失败（HTTP ${response.status}）`);
  const manifest = await response.json();
  if (!manifest.layers || !manifest.particles) throw new Error("生成结果缺少图层或流线素材");
  state.manifest = manifest;
  state.manifestUrl = new URL(manifestUrl, window.location.href).href;
  const levels = (manifest.levels || []).map(Number);
  const requestedLevel = Number(form.elements.level.value);
  state.level = levels.includes(requestedLevel)
    ? requestedLevel
    : (levels.includes(Number(manifest.defaultLevel)) ? Number(manifest.defaultLevel) : levels[0]);
  if (!manifest.layers[state.layer]) state.layer = manifest.defaultLayer || Object.keys(manifest.layers)[0];
  if (autoPlay) state.playing = true;
  await swapAssets(true);
  setPlaying(state.playing);
  setTimeout(prefetchAssets, 600);
}

function optionItems(raw) {
  if (!raw) return [];
  if (Array.isArray(raw)) {
    return raw.map((item) => typeof item === "object"
      ? { value: String(item.value ?? item.id), label: item.label ?? item.name ?? String(item.value ?? item.id) }
      : { value: String(item), label: String(item) });
  }
  return Object.entries(raw).map(([value, label]) => ({
    value,
    label: typeof label === "object" ? label.label || label.name || value : String(label),
  }));
}

function replaceOptions(name, raw, suffix = "") {
  const items = optionItems(raw);
  if (!items.length) return;
  const select = form.elements.namedItem(name);
  const previous = select.value;
  select.replaceChildren(...items.map((item) => {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label.includes(suffix) ? item.label : `${item.label}${suffix}`;
    return option;
  }));
  if ([...select.options].some((option) => option.value === previous)) select.value = previous;
}

function applyRangeConfig(name, definition) {
  if (!definition) return;
  const control = form.elements.namedItem(name);
  if (!control) return;
  if (Array.isArray(definition)) definition = { min: definition[0], max: definition[1] };
  for (const attribute of ["min", "max", "step"]) {
    const aliases = attribute === "min" ? ["min", "minimum"] : attribute === "max" ? ["max", "maximum"] : ["step"];
    const key = aliases.find((candidate) => definition[candidate] !== undefined);
    if (key) control[attribute] = definition[key];
  }
}

function assignAllowedValue(control, value) {
  if (!control || value === undefined || value === null) return false;
  if (control instanceof HTMLSelectElement) {
    if (![...control.options].some((option) => option.value === String(value))) return false;
    control.value = String(value);
    return true;
  }
  if (control.type === "number" || control.type === "range") {
    const numeric = Number(value);
    const min = control.min === "" ? -Infinity : Number(control.min);
    const max = control.max === "" ? Infinity : Number(control.max);
    if (!Number.isFinite(numeric)) return false;
    control.value = String(Math.min(max, Math.max(min, numeric)));
    return true;
  }
  control.value = value;
  return true;
}

function presetItems(raw) {
  if (Array.isArray(raw)) return raw;
  if (!raw || typeof raw !== "object") return [];
  return Object.entries(raw).map(([id, preset]) => ({ id, ...(preset || {}) }));
}

function renderPresets(rawPresets) {
  const presets = presetItems(rawPresets).filter((preset) => preset.settings && typeof preset.settings === "object");
  state.presets = presets;
  const strip = document.querySelector("#presetStrip");
  const container = document.querySelector("#presetButtons");
  container.replaceChildren();
  strip.hidden = presets.length === 0;
  for (const preset of presets) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preset-button";
    button.dataset.preset = preset.id || preset.value || preset.label;
    button.textContent = preset.label || preset.name || preset.id;
    button.addEventListener("click", () => {
      for (const [name, value] of Object.entries(preset.settings)) {
        assignAllowedValue(form.elements.namedItem(name), value);
      }
      refreshFormReadout();
      saveSettings();
      syncPresetSelection();
    });
    container.append(button);
  }
  syncPresetSelection();
}

function syncPresetSelection() {
  document.querySelectorAll(".preset-button").forEach((button) => {
    const preset = state.presets.find((item) => String(item.id || item.value || item.label) === button.dataset.preset);
    const matches = preset && Object.entries(preset.settings).every(([name, value]) => {
      const control = form.elements.namedItem(name);
      if (!control) return true;
      return numericFields.has(name) ? Number(control.value) === Number(value) : control.value === String(value);
    });
    button.classList.toggle("active", Boolean(matches));
  });
}

function applyConfig(config) {
  state.config = config;
  const options = config.options || {};
  const definitions = config.parameters || {};
  replaceOptions("resolution", config.resolutions || options.resolutions || options.resolution || definitions.resolution?.options, "°");
  replaceOptions("level", config.levels || options.levels || options.level || definitions.level?.options, " hPa");
  replaceOptions("region", config.regions || options.regions || options.region || definitions.region?.options);
  replaceOptions("season", config.seasons || options.seasons || options.season || definitions.season?.options);
  for (const name of persistedFields) {
    applyRangeConfig(name, config.ranges?.[name] || definitions[name] || config.limits?.[name]);
  }
  const defaults = { ...(config.defaults || config.defaultConfig || {}) };
  for (const [name, definition] of Object.entries(definitions)) {
    if (defaults[name] === undefined && definition?.default !== undefined) defaults[name] = definition.default;
  }
  const savedValues = readSettings().values || {};
  for (const [name, value] of Object.entries(defaults)) {
    const control = form.elements.namedItem(name);
    if (control && savedValues[name] === undefined) assignAllowedValue(control, value);
  }
  for (const [name, value] of Object.entries(savedValues)) {
    const control = form.elements.namedItem(name);
    if (control && !assignAllowedValue(control, value) && defaults[name] !== undefined) assignAllowedValue(control, defaults[name]);
  }
  renderPresets(config.presets);
  refreshFormReadout();
  setRendering(state.rendering);
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    applyConfig(await response.json());
  } catch (error) {
    console.warn("Render config unavailable; using safe defaults.", error);
    document.querySelector("#jobMessage").textContent = "计算接口尚未就绪；当前结果仍可浏览。";
    setRendering(false);
  }
}

function buildPayload() {
  const payload = {};
  for (const name of postFields) {
    const value = form.elements.namedItem(name).value;
    payload[name] = numericFields.has(name) ? Number(value) : value;
  }
  return payload;
}

function normalizedStage(raw) {
  const stage = String(raw || "queued").toLowerCase();
  return stageOrder.includes(stage) ? stage : (stageAliases[stage] || "queued");
}

function setJobVisual(status, progress, stage, message) {
  const jobPanel = document.querySelector("#jobPanel");
  const safeProgress = Math.max(0, Math.min(100, Number(progress) || 0));
  const normalized = normalizedStage(stage);
  const statusText = {
    idle: "示例已就绪",
    queued: "已提交 · 等待中",
    running: "正在计算",
    completed: "计算完成 · 正在播放",
    error: "计算失败",
  }[status] || status;
  jobPanel.className = `job-panel ${status}`;
  document.querySelector("#jobState").textContent = statusText;
  document.querySelector("#jobPercent").textContent = status === "idle" ? "—" : `${Math.round(safeProgress)}%`;
  document.querySelector("#jobProgress").style.width = `${safeProgress}%`;
  document.querySelector("#jobMessage").textContent = message || stageLabels[normalized] || "";
  const activeIndex = stageOrder.indexOf(normalized);
  document.querySelectorAll(".stage-dots li").forEach((item, index) => {
    item.classList.toggle("done", status === "completed" || index < activeIndex);
    item.classList.toggle("active", ["queued", "running"].includes(status) && index === activeIndex);
  });
  document.querySelector("#renderStatus").textContent = status === "running"
    ? `COMPUTING · ${Math.round(safeProgress)}%`
    : status === "queued" ? "QUEUED" : status === "error" ? "COMPUTE ERROR" : document.querySelector("#renderStatus").textContent;
}

function setRendering(rendering) {
  state.rendering = rendering;
  const available = Boolean(state.config);
  renderButton.disabled = rendering || !available;
  document.querySelector("#renderButtonLabel").textContent = rendering
    ? "正在计算…"
    : available ? "开始计算" : "计算服务不可用";
  form.setAttribute("aria-busy", String(rendering));
}

function jobPayload(data) {
  return data?.job || data?.render || data;
}

function saveActiveJob(jobId, statusUrl) {
  state.jobId = jobId;
  state.statusUrl = statusUrl || `/api/render/${encodeURIComponent(jobId)}`;
  localStorage.setItem(ACTIVE_JOB_KEY, JSON.stringify({ jobId, statusUrl: state.statusUrl }));
}

function clearActiveJob() {
  state.jobId = null;
  state.statusUrl = null;
  localStorage.removeItem(ACTIVE_JOB_KEY);
}

function rememberManifest(manifestUrl) {
  localStorage.setItem(LAST_MANIFEST_KEY, new URL(manifestUrl, window.location.href).href);
}

function readRememberedManifest() {
  return localStorage.getItem(LAST_MANIFEST_KEY) || INITIAL_MANIFEST_URL;
}

function readActiveJob() {
  try {
    const active = JSON.parse(localStorage.getItem(ACTIVE_JOB_KEY) || "null");
    return active?.jobId ? active : null;
  } catch {
    localStorage.removeItem(ACTIVE_JOB_KEY);
    return null;
  }
}

async function pollJob(jobId, token) {
  if (token !== state.pollToken) return;
  try {
    const statusUrl = state.statusUrl || `/api/render/${encodeURIComponent(jobId)}`;
    const response = await fetch(statusUrl, { cache: "no-store" });
    if (!response.ok) {
      const statusError = new Error(`状态请求失败（HTTP ${response.status}）`);
      statusError.status = response.status;
      throw statusError;
    }
    const job = jobPayload(await response.json());
    if (job.statusUrl && job.statusUrl !== state.statusUrl) saveActiveJob(jobId, job.statusUrl);
    state.pollFailures = 0;
    const status = String(job.status || "running").toLowerCase();
    const rawProgress = Number(job.progress ?? job.percent ?? 0);
    const progress = rawProgress > 0 && rawProgress <= 1 ? rawProgress * 100 : rawProgress;
    const stage = job.stage || job.phase || "model";
    if (["error", "failed", "cancelled"].includes(status)) {
      const jobError = new Error(job.error?.message || job.error || job.message || "渲染任务未完成");
      jobError.terminal = true;
      throw jobError;
    }
    if (["completed", "complete", "done", "success"].includes(status)) {
      setJobVisual("running", 100, "encode", "载入刚刚生成的播放素材…");
      const manifestUrl = job.manifestUrl || job.result?.manifestUrl || job.output?.manifestUrl;
      if (!manifestUrl) throw new Error("任务完成，但没有返回 manifestUrl");
      await activateManifest(manifestUrl, true);
      rememberManifest(manifestUrl);
      setJobVisual("completed", 100, "encode", "新结果已载入并自动播放。");
      setRendering(false);
      clearActiveJob();
      document.querySelector("#renderStatus").textContent = `READY · ${state.manifest.fps} FPS`;
      return;
    }
    setJobVisual(status === "queued" ? "queued" : "running", progress, stage, job.message);
    window.setTimeout(() => pollJob(jobId, token), Number(job.pollAfterMs || 800));
  } catch (error) {
    state.pollFailures += 1;
    if (!error.terminal && error.status !== 404 && token === state.pollToken) {
      const delay = Math.min(5000, 1000 + state.pollFailures * 700);
      setJobVisual("running", 0, "queued", "状态连接暂时中断；后台任务仍在继续，正在重连…");
      window.setTimeout(() => pollJob(jobId, token), delay);
      return;
    }
    failJob(error);
  }
}

function failJob(error) {
  state.pollToken += 1;
  setRendering(false);
  clearActiveJob();
  setJobVisual("error", 0, "queued", error.message || "计算失败，请检查参数后重试。");
  console.error(error);
}

async function submitRender(event) {
  event.preventDefault();
  if (state.rendering || !validateForm(true)) return;
  saveSettings();
  setRendering(true);
  setDrawer(true);
  setJobVisual("queued", 0, "queued", "参数已提交，正在创建计算任务…");
  const token = ++state.pollToken;
  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const fallback = response.status === 409 ? "已有计算任务正在运行，请等待它完成。" : `提交失败（HTTP ${response.status}）`;
      const serverMessage = typeof body.error === "string" ? body.error : body.error?.message;
      throw new Error(serverMessage || body.message || fallback);
    }
    const jobId = body.jobId || body.id || body.job?.id;
    if (!jobId) throw new Error("计算接口没有返回 jobId");
    saveActiveJob(jobId, body.statusUrl);
    state.pollFailures = 0;
    if (body.estimateSeconds) {
      document.querySelector("#estimateTime").textContent = durationLabel(Number(body.estimateSeconds));
    }
    pollJob(jobId, token);
  } catch (error) {
    failJob(error);
  }
}

document.querySelector("#drawerToggle").addEventListener("click", () => setDrawer(true));
document.querySelector("#drawerClose").addEventListener("click", () => setDrawer(false));
document.querySelectorAll(".parameter-group").forEach((section) => section.addEventListener("toggle", () => {
  if (state.config) saveSettings();
}));

form.addEventListener("input", () => {
  refreshFormReadout();
  syncPresetSelection();
  saveSettings();
});
form.addEventListener("change", () => {
  refreshFormReadout();
  syncPresetSelection();
  saveSettings();
});
form.addEventListener("submit", submitRender);

document.querySelectorAll(".layer").forEach((button) => button.addEventListener("click", async () => {
  if (!state.manifest || button.dataset.layer === state.layer || button.disabled) return;
  state.layer = button.dataset.layer;
  try {
    await swapAssets(false);
  } catch (error) {
    document.querySelector("#renderStatus").textContent = "LAYER ASSET ERROR";
    console.error(error);
  }
}));

document.querySelector("#playButton").addEventListener("click", () => setPlaying(!state.playing));

async function start() {
  restoreSettings();
  refreshFormReadout();
  setLoading(true);
  const tasks = [loadConfig()];
  const rememberedManifest = readRememberedManifest();
  try {
    await activateManifest(rememberedManifest);
  } catch (error) {
    if (rememberedManifest !== INITIAL_MANIFEST_URL) {
      localStorage.removeItem(LAST_MANIFEST_KEY);
      try {
        await activateManifest(INITIAL_MANIFEST_URL);
      } catch (fallbackError) {
        document.querySelector("#renderStatus").textContent = "INITIAL RENDER ERROR";
        setJobVisual("error", 0, "queued", fallbackError.message);
        console.error(fallbackError);
      }
    } else {
      document.querySelector("#renderStatus").textContent = "INITIAL RENDER ERROR";
      setJobVisual("error", 0, "queued", error.message);
      console.error(error);
    }
  } finally {
    setLoading(false);
  }
  await Promise.all(tasks);
  const active = readActiveJob();
  if (active) {
    saveActiveJob(active.jobId, active.statusUrl);
    state.pollFailures = 0;
    setRendering(true);
    setDrawer(true);
    setJobVisual("queued", 0, "queued", "已恢复计算任务，正在获取最新状态…");
    pollJob(active.jobId, ++state.pollToken);
  }
}

start();
