const INITIAL_MANIFEST_URL = "/assets/prerender/manifest.json";
const SETTINGS_KEY = "atmos20-render-settings-v3";
const ACTIVE_JOB_KEY = "atmos20-active-render-job-v2";
const LAST_MANIFEST_KEY = "atmos20-last-render-manifest-v5";
const LAYER_KEY = "atmos20-prerender-layer-v3";
const LEVEL_KEY = "atmos20-prerender-level-v2";
const UI_DEFAULTS = {
  scenario: "circulation",
  spinupHours: 120,
  analysisHours: 120,
  equatorToPoleContrastK: 60,
  surfaceDragDays: 1,
  seasonalHeatEquatorDeg: 0,
  tibetScale: 1,
  landHeatingScale: 1,
  oceanCurrentScale: 1,
};
const palettes = {
  wind: ["#3d50a3", "#3374b5", "#269ba9", "#36b875", "#8dcc55", "#e2d653", "#e5974e", "#b64173"],
  temperature: ["#4545a5", "#347cc2", "#42b7b1", "#c3d779", "#f4c85f", "#e87545", "#b93455"],
  pressure: ["#513392", "#3d70ba", "#43a9a4", "#d4d18e", "#e39b4e", "#b84961"],
  omega: ["#67339b", "#397dcc", "#47c4c2", "#d6d6b1", "#e99952", "#bc3f62"],
  geopotential: ["#343d82", "#3769a6", "#399a9d", "#75b567", "#c7c65b", "#d48b54"],
  terrain: ["#284f85", "#367d75", "#66a762", "#aabb69", "#b9915e", "#8a684f", "#e1ddd0"],
  fronts: ["#38439b", "#3478bc", "#47b5b0", "#d4d684", "#efb34e", "#cf5262"],
  zonalWind: ["#42379a", "#3169bd", "#35a2c1", "#d9e1d9", "#e5b95a", "#df7049", "#a93255"],
};

const postFields = [
  "scenario", "resolution", "level", "season", "spinupHours", "analysisHours", "frames", "fps", "particles",
  "flowSpeed", "trail", "equatorToPoleContrastK", "surfaceDragDays", "seasonalHeatEquatorDeg",
  "tibetScale", "landHeatingScale", "oceanCurrentScale",
  "jetStrength", "perturbationAmplitude", "hemisphere",
];
const numericFields = new Set(postFields.filter((name) => !["scenario", "season", "hemisphere"].includes(name)));
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
const globeCanvas = document.querySelector("#globeCanvas");
const fieldImage = document.querySelector("#fieldImage");
const flowImage = document.querySelector("#flowImage");
const loader = document.querySelector("#assetLoader");
const renderButton = document.querySelector("#renderButton");
let globeRenderer = null;

const state = {
  config: null,
  manifest: null,
  manifestUrl: INITIAL_MANIFEST_URL,
  layer: localStorage.getItem(LAYER_KEY) || "zonalWind",
  level: Number(localStorage.getItem(LEVEL_KEY) || 900),
  playing: !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  rendering: false,
  jobId: null,
  statusUrl: null,
  pollToken: 0,
  pollFailures: 0,
  assetSwap: 0,
  presets: [],
};

try {
  globeRenderer = new window.AtmosGlobeRenderer(globeCanvas, {
    mapElement: map,
    fallbackElement: document.querySelector("#globeFallback"),
  });
} catch (error) {
  document.querySelector(".map-media")?.classList.add("globe-fallback-mode");
  document.querySelector("#globeFallback").hidden = false;
  console.error("Globe renderer initialization failed.", error);
}

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
  // Select options are supplied by /api/config, so saved values are restored
  // later in applyConfig().
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
  globeRenderer?.resize();
  window.setTimeout(() => globeRenderer?.resize(), 300);
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
  if (name === "spinupHours" || name === "analysisHours") {
    const days = numeric / 24;
    return `${days.toFixed(Number.isInteger(days) ? 0 : 1)} d`;
  }
  if (name === "jetStrength") return `${numeric.toFixed(0)} m/s`;
  if (name === "perturbationAmplitude") return `${numeric.toFixed(1)} m/s`;
  if (name === "equatorToPoleContrastK") return `${numeric.toFixed(0)} K`;
  if (name === "surfaceDragDays") return `${numeric.toFixed(2)} d`;
  if (name === "seasonalHeatEquatorDeg") {
    if (numeric === 0) return "0°";
    return `${Math.abs(numeric).toFixed(0)}°${numeric > 0 ? "N" : "S"}`;
  }
  if (name === "trail" || name === "flowOpacity" || name === "fieldOpacity") return `${Math.round(numeric * 100)}%`;
  return `${numeric.toFixed(2)}×`;
}

function applyScenarioControls() {
  const baroclinic = form.elements.scenario?.value === "baroclinic";
  document.querySelectorAll("[data-baroclinic-only]").forEach((element) => {
    element.classList.toggle("scenario-hidden", !baroclinic);
  });
  document.querySelectorAll("[data-circulation-only]").forEach((element) => {
    element.classList.toggle("scenario-hidden", baroclinic);
  });
  const levelSelect = form.elements.level;
  if (levelSelect) {
    Array.from(levelSelect.options).forEach((option) => {
      option.disabled = baroclinic && Number(option.value) > 900;
    });
    if (baroclinic && Number(levelSelect.value) > 900) levelSelect.value = "850";
  }
  const duration = form.elements.spinupHours;
  if (duration) {
    duration.min = baroclinic ? "72" : "48";
    duration.max = baroclinic ? "240" : "720";
    duration.step = baroclinic ? "6" : "24";
    const numeric = Number(duration.value);
    if (baroclinic && (numeric < 72 || numeric > 240)) duration.value = "240";
    if (!baroclinic && (numeric < 48 || numeric > 720)) duration.value = "120";
  }
  const spinupLabel = document.querySelector("#spinupLabel");
  if (spinupLabel) spinupLabel.textContent = baroclinic ? "生命周期积分" : "环流建立期";
  const physicsSummary = document.querySelector("#physicsSummary");
  if (physicsSummary) physicsSummary.textContent = baroclinic ? "斜压波验证" : "HS + ETOPO + 三圈闭合";
}

function refreshFormReadout() {
  applyScenarioControls();
  document.querySelectorAll("[data-output]").forEach((output) => {
    const control = form.elements.namedItem(output.dataset.output);
    if (control) output.value = formatOutput(output.dataset.output, control.value);
  });
  const baroclinic = form.elements.scenario?.value === "baroclinic";
  const seasonLabel = baroclinic ? "斜压波验证" : "HS + 地形环流";
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
  document.querySelector(".map-media").style.setProperty("--field-opacity", String(fieldOpacity));
  document.querySelector(".map-media").style.setProperty("--flow-opacity", String(flowOpacity));
  globeRenderer?.setOpacity(fieldOpacity, flowOpacity);
}

function calculateEstimateSeconds() {
  const resolution = Number(form.elements.resolution.value);
  const frames = Number(form.elements.frames.value);
  const particles = Number(form.elements.particles.value);
  const spinupHours = Number(form.elements.spinupHours.value);
  const analysisHours = Number(form.elements.analysisHours.value);
  const scenario = form.elements.scenario?.value || "circulation";
  const defaultsByResolution = scenario === "baroclinic"
    ? (resolution <= 1 ? 18000 : resolution <= 2.5 ? 2400 : 300)
    : (resolution <= 1 ? 24000 : resolution <= 2.5 ? 3000 : 360);
  const configured = state.config?.estimate?.secondsAtDefaults || state.config?.estimateSecondsByResolution || {};
  const configuredBase = Number(configured[String(resolution)] ?? defaultsByResolution);
  const referenceHours = scenario === "baroclinic" ? 240 : 240;
  const experimentHours = scenario === "baroclinic" ? spinupHours : spinupHours + analysisHours;
  const spinupFactor = 0.35 + 0.65 * Math.max(experimentHours, 0) / referenceHours;
  const mediaFactor = 0.45 + 0.55 * (frames / 72) * (particles / 3600);
  return Math.max(1, configuredBase * (0.72 * spinupFactor + 0.28 * mediaFactor));
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
  const hours = Number(form.elements.spinupHours.value);
  if (form.elements.scenario.value === "baroclinic" && (hours < 72 || hours > 240)) {
    if (showMessage) message.textContent = "干温带气旋需要积分 3–10 天。";
    return false;
  }
  if (form.elements.scenario.value === "circulation" && (hours < 48 || hours > 720)) {
    if (showMessage) message.textContent = "三圈环流建立期需设置为 2–30 天。";
    return false;
  }
  const analysisHours = Number(form.elements.analysisHours.value);
  if (form.elements.scenario.value === "circulation" && (analysisHours < 24 || analysisHours > 720)) {
    if (showMessage) message.textContent = "分析 / 播放时段需设置为 1–30 天。";
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

function updateTimelineReadout(frameIndex = 0) {
  if (!state.manifest) return;
  const timeline = Array.isArray(state.manifest.timeline) ? state.manifest.timeline : [];
  const entry = timeline[Math.max(0, Math.min(timeline.length - 1, Number(frameIndex) || 0))];
  if (!entry) {
    document.querySelector("#loopLabel").textContent = `${Number(state.manifest.durationSeconds || state.manifest.frames / state.manifest.fps).toFixed(1)} s loop`;
    document.querySelector("#mediaNote").textContent = `${state.manifest.frames} FRAMES · READY`;
    return;
  }
  const forecastHour = Number(entry.forecastHour || 0);
  const day = forecastHour / 24;
  document.querySelector("#loopLabel").textContent = `D+${day.toFixed(day < 1 ? 2 : 1)} · ${forecastHour.toFixed(0)} h`;
  const pressure = Number(entry.surfacePressureMinHpa);
  const pressureLabel = Number.isFinite(pressure) ? ` · ${pressure.toFixed(0)} hPa` : "";
  document.querySelector("#mediaNote").textContent = `${Number(frameIndex) + 1}/${state.manifest.frames}${pressureLabel}`;
}

function updateMapText() {
  if (!state.manifest) return;
  const layer = state.manifest.layers[state.layer];
  const levelKey = String(state.level);
  const range = layer.ranges?.[levelKey] || [0, 1];
  document.querySelector("#layerEnglish").textContent = layer.english;
  document.querySelector("#layerName").textContent = layer.label;
  document.querySelector("#levelLabel").textContent = `${state.level} hPa`;
  document.querySelector("#legendName").textContent = layer.english;
  document.querySelector("#legendUnit").textContent = layer.unit;
  document.querySelector("#legendMin").textContent = numberLabel(range[0]);
  document.querySelector("#legendMid").textContent = state.layer === "zonalWind"
    ? "0"
    : numberLabel((Number(range[0]) + Number(range[1])) / 2);
  document.querySelector("#legendMax").textContent = numberLabel(range[1]);
  document.querySelector("#legendBar").style.background = `linear-gradient(90deg,${(palettes[state.layer] || palettes.wind).join(",")})`;
  if (!state.rendering) document.querySelector("#renderStatus").textContent = `READY · ${state.manifest.fps} FPS`;
  document.querySelector("#regionLabel").textContent = "WORLD · ORTHOGRAPHIC";
  const frontLegend = document.querySelector("#frontLegend");
  if (frontLegend) {
    frontLegend.hidden = !state.manifest.frontLegend
      || !["fronts", "wind", "temperature", "pressure"].includes(state.layer);
  }
  const zonalLegend = document.querySelector("#zonalLegend");
  if (zonalLegend) zonalLegend.hidden = state.layer !== "zonalWind";
  document.querySelectorAll(".layer").forEach((button) => {
    const available = Boolean(state.manifest.layers[button.dataset.layer]);
    button.disabled = !available;
    button.classList.toggle("active", button.dataset.layer === state.layer);
  });
  localStorage.setItem(LAYER_KEY, state.layer);
  localStorage.setItem(LEVEL_KEY, String(state.level));
  renderLevelLabel();
  updateTimelineReadout(Number(globeCanvas.dataset.frame || 0));
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

function setPlaying(playing) {
  state.playing = playing;
  const playButton = document.querySelector("#playButton");
  globeRenderer?.setPlaying(playing);
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
  try {
    const loads = [loadImage(fieldImage, fieldSource)];
    if (changeFlow) loads.push(loadImage(flowImage, flowSource));
    await Promise.all(loads);
    if (swap !== state.assetSwap) return;
    const timelineFrame = changeFlow ? 0 : Number(globeCanvas.dataset.frame || 0);
    await globeRenderer?.setSources(
      fieldImage,
      flowImage,
      fieldImage.currentSrc || fieldImage.src,
      flowImage.currentSrc || flowImage.src,
      { timelineFrame },
    );
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

function isWorldManifest(manifest) {
  const viewport = manifest.viewport || {};
  const lonMin = Number(viewport.lon_min ?? viewport.lonMin);
  let lonMax = Number(viewport.lon_max ?? viewport.lonMax);
  if (Number.isFinite(lonMin) && Number.isFinite(lonMax) && lonMax <= lonMin) lonMax += 360;
  const lonSpan = lonMax - lonMin;
  const regionValues = [
    viewport.key,
    viewport.name,
    manifest.settings?.region,
    typeof manifest.region === "string" ? manifest.region : manifest.region?.key,
  ].filter(Boolean).map((value) => String(value).toLowerCase());
  return lonSpan >= 359.5 || regionValues.some((value) => ["world", "global"].includes(value));
}

async function activateManifest(manifestUrl, autoPlay = false) {
  const response = await fetch(manifestUrl, { cache: "no-store" });
  if (!response.ok) throw new Error(`清单请求失败（HTTP ${response.status}）`);
  const manifest = await response.json();
  if (!manifest.layers || !manifest.particles) throw new Error("生成结果缺少图层或流线素材");
  if (manifest.settings?.scenario === "circulation"
      && (Number(manifest.schemaVersion || 0) < 5
        || manifest.experiment?.orographicLift !== true)) {
    throw new Error("这是旧的平坦水球结果；请载入或重新计算带 ETOPO 地形的新结果");
  }
  if (manifest.qualityGatePassed === false) {
    throw new Error("该结果未通过三风带或数值稳定性门禁，已拒绝播放");
  }
  if (!isWorldManifest(manifest)) throw new Error("旧的区域渲染不适用于全球球面，请重新计算全球结果");
  state.manifest = manifest;
  state.manifestUrl = new URL(manifestUrl, window.location.href).href;
  globeRenderer?.setManifest(manifest);
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

function normalizeScenarioOptions() {
  const select = form.elements.scenario;
  if (!select) return;
  const labels = {
    circulation: "全球地形环流 · HS + ETOPO + 闭合",
    baroclinic: "斜压波验证 · 干动力核",
  };
  for (const option of select.options) {
    if (labels[option.value]) option.textContent = labels[option.value];
  }
  const circulation = [...select.options].find((option) => option.value === "circulation");
  if (circulation) select.insertBefore(circulation, select.firstElementChild);
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
  replaceOptions("scenario", config.scenarios || options.scenarios || options.scenario || definitions.scenario?.options);
  normalizeScenarioOptions();
  replaceOptions("resolution", config.resolutions || options.resolutions || options.resolution || definitions.resolution?.options, "°");
  replaceOptions("level", config.levels || options.levels || options.level || definitions.level?.options, " hPa");
  replaceOptions("season", config.seasons || options.seasons || options.season || definitions.season?.options);
  replaceOptions("hemisphere", config.hemispheres || options.hemispheres || options.hemisphere || definitions.hemisphere?.options);
  for (const name of persistedFields) {
    applyRangeConfig(name, config.ranges?.[name] || definitions[name] || config.limits?.[name]);
  }
  const defaults = { ...(config.defaults || config.defaultConfig || {}) };
  for (const [name, definition] of Object.entries(definitions)) {
    if (defaults[name] === undefined && definition?.default !== undefined) defaults[name] = definition.default;
  }
  Object.assign(defaults, UI_DEFAULTS);
  const savedValues = readSettings().values || {};
  assignAllowedValue(form.elements.scenario, savedValues.scenario ?? defaults.scenario);
  applyScenarioControls();
  for (const [name, value] of Object.entries(defaults)) {
    const control = form.elements.namedItem(name);
    if (name !== "scenario" && control && savedValues[name] === undefined) assignAllowedValue(control, value);
  }
  for (const [name, value] of Object.entries(savedValues)) {
    const control = form.elements.namedItem(name);
    if (name !== "scenario" && control && !assignAllowedValue(control, value) && defaults[name] !== undefined) assignAllowedValue(control, defaults[name]);
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
  const payload = { region: "world" };
  const scenario = form.elements.scenario.value;
  if (scenario === "circulation") {
    const heatEquator = Number(form.elements.seasonalHeatEquatorDeg.value);
    const compatibleSeason = heatEquator > 2 ? "summer" : heatEquator < -2 ? "winter" : "equinox";
    assignAllowedValue(form.elements.season, compatibleSeason);
  }
  const circulationOnly = new Set([
    "analysisHours", "equatorToPoleContrastK", "surfaceDragDays", "seasonalHeatEquatorDeg",
    "tibetScale", "landHeatingScale", "oceanCurrentScale",
  ]);
  const baroclinicOnly = new Set(["jetStrength", "perturbationAmplitude", "hemisphere"]);
  for (const name of postFields) {
    if (scenario === "baroclinic" && circulationOnly.has(name)) continue;
    if (scenario === "circulation" && baroclinicOnly.has(name)) continue;
    const control = form.elements.namedItem(name);
    if (!control) continue;
    const value = control.value;
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
globeCanvas.addEventListener("atmos-frame", (event) => {
  updateTimelineReadout(event.detail?.frame || 0);
});

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
