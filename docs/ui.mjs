/**
 * UI module for BOxCrete interactive demo.
 * Handles sliders, canvas rendering (scatter + strength curve),
 * and interaction between views.
 */

import { predictStrengthCurve, predictStrengthMeanOnly, predictGWP, predictCost, initStrengthModel, initWASM } from "./gp.mjs";
import {
  UNITS,
  compToDisplay,
  compFromDisplay,
  sliderUnitLabel as sliderUnitLabelFor,
} from "./units.mjs";

// --- Shared Helpers ---
function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/**
 * Resolve a column name to an index in the schema, throwing loudly if
 * the name isn't present. Replaces the previous ``cols.indexOf(name) ||
 * 0`` pattern, which silently fell back to column 0 if the name didn't
 * match (e.g., after a Python-side rename of ``"Cement (kg/m3)"`` to
 * ``"Cement"`` or a Unicode ``"Cement (kg/m³)"``).
 */
function colIdx(cols, name) {
  const i = cols.indexOf(name);
  if (i < 0) {
    throw new Error(
      `[boxcrete] column ${JSON.stringify(name)} is not in the compositions ` +
      `schema. Known columns: ${JSON.stringify(cols)}. ` +
      "If column names changed in Python, update both ``DEFAULT_X_COLUMNS`` " +
      "(``boxcrete/utils.py``) AND the JS-side references in docs/ui.mjs."
    );
  }
  return i;
}

// Generate `nPts` log-spaced curing times in [0, 28] days. Denser at early
// times where strength changes fastest — inverse of `log10(t+1)/log10(29)`.
// Used by `drawStrengthCurve` (32 pts interactive / 64 pts idle), the
// Material Source curve transition (64 pts), and the always-on preview
// curve (PREVIEW_PTS pts).
function logSpacedTimes(nPts) {
  return Array.from({ length: nPts }, (_, i) => {
    const t01 = i / (nPts - 1);
    return Math.pow(29, t01) - 1;
  });
}

// Convert GP variances to standard deviations including model noise.
// `variances` is the array returned by `predictStrengthCurve`.
//
// For schema-v2 models with a heteroscedastic gated likelihood, the
// noise variance returned by predictStrengthCurve already incorporates
// h(t)² scaling and is summed into `variances` (advertised by
// `variance_includes_aleatoric: true`). For legacy v1 models or
// schema-v2 with global noise, we add `noise_variance * y_std²` here.
function computeStds(variances, params) {
  if (params.variance_includes_aleatoric) {
    return variances.map((v) => Math.sqrt(Math.max(0, v)));
  }
  const noiseVar = params.noise_variance * params.y_std * params.y_std;
  return variances.map((v) => Math.sqrt(v + noiseVar));
}

// --- Cached DOM Elements & Indices ---
let _sliderInputs = null; // cached after buildSliders()
let COL_MS = -1; // "Material Source" column index
let COL_TEMP = -1; // "Temp (C)" column index

// --- State ---
let strengthParams = null;
let gwpParams = null;
let costParams = null;
let compositionsData = null;
let currentComposition = null; // current slider values (without time)
let scatterDay = 28;
let scatterXAxis = "gwp"; // "gwp" or "cost"
let curveObsPositions = []; // [{px, py, time, strength}] for tooltip hit-testing
let animationId = null; // for smooth transitions
// Most recent animation TARGET (intended end state). When the user commits a
// click-to-edit value during an in-flight animation, we build the new target
// from this — not from mid-lerp `currentComposition` — so other still-animating
// sliders land at their intended positions.
let _lastAnimTarget = null;
let scatterFilter = null; // [{colIdx, min, max}] array or null
let mixAnalyses = null; // pre-computed mix descriptions
let animLoopId = null; // unified animation loop frame ID
let lastFrameTime = 0; // for frame-rate-independent interpolation
let scatterTransition = null; // {startTime, duration, fromX, fromY, toX, toY, fromPareto, toPareto}
let _curveYMax = null; // smoothly interpolated y-axis max for strength curve
let _curveYMaxTarget = null; // target y-max (for animation loop convergence check)
// Material Source curve transition: when the user toggles Material Source, we
// snapshot the pre-toggle strength curve and blend it linearly with the
// post-toggle curve over `duration` ms. Because Material Source is binary,
// composition-level interpolation would feed the GP non-categorical values
// and yield a noisy intermediate prediction. Curve-level interpolation keeps
// the visual aesthetic smooth without violating the GP's input domain.
let _msCurveTransition = null; // {startTime, duration, times, fromMeans, fromStds}

// --- Unit System ---
let unitSystem = "metric"; // "metric" or "imperial"
// `UNITS` is imported from `./units.mjs` (single source of truth, also used
// by the Node-based `test/test_js_units.mjs` parity tests).
function U() { return UNITS[unitSystem]; }

// Animated unit transition
let unitTransition = null; // {startTime, duration, from, to}

function getDisplayFactors() {
  if (!unitTransition) return U();
  const elapsed = performance.now() - unitTransition.startTime;
  const t = Math.min(elapsed / unitTransition.duration, 1);
  const ease = easeInOutCubic(t);
  const from = unitTransition.from;
  const to = unitTransition.to;
  if (t >= 1) { unitTransition = null; return to; }
  return {
    strength: to.strength,
    strengthFactor: from.strengthFactor + (to.strengthFactor - from.strengthFactor) * ease,
    mass: to.mass,
    massFactor: from.massFactor + (to.massFactor - from.massFactor) * ease,
    gwp: to.gwp,
    gwpFactor: from.gwpFactor + (to.gwpFactor - from.gwpFactor) * ease,
    cost: to.cost,
    costFactor: from.costFactor + (to.costFactor - from.costFactor) * ease,
  };
}

// Listen for unit toggle
document.addEventListener("toggle-units", () => {
  // Blur any in-progress click-to-edit before the unit transition. Otherwise,
  // a number typed in (e.g.) kg/m³ would be interpreted in lb/yd³ on commit.
  // Blur fires the input's blur listener, which commits the edit in the
  // pre-toggle unit context.
  const active = document.activeElement;
  if (active && active.classList && active.classList.contains("slider-value")) {
    active.blur();
  }
  const oldFactors = { ...U() };
  unitSystem = unitSystem === "metric" ? "imperial" : "metric";
  const newFactors = { ...U() };
  unitTransition = { startTime: performance.now(), duration: 350, from: oldFactors, to: newFactors };
  document.getElementById("unit-label").textContent = unitSystem === "metric" ? "SI" : "US";
  const mobileUnitLabel = document.getElementById("mobile-unit-label");
  if (mobileUnitLabel) mobileUnitLabel.textContent = unitSystem === "metric" ? "SI" : "US";
  // Update composition toggle button label with current unit
  const mobileSlidersBtn = document.getElementById("mobile-show-sliders");
  if (mobileSlidersBtn) {
    mobileSlidersBtn.textContent = unitSystem === "metric" ? "Composition (kg/m³)" : "Composition (lb/yd³)";
  }
  document.getElementById("gwp-unit").textContent = U().gwp;
  document.getElementById("cost-unit").textContent = U().cost;
  document.getElementById("sliders-title").textContent =
    unitSystem === "metric" ? "Composition (kg/m³)" : "Composition (lb/yd³)";
  updateSliderLabels();
  startAnimLoop();
});

// --- Load model data ---
async function loadJSON(path) {
  const resp = await fetch(path);
  return resp.json();
}

async function init() {
  [strengthParams, gwpParams, costParams, compositionsData] = await Promise.all([
    loadJSON("model/strength.json"),
    loadJSON("model/gwp.json"),
    loadJSON("model/cost.json"),
    loadJSON("model/compositions.json"),
  ]);

  // Load mix analyses (non-blocking, optional). Warn loudly on failure
  // rather than swallowing silently — a missing or malformed
  // mix_analyses.json should be visible in the console so it can be
  // diagnosed during development.
  loadJSON("model/mix_analyses.json")
    .then(d => { mixAnalyses = d; updateMixInsight(); })
    .catch(err => {
      console.warn("[boxcrete] mix_analyses.json failed to load; mix-insight panel will be empty:", err);
    });

  // Compute Cholesky and alpha from training data + kernel params
  initStrengthModel(strengthParams);

  // Initialize WASM BLAS for accelerated variance (non-blocking, falls back to JS)
  initWASM(strengthParams);

  buildSliders();
  _sliderInputs = document.querySelectorAll("#sliders input[type=range]");
  COL_MS = compositionsData.column_names.indexOf("Material Source");
  COL_TEMP = compositionsData.column_names.indexOf("Temp (C)");
  setupEventListeners();
  update();
  startAnimLoop();
  updateMixInsight(); // initial insight for default composition
}

// --- Sliders ---
// Ingredient descriptions — shown when clicking the ingredient name
const ingredientInfo = {
  "Cement": "Portland cement (OPC) is the primary binder in concrete. Hydration of its clinker minerals (C₃S, C₂S, C₃A, C₄AF) produces calcium silicate hydrate (C-S-H) gel, which gives concrete its strength. High early strength contribution but the most carbon-intensive ingredient — producing 1 tonne of cement releases ~0.6–0.9 tonnes of CO₂ from calcination and kiln fuel.",
  "Fly Ash": "A pozzolanic byproduct of coal combustion. Glassy silica spheres react slowly with calcium hydroxide from cement hydration to form additional C-S-H gel. Improves long-term strength and durability, reduces permeability, and has near-zero embodied carbon (it's a waste product). The spherical particles also improve workability (ball-bearing effect). Slower early strength gain than cement.",
  "Slag": "Ground granulated blast furnace slag (GGBFS) — a latent hydraulic byproduct of iron production. Activated by the alkaline environment from cement hydration, it produces C-S-H gel independently. Excellent late-age strength development, lower heat of hydration (reducing thermal cracking risk), and significantly lower GWP than cement. Can replace 30–70% of cement in typical mixes.",
  "Water": "Controls the water-to-binder (W/B) ratio, the single most important factor for concrete strength and durability. Lower W/B produces a denser, stronger, more durable matrix with less capillary porosity — but reduces workability. Superplasticizers (HRWR) allow low W/B while maintaining flowability.",
  "HRWR": "High-range water reducer (superplasticizer). A chemical admixture that disperses cement particles via electrostatic or steric repulsion, dramatically improving flowability without adding water. Enables ultra-low W/B ratios (0.20–0.25) that would otherwise be unworkable. Essential for high-performance concrete.",
  "Fine Aggregate": "Sand — provides bulk volume, dimensional stability, and load transfer in the morite matrix. Particle size distribution (gradation) affects packing density and paste demand. Typically river sand or manufactured sand from crushed rock.",
  "Coarse Aggregates": "Gravel or crushed stone (>4.75 mm) — forms the structural skeleton of concrete. The interfacial transition zone (ITZ) between paste and aggregate is often the weakest link. Well-graded aggregates improve packing and reduce paste demand. Typically 60–75% of concrete by volume.",
  "Material Source": "Identifies the source of raw materials. Different sources have varying mineral compositions, particle size distributions, and reactivity — all of which affect strength development, workability, and durability. Source-specific models account for this variability.",
  "Temperature": "Curing temperature significantly affects hydration kinetics. Higher temperatures accelerate early hydration (faster early strength) but can reduce ultimate strength due to non-uniform hydrate distribution. Low temperatures slow hydration but can improve long-term microstructure. The Arrhenius-based maturity concept links time and temperature to strength development.",
};

function buildSliders() {
  const container = document.getElementById("sliders");
  const bounds = compositionsData.slider_bounds;
  const colNames = compositionsData.column_names;

  // Use median composition as initial values
  const compositions = compositionsData.compositions;
  const n = compositions.length;
  const medianIdx = Math.floor(n / 2);
  currentComposition = [...compositions[medianIdx]];
  displayPreviewComp = [...currentComposition];

  for (let i = 0; i < colNames.length; i++) {
    const col = colNames[i];

    // Material Source gets a toggle instead of a slider
    if (col === "Material Source") {
      const group = document.createElement("div");
      // `material-source-group` lets mobile CSS hide the redundant value-span
      // and span the toggle-row across cols 2–3 without a `:has()` selector
      // (Safari < 15.4 still hits this page).
      group.className = "slider-group material-source-group";

      const label = document.createElement("label");
      const nameSpan = document.createElement("span");
      nameSpan.textContent = "Material Source";
      nameSpan.className = "ingredient-name";
      nameSpan.addEventListener("click", (e) => {
        e.preventDefault();
        toggleIngredientInfo(group, "Material Source");
      });
      const valueSpan = document.createElement("span");
      valueSpan.id = `val-${i}`;
      valueSpan.textContent = currentComposition[i] === 0 ? "Source A" : "Source B";
      label.append(nameSpan, valueSpan);

      const toggle = document.createElement("div");
      toggle.className = "toggle-row";
      const btn0 = document.createElement("button");
      btn0.textContent = "Source A";
      btn0.className = currentComposition[i] === 0 ? "toggle-btn active" : "toggle-btn";
      btn0.addEventListener("click", () => {
        // Smooth curve-level transition (see `triggerMaterialSourceTransition`).
        // Updates `currentComposition[i]` and `displayPreviewComp[i]` internally.
        triggerMaterialSourceTransition(i, 0);
        btn0.className = "toggle-btn active";
        btn1.className = "toggle-btn";
        document.getElementById(`val-${i}`).textContent = "Source A";
        update();
        // Refresh mix insight: the new composition (median + other MS) is
        // typically NOT in the training set, so the previous mix's description
        // would otherwise persist stale. Schedule with the same delay used by
        // `animateToComposition` so the insight settles after the curve does.
        scheduleInsightUpdate();
        checkExtrapolationWarning();
      });
      const btn1 = document.createElement("button");
      btn1.textContent = "Source B";
      btn1.className = currentComposition[i] === 1 ? "toggle-btn active" : "toggle-btn";
      btn1.addEventListener("click", () => {
        triggerMaterialSourceTransition(i, 1);
        btn0.className = "toggle-btn";
        btn1.className = "toggle-btn active";
        document.getElementById(`val-${i}`).textContent = "Source B";
        update();
        scheduleInsightUpdate();
        checkExtrapolationWarning();
      });
      toggle.append(btn0, btn1);

      group.append(label, toggle);
      container.appendChild(group);
      continue;
    }

    const b = bounds[col];
    if (b.min === b.max) continue; // skip zero-range sliders

    const group = document.createElement("div");
    group.className = "slider-group";

    const label = document.createElement("label");
    const nameSpan = document.createElement("span");
    // Display name: strip unit suffix and rename "Temp" → "Temperature" for
    // a friendlier label. The underlying column name in `compositionsData`
    // is unchanged (still "Temp (C)") so model code keeps working.
    let shortName = col.replace(" (kg/m3)", "").replace(" (C)", "");
    if (shortName === "Temp") shortName = "Temperature";
    nameSpan.textContent = shortName;
    // Make ingredient names clickable for info
    const infoKey = shortName;
    if (ingredientInfo[infoKey]) {
      nameSpan.className = "ingredient-name";
      nameSpan.addEventListener("click", (e) => {
        e.preventDefault();
        toggleIngredientInfo(group, infoKey);
      });
    }
    const valueInput = document.createElement("input");
    valueInput.id = `val-${i}`;
    valueInput.className = "slider-value";
    valueInput.type = "text";
    // `decimal` is safe here: all composition columns have b.min >= 0, so no
    // negative values are ever entered (no need for `-` key on iOS Safari).
    valueInput.inputMode = "decimal";
    valueInput.setAttribute("aria-label", `${shortName} value`);
    valueInput.value = displayCompValue(col, currentComposition[i]).toFixed(1);
    valueInput.dataset.idx = i;
    valueInput.dataset.col = col;
    attachValueEditHandlers(valueInput, i, col, b);

    // Per-row unit suffix (kg/m³ ↔ lb/yd³ on toggle; °C for Temperature).
    // Wrapped in a flex container so the label keeps its two-child
    // `space-between` layout (name on the left, value+unit packed on the right).
    const valueWrap = document.createElement("span");
    valueWrap.className = "slider-value-wrap";
    const unitSpan = document.createElement("span");
    unitSpan.className = "slider-unit";
    unitSpan.id = `unit-${i}`;
    unitSpan.textContent = sliderUnitLabel(col);
    valueWrap.append(valueInput, unitSpan);
    label.append(nameSpan, valueWrap);

    const input = document.createElement("input");
    input.type = "range";
    input.min = b.min;
    input.max = b.max;
    input.step = (b.max - b.min) / 200;
    input.value = currentComposition[i];
    input.dataset.idx = i;
    input.dataset.col = col;
    input.addEventListener("input", onSliderChange);

    const infoRow = document.createElement("div");
    infoRow.className = "info-row";
    infoRow.innerHTML = `<span>${b.min.toFixed(0)}</span><span>${b.max.toFixed(0)}</span>`;

    group.append(label, input, infoRow);
    container.appendChild(group);
  }
}

// Show ingredient info in the dedicated panel
let _activeIngredientKey = null;

function animateContentSwap(bodyEl, textEl, newHTML) {
  const prevHeight = bodyEl.offsetHeight;
  textEl.classList.add("fade-out");
  setTimeout(() => {
    textEl.innerHTML = newHTML;
    bodyEl.style.height = "auto";
    const newHeight = bodyEl.offsetHeight;
    bodyEl.style.height = prevHeight + "px";
    requestAnimationFrame(() => {
      bodyEl.style.height = newHeight + "px";
    });
    textEl.classList.remove("fade-out");
    textEl.classList.add("fade-in");
    requestAnimationFrame(() => textEl.classList.remove("fade-in"));
    setTimeout(() => { bodyEl.style.height = "auto"; }, 300);
  }, 300);
}

function toggleIngredientInfo(group, key) {
  const textEl = document.getElementById("ingredient-insight-text");
  const bodyEl = document.querySelector(".ingredient-insight-body");

  // Toggle off if same ingredient clicked again
  if (_activeIngredientKey === key) {
    animateContentSwap(bodyEl, textEl, '<span class="mix-insight-placeholder">Click an ingredient name in the Composition panel to learn more.</span>');
    _activeIngredientKey = null;
    for (const el of document.querySelectorAll(".ingredient-name.active")) {
      el.classList.remove("active");
    }
    return;
  }

  // Update active highlight
  for (const el of document.querySelectorAll(".ingredient-name.active")) {
    el.classList.remove("active");
  }
  const nameSpan = group.querySelector(".ingredient-name");
  if (nameSpan) nameSpan.classList.add("active");

  // FLIP: measure current height, crossfade content, animate to new height
  animateContentSwap(bodyEl, textEl, `<strong>${key}</strong> — ${ingredientInfo[key]}`);

  _activeIngredientKey = key;
}

// --- Slider Preview (hover composition preview) ---
// Update both the range input position and the editable value display.
// The value display is an <input> for regular sliders (click-to-edit) and a
// <span> for the Material Source row. We must use `.value` for inputs and
// `.textContent` for spans, and we must NOT clobber an in-progress edit
// (the focused element).
function syncSliderDOM(comp, updateValues = true) {
  if (!_sliderInputs) return;
  for (const slider of _sliderInputs) {
    const idx = parseInt(slider.dataset.idx);
    if (updateValues) slider.value = comp[idx];
    setValueDisplay(idx, displayCompValue(slider.dataset.col, comp[idx]).toFixed(1));
  }
}

function setValueDisplay(idx, formatted) {
  const el = document.getElementById(`val-${idx}`);
  if (!el) return;
  // Don't clobber an in-progress click-to-edit. The blur/Enter handlers will
  // refresh the display once the edit commits or reverts.
  if (el === document.activeElement) return;
  if (el.tagName === "INPUT") {
    el.value = formatted;
  } else {
    el.textContent = formatted;
  }
}

function showSliderPreview(comp) {
  if (!_sliderInputs) return;
  for (const slider of _sliderInputs) {
    const idx = parseInt(slider.dataset.idx);
    const val = comp[idx];
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    const fraction = (val - min) / (max - min);

    // Get or create preview marker
    let marker = slider.parentElement.querySelector(".slider-preview-marker");
    if (!marker) {
      marker = document.createElement("div");
      marker.className = "slider-preview-marker";
      slider.parentElement.insertBefore(marker, slider.nextSibling);
    }
    // Account for range input thumb inset (thumb center at min is `thumbHalf`
    // px from each edge of the input box). The half-width is exposed as a
    // CSS variable on `.slider-group` so the desktop (9 px) and mobile
    // (8 px, smaller thumb) values stay in sync with the actual rendered
    // thumb size — falls back to 9 if the variable isn't set.
    // `slider.offsetLeft` is 0 on desktop (the slider is a full-width child of
    // `.slider-group`), but on mobile the slider lives in column 2 of a CSS grid
    // so we must include its offset within the positioning parent.
    const thumbHalfRaw = getComputedStyle(slider.parentElement).getPropertyValue("--thumb-half");
    const thumbHalf = parseFloat(thumbHalfRaw) || 9;
    const trackWidth = slider.offsetWidth - 2 * thumbHalf;
    const leftPx = slider.offsetLeft + thumbHalf + fraction * trackWidth;
    marker.style.left = leftPx + "px";
    // Align vertically with the slider thumb center
    marker.style.top = `${slider.offsetTop + slider.offsetHeight / 2}px`;
    marker.style.display = "block";
  }

  const panel = document.getElementById("sliders-panel");
  panel.classList.add("previewing");
}

function hideSliderPreview() {
  const markers = document.querySelectorAll(".slider-preview-marker");
  for (const m of markers) m.style.display = "none";
  const panel = document.getElementById("sliders-panel");
  panel.classList.remove("previewing");
}

let _sliderActive = false;
let _sliderIdleTimer = null;

function onSliderChange(e) {
  const idx = parseInt(e.target.dataset.idx);
  currentComposition[idx] = parseFloat(e.target.value);
  displayPreviewComp[idx] = currentComposition[idx];
  const displayVal = displayCompValue(e.target.dataset.col, currentComposition[idx]);
  setValueDisplay(idx, displayVal.toFixed(1));
  _sliderActive = true;
  if (_sliderIdleTimer) clearTimeout(_sliderIdleTimer);
  _sliderIdleTimer = setTimeout(() => { _sliderActive = false; update(); }, 150);
  update();
  startAnimLoop(); // keep loop alive for smooth y-axis expansion
  updateMixInsight(); // immediate for manual slider adjustments
  checkExtrapolationWarning();
}

// Display value for a composition column under the active unit system.
// Handles both mass (factor) and temperature (factor + offset).
function displayCompValue(colName, internal) {
  return compToDisplay(colName, internal, unitSystem);
}
// Inverse of `displayCompValue`: parse a user-typed display value back to
// the model-native (kg/m³ or °C) value before clamping/storage.
function internalCompValue(colName, display) {
  return compFromDisplay(colName, display, unitSystem);
}
// Unit suffix label for a slider column (delegates to `units.mjs`).
function sliderUnitLabel(colName) {
  return sliderUnitLabelFor(colName, unitSystem);
}
function updateSliderLabels() {
  syncSliderDOM(currentComposition, false);
  // Update info rows (min/max labels) and per-row unit suffixes
  const bounds = compositionsData.slider_bounds;
  const colNames = compositionsData.column_names;
  const infoRows = document.querySelectorAll("#sliders .info-row");
  let rowIdx = 0;
  for (let i = 0; i < colNames.length; i++) {
    const col = colNames[i];
    if (col === "Material Source") continue;
    const b = bounds[col];
    if (b.min === b.max) continue;
    if (rowIdx < infoRows.length) {
      // Use offset-aware converter so temperature bounds render correctly
      // in °F (e.g. -20°C → -4°F, 22°C → 72°F) under imperial.
      const minDisp = displayCompValue(col, b.min).toFixed(0);
      const maxDisp = displayCompValue(col, b.max).toFixed(0);
      infoRows[rowIdx].innerHTML = `<span>${minDisp}</span><span>${maxDisp}</span>`;
      rowIdx++;
    }
    // Refresh per-row unit suffix (kg/m³ ↔ lb/yd³, °C ↔ °F)
    const unitEl = document.getElementById(`unit-${i}`);
    if (unitEl) unitEl.textContent = sliderUnitLabel(col);
  }
}

// --- Animated transition to a new composition ---
function animateToComposition(targetComp) {
  if (animationId) cancelAnimationFrame(animationId);
  // A scatter-click animation supersedes any in-flight Material Source curve
  // transition: the new composition takes over and we recompute the curve
  // from the lerped composition each frame.
  _msCurveTransition = null;
  hideExtrapolationWarning(); // suppress during transition
  startAnimLoop();

  // Track the *intended* end state so a click-to-edit during this animation
  // can build the new target from un-clobbered values. Cleared on completion.
  _lastAnimTarget = [...targetComp];

  const startComp = [...currentComposition];
  const duration = 350; // ms
  const startTime = performance.now();

  function step(now) {
    const t = Math.min((now - startTime) / duration, 1);
    // Smooth easing (ease-in-out: starts at zero velocity, ends at zero velocity)
    const ease = easeInOutCubic(t);

    // Lerp each dimension
    for (let i = 0; i < startComp.length; i++) {
      currentComposition[i] = startComp[i] + (targetComp[i] - startComp[i]) * ease;
    }

    syncSliderDOM(currentComposition);

    update();

    if (t < 1) {
      animationId = requestAnimationFrame(step);
    } else {
      animationId = null;
      _lastAnimTarget = null;
      // Snap to exact target and update toggle
      setComposition(targetComp);
      // Sequenced: update insight after the curve has settled
      scheduleInsightUpdate();
      checkExtrapolationWarning();
    }
  }

  animationId = requestAnimationFrame(step);
}

// --- Material Source curve-level transition ---
// Material Source is a binary categorical input; feeding the GP fractional
// values (0.5) gives a noisy intermediate prediction outside the training
// distribution. Instead, snapshot the pre-toggle posterior curve, commit the
// new MS value to `currentComposition`, and let `drawStrengthCurve` blend
// the cached `from` curve with each frame's freshly computed `to` curve over
// `MS_TRANSITION_MS`. The result is a smooth visual that respects the GP's
// input domain.
const MS_TRANSITION_MS = 350;
function triggerMaterialSourceTransition(idx, newVal) {
  if (!strengthParams) {
    // Predictor not yet initialized — fall back to instant commit so the UI
    // still responds. (Should not happen in practice; init() awaits params.)
    currentComposition[idx] = newVal;
    displayPreviewComp[idx] = newVal;
    return;
  }
  if (currentComposition[idx] === newVal) return; // no-op

  // Snapshot pre-toggle curve at fixed 64-point log-spaced times. Same time
  // grid is reused throughout the blend so per-frame work is just a lerp.
  const times = logSpacedTimes(64);
  const { means: fromMeans, variances: fromVars } = predictStrengthCurve(
    currentComposition, times, strengthParams
  );
  const fromStds = computeStds(fromVars, strengthParams);

  // Commit the new MS value before kicking off the visual blend so the GP
  // calls during the transition use the post-toggle composition.
  currentComposition[idx] = newVal;
  displayPreviewComp[idx] = newVal;

  _msCurveTransition = {
    startTime: performance.now(),
    duration: MS_TRANSITION_MS,
    times,
    fromMeans,
    fromStds,
  };
  startAnimLoop();
}

// --- Click-to-edit value handlers (regular sliders only) ---
// The plan: focus selects all; Enter commits; Escape reverts; blur commits.
// On commit, parse the displayed (unit-aware) number, divide by the column's
// display factor, clamp to [b.min, b.max], and animate to the new state.
function attachValueEditHandlers(inputEl, idx, col, b) {
  function commit() {
    const raw = inputEl.value.trim();
    const parsed = parseFloat(raw);
    if (!Number.isFinite(parsed)) {
      // Non-numeric → revert displayed text
      inputEl.value = displayCompValue(col, currentComposition[idx]).toFixed(1);
      return;
    }
    // Convert displayed value back to internal units, then clamp.
    // For Temperature this also handles the °F → °C offset.
    const internal = internalCompValue(col, parsed);
    const clamped = Math.max(b.min, Math.min(b.max, internal));
    // Build target from the most recent intended end state to avoid landing
    // mid-animation values for sliders that are currently in flight.
    const base = animationId !== null && _lastAnimTarget !== null
      ? [..._lastAnimTarget]
      : [...currentComposition];
    base[idx] = clamped;
    animateToComposition(base);
    // Refresh display in case clamping or rounding changed it (the animation
    // will overwrite, but we want the input to read correctly during the lerp
    // since `setValueDisplay` skips focused elements — and this element is
    // still focused if commit was triggered by Enter).
    inputEl.value = displayCompValue(col, clamped).toFixed(1);
  }
  inputEl.addEventListener("focus", () => inputEl.select());
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commit();
      inputEl.blur();
    } else if (e.key === "Escape") {
      e.preventDefault();
      // Revert without committing
      inputEl.value = displayCompValue(col, currentComposition[idx]).toFixed(1);
      inputEl.blur();
    }
  });
  inputEl.addEventListener("blur", () => {
    // Blur commits the edit (same as Enter), but only if the value was changed.
    // If the value matches the current displayed state, do nothing.
    const expected = displayCompValue(col, currentComposition[idx]).toFixed(1);
    if (inputEl.value.trim() !== expected) commit();
  });
}

// --- Set sliders from a composition (instant) ---
function setComposition(comp) {
  currentComposition = [...comp];
  displayPreviewComp = [...comp];
  syncSliderDOM(comp);
  // Update Material Source toggle (val-${msIdx} is a <span>, not an <input>,
  // so we write textContent directly. syncSliderDOM only iterates range
  // sliders, which doesn't include Material Source.)
  const msIdx = COL_MS;
  if (msIdx >= 0) {
    const msVal = Math.round(comp[msIdx]);
    const msEl = document.getElementById(`val-${msIdx}`);
    if (msEl) msEl.textContent = msVal === 0 ? "Source A" : "Source B";
    const buttons = document.querySelectorAll(".toggle-btn");
    if (buttons.length >= 2) {
      buttons[0].className = msVal === 0 ? "toggle-btn active" : "toggle-btn";
      buttons[1].className = msVal === 1 ? "toggle-btn active" : "toggle-btn";
    }
  }
  update();
}

// --- Update everything ---
function update() {
  updateReadouts();
  drawStrengthCurve();
  drawScatter();
  // Mix insight updates are triggered separately with delay (see animateToComposition)
}

// Delayed mix insight update — called after strength curve animation settles
function scheduleInsightUpdate() {
  setTimeout(updateMixInsight, 300); // delay after curve settles for sequenced feel
}

let _currentInsightIdx = null; // track which mix is currently displayed
const _placeholderHTML = '<span class="mix-insight-placeholder">Click a data point to see mix analysis.</span>';

function updateMixInsight() {
  const textEl = document.getElementById("mix-insight-text");
  const bodyEl = document.querySelector(".mix-insight-body");
  const paretoPill = document.getElementById("pareto-pill");
  if (!mixAnalyses) return;

  const nearIdx = findNearestCompositionIdx(currentComposition);

  // Update Pareto pill independently (instant, no content swap needed)
  if (nearIdx !== null) {
    const paretoTag = getParetoTag(nearIdx);
    if (paretoTag) {
      paretoPill.textContent = paretoTag;
      paretoPill.classList.add("visible");
    } else {
      paretoPill.classList.remove("visible");
    }
  } else {
    paretoPill.classList.remove("visible");
  }

  if (nearIdx !== null && mixAnalyses[String(nearIdx)]) {
    if (_currentInsightIdx === nearIdx) return;

    animateContentSwap(bodyEl, textEl, buildInsightHTML(nearIdx));
    _currentInsightIdx = nearIdx;
  } else if (nearIdx !== null) {
    if (_currentInsightIdx !== nearIdx) {
      animateContentSwap(bodyEl, textEl, '<span class="mix-insight-placeholder">Mix insight not available for this composition.</span>');
      _currentInsightIdx = nearIdx;
    }
  } else {
    if (_currentInsightIdx !== null) {
      animateContentSwap(bodyEl, textEl, _placeholderHTML);
      _currentInsightIdx = null;
    }
  }
}

function buildInsightHTML(idx) {
  let desc = mixAnalyses[String(idx)];
  desc = desc.replace(/\*\*(.+?)\*\*/g, (_, text) => `<strong>${text}</strong>`);
  return desc;
}

// Dynamically compute Pareto label for the current scatter objectives
function getParetoTag(idx) {
  if (!compositionsData) return null;
  const gwpPreds = compositionsData.gwp_predictions;
  const costPreds = compositionsData.cost_predictions;
  const strDay = String(scatterDay);
  const strPreds = compositionsData.strength_predictions[strDay];
  if (!strPreds) return null;

  const n = gwpPreds.length;

  // Check if idx is Pareto-optimal for current x-axis vs strength
  const xVals = scatterXAxis === "cost"
    ? costPreds.map(v => -v)
    : gwpPreds.map(v => -v);
  const yVals = strPreds;

  // Is idx dominated by any other point?
  const xi = xVals[idx], yi = yVals[idx];
  for (let j = 0; j < n; j++) {
    if (j === idx) continue;
    if (xVals[j] <= xi && yVals[j] >= yi && (xVals[j] < xi || yVals[j] > yi)) {
      return null; // dominated
    }
  }
  return "Pareto-optimal";
}

function updateReadouts() {
  const u = U();
  // GWP — use fixed Temp=22°C since GWP is a material property, not temperature-dependent
  const msIdx = COL_MS;
  const tempIdx = COL_TEMP;
  const ms = msIdx >= 0 ? Math.round(currentComposition[msIdx]) : 0;
  const compForGWP = [...currentComposition];
  if (tempIdx >= 0) compForGWP[tempIdx] = 22; // reference temperature
  const gwp = predictGWP(compForGWP, gwpParams, ms);
  // GWP model predicts -GWP (negated), so negate to get positive GWP
  document.getElementById("gwp-value").textContent =
    (Math.abs(gwp.mean) * u.gwpFactor).toFixed(1);

  // Cost (with uncertainty). Show ±2σ to match the strength-curve
  // band convention; the explicit "(±2σ)" suffix tells users which
  // confidence band they're seeing rather than leaving them to guess
  // at the meaning of the bare ``±``.
  const cost = predictCost(compForGWP, costParams);
  const costMean = Math.abs(cost.mean) * u.costFactor;
  const costStd = Math.sqrt(cost.variance) * u.costFactor;
  document.getElementById("cost-value").textContent = costMean.toFixed(1);
  document.getElementById("cost-uncertainty").textContent =
    `± ${(2 * costStd).toFixed(1)} (2σ)`;

  // W/B ratio
  const cols = compositionsData.column_names;
  const cement = currentComposition[colIdx(cols, "Cement (kg/m3)")];
  const flyAsh = currentComposition[colIdx(cols, "Fly Ash (kg/m3)")];
  const slag = currentComposition[colIdx(cols, "Slag (kg/m3)")];
  const water = currentComposition[colIdx(cols, "Water (kg/m3)")];
  const binder = cement + flyAsh + slag;
  const wb = binder > 0 ? (water / binder).toFixed(3) : "–";
  document.getElementById("wb-value").textContent = wb;

  // TODO: Slump prediction — requires model/slump.json with trained GP params.
  // Once available: load slumpParams in init(), add predictSlump to gp.mjs,
  // then: const slump = predictSlump(compForGWP, slumpParams);
  // Display in "slump-value" element with unit conversion (mm ↔ in).
}

// --- HiDPI Canvas Helpers ---
// Cache canvas dimensions to avoid forced reflow on every frame
const _canvasCache = new WeakMap();
let _resizeObserver = null;

function setupHiDPICanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;

  // Use cached dimensions if available (avoids forced reflow from getBoundingClientRect)
  let rect = _canvasCache.get(canvas);
  if (!rect) {
    rect = canvas.getBoundingClientRect();
    _canvasCache.set(canvas, { width: rect.width, height: rect.height });
    rect = _canvasCache.get(canvas);
    // Observe resize to invalidate cache and trigger redraw
    if (!_resizeObserver) {
      let resizeRAF = null;
      _resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          _canvasCache.delete(entry.target);
        }
        // Debounce redraw to next animation frame for snappy resize
        if (!resizeRAF) {
          resizeRAF = requestAnimationFrame(() => {
            resizeRAF = null;
            update();
          });
        }
      });
    }
    _resizeObserver.observe(canvas);
  }

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, W: rect.width, H: rect.height };
}

/**
 * Generate nice round tick values for an axis range.
 */
function niceTickValues(min, max, approxCount) {
  const range = max - min;
  const rawStep = range / approxCount;
  // Round step to 1, 2, or 5 × 10^n
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  let step;
  if (rawStep / mag < 1.5) step = mag;
  else if (rawStep / mag < 3.5) step = 2 * mag;
  else if (rawStep / mag < 7.5) step = 5 * mag;
  else step = 10 * mag;

  const ticks = [];
  const start = Math.ceil(min / step) * step;
  for (let v = start; v <= max; v += step) {
    ticks.push(Math.round(v * 1e6) / 1e6); // avoid floating point drift
  }
  return ticks;
}

/**
 * Compute Pareto non-dominated mask for minimizing x and maximizing y.
 * A point is Pareto-optimal if no other point has both lower x AND higher y.
 */
function computeParetoMask(xVals, yVals) {
  const n = xVals.length;
  const mask = new Array(n).fill(true);
  for (let i = 0; i < n; i++) {
    if (!mask[i]) continue;
    for (let j = 0; j < n; j++) {
      if (i === j || !mask[j]) continue;
      // j dominates i if j has lower-or-equal x AND higher-or-equal y (with at least one strict)
      if (xVals[j] <= xVals[i] && yVals[j] >= yVals[i] &&
          (xVals[j] < xVals[i] || yVals[j] > yVals[i])) {
        mask[i] = false;
        break;
      }
    }
  }
  return mask;
}

// --- Find nearest composition index in the dataset ---
function findNearestCompositionIdx(comp) {
  if (!compositionsData || !compositionsData.compositions) return null;
  const compositions = compositionsData.compositions;
  let bestDist = Infinity;
  let bestIdx = null;
  for (let i = 0; i < compositions.length; i++) {
    let dist = 0;
    for (let j = 0; j < comp.length; j++) {
      const d = comp[j] - compositions[i][j];
      dist += d * d;
    }
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = i;
    }
  }
  // Only match if essentially exact (squared distance < 1e-6)
  return bestDist < 1e-6 ? bestIdx : null;
}

// --- Extrapolation Warning ---
// Shows a warning when the current composition is far from any training data point.
// Uses normalized Euclidean distance (each dimension / its range) to the nearest point.
//
// The threshold is defined as a per-dimension average normalized difference.
// With EXTRAPOLATION_PER_DIM_THRESHOLD = 0.13, the warning fires when the nearest
// training point differs by ~13% of each dimension's range on average. The actual
// L2 threshold scales as: per_dim_threshold * sqrt(n_active_dims), adapting
// automatically to datasets with different dimensionality.
const EXTRAPOLATION_PER_DIM_THRESHOLD = 0.13;

function getExtrapolationThreshold() {
  if (!compositionsData) return Infinity;
  const bounds = compositionsData.slider_bounds;
  const colNames = compositionsData.column_names;
  let nActiveDims = 0;
  for (const col of colNames) {
    const b = bounds[col];
    if (b.max - b.min > 0) nActiveDims++;
  }
  return EXTRAPOLATION_PER_DIM_THRESHOLD * Math.sqrt(nActiveDims);
}

function checkExtrapolationWarning() {
  const warningEl = document.getElementById("extrapolation-warning");
  if (!warningEl || !compositionsData) return;

  const bounds = compositionsData.slider_bounds;
  const colNames = compositionsData.column_names;
  const compositions = compositionsData.compositions;
  const threshold = getExtrapolationThreshold();

  // Compute normalized distance to nearest training point
  let minDist = Infinity;
  for (let i = 0; i < compositions.length; i++) {
    let dist = 0;
    for (let j = 0; j < currentComposition.length; j++) {
      const col = colNames[j];
      const b = bounds[col];
      const range = b.max - b.min;
      if (range === 0) continue;
      const diff = (currentComposition[j] - compositions[i][j]) / range;
      dist += diff * diff;
    }
    dist = Math.sqrt(dist);
    if (dist < minDist) minDist = dist;
  }

  if (minDist > threshold) {
    warningEl.classList.add("visible");
  } else {
    warningEl.classList.remove("visible");
  }
}

// Hide warning during animated transitions (scatter click)
function hideExtrapolationWarning() {
  const warningEl = document.getElementById("extrapolation-warning");
  if (warningEl) warningEl.classList.remove("visible");
}

// --- Strength Curve Canvas ---
function drawStrengthCurve() {
  const canvas = document.getElementById("curve-canvas");
  const { ctx, W, H } = setupHiDPICanvas(canvas);
  const pad = { top: 20, right: 20, bottom: 40, left: 70 };

  // Compute predictions (use log-spaced time points for smooth early-time resolution).
  // Two paths:
  //   (1) Material Source transition active — blend cached pre-toggle curve
  //       with freshly computed post-toggle curve over MS_TRANSITION_MS.
  //   (2) Otherwise — standard predict at current composition.
  let times, means, stds, nPts;
  if (_msCurveTransition !== null) {
    const elapsed = performance.now() - _msCurveTransition.startTime;
    const t = Math.min(elapsed / _msCurveTransition.duration, 1);
    if (t >= 1) {
      _msCurveTransition = null; // fall through to standard path
    } else {
      const ease = easeInOutCubic(t);
      times = _msCurveTransition.times;
      nPts = times.length;
      const { means: toMeans, variances: toVars } = predictStrengthCurve(
        currentComposition, times, strengthParams
      );
      const toStds = computeStds(toVars, strengthParams);
      const { fromMeans, fromStds } = _msCurveTransition;
      means = fromMeans.map((m, i) => m + (toMeans[i] - m) * ease);
      stds = fromStds.map((s, i) => s + (toStds[i] - s) * ease);
    }
  }
  if (means === undefined) {
    const isAnimating = animationId !== null;
    const isInteracting = _sliderActive || isAnimating;
    nPts = isInteracting ? 32 : 64;
    times = logSpacedTimes(nPts);
    const { means: m, variances } = predictStrengthCurve(
      currentComposition, times, strengthParams
    );
    means = m;
    stds = computeStds(variances, strengthParams);
  }

  // Dynamic Y range: floor at 16500 psi (max observed: 16029), expands smoothly if needed
  // _curveYMax is stored in RAW (psi) space to be unit-invariant — prevents visual drift
  // during unit transitions where sf changes every frame.
  const df = getDisplayFactors();
  const sf = df.strengthFactor;
  const yMin = 0;
  const yMaxFloorRaw = 16500; // in psi (raw units)
  // Compute required max in RAW units (before unit conversion)
  const peakValRaw = Math.max(...means.map((m, i) => m + 2 * stds[i]));
  const yMaxNeededRaw = Math.max(yMaxFloorRaw, peakValRaw * 1.1);
  _curveYMaxTarget = yMaxNeededRaw;
  // Smooth interpolation in raw space — snap for small differences or during unit transition
  if (_curveYMax === null || unitTransition !== null) {
    _curveYMax = yMaxNeededRaw;
  } else {
    const diff = Math.abs(_curveYMax - yMaxNeededRaw);
    if (diff < yMaxNeededRaw * 0.02) {
      _curveYMax = yMaxNeededRaw;
    } else {
      _curveYMax += (yMaxNeededRaw - _curveYMax) * 0.15;
    }
  }
  // Convert to display units at the last moment
  const yMax = _curveYMax * sf;

  // Coordinate transforms
  const xScale = (t) => pad.left + ((t / 28) * (W - pad.left - pad.right));
  const yScale = (v) => H - pad.bottom - ((v - yMin) / (yMax - yMin)) * (H - pad.top - pad.bottom);

  ctx.clearRect(0, 0, W, H);
  const colors = getCanvasColors();

  // Draw uncertainty band (clipped at 0)
  ctx.beginPath();
  for (let i = 0; i < nPts; i++) {
    const x = xScale(times[i]);
    const y = yScale(Math.max(0, (means[i] + 2 * stds[i]) * sf));
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  for (let i = nPts - 1; i >= 0; i--) {
    const x = xScale(times[i]);
    const y = yScale(Math.max(0, (means[i] - 2 * stds[i]) * sf));
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = colors.band;
  ctx.fill();

  // Draw mean curve (clipped at 0 — strength is non-negative)
  ctx.beginPath();
  for (let i = 0; i < nPts; i++) {
    const x = xScale(times[i]);
    const y = yScale(Math.max(0, means[i] * sf));
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.strokeStyle = colors.accent;
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Preview curve: compute directly from interpolated displayPreviewComp
  const showPreview = isPreviewActive || !isCompositionConverged();
  if (showPreview) {
    const previewMeans = predictStrengthMeanOnly(displayPreviewComp, previewTimesCache, strengthParams);
    ctx.beginPath();
    for (let i = 0; i < PREVIEW_PTS; i++) {
      const x = xScale(previewTimesCache[i]);
      const y = yScale(Math.max(0, previewMeans[i] * sf));
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = colors.point;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  // Overlay actual observations for the nearest matching composition
  curveObsPositions = [];
  const nearestIdx = findNearestCompositionIdx(currentComposition);
  if (nearestIdx !== null && compositionsData.observations) {
    const obs = compositionsData.observations[String(nearestIdx)];
    if (obs && obsOpacity > 0.01) {
      ctx.globalAlpha = obsOpacity;
      for (let oi = 0; oi < obs.length; oi++) {
        const [t, rawY] = obs[oi];
        const dispY = rawY * sf;
        const x = xScale(t);
        const yp = yScale(dispY);
        curveObsPositions.push({ px: x, py: yp, time: t, strength: rawY, idx: oi });
        // Animated hover: enlarge if this is the hovered observation
        const isHovered = (hoveredCurveObsIdx === oi);
        const radius = isHovered ? 4 + curveObsHoverScale * 3 : 4;
        if (isHovered && curveObsHoverScale > 0.01) {
          // Glow halo behind the marker
          ctx.beginPath();
          ctx.arc(x, yp, radius + 4 * curveObsHoverScale, 0, 2 * Math.PI);
          ctx.fillStyle = `rgba(217, 119, 6, ${0.2 * curveObsHoverScale})`;
          ctx.fill();
        }
        // Marker fill + white outline (single draw, was previously
        // drawn twice).
        ctx.beginPath();
        ctx.arc(x, yp, radius, 0, 2 * Math.PI);
        ctx.fillStyle = colors.observation;
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }

  // Axes
  ctx.strokeStyle = colors.axis;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, H - pad.bottom);
  ctx.lineTo(W - pad.right, H - pad.bottom);
  ctx.stroke();

  // X ticks
  ctx.fillStyle = colors.text;
  ctx.font = "bold 12px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.textAlign = "center";
  for (const t of [0, 1, 3, 7, 14, 28]) {
    const x = xScale(t);
    ctx.beginPath();
    ctx.moveTo(x, H - pad.bottom);
    ctx.lineTo(x, H - pad.bottom + 4);
    ctx.stroke();
    ctx.fillText(t, x, H - pad.bottom + 16);
  }
  ctx.fillText("Curing Age (days)", W / 2, H - 4);

  // Y ticks (use nice ticks for current unit)
  ctx.textAlign = "right";
  const yTickValues = niceTickValues(yMin, yMax, 5);
  for (const v of yTickValues) {
    const y = yScale(v);
    ctx.beginPath();
    ctx.moveTo(pad.left - 4, y);
    ctx.lineTo(pad.left, y);
    ctx.stroke();
    ctx.fillText(Math.round(v).toLocaleString(), pad.left - 8, y + 4);
  }
  // Y label
  ctx.save();
  ctx.translate(14, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(`Strength (${U().strength})`, 0, 0);
  ctx.restore();

  // Uncertainty-band legend (top-right corner, in-canvas). Tells
  // viewers what the shaded band means without needing to open the
  // About modal. Matches the ``± 2σ`` convention used by the cost
  // readout and the About-text description.
  ctx.save();
  ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  // Legend swatch (small filled rectangle in band color).
  const swatchSize = 10;
  const legendPadX = 8;
  const legendPadY = 6;
  const swatchX = W - pad.right - 70;
  const swatchY = pad.top + legendPadY;
  ctx.fillStyle = colors.band;
  ctx.fillRect(swatchX, swatchY + 1, swatchSize, swatchSize);
  ctx.fillStyle = colors.text;
  ctx.fillText(
    "shaded: ±2σ",
    W - pad.right - legendPadX,
    pad.top + legendPadY,
  );
  ctx.restore();
}

// --- Cached Pareto Mask (O(n²) — avoid recomputing every frame) ---
let _paretoCache = null; // { key, mask }

function getCachedParetoMask(xVals, yVals) {
  const key = `${scatterDay}_${scatterXAxis}`;
  if (_paretoCache && _paretoCache.key === key) return _paretoCache.mask;
  const mask = computeParetoMask(xVals, yVals);
  _paretoCache = { key, mask };
  return mask;
}

// --- Scatter Plot Transition Animation ---
function getScatterData() {
  const df = getDisplayFactors();
  const gwpPreds = compositionsData.gwp_predictions.map((v) => -v * df.gwpFactor);
  const costPreds = compositionsData.cost_predictions.map((v) => -v * df.costFactor);
  const xPreds = scatterXAxis === "cost" ? costPreds : gwpPreds;
  const strPreds = compositionsData.strength_predictions[String(scatterDay)].map(v => v * df.strengthFactor);
  const paretoMask = getCachedParetoMask(xPreds, strPreds);
  return { xPreds, yPreds: strPreds, paretoMask };
}

function startScatterTransition(applyChange) {
  // Capture current positions and axis ranges
  const before = getScatterData();
  const beforeRange = getAxisRange(before.xPreds, before.yPreds);

  // Apply the change (updates scatterDay or scatterXAxis)
  applyChange();

  // Capture new positions and axis ranges
  const after = getScatterData();
  const afterRange = getAxisRange(after.xPreds, after.yPreds);

  scatterTransition = {
    startTime: performance.now(),
    duration: 350,
    fromX: before.xPreds,
    fromY: before.yPreds,
    fromPareto: before.paretoMask,
    toX: after.xPreds,
    toY: after.yPreds,
    toPareto: after.paretoMask,
    fromXMax: beforeRange.xMax,
    fromYMax: beforeRange.yMax,
    toXMax: afterRange.xMax,
    toYMax: afterRange.yMax,
  };

  startAnimLoop();
}

// Null out scatter transition when complete (called from drawScatter)
function checkScatterTransitionDone() {
  if (scatterTransition) {
    const elapsed = performance.now() - scatterTransition.startTime;
    if (elapsed >= scatterTransition.duration) {
      scatterTransition = null;
    }
  }
}

function getAxisRange(xPreds, yPreds) {
  const xMax = Math.max(...xPreds) * 1.05;
  const yMax = Math.max(...yPreds.map(v => Math.max(0, v))) * 1.1;
  return { xMax, yMax };
}

// --- Scatter Plot Canvas ---
// NOTE: The scatter plot displays MODEL PREDICTIONS (GP posterior means), not raw
// observations. This is necessary because: (1) interactivity requires predictions at
// arbitrary slider-controlled compositions, (2) each point shows predicted strength at
// a fixed curing day across all compositions, and (3) GWP is always a linear model
// prediction. Actual observations are overlaid on the strength curve (right panel)
// when viewing a specific mix, for ground-truth validation.
function drawScatter() {
  checkScatterTransitionDone();
  const canvas = document.getElementById("scatter-canvas");
  const { ctx, W, H } = setupHiDPICanvas(canvas);
  const pad = { top: 20, right: 20, bottom: 40, left: 70 };

  // gwp_predictions stores -GWP; negate to get positive GWP for display
  // cost_predictions stores -Cost; negate to get positive Cost for display
  // Apply unit conversion factors for display
  const df = getDisplayFactors();
  const yFactor = df.strengthFactor;
  const gwpPreds = compositionsData.gwp_predictions.map((v) => -v * df.gwpFactor);
  const costPreds = compositionsData.cost_predictions.map((v) => -v * df.costFactor);
  let xPreds = scatterXAxis === "cost" ? costPreds : gwpPreds;
  let strPreds = compositionsData.strength_predictions[String(scatterDay)].map(v => v * yFactor);

  // Use cached Pareto mask (scale-invariant, keyed by scatterDay + scatterXAxis)
  let paretoMask = getCachedParetoMask(xPreds, strPreds);

  // If animating between objectives, interpolate positions
  let transT = 1;
  let overrideXMax = null;
  let overrideYMax = null;
  if (scatterTransition) {
    const elapsed = performance.now() - scatterTransition.startTime;
    transT = Math.min(elapsed / scatterTransition.duration, 1);
    // Ease-in-out cubic
    const ease = easeInOutCubic(transT);

    const n = xPreds.length;
    const interpX = new Array(n);
    const interpY = new Array(n);
    for (let i = 0; i < n; i++) {
      interpX[i] = scatterTransition.fromX[i] + (scatterTransition.toX[i] - scatterTransition.fromX[i]) * ease;
      interpY[i] = scatterTransition.fromY[i] + (scatterTransition.toY[i] - scatterTransition.fromY[i]) * ease;
    }
    xPreds = interpX;
    strPreds = interpY;
    // Interpolate axis ranges smoothly
    overrideXMax = scatterTransition.fromXMax + (scatterTransition.toXMax - scatterTransition.fromXMax) * ease;
    overrideYMax = scatterTransition.fromYMax + (scatterTransition.toYMax - scatterTransition.fromYMax) * ease;
    // Use target Pareto mask (snaps at midpoint)
    paretoMask = ease > 0.5 ? scatterTransition.toPareto : scatterTransition.fromPareto;
  }

  // Compute current point (use fixed Temp for GWP/Cost since they're material properties)
  const msIdx = COL_MS;
  const tempIdx = COL_TEMP;
  const ms = msIdx >= 0 ? Math.round(currentComposition[msIdx]) : 0;
  const compForCost = [...currentComposition];
  if (tempIdx >= 0) compForCost[tempIdx] = 22; // reference temperature
  const curGWPRaw = predictGWP(compForCost, gwpParams, ms).mean;
  const curCostRaw = predictCost(compForCost, costParams).mean;
  const curX = scatterXAxis === "cost" ? -curCostRaw * df.costFactor : -curGWPRaw * df.gwpFactor;
  const curStr = predictStrengthCurve(
    currentComposition, [scatterDay], strengthParams
  ).means[0] * yFactor;

  // Axis ranges (use interpolated ranges during transition to avoid jumps)
  const xVals = xPreds;
  const yVals = strPreds.map((v) => Math.max(0, v)); // clip negative predictions
  const xMin = 0; // physical lower bound for both GWP and Cost
  const xMax = overrideXMax !== null ? overrideXMax : Math.max(...xVals, curX) * 1.05;
  const yMin = 0;
  const yMax = overrideYMax !== null ? overrideYMax : Math.max(...yVals, Math.max(0, curStr)) * 1.1;

  const xScale = (v) => pad.left + ((v - xMin) / (xMax - xMin)) * (W - pad.left - pad.right);
  const yScale = (v) => H - pad.bottom - ((v - yMin) / (yMax - yMin)) * (H - pad.top - pad.bottom);

  ctx.clearRect(0, 0, W, H);

  // Theme-aware colors
  const clr = getCanvasColors();

  // Draw Pareto frontier staircase (behind points)
  const paretoIndices = [];
  for (let i = 0; i < xVals.length; i++) {
    if (paretoMask[i]) paretoIndices.push(i);
  }
  if (paretoIndices.length > 1) {
    // Sort Pareto points by x ascending
    paretoIndices.sort((a, b) => xVals[a] - xVals[b]);
    ctx.beginPath();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = clr.pareto;
    ctx.globalAlpha = 0.35;
    ctx.lineWidth = 1.5;
    const firstIdx = paretoIndices[0];
    ctx.moveTo(xScale(xVals[firstIdx]), yScale(yVals[firstIdx]));
    for (let k = 1; k < paretoIndices.length; k++) {
      const prevIdx = paretoIndices[k - 1];
      const curIdx = paretoIndices[k];
      // Horizontal line at previous y to current x
      ctx.lineTo(xScale(xVals[curIdx]), yScale(yVals[prevIdx]));
      // Vertical line down to current y
      ctx.lineTo(xScale(xVals[curIdx]), yScale(yVals[curIdx]));
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1.0;
  }

  // Draw points (apply filter if active)
  for (let i = 0; i < xVals.length; i++) {
    // Check all filter conditions
    if (scatterFilter && scatterFilter.length > 0) {
      const comp = compositionsData.compositions[i];
      let filtered = false;
      for (const f of scatterFilter) {
        const val = f.computed ? f.computed(comp) : comp[f.colIdx];
        if (val < f.min || val > f.max) { filtered = true; break; }
      }
      if (filtered) {
        const x = xScale(xVals[i]);
        const y = yScale(yVals[i]);
        ctx.beginPath();
        ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(148, 163, 184, 0.3)";
        ctx.fill();
        continue;
      }
    }
    const x = xScale(xVals[i]);
    const y = yScale(yVals[i]);
    const isPareto = paretoMask[i];
    ctx.beginPath();
    ctx.arc(x, y, isPareto ? 5 : 3.5, 0, 2 * Math.PI);
    ctx.fillStyle = isPareto ? clr.pareto : clr.point;
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.7)";
    ctx.lineWidth = 0.8;
    ctx.stroke();
  }

  // Draw shrinking previous hovered point
  if (prevHoveredPointIdx !== null && prevHoverScale > 0.01 && prevHoveredPointIdx < xVals.length) {
    const phx = xScale(xVals[prevHoveredPointIdx]);
    const phy = yScale(yVals[prevHoveredPointIdx]);
    const isPareto = paretoMask[prevHoveredPointIdx];
    const baseR = isPareto ? 5 : 3.5;
    const scaleFactor = 1 + prevHoverScale * 0.5;
    const glowFactor = 1 + prevHoverScale * 1.2;
    ctx.beginPath();
    ctx.arc(phx, phy, baseR * glowFactor, 0, 2 * Math.PI);
    ctx.fillStyle = isPareto
      ? `rgba(225, 29, 72, ${0.15 * prevHoverScale})`
      : `rgba(71, 85, 105, ${0.15 * prevHoverScale})`;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(phx, phy, baseR * scaleFactor, 0, 2 * Math.PI);
    ctx.fillStyle = isPareto ? clr.pareto : clr.point;
    ctx.fill();
    ctx.strokeStyle = `rgba(255,255,255,${0.9 * prevHoverScale})`;
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }

  // Hover glow effect on nearest point (animated size)
  if (hoverScale > 0.01 && hoveredPointIdx !== null && hoveredPointIdx < xVals.length) {
    const hx = xScale(xVals[hoveredPointIdx]);
    const hy = yScale(yVals[hoveredPointIdx]);
    const isPareto = paretoMask[hoveredPointIdx];
    const baseR = isPareto ? 5 : 3.5;
    const scaleFactor = 1 + hoverScale * 0.5; // 1.0 → 1.5
    const glowFactor = 1 + hoverScale * 1.2; // 1.0 → 2.2
    // Draw glow
    ctx.beginPath();
    ctx.arc(hx, hy, baseR * glowFactor, 0, 2 * Math.PI);
    ctx.fillStyle = isPareto
      ? `rgba(225, 29, 72, ${0.15 * hoverScale})`
      : `rgba(71, 85, 105, ${0.15 * hoverScale})`;
    ctx.fill();
    // Draw enlarged point
    ctx.beginPath();
    ctx.arc(hx, hy, baseR * scaleFactor, 0, 2 * Math.PI);
    ctx.fillStyle = isPareto ? clr.pareto : clr.point;
    ctx.fill();
    ctx.strokeStyle = `rgba(255,255,255,${0.9 * hoverScale})`;
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }

  // Highlight selected composition with a pulsing glow ring (no crosshair)
  const cx = xScale(curX);
  const cy = yScale(Math.max(0, curStr));
  const nearIdx = findNearestCompositionIdx(currentComposition);

  // Animated pulse: subtle radius oscillation using time
  const pulse = Math.sin(Date.now() / 400) * 0.5 + 0.5; // 0-1 oscillation
  const ringRadius = 6.5 + pulse * 1.5;
  const ringAlpha = 0.7 + pulse * 0.3;

  // Use brighter ring color (theme-aware via accent)
  const ringColor = clr.accent;
  // Parse hex to RGB for alpha control
  const rr = parseInt(ringColor.slice(1, 3), 16) || 96;
  const rg = parseInt(ringColor.slice(3, 5), 16) || 165;
  const rb = parseInt(ringColor.slice(5, 7), 16) || 250;

  if (nearIdx !== null) {
    const hx = xScale(xVals[nearIdx]);
    const hy = yScale(yVals[nearIdx]);
    // Outer glow
    ctx.beginPath();
    ctx.arc(hx, hy, ringRadius + 4, 0, 2 * Math.PI);
    ctx.strokeStyle = `rgba(${rr}, ${rg}, ${rb}, ${ringAlpha * 0.3})`;
    ctx.lineWidth = 5;
    ctx.stroke();
    // Inner ring
    ctx.beginPath();
    ctx.arc(hx, hy, ringRadius, 0, 2 * Math.PI);
    ctx.strokeStyle = `rgba(${rr}, ${rg}, ${rb}, ${ringAlpha})`;
    ctx.lineWidth = 2.5;
    ctx.stroke();
  } else {
    // Not a dataset point — show a subtle accent ring
    ctx.beginPath();
    ctx.arc(cx, cy, ringRadius, 0, 2 * Math.PI);
    ctx.strokeStyle = `rgba(${rr}, ${rg}, ${rb}, ${ringAlpha})`;
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }


  // Axes
  ctx.strokeStyle = clr.axis;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, H - pad.bottom);
  ctx.lineTo(W - pad.right, H - pad.bottom);
  ctx.stroke();

  // Labels (unit-aware)
  ctx.fillStyle = clr.text;
  ctx.font = "bold 12px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.textAlign = "center";
  const xLabel = scatterXAxis === "cost" ? `Cost (${df.cost})` : `GWP (${df.gwp})`;
  ctx.fillText(xLabel, W / 2, H - 4);

  // X ticks (round numbers)
  const xTickValues = niceTickValues(xMin, xMax, 5);
  for (const v of xTickValues) {
    const x = xScale(v);
    ctx.beginPath();
    ctx.moveTo(x, H - pad.bottom);
    ctx.lineTo(x, H - pad.bottom + 4);
    ctx.stroke();
    ctx.fillText(Math.round(v), x, H - pad.bottom + 16);
  }

  // Y ticks (round numbers)
  ctx.textAlign = "right";
  const yTickValues = niceTickValues(yMin, yMax, 5);
  for (const v of yTickValues) {
    const y = yScale(v);
    ctx.beginPath();
    ctx.moveTo(pad.left - 4, y);
    ctx.lineTo(pad.left, y);
    ctx.stroke();
    ctx.fillText(Math.round(v).toLocaleString(), pad.left - 8, y + 4);
  }
  ctx.save();
  ctx.translate(14, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(`${scatterDay}-day Strength (${df.strength})`, 0, 0);
  ctx.restore();

  // Store scale functions for click handling
  canvas._xMin = xMin;
  canvas._xMax = xMax;
  canvas._yMin = yMin;
  canvas._yMax = yMax;
  canvas._pad = pad;
  canvas._W = W;
  canvas._H = H;
}

// --- Event Listeners ---
let hoveredPointIdx = null; // track hovered scatter point for glow effect
let hoverScale = 0; // animated hover scale (0 = normal, 1 = fully enlarged)
let hoverScaleTarget = 0;
let prevHoveredPointIdx = null; // previous hovered point (shrinking out)
let prevHoverScale = 0; // shrink-out scale for previous point
let hoveredCurveObsIdx = null; // hovered observation in strength curve
let curveObsHoverScale = 0; // animated hover scale for curve obs
let obsOpacity = 0; // smooth fade for observation points
let prevObsIdx = null; // track which observations are currently shown

// --- Unified Smooth Preview System ---
let previewTarget = null; // composition we're interpolating TOWARD (set on hover)
let displayPreviewComp = null; // always-valid interpolated composition (initialized on load)
let isPreviewActive = false; // true when hovering a scatter point
const PREVIEW_PTS = 48;
const previewTimesCache = logSpacedTimes(PREVIEW_PTS);

function isCompositionConverged() {
  for (let i = 0; i < displayPreviewComp.length; i++) {
    if (Math.abs(displayPreviewComp[i] - currentComposition[i]) > 1e-6) return false;
  }
  return true;
}

// --- Unified Animation Loop ---
function startAnimLoop() {
  if (animLoopId) return;
  lastFrameTime = performance.now();
  animLoopId = requestAnimationFrame(animLoop);
}

function animLoop(now) {
  const dt = now - lastFrameTime;
  lastFrameTime = now;

  // Frame-rate-independent interpolation factor
  const factor = 1 - Math.pow(0.85, dt / 16.67);

  // Interpolate displayPreviewComp toward target
  const target = isPreviewActive ? previewTarget : currentComposition;
  let previewConverged = true;
  for (let i = 0; i < displayPreviewComp.length; i++) {
    const diff = target[i] - displayPreviewComp[i];
    if (Math.abs(diff) > 1e-6) {
      displayPreviewComp[i] += diff * factor;
      previewConverged = false;
    } else {
      displayPreviewComp[i] = target[i];
    }
  }

  // Show/hide slider preview markers from interpolated composition
  if (isPreviewActive || !previewConverged) {
    showSliderPreview(displayPreviewComp);
  } else {
    hideSliderPreview();
  }

  // Animate hover scales
  hoverScale += (hoverScaleTarget - hoverScale) * 0.3;
  if (Math.abs(hoverScale - hoverScaleTarget) < 0.01) hoverScale = hoverScaleTarget;
  if (prevHoveredPointIdx !== null) {
    prevHoverScale *= 0.7;
    if (prevHoverScale < 0.02) { prevHoveredPointIdx = null; prevHoverScale = 0; }
  }
  const curveHoverTarget = hoveredCurveObsIdx !== null ? 1 : 0;
  curveObsHoverScale += (curveHoverTarget - curveObsHoverScale) * 0.3;
  if (Math.abs(curveObsHoverScale - curveHoverTarget) < 0.01) curveObsHoverScale = curveHoverTarget;

  // Smooth fade for observation points
  const nearIdx = findNearestCompositionIdx(currentComposition);
  const hasObs = nearIdx !== null && compositionsData.observations && compositionsData.observations[String(nearIdx)];
  const obsTarget = hasObs ? 1 : 0;
  if (nearIdx !== prevObsIdx) {
    // Composition changed to different point — if new point has obs, fade in from 0
    if (hasObs && prevObsIdx !== null) obsOpacity = 0;
    prevObsIdx = nearIdx;
  }
  obsOpacity += (obsTarget - obsOpacity) * 0.15;
  if (Math.abs(obsOpacity - obsTarget) < 0.01) obsOpacity = obsTarget;

  // Redraw
  drawScatter();
  drawStrengthCurve();

  // Continue or stop (no wasted work when idle)
  const hasHoverAnim = (Math.abs(hoverScale - hoverScaleTarget) > 0.01) || prevHoveredPointIdx !== null;
  const hasCurveAnim = Math.abs(curveObsHoverScale - curveHoverTarget) > 0.01;
  const hasObsAnim = Math.abs(obsOpacity - obsTarget) > 0.01;
  const hasYAxisAnim = _curveYMaxTarget !== null && Math.abs(_curveYMax - _curveYMaxTarget) > 0.5;
  if (!previewConverged || isPreviewActive || hasHoverAnim || hasCurveAnim || hasObsAnim || hasYAxisAnim || animationId !== null || scatterTransition !== null || unitTransition !== null || _msCurveTransition !== null) {
    animLoopId = requestAnimationFrame(animLoop);
  } else {
    animLoopId = null;
  }
}

function setupEventListeners() {
  const scatterCanvas = document.getElementById("scatter-canvas");

  // Scatter hover — enlarge point with glow on hover
  scatterCanvas.addEventListener("mousemove", (e) => {
    const rect = scatterCanvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    const pad = scatterCanvas._pad;
    const W = scatterCanvas._W;
    const H = scatterCanvas._H;
    if (!pad) return;

    // Convert pixel to data coordinates
    const dataX = scatterCanvas._xMin + ((px - pad.left) / (W - pad.left - pad.right)) * (scatterCanvas._xMax - scatterCanvas._xMin);
    const dataY = scatterCanvas._yMin + ((H - pad.bottom - py) / (H - pad.top - pad.bottom)) * (scatterCanvas._yMax - scatterCanvas._yMin);

    // Find nearest point in pixel space
    const gwpPreds = compositionsData.gwp_predictions.map((v) => -v * U().gwpFactor);
    const costPreds = compositionsData.cost_predictions.map((v) => -v * U().costFactor);
    const hoverXPreds = scatterXAxis === "cost" ? costPreds : gwpPreds;
    const strPreds = compositionsData.strength_predictions[String(scatterDay)].map(v => v * U().strengthFactor);

    let bestDist = Infinity;
    let bestIdx = -1;
    const xRange = scatterCanvas._xMax - scatterCanvas._xMin;
    const yRange = scatterCanvas._yMax - scatterCanvas._yMin;
    for (let i = 0; i < hoverXPreds.length; i++) {
      const dx = (hoverXPreds[i] - dataX) / xRange * (W - pad.left - pad.right);
      const dy = (strPreds[i] - dataY) / yRange * (H - pad.top - pad.bottom);
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }

    const newHovered = bestDist < 12 ? bestIdx : null;
    if (newHovered !== hoveredPointIdx) {
      // Transfer old hovered point to "previous" for smooth shrink-out
      if (hoveredPointIdx !== null && hoverScale > 0.01) {
        prevHoveredPointIdx = hoveredPointIdx;
        prevHoverScale = hoverScale;
      }
      hoveredPointIdx = newHovered;
      hoverScale = 0;
      hoverScaleTarget = newHovered !== null ? 1 : 0;
      scatterCanvas.style.cursor = hoveredPointIdx !== null ? "pointer" : "default";
      // Update preview target
      if (hoveredPointIdx !== null) {
        previewTarget = compositionsData.compositions[hoveredPointIdx];
        isPreviewActive = true;
      } else {
        isPreviewActive = false;
      }
      startAnimLoop();
    }
  });

  scatterCanvas.addEventListener("mouseleave", () => {
    if (hoveredPointIdx !== null) {
      // Transfer to previous for smooth shrink-out
      prevHoveredPointIdx = hoveredPointIdx;
      prevHoverScale = hoverScale;
      hoveredPointIdx = null;
      hoverScale = 0;
      hoverScaleTarget = 0;
      isPreviewActive = false;
      scatterCanvas.style.cursor = "default";
      startAnimLoop();
    }
  });

  // Scatter click → set sliders to nearest composition
  scatterCanvas.addEventListener("click", () => {
    // If a point is already hovered, use it directly (guarantees preview matches selection)
    const idx = hoveredPointIdx !== null ? hoveredPointIdx : null;
    if (idx !== null) {
      animateToComposition(compositionsData.compositions[idx]);
    }
  });

  // Clickable axis toggles (replacing dropdowns)
  const toggleX = document.getElementById("toggle-x");
  const toggleDay = document.getElementById("toggle-day");
  const xOptions = ["gwp", "cost"];
  const xLabels = ["GWP", "Cost"];
  const dayOptions = [28, 1];
  const dayLabels = ["28-day strength", "1-day strength"];

  toggleX.addEventListener("click", () => {
    const curIdx = xOptions.indexOf(scatterXAxis);
    const nextIdx = (curIdx + 1) % xOptions.length;
    toggleX.textContent = xLabels[nextIdx];
    startScatterTransition(() => { scatterXAxis = xOptions[nextIdx]; });
    updateMixInsight();
  });

  toggleDay.addEventListener("click", () => {
    const curIdx = dayOptions.indexOf(scatterDay);
    const nextIdx = (curIdx + 1) % dayOptions.length;
    toggleDay.textContent = dayLabels[nextIdx];
    startScatterTransition(() => { scatterDay = dayOptions[nextIdx]; });
    updateMixInsight();
  });

  // Filter panel — multi-dimensional filtering
  const filterRows = document.getElementById("filter-rows");
  const colNames = compositionsData.column_names;

  // Computed filter quantities (derived from composition)
  const computedFilters = [
    { id: "wb", label: "W/B Ratio", compute: (comp) => {
      const c = comp[colIdx(colNames, "Cement (kg/m3)")];
      const fa = comp[colIdx(colNames, "Fly Ash (kg/m3)")];
      const s = comp[colIdx(colNames, "Slag (kg/m3)")];
      const w = comp[colIdx(colNames, "Water (kg/m3)")];
      const b = c + fa + s;
      return b > 0 ? w / b : Infinity;
    }},
    { id: "binder", label: "Total Binder", compute: (comp) => {
      const c = comp[colIdx(colNames, "Cement (kg/m3)")];
      const fa = comp[colIdx(colNames, "Fly Ash (kg/m3)")];
      const s = comp[colIdx(colNames, "Slag (kg/m3)")];
      return c + fa + s;
    }},
    { id: "scm", label: "SCM Replacement %", compute: (comp) => {
      const c = comp[colIdx(colNames, "Cement (kg/m3)")];
      const fa = comp[colIdx(colNames, "Fly Ash (kg/m3)")];
      const s = comp[colIdx(colNames, "Slag (kg/m3)")];
      const b = c + fa + s;
      return b > 0 ? (fa + s) / b * 100 : 0;
    }},
    { id: "paste", label: "Paste Fraction", compute: (comp) => {
      const c = comp[colIdx(colNames, "Cement (kg/m3)")];
      const fa = comp[colIdx(colNames, "Fly Ash (kg/m3)")];
      const s = comp[colIdx(colNames, "Slag (kg/m3)")];
      const w = comp[colIdx(colNames, "Water (kg/m3)")];
      const ca = comp[colIdx(colNames, "Coarse Aggregates (kg/m3)")];
      const fna = comp[colIdx(colNames, "Fine Aggregate (kg/m3)")];
      const total = c + fa + s + w + ca + fna;
      return total > 0 ? (c + fa + s + w) / total : 0;
    }},
  ];

  function createFilterColOptions() {
    let html = '<optgroup label="Composition">';
    for (let i = 0; i < colNames.length; i++) {
      html += `<option value="${i}">${colNames[i].replace(" (kg/m3)", "")}</option>`;
    }
    html += '</optgroup><optgroup label="Computed">';
    for (const cf of computedFilters) {
      html += `<option value="computed:${cf.id}">${cf.label}</option>`;
    }
    html += '</optgroup>';
    return html;
  }

  // --- Filter Row Add/Remove Animation ---
  // Uses the Web Animations API (WAAPI) for smooth height transitions.
  //
  // Why WAAPI instead of CSS transitions or grid-template-rows?
  // 1. CSS `max-height` transitions have dead zones (animating from an arbitrary
  //    large value) and timing imprecision with `transitionend`.
  // 2. CSS `grid-template-rows: 1fr → 0fr` doesn't resolve to exactly 0px due to
  //    sub-pixel grid track minimums, causing a visible "snap" at the end.
  // 3. CSS `height` transitions require forced reflows (`element.offsetHeight`) to
  //    synchronize the browser's layout state before animating, which can still
  //    cause micro-jank on complex layouts.
  //
  // WAAPI solves all of these: it natively interpolates `height` from a measured
  // pixel value to exactly 0px, runs on the compositor, fires `onfinish` precisely
  // when the animation completes, and doesn't require forced reflows.

  function addFilterRow() {
    // Remove any dead wrappers from previous removals
    for (const dead of filterRows.querySelectorAll(".filter-row-wrapper.collapsed")) {
      dead.remove();
    }
    const wrapper = document.createElement("div");
    wrapper.className = "filter-row-wrapper";
    wrapper.style.overflow = "hidden";
    const row = document.createElement("div");
    row.className = "filter-row";
    row.innerHTML = `
      <select class="filter-col">${createFilterColOptions()}</select>
      <input class="filter-min" type="number" placeholder="min">
      <span>–</span>
      <input class="filter-max" type="number" placeholder="max">
      <button class="filter-remove-btn" title="Remove this filter">−</button>
    `;
    row.querySelector(".filter-remove-btn").addEventListener("click", () => {
      // Measure current rendered height, then animate to 0
      const h = wrapper.offsetHeight;
      wrapper.style.overflow = "hidden";
      const anim = wrapper.animate([
        { height: `${h}px`, opacity: 1, marginBottom: "0.3rem" },
        { height: "0px", opacity: 0, marginBottom: "0px" }
      ], { duration: 250, easing: "cubic-bezier(0.4, 0, 0.2, 1)", fill: "forwards" });
      anim.onfinish = () => {
        wrapper.classList.add("collapsed");
        wrapper.style.display = "none";
        applyFilters();
      };
    });
    row.querySelector(".filter-min").addEventListener("change", applyFilters);
    row.querySelector(".filter-max").addEventListener("change", applyFilters);
    row.querySelector(".filter-col").addEventListener("change", applyFilters);
    wrapper.appendChild(row);
    filterRows.appendChild(wrapper);
    // Animate expansion: measure natural height, then animate from 0 to that height
    const naturalHeight = wrapper.scrollHeight;
    wrapper.animate([
      { height: "0px", opacity: 0, marginBottom: "0px" },
      { height: `${naturalHeight}px`, opacity: 1, marginBottom: "0.3rem" }
    ], { duration: 250, easing: "cubic-bezier(0.4, 0, 0.2, 1)", fill: "forwards" });
  }

  function applyFilters() {
    const rows = filterRows.querySelectorAll(".filter-row-wrapper:not(.collapsed) .filter-row");
    if (rows.length === 0) {
      scatterFilter = null;
    } else {
      scatterFilter = [];
      for (const row of rows) {
        const colVal = row.querySelector(".filter-col").value;
        const minInput = row.querySelector(".filter-min");
        const maxInput = row.querySelector(".filter-max");
        const minVal = minInput.value;
        const maxVal = maxInput.value;
        // Validate numeric input: parseFloat("") → NaN and parseFloat("foo") →
        // NaN, both of which would silently make the filter dead (any
        // comparison against NaN is false, so nothing gets excluded). Treat
        // an empty input as "no bound" (-/+ Infinity); flag any non-empty
        // non-numeric input visually so the user knows the filter is bad.
        const parseBound = (raw, defaultVal, input) => {
          if (raw === "") {
            input.classList.remove("filter-input-invalid");
            return defaultVal;
          }
          const v = parseFloat(raw);
          if (Number.isNaN(v)) {
            input.classList.add("filter-input-invalid");
            return defaultVal;
          }
          input.classList.remove("filter-input-invalid");
          return v;
        };
        const min = parseBound(minVal, -Infinity, minInput);
        const max = parseBound(maxVal, Infinity, maxInput);
        if (colVal.startsWith("computed:")) {
          const cfId = colVal.replace("computed:", "");
          const cf = computedFilters.find(c => c.id === cfId);
          if (cf) scatterFilter.push({ computed: cf.compute, min, max });
        } else {
          scatterFilter.push({ colIdx: parseInt(colVal), min, max });
        }
      }
    }
    drawScatter();
  }

  document.getElementById("filter-add").addEventListener("click", () => {
    addFilterRow();
  });

  document.getElementById("filter-clear").addEventListener("click", () => {
    // Animate all active filters collapsing simultaneously
    const wrappers = [...filterRows.querySelectorAll(".filter-row-wrapper:not(.collapsed)")];
    if (wrappers.length === 0) return;
    let finished = 0;
    for (const w of wrappers) {
      const h = w.offsetHeight;
      w.style.overflow = "hidden";
      const anim = w.animate([
        { height: `${h}px`, opacity: 1, marginBottom: "0.3rem" },
        { height: "0px", opacity: 0, marginBottom: "0px" }
      ], { duration: 250, easing: "cubic-bezier(0.4, 0, 0.2, 1)", fill: "forwards" });
      anim.onfinish = () => {
        w.classList.add("collapsed");
        w.style.display = "none";
        finished++;
        if (finished === wrappers.length) {
          scatterFilter = null;
          drawScatter();
        }
      };
    }
  });

  // Tooltip on strength curve canvas for observed data points
  const curveCanvas = document.getElementById("curve-canvas");
  const tooltip = document.createElement("div");
  tooltip.className = "tooltip";
  tooltip.style.display = "none";
  document.body.appendChild(tooltip);

  curveCanvas.addEventListener("mousemove", (e) => {
    const rect = curveCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Check if cursor is near any observation point
    let hit = null;
    let hitIdx = null;
    for (const pt of curveObsPositions) {
      const dx = mx - pt.px;
      const dy = my - pt.py;
      if (dx * dx + dy * dy < 100) { // within 10px radius
        hit = pt;
        hitIdx = pt.idx;
        break;
      }
    }

    // Update hover state for animation
    if (hitIdx !== hoveredCurveObsIdx) {
      hoveredCurveObsIdx = hitIdx;
      curveObsHoverScale = 0;
      startAnimLoop();
    }

    if (hit) {
      const dispStrength = (hit.strength * U().strengthFactor);
      const unit = U().strength;
      tooltip.textContent = `Day ${hit.time}: ${dispStrength < 10 ? dispStrength.toFixed(1) : Math.round(dispStrength).toLocaleString()} ${unit}`;
      tooltip.style.display = "block";
      tooltip.style.left = `${e.clientX + 12}px`;
      tooltip.style.top = `${e.clientY - 28}px`;
      curveCanvas.style.cursor = "pointer";
    } else {
      tooltip.style.display = "none";
      curveCanvas.style.cursor = "default";
    }
  });

  curveCanvas.addEventListener("mouseleave", () => {
    tooltip.style.display = "none";
    curveCanvas.style.cursor = "default";
    if (hoveredCurveObsIdx !== null) {
      hoveredCurveObsIdx = null;
      startAnimLoop();
    }
  });
}

// --- Theme-aware canvas colors (cached, invalidated on theme change) ---
let _canvasColors = null;

function getCanvasColors() {
  if (_canvasColors) return _canvasColors;
  const style = getComputedStyle(document.documentElement);
  _canvasColors = {
    axis: style.getPropertyValue("--canvas-axis").trim() || "#1e293b",
    text: style.getPropertyValue("--canvas-text").trim() || "#1e293b",
    accent: style.getPropertyValue("--accent").trim() || "#2563eb",
    pareto: style.getPropertyValue("--pareto").trim() || "#e11d48",
    observation: style.getPropertyValue("--observation").trim() || "#d97706",
    point: style.getPropertyValue("--point").trim() || "#475569",
    band: style.getPropertyValue("--band").trim() || "rgba(37, 99, 235, 0.18)",
  };
  return _canvasColors;
}

// Re-render on theme change (invalidate color cache)
new MutationObserver(() => { _canvasColors = null; update(); }).observe(
  document.documentElement, { attributes: true, attributeFilter: ["data-theme"] }
);

// --- Invalidate scatter canvas (used by mobile toggle to fix empty canvas bug) ---
document.addEventListener("invalidate-scatter", () => {
  const canvas = document.getElementById("scatter-canvas");
  _canvasCache.delete(canvas);
  update();
});

// --- Test hook (gated behind ?test=1 to avoid leaking internals in prod) ---
// Exposes a tiny readonly view of internal state used only by Playwright specs.
if (typeof location !== "undefined" &&
    new URLSearchParams(location.search).get("test") === "1") {
  window.__test = {
    get currentComposition() { return currentComposition ? [...currentComposition] : null; },
    get displayPreviewComp() { return displayPreviewComp ? [...displayPreviewComp] : null; },
    // Whether the Material Source curve-level transition is currently in
    // flight. Used by the smooth-transition regression test in
    // `preview-curve.spec.ts` — it's the deterministic alternative to
    // racing screenshot timing against the 350 ms blend window.
    get isMsCurveTransitionActive() { return _msCurveTransition !== null; },
  };
}

// --- Start ---
// Surface model-load and init failures to the user instead of leaving
// the page in a silently-broken half-rendered state. A schema mismatch
// or missing artifact field throws hard inside ``initStrengthModel``
// (see docs/gp.mjs); without this catch the rejection becomes an
// unhandled error that no one notices.
init().catch(err => {
  console.error("[boxcrete] init failed:", err);
  const banner = document.createElement("div");
  banner.setAttribute("role", "alert");
  banner.style.cssText = (
    "position:fixed;top:0;left:0;right:0;z-index:99999;" +
    "background:#dc2626;color:#fff;padding:12px 16px;" +
    "font:14px/1.4 system-ui, sans-serif;text-align:center;"
  );
  banner.textContent = (
    "Model failed to load. Try a hard refresh; if this persists, " +
    "the model artefacts may be incompatible with the current explorer build. " +
    "See the browser console for the underlying error."
  );
  document.body.appendChild(banner);
});
