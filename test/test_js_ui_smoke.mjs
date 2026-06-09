/**
 * UI smoke test: mimics what `docs/ui.mjs` actually does to render the
 * strength curve in the browser. Catches the `params.noise_variance`-style
 * "the test passed but the UI is broken" bugs by exercising the SAME
 * code path the explorer uses, including the std-band computation that
 * reads `params.noise_variance` and `params.y_std` after init.
 *
 * Run: node test/test_js_ui_smoke.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

import {
  initStrengthModel,
  initWASM,
  predictStrengthCurve,
  predictStrengthMeanOnly,
  predictGWP,
  predictCost,
} from "../docs/gp.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const docsRoot = resolve(__dirname, "..", "docs");
const params = JSON.parse(readFileSync(resolve(docsRoot, "model/strength.json"), "utf-8"));
const gwpParams = JSON.parse(readFileSync(resolve(docsRoot, "model/gwp.json"), "utf-8"));
const costParams = JSON.parse(readFileSync(resolve(docsRoot, "model/cost.json"), "utf-8"));
const compositions = JSON.parse(
  readFileSync(resolve(docsRoot, "model/compositions.json"), "utf-8"),
).compositions;

console.log(`Loaded params (schema_version=${params.schema_version}), ${compositions.length} compositions.`);
initStrengthModel(params);
const wasmOk = await initWASM(params);
console.log(`initStrengthModel ✓, initWASM ${wasmOk ? "✓" : "(skipped)"}`);

// Mirror ui.mjs's `computeStds` helper: it reads params.noise_variance + y_std.
function computeStds(variances, p) {
  const noiseVar = p.noise_variance * p.y_std * p.y_std;
  return variances.map((v) => Math.sqrt(v + noiseVar));
}

// Iterate every composition and exercise the predict* functions exactly
// as ui.mjs does. Any NaN / non-finite output is a regression.
const times = Array.from({ length: 64 }, (_, i) =>
  Math.exp((i / 63) * Math.log(28)),
);
let nFail = 0;
let maxStd = 0;
for (let i = 0; i < compositions.length; i++) {
  const composition = compositions[i];
  const { means, variances } = predictStrengthCurve(composition, times, params);
  const stds = computeStds(variances, params);
  for (let ti = 0; ti < times.length; ti++) {
    if (!Number.isFinite(means[ti]) || !Number.isFinite(variances[ti]) || !Number.isFinite(stds[ti])) {
      nFail++;
      if (nFail <= 5) {
        console.error(
          `❌ comp ${i} time ${times[ti].toFixed(2)}: ` +
          `mean=${means[ti]} variance=${variances[ti]} std=${stds[ti]}`
        );
      }
    }
    if (stds[ti] > maxStd) maxStd = stds[ti];
  }
  // Also exercise mean-only predictor (used for the preview curve).
  const previewMeans = predictStrengthMeanOnly(composition, times, params);
  for (let ti = 0; ti < times.length; ti++) {
    if (!Number.isFinite(previewMeans[ti])) {
      nFail++;
      if (nFail <= 5) {
        console.error(`❌ comp ${i} time ${times[ti].toFixed(2)}: meanOnly=${previewMeans[ti]}`);
      }
    }
  }
  // Material source for GWP (composition[7] is 0 or 1).
  const ms = composition[7] >= 0.5 ? 1 : 0;
  const gwp = predictGWP(composition, gwpParams, ms);
  const cost = predictCost(composition, costParams);
  if (!Number.isFinite(gwp.mean) || !Number.isFinite(cost.mean)) {
    nFail++;
    if (nFail <= 5) {
      console.error(`❌ comp ${i}: gwp=${gwp.mean} cost=${cost.mean}`);
    }
  }
}

console.log(`\nUI smoke test: ${compositions.length} compositions × ${times.length} times`);
console.log(`Max std band displayed: ${maxStd.toFixed(0)} psi`);
if (nFail > 0) {
  console.error(`❌ ${nFail} non-finite outputs.`);
  process.exit(1);
} else {
  console.log("✅ All UI predictions are finite — explorer should render.");
}
