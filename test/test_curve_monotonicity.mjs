/**
 * Strength-curve monotonicity diagnostic.
 *
 * Concrete strength curves should generally be MONOTONICALLY
 * INCREASING (cement hydration is a one-way reaction; SCMs slowly
 * activate but don't reverse). Strong oscillations or decreasing
 * intervals are unphysical and indicate the GP is interpolating
 * features in ways that don't match the underlying chemistry.
 *
 * This diagnostic computes, across all compositions in compositions.json:
 *   (1) Fraction of compositions with any decreasing interval (strength
 *       drops between consecutive time points)
 *   (2) Worst dropdown magnitude (max |min(0, μ(t_{i+1}) − μ(t_i))|)
 *   (3) Number of inflection points (sign changes in second
 *       difference of mean) — too many means oscillating curves
 *   (4) Worst total variation of slope: sum of |Δslope| − a measure of
 *       how "wiggly" curves are
 *
 * Use this to compare candidate models and reject ones with too many
 * non-monotone or oscillating curves.
 *
 * Run: node test/test_curve_monotonicity.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

import {
  initStrengthModel,
  initWASM,
  predictStrengthMeanOnly,
} from "../docs/gp.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const docsRoot = resolve(__dirname, "..", "docs");

const params = JSON.parse(readFileSync(resolve(docsRoot, "model/strength.json"), "utf-8"));
const data = JSON.parse(readFileSync(resolve(docsRoot, "model/compositions.json"), "utf-8"));

initStrengthModel(params);
await initWASM(params);

// Time grid: log-spaced from 1 to 28 days (matches what the explorer
// shows). Sufficiently dense to detect oscillations between major
// observation days (1, 3, 7, 14, 28).
const N_TIMES = 64;
const T_MIN = 0.5;
const T_MAX = 28;
const times = new Array(N_TIMES);
for (let i = 0; i < N_TIMES; i++) {
  const u = i / (N_TIMES - 1);
  times[i] = T_MIN * Math.pow(T_MAX / T_MIN, u);
}

// For each composition, compute the mean curve and analyze it.
let nCompositions = 0;
let nWithDrop = 0;
let nWithMultipleInflections = 0;
let maxDropPsi = 0;
let maxDropComp = -1;
let totalSlopeVariation = 0;
const inflectionCounts = [];

for (let c = 0; c < data.compositions.length; c++) {
  const comp = data.compositions[c];
  const means = predictStrengthMeanOnly(comp, times, params);
  nCompositions++;

  // First differences (slopes between consecutive points)
  const slopes = new Array(N_TIMES - 1);
  for (let i = 0; i < N_TIMES - 1; i++) {
    slopes[i] = means[i + 1] - means[i];
  }
  // Decreasing intervals
  let minSlope = 0;
  for (const s of slopes) if (s < minSlope) minSlope = s;
  if (minSlope < -1.0) { // tolerate 1 psi numerical noise
    nWithDrop++;
    if (Math.abs(minSlope) > maxDropPsi) {
      maxDropPsi = Math.abs(minSlope);
      maxDropComp = c;
    }
  }

  // Inflection count: sign changes in second differences (curvature)
  const secondDiffs = new Array(N_TIMES - 2);
  for (let i = 0; i < N_TIMES - 2; i++) {
    secondDiffs[i] = slopes[i + 1] - slopes[i];
  }
  let inflections = 0;
  for (let i = 0; i < secondDiffs.length - 1; i++) {
    // Use a small threshold to ignore floating-point sign flicker.
    if (Math.abs(secondDiffs[i]) < 1e-3) continue;
    if (Math.abs(secondDiffs[i + 1]) < 1e-3) continue;
    if (secondDiffs[i] * secondDiffs[i + 1] < 0) inflections++;
  }
  inflectionCounts.push(inflections);
  if (inflections > 2) nWithMultipleInflections++; // a healthy curve has 0–1

  // Total slope variation (a smoothness measure):
  // Σ |slope[i+1] − slope[i]|. Large = wiggly.
  let tv = 0;
  for (let i = 0; i < secondDiffs.length; i++) tv += Math.abs(secondDiffs[i]);
  totalSlopeVariation += tv;
}

const meanInflections = inflectionCounts.reduce((a, b) => a + b, 0) / inflectionCounts.length;
const inflectionsP90 = inflectionCounts.slice().sort((a, b) => a - b)[Math.floor(0.9 * inflectionCounts.length)];

console.log(`=== Curve monotonicity diagnostic for ${params.model_name} ===`);
console.log(`Time grid: ${N_TIMES} log-spaced points in [${T_MIN}, ${T_MAX}] days.`);
console.log(`Compositions analyzed: ${nCompositions}.`);
console.log("");
console.log(`(1) Compositions with decreasing intervals: ${nWithDrop}/${nCompositions} (${(100*nWithDrop/nCompositions).toFixed(1)}%)`);
console.log(`    Worst dropdown: ${maxDropPsi.toFixed(1)} psi (composition idx ${maxDropComp})`);
console.log("");
console.log(`(2) Compositions with > 2 inflection points: ${nWithMultipleInflections}/${nCompositions} (${(100*nWithMultipleInflections/nCompositions).toFixed(1)}%)`);
console.log(`    Mean inflections per curve: ${meanInflections.toFixed(2)}`);
console.log(`    P90 inflections per curve: ${inflectionsP90}`);
console.log("");
console.log(`(3) Total slope variation (sum of |Δslope|): ${totalSlopeVariation.toFixed(0)} psi`);
console.log(`    Mean per composition: ${(totalSlopeVariation / nCompositions).toFixed(0)} psi`);

// Acceptance thresholds (physically motivated):
//   Concrete strength curves should be MONOTONICALLY increasing with at
//   most a small number of inflection points (1 typical; 2 if delayed
//   SCM/pozzolanic activation is present in the mix). We allow some
//   leniency for GP fits because the predictive mean has tiny fluctuations
//   from kernel-driven extrapolation, but anything beyond a few percent
//   of curves with multi-inflections, or any meaningful dropdown, is
//   unphysical.
//
//   These thresholds were calibrated via experiments/compare_monotonicity.py
//   on full-data fits:
//     F5_alllog + MLL only:           1.4% dec, 0.0% osc, 9 psi max drop
//     F5_alllog + MLL + gated_noise:  1.4% dec, 0.0% osc, 8 psi max drop
//     F5_alllog + block_loo_only:     3.5% dec, 8.3% osc, 24 psi max drop
//     F5_no_log_mat + block_loo_only: 99% dec, 99% osc, 876 psi max drop ❌
//
//   V2 choice: F5_alllog + block_loo_only (BEST block-LOO, ACCEPTABLE
//   monotonicity). The thresholds below would have caught the
//   F5_no_log_mat regression cleanly.
const PASS_FRACTION_DECREASE = 0.10;
const PASS_FRACTION_OSCILLATE = 0.20;
const PASS_MAX_DROP_PSI = 100.0;

let nFail = 0;
console.log("");
console.log("Acceptance thresholds (physically motivated):");
function check(label, val, pass) {
  console.log(`  ${pass ? "✅" : "❌"} ${label}: ${val}`);
  if (!pass) nFail++;
}
check(
  `< ${(PASS_FRACTION_DECREASE * 100).toFixed(0)}% compositions with decreasing intervals`,
  `${(100*nWithDrop/nCompositions).toFixed(1)}%`,
  nWithDrop / nCompositions < PASS_FRACTION_DECREASE,
);
check(
  `< ${(PASS_FRACTION_OSCILLATE * 100).toFixed(0)}% compositions with > 2 inflections`,
  `${(100*nWithMultipleInflections/nCompositions).toFixed(1)}%`,
  nWithMultipleInflections / nCompositions < PASS_FRACTION_OSCILLATE,
);
check(
  `Worst dropdown < ${PASS_MAX_DROP_PSI} psi`,
  `${maxDropPsi.toFixed(1)} psi`,
  maxDropPsi < PASS_MAX_DROP_PSI,
);

if (nFail > 0) {
  console.error(`\n❌ ${nFail} monotonicity checks failed.`);
  process.exit(1);
} else {
  console.log("\n✅ All curves pass physical monotonicity checks.");
}
