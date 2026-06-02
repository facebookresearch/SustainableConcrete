// Regression test: verify the JS-side predictions match the Python
// reference (stored in docs/model/test_vectors.json) for the V2
// strength GP model schema (gated multi-Matern + engineered features).
//
// Run with: node test/test_js_strength_v2.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

import {
  initStrengthModel, predictStrength,
} from "../docs/gp.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const strengthJsonPath = resolve(__dirname, "..", "docs", "model", "strength.json");
const testVectorsJsonPath = resolve(__dirname, "..", "docs", "model", "test_vectors.json");

console.log(`Loading ${strengthJsonPath} ...`);
const params = JSON.parse(readFileSync(strengthJsonPath, "utf-8"));
console.log(`  schema_version=${params.schema_version}, n_train=${params.n_train}, d_aug=${params.d_aug}`);
console.log(`  model: ${params.model_name}`);

console.log("Initializing model …");
const t0 = Date.now();
initStrengthModel(params);
console.log(`  init done in ${Date.now() - t0}ms`);

// UI compatibility checks: ui.mjs reads these v1-named fields after
// initStrengthModel(). They MUST be populated for predictions + std-bands
// to render. (Schema v2 stores them under different names; initStrengthModel
// is responsible for aliasing them so the UI can stay schema-agnostic.)
const requiredFields = ["noise_variance", "y_std", "y_mean", "prior_mean", "L_flat", "alpha_f64", "X_train_flat", "n"];
const missing = requiredFields.filter((k) => params[k] === undefined || params[k] === null);
if (missing.length > 0) {
  console.error(`❌ initStrengthModel did not populate: ${missing.join(", ")}`);
  process.exit(1);
}
console.log("✅ all required UI-compat fields populated by initStrengthModel");

console.log(`Loading ${testVectorsJsonPath} ...`);
const tvBundle = JSON.parse(readFileSync(testVectorsJsonPath, "utf-8"));
const testVectors = tvBundle.test_vectors;
console.log(`  ${testVectors.length} test vectors`);

let nFail = 0;
let maxRelErrMean = 0;
let maxRelErrVar = 0;
const tolMeanRel = 0.02;       // 2% relative for mean
const tolVarRel = 0.10;        // 10% relative for variance
const tolAbsZero = 0.5;        // psi at t=0 should be ~0
const constraintFails = [];

for (let i = 0; i < testVectors.length; i++) {
  const tv = testVectors[i];
  const { mean, variance } = predictStrength(tv.composition, tv.time, params);
  const expectedMean = tv.expected_mean;
  const expectedVar = tv.expected_variance;
  // Constraint check at t=0
  if (tv.time === 0.0) {
    if (Math.abs(mean) > tolAbsZero) {
      constraintFails.push({ i, mean, expected: 0 });
      nFail += 1;
      console.log(
        `  [FAIL t=0 constraint] idx=${i}: mean=${mean.toFixed(2)}, expected ≈ 0`
      );
    }
    continue;
  }
  // Numerical agreement check
  const relMean = Math.abs(mean - expectedMean) / Math.max(1.0, Math.abs(expectedMean));
  const relVar = Math.abs(variance - expectedVar) / Math.max(1e-3, Math.abs(expectedVar));
  maxRelErrMean = Math.max(maxRelErrMean, relMean);
  maxRelErrVar = Math.max(maxRelErrVar, relVar);
  if (relMean > tolMeanRel || relVar > tolVarRel) {
    nFail += 1;
    console.log(
      `  [FAIL] idx=${i}: t=${tv.time.toFixed(2)} mean=${mean.toFixed(0)} ` +
      `expected=${expectedMean.toFixed(0)} (rel ${(relMean * 100).toFixed(2)}%) | ` +
      `var=${variance.toFixed(0)} expected=${expectedVar.toFixed(0)} ` +
      `(rel ${(relVar * 100).toFixed(2)}%)`
    );
  }
}

console.log("\n--- Summary ---");
console.log(`Tests passed: ${testVectors.length - nFail}/${testVectors.length}`);
console.log(`Max relative error (mean): ${(maxRelErrMean * 100).toFixed(3)}%`);
console.log(`Max relative error (variance): ${(maxRelErrVar * 100).toFixed(3)}%`);
console.log(`Constraint fails (t=0 should be 0): ${constraintFails.length}`);

if (nFail > 0) {
  console.error(`\n❌ ${nFail} tests failed.`);
  process.exit(1);
} else {
  console.log("\n✅ All tests passed.");
  process.exit(0);
}
