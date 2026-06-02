/**
 * Port-equivalence test: verifies the JavaScript GP implementation
 * (in docs/gp.mjs, fed by docs/model/strength.json) produces the same
 * predictions as the Python reference (the expected_mean / expected_variance
 * values baked into docs/model/test_vectors.json by
 * experiments/regenerate_strength_json.py).
 *
 * See docs/model/README.md for schema + regen workflow.
 *
 * Run: node test/test_js_gp.mjs
 */

import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, "..");

// Import GP library
const gpModule = await import(join(REPO_ROOT, "docs", "gp.mjs"));
const { predictStrength, predictGWP, initStrengthModel } = gpModule;

// Load model parameters
const strengthParams = JSON.parse(
  readFileSync(join(REPO_ROOT, "docs", "model", "strength.json"), "utf-8")
);
const gwpParams = JSON.parse(
  readFileSync(join(REPO_ROOT, "docs", "model", "gwp.json"), "utf-8")
);
const testData = JSON.parse(
  readFileSync(join(REPO_ROOT, "docs", "model", "test_vectors.json"), "utf-8")
);

// Initialize model (compute Cholesky + alpha from X_train + kernel params)
console.time("initStrengthModel");
initStrengthModel(strengthParams);
console.timeEnd("initStrengthModel");

const RTOL = 1e-4;
const ATOL = 1e-2; // absolute tolerance for values near zero

function assertClose(actual, expected, name) {
  const absErr = Math.abs(actual - expected);
  const relErr = Math.abs(expected) > ATOL ? absErr / Math.abs(expected) : absErr;
  if (relErr > RTOL && absErr > ATOL) {
    throw new Error(
      `${name}: expected ${expected}, got ${actual} ` +
        `(relErr=${relErr.toExponential(2)}, absErr=${absErr.toExponential(2)})`
    );
  }
}

let passed = 0;
let failed = 0;

for (const [idx, vec] of testData.test_vectors.entries()) {
  const composition = vec.input;

  // Test GWP prediction
  try {
    // Determine material source from composition (class_dim in gwp model).
    // Use the same strict ``typeof === "number"`` check as the regen
    // script (``experiments/augment_test_vectors_with_gwp_cost.mjs``):
    // ``!== null`` would let ``undefined`` slip through and produce
    // ``composition[undefined] = undefined``, then ``Math.round(undefined)
    // = NaN``, then a confusing class-lookup failure downstream.
    const materialSource = typeof gwpParams.class_dim === "number"
      ? Math.round(composition[gwpParams.class_dim])
      : 0;
    const gwp = predictGWP(composition, gwpParams, materialSource);
    assertClose(gwp.mean, vec.gwp_mean, `vec[${idx}] GWP mean`);
    assertClose(gwp.variance, vec.gwp_variance, `vec[${idx}] GWP variance`);
    passed += 2;
  } catch (e) {
    console.error(e.message);
    failed += 2;
  }

  // Test strength predictions at each day
  for (const day of testData.strength_days) {
    try {
      const pred = predictStrength(composition, day, strengthParams);
      const ref = vec.strength[String(day)];
      assertClose(pred.mean, ref.mean, `vec[${idx}] day-${day} strength mean`);
      assertClose(
        pred.variance,
        ref.variance,
        `vec[${idx}] day-${day} strength variance`
      );
      passed += 2;
    } catch (e) {
      console.error(e.message);
      failed += 2;
    }
  }
}

const total = passed + failed;
console.log(`\nJS GP sync test: ${passed}/${total} assertions passed`);

if (failed > 0) {
  console.error(`\n❌ ${failed} assertions FAILED`);
  process.exit(1);
} else {
  console.log("✅ All assertions passed — JS matches Python reference");
}
