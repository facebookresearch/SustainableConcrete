/**
 * Post-process docs/model/test_vectors.json to add GWP/cost fields for
 * each test vector. Called immediately after experiments/regenerate_strength_json.py
 * has written the strength-related fields. The GWP and cost models are
 * unchanged from before; this just computes their predictions on the
 * new test vectors so the legacy test_js_gp.mjs (which exercises all
 * three predictors together) continues to pass.
 *
 * Run: node experiments/augment_test_vectors_with_gwp_cost.mjs
 */

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

import { predictGWP, predictCost } from "../docs/gp.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const TV_PATH = resolve(__dirname, "..", "docs", "model", "test_vectors.json");
const GWP_PATH = resolve(__dirname, "..", "docs", "model", "gwp.json");
const COST_PATH = resolve(__dirname, "..", "docs", "model", "cost.json");

const tv = JSON.parse(readFileSync(TV_PATH, "utf-8"));
const gwpParams = JSON.parse(readFileSync(GWP_PATH, "utf-8"));
const costParams = JSON.parse(readFileSync(COST_PATH, "utf-8"));

// The GWP model is class-indexed by the material-source column. Read
// the column index from gwp.json (single source of truth) rather than
// hardcoding ``composition[7]``; the previous version hardcoded the
// index AND used a ``>= 0.5`` threshold to collapse the value, which
// silently maps any future class >= 1 to class 1. ``Math.round`` is
// the canonical truncation for a categorical encoding (matches what
// the Python side does via ``int(...)``).
const classDim = gwpParams.class_dim;
if (typeof classDim !== "number") {
  throw new Error(
    `gwp.json::class_dim must be a number; got ${classDim} ` +
      `(${typeof classDim})`,
  );
}
const validClasses = new Set(Object.keys(gwpParams.coefficients));

console.log(`Augmenting ${tv.test_vectors.length} test vectors with GWP/cost…`);
for (const v of tv.test_vectors) {
  const composition = v.composition || v.input;
  const materialSource = Math.round(composition[classDim]);
  if (!validClasses.has(String(materialSource))) {
    throw new Error(
      `Test vector has material-source class ${materialSource} ` +
        `(from composition[${classDim}] = ${composition[classDim]}), ` +
        `but gwp.json only knows classes [${[...validClasses].join(", ")}]. ` +
        `Either extend gwp.json to cover the new class or filter the test vector.`,
    );
  }
  const gwp = predictGWP(composition, gwpParams, materialSource);
  const cost = predictCost(composition, costParams);
  v.gwp_mean = gwp.mean;
  v.gwp_variance = gwp.variance;
  v.cost_mean = cost.mean;
  v.cost_variance = cost.variance;
}

writeFileSync(TV_PATH, JSON.stringify(tv, null, 2));
console.log(`Wrote ${TV_PATH}`);
