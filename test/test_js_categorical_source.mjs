/**
 * Pins the model-side fact that makes interpolating Material Source a bug.
 *
 * The v5 strength GP handles Material Source with a Hamming
 * `CategoricalKernel`: `k(s_i, s_j) = 1` when the classes are equal, else the
 * learned `source_correlation`. It tests EQUALITY, not distance. So a value
 * that is not exactly one of the trained classes matches no training row at
 * all, and every row gets the same down-weight — the prediction collapses to
 * a single "unseen class" value that is not a blend of anything.
 *
 * Two UI bugs came from assuming otherwise (the composition animation and the
 * hover preview both used to lerp this dimension). This test states the
 * invariant directly, so if anyone later swaps in a smooth embedding kernel
 * this fails and they can revisit the pinning in the UI.
 *
 * Run: node test/test_js_categorical_source.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  initStrengthModel,
  initWASM,
  predictStrengthMeanOnly,
} from "../docs/gp.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const params = JSON.parse(
  readFileSync(join(ROOT, "docs/model/strength.json"), "utf8"),
);
const catalog = JSON.parse(
  readFileSync(join(ROOT, "docs/model/compositions.json"), "utf8"),
);

initStrengthModel(params);
await initWASM(params);

let failures = 0;
function check(name, cond, detail = "") {
  if (cond) {
    console.log(`  ok   ${name}`);
  } else {
    console.error(`  FAIL ${name}${detail ? " — " + detail : ""}`);
    failures++;
  }
}

const MS = catalog.column_names.indexOf("Material Source");
if (MS < 0) {
  console.error("Material Source column not found in compositions.json");
  process.exit(1);
}

const classes = [...new Set(catalog.compositions.map((c) => Math.round(c[MS])))].sort();
console.log(`Material Source is column ${MS}; trained classes: ${classes.join(", ")}`);

// Any real composition works; the source dim is the only thing we vary.
const base = [...catalog.compositions[0]];
const at = (v) => {
  const c = [...base];
  c[MS] = v;
  return predictStrengthMeanOnly(c, [28], params)[0];
};

console.log("\ncategorical source kernel");

// --- 1. Distinct classes give distinct predictions -------------------------
{
  const byClass = classes.map(at);
  const unique = new Set(byClass).size;
  check(
    "each trained class predicts a distinct value",
    unique === classes.length,
    `${unique} distinct values for ${classes.length} classes`,
  );
  classes.forEach((c, i) => console.log(`       class ${c}: ${byClass[i].toFixed(2)} psi`));
}

// --- 2. Every non-class value collapses to ONE value -----------------------
// This is the crux: there is no interpolation. 0.5 is not "between" classes
// 0 and 1 — it is simply "not any class", exactly like 1.5 or 42.
{
  const nonClass = [0.3, 0.5, 1.5, 1.98, 2.4, 42];
  const vals = nonClass.map(at);
  const allSame = vals.every((v) => v === vals[0]);
  check(
    "all non-class values collapse to a single prediction",
    allSame,
    `got ${JSON.stringify(vals.map((v) => +v.toFixed(4)))}`,
  );
  console.log(`       any non-class value: ${vals[0].toFixed(2)} psi`);

  const distinctFromClasses = classes.every((c) => at(c) !== vals[0]);
  check(
    "the collapsed value differs from every trained class",
    distinctFromClasses,
  );
}

// --- 3. A fractional value is NOT a blend of its neighbours ---------------
// The property that makes lerping visually wrong: a midpoint does not sit
// midway between the two classes it separates.
{
  let anyNonMonotonic = false;
  for (let i = 0; i + 1 < classes.length; i++) {
    const lo = at(classes[i]);
    const hi = at(classes[i + 1]);
    const mid = at((classes[i] + classes[i + 1]) / 2);
    const between = mid >= Math.min(lo, hi) && mid <= Math.max(lo, hi);
    const midpointOfEnds = (lo + hi) / 2;
    const off = Math.abs(mid - midpointOfEnds) / Math.max(1, Math.abs(midpointOfEnds));
    console.log(
      `       ${classes[i]} -> ${classes[i + 1]}: lo=${lo.toFixed(0)} mid=${mid.toFixed(0)} ` +
        `hi=${hi.toFixed(0)}  (${(100 * off).toFixed(1)}% off a true midpoint)`,
    );
    if (!between) anyNonMonotonic = true;
    check(
      `class ${classes[i]}->${classes[i + 1]}: midpoint is not the average of the endpoints`,
      off > 0.01,
      `only ${(100 * off).toFixed(2)}% off — suspiciously blend-like`,
    );
  }
  check(
    "at least one midpoint falls outside its endpoints entirely",
    anyNonMonotonic,
    "every midpoint happened to land between its neighbours",
  );
}

if (failures > 0) {
  console.error(`\n${failures} check(s) failed.`);
  process.exit(1);
}
console.log("\nAll categorical-source checks passed.");
