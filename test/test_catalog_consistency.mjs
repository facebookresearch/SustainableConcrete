/**
 * Catalog consistency tests: ensure docs/model/compositions.json is
 * anchored to the canonical v5 raw data (data/boxcrete_data.csv).
 *
 * These tests exist specifically to catch the class of bug where the
 * catalog silently drifts from the canonical data — e.g. v5 introduces
 * a new Material Source class 2 but the catalog still uses pre-v5
 * binary labelling. The freshness test (test_data_freshness.mjs)
 * verifies INTERNAL consistency (catalog ↔ predictions) but never
 * cross-references with raw data; this gap let an 83/144-row label
 * error pass undetected through the v5 migration.
 *
 * What we check:
 *   1. Every catalog composition matches a row in data/boxcrete_data.csv
 *      (by composition fingerprint: 7-column rounded tuple).
 *   2. The Material Source label in the catalog matches what the raw
 *      data says for that composition.
 *   3. The catalog's Material Source distribution covers every class
 *      present in the raw data (non-empty per-class bucket).
 *   4. The catalog's Material Source range fits within the model's
 *      declared bounds (slider_bounds[Material Source]).
 *   5. slider_bounds[Material Source].max matches the maximum class
 *      index actually present in the raw v5 data (catches stale
 *      pre-v5 bounds: max=1 when v5 has 3 classes).
 *
 * Run: node test/test_catalog_consistency.mjs
 */

import { readFileSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, "..");

const compositions = JSON.parse(
  readFileSync(resolve(REPO_ROOT, "docs", "model", "compositions.json"), "utf-8"),
);
const dataCsv = readFileSync(resolve(REPO_ROOT, "data", "boxcrete_data.csv"), "utf-8");

// --- Parse CSV into rows (header-aware) ---
const lines = dataCsv.trim().split("\n");
const header = lines[0].split(",");
const COMP_COLS = [
  "Cement (kg/m3)",
  "Fly Ash (kg/m3)",
  "Slag (kg/m3)",
  "Water (kg/m3)",
  "HRWR (kg/m3)",
  "Fine Aggregate (kg/m3)",
  "Coarse Aggregates (kg/m3)",
];
const colIdx = (name) => {
  const i = header.indexOf(name);
  if (i < 0) throw new Error(`column '${name}' not in raw data header: ${header}`);
  return i;
};
const COMP_IDX = COMP_COLS.map(colIdx);
const MS_IDX = colIdx("Material Source");

// Build fingerprint lookup over raw rows.
const fingerprint = (row, idxs = COMP_IDX) =>
  idxs.map((j) => Math.round(Number(row[j]) * 10) / 10).join("|");
const v5Lookup = new Map(); // fingerprint -> Material Source
for (let li = 1; li < lines.length; li++) {
  const row = lines[li].split(",");
  if (row.length < header.length) continue;
  const fp = fingerprint(row);
  const ms = Math.round(Number(row[MS_IDX]));
  if (!v5Lookup.has(fp)) v5Lookup.set(fp, ms);
}

const catColNames = compositions.column_names;
const catMsIdx = catColNames.indexOf("Material Source");
const catCompIdxs = COMP_COLS.map((c) => catColNames.indexOf(c));
if (catCompIdxs.some((i) => i < 0)) {
  throw new Error(
    `compositions.column_names missing one of: ${COMP_COLS.join(", ")}`,
  );
}

let nPass = 0;
let nFail = 0;
const failures = [];
function check(cond, msg) {
  if (cond) nPass++;
  else {
    nFail++;
    failures.push(msg);
  }
}

// --- (1) Every catalog composition matches a raw v5 row ---
{
  let nMissing = 0;
  for (let i = 0; i < compositions.compositions.length; i++) {
    const c = compositions.compositions[i];
    const fp = catCompIdxs.map((j) => Math.round(c[j] * 10) / 10).join("|");
    if (!v5Lookup.has(fp)) {
      nMissing++;
      if (nMissing <= 3) {
        check(
          false,
          `compositions[${i}] = [${COMP_COLS.map((col, k) => `${col.split(" ")[0]}=${c[catCompIdxs[k]]}`).join(", ")}] not found in data/boxcrete_data.csv`,
        );
      }
    }
  }
  check(
    nMissing === 0,
    `${nMissing} catalog compositions have no matching row in data/boxcrete_data.csv`,
  );
}

// --- (2) Catalog Material Source labels match raw data ---
{
  let nMismatch = 0;
  const examples = [];
  for (let i = 0; i < compositions.compositions.length; i++) {
    const c = compositions.compositions[i];
    const fp = catCompIdxs.map((j) => Math.round(c[j] * 10) / 10).join("|");
    const raw_ms = v5Lookup.get(fp);
    if (raw_ms === undefined) continue;
    const cat_ms = Math.round(c[catMsIdx]);
    if (cat_ms !== raw_ms) {
      nMismatch++;
      if (examples.length < 3) {
        examples.push(
          `compositions[${i}]: catalog Material Source=${cat_ms}, raw data Material Source=${raw_ms}`,
        );
      }
    }
  }
  for (const e of examples) check(false, e);
  check(
    nMismatch === 0,
    `${nMismatch}/${compositions.compositions.length} catalog Material Source labels disagree with data/boxcrete_data.csv`,
  );
}

// --- (3) Catalog covers every class present in raw v5 data ---
{
  const rawClasses = new Set(Array.from(v5Lookup.values()));
  const catClasses = new Set(
    compositions.compositions.map((c) => Math.round(c[catMsIdx])),
  );
  for (const c of rawClasses) {
    check(
      catClasses.has(c),
      `catalog has no compositions at Material Source=${c} (raw data has class ${c})`,
    );
  }
}

// --- (4) Catalog Material Source range within slider bounds ---
{
  const bounds = compositions.slider_bounds["Material Source"];
  for (let i = 0; i < compositions.compositions.length; i++) {
    const ms = compositions.compositions[i][catMsIdx];
    if (ms < bounds.min - 1e-6 || ms > bounds.max + 1e-6) {
      check(
        false,
        `compositions[${i}].Material Source=${ms} outside slider_bounds [${bounds.min}, ${bounds.max}]`,
      );
    }
  }
  check(true, "catalog Material Source values within slider_bounds");
}

// --- (5) slider_bounds[Material Source].max matches raw-data max ---
{
  const bounds = compositions.slider_bounds["Material Source"];
  const rawMax = Math.max(...Array.from(v5Lookup.values()));
  check(
    Math.round(bounds.max) === rawMax,
    `slider_bounds[Material Source].max=${bounds.max} but raw data max=${rawMax} (catches stale pre-v5 bounds)`,
  );
  check(
    Math.round(bounds.min) === 0,
    `slider_bounds[Material Source].min=${bounds.min} but data uses 0-indexed classes`,
  );
}

const total = nPass + nFail;
console.log(`\nCatalog consistency: ${nPass}/${total} assertions passed`);
if (nFail > 0) {
  console.error(`\n❌ ${nFail} assertions FAILED:`);
  for (const m of failures.slice(0, 8)) console.error(`  - ${m}`);
  if (failures.length > 8) console.error(`  …and ${failures.length - 8} more`);
  process.exit(1);
}
console.log("✅ docs/model/compositions.json is consistent with data/boxcrete_data.csv");
