/**
 * Unit tests for docs/filters.mjs.
 *
 * This logic previously lived inside a 400-line `setupEventListeners` closure
 * in ui.mjs, so it was reachable only through Playwright. The predicate is
 * where the correctness lives — particularly the categorical branch, since
 * Material Source is unordered and a min/max range is meaningless for it.
 *
 * Run: node test/test_js_filters.mjs
 */

import { makeComputedFilters, matchesFilters, colIdxStrict } from "../docs/filters.mjs";

let failures = 0;
function check(name, cond, detail = "") {
  if (cond) console.log(`  ok   ${name}`);
  else { console.error(`  FAIL ${name}${detail ? " — " + detail : ""}`); failures++; }
}
const close = (a, b, tol = 1e-9) => Math.abs(a - b) <= tol;

const COLS = [
  "Cement (kg/m3)", "Fly Ash (kg/m3)", "Slag (kg/m3)", "Water (kg/m3)",
  "HRWR (kg/m3)", "Fine Aggregate (kg/m3)", "Coarse Aggregates (kg/m3)",
  "Material Source", "Temp (C)",
];
const MS = 7;
// Mix C28: cement 263, slag 108, water 130, hrwr 2.04, fine 867, coarse 1129, class 2
const C28 = [263, 0, 108, 130, 2.04, 867, 1129, 2, 22];
const M1 = [353.33, 46.67, 266.67, 140, 6.67, 1833, 0, 0, 22];

const cf = makeComputedFilters(COLS);
const by = (id) => cf.find((c) => c.id === id).compute;

console.log("makeComputedFilters");
{
  check("exposes wb / binder / scm / paste",
    cf.map((c) => c.id).join(",") === "wb,binder,scm,paste", cf.map((c) => c.id).join(","));
  check("total binder", close(by("binder")(C28), 371));
  check("w/b ratio", close(by("wb")(C28), 130 / 371));
  check("SCM replacement %", close(by("scm")(C28), (108 / 371) * 100));
  const paste = 371 + 130;
  check("paste fraction", close(by("paste")(C28), paste / (paste + 1129 + 867)));

  // Degenerate inputs must not produce NaN — a NaN silently disables a filter
  // because every comparison against it is false.
  const zero = [0, 0, 0, 0, 0, 0, 0, 0, 22];
  check("zero binder -> wb is Infinity, not NaN", by("wb")(zero) === Infinity);
  check("zero binder -> scm is 0, not NaN", by("scm")(zero) === 0);
  check("zero mass -> paste is 0, not NaN", by("paste")(zero) === 0);
}

console.log("\ncolIdxStrict");
{
  check("resolves a known column", colIdxStrict(COLS, "Material Source") === MS);
  let threw = false;
  try { colIdxStrict(COLS, "Nope (kg/m3)"); } catch { threw = true; }
  check("throws on an unknown column rather than returning -1", threw);
}

console.log("\nmatchesFilters — numeric");
{
  check("null filters match everything", matchesFilters(C28, null) === true);
  check("empty filters match everything", matchesFilters(C28, []) === true);
  check("in-range passes", matchesFilters(C28, [{ colIdx: 0, min: 200, max: 300 }]) === true);
  check("below min fails", matchesFilters(C28, [{ colIdx: 0, min: 300, max: 400 }]) === false);
  check("above max fails", matchesFilters(C28, [{ colIdx: 0, min: 0, max: 100 }]) === false);
  check("bounds are inclusive",
    matchesFilters(C28, [{ colIdx: 0, min: 263, max: 263 }]) === true);
  check("computed filters are supported",
    matchesFilters(C28, [{ computed: by("binder"), min: 370, max: 372 }]) === true);
  check("multiple filters are ANDed",
    matchesFilters(C28, [
      { colIdx: 0, min: 200, max: 300 },
      { computed: by("binder"), min: 0, max: 100 },
    ]) === false);
}

console.log("\nmatchesFilters — categorical");
{
  const only2 = [{ colIdx: MS, classes: new Set([2]) }];
  check("member class passes", matchesFilters(C28, only2) === true);
  check("non-member class fails", matchesFilters(M1, only2) === false);

  const all = [{ colIdx: MS, classes: new Set([0, 1, 2]) }];
  check("all classes selected is a no-op (C28)", matchesFilters(C28, all) === true);
  check("all classes selected is a no-op (M1)", matchesFilters(M1, all) === true);

  const none = [{ colIdx: MS, classes: new Set() }];
  check("no classes selected matches nothing", matchesFilters(C28, none) === false);

  // The class value is rounded before lookup: mid-animation compositions can
  // carry a non-integral source, and Set membership would silently miss.
  const almost2 = [...C28]; almost2[MS] = 1.98;
  check("a near-integer class rounds to its class", matchesFilters(almost2, only2) === true);

  // A categorical filter must NOT be treated as a range.
  check("categorical ignores min/max if both are present",
    matchesFilters(M1, [{ colIdx: MS, classes: new Set([0]), min: 2, max: 2 }]) === true);
}

if (failures > 0) {
  console.error(`\n${failures} check(s) failed.`);
  process.exit(1);
}
console.log("\nAll filter checks passed.");
