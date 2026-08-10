/**
 * Scatter-filter logic, extracted from ui.mjs so it can be unit tested.
 *
 * These functions used to live inside `setupEventListeners` — a 400-line
 * closure — purely so they could capture `colNames`. That made the only part
 * of the filter subsystem with real correctness content (the predicate, and
 * the derived-quantity formulas) reachable only through Playwright. Both are
 * pure given their inputs, so they belong here with a fast node test.
 *
 * See test/test_js_filters.mjs.
 */

/**
 * Resolve a column name to its index, throwing loudly on a miss.
 *
 * A silent fallback here would be dangerous: `indexOf` returning -1 and being
 * used as an index yields `undefined`, which makes every numeric comparison
 * false and quietly disables the filter instead of failing.
 */
export function colIdxStrict(colNames, name) {
  const i = colNames.indexOf(name);
  if (i < 0) {
    throw new Error(
      `filters: column ${JSON.stringify(name)} not found in ${JSON.stringify(colNames)}. ` +
      "Column names must match DEFAULT_X_COLUMNS in boxcrete/utils.py.",
    );
  }
  return i;
}

/**
 * Derived quantities offered in the filter dropdown alongside raw columns.
 * Each `compute` takes a raw composition row and returns a scalar.
 *
 * NOTE: `paste` divides by a 6-term total that omits HRWR, whereas Python's
 * `_TOTAL_MASS_NAMES` (boxcrete/utils.py) includes it. HRWR is at most ~13 of
 * ~2400 kg/m3, so the two differ by well under 1%, but they are not the same
 * definition. Preserved as-is here to keep this extraction behaviour-neutral.
 *
 * @param {string[]} colNames - composition column names, in catalog order.
 */
export function makeComputedFilters(colNames) {
  const iCement = colIdxStrict(colNames, "Cement (kg/m3)");
  const iFlyAsh = colIdxStrict(colNames, "Fly Ash (kg/m3)");
  const iSlag = colIdxStrict(colNames, "Slag (kg/m3)");
  const iWater = colIdxStrict(colNames, "Water (kg/m3)");
  const iCoarse = colIdxStrict(colNames, "Coarse Aggregates (kg/m3)");
  const iFine = colIdxStrict(colNames, "Fine Aggregate (kg/m3)");

  const binderOf = (c) => c[iCement] + c[iFlyAsh] + c[iSlag];

  return [
    {
      id: "wb",
      label: "W/B Ratio",
      compute: (c) => {
        const b = binderOf(c);
        return b > 0 ? c[iWater] / b : Infinity;
      },
    },
    { id: "binder", label: "Total Binder", compute: binderOf },
    {
      id: "scm",
      label: "SCM Replacement %",
      compute: (c) => {
        const b = binderOf(c);
        return b > 0 ? ((c[iFlyAsh] + c[iSlag]) / b) * 100 : 0;
      },
    },
    {
      id: "paste",
      label: "Paste Fraction",
      compute: (c) => {
        const paste = binderOf(c) + c[iWater];
        const total = paste + c[iCoarse] + c[iFine];
        return total > 0 ? paste / total : 0;
      },
    },
  ];
}

/**
 * Does `comp` satisfy every active filter?
 *
 * A filter is either
 *   { colIdx, classes: Set<number> }        categorical: class membership
 *   { colIdx | computed, min, max }         numeric: inclusive bounds
 *
 * Categorical columns (Material Source) are unordered, so a min/max range is
 * meaningless for them — "between Source A and Source B" says nothing. They
 * test set membership on the rounded class instead.
 *
 * @param {number[]} comp - one composition row.
 * @param {Array|null} filters - active filter specs; null/empty means "match all".
 * @returns {boolean} true when the point should stay visible.
 */
export function matchesFilters(comp, filters) {
  if (!filters || filters.length === 0) return true;
  for (const f of filters) {
    const val = f.computed ? f.computed(comp) : comp[f.colIdx];
    if (f.classes) {
      if (!f.classes.has(Math.round(val))) return false;
    } else if (val < f.min || val > f.max) {
      return false;
    }
  }
  return true;
}
