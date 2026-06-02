/**
 * Engineered-feature registry for the strength GP.
 *
 * Each feature is computed from the RAW 10-dim composition vector
 * [Cement, Fly Ash, Slag, Water, HRWR, Fine, Coarse, Source, Temp, Time].
 * The same offsets (+1.0, +1e-3, +1e-4) used by the Python builders
 * apply here to match the GP's training-time numerical conditioning.
 *
 * Two callers consume this:
 *   - gp.mjs's `transformInput` (general path, uses `appendFeatures`)
 *   - gp_v2_fast.mjs's batched curve predictor (which inlines for speed,
 *     so it imports `FEATURE_FNS` and walks the names array directly).
 *
 * Adding a new feature: add it to `FEATURE_FNS`. The strength.json's
 * `engineered_feature_names` field is the source of truth for which
 * features are active in the deployed model.
 */

// Helper accessors for the raw composition vector.
const I_CEMENT = 0, I_FLYASH = 1, I_SLAG = 2, I_WATER = 3;
const I_HRWR = 4, I_FINE = 5, I_COARSE = 6;
// I_SOURCE = 7 (not a feature)
const I_TEMP = 8, I_TIME = 9;

/**
 * Map: feature_name → (xRaw: number[10]) → number.
 *
 * Source-of-truth: the V2 deployed builders in
 * `boxcrete/features.py::FEATURE_BUILDERS` (the 7 names in
 * `engineered_feature_names`). The deployed model consumes the
 * names it advertises in `engineered_feature_names`; additional
 * builders below (e.g., `wb_ratio`, `wc_ratio`) are kept for
 * forward-compat with any future strength.json that ships a non-
 * default `engineered_feature_names` list — the explorer would
 * resolve those names without a redeploy. Where Python uses
 * `x[..., _IDX["water"]:_IDX["water"]+1]`, we read `xRaw[I_WATER]`.
 */
export const FEATURE_FNS = {
  wb_ratio: (x) => x[I_WATER] / (x[I_CEMENT] + x[I_FLYASH] + x[I_SLAG] + 1.0),
  scm_frac: (x) => (x[I_FLYASH] + x[I_SLAG]) / (x[I_CEMENT] + x[I_FLYASH] + x[I_SLAG] + 1.0),
  hrwr_binder: (x) => x[I_HRWR] / (x[I_CEMENT] + x[I_FLYASH] + x[I_SLAG] + 1.0),
  log_hrwr_binder: (x) => Math.log(x[I_HRWR] / (x[I_CEMENT] + x[I_FLYASH] + x[I_SLAG] + 1.0) + 1e-4),
  wc_ratio: (x) => x[I_WATER] / (x[I_CEMENT] + 1.0),
  log_wc_ratio: (x) => Math.log(x[I_WATER] / (x[I_CEMENT] + 1.0) + 1e-3),
  coarse_fine: (x) => x[I_COARSE] / (x[I_FINE] + 1.0),
  log_coarse_fine: (x) => Math.log(x[I_COARSE] / (x[I_FINE] + 1.0) + 1e-3),
  agg_paste: (x) => (x[I_FINE] + x[I_COARSE]) /
    (x[I_CEMENT] + x[I_FLYASH] + x[I_SLAG] + x[I_WATER] + 1.0),
  log_agg_paste: (x) => Math.log(
    (x[I_FINE] + x[I_COARSE]) /
    (x[I_CEMENT] + x[I_FLYASH] + x[I_SLAG] + x[I_WATER] + 1.0) + 1e-3,
  ),
  maturity_robust: (x) => Math.max(0, x[I_TEMP] + 10.0) * x[I_TIME],
  log_maturity_robust: (x) => Math.log(Math.max(0, x[I_TEMP] + 10.0) * x[I_TIME] + 1.0),
};

/**
 * Append features to a raw input. Returns a NEW array of length
 * `xRaw.length + names.length`.
 */
export function appendFeatures(xRaw, names) {
  const out = new Array(xRaw.length + names.length);
  for (let i = 0; i < xRaw.length; i++) out[i] = xRaw[i];
  for (let k = 0; k < names.length; k++) {
    const fn = FEATURE_FNS[names[k]];
    if (!fn) {
      throw new Error(`Unknown engineered feature: '${names[k]}'. Add it to docs/feature_registry.mjs.`);
    }
    out[xRaw.length + k] = fn(xRaw);
  }
  return out;
}
