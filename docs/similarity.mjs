/**
 * Recipe-similarity between concrete mixes, as the deployed strength GP sees it.
 *
 * DOM-free so it can be unit tested under node. `docs/ui.mjs` touches the DOM at
 * module scope and cannot be imported outside a browser, which is why the pure
 * pieces live in siblings like this one (see `preview_state.mjs`).
 *
 * WHAT THE METRIC IS
 * ------------------
 * The strength GP's own normalized composition kernel — the model's believed
 * correlation between two mixes. It is a single learned joint metric: ARD
 * lengthscales over the continuous dims, plus a learned categorical weight for
 * Material Source (`source_correlation`), combined by the learned outputscales.
 * Nothing here is hand-tuned except the presentation mapping at the bottom.
 *
 * WHY THE TIME BRANCH IS EXCLUDED
 * -------------------------------
 * The production kernel (`gp.mjs:kernel`) is
 *   h(t1) * (M_blind + M_specific * hamming + RBF_time) * h(t2)
 * Two mixes are always compared at the same curing day, so the h(t) gate cancels
 * in the normalization and RBF_time reduces to its outputscale — a constant
 * 0.4578 out of the 0.5245 total. Including it would compress every similarity
 * into [0.907, 1.000], which renders as a uniform plot rather than an obviously
 * broken one. `test/test_js_similarity.mjs` pins the resulting dynamic range so
 * this cannot silently regress.
 */
import { matern52ActiveDims, transformInput } from "./gp.mjs";

/** Raw similarity at or below this renders fully hollow. Empirical catalog min is 0.267. */
export const SIMILARITY_FLOOR = 0.25;
/** Contrast exponent. 1.0 leaves everything looking solid, 3.0 leaves everything hollow. */
export const SIMILARITY_GAMMA = 2.0;
/** Fill alpha of the least similar mix. The outline carries position regardless. */
export const FILL_ALPHA_MIN = 0.06;

function hammingFactor(src1, src2, params) {
  const ms = params.matern_specific;
  if (!ms || ms.source_kernel_kind !== "hamming") return 1.0;
  return src1 === src2 ? 1.0 : ms.source_correlation;
}

/**
 * Hot path: both inputs already run through `transformInput`.
 *
 * @param {number[]} z1 - transformed 17-dim vector.
 * @param {number[]} z2 - transformed 17-dim vector.
 * @param {number} src1 - raw Material Source class of the first mix.
 * @param {number} src2 - raw Material Source class of the second mix.
 * @param {object} params - parsed strength.json.
 * @returns {number} similarity in [0, 1].
 */
export function similarityFromTransformed(z1, z2, src1, src2, params) {
  const b = params.matern_blind;
  const s = params.matern_specific;
  const kb = matern52ActiveDims(z1, z2, b.active_dims, b.lengthscales, b.outputscale);
  const ks =
    matern52ActiveDims(z1, z2, s.active_dims, s.lengthscales, s.outputscale) *
    hammingFactor(src1, src2, params);
  return (kb + ks) / (b.outputscale + s.outputscale);
}

/**
 * Normalized composition kernel between two 9-dim compositions at a fixed curing day.
 *
 * @param {number[]} comp1 - 9-dim composition.
 * @param {number[]} comp2 - 9-dim composition.
 * @param {object|null} params - parsed strength.json, or null before the model loads.
 * @param {number} curingDay - day both mixes are evaluated at.
 * @returns {number|null} similarity in [0, 1], or null when the model is unavailable.
 */
export function compositionSimilarity(comp1, comp2, params, curingDay) {
  if (!params || !params.matern_blind || !params.matern_specific) return null;
  return similarityFromTransformed(
    transformInput([...comp1, curingDay], params),
    transformInput([...comp2, curingDay], params),
    comp1[params.source_dim_raw],
    comp2[params.source_dim_raw],
    params,
  );
}
