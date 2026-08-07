/**
 * DOM-free preview-composition state, extracted so it can be unit tested.
 *
 * `docs/ui.mjs` touches the DOM at module scope and cannot be imported under
 * node, so the one piece of logic that has caused real bugs lives here
 * instead: advancing the ghost "preview" composition toward a hover target.
 *
 * The rule that keeps biting: a categorical dimension must never be
 * interpolated. The preview curve is predicted directly from this
 * composition, and the strength GP's source kernel is a Hamming
 * `CategoricalKernel` — it tests equality, not distance. A fractional class
 * matches no training row, so every row gets down-weighted and the posterior
 * collapses to a single "unseen class" value that is not a blend of anything.
 * See `test/test_js_preview_state.mjs` and `test/test_js_categorical_source.mjs`.
 */

/**
 * Advance `display` one frame toward `target`.
 *
 * Continuous dimensions ease toward the target by `factor`, snapping once
 * within `EPS`. The dimension at `categoricalIdx` is snapped immediately to
 * the nearest integer class and never interpolated. Pass a negative
 * `categoricalIdx` when the composition has no categorical dimension.
 *
 * Mutates `display` in place (it is a hot per-frame path) and returns whether
 * every dimension has converged.
 *
 * @param {number[]} display - composition being animated; mutated in place.
 * @param {number[]} target - composition to approach.
 * @param {number} factor - per-frame easing fraction in [0, 1].
 * @param {number} categoricalIdx - index to snap, or < 0 for none.
 * @returns {boolean} true when all continuous dims have reached the target.
 */
export const PREVIEW_EPS = 1e-6;

export function stepPreviewComposition(display, target, factor, categoricalIdx) {
  let converged = true;
  for (let i = 0; i < display.length; i++) {
    if (i === categoricalIdx) {
      display[i] = Math.round(target[i]);
      continue;
    }
    const diff = target[i] - display[i];
    if (Math.abs(diff) > PREVIEW_EPS) {
      display[i] += diff * factor;
      converged = false;
    } else {
      display[i] = target[i];
    }
  }
  return converged;
}
