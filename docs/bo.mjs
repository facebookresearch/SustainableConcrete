/**
 * Multi-objective Bayesian optimization over the concrete mix catalog.
 *
 * Pure and DOM-free so it can be unit-tested under `node --test` with a 100%
 * line/branch/function coverage gate (`make test-js-bo`). All presentation
 * logic lives in bo_view.mjs; ui.mjs only executes the resulting draw-list.
 *
 * Run: node --test test/test_js_bo.mjs
 */

/**
 * Reference point for the hypervolume indicator, mirroring
 * `boxcrete.CONCRETE_REFERENCE_POINT` = [-200, 1000, 5000] and
 * `boxcrete.CONCRETE_COST_THRESHOLD` = 250.
 *
 * Python stores GWP and cost negated (it maximizes everything); here the axes
 * are in natural display units, so the sign flip is already applied:
 * `gwp` and `cost` are upper bounds to stay under, `strength1`/`strength28`
 * are lower bounds to clear.
 */
export const REFERENCE_POINT = Object.freeze({
  gwp: 200.0,
  cost: 250.0,
  strength1: 1000.0,
  strength28: 5000.0,
});

/**
 * Resolve the reference point for one scatter configuration.
 *
 * `xAxis` is "gwp" or "cost"; `day` is 1 or 28. Throws rather than defaulting,
 * matching gp.mjs's schema guards: a silently-wrong reference point would make
 * every hypervolume number meaningless without any visible symptom.
 */
export function referenceFor(xAxis, day) {
  const refX = REFERENCE_POINT[xAxis];
  if (refX === undefined) {
    throw new Error(`referenceFor: unknown xAxis "${xAxis}" (expected "gwp" or "cost")`);
  }
  const refY = REFERENCE_POINT[`strength${day}`];
  if (refY === undefined) {
    throw new Error(`referenceFor: unknown day "${day}" (expected 1 or 28)`);
  }
  return { refX, refY };
}
