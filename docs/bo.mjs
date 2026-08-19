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

// ---------------------------------------------------------------------------
// Pareto geometry.
//
// Convention: MINIMISE x (GWP or cost), MAXIMISE y (strength). Point i is
// dominated by j iff x_j <= x_i and y_j >= y_i with one strict. So a
// non-dominated set sorted by x ascending has y ascending, and the region it
// covers is a RISING staircase: on [x_k, x_{k+1}) the ceiling is y_k.
// ---------------------------------------------------------------------------

/**
 * Non-dominated staircase, ascending in x (and therefore in y).
 *
 * Returns `{x, y, i}` with `i` the index into the input arrays, so callers can
 * map a front point back to the mix that produced it.
 *
 * Not clipped to any reference point — the view draws the true front, while
 * `hypervolume2D` clips internally. (Clipping commutes with the dominance
 * filter here: anything dominating an in-box point is itself in the box.)
 */
export function paretoStaircase(xs, ys) {
  if (xs.length !== ys.length) {
    throw new Error(
      `paretoStaircase: length mismatch (xs=${xs.length}, ys=${ys.length})`,
    );
  }
  const pts = [];
  for (let i = 0; i < xs.length; i++) pts.push({ x: xs[i], y: ys[i], i });
  // Ascending x, then DESCENDING y. The secondary key matters: with a tie in x
  // the weaker point is strictly dominated, and taking the stronger one first
  // lets the running-max filter below drop it. Sorting y ascending instead
  // would keep both and emit a spurious zero-width step.
  pts.sort((a, b) => (a.x - b.x) || (b.y - a.y));

  const front = [];
  let bestY = -Infinity;
  for (const p of pts) {
    if (p.y > bestY) {
      front.push(p);
      bestY = p.y;
    }
  }
  return front;
}

/**
 * Hypervolume of the region dominated by the front, bounded by the reference
 * point: x <= refX (an upper bound to stay under) and y >= refY (a lower bound
 * to clear).
 *
 * Zero — never NaN — when nothing clears the reference point, which is the
 * real situation at iteration 0 under CONCRETE_REFERENCE_POINT.
 */
export function hypervolume2D(xs, ys, refX, refY) {
  if (xs.length !== ys.length) {
    throw new Error(
      `hypervolume2D: length mismatch (xs=${xs.length}, ys=${ys.length})`,
    );
  }
  const inX = [];
  const inY = [];
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] <= refX && ys[i] >= refY) {
      inX.push(xs[i]);
      inY.push(ys[i]);
    }
  }
  const front = paretoStaircase(inX, inY);
  let total = 0;
  for (let k = 0; k < front.length; k++) {
    const xHi = k + 1 < front.length ? front[k + 1].x : refX;
    total += (xHi - front[k].x) * (front[k].y - refY);
  }
  return total;
}
