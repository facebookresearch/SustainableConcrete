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

// ---------------------------------------------------------------------------
// Gaussian helpers.
// ---------------------------------------------------------------------------

const SQRT2 = Math.SQRT2;
const INV_SQRT_2PI = 0.3989422804014327;

/**
 * Complementary error function, Numerical Recipes' Chebyshev form.
 *
 * Chosen over the more commonly copied Abramowitz & Stegun 7.1.26 because this
 * one is accurate in a RELATIVE sense, whereas A&S 7.1.26 is only accurate in
 * an ABSOLUTE sense (~1.5e-7). That distinction decides the far tail, which we
 * care about: Phi(-5) is 2.87e-7, so an absolute 1.5e-7 error there is ~50%
 * relative, and a candidate several sigma below the Pareto ceiling would get a
 * meaningless acquisition value.
 *
 * Numerical Recipes quotes fractional error < 1.2e-7, but that is the bound for
 * the truncated 7-coefficient form. With all 24 coefficients, as here, measured
 * relative error against known values is ~1e-15 from z=0 out to z=-7 -- see
 * "normalCdf keeps RELATIVE accuracy in the far tail" in test_js_bo.mjs.
 */
function erfc(x) {
  const z = Math.abs(x);
  const t = 2.0 / (2.0 + z);
  const ty = 4.0 * t - 2.0;
  const cof = [
    -1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2,
    -9.561514786808631e-3, -9.46595344482036e-4, 3.66839497852761e-4,
    4.2523324806907e-5, -2.0278578112534e-5, -1.624290004647e-6,
    1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8,
    6.529054439e-9, 5.059343495e-9, -9.91364156e-10,
    -2.27365122e-10, 9.6467911e-11, 2.394038e-12,
    -6.886027e-12, 8.94487e-13, 3.13092e-13,
    -1.12708e-13, 3.81e-16, 7.106e-15,
  ];
  let d = 0.0;
  let dd = 0.0;
  for (let j = cof.length - 1; j > 0; j--) {
    const tmp = d;
    d = ty * d - dd + cof[j];
    dd = tmp;
  }
  const ans = t * Math.exp(-z * z + 0.5 * (cof[0] + ty * d) - dd);
  return x >= 0.0 ? ans : 2.0 - ans;
}

/** Standard normal density. */
export function normalPdf(z) {
  return INV_SQRT_2PI * Math.exp(-0.5 * z * z);
}

/** Standard normal CDF. */
export function normalCdf(z) {
  return 0.5 * erfc(-z / SQRT2);
}

/**
 * E[max(0, Y - threshold)] for Y ~ N(mu, sd^2) — ordinary expected improvement.
 *
 * The `sd <= 0` branch is not defensive padding: a mix that has already been
 * acquired has (numerically) zero posterior variance at its observed times,
 * and `z` would be +/-Infinity.
 */
export function expectedImprovement(mu, sd, threshold) {
  const gap = mu - threshold;
  if (sd <= 0) return Math.max(0, gap);
  const z = gap / sd;
  return sd * normalPdf(z) + gap * normalCdf(z);
}

// ---------------------------------------------------------------------------
// Expected hypervolume improvement — closed form.
//
// Only strength is uncertain here; GWP and cost come from deterministic linear
// models, so a candidate's x is known exactly. With x fixed at g, the
// hypervolume improvement from observing y is
//
//     HVI(y) = integral over [g, refX] of max(0, y - ceiling(x)) dx
//
// which is piecewise-LINEAR in y, with one piece per staircase segment right
// of g. Expectation therefore passes straight through the sum:
//
//     E[HVI] = sum over segments of  width * EI(mu, sd; segment ceiling)
//
// Exact, O(P) per candidate, no quadrature and no sampling.
// ---------------------------------------------------------------------------

/**
 * @param front Ascending non-dominated staircase from `paretoStaircase`.
 *              Points outside the reference box are ignored, so the caller
 *              may pass an unclipped front.
 */
export function expectedHVI(mu, sd, g, front, refX, refY) {
  if (g > refX) return 0;

  let total = 0;
  let cursor = g;
  let ceiling = refY;
  for (const p of front) {
    if (p.x > refX || p.y < refY) continue;
    if (p.x > g) {
      total += (p.x - cursor) * expectedImprovement(mu, sd, ceiling);
      cursor = p.x;
    }
    // Left of the candidate the staircase only raises the ceiling it must beat.
    ceiling = Math.max(ceiling, p.y);
  }
  total += (refX - cursor) * expectedImprovement(mu, sd, ceiling);
  return total;
}

// ---------------------------------------------------------------------------
// Incremental Cholesky.
//
// The BO loop conditions the GP on a growing subset of the training rows. A
// direct refit is O(n^3) per iteration; extending the existing factor is
// O(n^2 b). Layout: L is row-major lower-triangular in a flat buffer with
// leading dimension `ld` (the full capacity), so growth never reallocates.
// The [n x b] blocks are column-major, matching gp_v2_fast.mjs's kernel layout.
// ---------------------------------------------------------------------------

/** Solve L X = B for X. B and X are [n x b] column-major. */
export function forwardSolve(L, ld, n, B, b) {
  const X = new Float64Array(n * b);
  for (let c = 0; c < b; c++) {
    const off = c * n;
    for (let i = 0; i < n; i++) {
      let acc = B[off + i];
      const row = i * ld;
      for (let k = 0; k < i; k++) acc -= L[row + k] * X[off + k];
      X[off + i] = acc / L[row + i];
    }
  }
  return X;
}

/**
 * Dense Cholesky of a small symmetric matrix, row-major [m x m], returning a
 * row-major lower-triangular factor.
 *
 * Escalating jitter mirrors gp.mjs::cholesky and, behind it, linear_operator's
 * psd_safe_cholesky. This is not hypothetical here: acquiring a mix whose
 * observations duplicate ones already in the conditioning set yields an exactly
 * rank-deficient block.
 */
export function choleskySmall(A, m) {
  const diag0 = new Float64Array(m);
  for (let i = 0; i < m; i++) diag0[i] = A[i * m + i];

  let jitter = 0;
  for (let attempt = 0; attempt < 10; attempt++) {
    const L = new Float64Array(m * m);
    let ok = true;
    for (let i = 0; i < m && ok; i++) {
      for (let j = 0; j <= i; j++) {
        let acc = A[i * m + j];
        if (i === j) acc += jitter;
        for (let k = 0; k < j; k++) acc -= L[i * m + k] * L[j * m + k];
        if (i === j) {
          if (!(acc > 0)) { ok = false; break; }
          L[i * m + i] = Math.sqrt(acc);
        } else {
          L[i * m + j] = acc / L[j * m + j];
        }
      }
    }
    if (ok) return L;
    jitter = jitter === 0 ? 1e-8 : jitter * 10;
    // Restore the pristine diagonal so each attempt sets jitter from the
    // original values rather than compounding the previous attempt's.
    for (let i = 0; i < m; i++) A[i * m + i] = diag0[i];
  }
  throw new Error("choleskySmall: Cholesky failed even after jitter escalation.");
}

/**
 * Extend a factored n x n block by b new rows.
 *
 *   Lb  = L^-1 K(X_a, X_new)                       [n x b] column-major
 *   Lnn = chol(K(X_new, X_new) + noise - Lb^T Lb)  [b x b] row-major
 *
 * @param Kna [n x b] column-major cross-covariance.
 * @param Knn [b x b] row-major self-covariance, noise already added.
 */
export function extendCholeskyBlock(L, ld, n, Kna, Knn, b) {
  const Lb = forwardSolve(L, ld, n, Kna, b);
  const S = new Float64Array(b * b);
  for (let i = 0; i < b; i++) {
    for (let j = 0; j < b; j++) {
      let acc = Knn[i * b + j];
      for (let k = 0; k < n; k++) acc -= Lb[i * n + k] * Lb[j * n + k];
      S[i * b + j] = acc;
    }
  }
  return { Lb, Lnn: choleskySmall(S, b) };
}

/** Write an extension block into L in place. Returns the new size, n + b. */
export function appendCholeskyBlock(L, ld, n, Lb, Lnn, b) {
  for (let i = 0; i < b; i++) {
    const row = (n + i) * ld;
    for (let k = 0; k < n; k++) L[row + k] = Lb[i * n + k];
    for (let j = 0; j <= i; j++) L[row + n + j] = Lnn[i * b + j];
  }
  return n + b;
}

// ---------------------------------------------------------------------------
// Kernel evaluation on post-transform rows.
//
// The composite kernel is
//     K(x1,x2) = h(t1) * [ M_blind + M_specific * hamming(s1,s2) + RBF_time ] * h(t2)
// exactly as in gp.mjs::kernel, which is the canonical definition.
//
// bo.mjs needs K between rows that are ALREADY in X_train's post-transform
// space, an access pattern neither gp.mjs (scalar, arbitrary vectors) nor
// gp_v2_fast.mjs (one composition across many times) provides. Rather than
// perform surgery on gp_v2_fast's deliberately-inlined hot loop -- its header
// records measured timings for that inlining -- this is a separate block
// builder, pinned to gp.mjs's scalar kernel by a parity test so the two cannot
// diverge on a schema change.
// ---------------------------------------------------------------------------

const SQRT5 = 2.23606797749979;

function unpackKernel(params) {
  return {
    blindDims: params.matern_blind.active_dims,
    blindLS: params.matern_blind.lengthscales,
    blindOS: params.matern_blind.outputscale,
    specDims: params.matern_specific.active_dims,
    specLS: params.matern_specific.lengthscales,
    specOS: params.matern_specific.outputscale,
    isHamming: params.matern_specific.source_kernel_kind === "hamming",
    srcCorr: params.matern_specific.source_correlation,
    srcDim: params.source_dim_raw,
    rbfIdx: params.rbf_time.active_dims[0],
    rbfLS: params.rbf_time.lengthscale,
    rbfOS: params.rbf_time.outputscale,
    gateTau: params.gate_tau,
    timeDim: params.time_dim_aug,
  };
}

function gate(t, tau) {
  return 1.0 - Math.exp(-Math.max(0.0, t) / tau);
}

/** k(x, x), which collapses to the summed outputscales times h(t)^2. */
export function selfKernel(rows, off, params) {
  const p = unpackKernel(params);
  const h = gate(rows[off + p.timeDim], p.gateTau);
  return (p.blindOS + p.specOS + p.rbfOS) * h * h;
}

/**
 * Cross-covariance block, returned column-major as [nA x nB]:
 * entry (i, c) lives at `out[c * nA + i]`.
 */
export function kernelBlock(A, nA, B, nB, dAug, params) {
  const p = unpackKernel(params);
  const nBlind = p.blindDims.length;
  const nSpec = p.specDims.length;

  const hA = new Float64Array(nA);
  for (let i = 0; i < nA; i++) hA[i] = gate(A[i * dAug + p.timeDim], p.gateTau);
  const hB = new Float64Array(nB);
  for (let c = 0; c < nB; c++) hB[c] = gate(B[c * dAug + p.timeDim], p.gateTau);

  const out = new Float64Array(nA * nB);
  for (let c = 0; c < nB; c++) {
    const bOff = c * dAug;
    const colOff = c * nA;
    for (let i = 0; i < nA; i++) {
      const aOff = i * dAug;

      let r2 = 0;
      for (let k = 0; k < nBlind; k++) {
        const d = p.blindDims[k];
        const dd = (A[aOff + d] - B[bOff + d]) / p.blindLS[k];
        r2 += dd * dd;
      }
      let r = Math.sqrt(r2);
      let s5r = SQRT5 * r;
      const kBlind = p.blindOS * (1 + s5r + (5 * r2) / 3) * Math.exp(-s5r);

      r2 = 0;
      for (let k = 0; k < nSpec; k++) {
        const d = p.specDims[k];
        const dd = (A[aOff + d] - B[bOff + d]) / p.specLS[k];
        r2 += dd * dd;
      }
      r = Math.sqrt(r2);
      s5r = SQRT5 * r;
      let kSpec = p.specOS * (1 + s5r + (5 * r2) / 3) * Math.exp(-s5r);
      if (p.isHamming && A[aOff + p.srcDim] !== B[bOff + p.srcDim]) kSpec *= p.srcCorr;

      const dt = (A[aOff + p.rbfIdx] - B[bOff + p.rbfIdx]) / p.rbfLS;
      const kRbf = p.rbfOS * Math.exp(-0.5 * dt * dt);

      out[colOff + i] = (kBlind + kSpec + kRbf) * hA[i] * hB[c];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Candidate set.
//
// The browser already has everything needed to map each training row back to
// its catalogue mix: un-normalising X_train with normalize_lower/upper recovers
// the 9 raw columns verbatim, and the time column comes back as
// 10^u[timeDim] - log_time_offset. Verified against the shipped artifacts: all
// 670 rows match exactly one of the 149 compositions. No new export required.
// ---------------------------------------------------------------------------

const SUPPORTED_DAYS = [1, 28];

/**
 * Index the training rows by mix and assemble the candidate set for one day.
 *
 * Candidates are the mixes that actually have an observation at `day`, and
 * each candidate's test row is lifted straight out of X_train rather than
 * rebuilt through appendFeatures -- it is the same point, already transformed.
 */
export function buildCandidateSet(strengthParams, compositionsData, day) {
  if (!SUPPORTED_DAYS.includes(day)) {
    throw new Error(
      `buildCandidateSet: unsupported day ${day} (expected one of ${SUPPORTED_DAYS.join(", ")})`,
    );
  }
  const lo = strengthParams.normalize_lower;
  const hi = strengthParams.normalize_upper;
  for (let k = 0; k < lo.length; k++) {
    if (hi[k] === lo[k]) {
      throw new Error(
        `buildCandidateSet: degenerate normalisation range at dim ${k} ` +
        `(lower === upper === ${lo[k]}); un-normalising would divide by zero.`,
      );
    }
  }

  const dAug = strengthParams.d_aug;
  const nTrain = strengthParams.n_train;
  const timeDim = strengthParams.time_dim_raw;
  const offset = strengthParams.log_time_offset || 1.0;
  const X = strengthParams.X_train;

  // Catalogue lookup keyed on the 9 rounded raw columns.
  const nRaw = compositionsData.column_names.length;
  const key = (v) => v.map((z) => z.toFixed(4)).join("|");
  const lookup = new Map();
  compositionsData.compositions.forEach((comp, m) => lookup.set(key(comp), m));

  const rowMix = new Int32Array(nTrain);
  const rowTime = new Float64Array(nTrain);
  for (let r = 0; r < nTrain; r++) {
    const raw = [];
    for (let k = 0; k < nRaw; k++) raw.push(X[r][k] * (hi[k] - lo[k]) + lo[k]);
    const m = lookup.get(key(raw));
    if (m === undefined) {
      throw new Error(
        `buildCandidateSet: training row ${r} does not match any catalogue mix. ` +
        "strength.json and compositions.json are out of sync; re-run " +
        "experiments/regenerate_all_artifacts.sh.",
      );
    }
    rowMix[r] = m;
    rowTime[r] = Math.pow(10, X[r][timeDim] * (hi[timeDim] - lo[timeDim]) + lo[timeDim]) - offset;
  }

  const rowsByMix = new Map();
  for (let r = 0; r < nTrain; r++) {
    if (!rowsByMix.has(rowMix[r])) rowsByMix.set(rowMix[r], []);
    rowsByMix.get(rowMix[r]).push(r);
  }

  const mix = [];
  const rowsOfCandidate = [];
  const testRowIdx = [];
  for (const [m, rows] of [...rowsByMix.entries()].sort((a, b) => a[0] - b[0])) {
    const atDay = rows.filter((r) => Math.abs(rowTime[r] - day) < 1e-6);
    if (atDay.length === 0) continue;
    mix.push(m);
    rowsOfCandidate.push(rows);
    testRowIdx.push(atDay[0]);
  }

  const m = mix.length;
  const candX = new Float64Array(m * dAug);
  const kSelf = new Float64Array(m);
  const observedY = new Float64Array(m);
  const gwp = new Float64Array(m);
  const cost = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    const src = X[testRowIdx[k]];
    for (let d = 0; d < dAug; d++) candX[k * dAug + d] = src[d];
    kSelf[k] = selfKernel(candX, k * dAug, strengthParams);
    const obs = compositionsData.observations[String(mix[k])].filter(([d]) => d === day);
    observedY[k] = Math.max(...obs.map(([, psi]) => psi));
    // JSON stores -GWP and -Cost (Python maximises everything).
    gwp[k] = -compositionsData.gwp_predictions[mix[k]];
    cost[k] = -compositionsData.cost_predictions[mix[k]];
  }

  return { day, mix, rowMix, rowTime, rowsOfCandidate, candX, kSelf, observedY, gwp, cost, dAug };
}
