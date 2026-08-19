/**
 * Multi-objective Bayesian optimization over the concrete mix catalog.
 *
 * Pure and DOM-free so it can be unit-tested under `node --test` with a 100%
 * line/branch/function coverage gate (`make test-js-bo`). All presentation
 * logic lives in bo_view.mjs; ui.mjs only executes the resulting draw-list.
 *
 * Run: node --test test/test_js_bo.mjs
 */

import { transformInput } from "./gp.mjs";

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
  // Jitter is added to the accumulator, not written back into A, so A is never
  // mutated and successive attempts cannot compound each other's jitter.
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

// ---------------------------------------------------------------------------
// The BO loop.
//
// Hyperparameters stay frozen at the shipped full-data fit; only the
// CONDITIONING SET grows. Carrying w = L^-1 y and V = L^-1 K_test forward
// alongside L is what makes this cheap: the posterior for all m candidates
// updates in O(n*m) per iteration, with no back-solve for alpha and no dtrsm.
//
//   mu_j  = y_max * (w . V[:,j])
//   var_j = y_max^2 * (k(x_j,x_j) - ||V[:,j]||^2 + gated noise)
//
// Acquiring a mix appends its b observation rows as one block; mu and var then
// need only the new block's contribution:
//
//   mu_j  += y_max   * sum_c wb[c] * Vb[c][j]
//   var_j -= y_max^2 * sum_c Vb[c][j]^2
//
// V is row-major [capacity x m] so the Lb^T V contraction runs contiguously
// over j, and appending rows is a straight write at the end.
// ---------------------------------------------------------------------------

/** Deterministic PRNG (mulberry32). Seeded so a run is always reproducible. */
export function makeRng(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** `count` distinct candidate indices, drawn deterministically from `seed`. */
export function chooseSeeds(m, count, seed) {
  if (count > m) {
    throw new Error(`chooseSeeds: asked for ${count} seed mixes but only ${m} candidates exist`);
  }
  const rng = makeRng(seed);
  const idx = Array.from({ length: m }, (_, i) => i);
  for (let i = m - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx.slice(0, count);
}

/**
 * Aleatoric variance at each candidate, in scaled units.
 *
 * Mirrors gp_v2_fast.mjs: with `noise_kind === "gated"` the noise scales as
 * h(t)^2 * sigma_n^2. Included because strength.json advertises
 * `variance_includes_aleatoric: true`, so the BO uncertainty matches the band
 * the site already draws. It also keeps an acquired candidate's sd strictly
 * positive, which stops EI from dividing by zero.
 */
export function candidateNoiseVariance(candidates, strengthParams) {
  const tau = strengthParams.noise_gate_tau ?? strengthParams.gate_tau;
  const gated = (strengthParams.noise_kind || "global") === "gated";
  const noise = strengthParams.noise || 0;
  const timeDim = strengthParams.time_dim_aug;
  const out = new Float64Array(candidates.mix.length);
  for (let k = 0; k < out.length; k++) {
    if (!gated) {
      out[k] = noise;
      continue;
    }
    const h = gate(candidates.candX[k * candidates.dAug + timeDim], tau);
    out[k] = h * h * noise;
  }
  return out;
}

function rowsFor(state, candidateIdx) {
  return state.candidates.rowsOfCandidate[candidateIdx];
}

/** Condition on one candidate's observation rows, updating L, w, V, mu, var. */
function conditionOn(state, k) {
  const { strengthParams: sp, candidates: cand } = state;
  const dAug = cand.dAug;
  const m = cand.mix.length;
  const rows = rowsFor(state, k);
  const b = rows.length;
  const n = state.n;

  const Xn = new Float64Array(b * dAug);
  for (let i = 0; i < b; i++) {
    const src = sp.X_train[rows[i]];
    for (let d = 0; d < dAug; d++) Xn[i * dAug + d] = src[d];
  }

  // Kna [n x b] column-major, Knn [b x b] row-major with noise on the diagonal.
  let Kna = new Float64Array(0);
  if (n > 0) {
    const Xa = new Float64Array(n * dAug);
    for (let i = 0; i < n; i++) {
      const src = sp.X_train[state.rows[i]];
      for (let d = 0; d < dAug; d++) Xa[i * dAug + d] = src[d];
    }
    Kna = kernelBlock(Xa, n, Xn, b, dAug, sp);
  }
  const KnnCm = kernelBlock(Xn, b, Xn, b, dAug, sp);
  const Knn = new Float64Array(b * b);
  for (let i = 0; i < b; i++) {
    for (let j = 0; j < b; j++) Knn[i * b + j] = KnnCm[j * b + i];
    // state.noise, not sp.noise: the field is optional in the schema, and an
    // undefined here would put NaN on the diagonal and fail the factorisation
    // with a misleading "Cholesky failed" rather than a missing-field error.
    Knn[i * b + i] += state.noise;
  }

  const { Lb, Lnn } = extendCholeskyBlock(state.L, state.ld, n, Kna, Knn, b);

  // wb = Lnn^-1 (y_new - Lb^T w)
  const rhsW = new Float64Array(b);
  for (let c = 0; c < b; c++) {
    let acc = sp.Y_train[rows[c]];
    for (let i = 0; i < n; i++) acc -= Lb[c * n + i] * state.w[i];
    rhsW[c] = acc;
  }
  const wb = forwardSolve(Lnn, b, b, rhsW, 1);

  // Vb = Lnn^-1 (K(X_new, X_cand) - Lb^T V)     [b x m] row-major
  const KnCand = kernelBlock(Xn, b, cand.candX, m, dAug, sp);
  const rhsV = new Float64Array(b * m);
  for (let c = 0; c < b; c++) {
    for (let j = 0; j < m; j++) {
      let acc = KnCand[j * b + c];
      for (let i = 0; i < n; i++) acc -= Lb[c * n + i] * state.V[i * m + j];
      rhsV[c * m + j] = acc;
    }
  }
  // Forward-substitute down the b rows in place.
  const Vb = new Float64Array(b * m);
  for (let c = 0; c < b; c++) {
    const diag = Lnn[c * b + c];
    for (let j = 0; j < m; j++) {
      let acc = rhsV[c * m + j];
      for (let k2 = 0; k2 < c; k2++) acc -= Lnn[c * b + k2] * Vb[k2 * m + j];
      Vb[c * m + j] = acc / diag;
    }
  }

  const ymax = sp.y_max;
  for (let j = 0; j < m; j++) {
    let dmu = 0;
    let dvar = 0;
    for (let c = 0; c < b; c++) {
      const v = Vb[c * m + j];
      dmu += wb[c] * v;
      dvar += v * v;
    }
    state.mu[j] += dmu * ymax;
    state.latentVar[j] = Math.max(0, state.latentVar[j] - dvar);
    state.variance[j] = (state.latentVar[j] + state.noiseVar[j]) * ymax * ymax;
  }

  appendCholeskyBlock(state.L, state.ld, n, Lb, Lnn, b);
  for (let c = 0; c < b; c++) {
    state.w[n + c] = wb[c];
    for (let j = 0; j < m; j++) state.V[(n + c) * m + j] = Vb[c * m + j];
    state.rows.push(rows[c]);
  }
  state.n = n + b;
}

/**
 * @param seedMixes Candidate indices to condition on before the loop starts.
 *                  The random arm must be given the SAME set (see the fairness
 *                  contract in test_js_bo.mjs).
 */
export function createBOState({
  strengthParams, candidates, objectiveX, refX, refY, seedMixes,
}) {
  const m = candidates.mix.length;
  const capacity = candidates.rowsOfCandidate.reduce((a, r) => a + r.length, 0);
  const state = {
    strengthParams,
    candidates,
    objectiveX,
    refX,
    refY,
    m,
    ld: capacity,
    noise: strengthParams.noise || 0,
    n: 0,
    rows: [],
    L: new Float64Array(capacity * capacity),
    w: new Float64Array(capacity),
    V: new Float64Array(capacity * m),
    mu: new Float64Array(m),
    latentVar: Float64Array.from(candidates.kSelf),
    noiseVar: candidateNoiseVariance(candidates, strengthParams),
    variance: new Float64Array(m),
    acquired: [],
    pool: Array.from({ length: m }, (_, i) => i),
    iteration: 0,
    lastPicked: null,
  };
  const ymax = strengthParams.y_max;
  for (let j = 0; j < m; j++) {
    state.variance[j] = (state.latentVar[j] + state.noiseVar[j]) * ymax * ymax;
  }
  for (const k of seedMixes) acquireInto(state, k);
  return state;
}

function acquireInto(state, k) {
  conditionOn(state, k);
  state.acquired.push(k);
  const at = state.pool.indexOf(k);
  if (at >= 0) state.pool.splice(at, 1);
  state.lastPicked = k;
}

/** Measured (x, y) of every acquired mix, as a non-dominated staircase. */
export function observedFront(state) {
  const xs = state.acquired.map((k) => state.objectiveX[k]);
  const ys = state.acquired.map((k) => state.candidates.observedY[k]);
  return paretoStaircase(xs, ys);
}

/**
 * Hypervolume of what has actually been MEASURED so far.
 *
 * Deliberately reads observedY, never the posterior: the model chooses what to
 * acquire, it never grades itself. This is what makes the BO-vs-random
 * comparison fair, and there is a test that poisons the GP to prove it.
 */
export function observedHypervolume(state) {
  return hypervolume2D(
    state.acquired.map((k) => state.objectiveX[k]),
    state.acquired.map((k) => state.candidates.observedY[k]),
    state.refX,
    state.refY,
  );
}

/**
 * One BO iteration: score the pool by EHVI, acquire the argmax, recondition.
 *
 * @param pick Optional explicit candidate index, for replaying a fixed
 *             sequence or driving a non-BO arm through the same machinery.
 */
export function stepBO(state, pick = undefined) {
  if (state.pool.length === 0) {
    return { picked: null, acqValues: new Float64Array(0), hypervolume: observedHypervolume(state) };
  }
  const front = observedFront(state);
  const acqValues = new Float64Array(state.pool.length);
  for (let i = 0; i < state.pool.length; i++) {
    const k = state.pool[i];
    acqValues[i] = expectedHVI(
      state.mu[k],
      Math.sqrt(state.variance[k]),
      state.objectiveX[k],
      front,
      state.refX,
      state.refY,
    );
  }

  let picked;
  if (pick === undefined) {
    let best = 0;
    // Strict >, so ties resolve to the lowest pool index and a run is
    // reproducible even when every candidate scores exactly 0.
    for (let i = 1; i < acqValues.length; i++) if (acqValues[i] > acqValues[best]) best = i;
    picked = state.pool[best];
  } else {
    if (state.acquired.includes(pick)) {
      throw new Error(
        `stepBO: candidate ${pick} is already acquired. Re-conditioning on rows ` +
        "already in the set makes the Schur complement exactly singular.",
      );
    }
    picked = pick;
  }

  acquireInto(state, picked);
  state.iteration += 1;
  return { picked, acqValues, hypervolume: observedHypervolume(state) };
}

/**
 * Hypervolume percentile bands for uniform random search.
 *
 * Takes no model. That is structural, not stylistic: the random arm must not
 * be able to consult the GP even by accident, and a test asserts this
 * function's source never mentions strengthParams.
 */
export function randomArmTraces({
  candidates, objectiveX, refX, refY, seedMixes, nIters, nRestarts, seed,
}) {
  const m = candidates.mix.length;
  const runs = [];
  for (let r = 0; r < nRestarts; r++) {
    const rng = makeRng(seed + r * 7919);
    const pool = [];
    for (let i = 0; i < m; i++) if (!seedMixes.includes(i)) pool.push(i);
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    const acquired = [...seedMixes];
    const trace = new Float64Array(nIters);
    for (let it = 0; it < nIters; it++) {
      trace[it] = hypervolume2D(
        acquired.map((k) => objectiveX[k]),
        acquired.map((k) => candidates.observedY[k]),
        refX,
        refY,
      );
      if (it < pool.length) acquired.push(pool[it]);
    }
    runs.push(trace);
  }
  const pct = (q) => {
    const out = new Float64Array(nIters);
    const col = new Float64Array(nRestarts);
    for (let it = 0; it < nIters; it++) {
      for (let r = 0; r < nRestarts; r++) col[r] = runs[r][it];
      const sorted = Float64Array.from(col).sort();
      out[it] = sorted[Math.min(nRestarts - 1, Math.floor(q * nRestarts))];
    }
    return out;
  };
  return { p10: pct(0.1), p50: pct(0.5), p90: pct(0.9) };
}

/** Solve L^T x = b for x, given a lower-triangular L. */
export function backSolve(L, ld, n, b) {
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let acc = b[i];
    for (let k = i + 1; k < n; k++) acc -= L[k * ld + i] * x[k];
    x[i] = acc / L[i * ld + i];
  }
  return x;
}

/**
 * Strength curve under the CURRENT conditioning set, not the full-data model.
 *
 * BO mode re-points the right-hand panel at the mix just acquired so its
 * uncertainty band visibly tightens. Leaving that panel on the full-data model
 * would show a confident band beside a scatter driven by a handful of
 * observations -- actively contradictory, not just a missed opportunity.
 *
 * Costs an O(n^2) back-solve for alpha plus an [n x K] triangular solve for the
 * band. Called once per iteration, never per frame.
 *
 * Test rows go through gp.mjs's exported `transformInput`, so the derive ->
 * log_offset -> log10 -> normalize chain has exactly one implementation.
 */
export function predictStrengthCurveSubset(state, composition, times) {
  const sp = state.strengthParams;
  const dAug = sp.d_aug;
  const dRaw = sp.d_in || 10;
  const nT = times.length;
  const n = state.n;

  const raw = new Array(dRaw).fill(0);
  for (let i = 0; i < dRaw; i++) raw[i] = composition[i] ?? 0;

  const Xt = new Float64Array(nT * dAug);
  for (let t = 0; t < nT; t++) {
    raw[sp.time_dim_raw] = times[t];
    const row = transformInput(raw, sp);
    for (let d = 0; d < dAug; d++) Xt[t * dAug + d] = row[d];
  }

  const Xa = new Float64Array(n * dAug);
  for (let i = 0; i < n; i++) {
    const src = sp.X_train[state.rows[i]];
    for (let d = 0; d < dAug; d++) Xa[i * dAug + d] = src[d];
  }

  const Ka = kernelBlock(Xa, n, Xt, nT, dAug, sp); // [n x nT] column-major
  const alpha = backSolve(state.L, state.ld, n, state.w);
  const Vc = forwardSolve(state.L, state.ld, n, Ka, nT);

  const gatedNoise = (sp.noise_kind || "global") === "gated";
  const noiseTau = sp.noise_gate_tau ?? sp.gate_tau;
  const noise = sp.noise || 0;
  const ymax = sp.y_max;

  const means = new Float64Array(nT);
  const variances = new Float64Array(nT);
  for (let t = 0; t < nT; t++) {
    let mean = 0;
    let nrm = 0;
    for (let i = 0; i < n; i++) {
      mean += Ka[t * n + i] * alpha[i];
      const v = Vc[t * n + i];
      nrm += v * v;
    }
    means[t] = mean * ymax;
    let latent = Math.max(0, selfKernel(Xt, t * dAug, sp) - nrm);
    if (gatedNoise) {
      const h = gate(Xt[t * dAug + sp.time_dim_aug], noiseTau);
      latent += h * h * noise;
    } else {
      latent += noise;
    }
    variances[t] = latent * ymax * ymax;
  }
  return { means, variances };
}
