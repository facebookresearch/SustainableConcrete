/**
 * Unit tests for docs/bo.mjs — the multi-objective BO engine.
 *
 * Uses node:test rather than the repo's hand-rolled `check()` helper because
 * that is what unlocks --experimental-test-coverage. Held to 100%
 * line/branch/function coverage by `make test-js-bo`.
 *
 * Run: node --test test/test_js_bo.mjs
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  REFERENCE_POINT,
  referenceFor,
  paretoStaircase,
  hypervolume2D,
  normalPdf,
  normalCdf,
  expectedImprovement,
  expectedHVI,
  forwardSolve,
  choleskySmall,
  extendCholeskyBlock,
  appendCholeskyBlock,
  kernelBlock,
  selfKernel,
  buildCandidateSet,
  candidateNoiseVariance,
  createBOState,
  stepBO,
  observedFront,
  observedHypervolume,
  randomArmTraces,
  chooseSeeds,
  makeRng,
} from "../docs/bo.mjs";
import { kernel as gpKernel } from "../docs/gp.mjs";
import { readFileSync } from "node:fs";

// The deployed model and catalogue, exactly as the browser fetches them.
const strengthParams = JSON.parse(
  readFileSync(new URL("../docs/model/strength.json", import.meta.url), "utf8"),
);
const compositionsData = JSON.parse(
  readFileSync(new URL("../docs/model/compositions.json", import.meta.url), "utf8"),
);
const D_AUG = strengthParams.d_aug;
const X_FLAT = Float64Array.from(strengthParams.X_train.flat());

/**
 * Independent reference implementation: a from-scratch refit on `rows`.
 * Deliberately composed differently from the incremental path (one dense
 * factorisation, no block extension), so agreement is a real check.
 */
function directPosterior(sp, cand, rows) {
  const dAug = cand.dAug;
  const n = rows.length;
  const m = cand.mix.length;
  const Xa = Float64Array.from(rows.flatMap((r) => sp.X_train[r]));
  const Kcm = kernelBlock(Xa, n, Xa, n, dAug, sp);
  const A = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i * n + j] = Kcm[j * n + i];
  for (let i = 0; i < n; i++) A[i * n + i] += sp.noise;
  const L = choleskySmall(A, n);
  const y = Float64Array.from(rows.map((r) => sp.Y_train[r]));
  const w = forwardSolve(L, n, n, y, 1);
  const Kt = kernelBlock(Xa, n, cand.candX, m, dAug, sp);
  const V = forwardSolve(L, n, n, Kt, m);
  const noiseVar = candidateNoiseVariance(cand, sp);
  const ymax = sp.y_max;
  const mu = new Float64Array(m);
  const variance = new Float64Array(m);
  for (let j = 0; j < m; j++) {
    let dot = 0;
    let nrm = 0;
    for (let i = 0; i < n; i++) {
      const v = V[j * n + i];
      dot += w[i] * v;
      nrm += v * v;
    }
    mu[j] = dot * ymax;
    variance[j] = (Math.max(0, cand.kSelf[j] - nrm) + noiseVar[j]) * ymax * ymax;
  }
  return { mu, variance };
}

function maxAbsDiff(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

// Deterministic uniform stream, shared by the linear-algebra fixtures.
function makeUniform(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return (s + 0.5) / 4294967296;
  };
}

/** Random symmetric positive-definite matrix, row-major [m x m]. */
function randomSPD(m, seed, ridge = 1.0) {
  const u = makeUniform(seed);
  const A = new Float64Array(m * m);
  const G = new Float64Array(m * m);
  for (let i = 0; i < m * m; i++) G[i] = u() * 2 - 1;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      let acc = 0;
      for (let k = 0; k < m; k++) acc += G[i * m + k] * G[j * m + k];
      A[i * m + j] = acc;
    }
    A[i * m + i] += ridge * m;
  }
  return A;
}

/** Max |(L L^T) - A| over a row-major lower-triangular L with leading dim ld. */
function reconstructionError(L, ld, m, A) {
  let worst = 0;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      let acc = 0;
      for (let k = 0; k <= Math.min(i, j); k++) acc += L[i * ld + k] * L[j * ld + k];
      worst = Math.max(worst, Math.abs(acc - A[i * m + j]));
    }
  }
  return worst;
}

// Deterministic normal sampler for the Monte-Carlo cross-checks. A flaky
// statistical test is worse than none, so this is a fixed-seed LCG + Box-Muller
// rather than Math.random().
function makeNormalSampler(seed) {
  let s = seed >>> 0;
  const u01 = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return (s + 0.5) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(u01())) * Math.cos(2 * Math.PI * u01());
}

// ---------------------------------------------------------------------------
// Reference point — mirrors boxcrete.CONCRETE_REFERENCE_POINT.
// ---------------------------------------------------------------------------

test("REFERENCE_POINT matches boxcrete.CONCRETE_REFERENCE_POINT", () => {
  // Python: CONCRETE_REFERENCE_POINT = [-200.0, 1000.0, 5000.0] (negated GWP),
  // CONCRETE_COST_THRESHOLD = 250.0. Here the signs are already flipped into
  // natural display units.
  assert.equal(REFERENCE_POINT.gwp, 200.0);
  assert.equal(REFERENCE_POINT.cost, 250.0);
  assert.equal(REFERENCE_POINT.strength1, 1000.0);
  assert.equal(REFERENCE_POINT.strength28, 5000.0);
});

test("REFERENCE_POINT is frozen so a caller cannot corrupt every later run", () => {
  assert.throws(() => {
    "use strict";
    REFERENCE_POINT.gwp = 1;
  }, TypeError);
});

test("referenceFor resolves each supported axis/day combination", () => {
  assert.deepEqual(referenceFor("gwp", 28), { refX: 200.0, refY: 5000.0 });
  assert.deepEqual(referenceFor("gwp", 1), { refX: 200.0, refY: 1000.0 });
  assert.deepEqual(referenceFor("cost", 28), { refX: 250.0, refY: 5000.0 });
  assert.deepEqual(referenceFor("cost", 1), { refX: 250.0, refY: 1000.0 });
});

test("referenceFor throws on an unknown axis rather than defaulting", () => {
  // A silently-wrong reference point makes every hypervolume number
  // meaningless with no visible symptom, so this must be loud.
  assert.throws(() => referenceFor("slump", 28), /unknown xAxis "slump"/);
});

test("referenceFor throws on an unknown day rather than defaulting", () => {
  assert.throws(() => referenceFor("gwp", 7), /unknown day "7"/);
});

// ---------------------------------------------------------------------------
// Pareto geometry.
//
// Convention throughout: MINIMISE x (GWP or cost), MAXIMISE y (strength).
// A point i is dominated by j iff x_j <= x_i and y_j >= y_i with one strict.
// Consequently a non-dominated set sorted by x ascending has y ascending too,
// and the covered region is a RISING staircase.
// ---------------------------------------------------------------------------

test("paretoStaircase returns an empty front for empty input", () => {
  assert.deepEqual(paretoStaircase([], []), []);
});

test("paretoStaircase returns a single point unchanged", () => {
  assert.deepEqual(paretoStaircase([150], [8000]), [{ x: 150, y: 8000, i: 0 }]);
});

test("paretoStaircase keeps both points of a genuine tradeoff", () => {
  // Cheaper-but-weaker and dearer-but-stronger: neither dominates.
  assert.deepEqual(paretoStaircase([100, 150], [5000, 8000]), [
    { x: 100, y: 5000, i: 0 },
    { x: 150, y: 8000, i: 1 },
  ]);
});

test("paretoStaircase drops a dominated point", () => {
  // (150, 5000) is worse on both axes than (100, 8000).
  assert.deepEqual(paretoStaircase([100, 150], [8000, 5000]), [
    { x: 100, y: 8000, i: 0 },
  ]);
});

test("paretoStaircase sorts unsorted input", () => {
  const front = paretoStaircase([150, 100, 175], [8000, 5000, 9000]);
  assert.deepEqual(front.map((p) => p.x), [100, 150, 175]);
  assert.deepEqual(front.map((p) => p.i), [1, 0, 2]);
});

test("paretoStaircase breaks a tie in x by keeping the higher y", () => {
  // Same GWP, different strength: the weaker mix is strictly dominated, so a
  // naive x-only sort that kept both would emit a zero-width segment.
  assert.deepEqual(paretoStaircase([100, 100], [5000, 8000]), [
    { x: 100, y: 8000, i: 1 },
  ]);
});

test("paretoStaircase collapses exact duplicates to one point", () => {
  assert.deepEqual(paretoStaircase([100, 100], [8000, 8000]), [
    { x: 100, y: 8000, i: 0 },
  ]);
});

test("paretoStaircase reports original indices so the view can highlight mixes", () => {
  const front = paretoStaircase([300, 100, 200], [1000, 9000, 2000]);
  assert.deepEqual(front, [{ x: 100, y: 9000, i: 1 }]);
});

test("paretoStaircase rejects mismatched array lengths", () => {
  // xs and ys arrive from separate sources (gwp_predictions vs a GP posterior),
  // so a length mismatch is a realistic wiring bug and must not read undefined.
  assert.throws(() => paretoStaircase([1, 2], [1]), /length/i);
});

test("hypervolume2D is zero for an empty front", () => {
  assert.equal(hypervolume2D([], [], 200, 5000), 0);
});

test("hypervolume2D rejects mismatched array lengths", () => {
  // Without this guard a short ys reads undefined, `undefined >= refY` is
  // false, and the point is silently dropped -- a wrong number with no symptom.
  assert.throws(() => hypervolume2D([1, 2], [1], 200, 5000), /length/i);
});

test("hypervolume2D computes the rectangle for a single point", () => {
  // (200 - 150) * (8000 - 5000)
  assert.equal(hypervolume2D([150], [8000], 200, 5000), 150000);
});

test("hypervolume2D computes a hand-checked two-step staircase", () => {
  // Front: (100, 6000) then (150, 8000); ref (200, 5000).
  //   x in [100,150): ceiling 6000 -> 50 * 1000 = 50000
  //   x in [150,200): ceiling 8000 -> 50 * 3000 = 150000
  assert.equal(hypervolume2D([100, 150], [6000, 8000], 200, 5000), 200000);
});

test("hypervolume2D ignores dominated points", () => {
  const without = hypervolume2D([100, 150], [6000, 8000], 200, 5000);
  const withDominated = hypervolume2D([100, 150, 180], [6000, 8000, 5500], 200, 5000);
  assert.equal(withDominated, without);
});

test("hypervolume2D is exactly zero when every point is outside the reference box", () => {
  // The real case at iteration 0 with CONCRETE_REFERENCE_POINT: seeds are
  // often too weak or too carbon-intensive to contribute anything. Must be a
  // hard 0, never NaN, or the learning curve axis breaks.
  const hv = hypervolume2D([250, 300], [4000, 4500], 200, 5000);
  assert.equal(hv, 0);
  assert.ok(!Number.isNaN(hv));
});

test("hypervolume2D is zero for a point exactly on the reference point", () => {
  assert.equal(hypervolume2D([200], [5000], 200, 5000), 0);
});

test("hypervolume2D clips a point that is better than the reference on only one axis", () => {
  // Strong enough but too much GWP -> contributes nothing.
  assert.equal(hypervolume2D([250], [9000], 200, 5000), 0);
  // Low GWP but too weak -> contributes nothing.
  assert.equal(hypervolume2D([100], [4000], 200, 5000), 0);
});

test("hypervolume2D never decreases when a point is added", () => {
  const xs = [180, 140, 120, 160, 110];
  const ys = [5500, 6500, 7000, 6000, 9000];
  let prev = 0;
  for (let k = 1; k <= xs.length; k++) {
    const hv = hypervolume2D(xs.slice(0, k), ys.slice(0, k), 200, 5000);
    assert.ok(hv >= prev, `hypervolume dropped at k=${k}: ${hv} < ${prev}`);
    prev = hv;
  }
});

// ---------------------------------------------------------------------------
// Gaussian helpers and the closed-form acquisition function.
// ---------------------------------------------------------------------------

test("normalPdf matches known values and is symmetric", () => {
  assert.ok(Math.abs(normalPdf(0) - 0.3989422804014327) < 1e-15);
  assert.ok(Math.abs(normalPdf(1) - 0.24197072451914337) < 1e-15);
  assert.equal(normalPdf(-2), normalPdf(2));
});

test("normalCdf matches known values to near machine precision", () => {
  // Not exact equality: erfc(0) is 1 only to within rounding, so Phi(0) lands
  // 1.6e-15 off 0.5. Measured relative error across these points is ~1e-15.
  const rel = (got, want) => Math.abs(got - want) / Math.abs(want);
  assert.ok(rel(normalCdf(0), 0.5) < 1e-14);
  assert.ok(rel(normalCdf(1), 0.8413447460685429) < 1e-14);
  assert.ok(rel(normalCdf(-1), 0.15865525393145707) < 1e-14);
  assert.ok(rel(normalCdf(1.96), 0.9750021048517795) < 1e-14);
});

test("normalCdf keeps RELATIVE accuracy in the far tail", () => {
  // The regime that matters: a candidate several sigma below the ceiling still
  // needs a sensibly-scaled EHVI. An erf approximation with only ABSOLUTE
  // accuracy (e.g. A&S 7.1.26 at 1.5e-7) returns garbage here -- Phi(-5) is
  // 2.87e-7, so absolute 1.5e-7 is a ~50% relative error.
  const rel = (got, want) => Math.abs(got - want) / want;
  // Measured: ~4e-15 and ~3e-15 respectively, i.e. this form holds full
  // double precision into the tail, not just the 1.2e-7 NR advertises.
  assert.ok(rel(normalCdf(-5), 2.866515718791939e-7) < 1e-13);
  assert.ok(rel(normalCdf(-7), 1.279812543885835e-12) < 1e-13);
});

test("normalCdf saturates without overflowing", () => {
  assert.equal(normalCdf(40), 1);
  assert.equal(normalCdf(-40), 0);
  assert.ok(normalCdf(-10) > 0, "must underflow to 0 only when truly negligible");
});

test("normalCdf is monotone", () => {
  let prev = -Infinity;
  for (let z = -6; z <= 6; z += 0.25) {
    const v = normalCdf(z);
    assert.ok(v >= prev, `not monotone at z=${z}`);
    prev = v;
  }
});

test("expectedImprovement is the deterministic gap when sd is zero", () => {
  assert.equal(expectedImprovement(5000, 0, 3000), 2000);
  assert.equal(expectedImprovement(2000, 0, 3000), 0);
});

test("expectedImprovement at the threshold is sd/sqrt(2*pi)", () => {
  const got = expectedImprovement(3000, 500, 3000);
  assert.ok(Math.abs(got - 500 * 0.3989422804014327) < 1e-9);
});

test("expectedImprovement approaches the gap when the mean dominates", () => {
  const got = expectedImprovement(9000, 100, 3000);
  assert.ok(Math.abs(got - 6000) < 1e-6);
});

test("expectedImprovement is never negative, even far below the threshold", () => {
  const got = expectedImprovement(1000, 200, 9000);
  assert.ok(got >= 0, `EI must not go negative, got ${got}`);
  assert.ok(got < 1e-6, "and should be negligible this far below");
});

test("expectedImprovement increases with both the mean and the uncertainty", () => {
  assert.ok(expectedImprovement(3100, 500, 3000) > expectedImprovement(3000, 500, 3000));
  assert.ok(expectedImprovement(3000, 800, 3000) > expectedImprovement(3000, 500, 3000));
});

test("expectedHVI reduces to the exact hypervolume improvement as sd -> 0", () => {
  const xs = [100, 150];
  const ys = [6000, 8000];
  const front = paretoStaircase(xs, ys);
  const base = hypervolume2D(xs, ys, 200, 5000);
  for (const [g, mu] of [[120, 9000], [90, 7000], [180, 8500], [130, 5500]]) {
    const exact = hypervolume2D([...xs, g], [...ys, mu], 200, 5000) - base;
    const got = expectedHVI(mu, 0, g, front, 200, 5000);
    assert.ok(Math.abs(got - exact) < 1e-9, `g=${g} mu=${mu}: ${got} vs ${exact}`);
  }
});

test("expectedHVI agrees with Monte Carlo", () => {
  const xs = [100, 150];
  const ys = [6000, 8000];
  const front = paretoStaircase(xs, ys);
  const base = hypervolume2D(xs, ys, 200, 5000);
  const randn = makeNormalSampler(12345);
  const N = 200000;
  for (const [g, mu, sd] of [[120, 7000, 1500], [90, 5200, 2000], [175, 9000, 800]]) {
    let sum = 0;
    for (let k = 0; k < N; k++) {
      const y = mu + sd * randn();
      sum += hypervolume2D([...xs, g], [...ys, y], 200, 5000) - base;
    }
    const mc = sum / N;
    const cf = expectedHVI(mu, sd, g, front, 200, 5000);
    // 1% is far outside MC noise at N=2e5 but well inside formula-error scale.
    assert.ok(Math.abs(cf - mc) / Math.max(mc, 1) < 0.01, `g=${g}: cf=${cf} mc=${mc}`);
  }
});

test("expectedHVI is zero for a candidate worse than the reference on x", () => {
  const front = paretoStaircase([100, 150], [6000, 8000]);
  assert.equal(expectedHVI(20000, 500, 250, front, 200, 5000), 0);
});

test("expectedHVI is zero when the candidate cannot beat the ceiling", () => {
  const front = paretoStaircase([100], [9000]);
  // g=150 sits under a ceiling of 9000; a mean of 2000 with sd 100 is ~70 sigma
  // short, so the expectation underflows to exactly zero.
  assert.equal(expectedHVI(2000, 100, 150, front, 200, 5000), 0);
});

test("expectedHVI treats an empty front as the bare reference box", () => {
  // With nothing observed the ceiling is refY across the whole width.
  const got = expectedHVI(8000, 1000, 150, [], 200, 5000);
  const want = (200 - 150) * expectedImprovement(8000, 1000, 5000);
  assert.ok(Math.abs(got - want) < 1e-9);
});

test("expectedHVI is never negative and increases with the mean", () => {
  const front = paretoStaircase([100, 150], [6000, 8000]);
  let prev = -1;
  for (let mu = 3000; mu <= 12000; mu += 500) {
    const v = expectedHVI(mu, 700, 130, front, 200, 5000);
    assert.ok(v >= 0, `negative EHVI at mu=${mu}`);
    assert.ok(v >= prev, `not monotone at mu=${mu}`);
    prev = v;
  }
});

test("expectedHVI ignores front points outside the reference box", () => {
  const clean = paretoStaircase([100, 150], [6000, 8000]);
  // Same front plus two points the reference point excludes.
  const dirty = paretoStaircase([100, 150, 260, 80], [6000, 8000, 12000, 100]);
  assert.equal(
    expectedHVI(9000, 600, 130, dirty, 200, 5000),
    expectedHVI(9000, 600, 130, clean, 200, 5000),
  );
});

// ---------------------------------------------------------------------------
// Incremental Cholesky.
//
// L is row-major lower-triangular in a flat buffer with leading dimension `ld`,
// so the conditioning set can grow in place without reallocating. The [n x b]
// blocks are column-major, matching gp_v2_fast.mjs's kernel layout.
// ---------------------------------------------------------------------------

test("forwardSolve returns B unchanged for an identity L", () => {
  const L = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const B = new Float64Array([1, 2, 3, 4, 5, 6]); // [3 x 2] column-major
  assert.deepEqual(Array.from(forwardSolve(L, 3, 3, B, 2)), [1, 2, 3, 4, 5, 6]);
});

test("forwardSolve solves a hand-checked lower-triangular system", () => {
  // L = [[2,0],[3,4]], b = [2, 11]  ->  x = [1, 2]
  const L = new Float64Array([2, 0, 3, 4]);
  const B = new Float64Array([2, 11]);
  const X = forwardSolve(L, 2, 2, B, 1);
  assert.ok(Math.abs(X[0] - 1) < 1e-12);
  assert.ok(Math.abs(X[1] - 2) < 1e-12);
});

test("forwardSolve honours a leading dimension larger than n", () => {
  // The BO state preallocates L at full capacity and uses the top-left block.
  const ld = 5;
  const L = new Float64Array(ld * ld);
  L[0 * ld + 0] = 2;
  L[1 * ld + 0] = 3;
  L[1 * ld + 1] = 4;
  const X = forwardSolve(L, ld, 2, new Float64Array([2, 11]), 1);
  assert.ok(Math.abs(X[0] - 1) < 1e-12);
  assert.ok(Math.abs(X[1] - 2) < 1e-12);
});

test("choleskySmall factors a hand-checked 2x2", () => {
  // A = [[4,2],[2,10]] -> L = [[2,0],[1,3]]
  const L = choleskySmall(new Float64Array([4, 2, 2, 10]), 2);
  assert.ok(Math.abs(L[0] - 2) < 1e-12);
  assert.equal(L[1], 0);
  assert.ok(Math.abs(L[2] - 1) < 1e-12);
  assert.ok(Math.abs(L[3] - 3) < 1e-12);
});

test("choleskySmall handles the 1x1 case", () => {
  const L = choleskySmall(new Float64Array([9]), 1);
  assert.ok(Math.abs(L[0] - 3) < 1e-12);
});

test("choleskySmall reconstructs random SPD matrices", () => {
  for (const m of [1, 2, 3, 5, 8]) {
    const A = randomSPD(m, 1000 + m);
    const L = choleskySmall(A, m);
    assert.ok(reconstructionError(L, m, m, A) < 1e-9, `m=${m}`);
  }
});

test("choleskySmall recovers from a singular block via jitter escalation", () => {
  // Two identical observations of the same mix produce exactly this: a rank-
  // deficient Gram block. gp.mjs::cholesky escalates jitter from 1e-8; so do we.
  const A = new Float64Array([1, 1, 1, 1]);
  const L = choleskySmall(A, 2);
  assert.ok(Number.isFinite(L[3]) && L[3] > 0, "jitter must make the block PD");
  assert.ok(reconstructionError(L, 2, 2, A) < 1e-5, "and stay close to the original");
});

test("choleskySmall throws rather than returning NaN on a hopeless block", () => {
  const A = new Float64Array([-1e6, 0, 0, -1e6]);
  assert.throws(() => choleskySmall(A, 2), /Cholesky/i);
});

test("extendCholeskyBlock reproduces a direct factorisation of the full matrix", () => {
  // The core correctness claim of the whole feature.
  for (const [n, b] of [[6, 1], [6, 3], [10, 4], [10, 5], [1, 1], [1, 5]]) {
    const m = n + b;
    const A = randomSPD(m, 7000 + m * 13 + b);
    const ld = m + 4; // deliberately padded, as the live state will be

    // Factor the leading n x n block.
    const Ann = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Ann[i * n + j] = A[i * m + j];
    const Lnn0 = choleskySmall(Ann, n);
    const L = new Float64Array(ld * ld);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) L[i * ld + j] = Lnn0[i * n + j];

    // Kna [n x b] column-major, Knn [b x b] row-major.
    const Kna = new Float64Array(n * b);
    for (let c = 0; c < b; c++) for (let i = 0; i < n; i++) Kna[c * n + i] = A[i * m + (n + c)];
    const Knn = new Float64Array(b * b);
    for (let i = 0; i < b; i++) for (let j = 0; j < b; j++) Knn[i * b + j] = A[(n + i) * m + (n + j)];

    const { Lb, Lnn } = extendCholeskyBlock(L, ld, n, Kna, Knn, b);
    appendCholeskyBlock(L, ld, n, Lb, Lnn, b);

    assert.ok(
      reconstructionError(L, ld, m, A) < 1e-8,
      `n=${n} b=${b}: reconstruction error ${reconstructionError(L, ld, m, A)}`,
    );
  }
});

test("extendCholeskyBlock from an empty conditioning set is a plain factorisation", () => {
  const b = 3;
  const A = randomSPD(b, 4242);
  const ld = b;
  const L = new Float64Array(ld * ld);
  const { Lb, Lnn } = extendCholeskyBlock(L, ld, 0, new Float64Array(0), A, b);
  assert.equal(Lb.length, 0);
  appendCholeskyBlock(L, ld, 0, Lb, Lnn, b);
  assert.ok(reconstructionError(L, ld, b, A) < 1e-9);
});

// ---------------------------------------------------------------------------
// Kernel evaluation on post-transform rows.
//
// bo.mjs needs K between rows that are ALREADY in X_train's post-transform
// space -- an access pattern neither gp.mjs (scalar, arbitrary vectors) nor
// gp_v2_fast.mjs (one composition across many times) has. Rather than perform
// surgery on gp_v2_fast's deliberately-inlined hot loop, bo.mjs has its own
// block builder, and these tests pin it to gp.mjs's canonical scalar kernel so
// the two cannot silently diverge on a schema change.
// ---------------------------------------------------------------------------

test("kernelBlock agrees with gp.mjs's canonical scalar kernel", () => {
  const rowsA = [0, 5, 100, 331, 669];
  const rowsB = [1, 42, 500];
  const A = Float64Array.from(rowsA.flatMap((r) => strengthParams.X_train[r]));
  const B = Float64Array.from(rowsB.flatMap((r) => strengthParams.X_train[r]));
  const K = kernelBlock(A, rowsA.length, B, rowsB.length, D_AUG, strengthParams);

  for (let c = 0; c < rowsB.length; c++) {
    for (let i = 0; i < rowsA.length; i++) {
      const want = gpKernel(
        strengthParams.X_train[rowsA[i]],
        strengthParams.X_train[rowsB[c]],
        strengthParams,
      );
      const got = K[c * rowsA.length + i]; // column-major
      assert.ok(
        Math.abs(got - want) <= 1e-12 * Math.max(1, Math.abs(want)),
        `row ${rowsA[i]} vs ${rowsB[c]}: ${got} != ${want}`,
      );
    }
  }
});

test("kernelBlock is symmetric on the diagonal block", () => {
  const rows = [3, 77, 400];
  const A = Float64Array.from(rows.flatMap((r) => strengthParams.X_train[r]));
  const K = kernelBlock(A, rows.length, A, rows.length, D_AUG, strengthParams);
  for (let i = 0; i < rows.length; i++) {
    for (let j = 0; j < rows.length; j++) {
      assert.ok(Math.abs(K[j * rows.length + i] - K[i * rows.length + j]) < 1e-15);
    }
  }
});

test("selfKernel matches the full kernel evaluated at a point against itself", () => {
  // gp_v2_fast.mjs shortcuts k(x,x) to (blindOS + specificOS + rbfOS) * h(t)^2
  // because every squared distance vanishes. Confirm the shortcut is exact.
  for (const r of [0, 17, 250, 669]) {
    const row = Float64Array.from(strengthParams.X_train[r]);
    const want = gpKernel(strengthParams.X_train[r], strengthParams.X_train[r], strengthParams);
    const got = selfKernel(row, 0, strengthParams);
    assert.ok(Math.abs(got - want) <= 1e-12 * want, `row ${r}: ${got} != ${want}`);
  }
});

// ---------------------------------------------------------------------------
// Candidate set: mapping training rows back onto catalogue mixes.
// ---------------------------------------------------------------------------

test("buildCandidateSet maps every training row to a catalogue mix", () => {
  const cs = buildCandidateSet(strengthParams, compositionsData, 28);
  assert.equal(cs.rowMix.length, strengthParams.n_train);
  for (let r = 0; r < cs.rowMix.length; r++) {
    assert.ok(
      cs.rowMix[r] >= 0 && cs.rowMix[r] < compositionsData.n_compositions,
      `row ${r} did not map to a mix`,
    );
  }
});

test("buildCandidateSet finds 147 day-28 candidates out of 149 mixes", () => {
  const cs = buildCandidateSet(strengthParams, compositionsData, 28);
  assert.equal(cs.mix.length, 147);
  assert.equal(new Set(cs.mix).size, 147, "each mix must appear at most once");
});

test("buildCandidateSet also works for day 1", () => {
  const cs = buildCandidateSet(strengthParams, compositionsData, 1);
  assert.equal(cs.mix.length, 147);
});

test("buildCandidateSet excludes mixes lacking an observation at the target day", () => {
  const at28 = new Set(buildCandidateSet(strengthParams, compositionsData, 28).mix);
  for (let m = 0; m < compositionsData.n_compositions; m++) {
    const hasDay28 = compositionsData.observations[String(m)].some(([d]) => d === 28);
    assert.equal(at28.has(m), hasDay28, `mix ${m} inclusion disagrees with its observations`);
  }
});

test("buildCandidateSet groups every observation row under its mix", () => {
  const cs = buildCandidateSet(strengthParams, compositionsData, 28);
  let total = 0;
  for (let k = 0; k < cs.mix.length; k++) {
    const rows = cs.rowsOfCandidate[k];
    total += rows.length;
    assert.ok(rows.length >= 3 && rows.length <= 5, `mix ${cs.mix[k]} has ${rows.length} rows`);
    for (const r of rows) assert.equal(cs.rowMix[r], cs.mix[k]);
  }
  // 147 of the 149 mixes, so a little short of the full 670 rows.
  assert.ok(total > 640 && total <= strengthParams.n_train, `grouped ${total} rows`);
});

test("buildCandidateSet recovers the measured strength at the target day", () => {
  const cs = buildCandidateSet(strengthParams, compositionsData, 28);
  for (let k = 0; k < cs.mix.length; k += 17) {
    const obs = compositionsData.observations[String(cs.mix[k])].filter(([d]) => d === 28);
    const want = Math.max(...obs.map(([, psi]) => psi));
    assert.ok(Math.abs(cs.observedY[k] - want) < 1e-6, `mix ${cs.mix[k]}`);
  }
});

test("buildCandidateSet carries the exact GWP and cost for each candidate", () => {
  const cs = buildCandidateSet(strengthParams, compositionsData, 28);
  for (let k = 0; k < cs.mix.length; k += 23) {
    assert.ok(Math.abs(cs.gwp[k] + compositionsData.gwp_predictions[cs.mix[k]]) < 1e-9);
    assert.ok(Math.abs(cs.cost[k] + compositionsData.cost_predictions[cs.mix[k]]) < 1e-9);
    assert.ok(cs.gwp[k] > 0 && cs.cost[k] > 0, "stored negated in JSON, positive here");
  }
});

test("buildCandidateSet rejects an unsupported day rather than returning an empty set", () => {
  assert.throws(() => buildCandidateSet(strengthParams, compositionsData, 7), /day/i);
});

test("buildCandidateSet defaults log_time_offset to 1.0 when absent", () => {
  // gp.mjs and gp_v2_fast.mjs both read this field as `|| 1.0`, treating it as
  // optional in the schema; bo.mjs matches so the three cannot disagree about
  // what a missing field means. The deployed model happens to ship 1.0, so
  // dropping it must be a no-op.
  const noOffset = JSON.parse(JSON.stringify(strengthParams));
  delete noOffset.log_time_offset;
  const a = buildCandidateSet(strengthParams, compositionsData, 28);
  const b = buildCandidateSet(noOffset, compositionsData, 28);
  assert.deepEqual(b.mix, a.mix);
});

test("buildCandidateSet fails loudly if the two artifacts have drifted apart", () => {
  // strength.json's X_train and compositions.json must describe the same mixes.
  // If a regeneration updates one and not the other, rows stop matching, and
  // silently dropping them would corrupt every posterior in the run.
  const drifted = JSON.parse(JSON.stringify(compositionsData));
  drifted.compositions = drifted.compositions.map((c) => c.map((v) => v + 1));
  assert.throws(
    () => buildCandidateSet(strengthParams, drifted, 28),
    /does not match any catalogue mix/,
  );
});

test("buildCandidateSet fails loudly on a degenerate normalisation range", () => {
  // hi == lo would make the un-normalisation divide by zero and every row map
  // to the wrong mix, silently.
  const broken = JSON.parse(JSON.stringify(strengthParams));
  broken.normalize_upper[0] = broken.normalize_lower[0];
  assert.throws(
    () => buildCandidateSet(broken, compositionsData, 28),
    /normalis|normaliz|degenerate/i,
  );
});

// ---------------------------------------------------------------------------
// The BO loop.
// ---------------------------------------------------------------------------

const CAND28 = buildCandidateSet(strengthParams, compositionsData, 28);
const REF28 = referenceFor("gwp", 28);

function freshState(seedCount = 3, seed = 0) {
  return createBOState({
    strengthParams,
    candidates: CAND28,
    objectiveX: CAND28.gwp,
    refX: REF28.refX,
    refY: REF28.refY,
    seedMixes: chooseSeeds(CAND28.mix.length, seedCount, seed),
  });
}

test("makeRng is deterministic and stays in [0,1)", () => {
  const a = makeRng(42);
  const b = makeRng(42);
  for (let i = 0; i < 50; i++) {
    const v = a();
    assert.equal(v, b());
    assert.ok(v >= 0 && v < 1);
  }
  assert.notEqual(makeRng(43)(), makeRng(42)());
});

test("chooseSeeds is deterministic, in range, and without repeats", () => {
  const s = chooseSeeds(147, 3, 0);
  assert.equal(s.length, 3);
  assert.equal(new Set(s).size, 3);
  for (const k of s) assert.ok(k >= 0 && k < 147);
  assert.deepEqual(chooseSeeds(147, 3, 0), s);
});

test("chooseSeeds cannot ask for more seeds than there are candidates", () => {
  assert.throws(() => chooseSeeds(3, 5, 0), /seed/i);
});

test("candidateNoiseVariance is positive and gated", () => {
  const nv = candidateNoiseVariance(CAND28, strengthParams);
  assert.equal(nv.length, CAND28.mix.length);
  for (const v of nv) assert.ok(v > 0 && v <= strengthParams.noise * 1.000001);
});

test("candidateNoiseVariance falls back to gate_tau when noise_gate_tau is absent", () => {
  // gp_v2_fast.mjs keeps the two gates separate in case a future variant
  // detunes them; today both are 0.05, so dropping one must change nothing.
  const noTau = JSON.parse(JSON.stringify(strengthParams));
  delete noTau.noise_gate_tau;
  assert.deepEqual(
    Array.from(candidateNoiseVariance(CAND28, noTau)),
    Array.from(candidateNoiseVariance(CAND28, strengthParams)),
  );
});

test("candidateNoiseVariance treats a model with no noise fields as noiseless", () => {
  const bare = JSON.parse(JSON.stringify(strengthParams));
  delete bare.noise_kind;
  delete bare.noise;
  for (const v of candidateNoiseVariance(CAND28, bare)) assert.equal(v, 0);
});

test("createBOState reproduces a from-scratch refit on the seed set", () => {
  const st = freshState(3);
  const rows = st.acquired.flatMap((k) => CAND28.rowsOfCandidate[k]);
  const ref = directPosterior(strengthParams, CAND28, rows);
  assert.ok(maxAbsDiff(st.mu, ref.mu) < 1e-6, `mu drift ${maxAbsDiff(st.mu, ref.mu)}`);
  const sd = Float64Array.from(st.variance, Math.sqrt);
  const refSd = Float64Array.from(ref.variance, Math.sqrt);
  assert.ok(maxAbsDiff(sd, refSd) < 1e-6, `sd drift ${maxAbsDiff(sd, refSd)}`);
});

test("stepBO still matches a from-scratch refit after many acquisitions", () => {
  // The central correctness claim: O(n^2 b) extension must not drift from an
  // O(n^3) refit, iteration after iteration.
  const st = freshState(3);
  for (let i = 0; i < 25; i++) stepBO(st);
  const rows = st.acquired.flatMap((k) => CAND28.rowsOfCandidate[k]);
  const ref = directPosterior(strengthParams, CAND28, rows);
  const muDrift = maxAbsDiff(st.mu, ref.mu);
  const sdDrift = maxAbsDiff(
    Float64Array.from(st.variance, Math.sqrt),
    Float64Array.from(ref.variance, Math.sqrt),
  );
  assert.ok(muDrift < 1e-6, `mean drift ${muDrift} psi after 25 acquisitions`);
  assert.ok(sdDrift < 1e-6, `sd drift ${sdDrift} psi after 25 acquisitions`);
});

test("stepBO picks the argmax of the acquisition values", () => {
  const st = freshState(3);
  const poolBefore = [...st.pool];
  const { picked, acqValues } = stepBO(st);
  let best = 0;
  for (let i = 1; i < acqValues.length; i++) if (acqValues[i] > acqValues[best]) best = i;
  assert.equal(picked, poolBefore[best]);
});

test("stepBO moves exactly one candidate from the pool to acquired", () => {
  const st = freshState(3);
  const nPool = st.pool.length;
  const nAcq = st.acquired.length;
  const { picked } = stepBO(st);
  assert.equal(st.pool.length, nPool - 1);
  assert.equal(st.acquired.length, nAcq + 1);
  assert.ok(st.acquired.includes(picked));
  assert.ok(!st.pool.includes(picked));
});

test("stepBO drives the observed hypervolume up and never down", () => {
  const st = freshState(3);
  let prev = observedHypervolume(st);
  for (let i = 0; i < 30; i++) {
    stepBO(st);
    const hv = observedHypervolume(st);
    assert.ok(hv >= prev - 1e-9, `hypervolume fell at iteration ${i}: ${hv} < ${prev}`);
    prev = hv;
  }
  assert.ok(prev > 0, "BO should find something inside the reference box");
});

test("stepBO collapses a candidate's variance to ~2x the noise floor once acquired", () => {
  const st = freshState(3);
  const before = [...st.variance];
  const { picked } = stepBO(st);
  const noiseFloor = candidateNoiseVariance(CAND28, strengthParams)[picked]
    * strengthParams.y_max * strengthParams.y_max;
  const after = st.variance[picked];

  // Twice, not once, and that is correct rather than a bug. Conditioning on a
  // NOISY observation at exactly this point leaves latent variance
  // k*sigma^2/(k+sigma^2) ~= sigma^2 (since k >> sigma^2 here), and the
  // predictive variance then adds the aleatoric term back on top. So the floor
  // is ~2*sigma^2, not sigma^2.
  assert.ok(after > 0, "must stay positive so EI never divides by a zero sd");
  assert.ok(after < before[picked] * 0.5, "acquiring must sharply reduce uncertainty");
  const ratio = after / noiseFloor;
  assert.ok(ratio > 1.5 && ratio < 2.5, `expected ~2x the noise floor, got ${ratio.toFixed(2)}x`);
});

test("candidateNoiseVariance falls back to ungated noise when the schema says so", () => {
  const global = JSON.parse(JSON.stringify(strengthParams));
  global.noise_kind = "global";
  const nv = candidateNoiseVariance(CAND28, global);
  for (const v of nv) assert.equal(v, global.noise);
});

test("stepBO honours an explicit pick, for the random arm and for replay", () => {
  const st = freshState(3);
  const target = st.pool[7];
  const { picked } = stepBO(st, target);
  assert.equal(picked, target);
});

test("stepBO refuses to acquire a candidate twice", () => {
  // Re-conditioning on rows already in the set makes the Schur complement
  // exactly singular; better to reject it than to lean on jitter.
  const st = freshState(3);
  const already = st.acquired[0];
  assert.throws(() => stepBO(st, already), /already acquired/i);
});

test("stepBO returns null once the pool is exhausted", () => {
  const st = freshState(3);
  let guard = 0;
  while (st.pool.length > 0 && guard++ < 200) stepBO(st);
  assert.equal(st.pool.length, 0);
  const { picked, acqValues } = stepBO(st);
  assert.equal(picked, null);
  assert.equal(acqValues.length, 0);
});

test("observedFront reports the measured tradeoff of the acquired mixes", () => {
  const st = freshState(3);
  stepBO(st);
  const front = observedFront(st);
  for (const p of front) {
    const k = st.acquired.find((c) => CAND28.gwp[c] === p.x);
    assert.notEqual(k, undefined, "front points must come from acquired mixes");
    assert.equal(CAND28.observedY[k], p.y);
  }
});

// ---------------------------------------------------------------------------
// The random-search arm.
// ---------------------------------------------------------------------------

test("randomArmTraces returns ordered percentile bands of the right length", () => {
  const bands = randomArmTraces({
    candidates: CAND28,
    objectiveX: CAND28.gwp,
    refX: REF28.refX,
    refY: REF28.refY,
    seedMixes: chooseSeeds(CAND28.mix.length, 3, 0),
    nIters: 20,
    nRestarts: 50,
    seed: 1,
  });
  assert.equal(bands.p50.length, 20);
  for (let i = 0; i < 20; i++) {
    assert.ok(bands.p10[i] <= bands.p50[i], `p10 > p50 at ${i}`);
    assert.ok(bands.p50[i] <= bands.p90[i], `p50 > p90 at ${i}`);
  }
});

test("randomArmTraces is monotone within each restart", () => {
  const bands = randomArmTraces({
    candidates: CAND28,
    objectiveX: CAND28.gwp,
    refX: REF28.refX,
    refY: REF28.refY,
    seedMixes: chooseSeeds(CAND28.mix.length, 3, 0),
    nIters: 25,
    nRestarts: 40,
    seed: 2,
  });
  for (const key of ["p10", "p50", "p90"]) {
    for (let i = 1; i < 25; i++) {
      assert.ok(bands[key][i] >= bands[key][i - 1] - 1e-9, `${key} fell at ${i}`);
    }
  }
});

// ---------------------------------------------------------------------------
// The fairness contract.
//
// The BO-vs-random comparison is the persuasive claim this feature makes, so
// the two properties it rests on are written down here as executable
// assertions before any of the machinery exists. They are skipped until
// Phase 4 lands `createBOState`, `stepBO` and `randomArmTraces`.
//
//   1. Both arms start from an identical seed set. Giving each arm its own
//      random seeds confounds the comparison -- the throwaway prototype did
//      exactly that and made BO look better than it is.
//   2. Both arms are scored on MEASURED outcomes, never on the GP's own
//      predictions. The model chooses what to acquire; it never grades itself.
//      Test 2 is the sharp one: poison the GP so it returns a constant mean,
//      then assert the hypervolume trace is bit-identical for a fixed
//      acquisition sequence. No coverage metric would ever catch this.
// ---------------------------------------------------------------------------

test("FAIRNESS: both arms start from an identical seed set", () => {
  const seeds = chooseSeeds(CAND28.mix.length, 3, 0);
  const st = freshState(3);
  assert.deepEqual([...st.acquired].sort((a, b) => a - b), [...seeds].sort((a, b) => a - b));

  const bands = randomArmTraces({
    candidates: CAND28,
    objectiveX: CAND28.gwp,
    refX: REF28.refX,
    refY: REF28.refY,
    seedMixes: seeds,
    nIters: 1,
    nRestarts: 8,
    seed: 3,
  });
  // Iteration 0 is scored before either arm has drawn anything, so both must
  // report exactly the seed set's hypervolume -- with no spread across
  // restarts, since they all start from the same three mixes.
  assert.equal(bands.p10[0], observedHypervolume(st));
  assert.equal(bands.p90[0], observedHypervolume(st));
});

test("FAIRNESS: hypervolume is scored on measured outcomes, not predictions", () => {
  // The sharp one. Replay a FIXED acquisition sequence twice: once against the
  // real GP, once against a model poisoned to predict a constant. If any part
  // of the hypervolume were being read off the posterior instead of the lab
  // measurements, the two traces would diverge. No coverage metric catches this.
  const sequence = [11, 40, 77, 5, 132, 90, 3];

  const trace = (sp) => {
    const cand = buildCandidateSet(sp, compositionsData, 28);
    const st = createBOState({
      strengthParams: sp,
      candidates: cand,
      objectiveX: cand.gwp,
      refX: REF28.refX,
      refY: REF28.refY,
      seedMixes: chooseSeeds(cand.mix.length, 3, 0),
    });
    const out = [observedHypervolume(st)];
    for (const k of sequence) {
      if (st.acquired.includes(k)) continue;
      stepBO(st, k);
      out.push(observedHypervolume(st));
    }
    return out;
  };

  const poisoned = JSON.parse(JSON.stringify(strengthParams));
  poisoned.Y_train = poisoned.Y_train.map(() => 0.5);

  assert.deepEqual(trace(poisoned), trace(strengthParams));
});

test("FAIRNESS: the random arm has no access to the model at all", () => {
  // Structural, not conventional: randomArmTraces takes no strengthParams, so
  // it cannot consult the GP even by accident.
  assert.ok(!/strengthParams/.test(randomArmTraces.toString()));
});
