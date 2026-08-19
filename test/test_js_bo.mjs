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
} from "../docs/bo.mjs";

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

test("FAIRNESS: both arms start from an identical seed set", { skip: "Phase 4" }, () => {
  assert.fail("implement with createBOState + randomArmTraces");
});

test("FAIRNESS: hypervolume is scored on measured outcomes, not predictions", { skip: "Phase 4" }, () => {
  assert.fail("poison the GP mean; the HV trace for a fixed sequence must not move");
});
