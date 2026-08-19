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
} from "../docs/bo.mjs";

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
