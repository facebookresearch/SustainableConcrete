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

import { REFERENCE_POINT, referenceFor } from "../docs/bo.mjs";

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
