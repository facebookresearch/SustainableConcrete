/**
 * Unit tests for docs/bo_view.mjs — BO presentation logic.
 *
 * Held to 100% line/branch/function coverage by `make test-js-bo`.
 *
 * Run: node --test test/test_js_bo_view.mjs
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  easingFactor,
  easeMeans,
  buildScatterDrawList,
  buildLearningCurveDrawList,
  narrate,
} from "../docs/bo_view.mjs";

// ---------------------------------------------------------------------------
// Easing.
// ---------------------------------------------------------------------------

test("easingFactor snaps immediately under reduced motion", () => {
  assert.equal(easingFactor(16.67, true), 1);
  assert.equal(easingFactor(0, true), 1);
});

test("easingFactor matches the ui.mjs preview easing at one frame", () => {
  // ui.mjs uses 1 - 0.85^(dt/16.67); at exactly one 60fps frame that is 0.15.
  assert.ok(Math.abs(easingFactor(16.67, false) - 0.15) < 1e-12);
});

test("easingFactor is framerate-independent and monotone in dt", () => {
  const short = easingFactor(8, false);
  const long = easingFactor(33, false);
  assert.ok(short > 0 && short < 1);
  assert.ok(long > short, "a longer frame must advance further");
  assert.ok(long < 1, "must never overshoot the target");
});

test("easingFactor is zero for a zero-length frame", () => {
  assert.equal(easingFactor(0, false), 0);
});

test("easeMeans moves the display towards the target and reports convergence", () => {
  const display = Float64Array.from([0, 100]);
  const target = Float64Array.from([100, 100]);
  const converged = easeMeans(display, target, 0.5);
  assert.equal(display[0], 50);
  assert.equal(display[1], 100);
  assert.equal(converged, false);
});

test("easeMeans snaps and reports converged once inside the epsilon", () => {
  const display = Float64Array.from([99.9999999]);
  const target = Float64Array.from([100]);
  assert.equal(easeMeans(display, target, 0.5), true);
  assert.equal(display[0], 100, "must land exactly on target, not asymptotically near");
});

test("easeMeans with factor 1 lands on the target in a single step", () => {
  const display = Float64Array.from([0, 500]);
  const target = Float64Array.from([100, 100]);
  assert.equal(easeMeans(display, target, 1), true);
  assert.deepEqual(Array.from(display), [100, 100]);
});

// ---------------------------------------------------------------------------
// Scatter draw-list.
// ---------------------------------------------------------------------------

const SCATTER = {
  xs: [100, 150, 175, 250],
  displayY: [6000, 8000, 8200, 9000],
  sd: [0, 0, 1000, 2000],
  observedY: [6100, 8100, 0, 0],
  acquired: [0, 1],
  newest: 1,
  refX: 200,
  refY: 5000,
};

test("buildScatterDrawList marks acquired, newest and unacquired points", () => {
  const dl = buildScatterDrawList(SCATTER);
  assert.equal(dl.points.length, 4);
  assert.equal(dl.points[0].kind, "acquired");
  assert.equal(dl.points[1].kind, "newest");
  assert.equal(dl.points[2].kind, "ghost");
  assert.equal(dl.points[3].kind, "ghost");
});

test("buildScatterDrawList makes ghosts translucent and acquired points solid", () => {
  const dl = buildScatterDrawList(SCATTER);
  assert.equal(dl.points[0].alpha, 1);
  assert.ok(dl.points[2].alpha < 1 && dl.points[2].alpha > 0);
});

test("buildScatterDrawList sizes the uncertainty halo by sd", () => {
  const dl = buildScatterDrawList(SCATTER);
  assert.equal(dl.points[0].halo, 0, "an acquired point with no sd needs no halo");
  assert.ok(dl.points[3].halo > dl.points[2].halo, "more uncertain means a bigger halo");
});

test("buildScatterDrawList caps the halo so one wild point cannot swamp the plot", () => {
  const dl = buildScatterDrawList({ ...SCATTER, sd: [0, 0, 1000, 1e9] });
  for (const p of dl.points) assert.ok(p.halo <= 1, `halo ${p.halo} exceeded the cap`);
});

test("buildScatterDrawList draws the observed front from MEASURED values", () => {
  // The observed staircase must never be built from the posterior; that is the
  // same fairness property the engine enforces, carried into the drawing.
  const dl = buildScatterDrawList(SCATTER);
  assert.deepEqual(dl.observedFront, [
    { x: 100, y: 6100 },
    { x: 150, y: 8100 },
  ]);
});

test("buildScatterDrawList draws the predicted front from the eased means", () => {
  const dl = buildScatterDrawList(SCATTER);
  // Point 3 is at x=250, outside refX=200, so it is clipped from the front.
  assert.deepEqual(dl.predictedFront, [
    { x: 100, y: 6000 },
    { x: 150, y: 8000 },
    { x: 175, y: 8200 },
  ]);
});

test("buildScatterDrawList copes with nothing acquired yet", () => {
  const dl = buildScatterDrawList({ ...SCATTER, acquired: [], newest: null });
  assert.deepEqual(dl.observedFront, []);
  for (const p of dl.points) assert.equal(p.kind, "ghost");
});

test("buildScatterDrawList copes with every point acquired", () => {
  const dl = buildScatterDrawList({ ...SCATTER, acquired: [0, 1, 2, 3], newest: 3 });
  assert.equal(dl.points.filter((p) => p.kind === "ghost").length, 0);
  assert.equal(dl.points.filter((p) => p.kind === "newest").length, 1);
});

test("buildScatterDrawList handles an all-zero sd without dividing by zero", () => {
  const dl = buildScatterDrawList({ ...SCATTER, sd: [0, 0, 0, 0] });
  for (const p of dl.points) {
    assert.equal(p.halo, 0);
    assert.ok(Number.isFinite(p.halo));
  }
});

// ---------------------------------------------------------------------------
// Learning curve.
// ---------------------------------------------------------------------------

const CURVE = {
  boTrace: [0, 0, 1.2e3, 5.05e5, 5.07e5],
  bands: {
    p10: [0, 0, 1e2, 2e3, 1e4],
    p50: [0, 5e2, 2e3, 1e4, 4e4],
    p90: [0, 2e3, 8e3, 4e4, 9e4],
  },
  nIters: 5,
};

test("buildLearningCurveDrawList emits a band and both traces", () => {
  const dl = buildLearningCurveDrawList(CURVE);
  assert.equal(dl.bo.length, 5);
  assert.equal(dl.median.length, 5);
  assert.equal(dl.band.length, 5);
  assert.deepEqual(dl.band[4], { i: 4, lo: 1e4, hi: 9e4 });
});

test("buildLearningCurveDrawList scales the axis to whichever arm is ahead", () => {
  const dl = buildLearningCurveDrawList(CURVE);
  assert.ok(dl.yMax >= 5.07e5, "BO's own trace must fit");
  const randomAhead = buildLearningCurveDrawList({
    ...CURVE,
    boTrace: [0, 0, 0, 0, 0],
  });
  assert.ok(randomAhead.yMax >= 9e4, "and so must the random band");
});

test("buildLearningCurveDrawList survives an all-zero start", () => {
  // The real case under CONCRETE_REFERENCE_POINT: nothing clears the reference
  // point for the first several iterations, so every value is 0. A naive
  // max-based axis would be 0 and every later division would be NaN.
  const dl = buildLearningCurveDrawList({
    boTrace: [0, 0],
    bands: { p10: [0, 0], p50: [0, 0], p90: [0, 0] },
    nIters: 2,
  });
  assert.ok(dl.yMax > 0, "axis must stay positive");
  assert.ok(Number.isFinite(dl.yMax));
});

test("buildLearningCurveDrawList only draws as far as the run has got", () => {
  const dl = buildLearningCurveDrawList({ ...CURVE, boTrace: [0, 1e3] });
  assert.equal(dl.bo.length, 2, "the BO trace stops at the current iteration");
  assert.equal(dl.band.length, 5, "the precomputed band still spans the full run");
});

// ---------------------------------------------------------------------------
// Narration.
// ---------------------------------------------------------------------------

test("narrate describes an ordinary iteration", () => {
  const s = narrate({
    iteration: 12,
    label: "Mix C28",
    ehvi: 34000,
    hypervolume: 5.4e5,
    previousHypervolume: 5.0e5,
    poolExhausted: false,
  });
  assert.match(s, /12/);
  assert.match(s, /Mix C28/);
  assert.match(s, /8%/, "should report the hypervolume gain");
});

test("narrate says so when an acquisition adds nothing", () => {
  const s = narrate({
    iteration: 3,
    label: "Mix A1",
    ehvi: 0,
    hypervolume: 1000,
    previousHypervolume: 1000,
    poolExhausted: false,
  });
  assert.match(s, /no change|0%/i);
});

test("narrate handles growth from a zero baseline without dividing by zero", () => {
  // Iteration 0 hypervolume is genuinely 0 whenever the seeds miss the
  // reference box, so the first real gain is an infinite percentage increase.
  const s = narrate({
    iteration: 1,
    label: "Mix B2",
    ehvi: 500,
    hypervolume: 1200,
    previousHypervolume: 0,
    poolExhausted: false,
  });
  assert.ok(!/NaN|Infinity/.test(s), `narration leaked a non-finite number: ${s}`);
  assert.match(s, /first/i);
});

test("narrate reports the terminal state", () => {
  const s = narrate({
    iteration: 147,
    label: null,
    ehvi: 0,
    hypervolume: 5.07e5,
    previousHypervolume: 5.07e5,
    poolExhausted: true,
  });
  assert.match(s, /every mix|exhausted|complete/i);
});

test("narrate describes the state before anything has been acquired", () => {
  const s = narrate({
    iteration: 0,
    label: null,
    ehvi: 0,
    hypervolume: 0,
    previousHypervolume: 0,
    poolExhausted: false,
  });
  assert.match(s, /seed/i);
});
