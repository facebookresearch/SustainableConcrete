/**
 * Presentation logic for BO mode: turns BO state into a draw-list and the
 * narration strings.
 *
 * Pure and DOM-free. This module exists so that the easing state machine, the
 * ghost/halo rules, the staircase vertex construction, the learning-curve band
 * geometry, and the narration text are all unit-testable. Without it they
 * would be trapped inside drawScatter and reachable only through screenshots.
 *
 * ui.mjs consumes these structures and does nothing but execute canvas calls.
 *
 * Run: node --test test/test_js_bo_view.mjs
 */

import { paretoStaircase } from "./bo.mjs";

/**
 * Per-frame easing factor, matching the preview system in ui.mjs: an
 * exponential approach that is framerate-independent, collapsing to an
 * immediate snap under reduced motion.
 */
export function easingFactor(dtMs, reduceMotion) {
  if (reduceMotion) return 1;
  return 1 - Math.pow(0.85, dtMs / 16.67);
}

/** Below this (in psi) a point is treated as arrived, and snapped. */
const EASE_EPS = 1e-6;

/**
 * Advance the displayed means one frame towards the posterior, in place.
 *
 * Snapping inside EASE_EPS matters: an exponential approach never actually
 * arrives, so without it the animation loop would never report convergence and
 * would keep requesting frames forever.
 *
 * @returns true once every point has arrived.
 */
export function easeMeans(display, target, factor) {
  let converged = true;
  for (let i = 0; i < display.length; i++) {
    const delta = target[i] - display[i];
    if (Math.abs(delta) < EASE_EPS) {
      display[i] = target[i];
      continue;
    }
    display[i] += delta * factor;
    if (Math.abs(target[i] - display[i]) < EASE_EPS) display[i] = target[i];
    else converged = false;
  }
  return converged;
}

const GHOST_ALPHA = 0.35;
const MAX_HALO = 1.0;

/**
 * Draw-list for the scatter panel.
 *
 * Two staircases, and the distinction is the whole point of the display:
 *   - `observedFront` comes from MEASURED outcomes of acquired mixes. This is
 *     what the hypervolume score is based on.
 *   - `predictedFront` comes from the model's current (eased) posterior means
 *     over every candidate. This is the frontier that visibly moves.
 *
 * Halo radius is normalised against the largest sd currently on screen and
 * capped, so a single very uncertain candidate cannot swamp the plot.
 */
export function buildScatterDrawList({
  xs, displayY, sd, observedY, acquired, newest, refX, refY,
}) {
  const isAcquired = new Set(acquired);
  let maxSd = 0;
  for (const v of sd) if (v > maxSd) maxSd = v;

  const points = [];
  for (let i = 0; i < xs.length; i++) {
    const acq = isAcquired.has(i);
    points.push({
      i,
      x: xs[i],
      y: displayY[i],
      kind: i === newest ? "newest" : acq ? "acquired" : "ghost",
      alpha: acq ? 1 : GHOST_ALPHA,
      // Zero when nothing is uncertain, so this never divides by zero.
      halo: maxSd > 0 ? Math.min(MAX_HALO, sd[i] / maxSd) : 0,
    });
  }

  const observedFront = paretoStaircase(
    acquired.map((i) => xs[i]),
    acquired.map((i) => observedY[i]),
  )
    .filter((p) => p.x <= refX && p.y >= refY)
    .map((p) => ({ x: p.x, y: p.y }));

  const predictedFront = paretoStaircase(Array.from(xs), Array.from(displayY))
    .filter((p) => p.x <= refX && p.y >= refY)
    .map((p) => ({ x: p.x, y: p.y }));

  return { points, observedFront, predictedFront };
}

/**
 * Draw-list for the hypervolume learning curve.
 *
 * `bands` spans the whole planned run (it is precomputed at mode entry, since
 * random search needs no model), while `boTrace` only reaches the current
 * iteration — so the band is always drawn ahead of the live BO line.
 */
export function buildLearningCurveDrawList({ boTrace, bands, nIters }) {
  let yMax = 0;
  for (const v of boTrace) if (v > yMax) yMax = v;
  for (const v of bands.p90) if (v > yMax) yMax = v;
  // Under CONCRETE_REFERENCE_POINT nothing clears the reference point for the
  // first several iterations, so every value is legitimately 0. A max-based
  // axis would then be 0 and every later normalisation would produce NaN.
  if (!(yMax > 0)) yMax = 1;

  const band = [];
  for (let i = 0; i < nIters; i++) {
    band.push({ i, lo: bands.p10[i], hi: bands.p90[i] });
  }
  return {
    yMax: yMax * 1.05,
    band,
    bo: Array.from(boTrace, (v, i) => ({ i, v })),
    median: Array.from(bands.p50.slice(0, nIters), (v, i) => ({ i, v })),
  };
}

/** One line of narration for the insight panel. */
export function narrate({
  iteration, label, ehvi, hypervolume, previousHypervolume, poolExhausted,
}) {
  if (poolExhausted) {
    return `Complete — every mix in the catalogue has been evaluated. Final hypervolume ${fmt(hypervolume)}.`;
  }
  if (iteration === 0 || label === null) {
    return "Seeded with 3 mixes. Press play to start optimising.";
  }
  const gain = describeGain(hypervolume, previousHypervolume);
  return `Iteration ${iteration} — acquired ${label} (EHVI ${fmt(ehvi)}); hypervolume ${gain}.`;
}

function describeGain(hv, prev) {
  if (hv <= prev) return "no change";
  // prev === 0 is the ordinary case early on, when the seed mixes all sit
  // outside the reference box; a percentage would be infinite.
  if (prev <= 0) return `${fmt(hv)} — first mix inside the reference point`;
  return `+${Math.round(((hv - prev) / prev) * 100)}%`;
}

function fmt(v) {
  if (v === 0) return "0";
  if (Math.abs(v) >= 1e4 || Math.abs(v) < 1e-2) return v.toExponential(1);
  return String(Math.round(v));
}
