/**
 * Unit tests for docs/preview_state.mjs — the ghost-preview composition
 * stepper.
 *
 * Why this exists: the preview curve is predicted directly from this
 * composition, and Material Source is a categorical class. Interpolating it
 * shipped a real bug — for the ~1.4 s the ease took to converge, the Hamming
 * kernel saw fractional classes that match no training row, so the ghost
 * rendered a collapsed "unseen class" posterior instead of the mix being
 * hovered. `test_js_categorical_source.mjs` pins the model-side fact; this
 * file pins the UI-side rule that produces the inputs.
 *
 * Run: node test/test_js_preview_state.mjs
 */

import { stepPreviewComposition, PREVIEW_EPS } from "../docs/preview_state.mjs";

let failures = 0;
function check(name, cond, detail = "") {
  if (cond) {
    console.log(`  ok   ${name}`);
  } else {
    console.error(`  FAIL ${name}${detail ? " — " + detail : ""}`);
    failures++;
  }
}

// Composition layout mirrors the deployed catalog: index 7 is Material Source.
const MS = 7;
const START = [353, 47, 267, 140, 6.7, 1833, 0, 0, 22];
const TARGET = [263, 0, 108, 130, 2.0, 867, 1129, 2, 22];
const FACTOR = 0.15; // ~ the 60 fps easing factor used by the animation loop

console.log("stepPreviewComposition");

// --- 1. The categorical dimension is never fractional, at any frame --------
{
  const display = [...START];
  const seen = new Set();
  let converged = false;
  for (let frame = 0; frame < 400 && !converged; frame++) {
    converged = stepPreviewComposition(display, TARGET, FACTOR, MS);
    seen.add(display[MS]);
  }
  const fractional = [...seen].filter((v) => !Number.isInteger(v));
  check(
    "Material Source is integral on every frame",
    fractional.length === 0,
    `saw ${JSON.stringify(fractional.slice(0, 5))}`,
  );
  check(
    "Material Source lands on the target class",
    display[MS] === TARGET[MS],
    `got ${display[MS]}, want ${TARGET[MS]}`,
  );
  check("continuous dims converge", converged, "loop hit the frame cap");
}

// --- 2. It snaps on the FIRST frame, not gradually -------------------------
{
  const display = [...START];
  stepPreviewComposition(display, TARGET, FACTOR, MS);
  check(
    "Material Source snaps on frame 1",
    display[MS] === TARGET[MS],
    `got ${display[MS]} after one step`,
  );
}

// --- 3. A fractional target class is rounded, never passed through ---------
{
  const display = [...START];
  const weird = [...TARGET];
  weird[MS] = 1.4; // defensive: nothing should ever hand us this
  stepPreviewComposition(display, weird, FACTOR, MS);
  check(
    "a fractional target class is rounded to an integer",
    Number.isInteger(display[MS]) && display[MS] === 1,
    `got ${display[MS]}`,
  );
}

// --- 4. Continuous dims still ease (the categorical rule must not leak) ----
{
  const display = [...START];
  stepPreviewComposition(display, TARGET, FACTOR, MS);
  const cementMoved = display[0] !== START[0] && display[0] !== TARGET[0];
  check("continuous dims ease rather than snap", cementMoved, `cement=${display[0]}`);
  const expected = START[0] + (TARGET[0] - START[0]) * FACTOR;
  check(
    "easing uses the supplied factor",
    Math.abs(display[0] - expected) < 1e-9,
    `got ${display[0]}, want ${expected}`,
  );
}

// --- 5. Convergence reporting ---------------------------------------------
{
  const display = [...TARGET];
  check(
    "already-equal composition reports converged",
    stepPreviewComposition(display, TARGET, FACTOR, MS) === true,
  );

  const near = [...TARGET];
  near[0] += PREVIEW_EPS / 2; // inside the snap threshold
  const conv = stepPreviewComposition(near, TARGET, FACTOR, MS);
  check("within-epsilon dims snap exactly", near[0] === TARGET[0], `got ${near[0]}`);
  check("within-epsilon composition reports converged", conv === true);

  const far = [...TARGET];
  far[0] += 10;
  check(
    "far composition reports not converged",
    stepPreviewComposition(far, TARGET, FACTOR, MS) === false,
  );
}

// --- 6. No categorical dimension (negative index) --------------------------
{
  const display = [...START];
  stepPreviewComposition(display, TARGET, FACTOR, -1);
  check(
    "categoricalIdx < 0 leaves every dim interpolated",
    display[MS] !== START[MS] && display[MS] !== TARGET[MS],
    `got ${display[MS]}`,
  );
}

if (failures > 0) {
  console.error(`\n${failures} check(s) failed.`);
  process.exit(1);
}
console.log("\nAll preview-state checks passed.");
