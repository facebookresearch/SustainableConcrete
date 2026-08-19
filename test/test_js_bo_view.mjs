/**
 * Unit tests for docs/bo_view.mjs — BO presentation logic.
 *
 * Held to 100% line/branch/function coverage by `make test-js-bo`.
 *
 * Run: node --test test/test_js_bo_view.mjs
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { easingFactor } from "../docs/bo_view.mjs";

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
