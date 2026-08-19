/**
 * Presentation logic for BO mode: turns BO state into a draw-list and the
 * narration strings.
 *
 * Pure and DOM-free. This module exists so that the easing state machine, the
 * ghost/halo rules, the staircase vertex construction, the learning-curve band
 * geometry, and the narration text are all unit-testable. Without it they
 * would be trapped inside drawScatter and reachable only through screenshots.
 *
 * Run: node --test test/test_js_bo_view.mjs
 */

/**
 * Per-frame easing factor, matching the preview system in ui.mjs: an
 * exponential approach that is framerate-independent, collapsing to an
 * immediate snap under reduced motion.
 */
export function easingFactor(dtMs, reduceMotion) {
  if (reduceMotion) return 1;
  return 1 - Math.pow(0.85, dtMs / 16.67);
}
