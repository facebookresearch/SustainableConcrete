import { test, expect } from "@playwright/test";

/**
 * Block landing if the served strength.json has any RAW feature lengthscale at
 * the optimiser's upper constraint bound. This mirrors
 * test/test_lengthscale_identifiability.py at the WEBSITE artifact level: the
 * Python test guards what the model script will produce; this test guards what
 * the website actually serves to users (the committed `docs/model/strength.json`).
 *
 * If a feature's lengthscale is at or near the cap, the corresponding slider
 * in the Composition panel becomes unresponsive — moving it produces no
 * visible change in the predicted strength curve. That's a silent UX failure
 * the visual tests can't catch.
 *
 * The V2 schema (the deployed V2 strength GP) emits per-subkernel lengthscales:
 *   matern_blind.lengthscales    — source-blind Matern branch
 *   matern_specific.lengthscales — source-aware Matern branch
 * Since the v5 three-class migration, Material Source is a categorical class
 * handled by a separate Hamming/CategoricalKernel factor, so it is excluded
 * from BOTH Matern branches. Each branch therefore carries lengthscales over
 * its own `active_dims` (the augmented dims it actually spans, excluding the
 * source dim). We read `active_dims` straight from the served JSON and map each
 * lengthscale back to its augmented feature name, so this test stays correct
 * even if the dim layout changes again. Missing fields fail loudly so we don't
 * silently fall back to stale assumptions.
 *
 * Rail-cap enforcement is limited to RAW (user-facing) features. Engineered
 * features (W/B, SCM frac, log(HRWR/binder), …) are derived from raw columns,
 * aren't directly slider-controllable, and may legitimately rail on some BLAS
 * backends (a valid "this ratio is redundant with the raw inputs" GP outcome).
 *
 * The artifact is emitted by `experiments/regenerate_strength_json.py`.
 */

// Mirrors `_LENGTHSCALE_CAP` in test/test_lengthscale_identifiability.py
// (LogTransformedInterval upper bound on the Matern lengthscale constraint).
const LENGTHSCALE_CAP = 1e3;

test("served strength.json has identifiable lengthscales for every feature", async ({ request }) => {
  const resp = await request.get("/model/strength.json");
  expect(resp.ok(), `failed to fetch /model/strength.json: ${resp.status()}`).toBeTruthy();
  const params = await resp.json();

  const rawNames = params.raw_feature_names as string[];
  const engineeredNames = params.engineered_feature_names as string[];

  expect(
    Array.isArray(rawNames) && rawNames.length > 0,
    `served model is missing 'raw_feature_names'. Re-run experiments/regenerate_strength_json.py and commit docs/model/strength.json.`,
  ).toBeTruthy();
  expect(
    Array.isArray(engineeredNames) && engineeredNames.length > 0,
    `served model is missing 'engineered_feature_names'. Re-run experiments/regenerate_strength_json.py and commit docs/model/strength.json.`,
  ).toBeTruthy();

  // Augmented feature names, indexed by augmented dim: raw dims first
  // (indices [0, nRaw)), then engineered features. `active_dims` indexes into
  // this list. Raw dims are the user-facing sliders subject to the rail cap.
  const augNames = [...rawNames, ...engineeredNames];
  const nRaw = rawNames.length;

  for (const key of ["matern_blind", "matern_specific"] as const) {
    const sub = params[key] as { lengthscales?: number[]; active_dims?: number[] };
    expect(
      sub && Array.isArray(sub.lengthscales),
      `served model is missing '${key}.lengthscales'. Re-run experiments/regenerate_strength_json.py and commit docs/model/strength.json.`,
    ).toBeTruthy();
    expect(
      sub && Array.isArray(sub.active_dims),
      `served model is missing '${key}.active_dims'. Re-run experiments/regenerate_strength_json.py and commit docs/model/strength.json.`,
    ).toBeTruthy();
    const ls = sub.lengthscales as number[];
    const dims = sub.active_dims as number[];
    expect(
      ls.length === dims.length,
      `${key}: expected one lengthscale per active dim (${dims.length}), got ${ls.length}`,
    ).toBeTruthy();

    const violations: string[] = [];
    for (let p = 0; p < dims.length; p++) {
      const augDim = dims[p];
      if (augDim >= nRaw) continue; // engineered feature — no rail-check
      if (ls[p] >= 0.99 * LENGTHSCALE_CAP) {
        violations.push(`${augNames[augDim]} (aug dim ${augDim}): ${ls[p].toFixed(2)}`);
      }
    }
    expect(
      violations.length,
      `${key} has feature(s) with non-identifiable lengthscales (≥ ${LENGTHSCALE_CAP}): ${violations.join(", ")}. ` +
        `These sliders will be unresponsive in the website. Re-run experiments/regenerate_strength_json.py and commit the regenerated docs/model/strength.json.`,
    ).toBe(0);
  }
});
