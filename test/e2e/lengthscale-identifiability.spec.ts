import { test, expect } from "@playwright/test";

/**
 * Block landing if the served strength.json has any feature lengthscale at
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
 *   matern_blind.lengthscales    — over the no-source augmented dims
 *   matern_specific.lengthscales — over all augmented dims
 * Both must stay clear of the cap. The cap (1e3) is hard-coded to mirror the
 * `LogTransformedInterval(1e-2, 1e3, ...)` constraint applied in
 * `boxcrete/strength_model.py::_ard_matern_with_within_group_prior`. Feature
 * names and the dim layout are read directly from the served JSON; missing
 * fields fail loudly so we don't silently fall back to stale local copies.
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
  const sourceDimRaw = params.source_dim_raw as number;

  expect(
    Array.isArray(rawNames) && rawNames.length > 0,
    `served model is missing 'raw_feature_names'. Re-run experiments/regenerate_strength_json.py and commit docs/model/strength.json.`,
  ).toBeTruthy();
  expect(
    Array.isArray(engineeredNames) && engineeredNames.length > 0,
    `served model is missing 'engineered_feature_names'. Re-run experiments/regenerate_strength_json.py and commit docs/model/strength.json.`,
  ).toBeTruthy();
  expect(
    typeof sourceDimRaw === "number" && Number.isFinite(sourceDimRaw),
    `served model is missing 'source_dim_raw'. Re-run experiments/regenerate_strength_json.py and commit docs/model/strength.json.`,
  ).toBeTruthy();

  const augNames = [...rawNames, ...engineeredNames];
  const augNamesNoSource = augNames.filter((_, i) => i !== sourceDimRaw);

  const subkernels: Array<{ key: "matern_blind" | "matern_specific"; names: string[] }> = [
    { key: "matern_specific", names: augNames },
    { key: "matern_blind", names: augNamesNoSource },
  ];

  for (const { key, names } of subkernels) {
    const sub = params[key] as { lengthscales?: number[] };
    expect(
      sub && Array.isArray(sub.lengthscales),
      `served model is missing '${key}.lengthscales'. Re-run experiments/regenerate_strength_json.py and commit docs/model/strength.json.`,
    ).toBeTruthy();
    const ls = sub.lengthscales as number[];
    expect(
      ls.length === names.length,
      `${key}: expected ${names.length} lengthscales, got ${ls.length}`,
    ).toBeTruthy();

    const violations: string[] = [];
    // Number of raw dims for this subkernel (matern_blind drops the
    // source dim, so its raw count is one less). Cap-rail check is only
    // enforced for RAW features — engineered features (W/B, SCM frac,
    // log(HRWR/binder), etc.) are derived from raw composition columns
    // and aren't directly user-controllable in the slider UI; if their
    // lengthscale rails, the kernel is saying "this engineered ratio
    // is redundant with the raw inputs", which is a valid GP fit
    // outcome (and one that empirically lands in different basins on
    // different BLAS implementations).
    const isBlind = key === "matern_blind";
    const nRawForThisSubkernel = rawNames.length - (isBlind ? 1 : 0);
    for (let i = 0; i < names.length; i++) {
      if (i >= nRawForThisSubkernel) {
        continue; // engineered feature — no rail-check
      }
      if (ls[i] >= 0.99 * LENGTHSCALE_CAP) {
        violations.push(`${names[i]} (idx ${i}): ${ls[i].toFixed(2)}`);
      }
    }
    expect(
      violations.length,
      `${key} has feature(s) with non-identifiable lengthscales (≥ ${LENGTHSCALE_CAP}): ${violations.join(", ")}. ` +
        `These sliders will be unresponsive in the website. Re-run experiments/regenerate_strength_json.py and commit the regenerated docs/model/strength.json.`,
    ).toBe(0);
  }
});
