/**
 * Schema-v2 lengthscale identifiability test.
 *
 * The strength GP uses an ARD-Matern kernel with the
 * ``WithinGroupShrinkagePrior`` keeping each per-dim lengthscale away
 * from extreme values. A lengthscale at (or near) the upper constraint
 * cap effectively means "this input is inactive" — sliders for that dim
 * in the explorer don't change predictions.
 *
 * This catches the failure mode where the prior is silently dropped
 * during fit / refinement and lengthscales drift to enormous values
 * (we observed ℓ_Fine=170, ℓ_Source=199, ℓ_Temp=384 in the deployed
 * model before fixing the prior-aware refinement bug).
 *
 * Run: node test/test_lengthscales_v2.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const docsRoot = resolve(__dirname, "..", "docs");

const params = JSON.parse(
  readFileSync(resolve(docsRoot, "model/strength.json"), "utf-8"),
);

if (params.schema_version !== 2) {
  console.log(`(skip) strength.json is not schema_version=2 (got ${params.schema_version})`);
  process.exit(0);
}

// Fitting constraint upper bound (LogTransformedInterval(1e-2, 1e3) in
// the kernel builder). Anything above 100 means the dim contributes
// negligibly to the kernel since (post-Normalize) inputs vary in [0, 1]
// — a lengthscale of 100 is 100x the input range, so distances are
// effectively zero and that dim's input is ignored.
//
// Different caps for raw vs engineered features:
//   - Raw dims (cement, water, etc.): must be active. The within-group
//     shrinkage prior keeps these tied to the bulk of the data; a
//     railed lengthscale here is a real bug (the corresponding slider
//     does nothing).
//   - Engineered features (wb_ratio, scm_frac, etc.): may legitimately
//     be redundant with each other (e.g., wb_ratio is partially
//     reconstructible from log_wc_ratio + binder dims). The prior does
//     NOT cover them, so the optimizer is free to mark them inactive.
//     We use a more permissive cap here, flagging only railed values.
const RAW_CAP = 100.0;
const ENGINEERED_CAP = 500.0;
// Hard cap (the LogTransformedInterval upper bound). Hitting this
// means the optimizer is railed against the constraint, which is even
// worse than just "large".
const HARD_CAP = 1000.0;

const allFeatureNames = [
  ...params.raw_feature_names,
  ...params.engineered_feature_names,
];
const sourceDim = params.source_dim_raw;
const blindFeatureNames = allFeatureNames.filter((_, i) => i !== sourceDim);

let nFail = 0;
const failures = [];

function check(condition, msg) {
  if (!condition) { nFail++; failures.push(msg); }
}

const nRawDims = params.raw_feature_names.length;

console.log("Specific Matern lengthscales (active on all 17 dims):");
for (let i = 0; i < params.matern_specific.lengthscales.length; i++) {
  const ls = params.matern_specific.lengthscales[i];
  const name = allFeatureNames[i];
  const isRaw = i < nRawDims;
  const cap = isRaw ? RAW_CAP : ENGINEERED_CAP;
  const flag = ls > cap ? " ❌" : (ls > cap * 0.5 ? " ⚠" : "");
  const tag = isRaw ? "raw " : "eng ";
  console.log(`  [specific ${tag}] ${name.padEnd(25)}: ${ls.toFixed(3).padStart(10)}${flag}`);
  // Cap-rail check: only enforced on RAW features. Engineered features
  // (W/B, SCM frac, log(HRWR/binder), etc.) are derived from the raw
  // composition columns and are not directly user-controllable in the
  // explorer's slider UI — if their lengthscale rails, the kernel is
  // saying "this engineered ratio is redundant with the raw inputs",
  // which is a valid GP fit outcome (and one that empirically lands
  // in different basins on different BLAS implementations: Linux
  // x86_64 sometimes rails wb_ratio at the constraint upper bound
  // while Apple Silicon / amd64-emulated Linux land at a finite
  // lengthscale ~10). Raw-feature rails would mean a slider with no
  // effect on predictions, so we keep the tight check there.
  if (isRaw) {
    check(
      ls < cap,
      `[specific ${tag}] ${name}: ℓ=${ls.toFixed(2)} > cap ${cap} (effectively inactive)`,
    );
    check(
      ls < HARD_CAP * 0.95,
      `[specific ${tag}] ${name}: ℓ=${ls.toFixed(2)} railed at constraint upper bound ${HARD_CAP}`,
    );
  }
}

console.log("");
console.log("Blind Matern lengthscales (excludes Source dim):");
// Blind active dims include all raw except Source, then all engineered.
// Index in blindFeatureNames: < (nRawDims-1) → raw; >= → engineered.
const nBlindRawDims = nRawDims - 1;
for (let i = 0; i < params.matern_blind.lengthscales.length; i++) {
  const ls = params.matern_blind.lengthscales[i];
  const name = blindFeatureNames[i];
  const isRaw = i < nBlindRawDims;
  const cap = isRaw ? RAW_CAP : ENGINEERED_CAP;
  const flag = ls > cap ? " ❌" : (ls > cap * 0.5 ? " ⚠" : "");
  const tag = isRaw ? "raw " : "eng ";
  console.log(`  [blind    ${tag}] ${name.padEnd(25)}: ${ls.toFixed(3).padStart(10)}${flag}`);
  // Cap-rail check: only enforced on RAW features (see specific Matern
  // for rationale).
  if (isRaw) {
    check(
      ls < cap,
      `[blind ${tag}] ${name}: ℓ=${ls.toFixed(2)} > cap ${cap} (effectively inactive)`,
    );
    check(
      ls < HARD_CAP * 0.95,
      `[blind ${tag}] ${name}: ℓ=${ls.toFixed(2)} railed at constraint upper bound ${HARD_CAP}`,
    );
  }
}

console.log("");
console.log(`RBF time lengthscale: ${params.rbf_time.lengthscale.toFixed(3)}`);
// RBF time lengthscale lives in a different scale (post-input-transform
// time is in [0, 1.46]); flag only at the hard cap.
check(
  params.rbf_time.lengthscale < HARD_CAP * 0.95,
  `[rbf_time]: ℓ=${params.rbf_time.lengthscale.toFixed(2)} railed at constraint upper bound`,
);

console.log("");
if (nFail > 0) {
  console.error(`❌ ${nFail} lengthscale identifiability checks failed:`);
  for (const f of failures) console.error("  - " + f);
  console.error(
    `\nLengthscales > ${HARD_CAP} (or near it: > ${0.95 * HARD_CAP}) mean ` +
    "the corresponding input contributes negligibly to the kernel — sliders " +
    "for that dim in the explorer will not affect predictions. The companion " +
    "Playwright gate at test/e2e/lengthscale-identifiability.spec.ts uses " +
    "a tighter 0.99 * cap rail-detection threshold; keep both in sync if you " +
    "loosen one.\n" +
    "\nLikely causes:\n" +
    "  1. WithinGroupShrinkagePrior was dropped during refinement\n" +
    "     (regression in block_loo_loss; ensure include_priors=True).\n" +
    "  2. Data signal in that dim is too weak for the prior strength.\n" +
    "  3. Optimization railed against the LogTransformedInterval cap.",
  );
  process.exit(1);
} else {
  console.log("✅ all lengthscales within identifiability cap.");
}
