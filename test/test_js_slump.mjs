/**
 * Port-equivalence test: docs/slump.mjs (fed by docs/model/slump.json) must
 * reproduce the Python posteriors baked into docs/model/slump_test_vectors.json
 * by experiments/regenerate_slump_json.py.
 *
 * Both files come from a single fit — regenerate them together.
 *
 * Run: node test/test_js_slump.mjs
 */

import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, "..");

const { initSlumpModel, predictSlump, augmentSlumpInput, slumpSupportsSource } =
  await import(join(REPO_ROOT, "docs", "slump.mjs"));

const params = JSON.parse(
  readFileSync(join(REPO_ROOT, "docs", "model", "slump.json"), "utf-8")
);
const golden = JSON.parse(
  readFileSync(join(REPO_ROOT, "docs", "model", "slump_test_vectors.json"), "utf-8")
);

initSlumpModel(params);

const RTOL = 1e-6;
const ATOL = 1e-8;

let passed = 0;
const failures = [];
let worstMean = 0;
let worstVar = 0;

function assertClose(actual, expected, name) {
  if (!Number.isFinite(actual)) {
    failures.push(`${name}: expected ${expected}, got non-finite ${actual}`);
    return 0;
  }
  const absErr = Math.abs(actual - expected);
  const relErr = Math.abs(expected) > ATOL ? absErr / Math.abs(expected) : absErr;
  if (relErr > RTOL && absErr > ATOL) {
    failures.push(
      `${name}: expected ${expected}, got ${actual} ` +
        `(relErr=${relErr.toExponential(2)}, absErr=${absErr.toExponential(2)})`
    );
  } else {
    passed++;
  }
  return relErr;
}

for (const [i, vec] of golden.test_vectors.entries()) {
  const { mean, variance } = predictSlump(vec.input, params);
  worstMean = Math.max(worstMean, assertClose(mean, vec.expected_mean, `vec[${i}].mean`));
  worstVar = Math.max(
    worstVar,
    assertClose(variance, vec.expected_variance, `vec[${i}].variance`)
  );
}

// The derived feature must be hrwr / max(binder, 1.0) — matching
// boxcrete.features.AppendDerivedFeatures — NOT hrwr / (binder + 1.0), which
// is what FEATURE_FNS.hrwr_binder in docs/feature_registry.mjs computes for
// the *strength* model. Mixing them up costs ~0.3% silently.
{
  // [cement, flyAsh, slag, water, hrwr, fine, coarse, source, temp]
  const comp = [263, 0, 108, 130, 2.04, 867, 1129, 2, 22];
  const binder = 263 + 0 + 108;
  const aug = augmentSlumpInput(comp);
  assertClose(aug[9], 2.04 / binder, "hrwr_binder uses max(binder, 1)");
  if (Math.abs(aug[9] - 2.04 / (binder + 1)) < 1e-12) {
    failures.push("hrwr_binder used the +1.0 strength formula");
  } else {
    passed++;
  }
}

// Degenerate all-zero binder must clamp, not divide by zero.
{
  const aug = augmentSlumpInput([0, 0, 0, 0, 3, 0, 0, 1, 22]);
  if (!Number.isFinite(aug[9]) || aug[9] !== 3) {
    failures.push(`zero binder: expected 3 (clamped to 1.0), got ${aug[9]}`);
  } else {
    passed++;
  }
}

// Mortar has no slump data; the UI relies on this gate to render "n/a".
{
  if (slumpSupportsSource(0, params)) {
    failures.push("slumpSupportsSource(0) must be false — mortar has no slump data");
  } else {
    passed++;
  }
  if (slumpSupportsSource(1, params) && slumpSupportsSource(2, params)) {
    passed++;
  } else {
    failures.push("slumpSupportsSource must be true for classes 1 and 2");
  }
}

// Far outside the training hull the posterior must stay finite and widen
// toward the prior rather than producing NaN — the readout renders this.
{
  const far = [894, 533, 1198, 443, 13.3, 2357, 0, 2, 4.5];
  const { mean, variance } = predictSlump(far, params);
  if (!Number.isFinite(mean) || !Number.isFinite(variance) || variance <= 0) {
    failures.push(
      `extrapolation produced mean=${mean} variance=${variance}; must be finite and positive`
    );
  } else {
    passed++;
  }
}

console.log(
  `worst rel err  mean: ${worstMean.toExponential(3)}  var: ${worstVar.toExponential(3)}`
);

if (failures.length) {
  console.error(`FAILED (${failures.length}):`);
  for (const f of failures.slice(0, 20)) console.error("  " + f);
  process.exit(1);
}
console.log(`All ${passed} slump assertions passed.`);
