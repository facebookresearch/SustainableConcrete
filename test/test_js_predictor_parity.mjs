// Variance-contract parity: the single-point `predictStrength` and the
// batched `predictStrengthCurve` MUST return the same total variance for
// the same (composition, time). The contract is documented in
// `strength.json::variance_includes_aleatoric: true`, and the regression
// gate exists because the two paths historically diverged by ~5× when one
// added the gated aleatoric `h(t)²·σ²·y_max²` and the other did not.
//
// Run: node test/test_js_predictor_parity.mjs

import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, "..");

const { initStrengthModel, predictStrength, predictStrengthCurve } = await import(
  join(REPO_ROOT, "docs", "gp.mjs")
);

const params = JSON.parse(
  readFileSync(join(REPO_ROOT, "docs", "model", "strength.json"), "utf-8"),
);
initStrengthModel(params);

const composition = [150, 60, 90, 120, 1, 895, 1166, 1, 22];
const times = [0.5, 1, 7, 28];

const curve = predictStrengthCurve(composition, times, params);

const RTOL = 1e-6;
const ATOL = 1e-6;
let passed = 0;
let failed = 0;

for (let i = 0; i < times.length; i++) {
  const single = predictStrength(composition, times[i], params);
  const batchMean = curve.means[i];
  const batchVar = curve.variances[i];
  for (const [name, a, b] of [
    [`t=${times[i]} mean`, single.mean, batchMean],
    [`t=${times[i]} variance`, single.variance, batchVar],
  ]) {
    const absErr = Math.abs(a - b);
    const denom = Math.max(Math.abs(a), Math.abs(b), ATOL);
    const relErr = absErr / denom;
    if (relErr > RTOL && absErr > ATOL) {
      console.error(
        `MISMATCH ${name}: single=${a}, batch=${b}, ` +
          `relErr=${relErr.toExponential(2)}, absErr=${absErr.toExponential(2)}`,
      );
      failed += 1;
    } else {
      passed += 1;
    }
  }
}

console.log(`\nsingle-vs-batch parity: ${passed}/${passed + failed} assertions passed`);
if (failed > 0) {
  console.error(`\n❌ ${failed} assertions FAILED`);
  process.exit(1);
}
console.log("✅ predictStrength and predictStrengthCurve agree.");
