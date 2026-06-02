/**
 * Physical-constraint regression test for the JS strength GP.
 *
 * Verifies that the deployed JS strength model satisfies the
 * physics constraint::
 *
 *     f(x, t = 0) = 0   for every composition x
 *
 * The Python recommended model uses a multiplicatively-gated kernel
 * (`k_gated = h(t) * k_base * h(t')` with `h(0) = 0`) plus
 * multiplicative-only Y scaling and `ZeroMean` to produce exactly 0
 * at t=0. The JS port in `docs/gp.mjs` must replicate this property
 * once it is updated to use the gated kernel.
 *
 * Regression guard for the gated-kernel JS port: any change that breaks
 * the structural f(x, t=0) = 0 invariant on the JS side will fail this
 * test. See `docs/model/README.md` for the schema and
 * `experiments/STRENGTH_GP_BENCHMARK.md` for the architecture writeup.
 *
 * Run: node test/test_js_physical_constraints.mjs
 */

import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, "..");

// Import GP library
const gpModule = await import(join(REPO_ROOT, "docs", "gp.mjs"));
const { predictStrength, initStrengthModel } = gpModule;

// Load model parameters and test vectors
const strengthParams = JSON.parse(
  readFileSync(join(REPO_ROOT, "docs", "model", "strength.json"), "utf-8")
);
const testData = JSON.parse(
  readFileSync(join(REPO_ROOT, "docs", "model", "test_vectors.json"), "utf-8")
);

initStrengthModel(strengthParams);

// Constraint tolerance: 1.0 psi (matches the Python test)
const PHYSICS_CONSTRAINT_TOLERANCE_PSI = 1.0;

let passed = 0;
let failed = 0;
const failureReports = [];

for (const [idx, vec] of testData.test_vectors.entries()) {
  // Compatibility: new test_vectors use `composition`; old used `input`.
  const composition = vec.composition || vec.input;

  try {
    const predAtZero = predictStrength(composition, 0, strengthParams);
    const absMean = Math.abs(predAtZero.mean);

    if (absMean > PHYSICS_CONSTRAINT_TOLERANCE_PSI) {
      throw new Error(
        `Physics constraint VIOLATED for vec[${idx}]: ` +
          `predicted strength at t=0 is ${predAtZero.mean.toFixed(2)} psi, ` +
          `expected |pred| < ${PHYSICS_CONSTRAINT_TOLERANCE_PSI}. ` +
          `The JS model must enforce f(x, 0) = 0 via the gated kernel.`
      );
    }

    // The variance at t=0 should also be near-zero with the gated kernel
    // (h(0)=0 makes prior covariance at t=0 exactly zero). Allow some slack
    // for floating-point and any noise contribution at test time.
    const stdAtZero = Math.sqrt(Math.max(0, predAtZero.variance));
    if (stdAtZero > PHYSICS_CONSTRAINT_TOLERANCE_PSI * 10) {
      console.warn(
        `vec[${idx}]: posterior std at t=0 is ${stdAtZero.toFixed(2)} psi ` +
          `(expected ~0 for gated kernel). May indicate variance from a ` +
          `noise term applied at test time.`
      );
    }

    passed++;
  } catch (e) {
    failed++;
    failureReports.push(e.message);
  }
}

const total = passed + failed;
console.log(`\nJS physical-constraint test: ${passed}/${total} compositions satisfy f(x, 0) ≈ 0`);

if (failed > 0) {
  console.error(`\n❌ ${failed} compositions FAILED the physics constraint`);
  console.error(`First few failures:`);
  for (const msg of failureReports.slice(0, 3)) {
    console.error(`  ${msg}`);
  }
  console.error(
    `\nThe JS GP's gating mechanism is broken — a non-zero variance at ` +
      `t=0 means h(0) is not exactly 0 in docs/gp.mjs, or the kernel-side ` +
      `multiplicative gate isn't being applied. See:\n` +
      `  - experiments/STRENGTH_GP_BENCHMARK.md (architecture)\n` +
      `  - docs/gp.mjs::gateFunction (the ∟ implementation)\n`
  );
  process.exit(1);
} else {
  console.log("✅ All compositions satisfy the physics constraint");
}
