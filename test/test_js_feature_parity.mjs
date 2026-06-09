/**
 * JS ↔ Python feature-builder parity test.
 *
 * Loads a fixture written by Python's
 * ``experiments/regenerate_feature_parity_fixture.py`` (which runs
 * ``boxcrete.features.FEATURE_BUILDERS`` over a small battery of
 * input vectors) and asserts each ``docs/feature_registry.mjs::FEATURE_FNS``
 * builder reproduces the Python output element-for-element within a
 * tight tolerance.
 *
 * Why a separate test (vs. the existing test_vectors.json end-to-end
 * check):
 *   * test_vectors.json catches drift but doesn't pinpoint *which*
 *     feature builder diverged — failures show up as posterior-mean
 *     mismatches at unrelated test points.
 *   * This test exercises each builder in isolation, so a math bug
 *     surfaces with the offending feature name in the assertion message.
 *
 * Tolerance: ``rtol=1e-12, atol=1e-15`` — tight enough to catch any
 * real arithmetic bug (a +1e-3 vs +1e-4 epsilon in the log builders
 * would shift outputs by >>1e-9), loose enough to absorb any 1-ULP
 * difference between V8's Math.log and PyTorch's torch.log on the
 * same machine.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

import { FEATURE_FNS } from "../docs/feature_registry.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const FIXTURE_PATH = resolve(
  __dirname,
  "fixtures",
  "feature_parity_fixture.json",
);

const fixture = JSON.parse(readFileSync(FIXTURE_PATH, "utf-8"));
const { rtol, atol } = fixture.tolerances;
// Pin a tolerance ceiling so a future fixture regen with looser
// thresholds doesn't silently weaken the parity guard. ``rtol=1e-9``
// would still admit a real arithmetic regression by orders of
// magnitude; the actual tolerances should be MUCH tighter (the
// committed fixture currently uses 1e-12 / 1e-15).
const RTOL_CEILING = 1e-9;
const ATOL_CEILING = 1e-12;
if (rtol > RTOL_CEILING || atol > ATOL_CEILING) {
  console.error(
    `❌ feature-parity fixture has tolerances rtol=${rtol}, atol=${atol} ` +
      `which exceed the safety ceiling (rtol<=${RTOL_CEILING}, atol<=${ATOL_CEILING}). ` +
      "If you need looser tolerances, justify it in a code review and bump these constants.",
  );
  process.exit(1);
}

let failures = 0;
let assertionCount = 0;

for (const name of fixture.feature_names) {
  if (!(name in FEATURE_FNS)) {
    console.error(
      `❌ feature ${JSON.stringify(name)} declared in the fixture but missing from ` +
        `docs/feature_registry.mjs::FEATURE_FNS — JS port is incomplete.`,
    );
    failures++;
    continue;
  }
  const fn = FEATURE_FNS[name];
  const expectedCol = fixture.expected[name]; // length = N inputs

  for (let i = 0; i < fixture.inputs.length; i++) {
    const x = fixture.inputs[i];
    const expected = expectedCol[i];
    const got = fn(x);
    assertionCount++;

    const tol = rtol * Math.abs(expected) + atol;
    const absDiff = Math.abs(got - expected);
    if (!(absDiff <= tol)) {
      console.error(
        `❌ ${name}[input ${i}]: got=${got}, expected=${expected}, ` +
          `|diff|=${absDiff.toExponential(3)} > tol=${tol.toExponential(3)} ` +
          `(rtol=${rtol}, atol=${atol}).\n   Input vector: ${JSON.stringify(x)}`,
      );
      failures++;
    }
  }
}

if (failures > 0) {
  console.error(
    `\n❌ ${failures} of ${assertionCount} feature parity assertion(s) failed.`,
  );
  console.error(
    "If this is an intentional change to a feature builder, update both " +
      "boxcrete/features.py::FEATURE_BUILDERS AND " +
      "docs/feature_registry.mjs::FEATURE_FNS, then re-generate the " +
      "fixture via:\n  python experiments/regenerate_feature_parity_fixture.py",
  );
  process.exit(1);
}

console.log(
  `✅ JS ↔ Python feature parity: ${assertionCount} assertions across ` +
    `${fixture.feature_names.length} builders × ${fixture.inputs.length} ` +
    `inputs all within rtol=${rtol}, atol=${atol}.`,
);
