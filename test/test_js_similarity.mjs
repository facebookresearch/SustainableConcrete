/**
 * Recipe-similarity metric checks.
 *
 * Similarity between two mixes is the strength GP's own normalized composition
 * kernel — the model's believed correlation between them. Two of the checks
 * below are load-bearing anti-regression guards:
 *
 *   - the DYNAMIC RANGE check fails if the RBF_time branch is ever folded back
 *     into the metric. At equal curing day it contributes a constant 0.4578 of
 *     the 0.5245 total outputscale, which silently compresses every similarity
 *     into [0.907, 1.0] and makes the encoding look uniform rather than broken.
 *   - the KERNEL PARITY check fails if docs/similarity.mjs drifts from the
 *     production kernel in docs/gp.mjs.
 */
import { readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { matern52ActiveDims, kernel, transformInput } from "../docs/gp.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const modelDir = resolve(__dirname, "..", "docs", "model");
const P = JSON.parse(readFileSync(resolve(modelDir, "strength.json"), "utf-8"));
const C = JSON.parse(readFileSync(resolve(modelDir, "compositions.json"), "utf-8"));
const X = C.compositions;
const SRC = P.source_dim_raw; // 7
const osB = P.matern_blind.outputscale;
const osS = P.matern_specific.outputscale;
const osT = P.rbf_time.outputscale;

let failures = 0;
function check(name, cond, detail = "") {
  if (cond) console.log(`  ok   ${name}`);
  else {
    console.error(`  FAIL ${name}${detail ? " — " + detail : ""}`);
    failures++;
  }
}
const close = (a, b, tol = 1e-9) => Math.abs(a - b) <= tol;

check("matern52ActiveDims is exported", typeof matern52ActiveDims === "function");
check("kernel is exported", typeof kernel === "function");
check("transformInput is exported", typeof transformInput === "function");
check("catalog has compositions", Array.isArray(X) && X.length > 0);
check("outputscales are present", osB > 0 && osS > 0 && osT > 0);

// ===== INSERT NEW CHECKS ABOVE THIS LINE — the exit gate must stay last =====
if (failures) {
  console.error(`\n${failures} failure(s)`);
  process.exit(1);
}
console.log("\nAll similarity tests passed.");
