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
import { compositionSimilarity } from "../docs/similarity.mjs";

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

// --- core metric -----------------------------------------------------------
check("s(x,x) === 1", close(compositionSimilarity(X[0], X[0], P, 28), 1, 1e-12));
check(
  "symmetry",
  close(
    compositionSimilarity(X[3], X[9], P, 28),
    compositionSimilarity(X[9], X[3], P, 28),
    1e-12,
  ),
);

let minS = Infinity;
let maxS = -Infinity;
let bad = 0;
for (let i = 0; i < X.length; i++) {
  for (let j = 0; j < X.length; j++) {
    const s = compositionSimilarity(X[i], X[j], P, 28);
    if (!Number.isFinite(s) || s < 0 || s > 1 + 1e-12) bad++;
    if (s < minS) minS = s;
    if (s > maxS) maxS = s;
  }
}
check("all 149x149 similarities finite and in [0,1]", bad === 0, `${bad} bad`);

// GUARD: with RBF_time wrongly folded in, the floor is 0.907 rather than 0.267.
check("dynamic range is usable (min < 0.40)", minS < 0.4, `min=${minS.toFixed(4)}`);
check("max similarity is 1", close(maxS, 1, 1e-12));

// The source dim is categorical: the model uses a Hamming kernel for it, so a
// 0->2 class change must be exactly as dissimilar as 0->1, never "twice as far".
const b0 = [...X[0]];
const b1 = [...X[0]];
const b2 = [...X[0]];
b0[SRC] = 0;
b1[SRC] = 1;
b2[SRC] = 2;
check(
  "source 0->1 equals 0->2 (Hamming, not ordinal)",
  close(
    compositionSimilarity(b0, b1, P, 28),
    compositionSimilarity(b0, b2, P, 28),
    1e-12,
  ),
);
const expectedCross =
  (osB + osS * P.matern_specific.source_correlation) / (osB + osS);
check(
  "cross-class similarity uses the learned source_correlation",
  close(compositionSimilarity(b0, b1, P, 28), expectedCross, 1e-10),
);

let prev = 1.0;
let mono = true;
for (const d of [10, 50, 120, 300]) {
  const p = [...X[0]];
  p[0] += d; // dim 0 = Cement
  const s = compositionSimilarity(X[0], p, P, 28);
  if (!(s < prev)) mono = false;
  prev = s;
}
check("similarity strictly decreases with a growing perturbation", mono);

// Temp (raw dim 8) is part of the metric -- an explicit product requirement.
const tempShift = [...X[0]];
tempShift[8] += 15;
check(
  "Temp contributes to the distance",
  compositionSimilarity(X[0], tempShift, P, 28) < 1 - 1e-6,
);

check("null params yields null", compositionSimilarity(X[0], X[1], null, 28) === null);

// ===== INSERT NEW CHECKS ABOVE THIS LINE — the exit gate must stay last =====
if (failures) {
  console.error(`\n${failures} failure(s)`);
  process.exit(1);
}
console.log("\nAll similarity tests passed.");
