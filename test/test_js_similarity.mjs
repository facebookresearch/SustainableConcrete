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
import {
  compositionSimilarity,
  similarityToFillAlpha,
  buildSimilarityContext,
  computeSimilarities,
  SIMILARITY_FLOOR,
  FILL_ALPHA_MIN,
} from "../docs/similarity.mjs";

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

// --- parity with the production kernel -------------------------------------
// Our composition-only similarity must reconstruct gp.mjs's full normalized
// kernel exactly once the (constant, at equal curing day) time branch is added
// back. This is what stops similarity.mjs drifting into a private, divergent
// reimplementation of the model. If it ever fails, fix similarity.mjs, not this.
let worstParity = 0;
for (const [i, j] of [
  [0, 1],
  [5, 40],
  [12, 12],
  [100, 3],
  [77, 148],
]) {
  const z1 = transformInput([...X[i], 28], P);
  const z2 = transformInput([...X[j], 28], P);
  const full = kernel(z1, z2, P) / Math.sqrt(kernel(z1, z1, P) * kernel(z2, z2, P));
  const s = compositionSimilarity(X[i], X[j], P, 28);
  const reconstructed = (s * (osB + osS) + osT) / (osB + osS + osT);
  worstParity = Math.max(worstParity, Math.abs(full - reconstructed));
}
check(
  "similarity reconstructs gp.mjs kernel() exactly",
  worstParity < 1e-10,
  `max err ${worstParity}`,
);

// --- contrast stretch ------------------------------------------------------
check("alpha(1) === 1", close(similarityToFillAlpha(1), 1, 1e-12));
check(
  "alpha at/below the floor === FILL_ALPHA_MIN",
  close(similarityToFillAlpha(SIMILARITY_FLOOR), FILL_ALPHA_MIN, 1e-12) &&
    close(similarityToFillAlpha(0), FILL_ALPHA_MIN, 1e-12),
);
check("alpha is never zero (the point stays visible)", similarityToFillAlpha(0) > 0);

let alphaMonotone = true;
let lastAlpha = -1;
for (let s = 0; s <= 1.0001; s += 0.02) {
  const a = similarityToFillAlpha(s);
  if (a < lastAlpha - 1e-12 || a < 0 || a > 1) alphaMonotone = false;
  lastAlpha = a;
}
check("alpha is monotone non-decreasing within [0,1]", alphaMonotone);
check(
  "out-of-range inputs clamp rather than produce NaN",
  similarityToFillAlpha(-5) === FILL_ALPHA_MIN &&
    close(similarityToFillAlpha(9), 1, 1e-12),
);
check("NaN degrades to the floor", similarityToFillAlpha(NaN) === FILL_ALPHA_MIN);

// The stretch exists to create contrast: raw similarities sit in a narrow IQR
// of roughly [0.64, 0.83], which would render as a uniformly ~70% solid plot.
const frameAlphas = X.map((c) =>
  similarityToFillAlpha(compositionSimilarity(X[0], c, P, 28)),
);
const alphaSpread = Math.max(...frameAlphas) - Math.min(...frameAlphas);
check(
  "rendered alpha spread across a frame exceeds 0.5",
  alphaSpread > 0.5,
  `spread=${alphaSpread.toFixed(3)}`,
);

// --- batched path ----------------------------------------------------------
const ctx = buildSimilarityContext(X, P, 28);
const sims = computeSimilarities(X[0], ctx, P);
check("one similarity per catalog mix", sims.length === X.length);
check(
  "batched agrees with pairwise",
  X.every((c, i) => close(sims[i], compositionSimilarity(X[0], c, P, 28), 1e-12)),
);
check("self-similarity is 1 at the reference index", close(sims[0], 1, 1e-12));
check("null params yields null (pre-model)", computeSimilarities(X[0], ctx, null) === null);
check("null context yields null", computeSimilarities(X[0], null, P) === null);
check(
  "empty catalog yields an empty result",
  computeSimilarities(X[0], buildSimilarityContext([], P, 28), P).length === 0,
);
const simsAgain = computeSimilarities(X[0], ctx, P);
check("deterministic across calls", sims.every((v, i) => v === simsAgain[i]));

// A dragged slider puts the reference between catalog points, so the reference
// is routinely off-catalog. That path must stay finite and in range.
const offCatalog = X[0].map((v, i) => (i === SRC ? v : v * 1.37 + 5));
check(
  "off-catalog reference stays finite and in range",
  computeSimilarities(offCatalog, ctx, P).every(
    (v) => Number.isFinite(v) && v >= 0 && v <= 1 + 1e-12,
  ),
);

// --- golden fixture --------------------------------------------------------
// Pins behaviour against the shipped model artifact. Regenerate DELIBERATELY,
// and review the diff, if the strength model is ever retrained.
const GOLDEN_TOP5 = [0, 16, 22, 19, 15];
const top5 = Array.from(sims.keys())
  .sort((i, j) => sims[j] - sims[i])
  .slice(0, 5);
check(
  "golden: top-5 nearest neighbours of catalog mix #0 are unchanged",
  JSON.stringify(top5) === JSON.stringify(GOLDEN_TOP5),
  `got [${top5}], want [${GOLDEN_TOP5}]`,
);

// ===== INSERT NEW CHECKS ABOVE THIS LINE — the exit gate must stay last =====
if (failures) {
  console.error(`\n${failures} failure(s)`);
  process.exit(1);
}
console.log("\nAll similarity tests passed.");
