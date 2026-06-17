#!/usr/bin/env node
// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// Regenerate ``compositions.gwp_predictions`` from ``compositions.compositions``
// using the JS ``predictGWP`` predictor (the same one the explorer uses).
//
// Needed after any change to ``docs/model/gwp.json`` (e.g. adding a new
// Material Source class, re-deriving per-class GWP coefficients, etc.).
// The existing ``regenerate_compositions_strength_predictions.mjs`` only
// refreshes the strength curves — gwp_predictions stays stale unless
// this script (or an equivalent path) is run.

import { readFileSync, writeFileSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const docsRoot = resolve(__dirname, "..", "docs");

const { predictGWP } = await import(resolve(docsRoot, "gp.mjs"));

const compositions = JSON.parse(
  readFileSync(resolve(docsRoot, "model/compositions.json"), "utf-8"),
);
const gwpParams = JSON.parse(
  readFileSync(resolve(docsRoot, "model/gwp.json"), "utf-8"),
);

const classDim = gwpParams.class_dim;
if (typeof classDim !== "number") {
  throw new Error(
    `gwp.json::class_dim must be a number; got ${classDim} (${typeof classDim})`,
  );
}

const oldPreds = compositions.gwp_predictions || [];
const newPreds = new Array(compositions.compositions.length);
let sumAbs = 0;
let maxAbs = 0;
let n = 0;

for (let i = 0; i < compositions.compositions.length; i++) {
  const c = compositions.compositions[i];
  const cls = Math.round(c[classDim]);
  const { mean } = predictGWP(c, gwpParams, cls);
  newPreds[i] = mean;
  if (oldPreds[i] !== undefined && Number.isFinite(oldPreds[i])) {
    const d = Math.abs(mean - oldPreds[i]);
    sumAbs += d;
    if (d > maxAbs) maxAbs = d;
    n++;
  }
}

compositions.gwp_predictions = newPreds;
console.log(
  `Regenerated ${newPreds.length} gwp_predictions; ` +
  `mean |Δ| vs old = ${(sumAbs / Math.max(1, n)).toFixed(2)}, ` +
  `max |Δ| = ${maxAbs.toFixed(2)}.`,
);

writeFileSync(
  resolve(docsRoot, "model/compositions.json"),
  JSON.stringify(compositions),
);
console.log(`Wrote ${resolve(docsRoot, "model/compositions.json")}`);
