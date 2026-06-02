#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Regenerate all precomputed artifacts that depend on the deployed
# strength model. See docs/model/README.md for what each file is.
# Run this whenever you re-train the model or change anything that
# affects the JS port. Idempotent.
#
# Pipeline:
#   1. Re-export docs/model/strength.json AND docs/model/test_vectors.json
#      from the same Python champion fit (one process, one model object;
#      both files must come from the same fit).
#   2. Augment test_vectors.json with GWP/cost columns (computed via the
#      JS predictors so the QA harness captures JS-side math too).
#   3. Recompute docs/model/compositions.json::strength_predictions from
#      the fresh strength.json.
#   4. Run the freshness check + full JS test suite.
#
# Usage: bash experiments/regenerate_all_artifacts.sh

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

echo "==> 1. Re-exporting strength.json from Python champion …"
python -u experiments/regenerate_strength_json.py | tail -8

echo ""
echo "==> 2. Augmenting test_vectors with GWP/cost …"
node experiments/augment_test_vectors_with_gwp_cost.mjs | tail -3

echo ""
echo "==> 3. Recomputing compositions.json strength_predictions + pareto_mask …"
node experiments/regenerate_compositions_strength_predictions.mjs | tail -5

echo ""
echo "==> 4. Running freshness tests …"
node test/test_data_freshness.mjs

echo ""
echo "==> 5. Running full JS test suite …"
for t in test_js_strength_v2.mjs test_js_physical_constraints.mjs test_js_gp.mjs test_js_ui_smoke.mjs test_lengthscales_v2.mjs test_curve_monotonicity.mjs; do
  echo "  -- $t --"
  node "test/$t" | tail -3
done

echo ""
echo "✅ All artifacts regenerated and verified."
