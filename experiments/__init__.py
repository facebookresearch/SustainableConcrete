# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Empty marker — makes ``experiments/`` an importable package.

This package hosts the artifact-regeneration pipeline and CI helpers
that surround the boxcrete production code (``boxcrete/``):

* ``regenerate_all_artifacts.sh`` / ``regenerate_strength_json.py`` —
  re-export ``docs/model/*`` from a fresh fit; the canonical pipeline
  invoked by ``Model Artifacts Coherence`` CI.
* ``regenerate_compositions_strength_predictions.mjs`` /
  ``augment_test_vectors_with_gwp_cost.mjs`` — JS-side regen steps.
* ``check_artifacts_drift.py`` — cross-arch-portable artifact drift
  check used by the regen-idempotency CI workflow.
* ``run_notebook_with_progress.py`` — cell-by-cell timing helper used
  by the Notebooks CI workflow as a fast-fail diagnostic.
"""
