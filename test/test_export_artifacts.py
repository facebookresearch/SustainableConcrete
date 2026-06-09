#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Invariants for the JSON artifacts emitted by
``experiments/regenerate_strength_json.py``.

These guard properties of the *committed* files in ``docs/model/`` that the
website consumes. They are CI-blocking because a bad artifact silently
breaks the user-facing demo.
"""

import json
import os
import unittest

from boxcrete.utils import REPO_DIR

COMP_PATH = os.path.join(REPO_DIR, "docs", "model", "compositions.json")

# Expected number of rows in the public compositions list (derived from the
# canonical CSV ``data/boxcrete_data.csv``). Update only when the source
# CSV changes intentionally.
EXPECTED_N_PUBLIC_COMPOSITIONS = 144


class TestCompositionsArtifact(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(COMP_PATH):
            self.skipTest(f"committed compositions not found at {COMP_PATH}")
        with open(COMP_PATH) as f:
            self.comp = json.load(f)

    def test_public_composition_count(self):
        """The scatter-plot list should reflect the unique compositions in
        ``data/boxcrete_data.csv``. A row-count mismatch catches accidental
        changes to the data-processing pipeline that would silently add or
        drop training compositions in the deployed model."""
        self.assertEqual(
            len(self.comp["compositions"]),
            EXPECTED_N_PUBLIC_COMPOSITIONS,
            msg=(
                "Number of public compositions has drifted. If this was "
                "intentional (CSV update), bump EXPECTED_N_PUBLIC_COMPOSITIONS."
            ),
        )


if __name__ == "__main__":
    unittest.main()
