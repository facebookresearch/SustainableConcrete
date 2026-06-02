# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pretrained-loader fidelity test.

Asserts that ``boxcrete.load_pretrained_strength_gp()`` reconstructs the
deployed V2 strength GP from ``docs/model/strength_model.pt`` (a PyTorch
``state_dict``) into a model whose posterior matches the Python
reference predictions baked into ``docs/model/test_vectors.json``. Both
artifacts are produced atomically by
``experiments/regenerate_strength_json.py`` during the regen pipeline.
Catches any future schema drift between the export path and the loader.

The test is deliberately fast: it loads the model once and probes a
small number of test vectors. Full-data fidelity is implicitly covered
because both paths share the same ``X_train`` / kernel hyperparameters
/ likelihood noise via the saved ``state_dict``.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_VECTORS_PATH = REPO_ROOT / "docs" / "model" / "test_vectors.json"
STRENGTH_JSON_PATH = REPO_ROOT / "docs" / "model" / "strength.json"

# Tight tolerances. The loader and the regen path read the same JSON
# bytes, so post-load posteriors should be float-exact (modulo torch's
# constraint inverse-transforms accumulating ~1e-15 of round-off, well
# below atol=1e-5).
ATOL_MEAN_PSI = 1e-5
ATOL_VARIANCE_PSI2 = 1e-3


class TestPretrainedLoaderFidelity(unittest.TestCase):
    """Loaded model.posterior(...) ≈ test_vectors.json's expected_*."""

    @classmethod
    def setUpClass(cls):
        # Confirm the JSON artifacts exist before running.
        if not STRENGTH_JSON_PATH.exists():
            raise unittest.SkipTest(
                f"{STRENGTH_JSON_PATH} not found; run "
                "`bash experiments/regenerate_all_artifacts.sh` first."
            )
        if not TEST_VECTORS_PATH.exists():
            raise unittest.SkipTest(
                f"{TEST_VECTORS_PATH} not found; run "
                "`bash experiments/regenerate_all_artifacts.sh` first."
            )
        # Load the model once for all tests.
        from boxcrete import load_pretrained_strength_gp

        cls.model = load_pretrained_strength_gp()
        cls.model.eval()
        cls.y_max = float(cls.model._study_y_std)
        with open(TEST_VECTORS_PATH) as f:
            cls.test_data = json.load(f)

    def _predict(self, composition: list[float], time: float) -> tuple[float, float]:
        """Run the loaded model's posterior at the given raw input.

        Returns ``(mean_psi, total_variance_psi²)`` where the variance
        matches the public ``variance_includes_aleatoric=True`` contract
        used by JS-side `predictStrength` / `predictStrengthCurve` and
        ``test_vectors.json``: latent f-variance + gated aleatoric
        ``h(t_post)²·σ²``, all un-standardised by ``y_max²``.
        """
        import math

        x = torch.tensor([list(composition) + [float(time)]], dtype=torch.double)
        with torch.no_grad():
            post = self.model.posterior(x)
            mean_psi = float(post.mean.item()) * self.y_max
            latent_var_psi2 = float(post.variance.item()) * self.y_max**2
        # Aleatoric: σ² · h(log10(t+1))² · y_max², matching the regen
        # script's ``_gated_aleatoric_psi2`` helper.
        noise_scaled = float(
            self.model.likelihood.noise_covar.noise.detach().flatten()[0]
        )
        noise_gate_tau = float(self.model.likelihood.gate_tau)
        t_post = math.log10(float(time) + 1.0)
        h = 1.0 - math.exp(-max(t_post, 0.0) / noise_gate_tau)
        aleatoric_psi2 = h * h * noise_scaled * self.y_max**2
        return mean_psi, latent_var_psi2 + aleatoric_psi2

    def test_loader_matches_baked_expected_means(self):
        # First 10 vectors is plenty (more would just slow the test).
        for i, v in enumerate(self.test_data["test_vectors"]):
            with self.subTest(vec=i):
                got_mean, _ = self._predict(v["composition"], v["time"])
                self.assertAlmostEqual(
                    got_mean,
                    v["expected_mean"],
                    delta=ATOL_MEAN_PSI,
                    msg=(
                        f"vec[{i}] (composition={v['composition'][:3]}..., "
                        f"time={v['time']}): loader posterior mean diverged "
                        f"from test_vectors.json's expected_mean by "
                        f"{abs(got_mean - v['expected_mean']):.3e} psi "
                        f"(allowed atol={ATOL_MEAN_PSI})."
                    ),
                )

    def test_loader_matches_baked_expected_variances(self):
        for i, v in enumerate(self.test_data["test_vectors"]):
            with self.subTest(vec=i):
                _, got_var = self._predict(v["composition"], v["time"])
                self.assertAlmostEqual(
                    got_var,
                    v["expected_variance"],
                    delta=ATOL_VARIANCE_PSI2,
                    msg=(
                        f"vec[{i}] (composition={v['composition'][:3]}..., "
                        f"time={v['time']}): loader posterior variance "
                        f"diverged from test_vectors.json's "
                        f"expected_variance by "
                        f"{abs(got_var - v['expected_variance']):.3e} psi^2 "
                        f"(allowed atol={ATOL_VARIANCE_PSI2})."
                    ),
                )

    def test_loader_returns_singletaskgp(self):
        # Sanity check: the returned object is the same architecture as
        # what fit_strength_gp produces (so downstream callers can use
        # the same APIs).
        from botorch.models import SingleTaskGP

        self.assertIsInstance(self.model, SingleTaskGP)
        self.assertEqual(self.model.num_outputs, 1)
        # Should be in eval mode (the loader returns a model ready for
        # prediction, not training).
        self.assertFalse(self.model.training)


if __name__ == "__main__":
    unittest.main()
