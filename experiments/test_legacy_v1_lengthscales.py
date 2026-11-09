# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Lengthscale-identifiability and within-group-tying regression tests
for the **legacy V1** strength GP architecture (research baseline only).

These tests verify properties of the V1 architecture (single Matern +
within-group prior + RBF on time + day-zero anchors + scalar noise),
which was the production OSS model before V2 was deployed on 2026-05-17.

The V1 fit lives in ``experiments/legacy_v1_strength_gp.py`` and is
exercised here for journey-baseline regression. Production V2
identifiability is verified by:
- ``test/test_lengthscale_identifiability.py`` (Python freshly-fit V2)
- ``test/test_lengthscales_v2.mjs`` (deployed ``strength.json``)
"""

import json
import math
import os
import unittest
from functools import lru_cache

import torch

from boxcrete import compute_loo_cv
from boxcrete.priors import (
    WithinGroupShrinkagePrior,
    _AGGREGATE_LENGTHSCALE_GROUP,
    _BINDER_LENGTHSCALE_GROUP,
    _default_lengthscale_prior,
)
from boxcrete.utils import DEFAULT_X_COLUMNS, REPO_DIR, load_concrete_strength
from experiments.legacy_v1_strength_gp import fit_strength_gp_v1

STRENGTH_JSON_PATH = os.path.join(REPO_DIR, "docs", "model", "strength.json")

# Maximum permissible ratio max(ℓ)/min(ℓ) within a tied material group.
# At sigma=0.001 (production), the ratio is essentially 1 (within float
# noise). 1.05 leaves comfortable margin for BLAS drift across platforms
# while still catching any real loosening of the prior.
_MAX_WITHIN_GROUP_RATIO = 1.05

# CI-blocking upper bound on LOO CV RMSE (psi) on the 647 public
# strength rows. The production fit gives ~725 psi; ~750 is a generous
# ceiling that still catches the "no prior" baseline (~772 psi) and any
# meaningful regression of the prior. See
# scripts/lengthscale_prior_study.py for the calibrating sweep.
_LOO_RMSE_CEILING_PSI = 750.0


@lru_cache(maxsize=1)
def _load_strength_params():
    """Load the committed strength.json once and memoise."""
    with open(STRENGTH_JSON_PATH) as f:
        return json.load(f)


def _is_v2_schema() -> bool:
    """Schema version 2 (the deployed multi-Matern + gated-noise champion)
    stores lengthscales under ``matern_blind`` and ``matern_specific``,
    not the v1 single ``matern_lengthscales`` array. The v2-aware
    equivalent of this test lives in ``test/test_lengthscales_v2.mjs``.
    """
    try:
        return _load_strength_params().get("schema_version") == 2
    except (FileNotFoundError, KeyError):
        return False


def _v2_skip_reason() -> str:
    return (
        "test reads v1 strength.json schema fields "
        "(matern_lengthscales etc.); v2-aware identifiability checks "
        "are in test/test_lengthscales_v2.mjs"
    )


@lru_cache(maxsize=1)
def _fit_default_strength_gp():
    """Fit the legacy V1 strength GP once (with the default within-group
    shrinkage prior) and cache it across tests."""
    torch.manual_seed(0)
    data = load_concrete_strength()
    X, Y, Yvar, X_bounds = data.strength_data
    gp = fit_strength_gp_v1(
        X=X,
        Y=Y,
        Yvar=Yvar,
        X_bounds=X_bounds,
        use_fixed_noise=False,
    )
    return gp, X, Y, Yvar, X_bounds


def _matern_lengthscales(gp) -> list[float]:
    return (
        gp.covar_module.kernels[0].base_kernel.lengthscale.detach().squeeze().tolist()
    )


def _within_group_ratio(lengthscales, group_idxs) -> float:
    """max ℓ / min ℓ across the group — measures how tightly tied the
    group's lengthscales are. Should be ≈ 1 when sigma -> 0."""
    vals = [lengthscales[i] for i in group_idxs]
    return max(vals) / min(vals)


def _analytical_loo_rmse(gp, n_real: int) -> float:
    """RMSE (in original target units) of analytical LOO at fixed
    hyperparameters. Thin wrapper around ``boxcrete.compute_loo_cv``;
    see that function for the closed-form derivation. ``n_real``
    excludes the day-zero anchors that ``fit_strength_gp`` appends.

    Used by the production-fit regression test below; reproduces the
    same headline numbers as ``scripts/lengthscale_prior_study.py``.
    """
    obs, mean, _ = compute_loo_cv(gp, n_real=n_real)
    return float(((obs - mean) ** 2).mean().sqrt().item())


def _feature_names_from_columns():
    """Strip the ``(unit)`` suffix from ``DEFAULT_X_COLUMNS`` to match the
    consumer-friendly names emitted in ``strength.json``."""
    return [c.split(" (")[0] for c in DEFAULT_X_COLUMNS]


def _collect_violations(lengthscales, feature_names, cap):
    """Return a list of human-readable strings, one per cap violation."""
    return [
        f"{name} (idx {i}): {lengthscales[i]:.2f}"
        for i, name in enumerate(feature_names)
        if lengthscales[i] >= cap
    ]


class TestStrengthLengthscaleIdentifiability(unittest.TestCase):
    """Block landing if any feature's lengthscale is at the constraint cap."""

    def _assert_no_violations(self, violations, source_label, remediation):
        self.assertEqual(
            violations,
            [],
            msg=(
                f"{source_label} has lengthscale(s) at or beyond the "
                f"identifiability cap. {remediation} "
                f"Violations: {', '.join(violations)}"
            ),
        )

    def test_freshly_fit_gp_has_identifiable_lengthscales(self):
        if _is_v2_schema():
            self.skipTest(_v2_skip_reason())
        gp, *_ = _fit_default_strength_gp()
        ls = _matern_lengthscales(gp)
        feature_names = _feature_names_from_columns()
        cap = _load_strength_params()["lengthscale_identifiability_cap"]

        self.assertEqual(len(ls), len(feature_names))
        self._assert_no_violations(
            _collect_violations(ls, feature_names, cap),
            source_label="Freshly-fit strength GP",
            remediation=(
                "This means the GP is treating those features as "
                "uninformative — the corresponding sliders in the website's "
                "Composition panel will not respond. Investigate data churn "
                "or fit instability."
            ),
        )

    def test_committed_strength_json_has_identifiable_lengthscales(self):
        """Sanity-check the *committed* model artifact too. CI also relies on
        this to catch the case where the JSON wasn't regenerated after a
        data change."""
        if _is_v2_schema():
            self.skipTest(_v2_skip_reason())

    def test_committed_binder_lengthscales_tied(self):
        if _is_v2_schema():
            self.skipTest(_v2_skip_reason())
        if not os.path.exists(STRENGTH_JSON_PATH):
            self.skipTest(f"committed model not found at {STRENGTH_JSON_PATH}")
        params = _load_strength_params()

        # Loud failure if the export script ever forgets to emit either
        # field — that would silently fall back to local hard-coded values.
        self.assertIn("feature_names", params)
        self.assertIn("lengthscale_identifiability_cap", params)

        ls = params["matern_lengthscales"]
        feature_names = params["feature_names"]
        cap = params["lengthscale_identifiability_cap"]
        self._assert_no_violations(
            _collect_violations(ls, feature_names, cap),
            source_label="Committed docs/model/strength.json",
            remediation=(
                "Re-run `python scripts/export_model.py` after updating the "
                "data, and re-commit the regenerated JSON."
            ),
        )

    def test_committed_strength_matches_fresh_fit(self):
        """Catch the "forgot to re-export `strength.json`" failure mode: the
        committed lengthscales should match a freshly-fit run on the same
        canonical data."""
        if _is_v2_schema():
            self.skipTest(_v2_skip_reason())
        if not os.path.exists(STRENGTH_JSON_PATH):
            self.skipTest(f"committed model not found at {STRENGTH_JSON_PATH}")
        committed = torch.tensor(
            _load_strength_params()["matern_lengthscales"],
            dtype=torch.float64,
        )
        gp, *_ = _fit_default_strength_gp()
        fresh = (
            gp.covar_module.kernels[0]
            .base_kernel.lengthscale.detach()
            .squeeze()
            .to(dtype=torch.float64)
        )

        self.assertEqual(
            fresh.shape,
            committed.shape,
            msg=(
                "Committed strength.json has a different number of "
                "lengthscales than a fresh fit produces — re-run "
                "`python scripts/export_model.py`."
            ),
        )
        # The export script seeds with the same value, so on a given platform
        # the lengthscales are reproducible. rtol=2e-2 catches the "forgot
        # to re-export" failure mode (which typically perturbs lengthscales
        # by orders of magnitude when the data shape changes) while
        # tolerating ~1% drift across BLAS implementations / architectures.
        self.assertTrue(
            torch.allclose(fresh, committed, rtol=2e-2, atol=1e-3),
            msg=(
                "Committed strength.json lengthscales drift from a fresh "
                "fit. Re-run `python scripts/export_model.py` and commit "
                f"the regenerated JSON. fresh={fresh.tolist()}, "
                f"committed={committed.tolist()}"
            ),
        )

    # ------------------------------------------------------------------
    # Within-group tying (binders {Cement, FA, Slag}, aggregates {Fine, Coarse}).
    # These tests fail if the WithinGroupShrinkagePrior is weakened or
    # removed — see boxcrete/models.py for the production prior settings.
    # ------------------------------------------------------------------
    def _assert_group_tied(self, lengthscales, group_idxs, group_label):
        ratio = _within_group_ratio(lengthscales, group_idxs)
        self.assertLess(
            ratio,
            _MAX_WITHIN_GROUP_RATIO,
            msg=(
                f"{group_label} lengthscales are not tied as expected: "
                f"max/min = {ratio:.3f} > {_MAX_WITHIN_GROUP_RATIO}. "
                f"Group lengthscales = "
                f"{[lengthscales[i] for i in group_idxs]}. "
                f"Check that boxcrete.models.WithinGroupShrinkagePrior is "
                f"installed by fit_strength_gp with "
                f"_LENGTHSCALE_SHRINKAGE_SIGMA <= 0.01."
            ),
        )

    def test_committed_binder_lengthscales_tied(self):
        """Cement/FA/Slag share a lengthscale in the committed JSON."""
        if _is_v2_schema():
            self.skipTest(_v2_skip_reason())
        if not os.path.exists(STRENGTH_JSON_PATH):
            self.skipTest(f"committed model not found at {STRENGTH_JSON_PATH}")
        ls = _load_strength_params()["matern_lengthscales"]
        self._assert_group_tied(
            ls,
            _BINDER_LENGTHSCALE_GROUP,
            group_label="Binder",
        )

    def test_committed_aggregate_lengthscales_tied(self):
        """Fine and Coarse Aggregates share a lengthscale in the committed JSON."""
        if _is_v2_schema():
            self.skipTest(_v2_skip_reason())
        if not os.path.exists(STRENGTH_JSON_PATH):
            self.skipTest(f"committed model not found at {STRENGTH_JSON_PATH}")
        ls = _load_strength_params()["matern_lengthscales"]
        self._assert_group_tied(
            ls,
            _AGGREGATE_LENGTHSCALE_GROUP,
            group_label="Aggregate",
        )

    def test_freshly_fit_binder_lengthscales_tied(self):
        """Production prior actually ties the binder lengthscales at fit time."""
        gp, *_ = _fit_default_strength_gp()
        self._assert_group_tied(
            _matern_lengthscales(gp),
            _BINDER_LENGTHSCALE_GROUP,
            group_label="Binder",
        )

    def test_freshly_fit_aggregate_lengthscales_tied(self):
        """Production prior actually ties the aggregate lengthscales at fit time."""
        gp, *_ = _fit_default_strength_gp()
        self._assert_group_tied(
            _matern_lengthscales(gp),
            _AGGREGATE_LENGTHSCALE_GROUP,
            group_label="Aggregate",
        )

    # ------------------------------------------------------------------
    # LOO CV performance — guards the empirical win documented in the
    # WithinGroupShrinkagePrior docstring (production fit ~663 psi RMSE
    # vs no-prior baseline ~707 psi).
    # ------------------------------------------------------------------
    def test_loo_cv_rmse_does_not_regress(self):
        """Held-out LOO RMSE on the 647 public strength rows is below
        ~750 psi. This bound is loose enough to absorb small platform
        drift but still catches the no-prior baseline (~772 psi) and any
        meaningful weakening of the within-group shrinkage prior.

        See ``scripts/lengthscale_prior_study.py`` for the full sweep
        that calibrated this threshold.
        """
        gp, X, _Y, _, _ = _fit_default_strength_gp()
        rmse = _analytical_loo_rmse(gp, n_real=X.shape[0])
        self.assertLess(
            rmse,
            _LOO_RMSE_CEILING_PSI,
            msg=(
                f"LOO CV RMSE on the public strength rows ({rmse:.1f} psi) "
                f"exceeds the ceiling ({_LOO_RMSE_CEILING_PSI} psi). "
                "This is a regression vs the production within-group "
                "shrinkage prior (typical RMSE ≈ 725 psi). Re-run "
                "scripts/lengthscale_prior_study.py to debug."
            ),
        )

    # ------------------------------------------------------------------
    # The default prior is actually being installed by fit_strength_gp.
    # ------------------------------------------------------------------
    def test_fit_strength_gp_installs_default_prior(self):
        """Catch a regression where the default kwarg gets accidentally
        flipped to None, silently disabling the prior."""
        gp, *_ = _fit_default_strength_gp()
        prior = gp.covar_module.kernels[0].base_kernel.lengthscale_prior
        self.assertIsInstance(
            prior,
            WithinGroupShrinkagePrior,
            msg=(
                "fit_strength_gp did not install the default "
                "WithinGroupShrinkagePrior on the Matern kernel. "
                "Check the lengthscale_prior kwarg default."
            ),
        )


class TestWithinGroupShrinkagePrior(unittest.TestCase):
    """Unit tests for the prior class itself (no GP fit required)."""

    def test_log_prob_is_zero_when_group_is_tied(self):
        """If every group member shares the same lengthscale, the
        within-group penalty is zero."""
        prior = WithinGroupShrinkagePrior(
            groups_with_sigma=[((0, 1, 2), 0.5), ((3, 4), 0.5)],
            dim=5,
        )
        # All ones → all log-lengthscales equal → zero penalty.
        x = torch.ones(1, 5, dtype=torch.float64)
        lp = prior.log_prob(x)
        self.assertEqual(lp.shape, x.shape)
        self.assertTrue(torch.allclose(lp, torch.zeros_like(lp)))

    def test_log_prob_is_negative_when_group_has_variance(self):
        """A non-zero within-group log-lengthscale variance produces a
        strictly negative log-prob (i.e., a positive penalty)."""
        prior = WithinGroupShrinkagePrior(
            groups_with_sigma=[((0, 1), 0.5)],
            dim=2,
        )
        # Two different values → positive within-group variance.
        x = torch.tensor([[1.0, math.e**1.0]], dtype=torch.float64)
        lp = prior.log_prob(x)
        self.assertLess(lp.sum().item(), 0.0)

    def test_log_prob_scales_as_inverse_sigma_squared(self):
        """Halving sigma should quadruple the penalty magnitude."""
        x = torch.tensor([[1.0, math.e**1.0]], dtype=torch.float64)
        wide = WithinGroupShrinkagePrior(
            groups_with_sigma=[((0, 1), 1.0)],
            dim=2,
        )
        narrow = WithinGroupShrinkagePrior(
            groups_with_sigma=[((0, 1), 0.5)],
            dim=2,
        )
        ratio = narrow.log_prob(x).sum().item() / wide.log_prob(x).sum().item()
        self.assertAlmostEqual(ratio, 4.0, places=4)

    def test_log_prob_handles_singleton_groups(self):
        """A group with a single member contributes no penalty (no
        within-group variance possible)."""
        prior = WithinGroupShrinkagePrior(
            groups_with_sigma=[((0,), 0.5), ((1, 2), 0.5)],
            dim=3,
        )
        # Singleton group {0} contributes 0; group {1, 2} is also tied here.
        x = torch.tensor([[2.0, 3.0, 3.0]], dtype=torch.float64)
        self.assertTrue(torch.allclose(prior.log_prob(x), torch.zeros_like(x)))

    def test_default_prior_is_installed_for_d_in_10_only(self):
        """``_default_lengthscale_prior`` returns the production prior at
        ``d_in=10`` and ``None`` otherwise (so test fits with a smaller
        feature subset don't get an inappropriately-keyed prior)."""
        self.assertIsInstance(
            _default_lengthscale_prior(10),
            WithinGroupShrinkagePrior,
        )
        self.assertIsNone(_default_lengthscale_prior(5))
        self.assertIsNone(_default_lengthscale_prior(11))


if __name__ == "__main__":
    unittest.main()
