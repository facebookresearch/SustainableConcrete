# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Lengthscale-identifiability and within-group-tying regression tests.

Asserts three properties of the V2 strength GP that, together,
guarantee the website's interactive sliders are well-calibrated and
responsive:

    1) Identifiability — every Matern lengthscale on BOTH the
       ``matern_blind`` AND ``matern_specific`` subkernels of V2's
       V2 kernel is comfortably below the optimiser's upper
       constraint bound (cap = 1e3, set by ``LogTransformedInterval``
       in ``boxcrete/strength_model.py::_ard_matern_with_within_group_prior``).
       A railed lengthscale silently breaks the corresponding slider in
       the explorer.

    2) Within-group tying — Cement / Fly Ash / Slag share a lengthscale,
       and Fine / Coarse Aggregates share a lengthscale, on each Matern
       subkernel independently (the within-group prior is installed on
       both ``matern_blind`` and ``matern_specific`` per
       ``_build_b_double_prime_kernel_for_aug_dim``). If a future change
       weakens the prior or rewires the kernel, this breaks loudly here
       rather than as a UX regression on the site.

    3) LOO CV performance — held-out predictive RMSE on the 647 public
       strength rows is below ~750 psi, validating that the structural
       prior delivers the empirical win documented in the
       ``WithinGroupShrinkagePrior`` docstring.

Identifiability for Fly Ash and Coarse Aggregates — both under-sampled
in the public CSV — is enforced by the ``WithinGroupShrinkagePrior``
that ``boxcrete.fit_strength_gp`` (V2 production) installs on each
constituent Matern subkernel, softly tying binders {Cement, Fly Ash,
Slag} and aggregates {Fine, Coarse} within each material group.

V2's per-subkernel layout is reflected in ``docs/model/strength.json``:
both ``matern_blind`` (16-dim, no source) and ``matern_specific``
(17-dim, all dims) carry their own ``lengthscales`` / ``outputscale``
arrays, and the tests assert the tying constraint on each.
"""

import json
import math
import os
import unittest
from functools import lru_cache

import torch

from boxcrete import compute_loo_cv, fit_strength_gp
from boxcrete.priors import (
    WithinGroupShrinkagePrior,
    _AGGREGATE_LENGTHSCALE_GROUP,
    _BINDER_LENGTHSCALE_GROUP,
    _default_lengthscale_prior,
)
from boxcrete.utils import DEFAULT_X_COLUMNS, REPO_DIR, load_concrete_strength

STRENGTH_JSON_PATH = os.path.join(REPO_DIR, "docs", "model", "strength.json")
TEST_VECTORS_PATH = os.path.join(REPO_DIR, "docs", "model", "test_vectors.json")

# Maximum permissible ratio max(ℓ)/min(ℓ) within a tied material group.
# At sigma=0.001 (production), the ratio is essentially 1 (within float
# noise). 1.05 leaves comfortable margin for BLAS drift across platforms
# while still catching any real loosening of the prior.
_MAX_WITHIN_GROUP_RATIO = 1.05

# CI-blocking upper bound on LOO CV RMSE (psi) on the 647 public
# strength rows. The production fit gives ~725 psi; ~750 is a generous
# ceiling that still catches the "no prior" baseline (~772 psi) and any
# meaningful regression of the prior. The headline numbers come
# from the empirical comparison table in
# :class:`boxcrete.priors.WithinGroupShrinkagePrior`'s docstring.
_LOO_RMSE_CEILING_PSI = 750.0

# Index of the Material Source column in DEFAULT_X_COLUMNS. The
# ``matern_blind`` subkernel excludes this dim, so its lengthscale array
# is one shorter than ``matern_specific``'s and the no-source feature
# names need to match.
_SOURCE_DIM = 7

# Identifiability cap on Matern lengthscales: the V2 kernel construction
# in ``boxcrete/strength_model.py::_ard_matern_with_within_group_prior``
# wraps the Matern in ``LogTransformedInterval(1e-2, 1e3, ...)``, so a
# fitted lengthscale at or above 1e3 means the optimiser railed against
# the constraint upper bound and the corresponding feature is being
# treated as uninformative. V2 emits per-subkernel lengthscales in
# ``strength.json`` (``matern_blind.lengthscales``,
# ``matern_specific.lengthscales``) but does NOT serialise the cap
# itself; we hard-code it here, mirroring the production constraint.
_LENGTHSCALE_CAP = 1e3


@lru_cache(maxsize=1)
def _load_strength_params():
    """Load the committed strength.json once and memoise."""
    with open(STRENGTH_JSON_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _fit_default_strength_gp():
    """Fit the production strength GP once (with the default within-group
    shrinkage prior) and cache it across tests. Saves ~10s per extra test."""
    torch.manual_seed(0)
    data = load_concrete_strength()
    X, Y, Yvar, X_bounds = data.strength_data
    gp = fit_strength_gp(
        X=X,
        Y=Y,
        Yvar=Yvar,
        X_bounds=X_bounds,
    )
    return gp, X, Y, Yvar, X_bounds


def _matern_lengthscales(gp) -> list[list[float]]:
    """Extract per-subkernel lengthscales from a V2 fitted strength GP.

    V2's ``covar_module`` wraps an additive sum of three
    subkernels in a ``_TimeGatedKernel``:

        _TimeGatedKernel(
            AdditiveKernel(
                ScaleKernel(matern_blind),    # over no-source dims (+ extras)
                ScaleKernel(matern_specific), # over all 10 raw dims
                ScaleKernel(rbf_time),        # additive RBF on time
            )
        )

    The within-group shrinkage prior (binders {0,1,2}, aggregates {5,6})
    is installed independently on BOTH ``matern_blind`` AND
    ``matern_specific`` (see ``_build_b_double_prime_kernel_for_aug_dim``
    in ``boxcrete/strength_model.py``), so identifiability and tying must
    hold for both subkernels.

    Returns a list of two lists: ``[blind_lengthscales, specific_lengthscales]``.
    The two layouts differ:
      * ``blind`` lengthscales are over the no-source dims (in
        DEFAULT_X_COLUMNS order, with dim 7 / Material Source dropped).
        With the default ordering, the binder group {0,1,2} and aggregate
        group {5,6} indices remain unchanged because source sits at dim 7.
      * ``specific`` lengthscales are over all 10 raw dims; binder and
        aggregate group indices line up directly with
        ``_BINDER_LENGTHSCALE_GROUP`` and ``_AGGREGATE_LENGTHSCALE_GROUP``.
    """
    additive = gp.covar_module.base_kernel  # unwrap _TimeGatedKernel
    blind_scale = additive.kernels[0]  # ScaleKernel(matern_blind)
    specific_scale = additive.kernels[1]  # ScaleKernel(matern_specific)
    blind_ls = blind_scale.base_kernel.lengthscale.detach().squeeze().tolist()
    specific_ls = specific_scale.base_kernel.lengthscale.detach().squeeze().tolist()
    return [blind_ls, specific_ls]


def _committed_matern_lengthscales() -> list[list[float]]:
    """V2 schema: read both ``matern_blind`` and ``matern_specific``
    lengthscales from ``docs/model/strength.json``. Returns
    ``[blind_lengthscales, specific_lengthscales]``."""
    p = _load_strength_params()
    return [p["matern_blind"]["lengthscales"], p["matern_specific"]["lengthscales"]]


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
    same headline numbers documented in
    :class:`boxcrete.priors.WithinGroupShrinkagePrior`'s docstring.
    """
    obs, mean, _ = compute_loo_cv(gp, n_real=n_real)
    return float(((obs - mean) ** 2).mean().sqrt().item())


def _feature_names_from_columns():
    """Strip the ``(unit)`` suffix from ``DEFAULT_X_COLUMNS`` to match the
    consumer-friendly names emitted in ``strength.json``."""
    return [c.split(" (")[0] for c in DEFAULT_X_COLUMNS]


def _augmented_feature_names(params) -> list[str]:
    """Build the V2 augmented feature-name list (length d_aug = 17):
    10 raw composition + time dims followed by 7 engineered features.
    Mirrors the layout of ``matern_*.lengthscales`` in the JSON."""
    return list(params["raw_feature_names"]) + list(params["engineered_feature_names"])


def _collect_violations(lengthscales, feature_names, cap, n_raw_dims):
    """Return a list of human-readable strings, one per cap violation.

    Only flags RAW features. Engineered features (W/B, SCM frac,
    log(HRWR/binder), etc.) are derived from the raw composition columns
    and are not directly user-controllable in the explorer's slider UI;
    if their lengthscale rails at the cap, the kernel is saying "this
    engineered ratio is redundant with the raw inputs", which is a
    valid GP fit outcome (and one that empirically lands in different
    basins on different BLAS implementations: Linux x86_64 sometimes
    rails ``wb_ratio`` at the constraint upper bound while Apple
    Silicon / amd64-emulated Linux land at a finite lengthscale ~10).
    Raw-feature rails would mean a slider with no effect on
    predictions, so we keep the tight check there.
    """
    return [
        f"{name} (idx {i}): {lengthscales[i]:.2f}"
        for i, name in enumerate(feature_names)
        if i < n_raw_dims and lengthscales[i] >= cap
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
        gp, *_ = _fit_default_strength_gp()
        blind_ls, specific_ls = _matern_lengthscales(gp)
        params = _load_strength_params()
        aug_names = _augmented_feature_names(params)

        # ``matern_specific`` covers all d_aug = 17 augmented dims.
        # ``matern_blind`` excludes the source dim (index 7), so it has
        # 16 lengthscales — assert with the source feature dropped.
        no_source_aug_names = [n for i, n in enumerate(aug_names) if i != _SOURCE_DIM]
        self.assertEqual(len(specific_ls), len(aug_names))
        self.assertEqual(len(blind_ls), len(no_source_aug_names))
        # Number of raw dims (composition + time, before engineered
        # features are appended). matern_blind drops the source dim, so
        # its raw count is one less.
        n_raw_specific = len(params["raw_feature_names"])
        n_raw_blind = n_raw_specific - 1
        for ls, names, label, n_raw in [
            (specific_ls, aug_names, "matern_specific", n_raw_specific),
            (blind_ls, no_source_aug_names, "matern_blind", n_raw_blind),
        ]:
            self._assert_no_violations(
                _collect_violations(ls, names, _LENGTHSCALE_CAP, n_raw),
                source_label=f"Freshly-fit strength GP ({label})",
                remediation=(
                    "This means the GP is treating those features as "
                    "uninformative — the corresponding sliders in the "
                    "website's Composition panel will not respond. "
                    "Investigate data churn or fit instability."
                ),
            )

    def test_committed_strength_json_has_identifiable_lengthscales(self):
        """Sanity-check the *committed* model artifact too. CI also relies on
        this to catch the case where the JSON wasn't regenerated after a
        data change."""
        if not os.path.exists(STRENGTH_JSON_PATH):
            self.skipTest(f"committed model not found at {STRENGTH_JSON_PATH}")
        params = _load_strength_params()

        # Loud failure if the export script ever forgets to emit either
        # field — that would silently fall back to local hard-coded values.
        self.assertIn("raw_feature_names", params)
        self.assertIn("engineered_feature_names", params)

        aug_names = _augmented_feature_names(params)
        no_source_aug_names = [n for i, n in enumerate(aug_names) if i != _SOURCE_DIM]
        n_raw_specific = len(params["raw_feature_names"])
        n_raw_blind = n_raw_specific - 1
        for subkernel, names, n_raw in [
            ("matern_specific", aug_names, n_raw_specific),
            ("matern_blind", no_source_aug_names, n_raw_blind),
        ]:
            ls = params[subkernel]["lengthscales"]
            self._assert_no_violations(
                _collect_violations(ls, names, _LENGTHSCALE_CAP, n_raw),
                source_label=f"Committed docs/model/strength.json ({subkernel})",
                remediation=(
                    "Re-run `python experiments/regenerate_strength_json.py` after "
                    "updating the data, and re-commit the regenerated JSON."
                ),
            )

    def test_committed_strength_matches_fresh_fit(self):
        """Catch the "forgot to re-run regen after a model code change"
        failure mode: a freshly-fit V2 strength GP should produce
        predictions that agree with the baked ``expected_mean`` values
        in ``docs/model/test_vectors.json`` at the 37 out-of-training
        compositions × times in that file.

        Why we compare *predictions at OOT compositions* rather than
        *raw lengthscales* (the older approach):

        Two GP fits with different lengthscales can be near-equivalent
        as predictors. The V2 strength GP's MLL surface has multiple
        nearby local optima — different BLAS implementations on
        different architectures (macOS Accelerate, Linux OpenBLAS, …)
        steer L-BFGS-B into different basins, yielding lengthscales
        that drift by 25-200% per dim across machines yet predict
        nearly the same surface (empirically: max abs diff ~50 psi /
        max rel diff ~5% at the 37 test_vectors compositions for
        Mac vs Linux at the same gpytorch / botorch versions). Asserting
        on raw lengthscales is therefore not portable; asserting on the
        prediction surface — which is what the explorer + the JS port
        actually consume — is.

        The check intentionally evaluates at *out-of-training*
        compositions to avoid the GP-memorises-training-data tautology
        (any low-noise GP interpolates training data tightly regardless
        of which optimum the lengthscales landed in). The OOT grid in
        ``test_vectors.json`` is the right surface to compare on
        because (a) it is what the JS-side ``test_data_freshness.mjs``
        consumes and (b) it is what
        ``test_pretrained_loader_fidelity.py`` consumes — using the
        same grid here keeps the three artifacts coherent.

        Tolerance per test vector: ``delta = max(250 psi, 5% × |E|)``.
        Sized empirically against the cross-runner / cross-architecture
        drift envelope observed across this PR's CI saga (worst case
        ~230 psi absolute / ~4.3% relative on certain GitHub-hosted
        ubuntu-latest VM hosts). Different physical Azure hosts have
        slightly different CPU rounding, which feeds into L-BFGS-B and
        occasionally lands the multi-modal MLL fit in a different local
        optimum even with float64 throughout. The 250 psi absolute floor
        handles low-magnitude predictions cleanly (where 5% relative
        would require sub-psi precision), and the 5% relative ceiling
        still catches structural model changes (which typically shift
        OOT predictions by 10-50%+).

        Same contract as ``experiments/check_artifacts_drift.py``, which
        guards the regen-idempotency CI workflow.

        If a future architecture or BLAS implementation yields drift
        beyond this band on this committed dataset, the right fix is
        either (a) regenerate ``docs/model/test_vectors.json`` on that
        architecture (committed values match the test runner) or (b)
        widen these bounds with a follow-up commit + a measurement.

        Companion structural checks (architecture-portable, no
        cross-arch sensitivity):
        ``test_committed_strength_json_has_identifiable_lengthscales``
        (no rail-cap violations),
        ``test_committed_binder_lengthscales_tied`` /
        ``test_committed_aggregate_lengthscales_tied`` (within-group
        prior intact),
        ``test_loo_cv_rmse_does_not_regress`` (model performance).
        """
        if not os.path.exists(STRENGTH_JSON_PATH):
            self.skipTest(f"committed model not found at {STRENGTH_JSON_PATH}")
        if not os.path.exists(TEST_VECTORS_PATH):
            self.skipTest(f"test vectors not found at {TEST_VECTORS_PATH}")
        with open(TEST_VECTORS_PATH) as f:
            test_vectors = json.load(f)["test_vectors"]
        if not test_vectors:
            self.skipTest("test_vectors.json contains no test vectors")

        gp, *_ = _fit_default_strength_gp()
        gp.eval()
        # The V2 fit factory stores the un-standardisation stats as
        # buffers (``_study_y_mean_buf`` / ``_study_y_std_buf``) so they
        # round-trip through state_dict; the legacy attribute aliases
        # (``_study_y_mean`` / ``_study_y_std``) read the same buffers.
        y_std = float(gp._study_y_std)
        y_mean = float(gp._study_y_mean)

        worst_abs = 0.0
        worst_rel = 0.0
        worst_label = ""
        for v in test_vectors:
            x = torch.tensor([v["composition"] + [v["time"]]], dtype=torch.float64)
            with torch.no_grad():
                pred_scaled = gp.posterior(x).mean.item()
            predicted = pred_scaled * y_std + y_mean
            expected = v["expected_mean"]
            abs_diff = abs(predicted - expected)
            rel_diff = abs_diff / max(abs(expected), 1.0)
            # Per-vector tolerance: max(250 psi absolute, 5% relative).
            # Sized against the empirical cross-runner / cross-arch
            # drift envelope (~230 psi / ~4.3% rel on certain Azure
            # hosts in this PR's CI saga). Same contract as
            # ``experiments/check_artifacts_drift.py``, which guards
            # the regen-idempotency CI workflow.
            delta = max(250.0, 0.05 * abs(expected))
            if abs_diff > worst_abs:
                worst_abs = abs_diff
                worst_rel = rel_diff
                worst_label = (
                    f"composition={v['composition']}, time={v['time']}d, "
                    f"expected={expected:.2f} psi, predicted={predicted:.2f} psi"
                )
            self.assertLessEqual(
                abs_diff,
                delta,
                msg=(
                    f"Fresh fit's prediction at OOT composition "
                    f"{v['composition']} (t={v['time']}d) differs from "
                    f"the committed test_vectors.json's expected_mean "
                    f"by {abs_diff:.2f} psi ({rel_diff * 100:.1f}% rel; "
                    f"tolerance was {delta:.2f} psi). This usually "
                    f"means a stale export — the model code or features "
                    f"changed but docs/model/*.json wasn't regenerated. "
                    f"Re-run `python experiments/regenerate_strength_json.py` "
                    f"and commit the regenerated JSON. "
                    f"expected={expected:.4f}, predicted={predicted:.4f}"
                ),
            )
        # Diagnostic on the worst case (always reported in logs).
        print(
            f"\n[test_committed_strength_matches_fresh_fit] worst case: "
            f"{worst_abs:.2f} psi ({worst_rel * 100:.2f}% rel) at "
            f"{worst_label}"
        )

    # ------------------------------------------------------------------
    # Within-group tying (binders {Cement, FA, Slag}, aggregates {Fine, Coarse}).
    # These tests fail if the WithinGroupShrinkagePrior is weakened or
    # removed — see boxcrete/priors.py for the production prior settings.
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
                f"Check that boxcrete.priors.WithinGroupShrinkagePrior is "
                f"installed by fit_strength_gp with "
                f"_LENGTHSCALE_SHRINKAGE_SIGMA <= 0.01."
            ),
        )

    def test_committed_binder_lengthscales_tied(self):
        """Cement/FA/Slag share a lengthscale in the committed JSON.

        The within-group prior is installed on BOTH the source-aware
        ``matern_specific`` AND the source-blind ``matern_blind``
        subkernels, so we assert tying on each independently.
        """
        if not os.path.exists(STRENGTH_JSON_PATH):
            self.skipTest(f"committed model not found at {STRENGTH_JSON_PATH}")
        blind_ls, specific_ls = _committed_matern_lengthscales()
        self._assert_group_tied(
            specific_ls, _BINDER_LENGTHSCALE_GROUP, group_label="matern_specific Binder"
        )
        # In matern_blind, source dim 7 is removed; binder group {0,1,2}
        # indices stay the same because source sits at dim 7 (above them).
        self._assert_group_tied(
            blind_ls, _BINDER_LENGTHSCALE_GROUP, group_label="matern_blind Binder"
        )

    def test_committed_aggregate_lengthscales_tied(self):
        """Fine and Coarse Aggregates share a lengthscale in the committed JSON.

        Asserts tying on both ``matern_specific`` and ``matern_blind``.
        """
        if not os.path.exists(STRENGTH_JSON_PATH):
            self.skipTest(f"committed model not found at {STRENGTH_JSON_PATH}")
        blind_ls, specific_ls = _committed_matern_lengthscales()
        self._assert_group_tied(
            specific_ls,
            _AGGREGATE_LENGTHSCALE_GROUP,
            group_label="matern_specific Aggregate",
        )
        # In matern_blind, source dim 7 is removed; aggregate group {5,6}
        # stays the same because source sits at dim 7 (above them).
        self._assert_group_tied(
            blind_ls,
            _AGGREGATE_LENGTHSCALE_GROUP,
            group_label="matern_blind Aggregate",
        )

    def test_freshly_fit_binder_lengthscales_tied(self):
        """Production prior actually ties the binder lengthscales at fit time
        on both ``matern_specific`` and ``matern_blind``."""
        gp, *_ = _fit_default_strength_gp()
        blind_ls, specific_ls = _matern_lengthscales(gp)
        self._assert_group_tied(
            specific_ls, _BINDER_LENGTHSCALE_GROUP, group_label="matern_specific Binder"
        )
        self._assert_group_tied(
            blind_ls, _BINDER_LENGTHSCALE_GROUP, group_label="matern_blind Binder"
        )

    def test_freshly_fit_aggregate_lengthscales_tied(self):
        """Production prior actually ties the aggregate lengthscales at fit time
        on both ``matern_specific`` and ``matern_blind``."""
        gp, *_ = _fit_default_strength_gp()
        blind_ls, specific_ls = _matern_lengthscales(gp)
        self._assert_group_tied(
            specific_ls,
            _AGGREGATE_LENGTHSCALE_GROUP,
            group_label="matern_specific Aggregate",
        )
        self._assert_group_tied(
            blind_ls,
            _AGGREGATE_LENGTHSCALE_GROUP,
            group_label="matern_blind Aggregate",
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

        See :class:`boxcrete.priors.WithinGroupShrinkagePrior`'s
        docstring (Empirical comparison section) for the full sweep
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
                "shrinkage prior (typical RMSE ≈ 725 psi). See the "
                "WithinGroupShrinkagePrior docstring (Empirical "
                "comparison) for expected per-sigma RMSEs."
            ),
        )

    # ------------------------------------------------------------------
    # The default prior is actually being installed by fit_strength_gp.
    # ------------------------------------------------------------------
    def test_fit_strength_gp_installs_default_prior(self):
        """Catch a regression where the default kwarg gets accidentally
        flipped to None, silently disabling the prior. V2's production fit
        installs the ``WithinGroupShrinkagePrior`` independently on
        BOTH ``matern_blind`` AND ``matern_specific`` (see
        ``_build_b_double_prime_kernel_for_aug_dim`` in
        ``boxcrete/strength_model.py``)."""
        gp, *_ = _fit_default_strength_gp()
        # Navigate the V2 kernel:
        # _TimeGatedKernel(AdditiveKernel(
        #   ScaleKernel(matern_blind), ScaleKernel(matern_specific),
        #   ScaleKernel(rbf_time)
        # ))
        additive = gp.covar_module.base_kernel
        for label, idx in [("matern_blind", 0), ("matern_specific", 1)]:
            matern = additive.kernels[idx].base_kernel
            self.assertIsInstance(
                matern.lengthscale_prior,
                WithinGroupShrinkagePrior,
                msg=(
                    f"fit_strength_gp did not install the default "
                    f"WithinGroupShrinkagePrior on {label}. "
                    "Check the lengthscale_prior kwarg default in "
                    "_build_b_double_prime_kernel_for_aug_dim."
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
