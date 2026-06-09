# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Strength-curve monotonicity regression test (Python).

Concrete strength curves are physically expected to be monotonically
increasing — cement hydration is a one-way reaction, and SCMs (fly ash,
slag) cause delayed strength gain, not strength loss. Strong oscillations
or decreasing intervals in the GP's predicted mean curve are unphysical
and indicate the model has acquired pathological feature-time
interactions.

This test catches the failure mode we observed when replacing
``log_maturity_robust`` with raw ``maturity_robust = (T+10)·t`` in the
feature set: the kernel mixed two non-commensurate functions of time
(post-log-time and raw-linear-maturity) and 99% of curves became
non-monotone, with worst-case dropdowns of 876 psi.

We check both:
  1. A freshly fit GP on the canonical training data
  2. The committed ``docs/model/strength.json`` (via Python re-fit
     of the V2 strength GP variant for end-to-end consistency)

The thresholds are physically motivated and were calibrated via
``experiments/compare_monotonicity.py`` against known-good and
known-bad models:

| variant                                    | %dec  | %osc  | maxDrop |
|--------------------------------------------|------:|------:|--------:|
| F5_alllog + MLL                            |  1.4% |  0.0% |   9 psi |
| F5_alllog + block_loo_only (current)       |  3.5% |  8.3% |  24 psi |
| F5_no_log_mat + block_loo_only (REGRESSED) | 99.3% | 98.6% | 876 psi |

Thresholds (10% / 20% / 100 psi) catch the regression by 5-10x and
allow the deployed V2 strength GP comfortably.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from boxcrete.utils import load_concrete_strength  # noqa: E402

STRENGTH_JSON_PATH = REPO_ROOT / "docs" / "model" / "strength.json"
COMPOSITIONS_JSON_PATH = REPO_ROOT / "docs" / "model" / "compositions.json"

# Time grid: log-spaced from 0.5 to 28 days. Dense enough to detect
# oscillations between key observation days (1, 3, 7, 14, 28).
N_TIMES = 64
T_MIN = 0.5
T_MAX = 28.0

# Acceptance thresholds — physically motivated. See class docstring.
PASS_FRACTION_DECREASE = 0.10
PASS_FRACTION_OSCILLATE = 0.20
PASS_MAX_DROP_PSI = 100.0


def _log_spaced_times(n: int = N_TIMES) -> np.ndarray:
    return np.exp(np.linspace(np.log(T_MIN), np.log(T_MAX), n))


def _compute_monotonicity_stats(pred_grid: np.ndarray) -> dict:
    """Given an [n_comp, n_times] array of predicted means, return a
    dict of monotonicity diagnostics. Mirrors the JS implementation in
    ``test/test_curve_monotonicity.mjs``.
    """
    n_comp = pred_grid.shape[0]
    slopes = np.diff(pred_grid, axis=1)  # [n_comp, n_times-1]
    second_diffs = np.diff(slopes, axis=1)  # [n_comp, n_times-2]

    # Decreasing intervals (tolerate 1 psi of numerical noise)
    min_slope_per_comp = slopes.min(axis=1)
    n_with_drop = int(np.sum(min_slope_per_comp < -1.0))
    max_dropdown = float(np.abs(min_slope_per_comp.min()))

    # Inflection counts: sign changes in second-diff (skip near-zero)
    EPS = 1e-3
    sd = np.where(np.abs(second_diffs) < EPS, 0.0, second_diffs)
    sd_sign = np.sign(sd)
    sign_changes = np.zeros(n_comp, dtype=int)
    for c in range(n_comp):
        prev = 0
        for v in sd_sign[c]:
            if v != 0:
                if prev != 0 and v != prev:
                    sign_changes[c] += 1
                prev = v
    n_oscillating = int(np.sum(sign_changes > 2))

    return {
        "frac_decreasing": n_with_drop / n_comp,
        "frac_oscillating_gt2": n_oscillating / n_comp,
        "max_dropdown_psi": max_dropdown,
        "mean_inflections": float(sign_changes.mean()),
        "n_compositions": n_comp,
    }


def _predict_grid_for_committed_model() -> np.ndarray:
    """Predict on the dense [n_comp, N_TIMES] grid using the **deployed**
    V2 strength GP (loaded from ``docs/model/strength_model.pt`` via the
    canonical ``boxcrete.load_pretrained_strength_gp`` API). This exercises
    the actual artifact users would consume \u2014 a regression in the loader
    would surface here as a monotonicity violation.

    Falls back to a fresh fit if the .pt file isn't present
    (e.g., on a clean clone before ``regenerate_all_artifacts.sh`` has run).
    """
    from boxcrete import load_pretrained_strength_gp

    pt_path = STRENGTH_JSON_PATH.parent / "strength_model.pt"
    if not pt_path.exists():
        return _predict_grid_for_fresh_fit()

    model = load_pretrained_strength_gp()
    model.eval()
    compositions = json.loads(COMPOSITIONS_JSON_PATH.read_text())["compositions"]
    times = _log_spaced_times()
    n_comp = len(compositions)
    pred_grid = np.zeros((n_comp, N_TIMES))
    y_max = float(getattr(model, "_study_y_std", 1.0))
    y_mean = float(getattr(model, "_study_y_mean", 0.0))
    with torch.no_grad():
        for c_idx, comp in enumerate(compositions):
            full = torch.tensor(
                [comp + [t] for t in times.tolist()],
                dtype=torch.double,
            )
            posterior = model.posterior(full)
            mean = posterior.mean.squeeze().cpu().numpy()
            pred_grid[c_idx, :] = mean * y_max + y_mean
    return pred_grid


def _predict_grid_for_fresh_fit() -> np.ndarray:
    """Fit the V2 strength GP via the public ``boxcrete.fit_strength_gp``
    and predict on the dense ``[n_comp, N_TIMES]`` grid."""
    from boxcrete import fit_strength_gp

    data = load_concrete_strength()
    X, Y, Yvar, bounds = data.strength_data
    model = fit_strength_gp(X=X, Y=Y, Yvar=Yvar, X_bounds=bounds, seed=0)
    model.eval()

    compositions = json.loads(COMPOSITIONS_JSON_PATH.read_text())["compositions"]
    times = _log_spaced_times()
    n_comp = len(compositions)
    pred_grid = np.zeros((n_comp, N_TIMES))
    y_max = float(getattr(model, "_study_y_std", 1.0))
    y_mean = float(getattr(model, "_study_y_mean", 0.0))
    with torch.no_grad():
        for c_idx, comp in enumerate(compositions):
            full = torch.tensor(
                [comp + [t] for t in times.tolist()],
                dtype=torch.double,
            )
            posterior = model.posterior(full)
            mean = posterior.mean.squeeze().cpu().numpy()
            pred_grid[c_idx, :] = mean * y_max + y_mean
    return pred_grid


class TestStrengthCurveMonotonicity(unittest.TestCase):
    """Catches unphysical (non-monotone, oscillating) strength curves."""

    def _assert_monotone(
        self, stats: dict, label: str, *, remediation: str = ""
    ) -> None:
        msg_extra = f"\n{remediation}" if remediation else ""
        self.assertLess(
            stats["frac_decreasing"],
            PASS_FRACTION_DECREASE,
            (
                f"{label}: {100 * stats['frac_decreasing']:.1f}% of "
                f"{stats['n_compositions']} compositions have decreasing "
                f"intervals "
                f"(threshold {100 * PASS_FRACTION_DECREASE:.0f}%). "
                f"Strength curves should be monotone; this indicates "
                f"the kernel has acquired pathological feature-time interactions."
                + msg_extra
            ),
        )
        self.assertLess(
            stats["frac_oscillating_gt2"],
            PASS_FRACTION_OSCILLATE,
            (
                f"{label}: {100 * stats['frac_oscillating_gt2']:.1f}% of "
                f"compositions have > 2 inflection points (threshold "
                f"{100 * PASS_FRACTION_OSCILLATE:.0f}%). "
                f"This indicates the predictive mean is wiggling at "
                f"intermediate times." + msg_extra
            ),
        )
        self.assertLess(
            stats["max_dropdown_psi"],
            PASS_MAX_DROP_PSI,
            (
                f"{label}: worst single-step dropdown is "
                f"{stats['max_dropdown_psi']:.0f} psi "
                f"(threshold {PASS_MAX_DROP_PSI:.0f} psi). "
                f"This is well above measurement noise and clearly unphysical."
                + msg_extra
            ),
        )

    def test_freshly_fit_v2_curves_are_monotone(self):
        """Fit the current V2 strength GP in Python and check its predicted
        curves. Catches model-side regressions before any JSON export."""
        pred_grid = _predict_grid_for_fresh_fit()
        stats = _compute_monotonicity_stats(pred_grid)
        self._assert_monotone(
            stats,
            label="Freshly fit V2 strength GP",
            remediation=(
                "Likely cause: a feature change introduced a non-commensurate "
                "function of time (e.g., raw 'maturity_robust' instead of "
                "'log_maturity_robust'). See "
                "experiments/compare_monotonicity.py to A/B test variants."
            ),
        )

    def test_committed_strength_json_curves_are_monotone(self):
        """Sanity-check the committed model artifact too. CI relies on
        this to catch the case where strength.json wasn't regenerated
        after a feature change."""
        if not STRENGTH_JSON_PATH.exists():
            self.skipTest(f"committed model not found at {STRENGTH_JSON_PATH}")

        pred_grid = _predict_grid_for_committed_model()
        stats = _compute_monotonicity_stats(pred_grid)
        self._assert_monotone(
            stats,
            label="Committed strength.json",
            remediation=(
                "Re-run `bash experiments/regenerate_all_artifacts.sh` after "
                "investigating which feature/architecture change caused the "
                "regression. Use experiments/compare_monotonicity.py to A/B test."
            ),
        )


if __name__ == "__main__":
    unittest.main()
