#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for boxcrete.plotting."""

import unittest
from unittest.mock import MagicMock

import matplotlib
import matplotlib.pyplot as plt
import torch
from parameterized import parameterized

matplotlib.use("Agg")

from boxcrete.plotting import (  # noqa: E402
    compute_loo_cv,
    plot_calibration,
    plot_feature_importance,
    plot_strength_curve,
)

# Shared defaults to keep tests fast
_FAST = {"num_t": 16, "dpi": 50}


def _mock_model():
    """Creates a mock SustainableConcreteModel."""
    model = MagicMock()
    gwp_post = MagicMock()
    gwp_post.mean = torch.tensor([[-150.0]])
    model.gwp_model.posterior.return_value = gwp_post

    def _dynamic_strength_posterior(X, *args, **kwargs):
        n = X.shape[0]
        post = MagicMock()
        post.mean = torch.rand(n, 1)
        post.variance = torch.rand(n, 1).abs() + 0.01
        return post

    model.strength_model.posterior.side_effect = _dynamic_strength_posterior
    return model


class TestPlotStrengthCurve(unittest.TestCase):
    """Tests for the plot_strength_curve function."""

    def setUp(self):
        plt.close("all")
        self.model = _mock_model()
        self.compositions = torch.rand(1, 7)

    def tearDown(self):
        plt.close("all")

    def test_basic_call_delegates_to_model(self):
        plot_strength_curve(self.model, self.compositions, **_FAST)
        self.model.gwp_model.posterior.assert_called_once()
        self.model.strength_model.posterior.assert_called_once()

    def test_1d_composition_unsqueezed(self):
        plot_strength_curve(self.model, torch.rand(7), **_FAST)
        X_input = self.model.strength_model.posterior.call_args[0][0]
        self.assertEqual(X_input.shape[-1], 8)  # 7 composition + 1 time

    @parameterized.expand([(False,), (True,)])
    def test_create_fig(self, create_fig):
        plot_strength_curve(
            self.model, self.compositions, create_fig=create_fig, **_FAST
        )

    def test_no_uncertainties(self):
        plot_strength_curve(
            self.model, self.compositions, plot_uncertainties=False, **_FAST
        )

    def test_with_observed_data(self):
        plot_strength_curve(
            self.model,
            self.compositions,
            observed_data=torch.tensor([2000.0, 6000.0]),
            observed_times=torch.tensor([1.0, 28.0]),
            **_FAST,
        )

    def test_multiple_compositions_custom_colors(self):
        plot_strength_curve(
            self.model, torch.rand(3, 7), colors=["r", "g", "b"], **_FAST
        )
        self.assertEqual(self.model.strength_model.posterior.call_count, 3)


class TestPlotCalibration(unittest.TestCase):
    """Tests for the plot_calibration function."""

    def setUp(self):
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    def test_basic_call(self):
        observed = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        predicted = torch.tensor([1.2, 2.8, 5.1, 6.9, 8.8])
        fig = plot_calibration(observed, predicted, dpi=50)
        self.assertIsInstance(fig, plt.Figure)

    def test_with_error_bars(self):
        observed = torch.tensor([1.0, 3.0, 5.0])
        predicted = torch.tensor([1.2, 2.8, 5.1])
        std = torch.tensor([0.1, 0.2, 0.15])
        fig = plot_calibration(observed, predicted, predicted_std=std, dpi=50)
        self.assertIsInstance(fig, plt.Figure)

    def test_without_error_bars(self):
        observed = torch.tensor([1.0, 3.0, 5.0])
        predicted = torch.tensor([1.2, 2.8, 5.1])
        fig = plot_calibration(observed, predicted, dpi=50)
        self.assertIsInstance(fig, plt.Figure)

    def test_custom_labels(self):
        observed = torch.tensor([100.0, 200.0, 300.0])
        predicted = torch.tensor([110.0, 190.0, 310.0])
        fig = plot_calibration(
            observed,
            predicted,
            title="Strength LOO-CV",
            xlabel="Observed (psi)",
            ylabel="Predicted (psi)",
            dpi=50,
        )
        self.assertIsInstance(fig, plt.Figure)


class TestComputeLooCv(unittest.TestCase):
    """Tests for the compute_loo_cv function."""

    def test_basic_loo(self):
        """LOO-CV should return tensors of correct shape."""
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize

        torch.manual_seed(0)
        X = torch.rand(10, 3, dtype=torch.double)
        Y = (X @ torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)).unsqueeze(-1)
        model = SingleTaskGP(X, Y, outcome_transform=Standardize(1))

        loo_obs, loo_pred, loo_std = compute_loo_cv(model)
        self.assertEqual(loo_obs.shape[0], 10)
        self.assertEqual(loo_pred.shape[0], 10)
        self.assertEqual(loo_std.shape[0], 10)
        self.assertTrue((loo_std > 0).all())

    def test_n_real_truncation(self):
        """n_real should restrict output to first n_real points."""
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize

        torch.manual_seed(0)
        X = torch.rand(15, 3, dtype=torch.double)
        Y = torch.rand(15, 1, dtype=torch.double)
        model = SingleTaskGP(X, Y, outcome_transform=Standardize(1))

        loo_obs, loo_pred, loo_std = compute_loo_cv(model, n_real=10)
        self.assertEqual(loo_obs.shape[0], 10)
        self.assertEqual(loo_pred.shape[0], 10)
        self.assertEqual(loo_std.shape[0], 10)


class TestPlotFeatureImportance(unittest.TestCase):
    """Tests for the plot_feature_importance function."""

    def setUp(self):
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    def test_basic_call(self):
        lengthscales = torch.tensor([1.0, 0.5, 2.0, 0.1])
        names = ["Cement", "Fly Ash", "Slag", "Water"]
        fig = plot_feature_importance(lengthscales, names, dpi=50)
        self.assertIsInstance(fig, plt.Figure)

    def test_ordering(self):
        """Bars should be sorted by importance (1/lengthscale)."""
        lengthscales = torch.tensor([10.0, 1.0, 5.0])
        names = ["A", "B", "C"]
        fig = plot_feature_importance(lengthscales, names, dpi=50)
        ax = fig.axes[0]
        # B (smallest lengthscale = most important) should be first
        labels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertEqual(labels[0], "B")


if __name__ == "__main__":
    unittest.main()
