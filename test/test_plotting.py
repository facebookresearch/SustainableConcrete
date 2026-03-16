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

from boxcrete.plotting import plot_strength_curve  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
