#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for boxcrete.models."""

import unittest
from unittest.mock import MagicMock

import torch
from parameterized import parameterized

from boxcrete.models import (
    FixedFeatureModel,
    SustainableConcreteModel,
    fit_gwp_gp,
    fit_strength_gp,
    get_strength_gp_input_transform,
)

# Limit optimizer iterations in tests for speed (follows BoTorch testing convention)
FAST_FIT_KWARGS = {"options": {"maxiter": 1}}


class BaseModelTest(unittest.TestCase):
    """Base class with shared synthetic data for model tests."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.dtype = torch.double
        cls.n, cls.d = 20, 8
        cls.strength_days = [1, 28]
        cls.X = torch.rand(cls.n, cls.d, dtype=cls.dtype) * 500 + 100
        cls.X[:, -1] = torch.tensor(
            [1, 7, 28] * (cls.n // 3) + [1] * (cls.n % 3), dtype=cls.dtype
        )
        cls.Y_strength = torch.rand(cls.n, 1, dtype=cls.dtype) * 5000 + 1000
        cls.Y_gwp = torch.rand(cls.n, 1, dtype=cls.dtype) * 100 + 50
        cls.Yvar = torch.full((cls.n, 1), 0.01, dtype=cls.dtype)
        cls.bounds = torch.stack(
            [
                torch.zeros(cls.d, dtype=cls.dtype),
                torch.ones(cls.d, dtype=cls.dtype) * 1000,
            ]
        )
        cls.bounds[1, -1] = 28

    def _mock_dataset(self):
        mock = MagicMock()
        mock.strength_data = (self.X, self.Y_strength, self.Yvar, self.bounds)
        mock.gwp_data = (
            self.X[:, :-1],
            self.Y_gwp,
            self.Yvar,
            self.bounds[:, :-1],
        )
        return mock

    def _mock_base_model(self, num_outputs=1):
        m = MagicMock()
        m.num_outputs = num_outputs
        return m


class TestFixedFeatureModel(BaseModelTest):
    """Tests for the FixedFeatureModel class."""

    def test_initialization_and_properties(self):
        model = FixedFeatureModel(
            base_model=self._mock_base_model(3), dim=5, indices=[4], values=[1.0]
        )
        self.assertEqual(model._dim, 5)
        self.assertEqual(model.num_outputs, 3)

    def test_initialization_mismatched_indices_values(self):
        with self.assertRaises(ValueError):
            FixedFeatureModel(
                base_model=self._mock_base_model(),
                dim=5,
                indices=[4],
                values=[1.0, 2.0],
            )

    @parameterized.expand([([4], [7.0]), ([3, 4], [7.0, 14.0]), ([0], [1.0])])
    def test_add_fixed_features(self, indices, values):
        """Test shape and value insertion for _add_fixed_features."""
        original_dim = 4
        model = FixedFeatureModel(
            base_model=self._mock_base_model(),
            dim=original_dim + len(indices),
            indices=indices,
            values=values,
        )
        X = torch.rand(5, original_dim, dtype=self.dtype)
        Z = model._add_fixed_features(X)
        self.assertEqual(Z.shape, (5, original_dim + len(indices)))
        for idx, val in zip(indices, values):
            torch.testing.assert_close(
                Z[:, idx], torch.full((5,), val, dtype=self.dtype)
            )

    @parameterized.expand([("forward",), ("posterior",)])
    def test_delegates_to_base_model(self, method):
        """Test that forward/posterior delegate to base model with augmented input."""
        base = self._mock_base_model()
        setattr(base, method, MagicMock(return_value=MagicMock()))
        model = FixedFeatureModel(base_model=base, dim=5, indices=[4], values=[1.0])
        getattr(model, method)(torch.rand(5, 4, dtype=self.dtype))
        getattr(base, method).assert_called_once()
        self.assertEqual(getattr(base, method).call_args[0][0].shape[-1], 5)

    def test_subset_output(self):
        base = self._mock_base_model()
        base.subset_output = MagicMock(return_value=MagicMock())
        model = FixedFeatureModel(base_model=base, dim=5, indices=[4], values=[1.0])
        result = model.subset_output([0])
        self.assertIsInstance(result, FixedFeatureModel)
        base.subset_output.assert_called_once_with([0])


class TestSustainableConcreteModel(BaseModelTest):
    """Tests for SustainableConcreteModel. GP models are fit once in setUpClass."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Fit models once (expensive) and reuse across tests
        cls.model = SustainableConcreteModel(strength_days=[1, 28])
        mock_data = MagicMock()
        mock_data.strength_data = (cls.X, cls.Y_strength, cls.Yvar, cls.bounds)
        mock_data.gwp_data = (
            cls.X[:, :-1],
            cls.Y_gwp,
            cls.Yvar,
            cls.bounds[:, :-1],
        )
        cls.model.fit_gwp_model(mock_data)
        cls.model.fit_strength_model(mock_data)

    def test_initialization(self):
        model = SustainableConcreteModel(strength_days=self.strength_days)
        self.assertEqual(model.strength_days, self.strength_days)
        self.assertIsNone(model.strength_model)
        self.assertIsNone(model.gwp_model)
        self.assertIsNone(model.d)

    def test_set_d_preserves_existing(self):
        model = SustainableConcreteModel(strength_days=self.strength_days, d=5)
        model._set_d(8)
        self.assertEqual(model.d, 5)

    def test_get_model_list_before_fit_raises(self):
        with self.assertRaises(ValueError):
            SustainableConcreteModel(strength_days=self.strength_days).get_model_list()

    def test_fit_sets_models_and_d(self):
        self.assertIsNotNone(self.model.strength_model)
        self.assertIsNotNone(self.model.gwp_model)
        self.assertIsNotNone(self.model.d)

    @parameterized.expand([([1, 28], 3), ([1, 7, 28], 4), ([28], 2)])
    def test_model_list_structure(self, strength_days, expected_outputs):
        """Test model list with different strength_days (reuses pre-fit models)."""
        model = SustainableConcreteModel(
            strength_days=strength_days,
            strength_model=self.model.strength_model,
            gwp_model=self.model.gwp_model,
            d=self.model.d,
        )
        model_list = model.get_model_list()
        self.assertEqual(len(model_list.models), expected_outputs)


class TestFitGP(BaseModelTest):
    """Tests for fit_gwp_gp and fit_strength_gp with fast optimizer."""

    @parameterized.expand([(False,), (True,)])
    def test_fit_gwp_gp(self, use_fixed_noise):
        X, Y, Yvar = self.X[:, :-1], self.Y_gwp, self.Yvar
        model = fit_gwp_gp(
            X=X,
            Y=Y,
            Yvar=Yvar,
            X_bounds=self.bounds[:, :-1],
            use_fixed_noise=use_fixed_noise,
            optimizer_kwargs=FAST_FIT_KWARGS,
        )
        self.assertEqual(model.num_outputs, 1)
        post = model.posterior(torch.rand(3, X.shape[-1], dtype=self.dtype) * 500 + 100)
        self.assertTrue(torch.all(post.variance > 0))

    def test_fit_gwp_gp_invalid_output_dim(self):
        with self.assertRaises(ValueError):
            fit_gwp_gp(
                X=self.X[:, :-1],
                Y=torch.rand(self.n, 2, dtype=self.dtype),
                Yvar=torch.rand(self.n, 2, dtype=self.dtype),
            )

    @parameterized.expand([(False, True), (True, True), (False, False)])
    def test_fit_strength_gp(self, use_fixed_noise, with_bounds):
        bounds = self.bounds if with_bounds else None
        model = fit_strength_gp(
            X=self.X,
            Y=self.Y_strength,
            Yvar=self.Yvar,
            X_bounds=bounds,
            use_fixed_noise=use_fixed_noise,
            optimizer_kwargs=FAST_FIT_KWARGS,
        )
        self.assertEqual(model.num_outputs, 1)
        post = model.posterior(self.X[:3])
        self.assertTrue(torch.all(post.variance > 0))

    def test_fit_strength_gp_invalid_output_dim(self):
        with self.assertRaises(ValueError):
            fit_strength_gp(
                X=self.X,
                Y=torch.rand(self.n, 2, dtype=self.dtype),
                Yvar=torch.rand(self.n, 2, dtype=self.dtype),
            )


class TestGetStrengthGPInputTransform(BaseModelTest):
    """Tests for get_strength_gp_input_transform."""

    @parameterized.expand([(8, True), (5, True), (8, False)])
    def test_input_transform(self, d, with_bounds):
        bounds = self.bounds[:, :d] if with_bounds else None
        tf = get_strength_gp_input_transform(d=d, bounds=bounds)
        X = torch.rand(10, d, dtype=self.dtype)
        X[:, -1] = torch.randint(1, 29, (10,)).double()
        self.assertEqual(tf(X).shape, X.shape)


if __name__ == "__main__":
    unittest.main()
