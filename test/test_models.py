#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for boxcrete.models."""

import unittest
from unittest.mock import MagicMock

import torch
from botorch.models import SingleTaskGP
from boxcrete.models import (
    fit_gwp_gp,
    fit_slump_gp,
    fit_strength_gp,
    FixedFeatureModel,
    get_strength_gp_input_transform,
    SustainableConcreteModel,
)
from boxcrete.utils import DATA_PATH, load_concrete_strength, SLUMP_Y_COLUMNS
from parameterized import parameterized

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

    def test_fixed_features_unsorted_indices(self):
        """Regression test: unsorted indices must place values at correct positions."""
        base = self._mock_base_model()
        # Deliberately pass unsorted indices: [4, 2]
        model = FixedFeatureModel(
            base_model=base, dim=6, indices=[4, 2], values=[99.0, 77.0]
        )
        X = torch.rand(3, 4, dtype=self.dtype)
        Z = model._add_fixed_features(X)
        # Position 2 should have 77.0 and position 4 should have 99.0
        torch.testing.assert_close(Z[:, 2], torch.full((3,), 77.0, dtype=self.dtype))
        torch.testing.assert_close(Z[:, 4], torch.full((3,), 99.0, dtype=self.dtype))

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
        self.assertIsNone(model.slump_model)
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

    @parameterized.expand([([1, 28], 4), ([28], 3)])
    def test_model_list_structure_with_slump(self, strength_days, expected_outputs):
        """Test model list with slump model included."""
        # Use a real SingleTaskGP as the slump model since ModelList requires Module
        n, d = 10, self.d - 1  # slump has no time
        slump_model = SingleTaskGP(
            train_X=torch.rand(n, d, dtype=self.dtype),
            train_Y=torch.rand(n, 1, dtype=self.dtype),
        )
        model = SustainableConcreteModel(
            strength_days=strength_days,
            strength_model=self.model.strength_model,
            gwp_model=self.model.gwp_model,
            slump_model=slump_model,
            d=self.model.d,
        )
        model_list = model.get_model_list()
        self.assertEqual(len(model_list.models), expected_outputs)
        # Slump should be last (after GWP and strength models)
        self.assertIs(model_list.models[-1], slump_model)

    def test_get_model_list_without_fixed_features(self):
        """Default (no fixed_features) should fix only Time."""
        model_list = self.model.get_model_list()
        self.assertEqual(len(model_list.models), 3)
        # GWP model should be the raw gwp model (not wrapped)
        self.assertIs(model_list.models[0], self.model.gwp_model)
        # Strength models should be FixedFeatureModel
        for i in range(1, len(model_list.models)):
            self.assertIsInstance(model_list.models[i], FixedFeatureModel)

    def test_get_model_list_with_fixed_features(self):
        """With fixed_features, GWP model gets wrapped too."""
        fixed = {5: 0.0}  # fix feature index 5
        model_list = self.model.get_model_list(fixed_features=fixed)
        self.assertEqual(len(model_list.models), 3)
        # GWP model should be wrapped in FixedFeatureModel
        self.assertIsInstance(model_list.models[0], FixedFeatureModel)
        # Strength models should also have extra fixed features
        for i in range(1, len(model_list.models)):
            self.assertIsInstance(model_list.models[i], FixedFeatureModel)
            # Should fix Time + the extra feature
            self.assertEqual(len(model_list.models[i]._indices), 2)

    def test_get_model_list_fixed_features_time_only(self):
        """If fixed_features only contains Time index, GWP is not wrapped."""
        time_idx = self.model.d - 1
        fixed = {time_idx: 14.0}
        model_list = self.model.get_model_list(fixed_features=fixed)
        # GWP should not be wrapped since only Time was in fixed_features
        self.assertIs(model_list.models[0], self.model.gwp_model)

    def test_model_names_without_slump(self):
        """Test model_names when slump is not fitted."""
        names = self.model.model_names
        self.assertEqual(names, ["GWP", "1-day Strength", "28-day Strength"])

    def test_model_names_with_slump(self):
        """Test model_names when slump is fitted (should be last)."""
        n, d = 10, self.d - 1
        slump_model = SingleTaskGP(
            train_X=torch.rand(n, d, dtype=self.dtype),
            train_Y=torch.rand(n, 1, dtype=self.dtype),
        )
        model = SustainableConcreteModel(
            strength_days=[1, 28],
            strength_model=self.model.strength_model,
            gwp_model=self.model.gwp_model,
            slump_model=slump_model,
            d=self.model.d,
        )
        names = model.model_names
        self.assertEqual(
            names, ["GWP", "1-day Strength", "28-day Strength", "Slump (in)"]
        )

    def test_get_model_dict(self):
        """Test get_model_dict returns correct name-to-model mapping."""
        model_dict = self.model.get_model_dict()
        self.assertIn("GWP", model_dict)
        self.assertIn("1-day Strength", model_dict)
        self.assertEqual(len(model_dict), 3)
        self.assertIs(model_dict["GWP"], self.model.gwp_model)

    def test_fit_slump_model(self):
        """Test that fit_slump_model sets the slump_model attribute."""
        model = SustainableConcreteModel(strength_days=[1, 28])
        mock_data = MagicMock()
        n = 15
        d = self.d - 1  # slump data has no time
        mock_data.slump_data = (
            torch.rand(n, d, dtype=self.dtype) * 500 + 100,
            torch.rand(n, 1, dtype=self.dtype) * 10,
            torch.full((n, 1), 0.01, dtype=self.dtype),
            self.bounds[:, :-1],
        )
        model.fit_slump_model(mock_data)
        self.assertIsNotNone(model.slump_model)

    def test_fit_slump_model_no_slump_data_raises(self):
        """Test that fit_slump_model raises when slump data is not available."""
        model = SustainableConcreteModel(strength_days=[1, 28])
        mock_data = MagicMock()
        mock_data.slump_data = None
        with self.assertRaises(ValueError):
            model.fit_slump_model(mock_data)

    def test_get_model_list_with_slump_and_fixed_features(self):
        """Test model list wraps slump model when fixed_features has non-time entries."""
        n, d = 10, self.d - 1
        slump_model = SingleTaskGP(
            train_X=torch.rand(n, d, dtype=self.dtype),
            train_Y=torch.rand(n, 1, dtype=self.dtype),
        )
        model = SustainableConcreteModel(
            strength_days=[1, 28],
            strength_model=self.model.strength_model,
            gwp_model=self.model.gwp_model,
            slump_model=slump_model,
            d=self.model.d,
        )
        fixed = {5: 0.0}
        model_list = model.get_model_list(fixed_features=fixed)
        # GWP (wrapped) + Slump (wrapped) + 2 strength = 4
        self.assertEqual(len(model_list.models), 4)
        # Both GWP (index 0) and Slump (index 1) should be wrapped
        self.assertIsInstance(model_list.models[0], FixedFeatureModel)
        self.assertIsInstance(model_list.models[1], FixedFeatureModel)


class TestFitGP(BaseModelTest):
    """Tests for fit_gwp_gp, fit_strength_gp, and fit_slump_gp with fast optimizer."""

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

    @parameterized.expand([(False,), (True,)])
    def test_fit_slump_gp(self, use_fixed_noise):
        """Test slump GP fitting with inferred and fixed noise."""
        X = self.X[:, :-1]  # slump has no time
        Y = torch.rand(self.n, 1, dtype=self.dtype) * 10
        Yvar = torch.full((self.n, 1), 0.01, dtype=self.dtype)
        model = fit_slump_gp(
            X=X,
            Y=Y,
            Yvar=Yvar,
            use_fixed_noise=use_fixed_noise,
            optimizer_kwargs=FAST_FIT_KWARGS,
        )
        self.assertEqual(model.num_outputs, 1)
        post = model.posterior(X[:3])
        self.assertTrue(torch.all(post.variance > 0))


class TestAppendDerivedFeatures(unittest.TestCase):
    """Tests for AppendDerivedFeatures input transform."""

    def test_transform_shape(self):
        from boxcrete.models import AppendDerivedFeatures

        tf = AppendDerivedFeatures()
        X = torch.rand(10, 8, dtype=torch.float64) * 500
        X_out = tf.transform(X)
        self.assertEqual(X_out.shape, (10, 9))  # 8 + 1 appended

    def test_num_appended(self):
        from boxcrete.models import AppendDerivedFeatures

        self.assertEqual(AppendDerivedFeatures().num_appended, 1)

    def test_hrwr_binder_ratio(self):
        from boxcrete.models import AppendDerivedFeatures

        tf = AppendDerivedFeatures()
        X = torch.zeros(1, 8, dtype=torch.float64)
        X[0, 0] = 300.0  # Cement
        X[0, 1] = 100.0  # Fly Ash
        X[0, 2] = 100.0  # Slag
        X[0, 4] = 5.0  # HRWR
        X_out = tf.transform(X)
        expected_ratio = 5.0 / 500.0  # HRWR / binder
        self.assertAlmostEqual(X_out[0, -1].item(), expected_ratio, places=6)

    def test_zero_binder_clamp(self):
        from boxcrete.models import AppendDerivedFeatures

        tf = AppendDerivedFeatures()
        X = torch.zeros(1, 8, dtype=torch.float64)  # all zeros → binder=0
        X[0, 4] = 2.0  # HRWR
        X_out = tf.transform(X)
        # binder clamped to 1.0, so ratio = 2.0 / 1.0 = 2.0
        self.assertAlmostEqual(X_out[0, -1].item(), 2.0, places=6)


class TestGetStrengthGPInputTransform(BaseModelTest):
    """Tests for get_strength_gp_input_transform."""

    @parameterized.expand([(8, True), (5, True), (8, False)])
    def test_input_transform(self, d, with_bounds):
        bounds = self.bounds[:, :d] if with_bounds else None
        tf = get_strength_gp_input_transform(d=d, bounds=bounds)
        X = torch.rand(10, d, dtype=self.dtype)
        X[:, -1] = torch.randint(1, 29, (10,)).double()
        self.assertEqual(tf(X).shape, X.shape)


class TestPredictiveQualityRegression(unittest.TestCase):
    """Regression tests: verify LOO-CV R² meets expected thresholds.

    These catch regressions in model fitting that could silently degrade
    predictive quality. The thresholds are conservative lower bounds;
    actual performance should exceed them. Update thresholds only when
    intentional model/data changes justify it.
    """

    @staticmethod
    def _loo_r2(model):
        """Compute LOO R² via closed-form GP identity."""
        from linear_operator.utils.cholesky import psd_safe_cholesky

        train_X = model.train_inputs[0]
        train_Y = model.train_targets
        n = train_X.shape[-2]
        with torch.no_grad():
            prior = model.forward(train_X)
            noisy = model.likelihood(prior)
        K = noisy.lazy_covariance_matrix.to_dense()
        L = psd_safe_cholesky(K)
        res = (train_Y - prior.mean).unsqueeze(-1)
        Kinv_res = torch.cholesky_solve(res, L)
        I = torch.eye(n, dtype=L.dtype, device=L.device)
        Linv = torch.linalg.solve_triangular(L, I, upper=False)
        Kinv_diag = (Linv**2).sum(dim=-2)
        loo_var = (1.0 / Kinv_diag).unsqueeze(-1)
        loo_mean = train_Y.unsqueeze(-1) - Kinv_res * loo_var
        p, o = loo_mean.squeeze().detach(), train_Y
        return 1 - ((p - o) ** 2).sum().item() / ((o - o.mean()) ** 2).sum().item()

    def test_gwp_loo_r2(self):
        torch.manual_seed(42)
        data = load_concrete_strength(data_path=DATA_PATH)
        model = SustainableConcreteModel(strength_days=[1, 28])
        model.fit_gwp_model(data)
        r2 = self._loo_r2(model.gwp_model)
        self.assertGreater(r2, 0.99, f"GWP LOO R² = {r2:.3f}, expected > 0.99")

    def test_strength_loo_r2(self):
        torch.manual_seed(42)
        data = load_concrete_strength(data_path=DATA_PATH)
        model = SustainableConcreteModel(strength_days=[1, 28])
        model.fit_strength_model(data)
        r2 = self._loo_r2(model.strength_model)
        self.assertGreater(r2, 0.90, f"Strength LOO R² = {r2:.3f}, expected > 0.90")

    def test_slump_loo_r2(self):
        torch.manual_seed(42)
        data = load_concrete_strength(data_path=DATA_PATH, Y_columns=SLUMP_Y_COLUMNS)
        X, Y, Yvar, _ = data.slump_data
        gp = fit_slump_gp(X=X, Y=Y, Yvar=Yvar)
        r2 = self._loo_r2(gp)
        self.assertGreater(r2, 0.40, f"Slump LOO R² = {r2:.3f}, expected > 0.40")


if __name__ == "__main__":
    unittest.main()
