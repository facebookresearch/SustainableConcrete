#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the V2 strength GP, slump GP, and SustainableConcreteModel."""

import unittest
from unittest.mock import MagicMock

import torch
from botorch.models import SingleTaskGP
from boxcrete import fit_strength_gp
from boxcrete.concrete_model import SustainableConcreteModel
from boxcrete.model_utils import FixedFeatureModel
from boxcrete.slump_model import fit_slump_gp
from boxcrete.strength_model_legacy import get_strength_gp_input_transform
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
        # Production V2 architecture hardcodes the 10-dim
        # composition + time layout (see DEFAULT_X_COLUMNS); use the
        # same shape here so fit_strength_gp can be exercised end-to-end.
        cls.n, cls.d = 20, 10
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

    def _mock_base_model(self, num_outputs=1):
        m = MagicMock()
        m.num_outputs = num_outputs
        return m


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
        """Slump model wrapped when fixed_features has non-time entries."""
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
        # GWP (wrapped) + 2 strength (wrapped) + Slump (wrapped) = 4
        self.assertEqual(len(model_list.models), 4)
        # GWP (index 0) and Slump (last) should be wrapped
        self.assertIsInstance(model_list.models[0], FixedFeatureModel)
        self.assertIsInstance(model_list.models[-1], FixedFeatureModel)


class TestFitGP(BaseModelTest):
    """Tests for fit_strength_gp and fit_slump_gp with fast optimizer."""

    @parameterized.expand([(False,), (True,)])
    def test_fit_strength_gp(self, with_bounds):
        # When `with_bounds=False`, exercise the public-API fallback that
        # derives bounds from X (covered separately by
        # ``test_fit_strength_gp_x_bounds_none_*``).
        bounds = self.bounds if with_bounds else None
        model = fit_strength_gp(
            X=self.X,
            Y=self.Y_strength,
            Yvar=self.Yvar,
            X_bounds=bounds,
            # V2 test affordance: cap the inner L-BFGS at 20 iters and
            # use a single restart so the per-test wall time stays
            # under ~1s. None preserves production defaults; values
            # pass through to fit_gpytorch_mll's optimizer_kwargs.
            max_optimizer_iter=20,
            num_restarts=1,
        )
        self.assertEqual(model.num_outputs, 1)
        post = model.posterior(self.X[:3])
        self.assertTrue(torch.all(post.variance > 0))

    def test_fit_strength_gp_x_bounds_none_rejects_3d_X(self):
        """The X_bounds=None public-API fallback rejects non-2D X cleanly
        instead of producing wrong-shape bounds via reduction over the
        batch dim."""
        X3d = torch.rand(2, self.n, self.X.shape[-1], dtype=self.dtype)
        with self.assertRaisesRegex(ValueError, "X.ndim"):
            fit_strength_gp(X=X3d, Y=self.Y_strength, X_bounds=None)

    def test_fit_strength_gp_x_bounds_none_rejects_empty_X(self):
        """The X_bounds=None public-API fallback rejects empty X cleanly."""
        X_empty = torch.empty(0, self.X.shape[-1], dtype=self.dtype)
        Y_empty = torch.empty(0, 1, dtype=self.dtype)
        with self.assertRaisesRegex(ValueError, "X.shape\\[0\\]"):
            fit_strength_gp(X=X_empty, Y=Y_empty, X_bounds=None)

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
        from boxcrete.features import AppendDerivedFeatures

        tf = AppendDerivedFeatures()
        X = torch.rand(10, 8, dtype=torch.float64) * 500
        X_out = tf.transform(X)
        self.assertEqual(X_out.shape, (10, 9))  # 8 + 1 appended

    def test_num_appended(self):
        from boxcrete.features import AppendDerivedFeatures

        self.assertEqual(AppendDerivedFeatures().num_appended, 1)

    def test_hrwr_binder_ratio(self):
        from boxcrete.features import AppendDerivedFeatures

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
        from boxcrete.features import AppendDerivedFeatures

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
        # Pin the float64 contract on the V1 input transform's affine
        # parameters. A silent regression to default float32 would
        # produce dtype-promotion warnings when chained with the V2
        # float64 fit pipeline and erode the precision-fix story.
        self.assertEqual(tf.tf1.coefficient.dtype, torch.float64)
        self.assertEqual(tf.tf1.offset.dtype, torch.float64)


class TestPredictiveQualityRegression(unittest.TestCase):
    """Regression tests: verify LOO-CV R² meets expected thresholds.

    These catch regressions in model fitting that could silently degrade
    predictive quality. The thresholds are conservative lower bounds;
    actual performance should exceed them. Update thresholds only when
    intentional model/data changes justify it.

    Each fit is expensive (~80s) and the same fit is needed by multiple
    tests. We fit once in ``setUpClass`` and reuse via attribute access —
    same pattern as ``TestGetModelListWithCost``. The tests are read-only
    against the fitted models, so no per-test deepcopy is needed.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch.manual_seed(42)
        cls.data = load_concrete_strength(data_path=DATA_PATH)
        cls.shared_model = SustainableConcreteModel(strength_days=[1, 28])
        cls.shared_model.fit_gwp_model(cls.data)
        cls.shared_model.fit_strength_model(cls.data)

        # Slump uses a separate dataset shape (different Y_columns).
        torch.manual_seed(42)
        cls.slump_data = load_concrete_strength(
            data_path=DATA_PATH, Y_columns=SLUMP_Y_COLUMNS
        )
        X, Y, Yvar, _ = cls.slump_data.slump_data
        cls.shared_slump_gp = fit_slump_gp(X=X, Y=Y, Yvar=Yvar)

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
        eye_n = torch.eye(n, dtype=L.dtype, device=L.device)
        Linv = torch.linalg.solve_triangular(L, eye_n, upper=False)
        Kinv_diag = (Linv**2).sum(dim=-2)
        loo_var = (1.0 / Kinv_diag).unsqueeze(-1)
        loo_mean = train_Y.unsqueeze(-1) - Kinv_res * loo_var
        p, o = loo_mean.squeeze().detach(), train_Y
        return 1 - ((p - o) ** 2).sum().item() / ((o - o.mean()) ** 2).sum().item()

    def test_gwp_calibration_r2(self):
        """GWP LinearModel should achieve near-perfect R² on training data."""
        X, Y, _, _ = self.data.gwp_data
        Y_pred = self.shared_model.gwp_model.posterior(X).mean.squeeze().detach()
        residuals = Y.squeeze() - Y_pred
        ss_res = (residuals**2).sum().item()
        ss_tot = ((Y.squeeze() - Y.squeeze().mean()) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot
        # GWP is a linear function of composition — R² should be near 1.
        # Threshold set to 0.999 to catch data integrity issues (e.g. mixes
        # with unmodeled ingredients whose GWP doesn't match composition).
        self.assertGreater(r2, 0.999, f"GWP R² = {r2:.6f}, expected > 0.999")

    def test_strength_loo_r2(self):
        r2 = self._loo_r2(self.shared_model.strength_model)
        self.assertGreater(r2, 0.90, f"Strength LOO R² = {r2:.3f}, expected > 0.90")

    def test_slump_loo_r2(self):
        r2 = self._loo_r2(self.shared_slump_gp)
        # Slump is the noisiest of our targets and the test acts as a tight
        # regression guard around the current LOO R². With seed 42 the value
        # reproduces at ≈ 0.336 (down from the pre-cleanup ≈ 0.40 because
        # removing the constant-zero `MRWR (kg/m3)` column from
        # `DEFAULT_X_COLUMNS` slightly shifts the unit-cube normalization and
        # the ARD-lengthscale optimizer's local optimum — no actual signal
        # was lost). The threshold is kept just below the current value so a
        # meaningful drop (≥ 0.006 absolute) is caught immediately. If the
        # slump model is improved in a future PR, raise this threshold along
        # with the change so it stays a tight floor.
        self.assertGreater(r2, 0.33, f"Slump LOO R² = {r2:.3f}, expected > 0.33")


class TestFitCostModel(unittest.TestCase):
    """Tests for SustainableConcreteModel.fit_cost_model."""

    def setUp(self):
        torch.manual_seed(42)
        self.data = load_concrete_strength(data_path=DATA_PATH)
        self.model = SustainableConcreteModel(strength_days=[1, 28])

    def test_fit_cost_model(self):
        """fit_cost_model should construct a LinearModel."""
        from boxcrete.model_utils import LinearModel

        result = self.model.fit_cost_model(self.data)
        self.assertIsInstance(result, LinearModel)
        self.assertIsNotNone(self.model.cost_model)

    def test_custom_coefficients(self):
        """Should accept custom coefficient dict."""
        custom = {"Cement (kg/m3)": (0.15, 0.02), "Water (kg/m3)": (0.003, 0.001)}
        self.model.fit_cost_model(self.data, cost_coefficients=custom)
        # Verify cement coefficient is negated internally for maximization
        X_columns = self.data.X_columns[:-1]
        cement_idx = X_columns.index("Cement (kg/m3)")
        self.assertAlmostEqual(
            self.model.cost_model._coefficients[cement_idx].item(), -0.15
        )

    def test_uncertainty_varies_by_composition(self):
        """HRWR-heavy compositions should have higher cost variance."""
        self.model.fit_cost_model(self.data)
        X_columns = self.data.X_columns[:-1]
        hrwr_idx = X_columns.index("HRWR (kg/m3)")
        # Create two compositions: one with high HRWR, one with low
        X_high_hrwr = torch.zeros(1, len(X_columns), dtype=torch.double)
        X_high_hrwr[0, hrwr_idx] = 10.0
        X_low_hrwr = torch.zeros(1, len(X_columns), dtype=torch.double)
        X_low_hrwr[0, hrwr_idx] = 1.0
        var_high = self.model.cost_model.posterior(X_high_hrwr).variance.item()
        var_low = self.model.cost_model.posterior(X_low_hrwr).variance.item()
        self.assertGreater(var_high, var_low)

    def test_cost_predictions_are_negative(self):
        """Cost predictions should be negative (maximization convention)."""
        self.model.fit_cost_model(self.data)
        X, _, _, _ = self.data.gwp_data
        pred = self.model.cost_model.posterior(X[:5]).mean
        self.assertTrue((pred < 0).all(), "Cost predictions should be negative")


class TestOutputIndex(unittest.TestCase):
    """Tests for SustainableConcreteModel.output_index."""

    def setUp(self):
        self.model = SustainableConcreteModel(strength_days=[1, 28])

    def test_basic_indices(self):
        """Should return correct indices for default outputs."""
        # Without fitting, model_names = ["GWP", "1-day Strength", "28-day Strength"]
        self.assertEqual(self.model.output_index("GWP"), 0)
        self.assertEqual(self.model.output_index("1-day Strength"), 1)
        self.assertEqual(self.model.output_index("28-day Strength"), 2)

    def test_with_cost(self):
        """Cost should be at end when cost_model is set."""
        from boxcrete.model_utils import LinearModel

        self.model.cost_model = LinearModel(torch.zeros(5), torch.zeros(5))
        self.assertEqual(self.model.output_index("Cost"), 3)

    def test_invalid_name_raises(self):
        """Should raise ValueError for unknown name."""
        with self.assertRaises(ValueError):
            self.model.output_index("Unknown")


class TestGetModelListWithCost(unittest.TestCase):
    """Tests for get_model_list with cost model.

    The strength GP fit is expensive (~80s per fit on a CPU runner). To
    avoid the ~10-minute total when each of the 8 tests in this class
    re-fits, we fit ONCE in ``setUpClass`` and ``copy.deepcopy`` the
    pre-fit model into ``self.model`` per test. Per-test deep-copy
    keeps tests isolated (each can mutate ``self.model`` via
    ``fit_cost_model`` etc. without polluting siblings) but skips the
    expensive GP fit. Deep-copy of the fitted SingleTaskGP is on the
    order of tens of milliseconds — negligible vs the ~80s fit.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch.manual_seed(42)
        cls.shared_data = load_concrete_strength(data_path=DATA_PATH)
        cls.shared_base_model = SustainableConcreteModel(strength_days=[1, 28])
        cls.shared_base_model.fit_gwp_model(cls.shared_data)
        cls.shared_base_model.fit_strength_model(cls.shared_data)

    def setUp(self):
        import copy

        self.data = type(self).shared_data
        self.model = copy.deepcopy(type(self).shared_base_model)

    def test_without_cost(self):
        """Default model_list has 3 outputs (GWP + 2 strength days)."""
        ml = self.model.get_model_list()
        X = self.data.gwp_data[0][:5]
        post = ml.posterior(X)
        self.assertEqual(post.mean.shape[-1], 3)

    def test_with_cost(self):
        """With cost model, model_list has 4 outputs."""
        self.model.fit_cost_model(self.data)
        ml = self.model.get_model_list()
        X = self.data.gwp_data[0][:5]
        post = ml.posterior(X)
        self.assertEqual(post.mean.shape[-1], 4)

    def test_model_names_with_cost(self):
        """model_names should include Cost after strength."""
        self.model.fit_cost_model(self.data)
        names = self.model.model_names
        self.assertEqual(names, ["GWP", "1-day Strength", "28-day Strength", "Cost"])

    def test_model_list_num_outputs_matches_names(self):
        """ModelList length must always match model_names length."""
        self.model.fit_cost_model(self.data)
        ml = self.model.get_model_list()
        self.assertEqual(len(ml.models), len(self.model.model_names))

    def test_output_index_consistent_with_model_list(self):
        """output_index values should produce correct predictions from ModelList."""
        self.model.fit_cost_model(self.data)
        ml = self.model.get_model_list()
        X = self.data.gwp_data[0][:3]
        post = ml.posterior(X)

        # GWP predictions from model_list should match direct gwp_model call
        gwp_idx = self.model.output_index("GWP")
        gwp_from_list = post.mean[:, gwp_idx]
        gwp_direct = self.model.gwp_model.posterior(X).mean.squeeze(-1)
        torch.testing.assert_close(gwp_from_list, gwp_direct)

    def test_model_list_gwp_numerical_correctness_with_fixed_features(self):
        """End-to-end test: model_list with fixed features produces correct GWP.

        Verifies the full pipeline: FixedFeatureModel reconstructs input →
        LinearModel indexes class_dim → correct per-class coefficients used.
        """
        ms_idx = self.data.X_columns.index("Material Source")
        temp_idx = self.data.X_columns.index("Temp (C)")
        fixed = {ms_idx: 0.0, temp_idx: 22.0}
        ml = self.model.get_model_list(fixed_features=fixed)

        # Create a test input in the reduced (free) space
        X_gwp, Y_gwp, _, _ = self.data.gwp_data
        # Select Source 0 mixes, remove fixed columns for optimization space
        mask = X_gwp[:, ms_idx] == 0
        keep = [i for i in range(X_gwp.shape[1]) if i not in (ms_idx, temp_idx)]
        X_free = X_gwp[mask][:5, keep]

        # Get predictions via ModelList
        post = ml.posterior(X_free)
        gwp_pred = post.mean[:, self.model.output_index("GWP")]

        # Compare with direct GWP model call (reconstruct full input manually)
        X_full = torch.zeros(5, X_gwp.shape[1], dtype=X_gwp.dtype)
        free_idx = 0
        for j in range(X_gwp.shape[1]):
            if j == ms_idx:
                X_full[:, j] = 0.0
            elif j == temp_idx:
                X_full[:, j] = 22.0
            else:
                X_full[:, j] = X_free[:, free_idx]
                free_idx += 1
        gwp_direct = self.model.gwp_model.posterior(X_full).mean.squeeze(-1)
        torch.testing.assert_close(gwp_pred, gwp_direct)

    def test_model_list_cost_numerical_correctness(self):
        """End-to-end: cost predictions via ModelList match manual computation."""
        self.model.fit_cost_model(self.data)
        ml = self.model.get_model_list()
        X_gwp, _, _, _ = self.data.gwp_data
        X_test = X_gwp[:5]

        # Get cost via ModelList
        post = ml.posterior(X_test)
        cost_idx = self.model.output_index("Cost")
        cost_from_list = post.mean[:, cost_idx]

        # Compute cost manually: -sum(coeff_i * x_i) (negated for maximization)
        from boxcrete.utils import DEFAULT_COST_COEFFICIENTS, make_linear_coefficients

        X_columns = self.data.X_columns[:-1]
        means, _ = make_linear_coefficients(X_columns, DEFAULT_COST_COEFFICIENTS)
        cost_manual = -(X_test * means).sum(-1)  # negated like the model
        torch.testing.assert_close(cost_from_list, cost_manual, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
