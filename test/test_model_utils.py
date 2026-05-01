#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for boxcrete.model_utils (LinearModel and FixedFeatureModel)."""

import unittest
from unittest.mock import MagicMock

import torch
from botorch.models import ModelList, SingleTaskGP
from boxcrete.model_utils import FixedFeatureModel, LinearModel


class TestLinearModelSimple(unittest.TestCase):
    """Tests for LinearModel in simple (non-class-indexed) mode."""

    def setUp(self):
        torch.manual_seed(42)
        self.d = 5
        self.coeffs = torch.tensor([0.12, 0.04, 0.09, 3.0, 0.02], dtype=torch.double)
        self.coeff_vars = (
            torch.tensor([0.015, 0.01, 0.015, 0.9, 0.006], dtype=torch.double) ** 2
        )
        self.model = LinearModel(self.coeffs, self.coeff_vars)
        self.X = torch.rand(10, self.d, dtype=torch.double) * 500

    def test_mean(self):
        """Posterior mean should be X @ coefficients."""
        post = self.model.posterior(self.X)
        expected = (self.X @ self.coeffs).unsqueeze(-1)
        torch.testing.assert_close(post.mean, expected)

    def test_variance(self):
        """Posterior variance should be (X**2) @ coefficient_vars."""
        post = self.model.posterior(self.X)
        expected = ((self.X**2) @ self.coeff_vars).unsqueeze(-1).clamp(min=1e-8)
        torch.testing.assert_close(post.variance, expected)

    def test_negative_coefficient_vars_raises(self):
        """Negative coefficient_vars should raise ValueError."""
        with self.assertRaises(ValueError):
            LinearModel(
                torch.ones(3, dtype=torch.double),
                torch.tensor([0.1, -0.1, 0.1], dtype=torch.double),
            )

    def test_zero_variance_clamped(self):
        """Zero coefficient vars should produce clamped positive variance."""
        model = LinearModel(self.coeffs, torch.zeros(self.d, dtype=torch.double))
        post = model.posterior(self.X)
        self.assertTrue((post.variance > 0).all())

    def test_num_outputs(self):
        self.assertEqual(self.model.num_outputs, 1)

    def test_rsample(self):
        """rsample should return correct shape."""
        post = self.model.posterior(self.X)
        samples = post.rsample(torch.Size([64]))
        self.assertEqual(samples.shape, torch.Size([64, 10, 1]))

    def test_batch_dimensions(self):
        """Should handle batched inputs (b, q, d) correctly."""
        X = torch.ones(2, 3, self.d, dtype=torch.double)
        post = self.model.posterior(X)
        self.assertEqual(post.mean.shape, torch.Size([2, 3, 1]))
        # Each point: [1,...,1] @ coeffs = sum(coeffs)
        expected_val = self.coeffs.sum().item()
        torch.testing.assert_close(
            post.mean,
            torch.full((2, 3, 1), expected_val, dtype=torch.double),
        )

    def test_in_model_list(self):
        """Should work in a ModelList."""
        gp = SingleTaskGP(
            torch.rand(5, self.d, dtype=torch.double),
            torch.rand(5, 1, dtype=torch.double),
        )
        ml = ModelList(gp, self.model)
        post = ml.posterior(self.X)
        self.assertEqual(post.mean.shape, torch.Size([10, 2]))


class TestLinearModelClassIndexed(unittest.TestCase):
    """Tests for LinearModel in class-indexed mode."""

    def test_basic_class_indexing(self):
        """Correct indexing into per-class coefficients."""
        K, d = 3, 4
        coeffs = torch.arange(K * d, dtype=torch.double).reshape(K, d)
        coeff_vars = torch.ones(K, d, dtype=torch.double) * 0.01
        model = LinearModel(coeffs, coeff_vars, class_dim=0)

        # Input: [class_idx, x1, x2, x3, x4]
        X = torch.zeros(3, 5, dtype=torch.double)
        X[0, 0] = 0  # class 0
        X[1, 0] = 1  # class 1
        X[2, 0] = 2  # class 2
        X[:, 1:] = 1.0  # all features = 1

        post = model.posterior(X)
        # For class k with all features=1: mean = sum(coeffs[k])
        expected = coeffs.sum(dim=1)
        torch.testing.assert_close(post.mean.squeeze(), expected)

    def test_class_dim_in_middle(self):
        """class_dim can be in the middle of input columns."""
        coeffs = torch.tensor([[1.0, 2.0], [10.0, 20.0]], dtype=torch.double)
        coeff_vars = torch.zeros(2, 2, dtype=torch.double)
        model = LinearModel(coeffs, coeff_vars, class_dim=1)

        # Input: [x1, class_idx, x2]
        X = torch.tensor([[5.0, 0.0, 10.0], [5.0, 1.0, 10.0]], dtype=torch.double)
        post = model.posterior(X)
        # class 0: coeffs=[1,2], features=[5,10] → 5*1 + 10*2 = 25
        # class 1: coeffs=[10,20], features=[5,10] → 5*10 + 10*20 = 250
        expected = torch.tensor([25.0, 250.0], dtype=torch.double)
        torch.testing.assert_close(post.mean.squeeze(), expected)

    def test_batch_dimensions(self):
        """Class-indexed mode with batched inputs."""
        coeffs = torch.tensor([[1.0, 2.0], [10.0, 20.0]], dtype=torch.double)
        coeff_vars = torch.zeros(2, 2, dtype=torch.double)
        model = LinearModel(coeffs, coeff_vars, class_dim=0)

        # (batch=2, q=3, d=3) where column 0 is class
        X = torch.zeros(2, 3, 3, dtype=torch.double)
        X[..., 0] = 0  # all class 0
        X[..., 1:] = 1.0  # features = [1, 1]
        post = model.posterior(X)
        self.assertEqual(post.mean.shape, torch.Size([2, 3, 1]))
        # Class 0: [1, 2] @ [1, 1] = 3
        torch.testing.assert_close(
            post.mean, torch.full((2, 3, 1), 3.0, dtype=torch.double)
        )

    def test_invalid_class_index_raises(self):
        """Out-of-bounds class index should raise an IndexError."""
        coeffs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.double)
        coeff_vars = torch.zeros(2, 2, dtype=torch.double)
        model = LinearModel(coeffs, coeff_vars, class_dim=0)

        X = torch.tensor([[5.0, 1.0, 2.0]], dtype=torch.double)  # class=5, invalid
        with self.assertRaises(IndexError):
            model.posterior(X)

    def test_with_fixed_feature_model(self):
        """Integration: class_dim LinearModel wrapped by FixedFeatureModel.

        Tests the production scenario where the class column is fixed via
        FixedFeatureModel and the LinearModel receives the reconstructed input.
        """
        # 2 classes, 3 features (after removing class column)
        coeffs = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=torch.double)
        coeff_vars = torch.zeros(2, 3, dtype=torch.double)
        # class_dim=1: second column of the full 4-column input is the class
        model = LinearModel(coeffs, coeff_vars, class_dim=1)

        # Wrap: fix column 1 (class) to value 0.0
        ffm = FixedFeatureModel(base_model=model, dim=4, indices=[1], values=[0.0])
        # Input: 3 free columns → reconstructs [x0, 0, x2, x3]
        X_free = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.double)
        post = ffm.posterior(X_free)
        # Class 0 coefficients [1, 2, 3], features [1, 1, 1] → mean = 6.0
        self.assertAlmostEqual(post.mean.item(), 6.0, places=5)

        # Now fix class to 1.0
        ffm1 = FixedFeatureModel(base_model=model, dim=4, indices=[1], values=[1.0])
        post1 = ffm1.posterior(X_free)
        # Class 1 coefficients [10, 20, 30], features [1, 1, 1] → mean = 60.0
        self.assertAlmostEqual(post1.mean.item(), 60.0, places=5)


class TestFixedFeatureModel(unittest.TestCase):
    """Tests for the FixedFeatureModel class."""

    def setUp(self):
        torch.manual_seed(42)
        self.dtype = torch.double

    def _mock_base_model(self, num_outputs=1):
        m = MagicMock()
        m.num_outputs = num_outputs
        return m

    def test_initialization_and_properties(self):
        model = FixedFeatureModel(
            base_model=self._mock_base_model(3), dim=5, indices=[4], values=[1.0]
        )
        self.assertEqual(model._dim, 5)
        self.assertEqual(model.num_outputs, 3)

    def test_initialization_mismatched_indices_values(self):
        with self.assertRaises(ValueError):
            FixedFeatureModel(
                base_model=self._mock_base_model(), dim=5, indices=[0, 1], values=[1.0]
            )

    def test_add_fixed_features_shape(self):
        model = FixedFeatureModel(
            base_model=self._mock_base_model(), dim=5, indices=[2, 4], values=[10, 20]
        )
        X = torch.rand(3, 3, dtype=self.dtype)  # 3 free features
        Z = model._add_fixed_features(X)
        self.assertEqual(Z.shape, (3, 5))  # 3 + 2 fixed = 5
        self.assertTrue(torch.all(Z[:, 2] == 10))
        self.assertTrue(torch.all(Z[:, 4] == 20))

    def test_add_fixed_features_unsorted_indices(self):
        """Indices provided in non-sorted order should still work."""
        model = FixedFeatureModel(
            base_model=self._mock_base_model(),
            dim=5,
            indices=[4, 1],
            values=[40.0, 10.0],
        )
        X = torch.rand(2, 3, dtype=self.dtype)
        Z = model._add_fixed_features(X)
        self.assertEqual(Z.shape, (2, 5))
        self.assertTrue(torch.all(Z[:, 1] == 10.0))
        self.assertTrue(torch.all(Z[:, 4] == 40.0))

    def test_posterior_delegates_to_base(self):
        """posterior should call base_model.posterior with augmented input."""
        base = self._mock_base_model()
        base.posterior = MagicMock(return_value=MagicMock())
        model = FixedFeatureModel(base_model=base, dim=4, indices=[3], values=[99.0])
        X = torch.rand(5, 3, dtype=self.dtype)
        model.posterior(X)
        base.posterior.assert_called_once()
        call_X = base.posterior.call_args[0][0]
        self.assertEqual(call_X.shape, (5, 4))
        self.assertTrue(
            torch.allclose(call_X[:, 3], torch.tensor(99.0, dtype=self.dtype))
        )

    def test_forward_delegates_to_base(self):
        """forward should call base_model.forward with augmented input."""
        base = self._mock_base_model()
        base.forward = MagicMock(return_value=MagicMock())
        model = FixedFeatureModel(base_model=base, dim=4, indices=[0], values=[5.0])
        X = torch.rand(5, 3, dtype=self.dtype)
        model.forward(X)
        base.forward.assert_called_once()
        call_X = base.forward.call_args[0][0]
        self.assertEqual(call_X.shape, (5, 4))
        self.assertTrue(
            torch.allclose(call_X[:, 0], torch.tensor(5.0, dtype=self.dtype))
        )

    def test_subset_output(self):
        """subset_output should return a new FixedFeatureModel."""
        base = self._mock_base_model()
        base.subset_output = MagicMock(return_value=MagicMock())
        model = FixedFeatureModel(base_model=base, dim=5, indices=[2], values=[7.0])
        result = model.subset_output([0])
        self.assertIsInstance(result, FixedFeatureModel)
        base.subset_output.assert_called_once_with([0])

    def test_batch_dimensions(self):
        """FixedFeatureModel should handle batched inputs."""
        coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=self.dtype)
        coeff_vars = torch.zeros(4, dtype=self.dtype)
        base = LinearModel(coeffs, coeff_vars)
        # Fix first of 4 features to 10.0; 3 free features remain
        model = FixedFeatureModel(base_model=base, dim=4, indices=[0], values=[10.0])
        X = torch.ones(2, 5, 3, dtype=self.dtype)  # batch=2, q=5, 3 free features
        post = model.posterior(X)
        self.assertEqual(post.mean.shape, torch.Size([2, 5, 1]))
        # Full input: [10, 1, 1, 1] → 10*1 + 1*2 + 1*3 + 1*4 = 19
        torch.testing.assert_close(
            post.mean, torch.full((2, 5, 1), 19.0, dtype=self.dtype)
        )


if __name__ == "__main__":
    unittest.main()
