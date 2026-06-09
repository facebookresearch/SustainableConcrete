#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for PartialFixedNoiseLikelihood."""

import unittest

import torch
from botorch.models import SingleTaskGP
from boxcrete.likelihoods import PartialFixedNoiseLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from linear_operator.operators import DiagLinearOperator


class TestPartialFixedNoiseLikelihood(unittest.TestCase):
    """Self-contained tests for the PartialFixedNoiseLikelihood class."""

    def setUp(self):
        torch.manual_seed(0)
        self.n_real = 10
        self.n_pseudo = 5
        self.pseudo_noise = 1e-6
        self.likelihood = PartialFixedNoiseLikelihood(
            n_real=self.n_real,
            n_pseudo=self.n_pseudo,
            pseudo_noise=self.pseudo_noise,
        )

    def test_properties(self):
        """Test that properties return correct values."""
        self.assertEqual(self.likelihood.n_real, self.n_real)
        self.assertEqual(self.likelihood.n_pseudo, self.n_pseudo)
        self.assertEqual(self.likelihood.pseudo_noise, self.pseudo_noise)

    def test_noise_diagonal_training(self):
        """Test that training-size input produces mixed noise diagonal."""
        n_total = self.n_real + self.n_pseudo
        base_shape = torch.Size([n_total])
        noise_covar = self.likelihood._shaped_noise_covar(base_shape)

        # Should return a DiagLinearOperator
        self.assertIsInstance(noise_covar, DiagLinearOperator)

        diag = noise_covar.diagonal()
        self.assertEqual(diag.shape[0], n_total)

        # Real observations should have the learned noise
        learned_noise = self.likelihood.noise.item()
        for i in range(self.n_real):
            self.assertAlmostEqual(
                diag[i].item(),
                learned_noise,
                places=8,
                msg=f"Real obs {i} should have learned noise",
            )

        # Pseudo-observations should have fixed pseudo_noise
        for i in range(self.n_real, n_total):
            self.assertAlmostEqual(
                diag[i].item(),
                self.pseudo_noise,
                places=10,
                msg=f"Pseudo obs {i} should have pseudo_noise",
            )

    def test_noise_diagonal_prediction(self):
        """Test that non-training-size input uses standard homoscedastic noise."""
        # During prediction, n != n_real + n_pseudo
        n_test = 7
        base_shape = torch.Size([n_test])
        noise_covar = self.likelihood._shaped_noise_covar(base_shape)

        diag = noise_covar.diagonal()
        learned_noise = self.likelihood.noise.item()
        for i in range(n_test):
            self.assertAlmostEqual(diag[i].item(), learned_noise, places=8)

    def test_gradient_flows_through_learned_noise(self):
        """Test that the learned noise parameter receives gradients."""
        n_total = self.n_real + self.n_pseudo
        base_shape = torch.Size([n_total])

        # Ensure noise parameter requires grad
        self.assertTrue(self.likelihood.noise_covar.raw_noise.requires_grad)

        noise_covar = self.likelihood._shaped_noise_covar(base_shape)
        diag = noise_covar.diagonal()

        # Backprop through the real-obs part of the diagonal
        loss = diag[: self.n_real].sum()
        loss.backward()

        # The raw_noise parameter should have received a gradient
        self.assertIsNotNone(self.likelihood.noise_covar.raw_noise.grad)
        self.assertNotEqual(self.likelihood.noise_covar.raw_noise.grad.item(), 0.0)

    def test_pseudo_noise_has_no_gradient(self):
        """Test that pseudo-obs noise does not depend on learned parameters."""
        n_total = self.n_real + self.n_pseudo
        base_shape = torch.Size([n_total])

        self.likelihood.zero_grad()
        noise_covar = self.likelihood._shaped_noise_covar(base_shape)
        diag = noise_covar.diagonal()

        # Backprop through the pseudo-obs part only
        loss = diag[self.n_real :].sum()
        loss.backward()

        # The raw_noise gradient should be zero (pseudo part is a constant)
        grad = self.likelihood.noise_covar.raw_noise.grad
        self.assertIsNotNone(grad)
        self.assertAlmostEqual(grad.item(), 0.0, places=10)

    def test_integration_with_single_task_gp(self):
        """Test that the likelihood integrates correctly with SingleTaskGP."""
        torch.manual_seed(42)
        d = 3
        n_real = 15
        n_pseudo = 5

        # Real observations
        X_real = torch.rand(n_real, d, dtype=torch.double)
        Y_real = torch.sin(X_real.sum(dim=-1, keepdim=True))

        # Pseudo-observations (y=0 at specific locations)
        X_pseudo = torch.rand(n_pseudo, d, dtype=torch.double)
        Y_pseudo = torch.zeros(n_pseudo, 1, dtype=torch.double)

        X = torch.cat([X_real, X_pseudo])
        Y = torch.cat([Y_real, Y_pseudo])

        likelihood = PartialFixedNoiseLikelihood(
            n_real=n_real,
            n_pseudo=n_pseudo,
            pseudo_noise=1e-6,
        )
        model = SingleTaskGP(train_X=X, train_Y=Y, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Ensure forward pass works
        model.train()
        output = model(X)
        loss = -mll(output, model.train_targets)
        self.assertTrue(torch.isfinite(loss))

        # Ensure backward pass works
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, msg=f"No gradient for {name}")

    def test_posterior_at_pseudo_locations(self):
        """Test that predictions at pseudo-observation locations are near zero."""
        torch.manual_seed(42)
        d = 3
        n_real = 30
        n_pseudo = 10

        # Real observations with positive values
        X_real = torch.rand(n_real, d, dtype=torch.double)
        Y_real = torch.rand(n_real, 1, dtype=torch.double) * 5 + 2

        # Pseudo-observations (y=0)
        X_pseudo = torch.rand(n_pseudo, d, dtype=torch.double)
        Y_pseudo = torch.zeros(n_pseudo, 1, dtype=torch.double)

        X = torch.cat([X_real, X_pseudo])
        Y = torch.cat([Y_real, Y_pseudo])

        likelihood = PartialFixedNoiseLikelihood(
            n_real=n_real,
            n_pseudo=n_pseudo,
            pseudo_noise=1e-6,
        )
        model = SingleTaskGP(train_X=X, train_Y=Y, likelihood=likelihood)

        # Train using BoTorch's fit_gpytorch_mll (handles constraints properly)
        from botorch import fit_gpytorch_mll

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 50}})

        # Predict at pseudo-observation locations
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X_pseudo)
            pred_mean = posterior.mean.squeeze()

        # Predictions at pseudo locations should be near zero
        # (with tight pseudo_noise, the GP is forced through these points)
        self.assertTrue(
            pred_mean.abs().max().item() < 0.5,
            msg=f"Predictions at pseudo locations should be near zero, "
            f"got max abs = {pred_mean.abs().max().item():.4f}",
        )


if __name__ == "__main__":
    unittest.main()
