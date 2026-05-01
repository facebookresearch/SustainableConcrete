#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Generic BoTorch-compatible model wrappers.

This module provides lightweight Model subclasses that integrate with BoTorch's
ModelList, acquisition functions, and optimize_acqf without requiring GP training:

- ``LinearModel``: A linear model f(x) = c^T x with known coefficient uncertainty
  and optional per-class coefficient indexing.
- ``FixedFeatureModel``: Wraps any model to fix a subset of inputs to constants.
"""

from __future__ import annotations

import torch
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior, Posterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator
from torch import Tensor


class LinearModel(Model):
    """A linear model f(x) = c^T x with known coefficient uncertainty.

    Supports two modes:

    1. **Simple** (``class_dim=None``): A single set of coefficients applied to
       all inputs. ``coefficients`` and ``coefficient_vars`` are ``(d,)``-dim.

    2. **Class-indexed** (``class_dim`` specified): Per-class coefficients stored
       as a ``(K, d)`` tensor. The input column at ``class_dim`` is used as an
       integer index to select the appropriate coefficient row. The class column
       is excluded from the dot product. This generalizes naturally to any
       number of classes K.

    No training data or fitting required. This is equivalent to Bayesian
    linear regression with a diagonal prior on the coefficients and no
    observations — the "posterior" is the prior itself.

    This model integrates seamlessly with BoTorch's ``ModelList``,
    ``FixedFeatureModel``, acquisition functions, and ``optimize_acqf``.
    """

    def __init__(
        self,
        coefficients: Tensor,
        coefficient_vars: Tensor,
        class_dim: int | None = None,
    ):
        """Constructs a LinearModel with known coefficients and their variances.

        Args:
            coefficients: Coefficient means. Shape ``(d,)`` for simple mode or
                ``(K, d)`` for class-indexed mode (K classes, d features).
            coefficient_vars: Coefficient variances. Same shape as
                ``coefficients``.
            class_dim: If specified, the input column index that contains the
                integer class label. This column is used for indexing into
                the coefficient tensor and excluded from the linear computation.
        """
        super().__init__()
        if (coefficient_vars < 0).any():
            raise ValueError("coefficient_vars must be non-negative.")
        self.register_buffer("_coefficients", coefficients)
        self.register_buffer("_coefficient_vars", coefficient_vars)
        self._class_dim = class_dim
        self._num_outputs = 1

    @property
    def num_outputs(self) -> int:
        """The number of outputs (always 1)."""
        return self._num_outputs

    def posterior(
        self, X: Tensor, observation_noise: bool = False, **kwargs
    ) -> GPyTorchPosterior:
        """Computes the posterior at input X.

        Args:
            X: Input Tensor. For simple mode: ``(... x d)``-dim.
                For class-indexed mode: ``(... x (d + 1))``-dim where column
                ``class_dim`` contains integer class indices.
            observation_noise: Ignored (included for API compatibility).

        Returns:
            A ``GPyTorchPosterior`` with the computed mean and variance.
        """
        if self._class_dim is not None:
            # Extract class indices and remove class column from features
            class_indices = X[..., self._class_dim].long()
            feature_dims = [i for i in range(X.shape[-1]) if i != self._class_dim]
            X_features = X[..., feature_dims]
            # Index into per-class coefficients (cast to input dtype)
            coeffs = self._coefficients[class_indices].to(X.dtype)
            coeff_vars = self._coefficient_vars[class_indices].to(X.dtype)
            mean = (X_features * coeffs).sum(-1)
            var = (X_features**2 * coeff_vars).sum(-1)
        else:
            coefficients = self._coefficients.to(X.dtype)
            coefficient_vars = self._coefficient_vars.to(X.dtype)
            mean = X @ coefficients
            var = (X**2) @ coefficient_vars

        mvn = MultivariateNormal(mean, DiagLinearOperator(var))
        return GPyTorchPosterior(mvn)


class FixedFeatureModel(Model):
    """Wraps a model to fix a subset of inputs to constant values.

    At evaluation time the fixed features are spliced back into the input
    tensor before delegating to the ``base_model``. This is useful for
    optimization over a subset of input dimensions while keeping others
    constant (e.g., fixing Material Source or Temperature).
    """

    def __init__(
        self,
        base_model: Model,
        dim: int,
        indices: list[int] | Tensor,
        values: list[float] | Tensor,
    ):
        """A wrapper around a model that fixes some inputs to specific values.

        Args:
            base_model: The base model to wrap.
            dim: The total input dimensionality expected by the base model.
                The FixedFeatureModel accepts ``(dim - len(indices))``-dimensional
                inputs and reconstructs the full ``dim``-dimensional input.
            indices: The indices of the inputs to fix.
            values: The values to fix the inputs to.

        Raises:
            ValueError: If indices and values do not have the same length.
        """
        super().__init__()
        self.base_model = base_model
        if len(indices) != len(values):
            raise ValueError("indices and values do not have the same length.")
        # Sort by index so that the boolean mask assignment in
        # _add_fixed_features places values at the correct positions.
        indices = torch.as_tensor(indices)
        values = torch.as_tensor(values)
        sort_order = indices.argsort()
        indices = indices[sort_order]
        values = values[sort_order]
        self._dim = dim
        self.register_buffer("_indices", indices)
        self.register_buffer(
            "_fixed",
            torch.tensor(
                [i in indices for i in torch.arange(dim, dtype=indices.dtype)]
            ),
        )
        self.register_buffer("_values", values)

    def _add_fixed_features(self, X: Tensor) -> Tensor:
        """Reconstructs the full input tensor by splicing in fixed values.

        Args:
            X: A ``(... x d_free)``-dim Tensor of free features.

        Returns:
            A ``(... x d_full)``-dim Tensor with fixed features inserted.
        """
        tkwargs = {"dtype": X.dtype, "device": X.device}
        Z = torch.zeros(*X.shape[:-1], X.shape[-1] + len(self._indices), **tkwargs)
        Z[..., self._fixed] = self._values.to(X.dtype)
        Z[..., ~self._fixed] = X
        return Z

    def forward(self, X: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass with fixed features added.

        Args:
            X: The ``(batch_shape x d_free)``-dim input Tensor.

        Returns:
            The model output Tensor.
        """
        return self.base_model.forward(self._add_fixed_features(X), *args, **kwargs)

    def posterior(self, X: Tensor, *args, **kwargs) -> Posterior:
        """Computes the posterior with fixed features added.

        Args:
            X: The ``(batch_shape x d_free)``-dim input Tensor.

        Returns:
            The posterior of the base model evaluated at the augmented input.
        """
        return self.base_model.posterior(self._add_fixed_features(X), *args, **kwargs)

    @property
    def num_outputs(self) -> int:
        """The number of outputs of the base model."""
        return self.base_model.num_outputs

    def subset_output(self, idcs: list[int]) -> FixedFeatureModel:
        """Returns a new ``FixedFeatureModel`` whose base model is subset to
        the given output indices.

        Args:
            idcs: Output indices to keep.

        Returns:
            A ``FixedFeatureModel`` wrapping the subset base model.
        """
        return FixedFeatureModel(
            base_model=self.base_model.subset_output(idcs),
            dim=self._dim,
            indices=self._indices,
            values=self._values,
        )
