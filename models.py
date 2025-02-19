#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Defines concrete strength and global warming potential (GWP) models.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from botorch import fit_gpytorch_mll
from botorch.models import ModelList, ModelListGP, SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.input import (
    AffineInputTransform,
    ChainedInputTransform,
    Log10,
    Normalize,
)
from botorch.models.transforms.outcome import Standardize

from botorch.posteriors import Posterior
from botorch.utils.constraints import LogTransformedInterval
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import LinearKernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from torch import Tensor
from utils import get_day_zero_data, SustainableConcreteDataset


class SustainableConcreteModel(object):
    def __init__(
        self,
        strength_days: List[int],
        strength_model: Model | None = None,
        gwp_model: Model | None = None,
        d: int | None = None,
    ):
        """A multi-output model that jointly predicts GWP and compressive strength at
        pre-defined days `strength_days`.

        Args:
            strength_days (List[int]): A list days to predict stength for.
            strength_model (Optional[Model], optional): The strength model. Defaults to None.
            gwp_model (Optional[Model], optional): The GWP model. Defaults to None.
            d (Optional[int], optional): The dimensionality of the input to the strength model.
                Is inferred automatically if the fit functions are called. NOTE: The model
                assumes that the last element of the input corresponds to the time dimension.
        """
        self.strength_days = strength_days
        self.strength_model = None
        self.gwp_model = None
        self.d = d

    def fit_strength_model(
        self, data: SustainableConcreteDataset, use_fixed_noise: bool = False
    ) -> SingleTaskGP:
        """Fits the strength model to the given `data`. Upon completion, the model
        can be accessed with the `strength_model` attribute.

        Args:
            data: A SustainableConcreteDataset containing the strength data.
            use_fixed_noise: Toggles the use of known observation variances.

        Returns:
            The fitted strength model.
        """
        X, Y, Yvar, X_bounds = data.strength_data
        self._set_d(X.shape[-1])
        self.strength_model = fit_strength_gp(
            X=X, Y=Y, Yvar=Yvar, X_bounds=X_bounds, use_fixed_noise=use_fixed_noise
        )
        return self.strength_model

    def fit_gwp_model(
        self, data: SustainableConcreteDataset, use_fixed_noise: bool = False
    ) -> SingleTaskGP:
        """Fits the global warming potential (GWP) model to the given `data`.
        Upon completion, the model can be accessed with the `gwp_model` attribute.

        Args:
            data: A SustainableConcreteDataset containing the GWP data.
            use_fixed_noise: Toggles the use of known observation variances.

        Returns:
            The fitted GWP model.
        """
        X, Y, Yvar, X_bounds = data.gwp_data
        self._set_d(X.shape[-1] + 1)
        self.gwp_model = fit_gwp_gp(
            X=X, Y=Y, Yvar=Yvar, X_bounds=X_bounds, use_fixed_noise=use_fixed_noise
        )
        return self.gwp_model

    def _set_d(self, d: int) -> None:
        if self.d is None:
            self.d = d

    def get_model_list(self) -> ModelListGP:
        """Returns a ModelListGP modeling the GWP and compressive strength objectives as a function
        of composition only.
        Converts the strength and gwp models into a model list of independent models for gwp,
        and x-day strengths, by fixing the time input of the strength model at 1 and 28 days.
        """
        if self.d is None:
            raise ValueError("Model not fit yet.")
        models = [
            self.gwp_model,
            *(
                FixedFeatureModel(
                    base_model=self.strength_model,
                    dim=self.d,
                    indices=[self.d - 1],
                    values=[day],
                )
                for day in self.strength_days
            ),
        ]
        model = ModelList(*models)
        return model  # for use with multi-objective optimization

    def plot_strength_curve(self, composition: Tensor, max_day: int = 28) -> None:
        time = torch.arange(max_day + 1)
        composition = composition.unsqueeze(0).expand(len(time))
        # IDEA: use FixedFeatureModel?


# BatchedMultiOutputGPyTorchModel, ExactGP
class FixedFeatureModel(Model):
    # advantage: only need to implement posterior for it to work with qNEHI
    # disadvantage: makes the strength outputs independent (IDEA: could add joint model)
    # TODO: check that these are appended before the InputTransforms are applied, not after.
    def __init__(
        self,
        base_model: Model,
        dim: int,
        indices: Union[List[int], Tensor],
        values: Union[List[float], Tensor],
    ):
        """A wrapper around a GP model that fixes some inputs to specific values.

        Args:
            base_model: The base model to wrap.
            dim: The input dimensionality of the FixedFeatureModel. This is usually the
                input dimensionality of base_model minus the number of fixed features.
            indices: The indices of the inputs to fix.
            values: The values to fix the inputs to.

        Raises:
            ValueError: If indices and values do not have the same length.
        """
        super().__init__()
        self.base_model = base_model
        if len(indices) != len(values):
            raise ValueError("indices and values do not have the same length.")
        values = torch.as_tensor(values)
        self._dim = dim
        self._indices: Tensor = torch.as_tensor(indices)
        self._fixed = torch.tensor(
            [i in indices for i in torch.arange(dim, dtype=self._indices.dtype)]
        )
        self._values = values

    def _add_fixed_features(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `n x d`-dim Tensor.

        Returns:
            A `n x (d + len(self._indices))`-dim Tensor.
        """
        tkwargs = {"dtype": X.dtype, "device": X.device}
        Z = torch.zeros(*X.shape[:-1], X.shape[-1] + len(self._indices), **tkwargs)
        Z[..., self._fixed] = self._values
        Z[..., ~self._fixed] = X
        return Z

    def forward(self, X: Tensor, *args, **kwargs) -> Tensor:
        """The forward method of the FixedFeatureModel, based on the forward method of
        the base model with the fixed features added.

        Args:
            X: The `batch_shape x d`-dim input Tensor.

        Returns:
            The `batch_shape x m`-dim output Tensor.
        """
        return self.base_model.forward(self._add_fixed_features(X), *args, **kwargs)

    def posterior(self, X: Tensor, *args, **kwargs) -> Posterior:
        """Computes the posterior of the FixedFeatureModel, based on the posterior of
        the base model with the fixed features added.

        Args:
            X: The `batch_shape x d`-dim input Tensor.

        Returns:
            The posterior of the FixedFeatureModel evaluated at `X`.
        """
        return self.base_model.posterior(self._add_fixed_features(X), *args, **kwargs)

    @property
    def num_outputs(self) -> int:
        return self.base_model.num_outputs  # need to adjust if we batch fixed features

    def subset_output(self, idcs: List[int]) -> FixedFeatureModel:
        raise FixedFeatureModel(
            base_model=self.base_model.subset_output(idcs),
            dim=self._dim,
            indices=self._indices,
            values=self._value,
        )


def fit_gwp_gp(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    X_bounds: Optional[Tensor] = None,
    use_fixed_noise: bool = False,
) -> SingleTaskGP:
    """Fits a Gaussian process model to the given global warming potential (GWP) data.

    Args:
        X: `n x d`-dim Tensor of composition inputs without time.
        Y: `n x 1`-dim Tensor of GWP values.
        Yvar: `n x 1`-dim Tensor of GWP variances.

    Returns:
        A SingleTaskGP model fit to the data.
    """
    d_in = X.shape[-1]
    d_out = Y.shape[-1]
    if d_out != 1:
        raise ValueError("Output dimensions is not one in gwp fitting.")
    # GWP is a linear function of the inputs
    covar_module = LinearKernel()
    model_kwargs = {
        "train_X": X,
        "train_Y": Y,
        "covar_module": covar_module,
        "input_transform": Normalize(d_in, bounds=X_bounds),
        "outcome_transform": Standardize(d_out),
    }
    if use_fixed_noise:
        model_kwargs["train_Yvar"] = Yvar
    else:
        model_kwargs["likelihood"] = GaussianLikelihood(
            noise_constraint=LogTransformedInterval(1e-4, 1.0, initial_value=1e-2)
        )
    model = SingleTaskGP(**model_kwargs)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def fit_strength_gp(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    X_bounds: Tensor | None = None,
    use_fixed_noise: bool = False,
) -> ExactGP:
    """Fits a Gaussian process model to the given strength data.

    IDEAS:
        - Features:
            - w / b ratio
            - maturity i.e. sum_i(max(0, temperature_i) * delta_time_i)
        - Kernels:
            - Try orthogonal additive kernel again
            - temperature modeling via additive kernel?

    Args:
        X: Tensor of composition inputs including time (n x d).
        Y: Tensor of strength values (n x 1).
        Yvar: Tensor of strength variances (n x 1).

    Returns:
        A SingleTaskGP model fit to the strength data.
    """
    d_in = X.shape[-1]
    d_out = Y.shape[-1]
    if d_out != 1:
        raise ValueError("Output dimensions is not one in strength curve fitting.")

    # add data to condition GP to be zero at day zero
    X_0, Y_0, Yvar_0 = get_day_zero_data(X=X, bounds=X_bounds, n=128)
    X = torch.cat((X, X_0), dim=0)
    Y = torch.cat((Y, Y_0), dim=0)
    Yvar = torch.cat((Yvar, Yvar_0), dim=0)

    # joint kernel to model all interactions
    base_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=d_in,
        lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        lengthscale_prior=None,
    )
    scaled_base_kernel = ScaleKernel(
        base_kernel=base_kernel,
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
        outputscale_prior=None,
    )

    # additive kernel to model behavior w.r.t. time
    # try matern?
    time_kernel = RBFKernel(  # MaternKernel # smoother RBF seems to work better for additive components
        active_dims=torch.tensor([d_in - 1]),  # last dimension is time
        ard_num_dims=1,
        lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        lengthscale_prior=None,
    )
    scaled_time_kernel = ScaleKernel(
        base_kernel=time_kernel,
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
        outputscale_prior=None,
        # batch_shape=batch_shape,
    )

    # IDEA: + scaled_water_kernel and other additive components
    kernel = scaled_base_kernel + scaled_time_kernel
    model_kwargs = {
        "train_X": X,
        "train_Y": Y,
        "covar_module": kernel,
        "input_transform": get_strength_gp_input_transform(d=d_in, bounds=X_bounds),
        "outcome_transform": Standardize(d_out),
    }
    if use_fixed_noise:
        model_kwargs["train_Yvar"] = Yvar
    else:
        model_kwargs["likelihood"] = GaussianLikelihood(
            noise_constraint=LogTransformedInterval(1e-6, 1.0, initial_value=1e-1)
        )
    model = SingleTaskGP(**model_kwargs)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def get_strength_gp_input_transform(
    d: int, bounds: Optional[Tensor]
) -> ChainedInputTransform:
    """Chains a log(time + 1) and Normalize transform on d dimensional input data,
    with the provided bounds.

    Args:
        bounds: `2 x d` tensor of lower and upper bounds for each dimension.

    Returns:
        A ChainedInputTransform that log-transforms the time dimension and subsequently
        normalizes all dimensions to the unit hyper-cube.
    """
    time_index = [d - 1]
    tf1 = AffineInputTransform(  # adds one to time dimension before taking log
        d,
        coefficient=torch.ones(1),
        offset=torch.ones(1),
        indices=time_index,
        reverse=True,
    )
    tf2 = Log10(
        indices=time_index
    )  # taking log of time dimension for better extrapolation
    if bounds is not None:
        transformed_bounds = tf2(tf1(bounds))
        tf3 = Normalize(
            d, bounds=transformed_bounds
        )  # normalizing after log(t + 1) transform
    else:
        tf3 = Normalize(d)  # normalizing after log(t + 1) transform
    return ChainedInputTransform(tf1=tf1, tf2=tf2, tf3=tf3)
