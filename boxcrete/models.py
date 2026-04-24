#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Defines concrete strength, slump, and global warming potential (GWP) models.
"""

from __future__ import annotations

import torch
from botorch import fit_gpytorch_mll
from botorch.models import ModelList, SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.input import (
    AffineInputTransform,
    ChainedInputTransform,
    InputTransform,
    Log10,
    Normalize,
)
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import Posterior
from botorch.utils.constraints import LogTransformedInterval
from boxcrete.utils import get_day_zero_data, SustainableConcreteDataset
from gpytorch.kernels import LinearKernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

# Indices into DEFAULT_X_COLUMNS (without Time) for derived feature computation
_CEMENT_IDX = 0
_FLY_ASH_IDX = 1
_SLAG_IDX = 2
_HRWR_IDX = 4


class AppendDerivedFeatures(InputTransform, torch.nn.Module):
    """Input transform that appends the HRWR-to-binder ratio.

    The HRWR/binder ratio encodes the admixture dosage relative to total
    binder content — a key determinant of concrete workability (slump)
    that stationary GP kernels cannot learn from raw composition values.
    """

    is_one_to_many = False

    def __init__(
        self,
        cement_idx: int = _CEMENT_IDX,
        fly_ash_idx: int = _FLY_ASH_IDX,
        slag_idx: int = _SLAG_IDX,
        hrwr_idx: int = _HRWR_IDX,
    ):
        super().__init__()
        self.cement_idx = cement_idx
        self.fly_ash_idx = fly_ash_idx
        self.slag_idx = slag_idx
        self.hrwr_idx = hrwr_idx
        self.transform_on_train = True
        self.transform_on_eval = True
        self.transform_on_fantasize = True

    def transform(self, X: Tensor) -> Tensor:
        binder = (
            X[..., self.cement_idx : self.cement_idx + 1]
            + X[..., self.fly_ash_idx : self.fly_ash_idx + 1]
            + X[..., self.slag_idx : self.slag_idx + 1]
        ).clamp(min=1.0)
        hrwr_b = X[..., self.hrwr_idx : self.hrwr_idx + 1] / binder
        return torch.cat([X, hrwr_b], dim=-1)

    @property
    def num_appended(self) -> int:
        """Number of features appended by this transform."""
        return 1


class SustainableConcreteModel:
    """Multi-output model that jointly predicts GWP, slump, and compressive strength.

    The model consists of a GWP model and an optional slump model (both independent
    of curing time) and a strength model (dependent on composition *and* time).
    At optimisation time the strength model is sliced at each of the
    ``strength_days`` via ``FixedFeatureModel`` to produce a ``ModelList`` that
    maps composition only to ``[GWP, (Slump), 1-day strength, 28-day strength, ...]``.
    """

    def __init__(
        self,
        strength_days: list[int],
        strength_model: Model | None = None,
        gwp_model: Model | None = None,
        slump_model: Model | None = None,
        d: int | None = None,
    ):
        """A multi-output model that jointly predicts GWP, slump, and compressive
        strength at pre-defined days `strength_days`.

        Args:
            strength_days: A list of days to predict strength for.
            strength_model: The strength model. Defaults to None.
            gwp_model: The GWP model. Defaults to None.
            slump_model: The slump model. Defaults to None.
            d: The dimensionality of the input to the strength model.
                Is inferred automatically if the fit functions are called. NOTE: The model
                assumes that the last element of the input corresponds to the time dimension.
        """
        self.strength_days = strength_days
        self.strength_model = strength_model
        self.gwp_model = gwp_model
        self.slump_model = slump_model
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

    def fit_slump_model(
        self, data: SustainableConcreteDataset, use_fixed_noise: bool = False
    ) -> SingleTaskGP:
        """Fits the slump model to the given `data`.
        Upon completion, the model can be accessed with the `slump_model` attribute.

        Args:
            data: A SustainableConcreteDataset containing slump data.
            use_fixed_noise: Toggles the use of known observation variances.

        Returns:
            The fitted slump model.

        Raises:
            ValueError: If slump data is not available in the dataset.
        """
        slump_data = data.slump_data
        if slump_data is None:
            raise ValueError(
                "Slump data not available. Ensure 'Slump (in)' is in Y_columns."
            )
        X, Y, Yvar, _ = slump_data
        self._set_d(X.shape[-1] + 1)
        self.slump_model = fit_slump_gp(
            X=X, Y=Y, Yvar=Yvar, use_fixed_noise=use_fixed_noise
        )
        return self.slump_model

    def _set_d(self, d: int) -> None:
        if self.d is None:
            self.d = d

    def get_model_list(
        self, fixed_features: dict[int, float] | None = None
    ) -> ModelList:
        """Returns a ``ModelList`` modelling GWP, optional slump, and compressive
        strength as a function of composition only.

        Converts the strength, GWP, and optional slump models into a model list
        of independent models by fixing the time input of the strength model at
        each ``strength_day``.

        Args:
            fixed_features: Optional mapping from input column **index** to a
                fixed value.  When provided these features are fixed *in
                addition to* the Time dimension for the strength models, and
                the non-Time entries are also applied to the GWP and slump
                models via ``FixedFeatureModel``.  Useful for fixing e.g.
                ``Coarse Aggregates = 0`` in mortar mode.

        Returns:
            A ``ModelList`` with ``1 + len(strength_days) + (1 if slump)``
            sub-models:

            - Index 0: GWP model (composition → GWP)
            - Indices 1..n: strength at each ``strength_day``
            - Last (if fitted): slump model (composition → Slump)

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self.d is None or self.strength_model is None or self.gwp_model is None:
            raise ValueError(
                "Model not fit yet. Call fit_gwp_model() and fit_strength_model() first."
            )

        time_idx = self.d - 1  # last column is Time

        # Helper to optionally wrap a time-independent model with FixedFeatureModel
        def _maybe_wrap(base_model: Model) -> Model:
            if fixed_features is None:
                return base_model
            non_time = {k: v for k, v in fixed_features.items() if k != time_idx}
            if not non_time:
                return base_model
            ff_indices = sorted(non_time.keys())
            ff_values = [non_time[i] for i in ff_indices]
            return FixedFeatureModel(
                base_model=base_model,
                dim=self.d - 1,  # time-independent models have no Time
                indices=ff_indices,
                values=ff_values,
            )

        models: list[Model] = [_maybe_wrap(self.gwp_model)]

        for day in self.strength_days:
            indices = [time_idx]
            values: list[float] = [float(day)]
            if fixed_features is not None:
                for idx, val in sorted(fixed_features.items()):
                    if idx != time_idx:
                        indices.append(idx)
                        values.append(val)
            models.append(
                FixedFeatureModel(
                    base_model=self.strength_model,
                    dim=self.d,
                    indices=indices,
                    values=values,
                )
            )

        if self.slump_model is not None:
            models.append(_maybe_wrap(self.slump_model))

        return ModelList(*models)

    @property
    def model_names(self) -> list[str]:
        """Ordered names of outputs in the ``ModelList`` from ``get_model_list``.

        Returns:
            A list like ``["GWP", "Slump (in)", "1-day Strength", "28-day Strength"]``.
        """
        names = ["GWP"]
        for day in self.strength_days:
            names.append(f"{day}-day Strength")
        if self.slump_model is not None:
            names.append("Slump (in)")
        return names

    def get_model_dict(
        self, fixed_features: dict[int, float] | None = None
    ) -> dict[str, Model]:
        """Returns a name-to-model dictionary for the multi-output model.

        Equivalent to ``dict(zip(model.model_names, model.get_model_list(...).models))``.

        Args:
            fixed_features: Same as ``get_model_list``.

        Returns:
            A dictionary mapping output names to sub-models.
        """
        model_list = self.get_model_list(fixed_features=fixed_features)
        return dict(zip(self.model_names, model_list.models))


class FixedFeatureModel(Model):
    """Wraps a GP model to fix a subset of inputs to constant values.

    At evaluation time the fixed features are spliced back into the input
    tensor before delegating to the ``base_model``.
    """

    def __init__(
        self,
        base_model: Model,
        dim: int,
        indices: list[int] | Tensor,
        values: list[float] | Tensor,
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
        # Sort by index so that the boolean mask assignment in
        # _add_fixed_features places values at the correct positions.
        indices = torch.as_tensor(indices)
        values = torch.as_tensor(values)
        sort_order = indices.argsort()
        indices = indices[sort_order]
        values = values[sort_order]
        self._dim = dim
        self._indices: Tensor = indices
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
        Z[..., self._fixed] = self._values.to(X.dtype)
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


def fit_gwp_gp(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    X_bounds: Tensor | None = None,
    use_fixed_noise: bool = False,
    optimizer_kwargs: dict | None = None,
) -> SingleTaskGP:
    """Fits a Gaussian process model to the given global warming potential (GWP) data.

    Args:
        X: `n x d`-dim Tensor of composition inputs without time.
        Y: `n x 1`-dim Tensor of GWP values.
        Yvar: `n x 1`-dim Tensor of GWP variances.
        X_bounds: Optional `2 x d`-dim bounds Tensor.
        use_fixed_noise: Whether to use fixed observation noise.
        optimizer_kwargs: Optional keyword arguments for the optimizer.

    Returns:
        A SingleTaskGP model fit to the data.
    """
    d_out = Y.shape[-1]
    if d_out != 1:
        raise ValueError("Output dimensions is not one in gwp fitting.")
    # GWP is a linear function of the inputs
    covar_module = LinearKernel()
    # removing any input and outcome transforms, as well as the prior mean from
    # the model to force it to be homogeneous, i.e. it has no offset.
    model_kwargs = {
        "train_X": X,
        "train_Y": Y,
        "mean_module": ZeroMean(),
        "covar_module": covar_module,
        "input_transform": None,
        "outcome_transform": None,
    }
    if use_fixed_noise:
        model_kwargs["train_Yvar"] = Yvar
    else:
        model_kwargs["likelihood"] = GaussianLikelihood(  # pyre-ignore
            noise_constraint=LogTransformedInterval(1e-4, 1.0, initial_value=1e-2)
        )
    model = SingleTaskGP(**model_kwargs)  # pyre-ignore
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, optimizer_kwargs=optimizer_kwargs)
    return model


def fit_strength_gp(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    X_bounds: Tensor | None = None,
    use_fixed_noise: bool = False,
    optimizer_kwargs: dict | None = None,
) -> SingleTaskGP:
    """Fits a Gaussian process model to the given strength data.

    Args:
        X: Tensor of composition inputs including time (n x d).
        Y: Tensor of strength values (n x 1).
        Yvar: Tensor of strength variances (n x 1).
        X_bounds: Optional `2 x d`-dim bounds Tensor.
        use_fixed_noise: Whether to use fixed observation noise.
        optimizer_kwargs: Optional keyword arguments for the optimizer.

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
    time_kernel = RBFKernel(
        active_dims=torch.tensor([d_in - 1]),  # last dimension is time
        ard_num_dims=1,
        lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        lengthscale_prior=None,
    )
    scaled_time_kernel = ScaleKernel(
        base_kernel=time_kernel,
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
        outputscale_prior=None,
    )

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
    fit_gpytorch_mll(mll, optimizer_kwargs=optimizer_kwargs)
    return model


def get_strength_gp_input_transform(
    d: int, bounds: Tensor | None
) -> ChainedInputTransform:
    """Chains a log(time + 1) and Normalize transform on d dimensional input data,
    with the provided bounds.

    Args:
        d: The input dimensionality.
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


def fit_slump_gp(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    use_fixed_noise: bool = False,
    optimizer_kwargs: dict | None = None,
) -> SingleTaskGP:
    """Fits a GP model to slump data with derived composition features.

    Automatically appends the HRWR/binder ratio via ``AppendDerivedFeatures``
    before fitting a ``SingleTaskGP``.

    Args:
        X: ``n x d``-dim Tensor of composition inputs (without time).
        Y: ``n x 1``-dim Tensor of slump values.
        Yvar: ``n x 1``-dim Tensor of slump variances.
        use_fixed_noise: Whether to use fixed observation noise.
        optimizer_kwargs: Optional keyword arguments for the optimizer.

    Returns:
        A fitted ``SingleTaskGP`` model.
    """
    d_in = X.shape[-1]
    derive = AppendDerivedFeatures()
    d_aug = d_in + derive.num_appended

    if optimizer_kwargs is None:
        optimizer_kwargs = {"options": {"maxiter": 1024}}

    # Chain: append derived features → normalize to unit cube
    X_aug = derive.transform(X)
    aug_min = X_aug.amin(dim=0)
    aug_max = X_aug.amax(dim=0)
    # Avoid zero-width bounds (causes NaN in normalization)
    zero_width = aug_max - aug_min < 1e-8
    aug_max[zero_width] = aug_min[zero_width] + 1.0
    aug_bounds = torch.stack([aug_min, aug_max])
    input_tf = ChainedInputTransform(
        derive=derive,
        normalize=Normalize(d=d_aug, bounds=aug_bounds),
    )

    model_kwargs: dict = {
        "train_X": X,
        "train_Y": Y,
        "input_transform": input_tf,
        "outcome_transform": Standardize(1),
    }
    if use_fixed_noise:
        model_kwargs["train_Yvar"] = Yvar
    else:
        # Constrain noise variance to [1e-4, 1e1]. The lower bound of 1e-4
        # (noise std ~1% of standardized data) prevents numerical issues
        # while allowing the optimizer to find the right noise level.
        model_kwargs["likelihood"] = GaussianLikelihood(
            noise_constraint=LogTransformedInterval(1e-4, 1.0, initial_value=1e-2)
        )

    model = SingleTaskGP(**model_kwargs)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, optimizer_kwargs=optimizer_kwargs)
    return model
