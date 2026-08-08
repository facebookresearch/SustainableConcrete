# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Legacy V1 strength GP — research artifact, NOT the deployed model.

The V1 architecture (single Matern + within-group prior + RBF on time
+ day-zero anchor pseudo-observations + scalar learnable Gaussian
noise) was the production OSS model before V2 was deployed on
2026-05-17. It is preserved here for:

1. The improvement-journey plot in ``experiments/plot_improvement_journey.py``
   which shows the V1→V2 progression.
2. Regression tests in ``experiments/test_legacy_v1_lengthscales.py`` that
   verify the within-group prior properties on the V1 baseline.
3. Any explicit caller that wants to A/B compare the legacy model.

For the production V2 fit, use ``boxcrete.fit_strength_gp``.
"""

from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.utils.constraints import LogTransformedInterval
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from boxcrete.likelihoods import PartialFixedNoiseLikelihood
from boxcrete.priors import _default_lengthscale_prior
from boxcrete.strength_model_legacy import get_strength_gp_input_transform
from boxcrete.utils import get_day_zero_data


def fit_strength_gp_v1(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    X_bounds: Tensor | None = None,
    use_fixed_noise: bool = False,
    optimizer_kwargs: dict | None = None,
    lengthscale_prior: object | None = "default",
) -> SingleTaskGP:
    """Fit the legacy V1 strength GP (research baseline only).

    Single Matern + RBF-time additive kernel + day-zero anchor pseudo-
    observations + scalar learnable Gaussian noise +
    ``WithinGroupShrinkagePrior`` on lengthscales.

    Args:
        X: ``[n, 10]`` raw input — composition (9 dims) + time (1 dim).
        Y: ``[n, 1]`` strength values in psi.
        Yvar: ``[n, 1]`` per-row strength variances (psi²).
        X_bounds: ``[2, 10]`` optional lower/upper bounds.
        use_fixed_noise: Whether to fix observation noise at ``Yvar``.
        optimizer_kwargs: Keyword arguments forwarded to ``fit_gpytorch_mll``.
        lengthscale_prior: Prior on the Matern lengthscales.
            ``"default"`` → ``WithinGroupShrinkagePrior`` (the V1 production
            default); ``None`` → no prior; or pass a ``gpytorch.priors.Prior``.

    Returns:
        A ``SingleTaskGP`` with the V1 architecture installed.
    """
    d_in = X.shape[-1]
    d_out = Y.shape[-1]
    if d_out != 1:
        raise ValueError("Output dimensions is not one in strength curve fitting.")

    if lengthscale_prior == "default":
        lengthscale_prior = _default_lengthscale_prior(d_in)

    # add data to condition GP to be zero at day zero
    X_0, Y_0, Yvar_0 = get_day_zero_data(X=X, n=128)
    n_real = X.shape[0]
    n_pseudo = X_0.shape[0]
    X = torch.cat((X, X_0), dim=0)
    Y = torch.cat((Y, Y_0), dim=0)
    Yvar = torch.cat((Yvar, Yvar_0), dim=0)

    # joint kernel to model all interactions
    base_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=d_in,
        lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        lengthscale_prior=lengthscale_prior,
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
        model_kwargs["likelihood"] = PartialFixedNoiseLikelihood(
            n_real=n_real,
            n_pseudo=n_pseudo,
            pseudo_noise=1e-6,
            noise_constraint=LogTransformedInterval(1e-6, 1.0, initial_value=1e-1),
        )
    model = SingleTaskGP(**model_kwargs)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, optimizer_kwargs=optimizer_kwargs)
    return model


__all__ = ["fit_strength_gp_v1"]
