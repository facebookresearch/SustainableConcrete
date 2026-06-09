# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Slump GP — single-Matern + appended HRWR/binder ratio.

Hosts the public fit API for the slump model (concrete workability). The
slump GP is intentionally simpler than the V2 strength GP: it's
time-independent (slump is measured pre-cure), uses a single Matern
kernel rather than the multi-Matern + gated-time decomposition, and
appends the HRWR/binder ratio as a derived feature so the kernel can
exploit a key admixture-dosage non-linearity that stationary kernels
otherwise cannot represent.

Public API:
  * :func:`fit_slump_gp` — fit a slump GP on raw (X, Y, Yvar) data.

The HRWR/binder ratio is appended via :class:`AppendDerivedFeatures`
from :mod:`boxcrete.features` (the same transform
:mod:`boxcrete.strength_model_legacy`'s V1 input transform composes with).
"""

from __future__ import annotations

from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import ChainedInputTransform, Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.utils.constraints import LogTransformedInterval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from boxcrete.features import AppendDerivedFeatures
from boxcrete.utils import derive_bounds_from_X


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
    aug_bounds = derive_bounds_from_X(X_aug)
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


__all__ = [
    "fit_slump_gp",
]
