# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Block-leave-one-out (block-LOO) training objective for the strength GP.

The deployed strength model is trained by minimising the **block-LOO
predictive negative-log-likelihood** rather than the marginal likelihood
(MLL). Each block is one unique composition (rows sharing the leading
composition dims), so block-LOO measures how well the model predicts a
*held-out composition's* strength curve — the quantity the explorer and
Bayesian-optimisation loop actually rely on. Training directly on this
objective (from initialisation, no MLL warm-up) gave the best and most
stable held-out RMSE in the design-exploration benchmark.

Closed form (Sundararajan-Keerthi block-inverse identity): with
``K = K_kernel + sigma^2 I``, ``alpha = K^{-1}(y - mu_prior)`` and, for a
block ``B``,

    residual_B = (K^{-1}_BB)^{-1} @ alpha_B
    var_B      = diag((K^{-1}_BB)^{-1})     # includes aleatoric noise

so the per-row loss is the Gaussian NLL ``0.5*(log 2pi + log var + r^2/var)``.
Day-zero anchor pseudo-rows (indices ``[n_real:]``) join their
composition's block — holding a composition out also holds out its anchor
(no leak) — but are not scored.

The deployed model is trained on a **single combined objective** that
blends this block-LOO loss with a fraction of the marginal likelihood
(see :func:`train_block_loo`): the MLL term supplies the ``-0.5 log|K|``
complexity penalty that block-LOO structurally lacks, keeping the strength
curves smooth/monotone between the training ages at negligible block-LOO
cost.
"""

from __future__ import annotations

import math

import torch
from botorch.models import SingleTaskGP
from gpytorch.utils.cholesky import psd_safe_cholesky

__all__ = ["block_loo_loss", "train_block_loo"]


def _model_prior_log_prob(model: SingleTaskGP) -> torch.Tensor:
    """Sum of ``log p(theta)`` over every prior registered on the model.

    Including this term makes block-LOO training maximise the posterior
    ``log p(theta | y)`` rather than the bare predictive likelihood, which
    keeps the lengthscale-estimation problem well-posed (the within-group
    shrinkage prior stays active). ``named_priors()`` yields
    ``(name, parent_module, prior, closure, setting_closure)``; the closure
    takes the parent module (the kernel the prior was registered on).
    """
    total = torch.zeros((), dtype=torch.double)
    for _name, parent_module, prior, closure, _setting in model.named_priors():
        contrib = prior.log_prob(closure(parent_module)).sum()
        total = total.to(contrib) + contrib
    return total


def block_loo_loss(
    model: SingleTaskGP,
    n_real: int,
    *,
    n_composition_dims: int = 9,
) -> torch.Tensor:
    """Differentiable mean block-LOO negative-log-likelihood per real row.

    Args:
        model: a ``SingleTaskGP`` whose hyperparameters require grad.
        n_real: number of real rows; anchor pseudo-rows live at
            ``[n_real:]`` and are held out with their block but not scored.
        n_composition_dims: number of leading input dims that define a
            block (default 9 = composition + temperature).

    Returns a scalar tensor (minimise it to improve held-out calibration).
    """
    train_X = model.train_inputs[0]
    train_Y = model.train_targets
    prior = model(train_X)
    noisy = model.likelihood(prior, train_X)
    K = noisy.lazy_covariance_matrix.to_dense()

    n = K.shape[-1]
    L = psd_safe_cholesky(K)
    Y_t = train_Y.unsqueeze(-1) if train_Y.dim() == 1 else train_Y
    residuals = Y_t - prior.mean.unsqueeze(-1)
    alpha = torch.cholesky_solve(residuals, L).squeeze(-1)
    K_inv = torch.cholesky_solve(torch.eye(n, dtype=K.dtype, device=K.device), L)

    fingerprints = train_X[..., :n_composition_dims]
    unique_fp, inverse = torch.unique(fingerprints, dim=0, return_inverse=True)

    log_two_pi = math.log(2.0 * math.pi)
    total_nll = torch.zeros((), dtype=K.dtype, device=K.device)
    n_scored = 0
    for g in range(unique_fp.shape[0]):
        idx = (inverse == g).nonzero(as_tuple=True)[0]
        K_inv_BB_inv = torch.linalg.inv(K_inv[idx][:, idx])
        block_residual = K_inv_BB_inv @ alpha[idx]
        block_var = torch.diagonal(K_inv_BB_inv).clamp_min(1e-12)
        block_nll = 0.5 * (
            log_two_pi + torch.log(block_var) + (block_residual**2) / block_var
        )
        real_mask = idx < n_real
        total_nll = total_nll + block_nll[real_mask].sum()
        n_scored += int(real_mask.sum())

    nll_per_row = total_nll / n_scored
    return nll_per_row - _model_prior_log_prob(model) / n_scored


def train_block_loo(
    model: SingleTaskGP,
    n_real: int,
    *,
    mll_weight: float = 0.5,
    max_iter: int = 150,
    lr: float = 0.1,
) -> float:
    """Train the model's hyperparameters from their current initialisation
    on a single combined objective, minimised with LBFGS::

        L(theta) = (1 - mll_weight) * block_loo_NLL(theta)
                   +      mll_weight * MLL_NLL(theta)

    The block-LOO term optimises held-out predictive calibration (the
    deployment metric), but is scored only at the training ages, so on its
    own it leaves the predictive mean free to *oscillate between* those ages
    (non-monotone strength curves). The marginal-likelihood (MLL) term adds
    back the ``-0.5 log|K|`` Occam/complexity penalty that block-LOO
    structurally lacks; a modest ``mll_weight`` restores smooth, monotone
    curves at negligible block-LOO cost. Trained from a freshly-constructed
    model (no MLL warm-up) this is a genuine single-objective fit, not a
    two-stage MLL-then-refine. Returns the final (pure) block-LOO loss.
    """
    from gpytorch.mlls import ExactMarginalLogLikelihood

    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.LBFGS(
        params,
        lr=lr,
        max_iter=max_iter,
        tolerance_grad=1e-6,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        output = model(*model.train_inputs)
        mll_nll = -mll(output, model.train_targets)
        loss = (1.0 - mll_weight) * block_loo_loss(model, n_real) + mll_weight * mll_nll
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return block_loo_loss(model, n_real).item()
