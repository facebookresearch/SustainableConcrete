#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark study: model variants for strength GP — calibration + LOO RMSE.

Each variant is fit on the public 647-row strength dataset and scored with
the GP's closed-form analytical LOO at fitted hyperparameters
(``boxcrete.compute_loo_cv``). Six metrics are reported:

  * RMSE / MAE         — point prediction error (psi)
  * Mean LPD           — log predictive density (proper scoring rule)
  * PIT-KS             — Kolmogorov-Smirnov distance of empirical
                         standardised residuals from N(0, 1) [calibration]
  * 95% coverage       — fraction with |y - mean| < 1.96 * std
                         (well-calibrated → ~0.95)
  * CRPS               — continuous ranked probability score (calibration
                         + sharpness combined)

Variants currently implemented:

  Baseline       — production model (Matern + RBF time, learned scalar
                   noise, WithinGroupShrinkagePrior).
  Variant A      — heteroscedastic noise: train_Yvar = Strength (Std)^2
                   per row, plumbed through FixedNoiseGaussianLikelihood.
  Variant B      — source-aware kernel decomposition:
                   K_shared(x_no_source) + K_source(x_no_source) * δ(s_i, s_j).
  Variant C      — log-Y outcome transform; day-zero anchors dropped.

Usage::

    python scripts/model_variant_study.py --seeds 0 1 2 3 4
    python scripts/model_variant_study.py --variants baseline A   # subset

Stacks (e.g. A+B) are also supported; see ``--variants A_plus_B`` etc.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch  # noqa: E402
from botorch.models import SingleTaskGP  # noqa: E402
from botorch.models.transforms import Standardize  # noqa: E402
from botorch.utils.constraints import LogTransformedInterval  # noqa: E402
from gpytorch.kernels import (
    IndexKernel,
    MaternKernel,
    RBFKernel,
    ScaleKernel,
)  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: E402
from botorch.fit import fit_gpytorch_mll  # noqa: E402

from boxcrete import compute_loo_cv  # noqa: E402
from boxcrete.kernels import NUM_MATERIAL_CLASSES  # noqa: E402
from boxcrete.likelihoods import (  # noqa: E402
    PartialFixedNoiseLikelihood,
)
from boxcrete.strength_model_legacy import (  # noqa: E402
    get_strength_gp_input_transform,
)
from boxcrete.utils import (  # noqa: E402
    DEFAULT_X_COLUMNS,
    get_day_zero_data,
    load_concrete_strength,
)

# Material Source is dim 7 in DEFAULT_X_COLUMNS; verify at runtime.
_SOURCE_DIM = DEFAULT_X_COLUMNS.index("Material Source")


# ---------------------------------------------------------------------------
# LOO calibration + accuracy metrics
# ---------------------------------------------------------------------------


def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2)))


def _normal_pdf(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)


def _standardize_Y(Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Manual Y standardisation used by all variants whose likelihood is not a
    plain GaussianLikelihood with `Standardize` outcome transform — i.e., the
    custom-likelihood floor / mult / full variants and the engineered-feature
    variants.

    Returns ``(Y_standardised, y_mean, y_std)``. Each caller is responsible
    for stashing ``y_mean`` and ``y_std`` on the model afterwards (as
    ``model._study_y_mean`` and ``model._study_y_std``) so that the metric
    helpers (`loo_metrics`, `block_loo_metrics`, `held_out_metrics`) can
    untransform predictions back to original psi space.
    """
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)
    y_mean = Y.mean(dim=0, keepdim=True)
    y_std = Y.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (Y - y_mean) / y_std, y_mean, y_std


def _crps_normal(err: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Closed-form CRPS for a Normal predictive distribution.

    CRPS(N(μ, σ²), y) = σ · ( z·(2·Φ(z) − 1) + 2·φ(z) − 1/√π ),
    where z = (y − μ) / σ, Φ and φ are the standard normal CDF / PDF.
    """
    z = err / std
    return std * (
        z * (2 * _normal_cdf(z) - 1) + 2 * _normal_pdf(z) - 1.0 / math.sqrt(math.pi)
    )


def _strength_curve_monotonicity_penalty(
    model: SingleTaskGP,
    raw_compositions: torch.Tensor,
    *,
    n_smooth_times: int = 20,
    t_min: float = 0.5,
    t_max: float = 28.0,
    p: float = 2.0,
) -> torch.Tensor:
    """Soft monotonicity penalty: ``λ · mean(relu(-dμ/dt)^p)``.

    Unlike the second-derivative smoothness penalty, this directly
    penalizes the bad thing — DECREASING intervals — while leaving the
    natural curvature of the strength curve untouched (diminishing
    returns saturation, delayed pozzolanic activation, etc. are all
    monotone-increasing patterns and incur ZERO penalty).

    Args mirror ``_strength_curve_smoothness_penalty``. ``p=1`` is the
    classic hinge (Riihimäki-Vehtari style); ``p=2`` is smoother and
    differentiable everywhere.
    """
    import gpytorch

    times = torch.exp(
        torch.linspace(
            math.log(t_min),
            math.log(t_max),
            n_smooth_times,
        )
    ).to(raw_compositions)
    n_comp = raw_compositions.shape[0]
    n_t = n_smooth_times
    comp_b = raw_compositions.unsqueeze(1).expand(n_comp, n_t, 9)
    time_b = times.view(1, n_t, 1).expand(n_comp, n_t, 1)
    raw_full = torch.cat([comp_b, time_b], dim=-1).reshape(-1, 10)

    was_training = model.training
    model.eval()
    try:
        with gpytorch.settings.detach_test_caches(False):
            posterior = model.posterior(raw_full)
            mean = posterior.mean
    finally:
        if was_training:
            model.train()

    mean = mean.reshape(n_comp, n_t)
    slopes = mean[:, 1:] - mean[:, :-1]
    # Hinge: only negative slopes (decreasing intervals) contribute.
    violations = torch.clamp(-slopes, min=0.0)
    return (violations**p).mean()


def _strength_curve_smoothness_penalty(
    model: SingleTaskGP,
    raw_compositions: torch.Tensor,
    *,
    n_smooth_times: int = 20,
    t_min: float = 0.5,
    t_max: float = 28.0,
    p: float = 2.0,
) -> torch.Tensor:
    """Penalty on the second derivative of the predicted strength curve
    over a dense time grid for the given training compositions.

    Block-LOO measures error AT the training time points (1, 7, 28 days)
    only. Oscillations BETWEEN training points are invisible to that
    objective — so we can add this smoothness penalty without sacrificing
    block-LOO predictive performance, while suppressing the unphysical
    between-day wiggles that the kernel can otherwise produce when
    given features with non-commensurate time-dependence.

    Returns a scalar tensor: mean over (composition × time) of
    |Δ²μ/Δt²|^p, where the second difference is taken over a log-spaced
    time grid in [t_min, t_max].

    Args:
        model: a SingleTaskGP with an input_transform that handles raw
            10-dim inputs [composition × 9, time].
        raw_compositions: [m, 9] tensor of training compositions (without
            time). Typically pass the unique compositions from the
            training set.
        n_smooth_times: number of log-spaced times in the grid.
        t_min, t_max: range of times in days.
        p: power for the penalty (1=L1, 2=L2). L2 emphasises sharp
            oscillations; L1 is more robust to occasional spikes.
    """
    import gpytorch

    times = torch.exp(
        torch.linspace(
            math.log(t_min),
            math.log(t_max),
            n_smooth_times,
        )
    ).to(raw_compositions)
    n_comp = raw_compositions.shape[0]
    n_t = n_smooth_times
    # Build [n_comp, n_t, 10] raw input (composition + time).
    comp_b = raw_compositions.unsqueeze(1).expand(n_comp, n_t, 9)
    time_b = times.view(1, n_t, 1).expand(n_comp, n_t, 1)
    raw_full = torch.cat([comp_b, time_b], dim=-1).reshape(-1, 10)

    # Switch to eval mode briefly so model.posterior() returns the
    # posterior (rather than the prior) — we restore mode at the end.
    # Caches must NOT be detached so gradients flow back to kernel HPs.
    was_training = model.training
    model.eval()
    try:
        with gpytorch.settings.detach_test_caches(False):
            posterior = model.posterior(raw_full)
            mean = posterior.mean
    finally:
        if was_training:
            model.train()

    # mean is in normalized-Y space ([0, 1] for max-scale models).
    # Reshape and take second differences over the time axis.
    mean = mean.reshape(n_comp, n_t)
    second_diffs = mean[:, 2:] - 2 * mean[:, 1:-1] + mean[:, :-2]
    return (second_diffs.abs() ** p).mean()


def _model_prior_log_prob(model: SingleTaskGP) -> torch.Tensor:
    """Sum of log p(hyperparameter) under all priors registered on the
    model (kernel ARD-lengthscale priors like ``WithinGroupShrinkagePrior``,
    outputscale priors, mean priors, etc.).

    Mirrors the prior-additive term in
    ``gpytorch.mlls.ExactMarginalLogLikelihood``:

        log p(y, theta) = log p(y | theta) + log p(theta)

    so that the block-LOO refinement maximises the *posterior*
    log p(theta | y_holdouts) rather than just the predictive
    log-likelihood. Without this term, the lengthscales were free to
    drift to extreme values during refinement (one input dim becoming
    "inactive" via huge lengthscale was a documented failure mode in
    earlier studies — see §4.4 of the parent benchmark on the within-
    group shrinkage prior). Adding the prior keeps the lengthscale
    estimation problem well-posed and preserves the regularisation
    we get from the MLL stage.

    Returns a scalar tensor with grad attached.

    Implementation note: ``named_priors()`` yields tuples of
    ``(full_name, parent_module, prior, closure, setting_closure)``
    where ``closure`` is a function that takes the PARENT MODULE (the
    kernel where the prior was registered) and returns the prior's
    target tensor. We must pass ``parent_module``, NOT the top-level
    SingleTaskGP. This bug previously caused silent failures —
    ``closure(model)`` raised AttributeError which was swallowed by an
    outer try/except, making the block-LOO refinement a silent no-op.
    """
    total = torch.tensor(0.0, dtype=torch.double)
    moved = False
    for _name, parent_module, prior, closure, _setting_closure in model.named_priors():
        # Try several calling conventions for compat across GPyTorch
        # versions / custom prior closures.
        try:
            value = closure(parent_module)
        except TypeError:
            try:
                value = closure()
            except TypeError as e:
                raise TypeError(
                    f"Cannot determine calling convention for prior closure '{_name}': {e}"
                )
        contrib = prior.log_prob(value).sum()
        if not moved:
            total = total.to(contrib)
            moved = True
        total = total + contrib
    return total


def block_loo_loss(
    model: SingleTaskGP,
    *,
    n_composition_dims: int = 9,
    real_only: bool = True,
    n_real: int | None = None,
    include_priors: bool = True,
    smoothness_lambda: float = 0.0,
    smoothness_compositions: torch.Tensor | None = None,
    smoothness_n_times: int = 20,
    smoothness_p: float = 2.0,
    monotonicity_lambda: float = 0.0,
    monotonicity_p: float = 2.0,
) -> torch.Tensor:
    """Differentiable block-LOO negative-log-likelihood loss.

    Same math as ``block_loo_metrics`` but in train-mode with grad
    enabled — usable as an objective for hyperparameter optimisation.
    Replaces the standard marginal log-likelihood (MLL) which has
    repeatedly been shown in this study to dissociate from block-LOO
    (the deployment metric we care about): MLL up but block-LOO down.

    Returns the **mean negative block-LOO log-likelihood per row** as
    a scalar tensor; minimising it directly improves block-LOO
    predictive distributions. The variance term ``block_var`` includes
    aleatoric noise (it is read from ``K_full = K + σ²I``), so this
    objective penalises mis-calibrated predictive intervals AS WELL AS
    point error — it is the proper *predictive log-likelihood*, not
    just RMSE.

    References
    ----------
    Single-row LOO in closed form:
        Sundararajan, S. & Keerthi, S. S. (2001). "Predictive
        Approaches for Choosing Hyperparameters in Gaussian
        Processes." NIPS 13.

    Theoretical advantage over MLL under model misspecification (which
    matches the empirical pattern we observe in this study —
    block-LOO refinement's gain grows with data sparsity, see §6.12 +
    §6.13 in the parent benchmark):
        Bachoc, F. (2013). "Cross Validation and Maximum Likelihood
        estimations of hyper-parameters of Gaussian processes."
        Journal of Statistical Planning and Inference 143(8).

    General review of Bayesian predictive selection / averaging:
        Vehtari, A. & Ojanen, J. (2012). "A survey of Bayesian
        predictive methods for model assessment, selection and
        comparison." Statistics Surveys 6.

    Block extension (group-LOO via the block-inverse identity) is
    folklore in the GP cross-validation literature; the closed-form
    K_BB formula used here is the standard generalisation::

        μ_LOO_B = y_B - (K_full^{-1}_BB)^{-1} α_B
        Σ_LOO_B = (K_full^{-1}_BB)^{-1}        # includes aleatoric

    where ``α = K_full^{-1} (y - μ_prior)`` and ``K_full = K + σ²I``.

    Args:
        model: a SingleTaskGP whose hyperparameters require_grad.
        n_composition_dims: number of leading dims that define a block
            (default 9 = composition + temp).
        real_only: if True (default), score only the real rows
            ([:n_real]); useful when the model has appended day-zero
            anchor pseudo-rows that we don't want included in the loss.
        n_real: required when ``real_only=True``. Number of real rows;
            anchor pseudo-rows are at indices [n_real:].

    Closed-form: uses the Sundararajan-Keerthi block-inverse identity
    (forms K^{-1} once, then for each block does a small b×b inverse).
    Cost dominated by the K^{-1} (O(n^3)) — same as one MLL gradient
    step, so this is essentially MLL-cost per outer iteration.
    """
    from gpytorch.utils.cholesky import psd_safe_cholesky

    train_X = model.train_inputs[0]
    train_Y = model.train_targets
    # Forward: kernel value with current hyperparameters + likelihood noise
    prior = model(train_X)
    noisy = model.likelihood(prior, train_X)
    K = noisy.lazy_covariance_matrix.to_dense()

    n = K.shape[-1]
    L = psd_safe_cholesky(K)
    Y_t = train_Y.unsqueeze(-1) if train_Y.dim() == 1 else train_Y
    residuals = Y_t - prior.mean.unsqueeze(-1)
    alpha = torch.cholesky_solve(residuals, L).squeeze(-1)
    eye = torch.eye(n, dtype=K.dtype, device=K.device)
    K_inv = torch.cholesky_solve(eye, L)

    fingerprints = train_X[..., :n_composition_dims]
    unique_fp, inverse = torch.unique(fingerprints, dim=0, return_inverse=True)

    log_two_pi = math.log(2.0 * math.pi)
    total_neg_log_lik = torch.zeros((), dtype=K.dtype, device=K.device)
    n_scored = 0
    score_limit = n_real if (real_only and n_real is not None) else n
    for g in range(unique_fp.shape[0]):
        idx = (inverse == g).nonzero(as_tuple=True)[0]
        # Restrict to the real-only subset if requested.
        if real_only and n_real is not None:
            idx = idx[idx < score_limit]
            if idx.numel() == 0:
                continue
        K_inv_BB = K_inv[idx][:, idx]
        K_inv_BB_inv = torch.linalg.inv(K_inv_BB)
        # In SK notation, the LOO residual at block i is K_inv_BB_inv @ alpha_block,
        # and the LOO predictive variance is the diagonal of K_inv_BB_inv.
        block_residual = K_inv_BB_inv @ alpha[idx]
        block_var = torch.diagonal(K_inv_BB_inv).clamp_min(1e-12)
        # log N(0 | residual, var) = -0.5 * (log(2pi) + log(var) + residual^2 / var)
        block_neg_log_lik = 0.5 * (
            log_two_pi + torch.log(block_var) + (block_residual**2) / block_var
        )
        total_neg_log_lik = total_neg_log_lik + block_neg_log_lik.sum()
        n_scored += int(idx.shape[0])
    nll_per_row = total_neg_log_lik / max(n_scored, 1)
    if include_priors:
        prior_lp = _model_prior_log_prob(model)
        nll_per_row = nll_per_row - prior_lp / max(n_scored, 1)
    if smoothness_lambda > 0 and smoothness_compositions is not None:
        # Add the smoothness regularizer. Operates on the predictive
        # mean at intermediate (non-training) times, so it does not
        # directly hurt block-LOO predictive performance.
        s_pen = _strength_curve_smoothness_penalty(
            model,
            smoothness_compositions,
            n_smooth_times=smoothness_n_times,
            p=smoothness_p,
        )
        nll_per_row = nll_per_row + smoothness_lambda * s_pen
    if monotonicity_lambda > 0 and smoothness_compositions is not None:
        # Hinge penalty on decreasing intervals only — leaves natural
        # curvature alone. Targets the actual physical violation we
        # care about (negative slopes) rather than total curvature.
        m_pen = _strength_curve_monotonicity_penalty(
            model,
            smoothness_compositions,
            n_smooth_times=smoothness_n_times,
            p=monotonicity_p,
        )
        nll_per_row = nll_per_row + monotonicity_lambda * m_pen
    return nll_per_row


def refine_with_block_loo(
    model: SingleTaskGP,
    *,
    n_real: int | None = None,
    max_iter: int = 50,
    lr: float = 0.1,
    smoothness_lambda: float = 0.0,
    smoothness_compositions: torch.Tensor | None = None,
    monotonicity_lambda: float = 0.0,
) -> float:
    """Refine model hyperparameters by minimising block-LOO loss via
    LBFGS, starting from whatever initialization the model already has
    (typically the MLL optimum from ``fit_gpytorch_mll``).

    Returns the final block-LOO loss value.

    Optional: pass ``smoothness_lambda > 0`` and
    ``smoothness_compositions`` (a [m, 9] tensor of training compositions)
    to add a between-training-points smoothness regularizer on the
    predicted strength curves. Block-LOO measures error AT training time
    points (1, 7, 28 days) only, so the curve's behaviour BETWEEN those
    points is unconstrained — adding the smoothness penalty here
    prevents unphysical wiggles without hurting block-LOO.
    """
    model.train()
    # Collect parameters that require grad
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
        loss = block_loo_loss(
            model,
            real_only=True,
            n_real=n_real,
            smoothness_lambda=smoothness_lambda,
            smoothness_compositions=smoothness_compositions,
            monotonicity_lambda=monotonicity_lambda,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        # Report the un-smoothed block-LOO loss as the "final" so it's
        # comparable across smoothness_lambda settings.
        final_loss = block_loo_loss(model, real_only=True, n_real=n_real).item()
    return final_loss


def loo_metrics(
    model: SingleTaskGP,
    n_real: int,
    *,
    include_anchors: bool = False,
) -> dict[str, float]:
    """LOO calibration + accuracy metrics in the model's outcome-transformed
    space (untransformed back to original units by ``compute_loo_cv``).

    For models without an outcome_transform but with manual standardisation
    (the noise-floor variants), we untransform via ``model._study_y_mean``
    and ``model._study_y_std`` instead.

    By default ignores the day-zero anchors that ``fit_strength_gp`` appends.
    Pass ``include_anchors=True`` to score them too — useful for tracking
    how well the model fits the physics constraint f(x, 0) ≈ 0.
    """
    n_to_score = n_real
    if include_anchors and hasattr(model, "_study_n_pseudo"):
        n_to_score = n_real + int(model._study_n_pseudo)
    obs, mean, std = compute_loo_cv(model, n_real=n_to_score)
    if hasattr(model, "_study_y_std"):
        y_mean = model._study_y_mean.to(mean)
        y_std = model._study_y_std.to(mean)
        mean = mean * y_std + y_mean
        obs = obs * y_std + y_mean
        std = std * y_std
    err = mean - obs
    rmse = (err.pow(2).mean().sqrt()).item()
    mae = err.abs().mean().item()

    log_pd = -0.5 * (err / std).pow(2) - std.log() - 0.5 * math.log(2 * math.pi)
    mean_lpd = log_pd.mean().item()

    z = err / std
    z_sorted, _ = torch.sort(z)
    n = z.numel()
    empirical_cdf = torch.arange(1, n + 1, dtype=z.dtype) / n
    pit_ks = (empirical_cdf - _normal_cdf(z_sorted)).abs().max().item()

    coverage_95 = (z.abs() < 1.959963984540054).float().mean().item()
    crps = _crps_normal(err, std).mean().item()

    return {
        "rmse": rmse,
        "mae": mae,
        "mean_lpd": mean_lpd,
        "pit_ks": pit_ks,
        "coverage_95": coverage_95,
        "crps": crps,
    }


def block_loo_metrics(
    model: SingleTaskGP,
    n_real: int,
    *,
    n_composition_dims: int = 9,
    include_anchors: bool = False,
    anchors_only: bool = False,
) -> dict[str, float]:
    """Block-LOO metrics: each block = one unique composition.

    Generalises Sundararajan-Keerthi single-row LOO to leaving out blocks
    of correlated rows. For block ``B`` with ``b = |B|``::

        mu_LOO[B] = y_B - inv(K^{-1}_BB) @ alpha_B
        Sigma_LOO[B] = inv(K^{-1}_BB)

    where ``K`` is the noisy kernel matrix and ``alpha = K^{-1}(y - mu_prior)``
    are the same precomputed quantities the standard single-row LOO uses.

    Block definition: rows sharing the same input fingerprint over the
    first ``n_composition_dims`` dims (the raw composition features
    Cement..Source..Temp, excluding time and any appended engineered
    features that may depend on time). For the strength dataset this
    gives ~144 blocks of ~4–5 rows each. Day-zero anchors are added by
    composition fingerprint, so they automatically join the block of
    their composition's real measurements — holding out a composition
    also holds out its day-zero anchor (no information leak).

    Cost: forms the full ``K^{-1}`` once (which is already needed for
    the single-row LOO diagonal) plus 144 small ``b x b`` solves —
    negligible relative to the GP fit itself.

    Score subset (one of three):
    - default: real rows only ([:n_real])
    - ``include_anchors=True`` (and not ``anchors_only``): all rows
      (real + anchors)
    - ``anchors_only=True``: anchors rows only ([n_real:n_real+n_pseudo])
      — useful for diagnosing how well the anchor mechanism is pinning
      f(x, 0) to 0 under composition-level holdout. Requires the model
      to have ``_study_n_pseudo`` set.
    """
    from gpytorch.utils.cholesky import psd_safe_cholesky

    model.eval()
    with torch.no_grad():
        train_X = model.train_inputs[0]  # already post-input-transform
        train_Y = model.train_targets
        # forward() skips the input_transform, matching compute_loo_cv's pattern
        prior = model.forward(train_X)
        noisy = model.likelihood(prior)
        K = noisy.lazy_covariance_matrix.to_dense()

    n = K.shape[-1]
    L = psd_safe_cholesky(K)
    if train_Y.dim() == 1:
        Y_t = train_Y.unsqueeze(-1)
    else:
        Y_t = train_Y
    residuals = Y_t - prior.mean.unsqueeze(-1)
    alpha = torch.cholesky_solve(residuals, L).squeeze(-1)
    I = torch.eye(n, dtype=K.dtype, device=K.device)
    K_inv = torch.cholesky_solve(I, L)

    # Block fingerprint: first 9 raw composition dims (Cement, Fly Ash,
    # Slag, Water, HRWR, Fine, Coarse, Material Source, Temp). Time
    # (dim 9) and any engineered features (dims 10+ in F1..F5 variants)
    # vary within a composition, so they're excluded from the fingerprint.
    fingerprints = train_X[..., :n_composition_dims]
    unique_fp, inverse = torch.unique(fingerprints, dim=0, return_inverse=True)

    loo_pred = torch.zeros(n, dtype=K.dtype, device=K.device)
    loo_var = torch.zeros(n, dtype=K.dtype, device=K.device)
    for g in range(unique_fp.shape[0]):
        idx = (inverse == g).nonzero(as_tuple=True)[0]
        K_inv_BB = K_inv[idx][:, idx]
        K_inv_BB_inv = torch.linalg.inv(K_inv_BB)
        block_mean_offset = K_inv_BB_inv @ alpha[idx]
        loo_pred[idx] = Y_t[idx, 0] - block_mean_offset
        loo_var[idx] = torch.diagonal(K_inv_BB_inv).clamp_min(1e-12)

    # Choose the score subset.
    if anchors_only:
        n_pseudo = int(getattr(model, "_study_n_pseudo", 0))
        if n_pseudo == 0:
            raise ValueError(
                "anchors_only=True requires the model to have _study_n_pseudo > 0"
            )
        obs = Y_t[n_real : n_real + n_pseudo, 0]
        mean_v = loo_pred[n_real : n_real + n_pseudo]
        std_v = loo_var[n_real : n_real + n_pseudo].sqrt()
    elif include_anchors:
        obs = Y_t[:, 0]
        mean_v = loo_pred
        std_v = loo_var.sqrt()
    else:
        obs = Y_t[:n_real, 0]
        mean_v = loo_pred[:n_real]
        std_v = loo_var[:n_real].sqrt()

    # Untransform back to original units (mirrors compute_loo_cv's logic).
    if hasattr(model, "outcome_transform"):
        otf = model.outcome_transform
        if hasattr(otf, "stdvs") and hasattr(otf, "means"):
            stdvs = otf.stdvs.squeeze().to(mean_v)
            means = otf.means.squeeze().to(mean_v)
            mean_v = mean_v * stdvs + means
            obs = obs * stdvs + means
            std_v = std_v * stdvs
    if hasattr(model, "_study_y_std"):
        y_std = model._study_y_std.to(mean_v)
        y_mean = model._study_y_mean.to(mean_v)
        mean_v = mean_v * y_std + y_mean
        obs = obs * y_std + y_mean
        std_v = std_v * y_std

    err = mean_v - obs
    rmse = err.pow(2).mean().sqrt().item()
    mae = err.abs().mean().item()
    log_pd = -0.5 * (err / std_v).pow(2) - std_v.log() - 0.5 * math.log(2 * math.pi)
    mean_lpd = log_pd.mean().item()
    z = err / std_v
    z_sorted, _ = torch.sort(z)
    nrows = z.numel()
    empirical_cdf = torch.arange(1, nrows + 1, dtype=z.dtype) / nrows
    pit_ks = (empirical_cdf - _normal_cdf(z_sorted)).abs().max().item()
    coverage_95 = (z.abs() < 1.959963984540054).float().mean().item()
    crps = _crps_normal(err, std_v).mean().item()

    return {
        "rmse": rmse,
        "mae": mae,
        "mean_lpd": mean_lpd,
        "pit_ks": pit_ks,
        "coverage_95": coverage_95,
        "crps": crps,
        "n_blocks": float(unique_fp.shape[0]),
        "avg_block_size": float(n) / unique_fp.shape[0],
    }


def phantom_anchor_metrics(
    model: SingleTaskGP,
    n_real: int,
    *,
    n_composition_dims: int = 9,
    out_of_training_n: int = 0,
    out_of_training_seed: int = 0,
    bounds: torch.Tensor | None = None,
) -> dict[str, float]:
    """Score the model's posterior predictions at *virtual* (c, t=0) test
    points. Two modes:

    1. **In-training** (default): one phantom test point per unique
       composition in the training set. The composition `c` is in
       training but `(c, t=0)` is not — measures how well the model
       extrapolates the strength curve to t=0 for known compositions.

    2. **Out-of-training** (`out_of_training_n > 0`): additionally sample
       `out_of_training_n` random compositions uniformly within the
       provided ``bounds`` and evaluate at t=0. Measures how well the
       model satisfies the physics constraint **everywhere in the input
       space**, not just at training compositions. Critical for
       comparing gated kernels (which produce 0 everywhere by
       construction) against anchor pseudo-observations (which are
       only enforced near training compositions and drift elsewhere).

    Returns a dict with:
        - rmse, mae, mean_pred, max_abs_pred (at training compositions)
        - oot_rmse, oot_mae, oot_max_abs_pred (at random OOT points,
          if requested)

    For the gated kernel, both training and OOT phantom RMSEs should be
    ~0 (the constraint is structural). For anchor pseudo-observations,
    the training metric will be small (anchors fit well at trained
    compositions) but OOT can drift substantially.

    ``bounds`` must be provided if ``out_of_training_n > 0`` — used to
    sample random OOT compositions uniformly. Format matches BoTorch:
    ``bounds[0]`` is lower per-dim, ``bounds[1]`` is upper per-dim;
    shape ``[2, 10]`` for the 10-dim raw input.
    """
    raw_X = getattr(model, "_study_X_train_raw", None)
    if raw_X is None:
        raise ValueError(
            "phantom_anchor_metrics requires model._study_X_train_raw "
            "(10-dim raw real-rows training X) to be stashed by the fit factory"
        )
    if raw_X.shape[0] != n_real:
        raise ValueError(
            f"Stashed _study_X_train_raw has {raw_X.shape[0]} rows but n_real={n_real} — "
            "the fit factory should stash only real rows (no anchor pseudo-rows)"
        )

    # In-training phantom test points: each unique training composition + t=0.
    fingerprint = raw_X[..., :n_composition_dims]
    unique_comp, _ = torch.unique(fingerprint, dim=0, return_inverse=True)
    n_unique = unique_comp.shape[0]
    zero_time = torch.zeros(
        n_unique,
        1,
        dtype=unique_comp.dtype,
        device=unique_comp.device,
    )
    X_phantom = torch.cat([unique_comp, zero_time], dim=-1)  # [n_unique, 10]

    # Out-of-training phantom test points (if requested): sample random
    # compositions from `bounds` and pin t=0.
    if out_of_training_n > 0:
        if bounds is None:
            raise ValueError(
                "out_of_training_n > 0 requires `bounds` to sample random OOT "
                "compositions"
            )
        gen = torch.Generator(device=raw_X.device).manual_seed(out_of_training_seed)
        # bounds has shape [2, 10]: bounds[0] = lower, bounds[1] = upper.
        lo = bounds[0].to(raw_X)
        hi = bounds[1].to(raw_X)
        u = torch.rand(
            out_of_training_n,
            lo.shape[-1],
            dtype=raw_X.dtype,
            device=raw_X.device,
            generator=gen,
        )
        X_oot = lo + u * (hi - lo)
        X_oot[..., 9] = 0.0  # set time to 0
    else:
        X_oot = None

    model.eval()
    with torch.no_grad():
        try:
            posterior = model.posterior(X_phantom)
        except (TypeError, RuntimeError) as e:
            raise RuntimeError(f"Phantom-anchor posterior failed: {e}") from e
        mean = posterior.mean.squeeze(-1)
        if X_oot is not None:
            try:
                posterior_oot = model.posterior(X_oot)
                mean_oot = posterior_oot.mean.squeeze(-1)
            except (TypeError, RuntimeError) as e:
                raise RuntimeError(f"OOT phantom-anchor posterior failed: {e}") from e
        else:
            mean_oot = None

    if hasattr(model, "_study_y_std"):
        y_mean = model._study_y_mean.to(mean)
        y_std = model._study_y_std.to(mean)
        mean = mean * y_std + y_mean
        if mean_oot is not None:
            mean_oot = mean_oot * y_std + y_mean

    err = mean
    rmse = err.pow(2).mean().sqrt().item()
    mae = err.abs().mean().item()
    out = {
        "rmse": float(rmse),
        "mae": float(mae),
        "mean_pred": float(mean.mean().item()),
        "max_abs_pred": float(mean.abs().max().item()),
        "n_phantoms": int(n_unique),
    }
    if mean_oot is not None:
        out["oot_rmse"] = float(mean_oot.pow(2).mean().sqrt().item())
        out["oot_mae"] = float(mean_oot.abs().mean().item())
        out["oot_max_abs_pred"] = float(mean_oot.abs().max().item())
        out["oot_n"] = int(out_of_training_n)
    return out


def held_out_metrics(
    model: SingleTaskGP,
    X_test: torch.Tensor,
    Y_test_psi: torch.Tensor,
) -> dict[str, float]:
    """Compute the same metrics as ``loo_metrics`` but on a *held-out* test set.

    Uses ``model.posterior(X_test, observation_noise=True)`` so the predictive
    variance includes the homoscedastic part of the likelihood (noise floor).
    The fixed-noise term contributes 0 at test points (size mismatch with
    train_Yvar), so we get pure latent variance + global noise floor — which
    is exactly what's appropriate for an unseen mix.

    For models with manual Y-standardisation (``_study_y_std`` attribute set),
    the posterior is in standardised space; we untransform by
    ``y_mean + y_std * mean_z``. For models with ``Standardize`` outcome
    transform, BoTorch returns the posterior already in psi space.

    ``Y_test_psi`` must be 1-D in psi units (squeeze any (..., 1) dim before
    calling).
    """
    model.eval()
    with torch.no_grad():
        try:
            posterior = model.posterior(X_test, observation_noise=True)
        except (TypeError, RuntimeError):
            # Fallback for likelihoods that don't support observation_noise kwarg
            posterior = model.posterior(X_test)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)

    if hasattr(model, "_study_y_std"):
        y_mean = model._study_y_mean.to(mean)
        y_std = model._study_y_std.to(mean)
        mean = mean * y_std + y_mean
        std = std * y_std

    Y_test_psi = Y_test_psi.to(mean)
    err = mean - Y_test_psi
    rmse = err.pow(2).mean().sqrt().item()
    mae = err.abs().mean().item()
    log_pd = -0.5 * (err / std).pow(2) - std.log() - 0.5 * math.log(2 * math.pi)
    mean_lpd = log_pd.mean().item()

    z = err / std
    z_sorted, _ = torch.sort(z)
    n = z.numel()
    empirical_cdf = torch.arange(1, n + 1, dtype=z.dtype) / n
    pit_ks = (empirical_cdf - _normal_cdf(z_sorted)).abs().max().item()
    coverage_95 = (z.abs() < 1.959963984540054).float().mean().item()
    crps = _crps_normal(err, std).mean().item()

    return {
        "rmse": rmse,
        "mae": mae,
        "mean_lpd": mean_lpd,
        "pit_ks": pit_ks,
        "coverage_95": coverage_95,
        "crps": crps,
    }


# ---------------------------------------------------------------------------
# Helpers shared across variants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Variant adapters — each returns (fitted_model, n_real)
# ---------------------------------------------------------------------------


def fit_baseline(X, Y, Yvar, bounds, seed):
    """Legacy V1 production model: Matern + additive RBF on time, learned
    scalar noise via PartialFixedNoiseLikelihood, WithinGroupShrinkagePrior.
    The journey-plot baseline."""
    from experiments.legacy_v1_strength_gp import fit_strength_gp_v1

    torch.manual_seed(seed)
    model = fit_strength_gp_v1(
        X=X,
        Y=Y,
        Yvar=Yvar,
        X_bounds=bounds,
        use_fixed_noise=False,
    )
    return model, X.shape[0]


def fit_baseline_no_prior(X, Y, Yvar, bounds, seed):
    """Pre-shrinkage-prior V1 baseline: same as `fit_baseline` but with the
    within-group lengthscale shrinkage prior REMOVED. Recovers the state
    of the production model **before** the prior was promoted. Used to
    quantify the prior's contribution in §4 of the markdown."""
    from experiments.legacy_v1_strength_gp import fit_strength_gp_v1

    torch.manual_seed(seed)
    model = fit_strength_gp_v1(
        X=X,
        Y=Y,
        Yvar=Yvar,
        X_bounds=bounds,
        use_fixed_noise=False,
        lengthscale_prior=None,
    )
    return model, X.shape[0]


def fit_variant_a(X, Y, Yvar, bounds, seed):
    """Variant A — heteroscedastic noise: pass empirical Yvar through.
    Floors zero-Yvar rows at (5 psi)^2 to avoid singular noise."""
    from experiments.legacy_v1_strength_gp import fit_strength_gp_v1

    torch.manual_seed(seed)
    floor = torch.full_like(Yvar, 25.0)  # (5 psi)^2 — well below median std
    Yvar_floored = torch.maximum(Yvar, floor)
    model = fit_strength_gp_v1(
        X=X,
        Y=Y,
        Yvar=Yvar_floored,
        X_bounds=bounds,
        use_fixed_noise=True,
    )
    return model, X.shape[0]


def _build_source_aware_kernel(d_in: int, source_dim: int) -> torch.nn.Module:
    """K = ScaleKernel(Matern(no_source)) + ScaleKernel(Matern(no_source)) ⊙ IndexKernel(source).

    The IndexKernel with rank=0 acts as a Kronecker delta on Material Source:
    the source-specific term contributes only when source_i == source_j. The
    shared Matern is over the 9 non-source dims and provides the cross-source
    pooling. The within-group lengthscale shrinkage prior is installed on the
    shared kernel (which contains the binder + aggregate dims).
    """
    no_source_dims = [i for i in range(d_in) if i != source_dim]

    shared_base = MaternKernel(
        nu=2.5,
        ard_num_dims=len(no_source_dims),
        active_dims=torch.tensor(no_source_dims),
        lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        lengthscale_prior=_within_group_prior(d_in=d_in, source_dim=source_dim),
    )
    shared = ScaleKernel(
        shared_base,
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
    )

    source_specific_base = MaternKernel(
        nu=2.5,
        ard_num_dims=len(no_source_dims),
        active_dims=torch.tensor(no_source_dims),
        lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
    )
    # IndexKernel(rank=0) = pure diagonal δ(s_i, s_j) (after var clamp); one task per Material Source class.
    source_index = IndexKernel(
        num_tasks=NUM_MATERIAL_CLASSES,
        rank=0,
        active_dims=torch.tensor([source_dim]),
    )
    source_specific = ScaleKernel(
        source_specific_base * source_index,
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=0.1),
    )

    time_kernel = ScaleKernel(
        RBFKernel(
            active_dims=torch.tensor([d_in - 1]),
            ard_num_dims=1,
            lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        ),
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
    )
    return shared + source_specific + time_kernel


def fit_variant_b(X, Y, Yvar, bounds, seed):
    """Variant B — source-aware kernel decomposition. Inlines a minimal
    fit_strength_gp variant since the production helper builds its own
    fixed kernel."""
    torch.manual_seed(seed)
    n_real = X.shape[0]
    d_in = X.shape[-1]
    d_out = 1 if Y.dim() == 1 else Y.shape[-1]
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)

    X_0, Y_0, Yvar_0 = get_day_zero_data(X=X, n=128)
    n_pseudo = X_0.shape[0]
    X = torch.cat((X, X_0), dim=0)
    Y = torch.cat((Y, Y_0), dim=0)
    Yvar = torch.cat((Yvar, Yvar_0), dim=0)

    kernel = _build_source_aware_kernel(d_in=d_in, source_dim=_SOURCE_DIM)
    likelihood = PartialFixedNoiseLikelihood(
        n_real=n_real,
        n_pseudo=n_pseudo,
        pseudo_noise=1e-6,
        noise_constraint=LogTransformedInterval(1e-6, 1.0, initial_value=1e-1),
    )
    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        covar_module=kernel,
        likelihood=likelihood,
        input_transform=get_strength_gp_input_transform(d=d_in, bounds=bounds),
        outcome_transform=Standardize(d_out),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model, n_real


def _loo_metrics_lognormal(model, n_real, Y_orig_psi):
    """LOO metrics in psi-space for a model fit in log-Y space.

    The model predicts a Normal in (standardized log-Y) space; ``compute_loo_cv``
    untransforms the Standardize step but not the outer Log. So the outputs are
    in log-psi units, distributed as N(μ_log, σ_log²). The corresponding psi-
    space distribution is lognormal with::

        E[Y]    = exp(μ_log + σ_log² / 2)
        Var[Y]  = E[Y]² · (exp(σ_log²) − 1)

    Calibration metrics (PIT-KS, coverage) are scale-invariant under monotone
    transforms, so we compute them in log-psi space and the answer is the same.
    """
    obs_log, mean_log, std_log = compute_loo_cv(model, n_real=n_real)

    # Point metrics in psi space — use the lognormal mean.
    mean_psi = (mean_log + 0.5 * std_log.pow(2)).exp()
    obs_psi = Y_orig_psi
    err = mean_psi - obs_psi
    rmse = err.pow(2).mean().sqrt().item()
    mae = err.abs().mean().item()

    # Lognormal predictive std in psi space (first-order via the moment formula).
    var_psi = mean_psi.pow(2) * (std_log.pow(2).exp() - 1)
    std_psi = var_psi.sqrt()

    # CRPS and mean LPD in psi space, using the lognormal predictive density.
    # Closed-form lognormal CRPS isn't standard; we evaluate via the relation
    # CRPS = E_Y|Y - y| - 0.5 E_Y|Y - Y'|. The simpler proxy: report Gaussian
    # CRPS using mean_psi/std_psi (first-order). Mean LPD uses the true
    # lognormal density.
    log_pdf_lognormal = (
        -torch.log(obs_psi.clamp(min=1.0))
        - torch.log(std_log)
        - 0.5 * math.log(2 * math.pi)
        - 0.5 * ((obs_psi.clamp(min=1.0).log() - mean_log) / std_log).pow(2)
    )
    mean_lpd = log_pdf_lognormal.mean().item()
    crps = _crps_normal(err, std_psi).mean().item()  # first-order proxy

    # Calibration metrics (scale-invariant): use log-space.
    err_log = mean_log - obs_log
    z = err_log / std_log
    z_sorted, _ = torch.sort(z)
    n = z.numel()
    empirical_cdf = torch.arange(1, n + 1, dtype=z.dtype) / n
    pit_ks = (empirical_cdf - _normal_cdf(z_sorted)).abs().max().item()
    coverage_95 = (z.abs() < 1.959963984540054).float().mean().item()

    return {
        "rmse": rmse,
        "mae": mae,
        "mean_lpd": mean_lpd,
        "pit_ks": pit_ks,
        "coverage_95": coverage_95,
        "crps": crps,
    }


def fit_variant_c(X, Y, Yvar, bounds, seed):
    """Variant C — log-Y outcome transform.

    Filters 2 anomalous rows with Y < 10 psi (likely lab failures; in log space
    they become extreme outliers anyway). Pre-logs Y outside the model and uses
    Standardize as the lone outcome transform, so ``compute_loo_cv`` returns
    clean log-Y predictions that we map back via lognormal moments.

    Day-zero anchors are dropped (log(0) is undefined); the GP no longer
    enforces strength → 0 as t → 0.
    """
    torch.manual_seed(seed)

    keep = (Y.squeeze() if Y.dim() > 1 else Y) >= 10.0
    X = X[keep]
    Y = Y[keep] if Y.dim() == 1 else Y[keep]
    Yvar = Yvar[keep]
    Y_psi = Y.clone()
    Y = Y.log() if Y.dim() == 1 else Y.log()

    n_real = X.shape[0]
    d_in = X.shape[-1]
    d_out = 1 if Y.dim() == 1 else Y.shape[-1]
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)
        Y_psi = Y_psi.unsqueeze(-1)

    base_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=d_in,
        lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        lengthscale_prior=_within_group_prior(d_in),
    )
    scaled_base = ScaleKernel(
        base_kernel,
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
    )
    time_kernel = ScaleKernel(
        RBFKernel(
            active_dims=torch.tensor([d_in - 1]),
            ard_num_dims=1,
            lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        ),
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
    )
    kernel = scaled_base + time_kernel

    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        covar_module=kernel,
        input_transform=get_strength_gp_input_transform(d=d_in, bounds=bounds),
        outcome_transform=Standardize(d_out),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    # Stash the original psi observations on the model for the metrics path.
    model._study_Y_psi = Y_psi.squeeze(-1)
    return model, n_real


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def fit_variant_a_plus_c(X, Y, Yvar, bounds, seed):
    """Stack A + C — log-Y outcome transform with heteroscedastic noise.

    Combines:
      A: train_Yvar populated with empirical Strength (Std)², transformed to
         log-psi space via the delta method: Var(log Y) ≈ Var(Y) / Y².
      C: pre-log Y outside the model; Standardize is the lone outcome
         transform; rows with Y < 10 psi filtered (lab anomalies).

    The expected synergy: C makes residuals more Gaussian (good calibration)
    while A pushes the GP to weight per-observation precision (good RMSE).
    """
    torch.manual_seed(seed)

    keep = (Y.squeeze() if Y.dim() > 1 else Y) >= 10.0
    X = X[keep]
    Y = Y[keep] if Y.dim() == 1 else Y[keep]
    Yvar = Yvar[keep]
    Y_psi = Y.clone()

    # Exact lognormal-noise transform (preserves accuracy for low-Y rows where
    # the delta-method approximation Var(Y)/Y² becomes ill-conditioned).
    #
    #   Y ~ Lognormal(μ, σ²) with E[Y]=m, Var[Y]=v
    #     ⇒ σ² = log(1 + v/m²)
    #
    # Floor Yvar at (5 psi)² so the 2 zero-empirical-std rows don't blow up.
    Yvar_psi_floored = torch.maximum(Yvar, torch.full_like(Yvar, 25.0))
    Y_squared = (Y if Y.dim() == 1 else Y.squeeze(-1)).pow(2)
    ratio = (
        Yvar_psi_floored / Y_squared.unsqueeze(-1)
        if Yvar.dim() > 1
        else Yvar_psi_floored / Y_squared
    )
    Yvar_log = torch.log1p(ratio)

    Y = Y.log() if Y.dim() == 1 else Y.log()

    n_real = X.shape[0]
    d_in = X.shape[-1]
    d_out = 1 if Y.dim() == 1 else Y.shape[-1]
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)
        Y_psi = Y_psi.unsqueeze(-1)
    if Yvar_log.dim() == 1:
        Yvar_log = Yvar_log.unsqueeze(-1)

    base_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=d_in,
        lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        lengthscale_prior=_within_group_prior(d_in),
    )
    scaled_base = ScaleKernel(
        base_kernel,
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
    )
    time_kernel = ScaleKernel(
        RBFKernel(
            active_dims=torch.tensor([d_in - 1]),
            ard_num_dims=1,
            lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        ),
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
    )
    kernel = scaled_base + time_kernel

    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        train_Yvar=Yvar_log,
        covar_module=kernel,
        input_transform=get_strength_gp_input_transform(d=d_in, bounds=bounds),
        outcome_transform=Standardize(d_out),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model._study_Y_psi = Y_psi.squeeze(-1)
    return model, n_real


def fit_variant_a_plus_b(X, Y, Yvar, bounds, seed):
    """Stack A + B — source-aware kernel with heteroscedastic noise.

    Hypothesis: A's per-row empirical noise stops the optimiser from
    attributing residual variance to B's source-specific kernel (which is
    what causes B alone to over-fit at n=647).
    """
    torch.manual_seed(seed)
    n_real = X.shape[0]
    d_in = X.shape[-1]
    d_out = 1 if Y.dim() == 1 else Y.shape[-1]
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)

    floor = torch.full_like(Yvar, 25.0)
    Yvar_floored = torch.maximum(Yvar, floor)

    X_0, Y_0, Yvar_0 = get_day_zero_data(X=X, n=128)
    X = torch.cat((X, X_0), dim=0)
    Y = torch.cat((Y, Y_0), dim=0)
    Yvar_floored = torch.cat((Yvar_floored, Yvar_0), dim=0)

    kernel = _build_source_aware_kernel(d_in=d_in, source_dim=_SOURCE_DIM)
    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        train_Yvar=Yvar_floored,
        covar_module=kernel,
        input_transform=get_strength_gp_input_transform(d=d_in, bounds=bounds),
        outcome_transform=Standardize(d_out),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model, n_real


def _build_source_aware_kernel_shared_ls(d_in: int, source_dim: int) -> torch.nn.Module:
    """B' variant: source-aware kernel with **shared** ARD lengthscales.

    The kernel is::

        K(x_i, x_j) = m(x_i^{ns}, x_j^{ns}) · ( σ²_shared + σ²_src · δ(s_i, s_j) )

    where m(·,·) is a single Matern-5/2 over the 9 non-source dimensions, and
    σ²_shared, σ²_src are independent ScaleKernel outputscales. The lengthscales
    of m are shared via Python reference, so:
      * The within-group shrinkage prior on the shared Matern automatically
        applies symmetrically to both parts of the decomposition.
      * Hyperparameter count drops from 18 ARD lengthscales (vanilla B) to 9.
    """
    no_source_dims = [i for i in range(d_in) if i != source_dim]

    shared_matern = MaternKernel(
        nu=2.5,
        ard_num_dims=len(no_source_dims),
        active_dims=torch.tensor(no_source_dims),
        lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        lengthscale_prior=_within_group_prior(d_in=d_in, source_dim=source_dim),
    )

    shared_part = ScaleKernel(
        shared_matern,
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
    )

    source_index = IndexKernel(
        num_tasks=NUM_MATERIAL_CLASSES,
        rank=0,
        active_dims=torch.tensor([source_dim]),
    )
    # The same shared_matern instance is referenced inside this product kernel,
    # so PyTorch's parameter accounting wires its lengthscales as a shared
    # parameter. Gradient updates from both branches accumulate correctly.
    source_specific_part = ScaleKernel(
        shared_matern * source_index,
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=0.1),
    )

    time_kernel = ScaleKernel(
        RBFKernel(
            active_dims=torch.tensor([d_in - 1]),
            ard_num_dims=1,
            lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        ),
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
    )
    return shared_part + source_specific_part + time_kernel


def fit_variant_b_prime(X, Y, Yvar, bounds, seed):
    """Variant B' — source-aware kernel with shared ARD lengthscales.

    See ``_build_source_aware_kernel_shared_ls`` for the kernel definition.
    Uses learned scalar noise (no heteroscedastic component) so we can isolate
    the effect of the shared-lengthscale parameterisation.
    """
    torch.manual_seed(seed)
    n_real = X.shape[0]
    d_in = X.shape[-1]
    d_out = 1 if Y.dim() == 1 else Y.shape[-1]
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)

    X_0, Y_0, Yvar_0 = get_day_zero_data(X=X, n=128)
    n_pseudo = X_0.shape[0]
    X = torch.cat((X, X_0), dim=0)
    Y = torch.cat((Y, Y_0), dim=0)

    kernel = _build_source_aware_kernel_shared_ls(d_in=d_in, source_dim=_SOURCE_DIM)
    likelihood = PartialFixedNoiseLikelihood(
        n_real=n_real,
        n_pseudo=n_pseudo,
        pseudo_noise=1e-6,
        noise_constraint=LogTransformedInterval(1e-6, 1.0, initial_value=1e-1),
    )
    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        covar_module=kernel,
        likelihood=likelihood,
        input_transform=get_strength_gp_input_transform(d=d_in, bounds=bounds),
        outcome_transform=Standardize(d_out),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model, n_real


def fit_variant_a_plus_b_prime(X, Y, Yvar, bounds, seed):
    """Stack A + B' — heteroscedastic noise + shared-lengthscale source-aware kernel."""
    torch.manual_seed(seed)
    n_real = X.shape[0]
    d_in = X.shape[-1]
    d_out = 1 if Y.dim() == 1 else Y.shape[-1]
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)

    floor = torch.full_like(Yvar, 25.0)
    Yvar_floored = torch.maximum(Yvar, floor)

    X_0, Y_0, Yvar_0 = get_day_zero_data(X=X, n=128)
    X = torch.cat((X, X_0), dim=0)
    Y = torch.cat((Y, Y_0), dim=0)
    Yvar_floored = torch.cat((Yvar_floored, Yvar_0), dim=0)

    kernel = _build_source_aware_kernel_shared_ls(d_in=d_in, source_dim=_SOURCE_DIM)
    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        train_Yvar=Yvar_floored,
        covar_module=kernel,
        input_transform=get_strength_gp_input_transform(d=d_in, bounds=bounds),
        outcome_transform=Standardize(d_out),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model, n_real


# V2 building blocks now live in boxcrete.strength_v2 (the canonical
# production home). Re-import them here so the variant catalog can
# compose them into research variants without duplicating their
# definitions. The research-only builders moved to
# ``experiments/_research_features.py``; the union below mirrors the
# pre-split ``_FEATURE_BUILDERS`` contents (production + research) so
# the variant catalog can continue composing them transparently. After
# Commit 2 trims ``boxcrete.strength_v2._FEATURE_BUILDERS`` to the
# 7-feature production set, the union below is the only place where
# production + research builders coexist.
# V2 building blocks. Originally lived under ``boxcrete.strength_v2`` with
# leading underscores; the V2 split moved them to public surfaces in
# ``boxcrete.features`` / ``boxcrete.kernels`` / ``boxcrete.likelihoods`` /
# ``boxcrete.priors``. We alias them back to the underscored names so the
# ~4700 lines of variant-catalog code below don't need a wholesale rename.
from boxcrete.features import (  # noqa: E402
    AppendDerivedFeatures as _AppendEngineeredFeatures,
    FEATURE_BUILDERS as _PROD_FEATURE_BUILDERS,
    IDX as _IDX,
    augmented_bounds as _augmented_bounds,
    max_scale_Y as _max_scale_Y,
)
from boxcrete.kernels import (  # noqa: E402
    TimeGatedKernel as _TimeGatedKernelBase,
    additive_time_kernel as _additive_time_kernel,
    ard_matern_with_within_group_prior as _ard_matern_with_within_group_prior,
    build_strength_kernel_for_aug_dim as _build_b_double_prime_kernel_for_aug_dim,
    make_gated_strength_kernel_builder as _make_gated_strength_kernel_builder_new,
)


def _TimeGatedKernel(*args, gate_learnable: bool = False, **kwargs):
    """Compat shim around :class:`boxcrete.kernels.TimeGatedKernel`.

    The historical ``gate_learnable`` constructor kwarg used by a small
    set of research variants in this catalog is unsupported on the V2
    production class (a learnable ``tau`` was explored during V2
    development but produced no measurable block-LOO RMSE improvement
    and destabilised L-BFGS-B; the buffer path is the only one we
    ship). All current call sites pass ``gate_learnable=False`` which
    is a no-op; we strip the kwarg here so the unmodified call sites
    keep working.
    """
    if gate_learnable:
        import warnings

        warnings.warn(
            "_TimeGatedKernel(gate_learnable=True) is no longer supported on "
            "the V2 production class; falling back to the buffered tau path.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _TimeGatedKernelBase(*args, **kwargs)


from boxcrete.likelihoods import (  # noqa: E402
    GatedGaussianLikelihood as _GatedGaussianLikelihood,
)
from boxcrete.priors import (  # noqa: E402
    within_group_prior as _within_group_prior,
)
from experiments._research_features import (  # noqa: E402
    RESEARCH_FEATURE_BUILDERS as _RESEARCH_FEATURE_BUILDERS,
)


def _make_b_double_prime_time_gated_builder(
    gate_tau: float = 0.05, gate_learnable: bool = False
):
    """Compat shim around the V2 production
    :func:`boxcrete.kernels.make_gated_strength_kernel_builder`.

    Accepts the historical ``gate_learnable`` kwarg used by a small set of
    research variants in this catalog (``B''+...+gated_t_learn`` etc.).
    The production V2 fit pinned ``gate_learnable=False`` (a learnable
    ``tau`` was explored during V2 development but produced no measurable
    block-LOO RMSE improvement and destabilised L-BFGS-B; the buffer path
    is the only one we ship). When ``gate_learnable=True`` is requested
    here we emit a one-shot warning and fall back to the buffered path,
    so the affected legacy research variants register but execute the
    production buffered tau (their results in this catalog are now
    aliases of the ``gated_t`` variant — flagged in any benchmark dump).
    """
    if gate_learnable and not getattr(
        _make_b_double_prime_time_gated_builder, "_warned", False
    ):
        import warnings

        warnings.warn(
            "gate_learnable=True is no longer supported on the V2 "
            "production builder; falling back to the buffered tau path. "
            "Affected research variants now alias the gated_t variant.",
            RuntimeWarning,
            stacklevel=2,
        )
        _make_b_double_prime_time_gated_builder._warned = True  # type: ignore[attr-defined]
    return _make_gated_strength_kernel_builder_new(gate_tau=gate_tau)


class _LearnableLogTimeTransform:
    """Compat stub for the legacy research-only learnable-log-time
    input transform (originally lived in ``boxcrete.strength_v2`` and
    used by a small set of research variants:
    ``B''+...+gated_learnable_t``). The class was not preserved through
    the V2 module split because the production fit pinned the buffered
    log-time transform; instantiation here raises ``NotImplementedError``
    so the affected research variants surface clearly rather than
    silently degrading to a different transform.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "_LearnableLogTimeTransform is a legacy research class that "
            "was not preserved through the V2 module split. The "
            "production fit uses the buffered log-time transform "
            "(see boxcrete/strength_model.py)."
        )


# Variant catalog uses the union. After Commit 2 trims
# ``boxcrete.strength_v2._FEATURE_BUILDERS`` to the 7-feature production
# set, this dict is the single source of truth for "all known feature
# builders", containing 7 production + 11 research entries.
_FEATURE_BUILDERS = {**_PROD_FEATURE_BUILDERS, **_RESEARCH_FEATURE_BUILDERS}


# ---------------------------------------------------------------------------
# Variant catalog navigation (audit 2026-05-19: all 258 entries pass on n=20).
# ---------------------------------------------------------------------------
# Variant keys are constructed by composing tags separated by ``+``. The
# left-most tag is the kernel family; subsequent tags refine features,
# noise model, or HP-fit objective. The deployed champion is
#   ``B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean``
# (see ``boxcrete/strength_v2.py::_CHAMPION_VARIANT`` and
# ``STRENGTH_GP_BENCHMARK.md`` §6.12 for the per-tag rationale).
#
# Tag glossary (see ``STRENGTH_GP_BENCHMARK.md`` for full per-tag results):
#   - kernel family: ``A`` ``B`` ``B'`` ``B''`` ``C`` ``D`` ``E1`` ``E2``
#     ``OAK1`` ``OAK2`` ``single_matern``
#   - features: ``F0`` ``F1`` … ``F5`` ``F5_alllog`` ``F5_alllog_steepmat``
#   - mean / scaling: ``maxscale_zeromean`` ``floor`` ``mult`` ``full``
#   - time gate: ``gated_t`` ``gated_t_learn`` ``gated_t_tau0.2``
#     ``gated_learnable_t`` ``no_time_kernel``
#   - noise: ``gated_noise`` ``per_source_noise`` ``learnable_per_source_noise``
#     ``fixed_noise_avg``
#   - HP fit objective: ``block_loo_only`` (block-LOO from random init) vs.
#     ``block_loo_refine`` (MLL warm-start + block-LOO refinement)
#   - misc / anchors: ``anchors`` ``learnoff`` ``no_prior`` ``time_matern``
#
# To audit catalog health, run ``experiments/_audit_variants.py`` (one-off,
# not committed).
VARIANTS = {
    "baseline": fit_baseline,
    "baseline_no_prior": fit_baseline_no_prior,
    "A": fit_variant_a,
    "B": fit_variant_b,
    "B'": fit_variant_b_prime,
    "C": fit_variant_c,
    "A+B": fit_variant_a_plus_b,
    "A+B'": fit_variant_a_plus_b_prime,
    "A+C": fit_variant_a_plus_c,
    # See section "Multi-task kernel grid" below.
    "B''": None,  # filled in after function definitions
    "A+B''": None,
    "D": None,
    "A+D": None,
    "E1": None,
    "A+E1": None,
    "E2": None,
    "A+E2": None,
}


# ---------------------------------------------------------------------------
# Multi-task kernel grid (B'', D, E1, E2)
#
# A 2x2 ablation factoring "with vs without explicit source-blind component"
# against "radial joint-ARD vs factored Matern x Hamming-RBF" parameterisation
# of the source-aware piece. See the design discussion in scripts docstring.
# ---------------------------------------------------------------------------


def _make_hamming_rbf(source_dim: int) -> RBFKernel:
    """Hamming-distance RBF on a single (binary) categorical feature.

    For binary `Material Source in {0, 1}`, squared Euclidean distance equals
    Hamming distance, so a 1-D RBF on the source dim is exactly Hamming-RBF
    with a learnable single lengthscale. Generalises to C > 2 if the column
    is one-hot encoded.
    """
    return RBFKernel(
        ard_num_dims=1,
        active_dims=torch.tensor([source_dim]),
        lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
    )


def _additive_time_matern_kernel(d_in: int) -> ScaleKernel:
    """Standalone additive Matern-5/2 on time (last dim). Alternative to
    `_additive_time_kernel` (which uses RBF). Matern-5/2 is less smooth at
    its mode than RBF; whether this helps depends on the data — concrete
    strength curves are smooth enough that RBF generally wins, but with the
    gated kernel modifying the kernel structurally near t=0, the question
    is worth re-asking."""
    return ScaleKernel(
        MaternKernel(
            nu=2.5,
            active_dims=torch.tensor([d_in - 1]),
            ard_num_dims=1,
            lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        ),
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
    )


def _build_b_double_prime_kernel(d_in: int, source_dim: int) -> torch.nn.Module:
    """B'' — radial joint-ARD source-aware kernel, with explicit blind term.

    K = M_blind(x_no_source) + M_specific(x_all_dims) + R_t(t)

    M_specific has source as one of its ARD dims; cross-source covariance is
    smoothly attenuated by ell_specific[source_dim]. Independent within-group
    priors on the two Materns.
    """
    no_source_dims = [i for i in range(d_in) if i != source_dim]
    all_dims = list(range(d_in))
    blind = _ard_matern_with_within_group_prior(
        ard_num_dims=len(no_source_dims),
        active_dims=torch.tensor(no_source_dims),
        prior=_within_group_prior(d_in=d_in, source_dim=source_dim),
        initial_outputscale=1.0,
    )
    specific = _ard_matern_with_within_group_prior(
        ard_num_dims=d_in,
        active_dims=torch.tensor(all_dims),
        prior=_within_group_prior(d_in),
        initial_outputscale=0.5,
    )
    return blind + specific + _additive_time_kernel(d_in)


def _build_d_kernel(d_in: int, source_dim: int) -> torch.nn.Module:
    """D — two unconstrained joint-ARD Materns over all dims, no blind term.

    K = M_1(x_all_dims) + M_2(x_all_dims) + R_t(t)

    Both Materns see all 10 dims (including source). Independent ARD ell
    vectors and independent within-group priors on each Matern. The optimiser
    decides whether to specialise one Matern to short-range / source-specific
    structure and one to long-range / shared physics.
    """
    del source_dim  # kept for sibling-builder API parity; not used here
    all_dims = list(range(d_in))
    m1 = _ard_matern_with_within_group_prior(
        ard_num_dims=d_in,
        active_dims=torch.tensor(all_dims),
        prior=_within_group_prior(d_in),
        initial_outputscale=1.0,
    )
    m2 = _ard_matern_with_within_group_prior(
        ard_num_dims=d_in,
        active_dims=torch.tensor(all_dims),
        prior=_within_group_prior(d_in),
        initial_outputscale=0.5,
    )
    return m1 + m2 + _additive_time_kernel(d_in)


def _build_e1_kernel(d_in: int, source_dim: int) -> torch.nn.Module:
    """E1 — blind + factored source-aware (Matern_no_source x Hamming-RBF).

    K = M_blind(x_no_source) + (M_specific(x_no_source) * k_cat(source)) + R_t(t)

    The factored form decouples continuous smoothness from categorical
    similarity: M_specific captures composition smoothness for the
    source-aware term, while k_cat (single-LS Hamming-RBF) controls how much
    that term transfers across sources.
    """
    no_source_dims = [i for i in range(d_in) if i != source_dim]
    blind = _ard_matern_with_within_group_prior(
        ard_num_dims=len(no_source_dims),
        active_dims=torch.tensor(no_source_dims),
        prior=_within_group_prior(d_in=d_in, source_dim=source_dim),
        initial_outputscale=1.0,
    )
    specific_base = MaternKernel(
        nu=2.5,
        ard_num_dims=len(no_source_dims),
        active_dims=torch.tensor(no_source_dims),
        lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        lengthscale_prior=_within_group_prior(d_in=d_in, source_dim=source_dim),
    )
    factored_specific = ScaleKernel(
        specific_base * _make_hamming_rbf(source_dim),
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=0.5),
    )
    return blind + factored_specific + _additive_time_kernel(d_in)


def _build_e2_kernel(d_in: int, source_dim: int) -> torch.nn.Module:
    """E2 — two factored components, no blind term.

    K = (M_1(x_no_source) * k_cat_1(source))
      + (M_2(x_no_source) * k_cat_2(source))
      + R_t(t)

    Two independent Materns on the non-source dims, each multiplied by an
    independent Hamming-RBF on source. The model can recover E1 by sending
    one ell_cat -> infinity (so its k_cat ~ 1 acts as a blind term), so E2
    is strictly more general than E1.
    """
    no_source_dims = [i for i in range(d_in) if i != source_dim]

    def _factored_branch(initial_outputscale: float) -> ScaleKernel:
        base = MaternKernel(
            nu=2.5,
            ard_num_dims=len(no_source_dims),
            active_dims=torch.tensor(no_source_dims),
            lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
            lengthscale_prior=_within_group_prior(d_in=d_in, source_dim=source_dim),
        )
        return ScaleKernel(
            base * _make_hamming_rbf(source_dim),
            outputscale_constraint=LogTransformedInterval(
                1e-2, 1e2, initial_value=initial_outputscale
            ),
        )

    return _factored_branch(1.0) + _factored_branch(0.5) + _additive_time_kernel(d_in)


def _fit_with_custom_kernel(
    kernel: torch.nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    Yvar: torch.Tensor,
    bounds: torch.Tensor,
    *,
    use_fixed_noise: bool,
) -> tuple[SingleTaskGP, int]:
    """Common fit path used by all multi-task-kernel variants.

    Replicates the production fit_strength_gp boilerplate (day-zero anchors,
    PartialFixedNoiseLikelihood vs FixedNoise) but accepts a pre-built kernel.
    Returns (fitted_model, n_real).
    """
    n_real = X.shape[0]
    d_in = X.shape[-1]
    d_out = 1 if Y.dim() == 1 else Y.shape[-1]
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)

    X_0, Y_0, Yvar_0 = get_day_zero_data(X=X, n=128)
    n_pseudo = X_0.shape[0]
    X_aug = torch.cat((X, X_0), dim=0)
    Y_aug = torch.cat((Y, Y_0), dim=0)

    if use_fixed_noise:
        floor = torch.full_like(Yvar, 25.0)  # (5 psi)^2 — handles 2 zero-Yvar rows
        Yvar_aug = torch.cat((torch.maximum(Yvar, floor), Yvar_0), dim=0)
        model = SingleTaskGP(
            train_X=X_aug,
            train_Y=Y_aug,
            train_Yvar=Yvar_aug,
            covar_module=kernel,
            input_transform=get_strength_gp_input_transform(d=d_in, bounds=bounds),
            outcome_transform=Standardize(d_out),
        )
    else:
        likelihood = PartialFixedNoiseLikelihood(
            n_real=n_real,
            n_pseudo=n_pseudo,
            pseudo_noise=1e-6,
            noise_constraint=LogTransformedInterval(1e-6, 1.0, initial_value=1e-1),
        )
        model = SingleTaskGP(
            train_X=X_aug,
            train_Y=Y_aug,
            covar_module=kernel,
            likelihood=likelihood,
            input_transform=get_strength_gp_input_transform(d=d_in, bounds=bounds),
            outcome_transform=Standardize(d_out),
        )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model, n_real


def _make_grid_fit(kernel_builder, *, use_fixed_noise: bool):
    """Closure factory that produces a fit-adapter compatible with the
    VARIANTS dispatch signature ``(X, Y, Yvar, bounds, seed) -> (model, n_real)``."""

    def _fit(X, Y, Yvar, bounds, seed):
        torch.manual_seed(seed)
        kernel = kernel_builder(X.shape[-1], _SOURCE_DIM)
        return _fit_with_custom_kernel(
            kernel,
            X,
            Y,
            Yvar,
            bounds,
            use_fixed_noise=use_fixed_noise,
        )

    return _fit


# Wire the new variants into the dispatch table.
VARIANTS["B''"] = _make_grid_fit(_build_b_double_prime_kernel, use_fixed_noise=False)
VARIANTS["A+B''"] = _make_grid_fit(_build_b_double_prime_kernel, use_fixed_noise=True)
VARIANTS["D"] = _make_grid_fit(_build_d_kernel, use_fixed_noise=False)
VARIANTS["A+D"] = _make_grid_fit(_build_d_kernel, use_fixed_noise=True)
VARIANTS["E1"] = _make_grid_fit(_build_e1_kernel, use_fixed_noise=False)
VARIANTS["A+E1"] = _make_grid_fit(_build_e1_kernel, use_fixed_noise=True)
VARIANTS["E2"] = _make_grid_fit(_build_e2_kernel, use_fixed_noise=False)
VARIANTS["A+E2"] = _make_grid_fit(_build_e2_kernel, use_fixed_noise=True)


# ---------------------------------------------------------------------------
# Noise-floor variants (Section 6 of the benchmark study).
#
# Empirical Yvar undercounts true noise (triplicates miss batch / lab-day /
# instrument drift / model-misspecification noise). Three parameterisations
# of the missing noise component:
#
#   additive:        K_noise = diag(Yvar) + sigma_g^2 * I    (constant absolute)
#   multiplicative:  K_noise = c * diag(Yvar)                (constant fractional)
#   full:            K_noise = c * diag(Yvar) + sigma_g^2 * I  (both)
#
# All operate on top of the A+B'' kernel; the additive form is also tested
# on A+B for comparison.
# ---------------------------------------------------------------------------


from gpytorch.likelihoods import _GaussianLikelihoodBase  # noqa: E402
from gpytorch.likelihoods.noise_models import (  # noqa: E402
    FixedGaussianNoise,
    HomoskedasticNoise,
)


class _PerSourceGaussianLikelihood(_GaussianLikelihoodBase):
    """Learnable per-source homoscedastic noise.

    Two (or more) learnable noise variances, one per source. Reads the
    source dim from the GP's input X at runtime and applies the matching
    noise per row. The motivation is the documented 2x heterogeneity
    between Source 0 (high-cement, ~163 psi residual std) and Source 1
    (low-cement, ~76 psi). A single learnable global noise can only
    average these; per-source noise lets each source converge to its
    own appropriate level while still letting the optimiser absorb
    model-misspecification slack (the failure mode of the FIXED
    per-source noise variant — see benchmark §6.10).

    Implementation: use a vanilla (non-batched) HomoskedasticNoise as the
    inherited ``noise_covar`` placeholder so BoTorch's SingleTaskGP
    doesn't see a batched noise model (which propagates batch dims into
    downstream matmuls and breaks the 2-D shape contract). The actual
    per-source learnable noise parameters are registered separately on
    this class as ``raw_per_source_noise``, with the constraint stored
    as ``per_source_noise_constraint``. ``_shaped_noise_covar`` reads
    these to compute per-row noise from the input X.
    """

    def __init__(
        self,
        n_sources: int = 2,
        source_dim: int = 7,
        noise_constraint=None,
        noise_prior=None,
        **kwargs,
    ):
        if noise_constraint is None:
            noise_constraint = LogTransformedInterval(
                1e-6,
                1.0,
                initial_value=1e-1,
            )
        # Non-batched placeholder so SingleTaskGP doesn't think we're
        # a batched model. We forward `noise_prior` here for API symmetry
        # with GaussianLikelihood; it has no effect on the actual
        # per-source noise (which lives on `raw_per_source_noise` and
        # has no prior in the current implementation).
        placeholder_noise_covar = HomoskedasticNoise(
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
        )
        super().__init__(noise_covar=placeholder_noise_covar)
        self.n_sources = int(n_sources)
        self.source_dim = int(source_dim)
        # The placeholder ``noise_covar.raw_noise`` is registered as a
        # parameter (inherited from HomoskedasticNoise) but is never used
        # by our overridden ``_shaped_noise_covar``. Freeze it so
        # ``fit_gpytorch_mll`` doesn't waste optimisation budget on a
        # redundant parameter that has no effect on the loss.
        self.noise_covar.raw_noise.requires_grad_(False)
        # Our own learnable per-source noise parameter (a 1-D vector
        # of length n_sources). Initialise to a SMALL value (1e-3 in
        # scaled space ≈ noise floor in psi for typical y_max ~16k psi).
        # Empirically this initial point gives more reliable convergence
        # than the constraint's `initial_value=0.1` (which pushed the
        # optimiser into a high-noise local optimum). The optimiser is
        # still free to grow noise as needed.
        small_init_value = 1e-3
        if hasattr(noise_constraint, "inverse_transform"):
            init_raw = float(
                noise_constraint.inverse_transform(
                    torch.tensor(small_init_value, dtype=torch.double)
                )
            )
        else:
            init_raw = -6.9
        self.register_parameter(
            "raw_per_source_noise",
            torch.nn.Parameter(torch.full((n_sources,), init_raw, dtype=torch.double)),
        )
        self.per_source_noise_constraint = noise_constraint

    @property
    def per_source_noise(self) -> torch.Tensor:
        """Constrained per-source noise variances, shape ``[n_sources]``."""
        return self.per_source_noise_constraint.transform(self.raw_per_source_noise)

    def set_train_sources(self, source_values: torch.Tensor) -> None:
        """Stash per-row source indices (1-D long tensor) for use during
        marginal-likelihood computation, when GPyTorch calls
        ``_shaped_noise_covar`` without passing X. Must be called before
        ``fit_gpytorch_mll`` for the noise to be applied per-row during
        training.
        """
        self._train_sources = (
            source_values.detach().round().long().clamp(0, self.n_sources - 1)
        )

    def _shaped_noise_covar(self, base_shape, *params, **kwargs):
        from linear_operator.operators import DiagLinearOperator

        # Determine source assignment per row.
        if params and hasattr(params[0], "shape") and params[0].dim() >= 2:
            # Prediction-time path: read source from input X.
            x = params[0]
            sources = (
                x[..., self.source_dim].round().long().clamp(0, self.n_sources - 1)
            )
        elif getattr(self, "_train_sources", None) is not None:
            # Training-time path: GPyTorch's mll() doesn't pass X to the
            # likelihood, so we read from the stash.
            sources = self._train_sources
        else:
            return super()._shaped_noise_covar(base_shape, *params, **kwargs)
        noise_per_source = self.per_source_noise  # shape [n_sources], differentiable
        per_row_noise = noise_per_source[sources.flatten()]
        n = int(base_shape[-1])
        if per_row_noise.shape[0] >= n:
            per_row_noise = per_row_noise[:n]
        else:
            pad = noise_per_source[0].expand(n - per_row_noise.shape[0])
            per_row_noise = torch.cat([per_row_noise, pad], dim=0)
        return DiagLinearOperator(per_row_noise)


class _PowersLawGatedMean(torch.nn.Module):
    """Powers'-law-style parametric mean function for concrete strength.

    Returns ``μ(x, t) = α · h(t)`` where:
    - ``α`` is a learnable scalar (the asymptotic strength as t → ∞,
      in the GP's outcome-scaled space).
    - ``h(t) = 1 - exp(-t / tau_mean)`` matches the gated-kernel
      transition function shape — vanishes at t=0 (preserving the
      physics constraint), saturates as t grows.

    This gives the GP a strong physics-aware prior trend; it then fits
    composition-specific deviations as residuals around this curve.
    The single ``α`` parameter (no per-composition variation) is the
    simplest possible Powers'-law mean. Per-composition extensions
    (α(x) = a₀ + a₁·W/B + ...) are higher-parameter variants that
    history suggests will overfit, so we start with the simplest form.

    Critical for the gated kernel: ``μ(x, 0) = α · 0 = 0`` exactly,
    preserving the structural physics constraint.
    """

    def __init__(self, tau_mean: float = 0.5, init_alpha: float = 0.4):
        """
        Args:
            tau_mean: timescale of the mean's saturation curve (in
                post-input-transform time units; default 0.5 means
                significant strength growth in roughly the first half
                of the time range).
            init_alpha: initial asymptote in scaled space (Y / y_max).
                Default 0.4 ≈ "average mature strength is ~40% of the
                strongest mix in the dataset".
        """
        super().__init__()
        self.register_parameter(
            "alpha",
            torch.nn.Parameter(torch.tensor(init_alpha, dtype=torch.double)),
        )
        # tau_mean is fixed, not learnable, to avoid the "extra
        # learnable parameter overfits" pattern that has plagued every
        # other flexibility addition in this study (anchor study §3.8,
        # benchmark §6.10). We pick a sensible default and trust the
        # GP residual to adapt around it.
        self.register_buffer("tau_mean", torch.tensor(tau_mean, dtype=torch.double))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape [..., n, d]; return shape [..., n]
        # Time dim is at index 9 in the post-input-transform space.
        t = x[..., 9]
        h = 1.0 - torch.exp(-t.clamp_min(0.0) / self.tau_mean.to(t))
        return self.alpha.to(x) * h


class FixedNoiseAdditiveLikelihood(_GaussianLikelihoodBase):
    """Heteroscedastic fixed noise + a learned global additive floor.

    K_noise = diag(train_Yvar) + sigma_g^2 * I

    The learned floor sigma_g captures noise sources missing from the
    triplicate-derived Yvar — batch drift, lab-day variation, instrument
    drift, model misspecification. At test points the per-row diagonal
    is empty (no known Yvar), so test-point noise inherits only sigma_g.
    """

    def __init__(self, noise: torch.Tensor, *, floor_initial: float = 1e-2):
        super().__init__(noise_covar=FixedGaussianNoise(noise=noise))
        self._floor_covar = HomoskedasticNoise(
            noise_constraint=LogTransformedInterval(
                1e-6, 1.0, initial_value=floor_initial
            ),
            batch_shape=torch.Size(),
        )

    def _shaped_noise_covar(self, base_shape, *params, **kwargs):
        fixed = self.noise_covar(*params, shape=base_shape, **kwargs)
        floor = self._floor_covar(*params, shape=base_shape, **kwargs)
        return fixed + floor


class FixedNoiseMultiplicativeLikelihood(_GaussianLikelihoodBase):
    """Heteroscedastic fixed noise scaled by a learned global coefficient.

    K_noise = c * diag(train_Yvar)

    Useful when triplicates are within-batch correlated and systematically
    under-report true variance by a constant fractional factor.
    """

    def __init__(self, noise: torch.Tensor, *, scale_initial: float = 1.0):
        super().__init__(noise_covar=FixedGaussianNoise(noise=noise))
        self._scale_covar = HomoskedasticNoise(
            noise_constraint=LogTransformedInterval(
                0.1, 100.0, initial_value=scale_initial
            ),
            batch_shape=torch.Size(),
        )

    def _shaped_noise_covar(self, base_shape, *params, **kwargs):
        fixed = self.noise_covar(*params, shape=base_shape, **kwargs)
        c = self._scale_covar.noise.squeeze()
        # FixedGaussianNoise returns a DiagLinearOperator; multiplying by a
        # positive scalar is an element-wise rescale of the diagonal.
        return fixed * c


class FixedNoiseFullCalibrationLikelihood(_GaussianLikelihoodBase):
    """Combined: K_noise = c * diag(train_Yvar) + sigma_g^2 * I.

    The fully-flexible noise model. With weak priors anchoring c near 1 and
    sigma_g near a small fraction of median Y, this is the principled answer
    when neither pure additive nor pure multiplicative captures the missing
    noise structure alone.
    """

    def __init__(
        self,
        noise: torch.Tensor,
        *,
        floor_initial: float = 1e-2,
        scale_initial: float = 1.0,
    ):
        super().__init__(noise_covar=FixedGaussianNoise(noise=noise))
        self._floor_covar = HomoskedasticNoise(
            noise_constraint=LogTransformedInterval(
                1e-6, 1.0, initial_value=floor_initial
            ),
            batch_shape=torch.Size(),
        )
        self._scale_covar = HomoskedasticNoise(
            noise_constraint=LogTransformedInterval(
                0.1, 100.0, initial_value=scale_initial
            ),
            batch_shape=torch.Size(),
        )

    def _shaped_noise_covar(self, base_shape, *params, **kwargs):
        fixed = self.noise_covar(*params, shape=base_shape, **kwargs)
        floor = self._floor_covar(*params, shape=base_shape, **kwargs)
        c = self._scale_covar.noise.squeeze()
        return fixed * c + floor


def _fit_with_noise_floor(
    kernel: torch.nn.Module,
    likelihood_factory,
    X: torch.Tensor,
    Y: torch.Tensor,
    Yvar: torch.Tensor,
    bounds: torch.Tensor,
) -> tuple[SingleTaskGP, int]:
    """Fit path for the noise-floor variants.

    Custom likelihoods (additive / multiplicative / full) operate on the
    likelihood-noise diagonal directly. BoTorch only auto-rescales Yvar when
    you use its built-in ``train_Yvar`` path with the default
    ``FixedNoiseGaussianLikelihood``; with a custom likelihood we have to
    pre-standardise Y and Yvar ourselves and skip the outcome_transform.
    The resulting LOO outputs come back in standardised space; the metrics
    code (``loo_metrics``) detects this via ``model._study_y_std`` and
    untransforms.

    Day-zero anchors are NOT appended: the homoscedastic floor parameter
    would otherwise drown the near-zero pseudo-observation noise.
    """
    n_real = X.shape[0]
    d_in = X.shape[-1]
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)
    if Yvar.dim() == 1:
        Yvar = Yvar.unsqueeze(-1)

    floor = torch.full_like(Yvar, 25.0)  # (5 psi)^2
    Yvar_floored = torch.maximum(Yvar, floor)

    Y_z, y_mean, y_std = _standardize_Y(Y)
    Yvar_z = Yvar_floored / (y_std**2)

    likelihood = likelihood_factory(Yvar_z.squeeze(-1))
    model = SingleTaskGP(
        train_X=X,
        train_Y=Y_z,
        covar_module=kernel,
        likelihood=likelihood,
        input_transform=get_strength_gp_input_transform(d=d_in, bounds=bounds),
        # NB: no outcome_transform — Y is already standardised in-line above.
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    # Stash for the metrics path to untransform LOO predictions back to psi.
    model._study_y_mean = y_mean.squeeze()
    model._study_y_std = y_std.squeeze()
    return model, n_real


def _make_floor_fit(kernel_builder, *, noise_factory):
    """Closure factory for noise-floor variants."""

    def _fit(X, Y, Yvar, bounds, seed):
        torch.manual_seed(seed)
        kernel = kernel_builder(X.shape[-1], _SOURCE_DIM)
        return _fit_with_noise_floor(kernel, noise_factory, X, Y, Yvar, bounds)

    return _fit


# Wire the noise-floor variants. Suffix conventions:
#   +floor : additive global noise floor (sigma_g^2 * I)
#   +mult  : multiplicative global scale (c * Yvar)
#   +full  : both
VARIANTS["A+B''+floor"] = _make_floor_fit(
    _build_b_double_prime_kernel,
    noise_factory=lambda yv: FixedNoiseAdditiveLikelihood(noise=yv),
)
VARIANTS["A+B''+mult"] = _make_floor_fit(
    _build_b_double_prime_kernel,
    noise_factory=lambda yv: FixedNoiseMultiplicativeLikelihood(noise=yv),
)
VARIANTS["A+B''+full"] = _make_floor_fit(
    _build_b_double_prime_kernel,
    noise_factory=lambda yv: FixedNoiseFullCalibrationLikelihood(noise=yv),
)
VARIANTS["A+B+floor"] = _make_floor_fit(
    _build_source_aware_kernel,
    noise_factory=lambda yv: FixedNoiseAdditiveLikelihood(noise=yv),
)


# ---------------------------------------------------------------------------
# OAK (Orthogonal Additive Kernel) variants — Section 7 of the benchmark study.
#
# OAK decomposes the function as
#
#   f(x) = c_0 + sum_i c_i * f_i(x_i) + sum_{i<j} c_{ij} * f_{ij}(x_i, x_j) + ...
#
# with orthogonality constraints between components for identifiability.
# Well-suited to data where many features have first-order effects and only
# a few have meaningful interactions.
# ---------------------------------------------------------------------------


from botorch.models.kernels.orthogonal_additive_kernel import (  # noqa: E402
    OrthogonalAdditiveKernel,
)
from botorch.models.transforms.input import (  # noqa: E402
    ChainedInputTransform,
    Normalize,
)


class _ClampedNormalize(Normalize):
    """Normalize with a post-transform clamp to [0, 1]. OAK requires strict
    hypercube inputs; without the clamp, float-precision overshoot at the
    bound (e.g. 1.0000001 for a max-bound input) trips OAK's input check.
    """

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return super().transform(X).clamp(0.0, 1.0)


def _get_oak_input_transform(d_in: int, bounds: torch.Tensor) -> _ClampedNormalize:
    """Plain Normalize + safety clamp — OAK requires strict [0,1]^d."""
    return _ClampedNormalize(d_in, bounds=bounds)


def _build_oak_kernel(
    d_in: int, source_dim: int, *, second_order: bool
) -> torch.nn.Module:
    """OAK as the only kernel — applied to all 10 dims (incl. time and source).

    Time is one of the 1st-order components; no separate additive time kernel.
    The base Matern is shared across dims (single lengthscale) — this is
    OAK's standard usage. The OAK coefficients give per-dim importance.
    """
    del source_dim  # kept for sibling-builder API parity; not used here
    base = MaternKernel(nu=2.5)
    return OrthogonalAdditiveKernel(
        base_kernel=base,
        dim=d_in,
        second_order=second_order,
        dtype=torch.float64,
    )


def _build_oak_plus_source_kernel(
    d_in: int, source_dim: int, *, second_order: bool
) -> torch.nn.Module:
    """OAK as a component + Matern x delta(source) source-aware + RBF time.

    Mirrors B's structure but replaces the source-blind Matern with OAK
    over all dims. Note: ``active_dims`` cannot be used to restrict OAK
    (it expects to see exactly ``dim`` dims), so OAK sees all dims.
    """
    no_source_dims = [i for i in range(d_in) if i != source_dim]

    oak_base = MaternKernel(nu=2.5)
    oak = OrthogonalAdditiveKernel(
        base_kernel=oak_base,
        dim=d_in,
        second_order=second_order,
        dtype=torch.float64,
    )

    source_specific_base = MaternKernel(
        nu=2.5,
        ard_num_dims=len(no_source_dims),
        active_dims=torch.tensor(no_source_dims),
        lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        lengthscale_prior=_within_group_prior(d_in=d_in, source_dim=source_dim),
    )
    source_index = IndexKernel(
        num_tasks=NUM_MATERIAL_CLASSES,
        rank=0,
        active_dims=torch.tensor([source_dim]),
    )
    source_specific = ScaleKernel(
        source_specific_base * source_index,
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=0.5),
    )

    return oak + source_specific + _additive_time_kernel(d_in)


def _fit_with_oak_kernel(
    kernel: torch.nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    Yvar: torch.Tensor,
    bounds: torch.Tensor,
    *,
    use_fixed_noise: bool,
) -> tuple[SingleTaskGP, int]:
    """Fit path using a Normalize+clamp input transform compatible with OAK.

    Mirrors ``_fit_with_custom_kernel`` but with two key differences:
    (1) Plain Normalize+clamp input transform (no AppendDerivedFeatures, since
        derived features may overshoot [0, 1]^d after Normalize and OAK
        strictly checks the hypercube).
    (2) **No day-zero anchors.** An additive model f(x, t) = f_x(x) + f_t(t)
        cannot satisfy f(*, t=0) = 0 for arbitrary compositions, so anchoring
        ``strength = 0 at t = 0`` would require f_x to be constant. Without
        anchors, OAK fits the joint surface in the time × composition space
        without that contradiction.
    """
    n_real = X.shape[0]
    d_in = X.shape[-1]
    d_out = 1 if Y.dim() == 1 else Y.shape[-1]
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)

    input_tf = _get_oak_input_transform(d_in, bounds)

    if use_fixed_noise:
        floor = torch.full_like(Yvar, 25.0)
        Yvar_floored = torch.maximum(Yvar, floor)
        model = SingleTaskGP(
            train_X=X,
            train_Y=Y,
            train_Yvar=Yvar_floored,
            covar_module=kernel,
            input_transform=input_tf,
            outcome_transform=Standardize(d_out),
        )
    else:
        # Plain learnable scalar Gaussian noise — no PartialFixedNoise since
        # there are no day-zero pseudo-observations to give zero noise.
        from gpytorch.likelihoods import GaussianLikelihood

        likelihood = GaussianLikelihood(
            noise_constraint=LogTransformedInterval(1e-6, 1.0, initial_value=1e-1),
        )
        model = SingleTaskGP(
            train_X=X,
            train_Y=Y,
            covar_module=kernel,
            likelihood=likelihood,
            input_transform=input_tf,
            outcome_transform=Standardize(d_out),
        )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model, n_real


def _make_oak_fit(kernel_builder, *, use_fixed_noise: bool):
    def _fit(X, Y, Yvar, bounds, seed):
        torch.manual_seed(seed)
        kernel = kernel_builder(X.shape[-1], _SOURCE_DIM)
        return _fit_with_oak_kernel(
            kernel,
            X,
            Y,
            Yvar,
            bounds,
            use_fixed_noise=use_fixed_noise,
        )

    return _fit


# Wire OAK variants
VARIANTS["OAK1"] = _make_oak_fit(
    lambda d, s: _build_oak_kernel(d, s, second_order=False),
    use_fixed_noise=False,
)
VARIANTS["A+OAK1"] = _make_oak_fit(
    lambda d, s: _build_oak_kernel(d, s, second_order=False),
    use_fixed_noise=True,
)
VARIANTS["OAK2"] = _make_oak_fit(
    lambda d, s: _build_oak_kernel(d, s, second_order=True),
    use_fixed_noise=False,
)
VARIANTS["A+OAK2"] = _make_oak_fit(
    lambda d, s: _build_oak_kernel(d, s, second_order=True),
    use_fixed_noise=True,
)


def _build_oak1_plus_b_double_prime_kernel(
    d_in: int,
    source_dim: int,
) -> torch.nn.Module:
    """1st-order OAK + B''-style blind + source-specific Materns (no time RBF).

    Kernel:
        K = OAK1(x_all_dims) + ScaleKernel(M_blind(x_no_source))
                              + ScaleKernel(M_specific(x_all_dims))

    Why no separate additive time kernel: OAK1 already includes time as one
    of its 1st-order univariate components, so an extra RBF on time would
    be redundant.

    Day-zero anchors ARE kept here (unlike in the OAK-alone variants):
    the multi-Matern part has enough capacity to satisfy f(·, 0) = 0,
    and OAK only needs to contribute small offsets in those rows.
    """
    no_source_dims = [i for i in range(d_in) if i != source_dim]

    oak_base = MaternKernel(nu=2.5)
    oak = OrthogonalAdditiveKernel(
        base_kernel=oak_base,
        dim=d_in,
        second_order=False,
        dtype=torch.float64,
    )

    blind = _ard_matern_with_within_group_prior(
        ard_num_dims=len(no_source_dims),
        active_dims=torch.tensor(no_source_dims),
        prior=_within_group_prior(d_in=d_in, source_dim=source_dim),
        initial_outputscale=1.0,
    )
    specific = _ard_matern_with_within_group_prior(
        ard_num_dims=d_in,
        active_dims=torch.tensor(list(range(d_in))),
        prior=_within_group_prior(d_in),
        initial_outputscale=0.5,
    )
    return oak + blind + specific


# Wire OAK1 + B'' variants
VARIANTS["OAK1+B''"] = _make_oak_fit(
    _build_oak1_plus_b_double_prime_kernel,
    use_fixed_noise=False,
)
VARIANTS["A+OAK1+B''"] = _make_oak_fit(
    _build_oak1_plus_b_double_prime_kernel,
    use_fixed_noise=True,
)


def _make_floor_oak_fit(kernel_builder, *, noise_factory):
    """Closure factory: noise-floor variant of an OAK-using kernel.

    Bypasses the day-zero anchor concatenation that the standard
    `_make_floor_fit` does (OAK-fit path needs the OAK input transform too).
    """

    def _fit(X, Y, Yvar, bounds, seed):
        torch.manual_seed(seed)
        kernel = kernel_builder(X.shape[-1], _SOURCE_DIM)
        n_real = X.shape[0]
        d_in = X.shape[-1]
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
        if Yvar.dim() == 1:
            Yvar = Yvar.unsqueeze(-1)
        floor = torch.full_like(Yvar, 25.0)
        Yvar_floored = torch.maximum(Yvar, floor)
        Y_z, y_mean, y_std = _standardize_Y(Y)
        Yvar_z = Yvar_floored / (y_std**2)
        likelihood = noise_factory(Yvar_z.squeeze(-1))
        input_tf = _get_oak_input_transform(d_in, bounds)
        model = SingleTaskGP(
            train_X=X,
            train_Y=Y_z,
            covar_module=kernel,
            likelihood=likelihood,
            input_transform=input_tf,
            # No outcome_transform — Y already standardised in-line.
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        model._study_y_mean = y_mean.squeeze()
        model._study_y_std = y_std.squeeze()
        return model, n_real

    return _fit


VARIANTS["A+OAK1+B''+floor"] = _make_floor_oak_fit(
    _build_oak1_plus_b_double_prime_kernel,
    noise_factory=lambda yv: FixedNoiseAdditiveLikelihood(noise=yv),
)


# ---------------------------------------------------------------------------
# Feature engineering — concrete-chemistry-informed derived features.
#
# DEFAULT_X_COLUMNS dim order:
#   0 Cement, 1 Fly Ash, 2 Slag, 3 Water, 4 HRWR,
#   5 Fine Aggregate, 6 Coarse Aggregates, 7 Material Source, 8 Temp, 9 Time
# ---------------------------------------------------------------------------

from botorch.models.transforms.input import (  # noqa: E402
    AffineInputTransform,
    ChainedInputTransform,
    Log10,
    Normalize,
)


def _get_engineered_input_transform(
    d_in: int,
    bounds: torch.Tensor,
    feature_names: list[str],
    X_for_bounds: torch.Tensor,
    log_time_offset: float = 1.0,
    learnable_log_offset: bool = False,
    skip_time_in_normalize: bool = False,
) -> ChainedInputTransform:
    """Chain: AppendEngineeredFeatures + log10(time + offset) + Normalize.

    The raw bounds are extended with empirically-derived bounds for each
    appended feature (computed on ``X_for_bounds``).

    ``log_time_offset`` controls how far apart the t=0 anchor and the t=1
    real measurements sit *after* the log transform. The default (1.0)
    matches the production `Log10(time + 1)` transform.

    ``learnable_log_offset``: if True, the offset becomes a learnable
    parameter. ``Normalize`` is then automatically applied only to the
    *non-time* dims (since Normalize's stored bounds would otherwise
    become stale as the optimiser moves ``offset``).

    ``skip_time_in_normalize``: if True (and ``learnable_log_offset`` is
    False), uses the same no-time-Normalize code path as the learnable
    case but with a *fixed* offset. Used as a control in the Phase 0
    diagnostic to isolate "does removing Normalize on time hurt?" from
    "does learning the offset hurt?".
    """
    augmented = _augmented_bounds(X_for_bounds, bounds, feature_names)
    d_aug = d_in + len(feature_names)
    time_index = [d_in - 1]
    derive = _AppendEngineeredFeatures(feature_names)

    no_time_normalize = learnable_log_offset or skip_time_in_normalize

    if no_time_normalize:
        if learnable_log_offset:
            log_time = _LearnableLogTimeTransform(
                time_idx=time_index[0],
                initial_offset=log_time_offset,
            )
        else:
            # Fixed-offset version of the same code path. Re-implement
            # AffineInputTransform + Log10 inline so we don't have to track
            # whether Normalize sees them.
            log_time = _LearnableLogTimeTransform(
                time_idx=time_index[0],
                initial_offset=log_time_offset,
            )
            # Freeze the parameter — we want a true fixed-offset control.
            log_time.raw_offset.requires_grad_(False)
        non_time_dims = [i for i in range(d_aug) if i not in time_index]
        tf_normalize = Normalize(
            d_aug,
            indices=torch.tensor(non_time_dims),
            bounds=augmented,
        )
        return ChainedInputTransform(
            derive=derive,
            log_time=log_time,
            normalize=tf_normalize,
        )

    # Default fixed-offset path (Normalize over all dims including time).
    tf_log_offset = AffineInputTransform(
        d_aug,
        coefficient=torch.ones(1),
        offset=torch.full((1,), log_time_offset),
        indices=time_index,
        reverse=True,
    )
    tf_log = Log10(indices=time_index)
    transformed_bounds = tf_log(tf_log_offset(augmented))
    tf_normalize = Normalize(d_aug, bounds=transformed_bounds)
    return ChainedInputTransform(
        derive=derive,
        log_offset=tf_log_offset,
        log=tf_log,
        normalize=tf_normalize,
    )


# Cumulative feature config sets — each builds on the previous.
# F0 = status quo for the strength model (no engineered features at all).
# Feature configurations.
#
# This dict has two tiers:
#   1. ACTIVE configs (F0..F6 and the §4.3 distribution-aware transforms) —
#      what the recommended fits actually use, plus the chemistry-progression
#      story (§4.2) and log-transform progression (§4.3).
#   2. NEGATIVE-RESULT configs — kept here so the negative results in
#      markdown §8 (alternative kernels, single-feature ablations, leave-one-
#      out, minimal-effective hypotheses, transformed-feature explorations)
#      remain reproducible. Each is annotated with the §8.X section that
#      uses it. Do NOT remove without first verifying the corresponding
#      markdown section no longer needs the variant.
_FEATURE_CONFIGS = {
    # ---- ACTIVE: §4.2 chemistry-feature progression ----
    "F0": [],
    "F1": ["wb_ratio"],
    "F2": ["wb_ratio", "scm_frac"],
    "F3": ["wb_ratio", "scm_frac", "hrwr_binder"],
    "F5": [
        "wb_ratio",
        "scm_frac",
        "hrwr_binder",
        "wc_ratio",
        "coarse_fine",
        "agg_paste",
        "maturity",
    ],
    # ---- ACTIVE: §4.3 distribution-aware log-transforms ----
    "F5_loghrwr": [
        "wb_ratio",
        "scm_frac",
        "log_hrwr_binder",
        "wc_ratio",
        "coarse_fine",
        "agg_paste",
        "maturity",
    ],
    "F5_lh_lmat": [
        "wb_ratio",
        "scm_frac",
        "log_hrwr_binder",
        "wc_ratio",
        "coarse_fine",
        "agg_paste",
        "log_maturity_robust",
    ],
    "F5_alllog": [
        "wb_ratio",
        "scm_frac",
        "log_hrwr_binder",
        "log_wc_ratio",
        "log_coarse_fine",
        "log_agg_paste",
        "log_maturity_robust",
    ],
    # ---- F6 / F7 / F8: explicit interaction features motivated by the
    # §6.3 feature ablation study ----
    # The Class-2 "synergy-only" features (log_agg_paste, log_coarse_fine)
    # showed they hurt when used alone but help in combination. F6+
    # variants test whether pre-computing explicit chemistry × packing
    # interactions gives the kernel a clearer signal to fit.
    "F6_wb_ap_interaction": [
        "wb_ratio",
        "scm_frac",
        "log_hrwr_binder",
        "log_wc_ratio",
        "log_coarse_fine",
        "log_agg_paste",
        "log_maturity_robust",
        "log_wb_x_agg_paste",
    ],
    "F7_two_interactions": [
        "wb_ratio",
        "scm_frac",
        "log_hrwr_binder",
        "log_wc_ratio",
        "log_coarse_fine",
        "log_agg_paste",
        "log_maturity_robust",
        "log_wb_x_agg_paste",
        "log_hrwr_binder_x_agg_paste",
    ],
    "F8_three_interactions": [
        "wb_ratio",
        "scm_frac",
        "log_hrwr_binder",
        "log_wc_ratio",
        "log_coarse_fine",
        "log_agg_paste",
        "log_maturity_robust",
        "log_wb_x_agg_paste",
        "log_hrwr_binder_x_agg_paste",
        "log_wc_x_coarse_fine",
    ],
    # Phase 0 follow-up: F5_alllog with a much steeper log-maturity transform
    # (eps=1e-3 inside log instead of +1). Decouples the anchor's
    # log_maturity coordinate from the t≥1 real-data values, so the kernel
    # doesn't have to compromise on the maturity-dim lengthscale to satisfy
    # both. Used together with the learnable-time-offset variant to test
    # whether decoupling on BOTH time-dependent dims (raw time + maturity)
    # closes the anchor regression.
    "F5_alllog_steepmat": [
        "wb_ratio",
        "scm_frac",
        "log_hrwr_binder",
        "log_wc_ratio",
        "log_coarse_fine",
        "log_agg_paste",
        "log_maturity_robust_eps1e-3",
    ],
    "F5_alllog_steepmat6": [
        "wb_ratio",
        "scm_frac",
        "log_hrwr_binder",
        "log_wc_ratio",
        "log_coarse_fine",
        "log_agg_paste",
        "log_maturity_robust_eps1e-6",
    ],
    # ---- ACTIVE: §8.7 negative results — F4 adds W/C, F6 uses eff_W/B ----
    "F4": ["wb_ratio", "scm_frac", "hrwr_binder", "wc_ratio"],
    "F6": [
        "eff_wb_ratio",
        "scm_frac",
        "hrwr_binder",
        "wc_ratio",
        "coarse_fine",
        "agg_paste",
        "maturity",
    ],
    # ---- §8.3: single-feature ablations ----
    # "Only HRWR/binder helps in isolation" finding. Other Fxxx single-feature
    # configs reproduce the rest of the table.
    "Fhrwr": ["hrwr_binder"],
    "Fwb": ["wb_ratio"],
    "Fscm": ["scm_frac"],
    "Fmat": ["maturity"],
    "Fwc": ["wc_ratio"],
    # ---- §8.3: F5 leave-one-out ablations (the "essential features" study
    # whose conclusion was disproven by §8.3's two-feature failure) ----
    "F5_minus_hrwr": [
        "wb_ratio",
        "scm_frac",
        "wc_ratio",
        "coarse_fine",
        "agg_paste",
        "maturity",
    ],
    "F5_minus_wb": [
        "scm_frac",
        "hrwr_binder",
        "wc_ratio",
        "coarse_fine",
        "agg_paste",
        "maturity",
    ],
    "F5_minus_scm": [
        "wb_ratio",
        "hrwr_binder",
        "wc_ratio",
        "coarse_fine",
        "agg_paste",
        "maturity",
    ],
    "F5_minus_wc": [
        "wb_ratio",
        "scm_frac",
        "hrwr_binder",
        "coarse_fine",
        "agg_paste",
        "maturity",
    ],
    "F5_minus_aggregates": [
        "wb_ratio",
        "scm_frac",
        "hrwr_binder",
        "wc_ratio",
        "maturity",
    ],
    "F5_minus_maturity": [
        "wb_ratio",
        "scm_frac",
        "hrwr_binder",
        "wc_ratio",
        "coarse_fine",
        "agg_paste",
    ],
    # ---- §8.3: minimal-essential hypothesis (HRWR + maturity should suffice;
    # the hypothesis FAILED — block-LOO ~754 vs F5_alllog 663) ----
    "Fhrwr_mat2": ["hrwr_binder", "maturity"],
    "Fessential3": ["hrwr_binder", "maturity", "scm_frac"],
    "Fessential4": ["hrwr_binder", "maturity", "scm_frac", "wb_ratio"],
    # ---- §8.4: W/C clipping investigation ----
    # Negative result — wc_ratio_clipped hurts; log_W/C is the right fix.
    "F5_robust": [
        "wb_ratio",
        "scm_frac",
        "log_hrwr_binder",
        "wc_ratio_clipped",
        "coarse_fine",
        "agg_paste",
        "maturity_robust",
    ],
    "F5_mat_robust": [
        "wb_ratio",
        "scm_frac",
        "hrwr_binder",
        "wc_ratio",
        "coarse_fine",
        "agg_paste",
        "maturity_robust",
    ],
    "F5_log_mat": [
        "wb_ratio",
        "scm_frac",
        "log_hrwr_binder",
        "wc_ratio",
        "coarse_fine",
        "agg_paste",
        "maturity_robust",
    ],
    # ---- §17 (the binary HRWR indicator and intermediate log-combos) ----
    "F3_log": ["wb_ratio", "scm_frac", "log_hrwr_binder"],
    "F3_indicator": ["wb_ratio", "scm_frac", "hrwr_binder", "hrwr_used"],
    "F3_logind": ["wb_ratio", "scm_frac", "log_hrwr_binder", "hrwr_used"],
    "Fhrwr_log": ["log_hrwr_binder"],
    "Fhrwr_ind": ["hrwr_used"],
    # ---- §18 intermediate log-transform pair experiments ----
    "F5_lh_lap": [
        "wb_ratio",
        "scm_frac",
        "log_hrwr_binder",
        "wc_ratio",
        "coarse_fine",
        "log_agg_paste",
        "maturity_robust",
    ],
    "F5_lh_lwc": [
        "wb_ratio",
        "scm_frac",
        "log_hrwr_binder",
        "log_wc_ratio",
        "coarse_fine",
        "agg_paste",
        "maturity_robust",
    ],
}


def _build_b_double_prime_kernel_for_aug_dim_time_matern(
    d_aug: int,
    lengthscale_lower: float = 1e-2,
) -> torch.nn.Module:
    """Same as ``_build_b_double_prime_kernel_for_aug_dim`` but the
    additive time component uses Matern-5/2 instead of RBF."""
    no_source_dims = [i for i in range(10) if i != _SOURCE_DIM]
    all_orig_dims = list(range(10))
    extra_dims = list(range(10, d_aug))
    blind = _ard_matern_with_within_group_prior(
        ard_num_dims=len(no_source_dims) + len(extra_dims),
        active_dims=torch.tensor(no_source_dims + extra_dims),
        prior=_within_group_prior(
            d_in=10, source_dim=_SOURCE_DIM, num_extras=len(extra_dims)
        ),
        initial_outputscale=1.0,
        lengthscale_lower=lengthscale_lower,
    )
    specific = _ard_matern_with_within_group_prior(
        ard_num_dims=10 + len(extra_dims),
        active_dims=torch.tensor(all_orig_dims + extra_dims),
        prior=_within_group_prior(d_in=10, num_extras=len(extra_dims)),
        initial_outputscale=0.5,
        lengthscale_lower=lengthscale_lower,
    )
    return blind + specific + _additive_time_matern_kernel(10)


def _build_b_double_prime_kernel_for_aug_dim_no_time_component(
    d_aug: int,
    lengthscale_lower: float = 1e-2,
) -> torch.nn.Module:
    """Same as ``_build_b_double_prime_kernel_for_aug_dim`` but WITHOUT
    the additive time-only component. Tests whether the time information
    in the joint multi-Matern is sufficient. Under the gated kernel,
    the additive time-only component might be redundant."""
    no_source_dims = [i for i in range(10) if i != _SOURCE_DIM]
    all_orig_dims = list(range(10))
    extra_dims = list(range(10, d_aug))
    blind = _ard_matern_with_within_group_prior(
        ard_num_dims=len(no_source_dims) + len(extra_dims),
        active_dims=torch.tensor(no_source_dims + extra_dims),
        prior=_within_group_prior(
            d_in=10, source_dim=_SOURCE_DIM, num_extras=len(extra_dims)
        ),
        initial_outputscale=1.0,
        lengthscale_lower=lengthscale_lower,
    )
    specific = _ard_matern_with_within_group_prior(
        ard_num_dims=10 + len(extra_dims),
        active_dims=torch.tensor(all_orig_dims + extra_dims),
        prior=_within_group_prior(d_in=10, num_extras=len(extra_dims)),
        initial_outputscale=0.5,
        lengthscale_lower=lengthscale_lower,
    )
    return blind + specific


def _build_single_matern_kernel_for_aug_dim(
    d_aug: int,
    lengthscale_lower: float = 1e-2,
) -> torch.nn.Module:
    """Single ARD Matern over all dims (no source-specific component, no
    additive time component, no within-group prior). The simplest possible
    kernel, comparable to the production baseline (without the prior) but
    extended to the augmented input."""
    return ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=d_aug,
            lengthscale_constraint=LogTransformedInterval(
                lengthscale_lower,
                1e3,
                initial_value=1.0,
            ),
        ),
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
    )


def _build_single_matern_with_prior_for_aug_dim(
    d_aug: int,
    lengthscale_lower: float = 1e-2,
) -> torch.nn.Module:
    """Single ARD Matern over all dims WITH the within-group shrinkage prior
    on the binder/aggregate dims. This is the closest single-Matern analogue
    of the production baseline, extended to augmented inputs. Used as the
    "first shippable model" in the journey plot — adds gating to the
    production baseline while preserving the prior, BEFORE swapping to
    Multi-Matern."""
    extra_dims = list(range(10, d_aug))
    prior = _within_group_prior(d_in=10, num_extras=len(extra_dims))
    return _ard_matern_with_within_group_prior(
        ard_num_dims=10 + len(extra_dims),
        active_dims=torch.tensor(list(range(10)) + extra_dims),
        prior=prior,
        initial_outputscale=1.0,
        lengthscale_lower=lengthscale_lower,
    )


def _build_single_matern_with_prior_and_rbf_t_for_aug_dim(
    d_aug: int,
    lengthscale_lower: float = 1e-2,
) -> torch.nn.Module:
    """Single ARD Matern over all dims WITH the within-group shrinkage prior
    AND an additive RBF(t) time-only kernel. Decomposes the "Multi-Matern"
    upgrade in the model journey into its two compositional pieces:
    (1) the additive time-only RBF kernel, and (2) the source-specific
    Matern decomposition. This intermediate stage isolates (1)."""
    base = _build_single_matern_with_prior_for_aug_dim(d_aug, lengthscale_lower)
    return base + _additive_time_kernel(10)


def _build_b_double_prime_kernel_for_aug_dim_no_prior(
    d_aug: int,
    lengthscale_lower: float = 1e-2,
) -> torch.nn.Module:
    """Same as ``_build_b_double_prime_kernel_for_aug_dim`` but WITHOUT
    the within-group lengthscale shrinkage prior. Tests whether the
    prior is still beneficial under the gated kernel."""
    no_source_dims = [i for i in range(10) if i != _SOURCE_DIM]
    all_orig_dims = list(range(10))
    extra_dims = list(range(10, d_aug))
    blind = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=len(no_source_dims) + len(extra_dims),
            active_dims=torch.tensor(no_source_dims + extra_dims),
            lengthscale_constraint=LogTransformedInterval(
                lengthscale_lower,
                1e3,
                initial_value=1.0,
            ),
        ),
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
    )
    specific = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=10 + len(extra_dims),
            active_dims=torch.tensor(all_orig_dims + extra_dims),
            lengthscale_constraint=LogTransformedInterval(
                lengthscale_lower,
                1e3,
                initial_value=1.0,
            ),
        ),
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=0.5),
    )
    return blind + specific + _additive_time_kernel(10)


def _build_b_double_prime_kernel_for_aug_dim_no_rbf(
    d_aug: int,
    lengthscale_lower: float = 1e-2,
) -> torch.nn.Module:
    """Same as ``_build_b_double_prime_kernel_for_aug_dim`` but without the
    additive ``RBF(t)`` time-only component.

    Used by markdown §4.1's "RBF(t) contribution" benchmark — the variant
    `B''+F5_alllog_norbf` ablates only that single kernel component while
    keeping everything else (B'' multi-Matern, F5_alllog features,
    learnable scalar noise, no A) constant.
    """
    no_source_dims = [i for i in range(10) if i != _SOURCE_DIM]
    all_orig_dims = list(range(10))
    extra_dims = list(range(10, d_aug))
    blind = _ard_matern_with_within_group_prior(
        ard_num_dims=len(no_source_dims) + len(extra_dims),
        active_dims=torch.tensor(no_source_dims + extra_dims),
        prior=_within_group_prior(
            d_in=10, source_dim=_SOURCE_DIM, num_extras=len(extra_dims)
        ),
        initial_outputscale=1.0,
        lengthscale_lower=lengthscale_lower,
    )
    specific = _ard_matern_with_within_group_prior(
        ard_num_dims=10 + len(extra_dims),
        active_dims=torch.tensor(all_orig_dims + extra_dims),
        prior=_within_group_prior(d_in=10, num_extras=len(extra_dims)),
        initial_outputscale=0.5,
        lengthscale_lower=lengthscale_lower,
    )
    return blind + specific


# ---------------------------------------------------------------------------
# §18: D and E1 kernels adapted to augmented input dim (raw + appended).
# Mirrors _build_b_double_prime_kernel_for_aug_dim. Used to test if the
# kernel-architectural advantages of D / E1 from §14.1 compound with the §17
# distribution-aware feature transformations.
# ---------------------------------------------------------------------------


def _build_d_kernel_for_aug_dim(
    d_aug: int,
    lengthscale_lower: float = 1e-2,
) -> torch.nn.Module:
    """D adapted to d_aug: two unconstrained joint-ARD Materns over all dims."""
    extra_dims = list(range(10, d_aug))
    all_dims = list(range(d_aug))
    prior = _within_group_prior(d_in=10, num_extras=len(extra_dims))
    m1 = _ard_matern_with_within_group_prior(
        ard_num_dims=d_aug,
        active_dims=torch.tensor(all_dims),
        prior=prior,
        initial_outputscale=1.0,
        lengthscale_lower=lengthscale_lower,
    )
    m2 = _ard_matern_with_within_group_prior(
        ard_num_dims=d_aug,
        active_dims=torch.tensor(all_dims),
        prior=prior,
        initial_outputscale=0.5,
        lengthscale_lower=lengthscale_lower,
    )
    return m1 + m2 + _additive_time_kernel(10)


def _build_e1_kernel_for_aug_dim(
    d_aug: int,
    lengthscale_lower: float = 1e-2,
) -> torch.nn.Module:
    """E1 adapted to d_aug: blind + (specific × Hamming-RBF on source) + RBF time."""
    no_source_dims = [i for i in range(10) if i != _SOURCE_DIM]
    extra_dims = list(range(10, d_aug))
    no_source_with_extras = no_source_dims + extra_dims
    blind_prior = _within_group_prior(
        d_in=10, source_dim=_SOURCE_DIM, num_extras=len(extra_dims)
    )
    blind = _ard_matern_with_within_group_prior(
        ard_num_dims=len(no_source_with_extras),
        active_dims=torch.tensor(no_source_with_extras),
        prior=blind_prior,
        initial_outputscale=1.0,
        lengthscale_lower=lengthscale_lower,
    )
    specific_no_source = _ard_matern_with_within_group_prior(
        ard_num_dims=len(no_source_with_extras),
        active_dims=torch.tensor(no_source_with_extras),
        prior=blind_prior,
        initial_outputscale=0.5,
        lengthscale_lower=lengthscale_lower,
    )
    k_cat = ScaleKernel(
        RBFKernel(active_dims=torch.tensor([_SOURCE_DIM])),
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
    )
    return blind + (specific_no_source * k_cat) + _additive_time_kernel(10)


def _make_engineered_no_a_fit(
    feature_names: list[str],
    kernel_builder=None,
    learnable_log_offset: bool = False,
    log_time_offset_init: float = 1.0,
    skip_time_in_normalize: bool = False,
    output_max_scale: bool = False,
    zero_mean: bool = False,
    per_source_noise: tuple[float, float] | None = None,
    learnable_per_source_noise: bool = False,
    powers_law_mean: bool = False,
    powers_law_tau: float = 0.5,
    refine_block_loo: bool = False,
    block_loo_only: bool = False,
    gated_noise: bool = False,
    smoothness_lambda: float = 0.0,
    monotonicity_lambda: float = 0.0,
    gated_noise_tau: float = 0.05,
):
    """No-A engineered-feature fit: B''-style (or any) multi-Matern kernel +
    chemistry features + Y-standardisation, with a single learnable scalar
    Gaussian likelihood (no per-row Yvar, no global floor, no day-zero anchors).

    This is the recommended fit path (see markdown §3 / §4.4). Without A, the
    kernel retains the right amount of expressiveness for unseen compositions
    instead of overfitting per-composition time curves.

    ``kernel_builder``: a callable ``(d_aug: int) -> torch.nn.Module``
    producing the covariance module. Defaults to the augmented-dim B'' kernel.
    Pass ``functools.partial(_build_b_double_prime_kernel_for_aug_dim,
    lengthscale_lower=1e-4)`` to relax the lengthscale constraint (the
    `_rl` variants in markdown §6.7), or pass `_build_d_kernel_for_aug_dim` /
    `_build_e1_kernel_for_aug_dim` for the alternative-kernel comparison
    (markdown §6.1).

    For the **anchored** counterpart (day-zero pseudo-observations encoding
    the physics constraint f(x, 0) ≈ 0), use
    ``_make_engineered_no_a_fit_with_anchors`` instead.

    ``learnable_log_offset``: pass True to make the additive offset inside
    the log-time transform a learnable parameter (optimised end-to-end via
    MLL). ``log_time_offset_init`` sets the initialisation. See
    ``_get_engineered_input_transform`` for differentiability caveats.

    ``output_max_scale``: when True, replace the default z-score
    standardisation of Y with multiplicative-only scaling ``Y / y_max``
    (no mean subtraction). Required for the gated-kernel variants where
    we need ``Y = 0`` to map to ``Y_scaled = 0`` so the kernel's structural
    enforcement of ``f(x, 0) = 0`` carries through to raw psi space.

    ``zero_mean``: when True, use ``ZeroMean`` instead of ``ConstantMean``
    so the GP's prior mean is exactly 0 in scaled space (rather than a
    learned constant). Combined with ``output_max_scale=True`` and a
    gated kernel, gives posterior 0 at t=0 in raw psi space exactly.
    """
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.means import ZeroMean

    if kernel_builder is None:
        kernel_builder = _build_b_double_prime_kernel_for_aug_dim

    def _fit(X, Y, Yvar, bounds, seed):
        torch.manual_seed(seed)
        d_in = X.shape[-1]
        d_aug = d_in + len(feature_names)
        if output_max_scale:
            Y_z, y_mean, y_std = _max_scale_Y(Y)
        else:
            Y_z, y_mean, y_std = _standardize_Y(Y)
        if learnable_per_source_noise:
            # Learnable per-source noise: 2 free parameters (one per source)
            # combined into a custom GaussianLikelihood subclass that reads
            # the source dim from the GP input and applies the matching
            # noise per row at runtime. Compromise between the (fixed
            # per-source) variant of §6.10 and the (single learnable global)
            # baseline: lets each source converge to its own appropriate
            # noise level while preserving the model-misspecification slack
            # that fixed noise lacks.
            likelihood = _PerSourceGaussianLikelihood(
                n_sources=2,
                source_dim=_SOURCE_DIM,
                noise_constraint=LogTransformedInterval(
                    1e-6,
                    1.0,
                    initial_value=1e-1,
                ),
            )
            # Stash per-row source indices on the likelihood so MLL paths
            # that don't receive X (the GPyTorch ExactMarginalLogLikelihood
            # call doesn't pass X to the likelihood) can still index the
            # per-source noise correctly during training.
            likelihood.set_train_sources(X[..., _SOURCE_DIM])
        elif gated_noise:
            # Heteroscedastic Gaussian likelihood whose noise is gated
            # by h(t)² — the same gate used by the kernel. This makes
            # the FULL predictive distribution (mean AND variance) vanish
            # at t=0, matching the physical prior that a just-mixed
            # concrete has 0 strength with 0 scatter.
            likelihood = _GatedGaussianLikelihood(
                time_idx=_IDX["time"],
                gate_tau=gated_noise_tau,
                noise_constraint=LogTransformedInterval(
                    1e-6,
                    1.0,
                    initial_value=1e-1,
                ),
            )
            # The MLL path doesn't pass X; stash post-input-transform
            # train times so per-row noise can be computed. Note: at
            # this point in the fit factory `X` is still the RAW
            # 10-dim input, so we use the raw time column. After the
            # input transform is built and applied below, we will
            # re-stash with post-transform times.
            likelihood.set_train_times(X[..., _IDX["time"]])
            # Mark it for re-stashing after model construction.
            _gated_noise_likelihood = likelihood
        elif per_source_noise is None:
            likelihood = GaussianLikelihood(
                noise_constraint=LogTransformedInterval(1e-6, 1.0, initial_value=1e-1),
            )
        else:
            # Per-source FIXED noise floor: noise(i) = source_noise[source_i]
            # in psi units, scaled to the GP's outcome space. This is a
            # constrained model (no learnable global noise) — addresses the
            # documented 2x heterogeneity between Source 0 (~163 psi) and
            # Source 1 (~76 psi) noise floors. Forces SingleTaskGP into
            # FixedNoiseGaussianLikelihood mode.
            likelihood = (
                None  # will be set automatically by SingleTaskGP from train_Yvar
            )
        kernel = kernel_builder(d_aug)
        input_tf = _get_engineered_input_transform(
            d_in=d_in,
            bounds=bounds,
            feature_names=feature_names,
            X_for_bounds=X,
            log_time_offset=log_time_offset_init,
            learnable_log_offset=learnable_log_offset,
            skip_time_in_normalize=skip_time_in_normalize,
        )
        gp_kwargs = dict(
            train_X=X,
            train_Y=Y_z,
            covar_module=kernel,
            input_transform=input_tf,
        )
        if likelihood is not None:
            gp_kwargs["likelihood"] = likelihood
        else:
            # Build per-row Yvar from per_source_noise tuple, scaled to the
            # GP's outcome space.
            sources = X[..., _IDX["source"]].long()
            s0_psi, s1_psi = per_source_noise
            per_row_var_psi = torch.where(
                sources == 0,
                torch.tensor(s0_psi, dtype=Y.dtype, device=Y.device) ** 2,
                torch.tensor(s1_psi, dtype=Y.dtype, device=Y.device) ** 2,
            ).unsqueeze(-1)
            # Y_scaled = Y / y_std (with y_mean = 0 for max-scale, or y_std=std for z-score).
            # Variance scales as 1/y_std^2.
            y_std_squeezed = y_std.squeeze() if y_std.dim() > 0 else y_std
            per_row_var_scaled = per_row_var_psi / (y_std_squeezed**2)
            gp_kwargs["train_Yvar"] = per_row_var_scaled.clamp_min(1e-6)
        if output_max_scale:
            # The gated-kernel path requires explicitly disabling BoTorch's
            # default Standardize: otherwise the additive `+y_mean` term
            # in un-standardisation re-introduces an offset that breaks the
            # GP=0 \u2194 raw=0 correspondence the gated kernel relies on.
            # For non-gated variants, leave the default in place so the GP
            # gets the (cosmetic-but-numerically-helpful) standardisation.
            gp_kwargs["outcome_transform"] = None
        if zero_mean:
            gp_kwargs["mean_module"] = ZeroMean()
        if powers_law_mean:
            # Powers'-law parametric mean: μ(x, t) = α · (1 - exp(-t/tau)).
            # Vanishes at t=0 (preserves the gated kernel's structural
            # constraint). The GP fits residuals around this curve.
            gp_kwargs["mean_module"] = _PowersLawGatedMean(
                tau_mean=powers_law_tau,
                init_alpha=0.4,
            )
        model = SingleTaskGP(**gp_kwargs)
        if gated_noise:
            # Re-stash with POST-input-transform train times. SingleTaskGP
            # has applied the input_transform during construction; pull
            # times from model.train_inputs[0] at the post-transform
            # time index (== 9 in the augmented input; same as _IDX["time"]).
            with torch.no_grad():
                post_transform_times = model.train_inputs[0][..., _IDX["time"]]
            model.likelihood.set_train_times(post_transform_times)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        if not block_loo_only:
            fit_gpytorch_mll(mll)
        if refine_block_loo or block_loo_only:
            # Pass training compositions (raw 9-dim, no time) for the
            # smoothness penalty, if active. We use the unique-by-row
            # raw inputs (which already include source + temp).
            if smoothness_lambda > 0 or monotonicity_lambda > 0:
                # Raw `X` here is the augmented 10-dim input (composition
                # + time). Drop the time column to get the 9-dim composition.
                smooth_comps = X[..., : _IDX["time"]].detach().clone()
                # Deduplicate to one row per unique composition (so we don't
                # spend computation on repeated time-replicates).
                smooth_comps = torch.unique(smooth_comps, dim=0)
            else:
                smooth_comps = None
            final_block_loo = refine_with_block_loo(
                model,
                n_real=X.shape[0],
                max_iter=200 if block_loo_only else 50,
                lr=0.1,
                smoothness_lambda=smoothness_lambda,
                smoothness_compositions=smooth_comps,
                monotonicity_lambda=monotonicity_lambda,
            )
            model._study_block_loo_loss_after_refine = float(final_block_loo)
        with torch.no_grad():
            output = model(*model.train_inputs)
            model._study_mll_per_row = float(mll(output, model.train_targets).item())
        model._study_y_mean = y_mean.squeeze()
        model._study_y_std = y_std.squeeze()
        # Stash raw 10-dim training X (real rows only) for use by
        # `phantom_anchor_metrics` to construct virtual (c, t=0) test points.
        model._study_X_train_raw = X.detach().clone()
        return model, X.shape[0]

    return _fit


def _make_engineered_no_a_fit_with_anchors(
    feature_names: list[str],
    kernel_builder=None,
    log_time_offset: float = 1.0,
    learnable_log_offset: bool = False,
):
    """Like ``_make_engineered_no_a_fit`` but adds 128 day-zero pseudo-
    observations encoding the physical constraint that strength ≈ 0 at time
    = 0. The anchors are appended to the training set with near-zero fixed
    noise via ``PartialFixedNoiseLikelihood``; real rows retain a single
    learnable scalar noise.

    See markdown §3.1 / §6.7 / Phase 0 of the productionization plan: the
    physics motivation for anchors is independent of the noise model, so they
    should be retained even in the recommended (no-A) regime. This factory
    makes that combination available so we can ablate it directly.

    **Standardisation**: Y is standardised using **real-data statistics
    only** (not real + anchors). The anchor Y=0 then maps to a z-value of
    ``-y_mean_real / y_std_real`` — a constant negative offset the model
    fits via the kernel. Standardising over the augmented Y instead would
    contaminate ``y_std`` with the 128 zeros and produce a bimodal target
    distribution; that approach matches the production baseline numerically
    but produces poor multi-Matern + engineered-feature fits.

    ``log_time_offset`` controls the (initial, if learnable) offset inside
    the log-time transform. Default 1.0 matches the production
    `Log10(time + 1)` transform; smaller values stretch the t=0 vs t=1
    distance.

    ``learnable_log_offset``: if True, the offset becomes a learnable
    parameter optimised end-to-end via marginal-likelihood. ``log_time_offset``
    is the initialisation in this mode. See ``_get_engineered_input_transform``
    for the differentiability caveats; in particular, Normalize is then
    applied only to the non-time dims.

    The fit closure stashes ``n_pseudo`` on the model so downstream metric
    helpers can choose to score the anchor rows too.
    """
    if kernel_builder is None:
        kernel_builder = _build_b_double_prime_kernel_for_aug_dim

    def _fit(X, Y, Yvar, bounds, seed):
        torch.manual_seed(seed)
        d_in = X.shape[-1]
        d_aug = d_in + len(feature_names)

        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
        n_real = X.shape[0]

        # Standardise over REAL Y only.
        Y_z_real, y_mean, y_std = _standardize_Y(Y)

        # Append 128 day-zero anchor pseudo-observations.
        X_0, Y_0, _Yvar_0 = get_day_zero_data(X=X, n=128)
        n_pseudo = X_0.shape[0]
        # Anchors have raw Y=0; their z-value is (0 - y_mean) / y_std.
        Y_0_z = (Y_0 - y_mean) / y_std

        X_aug = torch.cat((X, X_0), dim=0)
        Y_aug_z = torch.cat((Y_z_real, Y_0_z), dim=0)

        likelihood = PartialFixedNoiseLikelihood(
            n_real=n_real,
            n_pseudo=n_pseudo,
            pseudo_noise=1e-6,
            noise_constraint=LogTransformedInterval(1e-6, 1.0, initial_value=1e-1),
        )
        kernel = kernel_builder(d_aug)
        input_tf = _get_engineered_input_transform(
            d_in=d_in,
            bounds=bounds,
            feature_names=feature_names,
            X_for_bounds=X_aug,
            log_time_offset=log_time_offset,
            learnable_log_offset=learnable_log_offset,
        )
        model = SingleTaskGP(
            train_X=X_aug,
            train_Y=Y_aug_z,
            covar_module=kernel,
            likelihood=likelihood,
            input_transform=input_tf,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        # Stash final MLL value (per-row, normalised by n) for diagnostics.
        with torch.no_grad():
            output = model(*model.train_inputs)
            model._study_mll_per_row = float(mll(output, model.train_targets).item())
        model._study_y_mean = y_mean.squeeze()
        model._study_y_std = y_std.squeeze()
        # Stash raw 10-dim training X (REAL rows only — the anchor pseudo-rows
        # are excluded so the phantom-anchor metric uses just the real-data
        # composition set).
        model._study_X_train_raw = X.detach().clone()
        # Stash n_pseudo so the metric helpers can optionally include anchors.
        model._study_n_pseudo = n_pseudo
        # Stash the learnt log-time offset for diagnostics.
        if learnable_log_offset:
            try:
                model._study_log_time_offset = float(input_tf.log_time.offset.item())
            except (AttributeError, RuntimeError):
                pass
        return model, n_real

    return _fit


def _make_engineered_floor_fit(feature_names: list[str]):
    """Closure factory: A+B''+floor fit using engineered input features.

    Stacks on the current production champion (A+B''+floor) and varies only
    the input transform. Day-zero anchors are dropped (consistent with the
    floor variants) — the homoscedastic floor would otherwise drown the
    near-zero pseudo-observation noise.
    """

    def _fit(X, Y, Yvar, bounds, seed):
        torch.manual_seed(seed)
        d_in = X.shape[-1]
        d_aug = d_in + len(feature_names)
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
        if Yvar.dim() == 1:
            Yvar = Yvar.unsqueeze(-1)

        floor = torch.full_like(Yvar, 25.0)
        Yvar_floored = torch.maximum(Yvar, floor)
        Y_z, y_mean, y_std = _standardize_Y(Y)
        Yvar_z = Yvar_floored / (y_std**2)
        likelihood = FixedNoiseAdditiveLikelihood(noise=Yvar_z.squeeze(-1))

        kernel = _build_b_double_prime_kernel_for_aug_dim(d_aug)
        input_tf = _get_engineered_input_transform(
            d_in=d_in,
            bounds=bounds,
            feature_names=feature_names,
            X_for_bounds=X,
        )
        model = SingleTaskGP(
            train_X=X,
            train_Y=Y_z,
            covar_module=kernel,
            likelihood=likelihood,
            input_transform=input_tf,
            # No outcome_transform — Y already standardised in-line.
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        model._study_y_mean = y_mean.squeeze()
        model._study_y_std = y_std.squeeze()
        return model, X.shape[0]

    return _fit


# Wire engineered-feature variants. F0 = baseline (no derived features).
for _config_name, _features in _FEATURE_CONFIGS.items():
    VARIANTS[f"A+B''+floor+{_config_name}"] = _make_engineered_floor_fit(_features)


# Wire no-A counterparts of the engineered-feature variants.
# Naming: "B''+F<n>" = same kernel + features as A+B''+floor+F<n>, but with
# learnable scalar Gaussian noise (no A, no floor) instead of fixed Yvar + floor.
# These are the no-A series that produces the markdown's §3 / §4.3 results.
for _config_name, _features in _FEATURE_CONFIGS.items():
    VARIANTS[f"B''+{_config_name}"] = _make_engineered_no_a_fit(_features)

# §4.1 RBF(t) contribution benchmark: B'' without the additive RBF time
# component, on F5_alllog (champion) and F0 (no-FE) for context.
for _config_name in ["F0", "F5_alllog"]:
    VARIANTS[f"B''+{_config_name}_norbf"] = _make_engineered_no_a_fit(
        _FEATURE_CONFIGS[_config_name],
        kernel_builder=_build_b_double_prime_kernel_for_aug_dim_no_rbf,
    )

# §6.7 negative result: relaxed lengthscale lower bound (1e-4 instead of
# 1e-2). Documented in markdown as "tighter constraint doesn't fix the
# HRWR/binder lengthscale railing — feature distribution does (§4.3)".
from functools import partial as _partial

for _config_name in ["F3", "F5"]:
    VARIANTS[f"B''+{_config_name}_rl"] = _make_engineered_no_a_fit(
        _FEATURE_CONFIGS[_config_name],
        kernel_builder=_partial(
            _build_b_double_prime_kernel_for_aug_dim, lengthscale_lower=1e-4
        ),
    )

# Phase 0 of the productionization plan: anchored counterparts of the
# recommended (no-A) variants. Day-zero anchors encode the physics constraint
# f(x, 0) ≈ 0; the original benchmark dropped them as a code-path
# simplification, but the productionization plan re-includes them by default.
# Register the configs that appear in the §1.4 / §6 ablations so we can
# directly compare anchored vs unanchored on full data and on the §5 subset
# learning curves.
for _config_name in ["F0", "F3", "F5", "F5_alllog"]:
    VARIANTS[f"B''+{_config_name}+anchors"] = _make_engineered_no_a_fit_with_anchors(
        _FEATURE_CONFIGS[_config_name],
    )

# Phase 0 follow-up: same anchored variant but with a steeper log-time
# transform (offset 0.1 instead of 1.0). This stretches the t=0 vs t=1
# distance after `Log10(time + offset)` from 0.30 to 1.04 — roughly 3.4×
# more separation — so the GP's time lengthscale doesn't have to compromise
# between satisfying the anchors near t=0 and modelling the early-age real
# data at t=1.
for _config_name in ["F0", "F5_alllog"]:
    VARIANTS[f"B''+{_config_name}+anchors+steeptime0.1"] = (
        _make_engineered_no_a_fit_with_anchors(
            _FEATURE_CONFIGS[_config_name],
            log_time_offset=0.1,
        )
    )
    VARIANTS[f"B''+{_config_name}+anchors+steeptime0.01"] = (
        _make_engineered_no_a_fit_with_anchors(
            _FEATURE_CONFIGS[_config_name],
            log_time_offset=0.01,
        )
    )

# Phase 0 follow-up: learnable additive offset inside the log-time transform.
# Optimised end-to-end via marginal-likelihood; should subsume the fixed-offset
# steepness experiments above. Tested with and without anchors so we can
# isolate "does learnable offset help the no-anchor champion?" from "can it
# close the anchor regression?".
VARIANTS["B''+F5_alllog+learnoff"] = _make_engineered_no_a_fit(
    _FEATURE_CONFIGS["F5_alllog"],
    learnable_log_offset=True,
)
# Diagnostic control (per the user's optimisation question): same no-Normalize-on-time
# code path as the learnable variant but with the offset *fixed* at 1.0. Tells us
# whether the regression seen in `+learnoff` is caused by removing Normalize from
# the time dim or by the learnable parameter itself.
VARIANTS["B''+F5_alllog+nonormtime"] = _make_engineered_no_a_fit(
    _FEATURE_CONFIGS["F5_alllog"],
    skip_time_in_normalize=True,
)
VARIANTS["B''+F5_alllog+anchors+learnoff"] = _make_engineered_no_a_fit_with_anchors(
    _FEATURE_CONFIGS["F5_alllog"],
    learnable_log_offset=True,
)
# Initialise the learnable offset at a smaller value, in case the optimiser
# has trouble shrinking from 1.0 (gradient there is small for offsets near 1).
VARIANTS["B''+F5_alllog+anchors+learnoff_init0.01"] = (
    _make_engineered_no_a_fit_with_anchors(
        _FEATURE_CONFIGS["F5_alllog"],
        learnable_log_offset=True,
        log_time_offset=0.01,
    )
)

# Phase 0 follow-up (the user's "maturity blind spot" insight): combine the
# learnable time offset with a much-steeper log-maturity transform so the
# anchor and real rows are decoupled on BOTH time-dependent dims simultaneously.
VARIANTS["B''+F5_alllog_steepmat"] = _make_engineered_no_a_fit(
    _FEATURE_CONFIGS["F5_alllog_steepmat"],
)
VARIANTS["B''+F5_alllog_steepmat+anchors"] = _make_engineered_no_a_fit_with_anchors(
    _FEATURE_CONFIGS["F5_alllog_steepmat"],
)
VARIANTS["B''+F5_alllog_steepmat+anchors+learnoff"] = (
    _make_engineered_no_a_fit_with_anchors(
        _FEATURE_CONFIGS["F5_alllog_steepmat"],
        learnable_log_offset=True,
        log_time_offset=0.01,
    )
)
VARIANTS["B''+F5_alllog_steepmat6+anchors+learnoff"] = (
    _make_engineered_no_a_fit_with_anchors(
        _FEATURE_CONFIGS["F5_alllog_steepmat6"],
        learnable_log_offset=True,
        log_time_offset=0.01,
    )
)

# Phase 0 follow-up: time-gated kernel — multiplicatively gate the recommended
# kernel by h(t) so the prior variance at t=0 is exactly zero. Structurally
# enforces f(x, 0) = 0 without anchor pseudo-observations. See
# STRENGTH_GP_ANCHORS_STUDY.md §5 (Tier 3 idea 8) for the full design.
VARIANTS["B''+F5_alllog+gated_t"] = _make_engineered_no_a_fit(
    _FEATURE_CONFIGS["F5_alllog"],
    kernel_builder=_make_b_double_prime_time_gated_builder(
        gate_tau=0.05,
        gate_learnable=False,
    ),
)
VARIANTS["B''+F5_alllog+gated_t_learn"] = _make_engineered_no_a_fit(
    _FEATURE_CONFIGS["F5_alllog"],
    kernel_builder=_make_b_double_prime_time_gated_builder(
        gate_tau=0.05,
        gate_learnable=True,
    ),
)
# Sweep a few gate-tau values to find a good fixed setting before deciding
# whether the learnable version is needed.
for _tau in [0.02, 0.10, 0.20]:
    _name = f"B''+F5_alllog+gated_t_tau{_tau}"
    VARIANTS[_name] = _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=_tau,
            gate_learnable=False,
        ),
    )

# Phase 0 follow-up (the user's "standardisation breaks gating" insight):
# the gated kernel only enforces f(x, 0) = 0 in *raw psi space* if Y is
# scaled multiplicatively (not z-score standardised) AND the GP uses
# ZeroMean (so the prior mean is 0 in scaled space). With those two
# changes, gated-kernel posterior at t=0 should be 0 in raw psi exactly.
#
# Control variant (max-scale + zero-mean, NO gating): isolates the effect
# of changing the outcome scaling from the effect of the gating.
VARIANTS["B''+F5_alllog+maxscale_zeromean"] = _make_engineered_no_a_fit(
    _FEATURE_CONFIGS["F5_alllog"],
    output_max_scale=True,
    zero_mean=True,
)
# Recommended path: max-scale + zero-mean + gated kernel.
VARIANTS["B''+F5_alllog+gated_t+maxscale_zeromean"] = _make_engineered_no_a_fit(
    _FEATURE_CONFIGS["F5_alllog"],
    kernel_builder=_make_b_double_prime_time_gated_builder(
        gate_tau=0.05,
        gate_learnable=False,
    ),
    output_max_scale=True,
    zero_mean=True,
    skip_time_in_normalize=True,  # CRITICAL — keep post(t=1) ≈ 0.30 separated from post(t=0) = 0
)
VARIANTS["B''+F5_alllog+gated_t_learn+maxscale_zeromean"] = _make_engineered_no_a_fit(
    _FEATURE_CONFIGS["F5_alllog"],
    kernel_builder=_make_b_double_prime_time_gated_builder(
        gate_tau=0.05,
        gate_learnable=True,
    ),
    output_max_scale=True,
    zero_mean=True,
    skip_time_in_normalize=True,
)
# Diagnostic — does the learnable LOG offset still help once we have the
# gated kernel? Hypothesis (per the user): no — the log offset was a
# workaround to make anchor observations decouple from t=1 real data,
# and with a structurally-gated kernel the workaround is unnecessary.
VARIANTS["B''+F5_alllog+gated_t+maxscale_zeromean+learnoff"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        learnable_log_offset=True,
        log_time_offset_init=1.0,
    )
)
VARIANTS["B''+F5_alllog+gated_t+maxscale_zeromean+learnoff_init0.1"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        learnable_log_offset=True,
        log_time_offset_init=0.1,
    )
)

# Holistic re-benchmark sweep: feature configs with the gated-kernel
# recommended config. Confirms F5_alllog is still the best feature set.
for _config_name in ["F0", "F3", "F5", "F5_alllog"]:
    VARIANTS[f"B''+{_config_name}+gated_t+maxscale_zeromean"] = (
        _make_engineered_no_a_fit(
            _FEATURE_CONFIGS[_config_name],
            kernel_builder=_make_b_double_prime_time_gated_builder(
                gate_tau=0.05,
                gate_learnable=False,
            ),
            output_max_scale=True,
            zero_mean=True,
            skip_time_in_normalize=True,
        )
    )

# τ sweep with the recommended config (max-scale + ZeroMean + skip_time_norm).
# The learnable-τ version overfits; this sweep tests whether any fixed τ
# beats the τ=0.05 default.
for _tau in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]:
    VARIANTS[f"B''+F5_alllog+gated_t_tau{_tau}+maxscale_zeromean"] = (
        _make_engineered_no_a_fit(
            _FEATURE_CONFIGS["F5_alllog"],
            kernel_builder=_make_b_double_prime_time_gated_builder(
                gate_tau=_tau,
                gate_learnable=False,
            ),
            output_max_scale=True,
            zero_mean=True,
            skip_time_in_normalize=True,
        )
    )

# Holistic re-examination of model components under the gated-kernel objective.
# Each ablation tests whether a previously-justified component is still
# necessary given the new structural constraint and the joint metric.


# Time-component RBF vs Matern (user's specific question). The current B''
# kernel uses RBF for the additive time-only component. With the gated kernel
# modifying smoothness near t=0 structurally, the question is open whether
# Matern-5/2 (less smooth than RBF) might be preferred for the time axis.
def _gated_time_matern_builder(d_aug):
    base = _build_b_double_prime_kernel_for_aug_dim_time_matern(d_aug)
    return _TimeGatedKernel(base, time_idx=9, gate_tau=0.05, gate_learnable=False)


VARIANTS["B''+F5_alllog+gated_t+maxscale_zeromean+time_matern"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_gated_time_matern_builder,
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
    )
)


# No additive time-only component at all (rely only on the joint multi-Matern's
# time-dim ARD lengthscale). With the gated kernel, the additive time-only
# component might be redundant — the time information is also in the joint
# kernel's ARD over all dims.
def _gated_no_time_builder(d_aug):
    base = _build_b_double_prime_kernel_for_aug_dim_no_time_component(d_aug)
    return _TimeGatedKernel(base, time_idx=9, gate_tau=0.05, gate_learnable=False)


VARIANTS["B''+F5_alllog+gated_t+maxscale_zeromean+no_time_kernel"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_gated_no_time_builder,
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
    )
)


# Single Matern (no multi-Matern decomposition). Tests whether B'''s
# blind+specific+RBF structure is still beneficial under the gated kernel.
def _gated_single_matern_builder(d_aug):
    base = _build_single_matern_kernel_for_aug_dim(d_aug)
    return _TimeGatedKernel(base, time_idx=9, gate_tau=0.05, gate_learnable=False)


VARIANTS["single_matern+F5_alllog+gated_t+maxscale_zeromean"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_gated_single_matern_builder,
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
    )
)


# Production-style + gated kernel: single Matern WITH within-group prior +
# F0 (no engineered features) + gating. The "first shippable" intermediate
# stage in the journey: adds the structural physics constraint to the
# production baseline while preserving its kernel + prior choices, BEFORE
# upgrading to Multi-Matern.
def _gated_single_matern_with_prior_builder(d_aug):
    base = _build_single_matern_with_prior_for_aug_dim(d_aug)
    return _TimeGatedKernel(base, time_idx=9, gate_tau=0.05, gate_learnable=False)


VARIANTS["single_matern+F0+gated_t+maxscale_zeromean+prior"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F0"],
        kernel_builder=_gated_single_matern_with_prior_builder,
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
    )
)


# Intermediate stage that isolates the additive RBF(t) component of the
# "Multi-Matern" upgrade. Single Matern + prior + RBF(t) lets us separate
# the contribution of the additive time-only kernel from that of the
# source-specific Matern decomposition.
def _gated_single_matern_with_prior_and_rbf_t_builder(d_aug):
    base = _build_single_matern_with_prior_and_rbf_t_for_aug_dim(d_aug)
    return _TimeGatedKernel(base, time_idx=9, gate_tau=0.05, gate_learnable=False)


VARIANTS["single_matern+F0+gated_t+maxscale_zeromean+prior+rbf_t"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F0"],
        kernel_builder=_gated_single_matern_with_prior_and_rbf_t_builder,
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
    )
)

# Source-specific-Matern ablation: same kernel as above (single Matern +
# prior + RBF(t)) but paired with the full F3 / F5_alllog feature configs.
# Tests whether Multi-Matern's blind+specific decomposition is necessary
# at all, or whether single Matern + prior + RBF(t) + features beats
# the full Multi-Matern.
for _config_name in ["F3", "F5", "F5_alllog"]:
    VARIANTS[f"single_matern+{_config_name}+gated_t+maxscale_zeromean+prior+rbf_t"] = (
        _make_engineered_no_a_fit(
            _FEATURE_CONFIGS[_config_name],
            kernel_builder=_gated_single_matern_with_prior_and_rbf_t_builder,
            output_max_scale=True,
            zero_mean=True,
            skip_time_in_normalize=True,
        )
    )
    # Same with gated noise — used by the improvement-journey plot. The
    # gated kernel + gated noise pair is the unified "physics constraint
    # at t=0" step (zero predictive mean AND zero predictive variance).
    VARIANTS[
        f"single_matern+{_config_name}+gated_t+gated_noise+maxscale_zeromean+prior+rbf_t"
    ] = _make_engineered_no_a_fit(
        _FEATURE_CONFIGS[_config_name],
        kernel_builder=_gated_single_matern_with_prior_and_rbf_t_builder,
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        gated_noise=True,
        gated_noise_tau=0.05,
    )

# Variant matching the journey's stage-3 (gated kernel + gated noise on F0).
VARIANTS["single_matern+F0+gated_t+gated_noise+maxscale_zeromean+prior+rbf_t"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F0"],
        kernel_builder=_gated_single_matern_with_prior_and_rbf_t_builder,
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        gated_noise=True,
        gated_noise_tau=0.05,
    )
)

# Feature ablation under the new architecture (gated B'' + max-scale +
# ZeroMean + skip_time_in_normalize). Builds leave-one-out (LOO) and
# singleton variants for each of the 7 features in F5_alllog. Tests
# whether every feature is still necessary under the gated objective.
_F5_ALLLOG_FEATURES = list(_FEATURE_CONFIGS["F5_alllog"])
for _feat in _F5_ALLLOG_FEATURES:
    # Leave-one-out: F5_alllog minus this feature
    _loo_features = [f for f in _F5_ALLLOG_FEATURES if f != _feat]
    _config_key = f"F5_alllog_no_{_feat}"
    _FEATURE_CONFIGS[_config_key] = _loo_features
    VARIANTS[f"B''+{_config_key}+gated_t+maxscale_zeromean"] = (
        _make_engineered_no_a_fit(
            _loo_features,
            kernel_builder=_make_b_double_prime_time_gated_builder(
                gate_tau=0.05,
                gate_learnable=False,
            ),
            output_max_scale=True,
            zero_mean=True,
            skip_time_in_normalize=True,
        )
    )
    # Singleton: F0 + this single feature only
    _config_key_singleton = f"F0_only_{_feat}"
    _FEATURE_CONFIGS[_config_key_singleton] = [_feat]
    VARIANTS[f"B''+{_config_key_singleton}+gated_t+maxscale_zeromean"] = (
        _make_engineered_no_a_fit(
            [_feat],
            kernel_builder=_make_b_double_prime_time_gated_builder(
                gate_tau=0.05,
                gate_learnable=False,
            ),
            output_max_scale=True,
            zero_mean=True,
            skip_time_in_normalize=True,
        )
    )

# Per-source noise floor: addresses the documented 2x heterogeneity
# (Source 0 ~163 psi, Source 1 ~76 psi from triplicate-based estimates).
# Uses FixedNoiseGaussianLikelihood with empirical per-source noise
# values; no learnable noise parameter. Tests whether constraining the
# noise model to the empirically observed heterogeneity helps block-LOO.
VARIANTS["B''+F5_alllog+gated_t+maxscale_zeromean+per_source_noise"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        per_source_noise=(163.0, 76.0),  # Source 0, Source 1 in psi
    )
)
# Learnable per-source noise: 2 free parameters, one per source. The
# motivated extension to the §6.10 negative result — lets each source
# converge to its own appropriate noise level while preserving the
# model-misspecification slack that fixed noise lacks.
VARIANTS["B''+F5_alllog+gated_t+maxscale_zeromean+learnable_per_source_noise"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        learnable_per_source_noise=True,
    )
)
# Powers'-law parametric mean: μ(x, t) = α · (1 - exp(-t/tau)) replaces
# ZeroMean. Vanishes at t=0 (preserves the gated-kernel constraint) and
# gives the GP a physics-aware "expected curve" to fit residuals around.
# Tests whether a domain-knowledge prior trend reduces the burden on the
# kernel. See benchmark §7.1 K4 for the original idea, and the live
# discussion in this conversation for the gated-kernel adaptation.
for _tau in [0.2, 0.5, 1.0]:
    VARIANTS[f"B''+F5_alllog+gated_t+maxscale_zeromean+powers_tau{_tau}"] = (
        _make_engineered_no_a_fit(
            _FEATURE_CONFIGS["F5_alllog"],
            kernel_builder=_make_b_double_prime_time_gated_builder(
                gate_tau=0.05,
                gate_learnable=False,
            ),
            output_max_scale=True,
            zero_mean=False,  # use Powers' instead
            skip_time_in_normalize=True,
            powers_law_mean=True,
            powers_law_tau=_tau,
        )
    )

# Block-LOO objective for HP refinement: after fit_gpytorch_mll
# converges to the MLL optimum, run an additional LBFGS optimisation
# minimising block-LOO predictive negative-log-likelihood. Directly
# attacks the MLL ≠ block-LOO dissociation we have documented at
# every turn (each "extra parameter" overfits MLL but regresses
# block-LOO). If the block-LOO surface has a meaningfully different
# optimum than MLL, this should improve the deployment metric.
VARIANTS["B''+F5_alllog+gated_t+maxscale_zeromean+block_loo_refine"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        refine_block_loo=True,
    )
)

# Gated-noise champion: heteroscedastic gaussian likelihood with h(t)²
# scaling. Combined with the gated kernel, makes the FULL predictive
# distribution (mean AND variance) vanish at t=0 — physically faithful
# (a just-mixed concrete has 0 strength with 0 within-batch scatter).
# At training data t ≥ 1 day, gate ≈ 0.998 → h² ≈ 0.996, so global
# noise estimate is essentially unchanged from the standard champion.
VARIANTS["B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_refine"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        gated_noise=True,
        gated_noise_tau=0.05,
        refine_block_loo=True,
    )
)

# MLL-only counterpart of the gated-noise champion (no refinement).
# Used to A/B test whether refinement is worth its complexity.
VARIANTS["B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        gated_noise=True,
        gated_noise_tau=0.05,
    )
)

# Single-stage block-LOO + priors (no MLL warmup). Tests whether the
# MLL stage is necessary or whether the block-LOO+priors objective is
# sufficient on its own from a default kernel initialisation. Cleaner
# pipeline if it works.
VARIANTS["B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_only"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        gated_noise=True,
        gated_noise_tau=0.05,
        block_loo_only=True,
    )
)
# Same with τ=0.20 (winner from the calibration-aware τ sweep — see §6.13).
VARIANTS[
    "B''+F5_alllog+gated_t_tau0.2+gated_noise+maxscale_zeromean+block_loo_refine"
] = _make_engineered_no_a_fit(
    _FEATURE_CONFIGS["F5_alllog"],
    kernel_builder=_make_b_double_prime_time_gated_builder(
        gate_tau=0.20,
        gate_learnable=False,
    ),
    output_max_scale=True,
    zero_mean=True,
    skip_time_in_normalize=True,
    gated_noise=True,
    gated_noise_tau=0.20,
    refine_block_loo=True,
)

# Combined: gated_noise + drop the actively-harmful log transforms found
# by the per-feature log ablation (§6.X). Specifically log_coarse_fine
# REGRESSED block-LOO under refinement (-18 psi when replaced by raw),
# and log_wc_ratio + log_hrwr_binder were near-wash. We test both
# "drop-coarse-fine-only" and "drop-3-logs" to find the ultimate champion.
_F5_DROP_COARSE_FINE = [
    "wb_ratio",
    "scm_frac",
    "log_hrwr_binder",
    "log_wc_ratio",
    "coarse_fine",  # raw instead of log
    "log_agg_paste",
    "log_maturity_robust",
]
_F5_DROP_3_NEAR_WASH_LOGS = [
    "wb_ratio",
    "scm_frac",
    "hrwr_binder",  # raw
    "wc_ratio",  # raw
    "coarse_fine",  # raw
    "log_agg_paste",
    "log_maturity_robust",
]
_FEATURE_CONFIGS["F5_drop_log_coarse_fine"] = _F5_DROP_COARSE_FINE
_FEATURE_CONFIGS["F5_drop_3_near_wash_logs"] = _F5_DROP_3_NEAR_WASH_LOGS

for _config_name, _features in [
    ("F5_drop_log_coarse_fine", _F5_DROP_COARSE_FINE),
    ("F5_drop_3_near_wash_logs", _F5_DROP_3_NEAR_WASH_LOGS),
]:
    VARIANTS[
        f"B''+{_config_name}+gated_t+gated_noise+maxscale_zeromean+block_loo_refine"
    ] = _make_engineered_no_a_fit(
        _features,
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        gated_noise=True,
        gated_noise_tau=0.05,
        refine_block_loo=True,
    )

# Tau sweep WITH block-LOO refinement enabled. This is the calibration-
# aware re-evaluation of how the time gate's transition sharpness affects
# the production model. The earlier tau sweep used only MLL training and
# preferred small tau (sharp gate) because MLL doesn't penalise miscalibrated
# uncertainty. Under the block-LOO log-likelihood objective (which DOES
# penalise mis-calibrated variance, see §6.12 in benchmark md), a smoother
# gate may be preferred — and would also produce more physically-realistic
# strength curves near t=0 (gradual rise rather than near-step).
for _tau in [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]:
    _name = f"B''+F5_alllog+gated_t_tau{_tau}+maxscale_zeromean+block_loo_refine"
    VARIANTS[_name] = _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=_tau,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        refine_block_loo=True,
    )

# Per-feature log-transform ablation: F5_alllog with EXACTLY ONE log
# replaced by its raw counterpart. Tests whether each individual log
# transform is necessary and beneficial (rather than just the aggregate
# F5 vs F5_alllog comparison the parent benchmark §4.3 did).
#
# F5_alllog has 5 log-transformed features: log_hrwr_binder, log_wc_ratio,
# log_coarse_fine, log_agg_paste, log_maturity_robust. We replace each
# with its raw counterpart in turn.
_F5_ALLLOG_LOG_FEATS = [
    ("log_hrwr_binder", "hrwr_binder"),
    ("log_wc_ratio", "wc_ratio"),
    ("log_coarse_fine", "coarse_fine"),
    ("log_agg_paste", "agg_paste"),
    ("log_maturity_robust", "maturity_robust"),
]
for _log_feat, _raw_feat in _F5_ALLLOG_LOG_FEATS:
    _config_key = f"F5_alllog_minus_log_{_log_feat}"
    _features = [
        _raw_feat if f == _log_feat else f for f in _FEATURE_CONFIGS["F5_alllog"]
    ]
    _FEATURE_CONFIGS[_config_key] = _features
    # Both with and without block-LOO refinement so we can compare cleanly.
    VARIANTS[f"B''+{_config_key}+gated_t+maxscale_zeromean"] = (
        _make_engineered_no_a_fit(
            _features,
            kernel_builder=_make_b_double_prime_time_gated_builder(
                gate_tau=0.05,
                gate_learnable=False,
            ),
            output_max_scale=True,
            zero_mean=True,
            skip_time_in_normalize=True,
        )
    )
    VARIANTS[f"B''+{_config_key}+gated_t+maxscale_zeromean+block_loo_refine"] = (
        _make_engineered_no_a_fit(
            _features,
            kernel_builder=_make_b_double_prime_time_gated_builder(
                gate_tau=0.05,
                gate_learnable=False,
            ),
            output_max_scale=True,
            zero_mean=True,
            skip_time_in_normalize=True,
            refine_block_loo=True,
        )
    )
    # WITH gated_noise (the ultimate champion architecture). MLL-only,
    # used to re-evaluate per-feature log ablation under proper
    # optimization (the buggy refinement gave us a misleading "drop
    # log_coarse_fine" verdict; this variant tests under the architecture
    # we actually deploy).
    VARIANTS[f"B''+{_config_key}+gated_t+gated_noise+maxscale_zeromean"] = (
        _make_engineered_no_a_fit(
            _features,
            kernel_builder=_make_b_double_prime_time_gated_builder(
                gate_tau=0.05,
                gate_learnable=False,
            ),
            output_max_scale=True,
            zero_mean=True,
            skip_time_in_normalize=True,
            gated_noise=True,
            gated_noise_tau=0.05,
        )
    )
    # ALSO under block_loo_only — the actual champion training procedure
    # (single-stage block-LOO + priors, beats MLL-only by 80-200 psi at
    # small data once the silent-crash bug was fixed).
    VARIANTS[
        f"B''+{_config_key}+gated_t+gated_noise+maxscale_zeromean+block_loo_only"
    ] = _make_engineered_no_a_fit(
        _features,
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        gated_noise=True,
        gated_noise_tau=0.05,
        block_loo_only=True,
    )

# MLL-only + gated_noise re-evaluation of the F-config sweep. The
# original F-config decision (F5_alllog wins) was made under MLL alone
# without gated_noise; we re-verify under the deployed architecture
# (MLL + gated_noise) since the buggy refinement era produced spurious
# "drop log_coarse_fine" recommendations. If this re-sweep also picks
# F5_alllog, the architecture stands.
for _f_name in ["F0", "F3", "F5", "F5_alllog"]:
    VARIANTS[f"B''+{_f_name}+gated_t+gated_noise+maxscale_zeromean"] = (
        _make_engineered_no_a_fit(
            _FEATURE_CONFIGS[_f_name],
            kernel_builder=_make_b_double_prime_time_gated_builder(
                gate_tau=0.05,
                gate_learnable=False,
            ),
            output_max_scale=True,
            zero_mean=True,
            skip_time_in_normalize=True,
            gated_noise=True,
            gated_noise_tau=0.05,
        )
    )
    # Also under the actual production training procedure (block_loo_only).
    VARIANTS[f"B''+{_f_name}+gated_t+gated_noise+maxscale_zeromean+block_loo_only"] = (
        _make_engineered_no_a_fit(
            _FEATURE_CONFIGS[_f_name],
            kernel_builder=_make_b_double_prime_time_gated_builder(
                gate_tau=0.05,
                gate_learnable=False,
            ),
            output_max_scale=True,
            zero_mean=True,
            skip_time_in_normalize=True,
            gated_noise=True,
            gated_noise_tau=0.05,
            block_loo_only=True,
        )
    )

# F5 with multiple log replacements — block_loo_only is showing different
# answers than MLL-only on this question. Test pairwise drops to find
# the right answer.
_F5_NO_LOG_MAT = [
    f if f != "log_maturity_robust" else "maturity_robust"
    for f in _FEATURE_CONFIGS["F5_alllog"]
]
_F5_NO_LOG_MAT_NO_LOG_AP = [
    (
        f
        if f not in ("log_maturity_robust", "log_agg_paste")
        else ("maturity_robust" if f == "log_maturity_robust" else "agg_paste")
    )
    for f in _FEATURE_CONFIGS["F5_alllog"]
]
_FEATURE_CONFIGS["F5_no_log_mat"] = _F5_NO_LOG_MAT
_FEATURE_CONFIGS["F5_no_log_mat_no_log_ap"] = _F5_NO_LOG_MAT_NO_LOG_AP
for _name, _feats in [
    ("F5_no_log_mat", _F5_NO_LOG_MAT),
    ("F5_no_log_mat_no_log_ap", _F5_NO_LOG_MAT_NO_LOG_AP),
]:
    VARIANTS[f"B''+{_name}+gated_t+gated_noise+maxscale_zeromean+block_loo_only"] = (
        _make_engineered_no_a_fit(
            _feats,
            kernel_builder=_make_b_double_prime_time_gated_builder(
                gate_tau=0.05,
                gate_learnable=False,
            ),
            output_max_scale=True,
            zero_mean=True,
            skip_time_in_normalize=True,
            gated_noise=True,
            gated_noise_tau=0.05,
            block_loo_only=True,
        )
    )

# Smoothness-regularized sweep on F5_no_log_mat. Goal: recover its
# block-LOO advantage (655 vs F5_alllog's 672) without the unphysical
# oscillations (which it had at λ=0).
for _lam in [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
    _key = f"B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+block_loo_only+smooth{_lam}"
    VARIANTS[_key] = _make_engineered_no_a_fit(
        _F5_NO_LOG_MAT,
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        gated_noise=True,
        gated_noise_tau=0.05,
        block_loo_only=True,
        smoothness_lambda=_lam,
    )

# Monotonicity-only sweep on F5_no_log_mat. Direct hinge penalty
# `λ · mean(relu(-dμ/dt)^2)` targets the actual violation (decreasing
# intervals) rather than total curvature.
for _lam in [10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]:
    _key = f"B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+block_loo_only+mono{_lam}"
    VARIANTS[_key] = _make_engineered_no_a_fit(
        _F5_NO_LOG_MAT,
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        gated_noise=True,
        gated_noise_tau=0.05,
        block_loo_only=True,
        monotonicity_lambda=_lam,
    )
# Monotonicity on F5_alllog for comparison/validation
for _lam in [100.0, 1000.0, 10000.0]:
    _key = (
        f"B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_only+mono{_lam}"
    )
    VARIANTS[_key] = _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        gated_noise=True,
        gated_noise_tau=0.05,
        block_loo_only=True,
        monotonicity_lambda=_lam,
    )
# Combined smoothness + monotonicity sweep (best of both worlds?)
for _ms_lam in [100.0, 1000.0]:
    for _mo_lam in [100.0, 1000.0, 10000.0]:
        _key = (
            f"B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+"
            f"block_loo_only+smooth{_ms_lam}+mono{_mo_lam}"
        )
        VARIANTS[_key] = _make_engineered_no_a_fit(
            _F5_NO_LOG_MAT,
            kernel_builder=_make_b_double_prime_time_gated_builder(
                gate_tau=0.05,
                gate_learnable=False,
            ),
            output_max_scale=True,
            zero_mean=True,
            skip_time_in_normalize=True,
            gated_noise=True,
            gated_noise_tau=0.05,
            block_loo_only=True,
            smoothness_lambda=_ms_lam,
            monotonicity_lambda=_mo_lam,
        )

# All of these regressed block-LOO under pure MLL fitting, with the
# documented "MLL up, block-LOO down" pattern. With the new ability
# to refine HPs against block-LOO directly, the extra flexibility
# they provide may now find a productive optimum.

# F7 / F8 interaction features + block-LOO refine.
# Previous: F7 +73, F8 +64 psi block-LOO regression (§6.3).
for _config_name in ["F7_two_interactions", "F8_three_interactions"]:
    VARIANTS[f"B''+{_config_name}+gated_t+maxscale_zeromean+block_loo_refine"] = (
        _make_engineered_no_a_fit(
            _FEATURE_CONFIGS[_config_name],
            kernel_builder=_make_b_double_prime_time_gated_builder(
                gate_tau=0.05,
                gate_learnable=False,
            ),
            output_max_scale=True,
            zero_mean=True,
            skip_time_in_normalize=True,
            refine_block_loo=True,
        )
    )

# Learnable per-source noise + block-LOO refine.
# Previous: +75 psi block-LOO regression (§6.10).
VARIANTS[
    "B''+F5_alllog+gated_t+maxscale_zeromean+learnable_per_source_noise+block_loo_refine"
] = _make_engineered_no_a_fit(
    _FEATURE_CONFIGS["F5_alllog"],
    kernel_builder=_make_b_double_prime_time_gated_builder(
        gate_tau=0.05,
        gate_learnable=False,
    ),
    output_max_scale=True,
    zero_mean=True,
    skip_time_in_normalize=True,
    learnable_per_source_noise=True,
    refine_block_loo=True,
)

# Learnable τ + block-LOO refine.
# Previous: anchor-study §3.8 documented learnable τ regression.
VARIANTS["B''+F5_alllog+gated_learnable_t+maxscale_zeromean+block_loo_refine"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=True,  # ← learnable
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        refine_block_loo=True,
    )
)
# Try a slightly inflated version too, in case empirical noise underestimates.
VARIANTS["B''+F5_alllog+gated_t+maxscale_zeromean+per_source_noise_1.5x"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        per_source_noise=(244.5, 114.0),  # 1.5x of empirical
    )
)
# Equal-noise control: use the average per-source noise for both. Isolates
# the effect of "fixed vs learnable noise" from "per-source heterogeneity".
VARIANTS["B''+F5_alllog+gated_t+maxscale_zeromean+fixed_noise_avg"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_make_b_double_prime_time_gated_builder(
            gate_tau=0.05,
            gate_learnable=False,
        ),
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
        per_source_noise=(120.0, 120.0),  # ~average of (163+76)/2
    )
)
for _config_name in [
    "F6_wb_ap_interaction",
    "F7_two_interactions",
    "F8_three_interactions",
]:
    VARIANTS[f"B''+{_config_name}+gated_t+maxscale_zeromean"] = (
        _make_engineered_no_a_fit(
            _FEATURE_CONFIGS[_config_name],
            kernel_builder=_make_b_double_prime_time_gated_builder(
                gate_tau=0.05,
                gate_learnable=False,
            ),
            output_max_scale=True,
            zero_mean=True,
            skip_time_in_normalize=True,
        )
    )


# Within-group prior on/off ablation under gating.
def _gated_no_prior_builder(d_aug):
    base = _build_b_double_prime_kernel_for_aug_dim_no_prior(d_aug)
    return _TimeGatedKernel(base, time_idx=9, gate_tau=0.05, gate_learnable=False)


VARIANTS["B''+F5_alllog+gated_t+maxscale_zeromean+no_prior"] = (
    _make_engineered_no_a_fit(
        _FEATURE_CONFIGS["F5_alllog"],
        kernel_builder=_gated_no_prior_builder,
        output_max_scale=True,
        zero_mean=True,
        skip_time_in_normalize=True,
    )
)

# §6.1: D and E1 alternative kernels with various feature configs, no A.
# The full set is needed to reproduce the kernel-architecture comparison
# table — D / E1 variants did not beat B''+F5_alllog at any feature config.
for _config_name in ["F3", "F5", "F5_loghrwr", "F5_alllog"]:
    VARIANTS[f"D+{_config_name}"] = _make_engineered_no_a_fit(
        _FEATURE_CONFIGS[_config_name],
        kernel_builder=_build_d_kernel_for_aug_dim,
    )
    VARIANTS[f"E1+{_config_name}"] = _make_engineered_no_a_fit(
        _FEATURE_CONFIGS[_config_name],
        kernel_builder=_build_e1_kernel_for_aug_dim,
    )


def run(
    seeds: list[int],
    names: list[str],
    *,
    subset_n: int | None = None,
    subset_seed: int = 0,
    holdout_unit: str = "row",
) -> dict[str, dict[str, tuple[float, float]]]:
    data = load_concrete_strength()
    X, Y, Yvar, bounds = data.strength_data

    held_out: tuple[torch.Tensor, torch.Tensor] | None = None
    if subset_n is not None:
        if holdout_unit == "row":
            # Random row-level holdout. NB: leaks information across the time
            # curves of any composition with multiple measurements (the
            # training set may include 1d/3d strength of a mix whose 28d
            # strength is in the held-out set, which is not how production
            # data acquisition works).
            if subset_n < X.shape[0]:
                gen = torch.Generator().manual_seed(subset_seed)
                perm = torch.randperm(X.shape[0], generator=gen)
                train_idx = perm[:subset_n]
                held_idx = perm[subset_n:]
                X_held = X[held_idx]
                Y_held = Y[held_idx]
                X = X[train_idx]
                Y = Y[train_idx]
                Yvar = Yvar[train_idx]
                held_out = (X_held, Y_held.squeeze(-1) if Y_held.dim() > 1 else Y_held)
                print(
                    f"Subsampled to n={subset_n} ROWS "
                    f"(subset_seed={subset_seed}, holdout_unit=row); "
                    f"held-out test set n={X_held.shape[0]}; "
                    f"empirical std median={Yvar.sqrt().median().item():.1f} psi"
                )
            else:
                print(
                    f"Loaded n={X.shape[0]} strength rows; "
                    f"empirical std median={Yvar.sqrt().median().item():.1f} psi"
                )
        elif holdout_unit == "composition":
            # Composition-level holdout: hold out entire UNIQUE COMPOSITIONS
            # (and ALL their time-point measurements). Mirrors how concrete
            # data is actually acquired — every mix is measured at multiple
            # ages and either the whole curve is in training or it isn't.
            # ``subset_n`` here is the number of training compositions, not
            # the number of training rows.
            comp = X[:, :-1]  # composition fingerprint = all dims except Time
            unique_comps = torch.unique(comp, dim=0)
            n_comps = unique_comps.shape[0]
            if subset_n < n_comps:
                gen = torch.Generator().manual_seed(subset_seed)
                comp_perm = torch.randperm(n_comps, generator=gen)
                train_comp_idx = comp_perm[:subset_n]
                train_mask = torch.zeros(X.shape[0], dtype=torch.bool)
                for ci in train_comp_idx:
                    train_mask |= (comp == unique_comps[ci]).all(dim=-1)
                X_train = X[train_mask]
                Y_train = Y[train_mask]
                Yvar_train = Yvar[train_mask]
                X_held = X[~train_mask]
                Y_held = Y[~train_mask]
                held_out = (X_held, Y_held.squeeze(-1) if Y_held.dim() > 1 else Y_held)
                X, Y, Yvar = X_train, Y_train, Yvar_train
                print(
                    f"Subsampled to {subset_n} COMPOSITIONS "
                    f"(subset_seed={subset_seed}, holdout_unit=composition) "
                    f"= {X.shape[0]} training rows; "
                    f"held-out test set n={X_held.shape[0]} "
                    f"({n_comps - subset_n} compositions); "
                    f"empirical std median={Yvar.sqrt().median().item():.1f} psi"
                )
            else:
                print(
                    f"Requested {subset_n} >= {n_comps} unique compositions; "
                    f"using all data ({X.shape[0]} rows)."
                )
        else:
            raise ValueError(
                f"Unknown holdout_unit='{holdout_unit}'; choose 'row' or 'composition'."
            )
    else:
        print(
            f"Loaded n={X.shape[0]} strength rows; "
            f"empirical std median={Yvar.sqrt().median().item():.1f} psi"
        )

    results = {}
    for name in names:

        if name not in VARIANTS:
            raise ValueError(f"Unknown variant '{name}'. Choices: {list(VARIANTS)}")
        per_seed_loo = []
        per_seed_block = []
        per_seed_ho = []
        for seed in seeds:
            model, n_real = VARIANTS[name](X, Y, Yvar, bounds, seed)
            has_anchors = hasattr(model, "_study_n_pseudo")
            # Always print the final MLL value if available — the actual
            # objective the optimiser targets. A regression on block-LOO
            # WITHOUT a regression on MLL means the model is overfitting
            # or misspecified; a regression on both means optimisation is
            # stuck.
            if hasattr(model, "_study_mll_per_row"):
                print(
                    f"  [{name} seed={seed}] final MLL/row = {model._study_mll_per_row:+.4f}"
                )
            # Phantom-anchor metric: posterior at virtual (c, t=0) test points
            # for each unique composition. Computable for both anchored and
            # unanchored models — the apples-to-apples physics-constraint
            # metric per STRENGTH_GP_ANCHORS_STUDY.md §2.
            try:
                phantom = phantom_anchor_metrics(
                    model,
                    n_real,
                    out_of_training_n=144,
                    out_of_training_seed=0,
                    bounds=bounds,
                )
                print(
                    f"  [{name} seed={seed}] phantom-anchor t=0  "
                    f"in-train RMSE={phantom['rmse']:.1f}  "
                    f"OOT RMSE={phantom.get('oot_rmse', float('nan')):.1f}  "
                    f"max|pred|={phantom['max_abs_pred']:.1f}  "
                    f"OOT max|pred|={phantom.get('oot_max_abs_pred', float('nan')):.1f}"
                )
            except Exception as e:
                print(f"  [phantom-anchor failed for {name}: {e}]")
            if hasattr(model, "_study_Y_psi"):
                loo = _loo_metrics_lognormal(model, n_real, model._study_Y_psi)
                block = None  # block-LOO not implemented for log-Y models
            else:
                # Always report the REAL-ONLY metric as the primary number — this
                # is apples-to-apples comparable across anchored and unanchored
                # variants. For anchored variants we additionally report the
                # WITH-ANCHORS metric (physics-constraint satisfaction).
                loo = loo_metrics(model, n_real, include_anchors=False)
                try:
                    block = block_loo_metrics(model, n_real, include_anchors=False)
                except Exception as e:
                    print(f"  [block-LOO failed for {name} seed={seed}: {e}]")
                    block = None
                if has_anchors:
                    try:
                        loo_w = loo_metrics(model, n_real, include_anchors=True)
                        block_w = block_loo_metrics(model, n_real, include_anchors=True)
                        block_a_only = block_loo_metrics(
                            model, n_real, anchors_only=True
                        )
                        # Extract the learnt log-time offset for diagnostics, if any.
                        offset_str = ""
                        input_tf = getattr(model, "input_transform", None)
                        if input_tf is not None:
                            for sub_tf in (
                                input_tf.values() if hasattr(input_tf, "values") else []
                            ):
                                if isinstance(sub_tf, _LearnableLogTimeTransform):
                                    offset_str = (
                                        f" learnt_offset={sub_tf.offset.item():.4f}"
                                    )
                                    break
                        print(
                            f"  [{name} seed={seed}] "
                            f"real-only block-LOO={block['rmse']:.1f}  "
                            f"anchor-only block-LOO={block_a_only['rmse']:.1f}  "
                            f"combined block-LOO={block_w['rmse']:.1f}  "
                            f"(combined LOO RMSE={loo_w['rmse']:.1f}){offset_str}"
                        )
                    except Exception as e:
                        print(f"  [anchors-in-CV metrics failed for {name}: {e}]")
            per_seed_loo.append(loo)
            if block is not None:
                per_seed_block.append(block)

            if held_out is not None:
                X_held, Y_held_psi = held_out
                ho = held_out_metrics(model, X_held, Y_held_psi)
                per_seed_ho.append(ho)
                if block is not None:
                    print(
                        f"[{name:>20s}] seed={seed}  "
                        f"LOO: RMSE={loo['rmse']:6.1f} MLPD={loo['mean_lpd']:6.2f}  ||  "
                        f"BLOCK-LOO: RMSE={block['rmse']:6.1f} MLPD={block['mean_lpd']:6.2f}  ||  "
                        f"HO: RMSE={ho['rmse']:6.1f} MLPD={ho['mean_lpd']:6.2f}"
                    )
                else:
                    print(
                        f"[{name:>20s}] seed={seed}  "
                        f"LOO: RMSE={loo['rmse']:6.1f} MLPD={loo['mean_lpd']:6.2f} cov95={loo['coverage_95']:.3f}  "
                        f"||  HO: RMSE={ho['rmse']:6.1f} MLPD={ho['mean_lpd']:6.2f} cov95={ho['coverage_95']:.3f}"
                    )
            else:
                if block is not None:
                    print(
                        f"[{name:>20s}] seed={seed}  "
                        f"LOO: RMSE={loo['rmse']:6.1f} MLPD={loo['mean_lpd']:6.2f} cov95={loo['coverage_95']:.3f}  "
                        f"||  BLOCK-LOO ({block['n_blocks']} blocks): "
                        f"RMSE={block['rmse']:6.1f} MLPD={block['mean_lpd']:6.2f} cov95={block['coverage_95']:.3f}"
                    )
                else:
                    print(
                        f"[{name:>20s}] seed={seed}: "
                        f"RMSE={loo['rmse']:6.1f}  MAE={loo['mae']:6.1f}  "
                        f"MLPD={loo['mean_lpd']:6.2f}  KS={loo['pit_ks']:.3f}  "
                        f"cov95={loo['coverage_95']:.3f}  CRPS={loo['crps']:6.1f}"
                    )

        agg_loo = {}
        for k in per_seed_loo[0]:
            xs = torch.tensor([m[k] for m in per_seed_loo])
            agg_loo[k] = (xs.mean().item(), xs.std(unbiased=False).item())
        results[name] = {"loo": agg_loo}
        if per_seed_block:
            agg_block = {}
            for k in per_seed_block[0]:
                xs = torch.tensor([m[k] for m in per_seed_block])
                agg_block[k] = (xs.mean().item(), xs.std(unbiased=False).item())
            results[name]["block_loo"] = agg_block
        if per_seed_ho:
            agg_ho = {}
            for k in per_seed_ho[0]:
                xs = torch.tensor([m[k] for m in per_seed_ho])
                agg_ho[k] = (xs.mean().item(), xs.std(unbiased=False).item())
            results[name]["held_out"] = agg_ho

    print()
    print(
        _format_table(
            results,
            has_held_out=held_out is not None,
            has_block=any("block_loo" in v for v in results.values()),
        )
    )
    return results


def _format_table(
    results, *, has_held_out: bool = False, has_block: bool = False
) -> str:
    cols = ["rmse", "mae", "mean_lpd", "pit_ks", "coverage_95", "crps"]
    out = []
    table_specs = [("LOO (training-set leave-one-row-out)", "loo")]
    if has_block:
        table_specs.append(("BLOCK-LOO (leave-one-composition-out)", "block_loo"))
    if has_held_out:
        table_specs.append(("HELD-OUT (the n_total - n_subset complement)", "held_out"))

    if len(table_specs) > 1:
        for label, key in table_specs:
            out.append(f"  {label}:")
            header = f"  {'variant':<22s}  " + "  ".join(f"{c:>11s}" for c in cols)
            out.append(header)
            out.append("  " + "-" * (len(header) - 2))
            for name, agg in results.items():
                if key not in agg:
                    continue
                row = f"  {name:<22s}  " + "  ".join(
                    f"{agg[key][c][0]:7.2f}±{agg[key][c][1]:.2f}" for c in cols
                )
                out.append(row)
            out.append("")
    else:
        header = f"  {'variant':<22s}  " + "  ".join(f"{c:>11s}" for c in cols)
        out.append(header)
        out.append("  " + "-" * (len(header) - 2))
        for name, agg in results.items():
            row = f"  {name:<22s}  " + "  ".join(
                f"{agg['loo'][c][0]:7.2f}±{agg['loo'][c][1]:.2f}" for c in cols
            )
            out.append(row)
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS),
        help=f"Subset of variants to run (default: all). Choices: {list(VARIANTS)}",
    )
    parser.add_argument(
        "--subset_n",
        type=int,
        default=None,
        help="If set, randomly subsample the training data. With "
        "--holdout_unit=row this is a number of rows; with "
        "--holdout_unit=composition this is a number of unique "
        "compositions (and all their time-point measurements stay "
        "together).",
    )
    parser.add_argument(
        "--subset_seed",
        type=int,
        default=0,
        help="Seed for the data subsetting (so subsets are reproducible across variants).",
    )
    parser.add_argument(
        "--holdout_unit",
        type=str,
        choices=("row", "composition"),
        default="row",
        help="Granularity of the held-out set. 'row' (default) holds out "
        "individual rows uniformly at random — beware: this leaks "
        "information across the time-curve of any composition with "
        "multiple measurements. 'composition' holds out entire "
        "compositions and all of their time-point rows together — "
        "the realistic 'predict an unseen mix' scenario.",
    )
    args = parser.parse_args()
    run(
        seeds=args.seeds,
        names=args.variants,
        subset_n=args.subset_n,
        subset_seed=args.subset_seed,
        holdout_unit=args.holdout_unit,
    )
