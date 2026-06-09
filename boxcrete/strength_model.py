# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Strength GP V2 — the deployed production model (2026-05-17 onwards).

This module hosts the public fit / load API for the V2 strength-GP
architecture::

    Multi-Matern (B'') kernel (blind + source-specific + RBF-time)
        + F5_alllog engineered features
        + multiplicative time gate h(t)·k(x,x')·h(t')
        + heteroscedastic gated noise σ²(t) = h(t)² · σ²_global
        + Y/y_max scaling + ZeroMean prior
        + MLL fit with within-group shrinkage prior on lengthscales

Block-LOO RMSE: 680 psi at full data (n=144 compositions). See
``experiments/STRENGTH_GP_BENCHMARK.md`` for the full architecture
study and rejected variants.

Public API:
  * :func:`fit_strength_gp` — fit a fresh V2 model on raw (X, Y) data.
  * :func:`load_pretrained_strength_gp` — deserialise the deployed
    V2 strength GP from ``docs/model/strength_model.pt`` without
    re-fitting.

Building blocks (kernels, likelihoods, priors, features, transforms)
live in :mod:`boxcrete.kernels`, :mod:`boxcrete.likelihoods`,
:mod:`boxcrete.priors`, and :mod:`boxcrete.features` and are
re-exported by :mod:`boxcrete` for users / variant authors.

The legacy V1 architecture (single-Matern + anchors + scalar noise)
is intentionally retained alongside V2 for instructional purposes via
:mod:`boxcrete.strength_model_legacy`'s ``get_strength_gp_input_transform``
helper, making it easy for new readers to compare V2's input pipeline
against the prior-generation baseline. The full V1 fit factory and the
research variant catalog (which exercised V1) are not part of this
release.

New code should import :func:`fit_strength_gp` from ``boxcrete`` (this
module) — the V2 production model.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import (
    AffineInputTransform,
    AppendFeatures,
    ChainedInputTransform,
    Log10,
    Normalize,
)
from botorch.utils.constraints import LogTransformedInterval
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from boxcrete.features import (
    F5_ALLLOG_FEATURES,
    GATE_TAU,
    IDX,
    append_engineered_features_callable,
    augmented_bounds,
    max_scale_Y,
)
from boxcrete.kernels import make_gated_strength_kernel_builder
from boxcrete.likelihoods import GatedGaussianLikelihood
from boxcrete.utils import derive_bounds_from_X, load_concrete_strength

_CHAMPION_VARIANT = "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean"


# Module-level optimizer override consulted by the V2 fit. ``None``
# = production defaults. A non-None dict is passed straight through as
# ``optimizer_kwargs`` to ``fit_gpytorch_mll`` (e.g.
# ``{"options": {"maxiter": 20}}``). The public ``fit_strength_gp``
# exposes ``max_optimizer_iter`` / ``num_restarts`` kwargs that set/
# restore this override around the fit call (see ``fit_strength_gp``).
#
# Concurrency note: this is process-global mutable state. Single-threaded
# sequential reentry is safe (the try/finally in ``fit_strength_gp``
# restores the prior value), but two concurrent fits in the same
# interpreter (threading or pytest-xdist with ``loadgroup``) can race.
# If you ever need true concurrent fits, replace this with a
# ``contextvars.ContextVar`` or pass the override down through the call
# stack instead of via this module global.
_OPTIMIZER_OVERRIDE: dict | None = None


def _fit_gpytorch_mll_with_override(mll) -> None:
    """``fit_gpytorch_mll`` wrapper that respects ``_OPTIMIZER_OVERRIDE``."""
    if _OPTIMIZER_OVERRIDE is None:
        fit_gpytorch_mll(mll)
    else:
        fit_gpytorch_mll(mll, optimizer_kwargs=_OPTIMIZER_OVERRIDE)


def _get_v2_input_transform(
    d_in: int,
    bounds: Tensor,
    feature_names: Sequence[str],
    X_for_bounds: Tensor,
    log_time_offset: float = 1.0,
) -> ChainedInputTransform:
    """Build the V2 strength GP's ChainedInputTransform.

    Four steps applied in order:
      1. ``AppendFeatures(f=...)`` — appends the 7 F5_alllog engineered
         features (10 → 17 dims) using BoTorch's primitive with a
         callable that delegates to ``FEATURE_BUILDERS``.
      2. ``AffineInputTransform`` (``reverse=True``) on the time column —
         adds ``log_time_offset`` (= 1.0) so we never take ``log10(0)``.
      3. ``Log10`` on the time column — replaces ``t + offset`` with
         ``log10(t + offset)``.
      4. ``Normalize`` on the **non-time dims only** — rescales the
         raw composition columns + the appended engineered columns to
         ``[0, 1]`` (everything except ``IDX["time"]``).
         The time column intentionally bypasses Normalize: with the
         gated kernel ``h(t) = 1 - exp(-t / τ)``, mapping the smallest
         training time ``t = 1`` to post-Normalize ``0`` would trigger
         ``h(0) = 0`` and gate those rows out of the kernel — a
         multi-hundred-psi block-LOO regression
         (see ``experiments/STRENGTH_GP_BENCHMARK.md`` §3.6).

    Composed from BoTorch's ``AffineInputTransform`` (adds offset) +
    ``Log10`` primitives — the V2 strength GP's canonical log-time
    chain.
    """
    augmented = augmented_bounds(X_for_bounds, bounds, feature_names)
    d_aug = d_in + len(feature_names)
    time_index = [d_in - 1]
    derive = AppendFeatures(
        f=append_engineered_features_callable(feature_names),
        indices=list(range(d_in)),
        transform_on_train=True,
        transform_on_eval=True,
        transform_on_fantasize=True,
    )

    # Steps 2+3: t -> log10(t + offset). Composed from BoTorch primitives
    # (Affine reverse=True adds offset; Log10 takes log10).
    log_offset = AffineInputTransform(
        d_aug,
        coefficient=torch.ones(1, dtype=torch.float64),
        offset=torch.full((1,), log_time_offset, dtype=torch.float64),
        indices=time_index,
        reverse=True,
    )
    log = Log10(indices=time_index)

    # Step 4: Normalize on non-time dims only (BoTorch's Normalize
    # supports `indices=` to scope which columns it touches).
    non_time_dims = [i for i in range(d_aug) if i not in time_index]
    tf_normalize = Normalize(
        d_aug,
        indices=torch.tensor(non_time_dims),
        bounds=augmented,
    )
    return ChainedInputTransform(
        derive=derive,
        log_offset=log_offset,
        log=log,
        normalize=tf_normalize,
    )


def _fit_v2_strength_gp(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor | None,
    bounds: Tensor,
    seed: int = 0,
) -> SingleTaskGP:
    """Fit the deployed V2 production configuration.

    The variant identifier is
    ``B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean`` (see
    :data:`_CHAMPION_VARIANT`). This module is the canonical
    implementation; ``test/test_pretrained_loader_fidelity.py`` guards
    that the deployed ``docs/model/strength_model.pt`` round-trips
    through this fit factory's loader without drift.
    """
    # Yvar is accepted for signature compat with the research factory but
    # is unused: V2 uses a learnable gated heteroscedastic
    # noise (see ``GatedGaussianLikelihood``), not per-row Yvar.
    # The explicit ``del`` documents this accept-and-discard so static
    # analysers stop flagging the param as unused.
    del Yvar
    # Force float64 throughout the fit. ``X`` already arrives as
    # float64 (set in the data loader), but ``X_bounds`` and any
    # tensor created inside the kernel / GP without an explicit
    # ``dtype=`` kwarg inherit ``torch.get_default_dtype()`` which
    # is float32 by default. float32 has only ~7 decimal digits of
    # precision; the L-BFGS-B optimizer is sensitive to FP rounding
    # near saddles in the multi-modal MLL surface, and that ~7-digit
    # precision is exactly the magnitude where cross-host CPU
    # rounding differences (different physical Azure VMs, different
    # microcode) push the optimizer into different local optima
    # (e.g., wb_ratio railing at the kernel cap on some hosts).
    # Casting everything to float64 (~15-16 digit precision)
    # eliminates this basin variability empirically. To debug a
    # suspected fit-determinism issue, fit the GP N times with the
    # same seed and compare per-sub-kernel lengthscales (canonical
    # walk: ``model.covar_module.base_kernel.kernels[i].base_kernel``
    # — ``test/test_kernel_layout.py`` pins this contract); cross-run
    # bit-identical lengthscales at float64 indicate the BLAS path
    # is deterministic on your host.
    if bounds is not None:
        bounds = bounds.to(torch.float64)
    X = X.to(torch.float64)
    Y = Y.to(torch.float64)
    torch.manual_seed(seed)
    feature_names = F5_ALLLOG_FEATURES
    d_in = X.shape[-1]
    d_aug = d_in + len(feature_names)

    Y_z, y_mean, y_std = max_scale_Y(Y)

    likelihood = GatedGaussianLikelihood(
        time_idx=IDX["time"],
        gate_tau=GATE_TAU,
        noise_constraint=LogTransformedInterval(
            1e-6,
            1.0,
            initial_value=1e-1,
        ),
    )
    # MLL path doesn't pass X to the likelihood; stash raw train times so
    # ``GatedGaussianLikelihood._shaped_noise_covar`` has a value to read.
    # See ``boxcrete/likelihoods.py::set_train_times`` for the two-phase
    # contract — the second call below is also raw days (BoTorch returns
    # raw inputs from ``train_inputs``).
    likelihood.set_train_times(X[..., IDX["time"]])

    # V2 uses the time-gated kernel builder (NOT the bare multi-Matern).
    kernel_builder = make_gated_strength_kernel_builder(
        gate_tau=GATE_TAU,
    )
    kernel = kernel_builder(d_aug)

    input_tf = _get_v2_input_transform(
        d_in=d_in,
        bounds=bounds,
        feature_names=feature_names,
        X_for_bounds=X,
    )

    model = SingleTaskGP(
        train_X=X,
        train_Y=Y_z,
        covar_module=kernel,
        input_transform=input_tf,
        likelihood=likelihood,
        outcome_transform=None,
        mean_module=ZeroMean(),
    )

    # Re-stash train times after model construction. Note: BoTorch's
    # ``train_inputs`` returns RAW inputs (the input_transform is applied
    # at ``forward()`` time), so this pulls raw days, not
    # log10(t+1) values. For the strength dataset (raw t ≥ 1, gate_tau =
    # 0.05) ``h(raw_t / tau)`` saturates to ≈1.0, so the gated noise
    # diagonal is empirically equivalent to bare σ² at training. The
    # kernel-side gate ``h(t1) k h(t2)`` is unaffected; it sees
    # post-transform time straight from the augmented input.
    with torch.no_grad():
        train_input_times = model.train_inputs[0][..., IDX["time"]]
    model.likelihood.set_train_times(train_input_times)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _fit_gpytorch_mll_with_override(mll)

    with torch.no_grad():
        output = model(*model.train_inputs)
        model._study_mll_per_row = float(mll(output, model.train_targets).item())
    # Register Y-scaling stats as buffers so they round-trip through
    # ``state_dict`` (consumed by ``load_pretrained_strength_gp``). Plain
    # Python attributes on a torch.nn.Module would not be restored by
    # ``model.load_state_dict(...)``, which would silently leave the
    # loader's ``y_max`` recomputed against whatever ``data/`` happens
    # to look like at load time — a sharp edge if the CSV ever changes.
    model.register_buffer("_study_y_mean_buf", y_mean.squeeze().detach().clone())
    model.register_buffer("_study_y_std_buf", y_std.squeeze().detach().clone())
    # Plain-attribute aliases for the legacy access paths (notebooks,
    # research scripts) that read ``model._study_y_std`` directly.
    model._study_y_mean = model._study_y_mean_buf
    model._study_y_std = model._study_y_std_buf
    # Stash raw 10-dim training X (real rows only) for use by
    # phantom_anchor_metrics to construct virtual (c, t=0) test points.
    model._study_X_train_raw = X.detach().clone()
    return model


def fit_strength_gp(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor | None = None,
    X_bounds: Tensor | None = None,
    *,
    seed: int = 0,
    max_optimizer_iter: int | None = None,
    num_restarts: int | None = None,
) -> SingleTaskGP:
    """Fit the V2 strength GP — the deployed production architecture.

    Multi-Matern + gated kernel + gated noise + F5_alllog engineered
    features + Y/y_max scaling + ZeroMean prior + MLL fit. See module
    docstring and ``experiments/STRENGTH_GP_BENCHMARK.md`` for details.

    Args:
        X: ``[n, 10]`` raw input — composition (9 dims) + time (1 dim).
            Composition order matches ``boxcrete.utils.DEFAULT_X_COLUMNS``:
            Cement, Fly Ash, Slag, Water, HRWR, Fine Aggregate,
            Coarse Aggregates, Material Source, Temperature, Time.
        Y: ``[n, 1]`` strength values in psi.
        Yvar: Ignored — V2 uses a learnable global heteroscedastic-gated
            noise rather than per-row Yvar. Kept for backward compat.
        X_bounds: ``[2, 10]`` optional lower/upper bounds; if ``None``,
            derived from X.
        seed: Torch RNG seed for fit determinism.
        max_optimizer_iter: Optional cap on the inner L-BFGS iteration
            count used by ``fit_gpytorch_mll`` (passed via
            ``optimizer_kwargs={"options": {"maxiter": ...}}``). ``None``
            (default) preserves production behaviour. Set to a small
            integer in test suites to drop fit cost from ~85s to <1s
            without sacrificing structural correctness.
        num_restarts: Optional override for the number of optimizer
            restarts. ``None`` (default) preserves production behaviour.

    Returns:
        A ``SingleTaskGP`` with the V2 architecture installed.

    Example:
        >>> from boxcrete import fit_strength_gp, load_concrete_strength
        >>> data = load_concrete_strength()
        >>> X, Y, Yvar, bounds = data.strength_data
        >>> model = fit_strength_gp(X, Y, Yvar, bounds)
    """
    # Single-output guard: V2 is a univariate strength regressor.
    # Catch shape mistakes upstream so callers see a clean ValueError
    # rather than an opaque tensor/optimizer crash deep in the fit.
    if Y.ndim != 2 or Y.shape[-1] != 1:
        raise ValueError(
            f"fit_strength_gp expects Y with shape [n, 1]; got {tuple(Y.shape)}."
        )

    # X_bounds fallback: derive from X if not provided. The variant
    # catalog and many notebooks pass explicit bounds, but the public
    # API documents this as optional, so honour that contract.
    #
    # Defensive against degenerate inputs:
    #  * 2-D-only: a batched X (>= 3D) reduces over the wrong dim.
    #  * Non-empty: at least one row.
    #  * Non-zero-width: a constant column would otherwise produce
    #    a [lo == hi] interval, which makes the downstream Normalize
    #    transform divide by zero. Mirrors the slump GP's defense
    #    (see ``boxcrete/slump_model.py``).
    if X_bounds is None:
        if X.ndim != 2:
            raise ValueError(
                f"fit_strength_gp: X_bounds=None requires a 2-D X of shape "
                f"[n, d]; got X.ndim={X.ndim}."
            )
        if X.shape[0] < 1:
            raise ValueError("fit_strength_gp: X_bounds=None requires X.shape[0] >= 1.")
        X_bounds = derive_bounds_from_X(X)

    # Build the optimizer-override dict from kwargs. ``None`` for both
    # = production defaults; a non-None value sets the override around
    # the fit call and restores it afterwards (try/finally) so failures
    # don't poison subsequent fits in the same process.
    #
    # Note: ``num_restarts`` is recorded as a kwarg for forward-compat
    # but is not currently consumed by the underlying scipy-backed
    # ``fit_gpytorch_mll`` (it would be the right knob for a torch.optim
    # path). For now only ``max_optimizer_iter`` is propagated. The
    # explicit ``del`` documents the deliberate accept-and-discard so
    # static analysers stop flagging the param as unused.
    del num_restarts
    override: dict | None = None
    if max_optimizer_iter is not None:
        override = {"options": {"maxiter": int(max_optimizer_iter)}}

    # The full V2 fit factory lives at ``_fit_v2_strength_gp``
    # above; this public wrapper just plumbs through the
    # ``max_optimizer_iter`` / ``num_restarts`` overrides via the
    # module-global ``_OPTIMIZER_OVERRIDE`` (see comment block at
    # the top of this module for concurrency caveats).
    global _OPTIMIZER_OVERRIDE
    prior_override = _OPTIMIZER_OVERRIDE
    try:
        if override is not None:
            _OPTIMIZER_OVERRIDE = override
        model = _fit_v2_strength_gp(X, Y, Yvar, X_bounds, seed=seed)
    finally:
        _OPTIMIZER_OVERRIDE = prior_override
    return model


def load_pretrained_strength_gp(
    state_dict_path: str | None = None,
) -> SingleTaskGP:
    """Reconstruct the deployed V2 strength GP from
    ``docs/model/strength_model.pt`` without re-fitting.

    Returns a fully-functional ``SingleTaskGP`` whose ``posterior(X)``
    output matches the predictions Python made at export time. Use
    this for downstream code that needs the deployed model's
    predictions but doesn't want to (or can't) re-fit — notebooks,
    alternative ports, regression checks, etc.

    Implementation: standard PyTorch ``state_dict`` round-trip. We
    rebuild a fresh skeleton via ``_fit_v2_strength_gp(...,
    max_optimizer_iter=0)`` (constructs the architecture without
    optimization), then ``model.load_state_dict(...)`` to overwrite
    every learnable parameter and registered buffer with the values
    captured at export time.

    Args:
        state_dict_path: Optional path to a ``.pt`` state-dict file.
            ``None`` (default) loads from the canonical
            ``docs/model/strength_model.pt`` in the repo.

    Returns:
        A ``SingleTaskGP`` configured with the V2 strength-GP architecture
        and the parameters baked into ``strength_model.pt``. The
        returned model is in ``eval()`` mode.

    Fidelity:
        ``test/test_pretrained_loader_fidelity.py`` asserts that this
        function's posterior matches the Python reference predictions
        baked into ``docs/model/test_vectors.json`` (which is exported
        atomically alongside ``strength_model.pt`` by
        ``experiments/regenerate_strength_json.py``) at ``atol=1e-5``
        for the mean and ``atol=1e-3`` for the variance.

    JS deployment:
        This Python loader does NOT touch ``strength.json``. The browser
        explorer (``docs/gp.mjs`` / ``docs/gp_v2_fast.mjs``) consumes the
        human-readable JSON companion artifact (emitted by the same regen
        pipeline) and rebuilds its own posterior from the kernel
        ingredients there. The Python ``.pt`` and the JS ``strength.json``
        are independent serialisations of the same fitted model.

    Notes:
        Requires the canonical training data to be loadable via
        ``boxcrete.load_concrete_strength()`` (needed to construct the
        skeleton SingleTaskGP). The function does NOT re-fit; it only
        deserialises hyperparameters into a freshly-constructed model
        whose architecture matches what
        ``regenerate_strength_json.py`` exported.
    """
    if state_dict_path is None:
        state_dict_path = str(
            Path(__file__).resolve().parent.parent
            / "docs"
            / "model"
            / "strength_model.pt"
        )

    # 1. Build a fresh V2 model with NO optimization (maxiter=0
    #    runs the construction code path but exits before any
    #    parameter update). This produces the right architecture +
    #    correct shape; we then overwrite the converged hyperparams.
    data = load_concrete_strength()
    X, Y, Yvar, bounds = data.strength_data

    global _OPTIMIZER_OVERRIDE
    prior_override = _OPTIMIZER_OVERRIDE
    try:
        _OPTIMIZER_OVERRIDE = {"options": {"maxiter": 0}}
        model = _fit_v2_strength_gp(X, Y, Yvar, bounds, seed=0)
    finally:
        _OPTIMIZER_OVERRIDE = prior_override

    # 2. Load the saved state dict and overwrite all learnable
    #    parameters + registered buffers. ``weights_only=True`` is
    #    safe (no arbitrary code execution) and matches PyTorch's
    #    recommended security posture for trusted-but-don't-execute
    #    artifacts.
    state = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    # 3. Switch to eval mode and force GPyTorch to recompute the
    #    cached Cholesky / alpha from the just-loaded hyperparameters.
    #    train() then eval() invalidates the cache so the next
    #    posterior() call rebuilds everything from scratch.
    model.train()
    model.eval()
    return model


__all__ = [
    "fit_strength_gp",
    "load_pretrained_strength_gp",
]
