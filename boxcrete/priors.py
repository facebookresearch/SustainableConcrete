# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Lengthscale priors for the V2 strength GP.

Hosts:

  * :class:`WithinGroupShrinkagePrior` — the original soft-tying prior
    on cementitious-binder and aggregate ARD lengthscales (within a
    single Matern's lengthscale tensor).
  * :class:`ComposedLengthscalePrior` — the v5 default (introduced in
    the materials-classes-and-lengthscale-prior stack): the within-group
    quadratic penalty AND a per-element LogNormal baseline mirroring
    BoTorch's ARD-Matern default. Adding the LogNormal baseline pulls
    the previously-unprior'd lengthscales (Water, HRWR, Source, Temp,
    Time, and the engineered features) toward √d-scale and prevents the
    drift past the rail-detection threshold that produced the most
    recent CI flake (``Time: ℓ = 288.13 > cap 100``).
  * :class:`CrossComponentLengthscalePrior` — soft-tying prior across
    LENGTHSCALES OF DIFFERENT KERNEL SUB-COMPONENTS (blind Matern,
    source-specific Matern, additive RBF on Time). The strength kernel
    has three additive sub-kernels each carrying a Time lengthscale,
    and the V5 ablation showed they fit to wildly different values
    (97.94 / 16.13 / 72.90 at the ML optimum) — symptomatic of the
    optimiser depending on a single sub-kernel for composition × time
    interaction while letting the others drift to whatever rail. This
    prior expresses "the three log-Time-lengthscales should be in the
    same order of magnitude" without forcing them to be equal.

Public re-exports go through :mod:`boxcrete` for ergonomics.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from gpytorch.priors import LogNormalPrior

# --- Within-group lengthscale shrinkage prior --------------------------------

# Material-class groupings used by the production within-group shrinkage
# prior on the Matern kernel's lengthscales. Indices refer to
# DEFAULT_X_COLUMNS: Cement, Fly Ash, Slag, Water, HRWR, Fine Aggregate,
# Coarse Aggregates, Material Source, Temp, Time.
_BINDER_LENGTHSCALE_GROUP = (0, 1, 2)  # Cement, Fly Ash, Slag
_AGGREGATE_LENGTHSCALE_GROUP = (5, 6)  # Fine Aggregate, Coarse Aggregates

# σ → 0 effectively hard-ties the members of each group to a shared lengthscale.
# Empirical LOO CV (n=647 public rows) found σ=0.001 minimises held-out RMSE
# while keeping all lengthscales well below the identifiability cap. The
# headline numbers are documented in :class:`WithinGroupShrinkagePrior`'s
# docstring (Empirical comparison section).
_LENGTHSCALE_SHRINKAGE_SIGMA = 0.001


class WithinGroupShrinkagePrior(LogNormalPrior):
    """Soft hard-tying prior on Matern ARD lengthscales within material groups.

    Penalises within-group variance of log-lengthscales. Encodes the
    domain fact that interchangeable materials (e.g., cementitious binders
    or aggregates) should have similar smoothness scales in the GP. With
    ``sigma → 0`` this approaches a hard tying constraint that forces
    each group to share a single lengthscale.

    Why this prior exists
    ---------------------
    Several composition features in ``data/boxcrete_data.csv`` are
    under-sampled. In particular, Fly Ash is zero in the majority of rows
    (most concretes use only Cement + Slag), and Coarse Aggregates are
    zero for the mortar half of the dataset. Without a prior, ARD pushes
    the corresponding Matern lengthscales to the optimiser's upper bound
    (``1e3`` in normalised input space), making the GP effectively
    insensitive to those features — the website's interactive sliders
    for Fly Ash and Coarse Aggregates would not respond to user input.

    This prior softly ties the lengthscales of materials that play
    interchangeable physical roles, so the well-identified members of
    each group (Cement, Fine Aggregate) supply usable scale information
    to their under-identified peers (Fly Ash, Coarse Aggregates).

    Empirical comparison
    --------------------
    Analytical LOO CV (via ``boxcrete.compute_loo_cv``) on n=647 public
    strength rows; lower RMSE is better:

    +-----------------------------------------+-----------+
    | Variant                                 | LOO RMSE  |
    +=========================================+===========+
    | No prior (Fly Ash & Coarse Agg railed)  |  772 psi  |
    | Within-group shrinkage σ=0.50           |  754 psi  |
    | Within-group shrinkage σ=0.10           |  731 psi  |
    | Within-group shrinkage σ=0.001 (prod)   |  **725 psi** |
    +-----------------------------------------+-----------+

    The σ → 0 limit Pareto-dominates every alternative we evaluated:
    per-feature LogNormal priors, Cauchy / Student-t shrinkage,
    asymmetric per-group widths, Cement-anchored shrinkage, and additive
    kernel decompositions.

    Mathematical form
    -----------------
    The prior contributes the following log-density (up to a constant)
    to the marginal log-likelihood::

        log p(ℓ) = -Σ_g Σ_{i ∈ g} (log ℓ_i - mean_{j ∈ g} log ℓ_j)² / (2 σ_g²)

    where ``g`` ranges over the configured groups. Subclassing
    ``LogNormalPrior`` lets the prior satisfy GPyTorch's
    ``isinstance(_, Prior)`` check without reimplementing the Prior
    interface; the inherited ``loc`` / ``scale`` are unused placeholders.

    Args:
        groups_with_sigma: List of ``(dim_indices, sigma)`` tuples. Each
            entry contributes a within-group penalty with its own width.
            ``sigma → 0`` hard-ties the group; ``sigma → ∞`` is uniform.
        dim: Dimensionality of the lengthscale tensor (matches the
            kernel's ``ard_num_dims``).
    """

    def __init__(
        self,
        groups_with_sigma: list[tuple[tuple[int, ...], float]],
        dim: int,
    ):
        # Pass scale as a fully-shaped, contiguous tensor (not the
        # scalar ``1.0``). When scale is a Python scalar, gpytorch's
        # MultivariateNormal base class calls ``scale.expand_as(loc)``
        # internally, producing a tensor with aliased storage (one
        # underlying element backing all ``dim`` positions). PyTorch
        # 2.12+'s ``load_state_dict`` raises on writing into such
        # aliased destinations: "more than one element of the written-to
        # tensor refers to a single memory location. Please clone()...".
        # The contiguous (1, dim) shape avoids the expand path entirely.
        super().__init__(
            loc=torch.zeros(1, dim, dtype=torch.float64),
            scale=torch.ones(1, dim, dtype=torch.float64),
        )
        self._groups_with_sigma = groups_with_sigma

    def log_prob(self, x):
        """Within-group quadratic penalty on log-lengthscales.

        Returns a tensor with the same shape as ``x`` whose **sum** equals
        the desired scalar penalty ``- ½ Σ_g (1/σ_g²) Σ_{i ∈ g} (log ℓ_i − μ_g)²``,
        where ``μ_g`` is the per-group mean of the in-group log-lengthscales.

        Why this shape (and not a scalar): GPyTorch wraps prior-augmented
        marginal-likelihoods via ``prior.log_prob(ℓ).sum()``. Returning the
        already-summed scalar would double-count it; returning per-element
        contributions of the form ``total / x.numel()`` makes the
        framework's element-wise sum re-aggregate to exactly ``total``.
        """
        log_x = x.log()
        flat_log = log_x.flatten()
        total = torch.zeros((), dtype=x.dtype, device=x.device)
        for grp, sigma in self._groups_with_sigma:
            if len(grp) < 2:
                continue
            grp_log = flat_log[list(grp)]
            sq_dev = ((grp_log - grp_log.mean()) ** 2).sum()
            total = total - 0.5 * sq_dev / (sigma**2)
        return total / x.numel() * torch.ones_like(x)


def _lognormal_baseline_loc(dim: int) -> float:
    """BoTorch's standard ARD-Matern default ``loc`` for the LogNormal
    lengthscale prior: ``sqrt(2) + 0.5 * ln(dim)``. Centred so the mode
    of the LogNormal is at ``sqrt(dim)`` (the unit-cube ARD scale).
    """
    return math.sqrt(2.0) + 0.5 * math.log(max(int(dim), 1))


_LOGNORMAL_BASELINE_SCALE = math.sqrt(3.0)
"""BoTorch's standard ARD-Matern default ``scale`` for the LogNormal
lengthscale prior. Wide enough to admit lengthscales from ~0.05·√d to
~20·√d as roughly equally probable a priori; narrow enough to suppress
the rail-detection-threshold drift past ``ℓ > 100``."""


class ComposedLengthscalePrior(LogNormalPrior):
    """Composition of :class:`WithinGroupShrinkagePrior` and the
    BoTorch-default LogNormal ARD-Matern baseline.

    Per-element ``log_prob(ℓ)`` returns

        log p(ℓ_i) = log LogNormal(ℓ_i; loc, scale) + W_total / d

    where:
      * ``log LogNormal(ℓ_i; loc, scale)`` is the standard LogNormal
        log-density at element ``i`` with shape parameters
        ``loc = sqrt(2) + 0.5·ln(d)`` and ``scale = sqrt(3)``.
      * ``W_total`` is the WithinGroupShrinkagePrior's total
        (group-summed) penalty across the same ``ℓ`` vector.
      * ``d`` is ``ℓ.numel()``.

    GPyTorch wraps prior-augmented marginal-likelihoods via
    ``prior.log_prob(ℓ).sum()``. The element-wise sum re-aggregates the
    LogNormal log-density (full) and re-aggregates ``W_total / d * 1`` to
    exactly ``W_total``. Combined: ``Σ_i log LogNormal_i + W_total``.

    Args:
        groups_with_sigma: ``[(group, sigma), ...]`` — the within-group
            shrinkage groups (binder, aggregate). Same format as
            :class:`WithinGroupShrinkagePrior`.
        dim: lengthscale dimensionality.
        lognormal_loc: optional override for the per-element LogNormal
            ``loc``; defaults to BoTorch's ``sqrt(2) + 0.5·ln(dim)``.
        lognormal_scale: optional override for the per-element LogNormal
            ``scale``; defaults to BoTorch's ``sqrt(3)``.
    """

    def __init__(
        self,
        groups_with_sigma: list[tuple[tuple[int, ...], float]],
        dim: int,
        lognormal_loc: float | None = None,
        lognormal_scale: float = _LOGNORMAL_BASELINE_SCALE,
    ):
        loc = (
            _lognormal_baseline_loc(dim)
            if lognormal_loc is None
            else float(lognormal_loc)
        )
        # See WithinGroupShrinkagePrior.__init__ for the rationale on
        # passing fully-shaped contiguous (1, dim) float64 tensors instead
        # of scalar loc/scale (PyTorch 2.12+ aliased-storage error).
        super().__init__(
            loc=torch.full((1, dim), float(loc), dtype=torch.float64),
            scale=torch.full((1, dim), float(lognormal_scale), dtype=torch.float64),
        )
        self._groups_with_sigma = groups_with_sigma
        self._dim = dim
        self._lognormal_loc = float(loc)
        self._lognormal_scale = float(lognormal_scale)

    def _within_group_total(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the WithinGroupShrinkagePrior's scalar total penalty
        on ``x``.
        """
        log_x = x.log().flatten()
        total = torch.zeros((), dtype=x.dtype, device=x.device)
        for grp, sigma in self._groups_with_sigma:
            if len(grp) < 2:  # pragma: no cover
                continue
            grp_log = log_x[list(grp)]
            sq_dev = ((grp_log - grp_log.mean()) ** 2).sum()
            total = total - 0.5 * sq_dev / (sigma**2)
        return total

    def log_prob(self, x):
        """Per-element log-density: LogNormal baseline + spread
        within-group penalty.
        """
        # LogNormal baseline (per-element). Reuse the parent class's
        # implementation, which already produces a per-element tensor of
        # the same shape as x.
        lognormal = super().log_prob(x)
        within_total = self._within_group_total(x)
        return lognormal + (within_total / x.numel()) * torch.ones_like(x)


class CrossComponentLengthscalePrior(LogNormalPrior):  # pragma: no cover
    """Soft-tying prior on lengthscale parameters across DIFFERENT
    kernel sub-modules.

    Use case: the V2 strength kernel is an additive sum

        ScaleKernel(blind_matern over no_source + extras)
      + ScaleKernel(<categorical> * specific_matern over no_source + extras)
      + ScaleKernel(rbf_time over time only)

    each carrying a Time lengthscale. At the ML optimum these fit to
    very different values (typically 97.94 / 16.13 / 72.90 in the v5
    Stage-2c configuration) — the optimiser dumps composition × time
    interaction onto whichever single sub-kernel can carry it and lets
    the others drift to the lengthscale upper-bound rail. That outcome
    is fragile (seed-dependent which sub-kernel does the work) and
    masks model-comparison signals.

    This prior penalises the variance of the log-lengthscales across
    the three sub-kernels with a permissive ``sigma`` (default 0.5 in
    log-space, ~1.6× spread between min and max admitted as 1-σ
    a-priori), expressing "I expect the three time-lengthscales to be
    in the same order of magnitude" without forcing them equal.

    Mathematical form
    -----------------
    With handles ``ℓ_1, ℓ_2, ..., ℓ_K`` (each a positive scalar) and
    width ``σ``, the prior contributes

        log p(ℓ_1, ..., ℓ_K) = -½/σ² · Σ_k (log ℓ_k - mean_j log ℓ_j)²

    to the marginal log-likelihood. The penalty is computed as a side
    effect of evaluating ``log_prob(x)`` on whichever single tensor
    GPyTorch passes in (the parameter the prior is registered against)
    and is distributed across the elements of that tensor so the
    framework's element-wise sum re-aggregates to exactly the scalar
    penalty above. **Register this prior on EXACTLY ONE of the
    lengthscale parameters** (typically the rbf_time's, which has
    ``numel() == 1``); registering it on multiple would multi-count
    the same penalty.

    Implementation notes
    --------------------
    The cross-component coupling is implemented via callable handles
    (``lengthscale_getters``) rather than direct tensor references
    because GPyTorch may rebuild parameter tensors during the fit's
    state-dict round-trips; the callables fetch the live tensor at
    each ``log_prob`` evaluation. Autograd flows through the handles
    correctly because each ``getter()`` returns the current
    Parameter, and PyTorch's autograd tracks the dependency.

    Args:
        lengthscale_getters: list of zero-arg callables; each returns
            the live lengthscale tensor for one kernel sub-component.
            For the strength GP's three time lengthscales the callables
            slice into the relevant Matern's ARD lengthscale tensor at
            the appropriate active-dim index for the time column.
        sigma: shrinkage strength (in log-space). Smaller = harder
            tying. Default 0.5 admits ~e^0.5 ≈ 1.65× spread between
            sub-components at 1-σ a-priori.
        attached_dim: ``numel()`` of the ARD lengthscale tensor this
            prior is attached to. Used to pre-shape the parent
            ``LogNormalPrior``'s ``loc``/``scale`` tensors (matches the
            existing :class:`WithinGroupShrinkagePrior` pattern that
            avoids PyTorch 2.12+'s aliased-storage error).
    """

    def __init__(
        self,
        lengthscale_getters: list[Callable[[], torch.Tensor]],
        sigma: float = 0.5,
        attached_dim: int = 1,
    ):
        if len(lengthscale_getters) < 2:
            raise ValueError(
                "CrossComponentLengthscalePrior needs >=2 lengthscale "
                "handles to define a cross-component variance"
            )
        super().__init__(
            loc=torch.zeros(1, attached_dim, dtype=torch.float64),
            scale=torch.ones(1, attached_dim, dtype=torch.float64),
        )
        self._getters = lengthscale_getters
        self._sigma = float(sigma)

    def log_prob(self, x):
        # Stack the live lengthscale values from all sub-components.
        log_ls = torch.stack(
            [getter().squeeze().log() for getter in self._getters]
        ).flatten()
        sq_dev = ((log_ls - log_ls.mean()) ** 2).sum()
        total = -0.5 * sq_dev / (self._sigma**2)
        return (total / x.numel()) * torch.ones_like(x)


# --- Factory ----------------------------------------------------------------


def _default_lengthscale_prior(d_in: int) -> WithinGroupShrinkagePrior | None:
    """Returns the production within-group shrinkage prior, or None if the
    input dimensionality doesn't match the production schema (in which case
    we fall back to the unconstrained MLL fit)."""
    # Only apply if the model uses the production 10-dim DEFAULT_X_COLUMNS
    # layout (Cement, Fly Ash, Slag, Water, HRWR, Fine, Coarse, MS, Temp,
    # Time). For sub-dim or test fits, return None.
    if d_in != 10:
        return None
    return WithinGroupShrinkagePrior(
        groups_with_sigma=[
            (_BINDER_LENGTHSCALE_GROUP, _LENGTHSCALE_SHRINKAGE_SIGMA),
            (_AGGREGATE_LENGTHSCALE_GROUP, _LENGTHSCALE_SHRINKAGE_SIGMA),
        ],
        dim=d_in,
    )


def within_group_prior(
    d_in: int,
    *,
    source_dim: int | None = None,
    num_extras: int = 0,
    sigma: float = _LENGTHSCALE_SHRINKAGE_SIGMA,
    include_lognormal_baseline: bool = True,
):
    """Returns either :class:`WithinGroupShrinkagePrior` (legacy) or
    :class:`ComposedLengthscalePrior` (default for v5+) on the production
    Cement/FlyAsh/Slag and Fine/Coarse aggregate groups.

    Args:
        d_in: input dim of the kernel BEFORE excluding ``source_dim``.
        source_dim: if given, the Material Source column is dropped
            from the input (subkernel doesn't see it). Indices in the
            two groups are remapped to skip ``source_dim``. With the
            default ``DEFAULT_X_COLUMNS`` ordering, the binder group
            ``{0, 1, 2}`` and aggregate group ``{5, 6}`` are unchanged
            because ``source_dim=7`` sits between them.
        num_extras: appended-feature dims beyond ``d_in`` (or beyond
            ``d_in - 1`` if ``source_dim`` is given). Group indices are
            unchanged because extras come AFTER the original dims.
        sigma: shrinkage strength (smaller = harder tying). Default
            ``_LENGTHSCALE_SHRINKAGE_SIGMA`` makes the prior essentially
            a hard tying constraint.
        include_lognormal_baseline: if True (default), wrap the
            within-group penalty inside a
            :class:`ComposedLengthscalePrior` that also adds the
            BoTorch-default LogNormal ARD-Matern baseline on EVERY
            element (so the 11 currently-unprior'd lengthscales — Water,
            HRWR, Source, Temp, Time, and the 7 engineered features —
            stop drifting past the rail-detection threshold). Set to
            False to recover the pre-v5 :class:`WithinGroupShrinkagePrior`
            behaviour (binder/aggregate penalty only) for backward
            compatibility with sub-dim test fits and ablation studies.
    """
    if source_dim is None:
        binder = _BINDER_LENGTHSCALE_GROUP
        aggregate = _AGGREGATE_LENGTHSCALE_GROUP
        dim = d_in + num_extras
    else:
        no_source_dims = [i for i in range(d_in) if i != source_dim]
        remap = {orig: new for new, orig in enumerate(no_source_dims)}
        binder = tuple(remap[d] for d in _BINDER_LENGTHSCALE_GROUP if d in remap)
        aggregate = tuple(remap[d] for d in _AGGREGATE_LENGTHSCALE_GROUP if d in remap)
        dim = len(no_source_dims) + num_extras
    groups_with_sigma = [(binder, sigma), (aggregate, sigma)]
    if include_lognormal_baseline:
        return ComposedLengthscalePrior(
            groups_with_sigma=groups_with_sigma,
            dim=dim,
        )
    return WithinGroupShrinkagePrior(
        groups_with_sigma=groups_with_sigma,
        dim=dim,
    )


__all__ = [
    "ComposedLengthscalePrior",
    "CrossComponentLengthscalePrior",
    "WithinGroupShrinkagePrior",
    "within_group_prior",
]
