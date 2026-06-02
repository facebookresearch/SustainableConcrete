# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Lengthscale priors for the V2 strength GP.

Hosts the within-group shrinkage prior used to soft-tie ARD lengthscales
of materials playing interchangeable physical roles (cementitious binders
and aggregates). The prior lives alongside the group-index constants it
depends on so users searching for "where is the lengthscale prior?"
land in one obvious place.

Public re-exports go through :mod:`boxcrete` for ergonomics.
"""

from __future__ import annotations

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
) -> WithinGroupShrinkagePrior:
    """Within-group shrinkage prior over Cement/FlyAsh/Slag and
    Fine/Coarse aggregate groups.

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
    return WithinGroupShrinkagePrior(
        groups_with_sigma=[(binder, sigma), (aggregate, sigma)],
        dim=dim,
    )


__all__ = [
    "WithinGroupShrinkagePrior",
    "within_group_prior",
]
