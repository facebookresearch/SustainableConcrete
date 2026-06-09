# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Engineered features and input/output transforms for the V2 strength GP.

Public extension surface for users / variant authors:

  * :data:`F5_ALLLOG_FEATURES` — the deployed V2 strength GP's 7-feature set.
  * :data:`GATE_TAU` — the production time-gate timescale (0.05).
  * :data:`IDX` — column-name → integer-index map for the 10 raw input
    dims (Cement, Fly Ash, Slag, Water, HRWR, Fine, Coarse, Source,
    Temperature, Time).
  * :data:`FEATURE_BUILDERS` — name → ``(X) -> feature_col`` callables
    for the V2 strength GP's 7 engineered features.
  * :func:`append_engineered_features_callable` — produces the callable
    consumed by BoTorch's ``AppendFeatures(f=...)`` primitive.
  * :func:`augmented_bounds` — extends an X bounds tensor with empirical
    bounds for the appended engineered features.
  * :class:`AppendDerivedFeatures` — appends the HRWR-to-binder ratio
    log-time transform. The deployed V2 strength GP uses BoTorch's
    ``AffineInputTransform + Log10`` primitives directly; this class
    is retained for the ``learnable_log_offset=True`` ablation.
  * :func:`max_scale_Y` — multiplicative-only Y scaling
    (``Y / y_max`` with ``y_mean = 0``) used by the gated-kernel V2
    strength GP.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.models.transforms.input import InputTransform
from torch import Tensor

from boxcrete.utils import DEFAULT_X_COLUMNS

# Time gate constant: h(t) = 1 - exp(-t / GATE_TAU).
# tau=0.05 (post-input-transform time units) was found to be optimal in
# the τ-sweep; see §4.5 of the benchmark.
GATE_TAU = 0.05


# Default feature set for the V2 strength GP. See STRENGTH_GP_BENCHMARK.md
# §4.3 for the per-feature ablation showing all 5 log-transforms help.
F5_ALLLOG_FEATURES = (
    "wb_ratio",
    "scm_frac",
    "log_hrwr_binder",
    "log_wc_ratio",
    "log_coarse_fine",
    "log_agg_paste",
    "log_maturity_robust",
)


# Short alias → full column name in :data:`boxcrete.utils.DEFAULT_X_COLUMNS`.
# The integer indices in :data:`IDX` are *derived* from
# ``DEFAULT_X_COLUMNS`` at import time, so any reorder / addition /
# removal at the data-layer schema automatically propagates. A
# mistyped or removed full-column-name raises ``ValueError`` at import
# (loud) instead of silently indexing the wrong column at fit time.
_RAW_COLUMN_FOR_IDX = {
    "cement": "Cement (kg/m3)",
    "fly_ash": "Fly Ash (kg/m3)",
    "slag": "Slag (kg/m3)",
    "water": "Water (kg/m3)",
    "hrwr": "HRWR (kg/m3)",
    "fine": "Fine Aggregate (kg/m3)",
    "coarse": "Coarse Aggregates (kg/m3)",
    "source": "Material Source",
    "temp": "Temp (C)",
    "time": "Time",
}
IDX = {
    short: DEFAULT_X_COLUMNS.index(full) for short, full in _RAW_COLUMN_FOR_IDX.items()
}


FEATURE_BUILDERS = {
    # V2 strength GP's 7-feature set (F5_ALLLOG_FEATURES). The 11
    # research-only builders (eff_wb_ratio, wc_ratio, hrwr_binder,
    # Note: the F5_alllog feature set below is the deployed
    # production set. The full research catalog (additional engineered
    # features explored during V2 development) is not part of this
    # release.
    #
    # Abrams' law: water-to-binder ratio is the strongest single predictor of strength.
    "wb_ratio": lambda x: x[..., IDX["water"] : IDX["water"] + 1]
    / (
        x[..., IDX["cement"] : IDX["cement"] + 1]
        + x[..., IDX["fly_ash"] : IDX["fly_ash"] + 1]
        + x[..., IDX["slag"] : IDX["slag"] + 1]
        + 1.0
    ),
    # SCM (supplementary cementitious materials) replacement fraction.
    "scm_frac": lambda x: (
        x[..., IDX["fly_ash"] : IDX["fly_ash"] + 1]
        + x[..., IDX["slag"] : IDX["slag"] + 1]
    )
    / (
        x[..., IDX["cement"] : IDX["cement"] + 1]
        + x[..., IDX["fly_ash"] : IDX["fly_ash"] + 1]
        + x[..., IDX["slag"] : IDX["slag"] + 1]
        + 1.0
    ),
    # log(HRWR/binder + 1e-4): spreads the bimodal hrwr_binder distribution
    # (86% near-zero) into something more uniform. §17 found the original
    # hrwr_binder is essentially a spike-and-slab, which is why the kernel
    # railed at the lower lengthscale bound.
    "log_hrwr_binder": lambda x: torch.log(
        (
            x[..., IDX["hrwr"] : IDX["hrwr"] + 1]
            / (
                x[..., IDX["cement"] : IDX["cement"] + 1]
                + x[..., IDX["fly_ash"] : IDX["fly_ash"] + 1]
                + x[..., IDX["slag"] : IDX["slag"] + 1]
                + 1.0
            )
        )
        + 1e-4
    ),
    # ---- §18 D1: log-transforms of other heavy-tailed features ----
    "log_wc_ratio": lambda x: torch.log(
        (
            x[..., IDX["water"] : IDX["water"] + 1]
            / (x[..., IDX["cement"] : IDX["cement"] + 1] + 1.0)
        )
        + 1e-3
    ),
    "log_agg_paste": lambda x: torch.log(
        (
            (
                x[..., IDX["fine"] : IDX["fine"] + 1]
                + x[..., IDX["coarse"] : IDX["coarse"] + 1]
            )
            / (
                x[..., IDX["cement"] : IDX["cement"] + 1]
                + x[..., IDX["fly_ash"] : IDX["fly_ash"] + 1]
                + x[..., IDX["slag"] : IDX["slag"] + 1]
                + x[..., IDX["water"] : IDX["water"] + 1]
                + 1.0
            )
        )
        + 1e-3
    ),
    "log_maturity_robust": lambda x: torch.log(
        torch.clamp(x[..., IDX["temp"] : IDX["temp"] + 1] + 10.0, min=0.0)
        * x[..., IDX["time"] : IDX["time"] + 1]
        + 1.0
    ),
    "log_coarse_fine": lambda x: torch.log(
        x[..., IDX["coarse"] : IDX["coarse"] + 1]
        / (x[..., IDX["fine"] : IDX["fine"] + 1] + 1.0)
        + 1e-3
    ),
}


def append_engineered_features_callable(
    feature_names: Sequence[str],
):
    """Build a callable that appends F5_alllog engineered features.

    Returned callable matches BoTorch's ``AppendFeatures(f=...)`` API:
    given an input ``X`` of shape ``batch x q x d``, return the new
    features as ``batch x q x 1 x d_f`` (the n_f=1 axis is required by
    ``AppendFeatures.transform``'s expansion logic; for our use case
    each input gets exactly one set of d_f appended features).

    Each feature is computed from the raw (un-normalised) input columns.
    The full input ``[X | F]`` is then handed to a downstream Normalize
    that rescales every dim (including the appended features) to [0, 1].
    """

    def f(X: torch.Tensor) -> torch.Tensor:
        if not feature_names:
            # ``pragma: no cover`` -- V2 fit always has F5_alllog
            # (7 features); empty-feature_names branch only used by
            # research-only ablations.
            return X.new_empty((*X.shape[:-1], 1, 0))  # pragma: no cover
        feats = torch.cat([FEATURE_BUILDERS[n](X) for n in feature_names], dim=-1)
        return feats.unsqueeze(-2)

    return f


def max_scale_Y(Y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Multiplicative-only Y scaling: ``Y / y_max`` with ``y_mean = 0``.

    Used by the gated-kernel V2 strength GP: standard z-score standardisation
    breaks the kernel's ``f(x, 0) = 0`` constraint because un-standardisation
    adds back ``y_mean``. Max-scaling has no additive offset, so a posterior
    of 0 in scaled space stays 0 in raw psi space.

    Returns ``(Y_scaled, y_mean=0, y_std=y_max)`` so the existing untransform
    code path ``mean * y_std + y_mean`` works correctly.
    """
    if Y.dim() == 1:
        # ``pragma: no cover`` -- V2 callers pass [n, 1] Y (Y-shape
        # guard at ``fit_strength_gp`` top); 1D fallback retained for
        # symmetry with BoTorch's Standardize signature.
        Y = Y.unsqueeze(-1)  # pragma: no cover
    y_max = Y.abs().max(dim=0, keepdim=True).values.clamp_min(1e-6)
    y_mean = torch.zeros_like(y_max)
    return Y / y_max, y_mean, y_max


def augmented_bounds(
    X: Tensor,
    bounds: Tensor,
    feature_names: Sequence[str],
) -> Tensor:
    """Compute (2 x d_aug) bounds: original bounds for the raw dims, plus
    empirical [min, max] (with 5% padding) over the appended features
    evaluated on X.
    """
    if not feature_names:
        # ``pragma: no cover`` -- V2 fit always passes F5_alllog
        # (7 features); empty-feature_names branch only used by
        # research-only no-feature ablations.
        return bounds  # pragma: no cover
    appended_vals = torch.cat([FEATURE_BUILDERS[n](X) for n in feature_names], dim=-1)
    aug_lower = appended_vals.min(dim=0).values
    aug_upper = appended_vals.max(dim=0).values
    pad = 0.05 * (aug_upper - aug_lower).clamp_min(1e-3)
    aug_lower = aug_lower - pad
    aug_upper = aug_upper + pad
    return torch.cat([bounds, torch.stack([aug_lower, aug_upper])], dim=-1)


class AppendDerivedFeatures(InputTransform, torch.nn.Module):
    """Input transform that appends the HRWR-to-binder ratio.

    The HRWR/binder ratio encodes the admixture dosage relative to total
    binder content — a key determinant of concrete workability (slump)
    that stationary GP kernels cannot learn from raw composition values.

    Used by the V1 strength fit factory (kept for instructional baseline)
    AND by the production slump GP — workability is driven primarily by
    this single ratio.
    """

    is_one_to_many = False

    def __init__(
        self,
        cement_idx: int = IDX["cement"],
        fly_ash_idx: int = IDX["fly_ash"],
        slag_idx: int = IDX["slag"],
        hrwr_idx: int = IDX["hrwr"],
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


__all__ = [
    "AppendDerivedFeatures",
    "F5_ALLLOG_FEATURES",
    "FEATURE_BUILDERS",
    "GATE_TAU",
    "IDX",
    "append_engineered_features_callable",
    "augmented_bounds",
    "max_scale_Y",
]
