"""Research-only feature builders for the strength GP variant catalog.

These builders are kept OUT of the production package (``boxcrete``)
because they did not survive the §17–§18 ablations / distribution-aware
transform study; they are retained here so that historical variant
configurations (and future re-runs of the variant study) remain
reproducible. The production champion uses ``F5_ALLLOG_FEATURES`` from
``boxcrete.strength_v2``, which contains only the 7 distribution-aware
log-transformed features that won the per-feature LOO ablation.

Naming convention preserved from the original ``_FEATURE_BUILDERS``
dict in ``boxcrete/strength_v2.py`` (pre-cleanup); each builder is a
``Callable[[torch.Tensor], torch.Tensor]`` that takes a
``[..., 10]`` raw composition+time tensor and returns a
``[..., 1]`` engineered-feature tensor.

Composition column order (matches ``boxcrete.utils.DEFAULT_X_COLUMNS``):
0 cement, 1 fly_ash, 2 slag, 3 water, 4 hrwr, 5 fine, 6 coarse,
7 source, 8 temp, 9 time.

The 11 research-only builders re-exported here:
  - ``eff_wb_ratio``: EN 206-style effective W/B with k-values.
  - ``wc_ratio``: bare water/cement (un-SCM-weighted).
  - ``hrwr_binder``: raw HRWR/binder (replaced by
    ``log_hrwr_binder`` in production after §17 distribution audit).
  - ``coarse_fine``, ``agg_paste``: raw aggregate ratios (replaced by
    ``log_*`` versions in production).
  - ``maturity``: bare Saul/Nurse degree-days (T+10)·t. Production
    uses ``log_maturity_robust`` (clamped + log-transformed) — the bare
    form has temperature-dependent monotonicity issues for T < -10°C.
  - ``hrwr_used``: binary indicator (§17 negative result).
  - ``wc_ratio_clipped``: capped W/C — superseded by ``log_wc_ratio``.
  - ``log_maturity_robust_eps1e-3``, ``log_maturity_robust_eps1e-6``:
    alternative ε offsets used in the anchor-conditioning experiments.
  - ``log_wb_x_agg_paste``, ``log_hrwr_binder_x_agg_paste``,
    ``log_wc_x_coarse_fine``: pre-computed interaction features from
    the §6.3 leave-one-out study.

Use::

    from experiments._research_features import RESEARCH_FEATURE_BUILDERS
    from boxcrete.features import FEATURE_BUILDERS as _PROD_BUILDERS
    ALL_FEATURE_BUILDERS = {**_PROD_BUILDERS, **RESEARCH_FEATURE_BUILDERS}

Most callers should import :data:`ALL_FEATURE_BUILDERS` from
``experiments.model_variant_study`` (which already builds it).
"""

from __future__ import annotations

from typing import Callable

import torch

from boxcrete.features import IDX as _IDX

FeatureBuilder = Callable[[torch.Tensor], torch.Tensor]


RESEARCH_FEATURE_BUILDERS: dict[str, FeatureBuilder] = {
    # Effective W/B with EN 206-style k-values for SCMs (k_FA=0.3, k_slag=0.7).
    # Captures that fly ash and slag contribute less to early-age strength than cement
    # but still consume free water through pozzolanic / latent-hydraulic reactions.
    "eff_wb_ratio": lambda x: x[..., _IDX["water"] : _IDX["water"] + 1]
    / (
        x[..., _IDX["cement"] : _IDX["cement"] + 1]
        + 0.3 * x[..., _IDX["fly_ash"] : _IDX["fly_ash"] + 1]
        + 0.7 * x[..., _IDX["slag"] : _IDX["slag"] + 1]
        + 1.0
    ),
    # Original Abrams law: water-to-cement ratio (without SCM weighting).
    "wc_ratio": lambda x: x[..., _IDX["water"] : _IDX["water"] + 1]
    / (x[..., _IDX["cement"] : _IDX["cement"] + 1] + 1.0),
    # HRWR (high-range water reducer) dosage relative to total binder.
    # Already in production for slump model; raw form replaced by
    # log_hrwr_binder for strength after §17.
    "hrwr_binder": lambda x: x[..., _IDX["hrwr"] : _IDX["hrwr"] + 1]
    / (
        x[..., _IDX["cement"] : _IDX["cement"] + 1]
        + x[..., _IDX["fly_ash"] : _IDX["fly_ash"] + 1]
        + x[..., _IDX["slag"] : _IDX["slag"] + 1]
        + 1.0
    ),
    # Coarse-to-fine aggregate ratio. Affects packing density and modulus.
    "coarse_fine": lambda x: x[..., _IDX["coarse"] : _IDX["coarse"] + 1]
    / (x[..., _IDX["fine"] : _IDX["fine"] + 1] + 1.0),
    # Aggregate-to-paste mass ratio. Higher → less shrinkage, also less strength continuity.
    "agg_paste": lambda x: (
        x[..., _IDX["fine"] : _IDX["fine"] + 1]
        + x[..., _IDX["coarse"] : _IDX["coarse"] + 1]
    )
    / (
        x[..., _IDX["cement"] : _IDX["cement"] + 1]
        + x[..., _IDX["fly_ash"] : _IDX["fly_ash"] + 1]
        + x[..., _IDX["slag"] : _IDX["slag"] + 1]
        + x[..., _IDX["water"] : _IDX["water"] + 1]
        + 1.0
    ),
    # Maturity (Saul/Nurse degree-days): (T + 10) * t. Captures the temp×time
    # interaction that drives hydration kinetics. Bare form replaced by
    # log_maturity_robust in production.
    "maturity": lambda x: (
        (x[..., _IDX["temp"] : _IDX["temp"] + 1] + 10.0)
        * x[..., _IDX["time"] : _IDX["time"] + 1]
    ),
    # Binary "uses HRWR" indicator (§17 negative result; not in any retained
    # production config but kept for reference if the binary-categorical
    # hypothesis is ever revisited).
    "hrwr_used": lambda x: (x[..., _IDX["hrwr"] : _IDX["hrwr"] + 1] > 0).to(x.dtype),
    # Clipped W/C ratio: 8/647 mixes have W/C > 5 (max 287!), which swamps
    # Normalize. Cap at 1.5. Superseded in production by log_wc_ratio.
    "wc_ratio_clipped": lambda x: torch.clamp(
        x[..., _IDX["water"] : _IDX["water"] + 1]
        / (x[..., _IDX["cement"] : _IDX["cement"] + 1] + 1.0),
        max=1.5,
    ),
    # Alternative ε offsets used in the anchor-conditioning experiments
    # (§19). Production uses ε = 1.0 (in log_maturity_robust); these
    # variants stretch the maturity-axis differently for the anchor
    # conditioning experiments.
    "log_maturity_robust_eps1e-3": lambda x: torch.log(
        torch.clamp(x[..., _IDX["temp"] : _IDX["temp"] + 1] + 10.0, min=0.0)
        * x[..., _IDX["time"] : _IDX["time"] + 1]
        + 1e-3
    ),
    "log_maturity_robust_eps1e-6": lambda x: torch.log(
        torch.clamp(x[..., _IDX["temp"] : _IDX["temp"] + 1] + 10.0, min=0.0)
        * x[..., _IDX["time"] : _IDX["time"] + 1]
        + 1e-6
    ),
    # ---- Interaction features motivated by the §6.3 feature ablation ----
    # Pre-computed log-products give the kernel its own ARD lengthscale on
    # the interaction term, which it can fit independently from the
    # individual contributors.
    "log_wb_x_agg_paste": lambda x: torch.log(
        # log( (water/(C+FA+S)) * ((F+C_agg)/(C+FA+S+W)) + eps )
        (
            x[..., _IDX["water"] : _IDX["water"] + 1]
            / (
                x[..., _IDX["cement"] : _IDX["cement"] + 1]
                + x[..., _IDX["fly_ash"] : _IDX["fly_ash"] + 1]
                + x[..., _IDX["slag"] : _IDX["slag"] + 1]
                + 1.0
            )
        )
        * (
            (
                x[..., _IDX["fine"] : _IDX["fine"] + 1]
                + x[..., _IDX["coarse"] : _IDX["coarse"] + 1]
            )
            / (
                x[..., _IDX["cement"] : _IDX["cement"] + 1]
                + x[..., _IDX["fly_ash"] : _IDX["fly_ash"] + 1]
                + x[..., _IDX["slag"] : _IDX["slag"] + 1]
                + x[..., _IDX["water"] : _IDX["water"] + 1]
                + 1.0
            )
        )
        + 1e-3
    ),
    "log_hrwr_binder_x_agg_paste": lambda x: torch.log(
        # log( (HRWR/binder) * (A/P) + eps )
        (
            x[..., _IDX["hrwr"] : _IDX["hrwr"] + 1]
            / (
                x[..., _IDX["cement"] : _IDX["cement"] + 1]
                + x[..., _IDX["fly_ash"] : _IDX["fly_ash"] + 1]
                + x[..., _IDX["slag"] : _IDX["slag"] + 1]
                + 1.0
            )
        )
        * (
            (
                x[..., _IDX["fine"] : _IDX["fine"] + 1]
                + x[..., _IDX["coarse"] : _IDX["coarse"] + 1]
            )
            / (
                x[..., _IDX["cement"] : _IDX["cement"] + 1]
                + x[..., _IDX["fly_ash"] : _IDX["fly_ash"] + 1]
                + x[..., _IDX["slag"] : _IDX["slag"] + 1]
                + x[..., _IDX["water"] : _IDX["water"] + 1]
                + 1.0
            )
        )
        + 1e-3
    ),
    "log_wc_x_coarse_fine": lambda x: torch.log(
        # log( (W/C) * (Coarse/Fine) + eps )
        (
            x[..., _IDX["water"] : _IDX["water"] + 1]
            / (x[..., _IDX["cement"] : _IDX["cement"] + 1] + 1.0)
        )
        * (
            x[..., _IDX["coarse"] : _IDX["coarse"] + 1]
            / (x[..., _IDX["fine"] : _IDX["fine"] + 1] + 1.0)
        )
        + 1e-3
    ),
}


__all__ = ["RESEARCH_FEATURE_BUILDERS", "FeatureBuilder"]
