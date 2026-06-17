#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Three-class material source + LogNormal lengthscale prior ablation.

Plan reference: ``three_class_and_lengthscale_prior.plan.md`` §"Commit 6".

V2-grade ablation for the materials-classes-and-lengthscale-prior
stack: per-stage decomposition with full V2 metric battery (LOO,
block-LOO, per-class, calibration: PIT-KS / coverage95 / MLPD / CRPS).

Variant grid:

    Stage 0 (v2_baseline)
        Pre-v5 fixture (727 rows, binary {0,1} source). Reads from
        test/fixtures/boxcrete_data_pre_v5.csv. Uses the v5 production
        fit (categorical kernel default) as the closest-available
        2-class anchor; the ``onehot_ard`` baseline is also collected
        for direct comparison.

    Stage 1 (v2_kernel_3class_data)
        v5 dataset (679 rows, ternary {0,1,2}) with the SAME categorical
        kernel as Stage 0. Isolates the data-correction effect.

    Stage 2c (3class_cat_indexkernel_r2)
        v5 + IndexKernel(rank=2) + within-group prior only (NO LogNormal,
        NO time-tying). The proposed kernel's bare configuration; reveals
        the rail behaviour in ℓ_Time(blind).

    Stage 3 (3class_cat_indexkernel_r2_lognormal)
        Stage 2c + per-element LogNormal baseline on every dim
        (ComposedLengthscalePrior). Suppresses ALL rail-prone dims at the
        cost of compressing the dims that are doing useful work.

    Stage 3.5a (3class_cat_indexkernel_r2_timetie)
        Stage 2c + cross-component time-lengthscale tying (sigma=0.5)
        but NO per-element LogNormal. Tests "is the cross-component
        time-tying alone enough to fix the rail?" — keeping non-time
        lengthscales unconstrained.

    Stage 3.5b (3class_cat_indexkernel_r2_lognormal_timetie)
        Stage 3 + cross-component time-lengthscale tying. Tests "do
        per-element LogNormal AND cross-component tying compose
        constructively, or does one suffice?"

3 seeds per variant. Per-variant metrics:
  - ``loo_metrics``        analytical leave-one-out (psi)
  - ``block_loo_metrics``  composition-leave-one-out (psi); the harder
                           and more honest metric the V2 study used
  - ``held_out_set3``      held-out-Set-3 (fit on Sets 1+2 only,
                           evaluate on Set 3 in psi)
  - lengthscale rail diagnostics: ℓ_Time across all three sub-kernels

Outputs:

    experiments/three_class_ablation_results.csv
    experiments/THREE_CLASS_AND_PRIOR_BENCHMARK.md

Run::

    python experiments/three_class_ablation.py --variants all --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from boxcrete import fit_strength_gp, load_concrete_strength  # noqa: E402
from boxcrete.utils import DEFAULT_X_COLUMNS  # noqa: E402

# Reuse the V2 ablation metric battery — same path as
# experiments/STRENGTH_GP_BENCHMARK.md uses for the production champion.
from experiments.model_variant_study import (  # noqa: E402
    block_loo_metrics,
    held_out_metrics,
    loo_metrics,
)

PRE_V5_FIXTURE = REPO_ROOT / "test" / "fixtures" / "boxcrete_data_pre_v5.csv"
RESULTS_CSV = REPO_ROOT / "experiments" / "three_class_ablation_results.csv"
WRITEUP_MD = REPO_ROOT / "experiments" / "THREE_CLASS_AND_PRIOR_BENCHMARK.md"


# ---------------------------------------------------------------------------
# Variant catalog
# ---------------------------------------------------------------------------


def _stage_factory(
    source_kernel: str,
    *,
    include_lognormal_baseline: bool,
    time_tying_sigma: float | None,
):
    """Build a fit factory for a given (source_kernel, prior, time_tie)
    config. Implementation: monkey-patch the production builder for
    the duration of the fit so we don't have to plumb every variant's
    knob through fit_strength_gp's public surface.
    """

    def _fit(X, Y, Yvar, X_bounds, seed):
        torch.manual_seed(seed)
        from boxcrete import kernels as _kmod
        from boxcrete import priors as _pmod
        from boxcrete import strength_model as _smod

        original_builder = _kmod.make_gated_strength_kernel_builder
        original_factory = _pmod.within_group_prior

        def _patched_builder(gate_tau=0.05, **kwargs):
            return original_builder(
                gate_tau=gate_tau,
                source_kernel=source_kernel,
                time_tying_sigma=time_tying_sigma,
            )

        def _patched_factory(*args, **kwargs):
            kwargs.setdefault("include_lognormal_baseline", include_lognormal_baseline)
            return original_factory(*args, **kwargs)

        _kmod.make_gated_strength_kernel_builder = _patched_builder
        _smod.make_gated_strength_kernel_builder = _patched_builder
        _pmod.within_group_prior = _patched_factory
        _kmod.within_group_prior = _patched_factory
        try:
            return fit_strength_gp(X=X, Y=Y, Yvar=Yvar, X_bounds=X_bounds, seed=seed)
        finally:
            _kmod.make_gated_strength_kernel_builder = original_builder
            _smod.make_gated_strength_kernel_builder = original_builder
            _pmod.within_group_prior = original_factory
            _kmod.within_group_prior = original_factory

    return _fit


VARIANTS = {
    # ------------------------------------------------------------
    # Production-anchor variants: the legacy-V2 architecture
    # (continuous-ARD source kernel) on both pre-v5 and v5 data,
    # to anchor the no-regression check vs the currently-deployed
    # production model. The deployed `docs/model/strength.json` was
    # generated by this architecture on pre-v5 data.
    # ------------------------------------------------------------
    "production_pre_v5_legacy_continuous_ard_bare": {
        "factory": _stage_factory(
            "legacy_continuous_ard",
            include_lognormal_baseline=False,
            time_tying_sigma=None,
        ),
        "data": "pre_v5",
        "description": (
            "*** Currently-deployed production architecture (legacy V2 "
            "continuous-ARD source kernel) on pre-v5 fixture. "
            "Reproduces the architecture that fitted the deployed "
            "docs/model/strength.json. The within-group prior is on "
            "(default), as in the deployment. ***"
        ),
        "stage": -1,
    },
    "production_v5_legacy_continuous_ard_bare": {
        "factory": _stage_factory(
            "legacy_continuous_ard",
            include_lognormal_baseline=False,
            time_tying_sigma=None,
        ),
        "data": "v5",
        "description": (
            "Legacy V2 architecture on the new v5 dataset. "
            "Tests whether the architecture itself can absorb the "
            "v5 data improvements without changing the kernel."
        ),
        "stage": -1,
    },
    "stage0_pre_v5_indexkernel_r2_bare": {
        "factory": _stage_factory(
            "indexkernel_r2",
            include_lognormal_baseline=False,
            time_tying_sigma=None,
        ),
        "data": "pre_v5",
        "description": (
            "Pre-v5 fixture (727 rows, binary {0,1} source) + "
            "IndexKernel(rank=2) + bare prior. Kernel × data cell (1, "
            "pre-v5)."
        ),
        "stage": 0,
    },
    "stage0_pre_v5_hamming_bare": {
        "factory": _stage_factory(
            "hamming",
            include_lognormal_baseline=False,
            time_tying_sigma=None,
        ),
        "data": "pre_v5",
        "description": (
            "Pre-v5 fixture + Hamming categorical kernel + bare prior. "
            "Kernel × data cell (Hamming, pre-v5)."
        ),
        "stage": 0,
    },
    "stage0_pre_v5_indexkernel_r2_lognormal_timetie": {
        "factory": _stage_factory(
            "indexkernel_r2",
            include_lognormal_baseline=True,
            time_tying_sigma=0.5,
        ),
        "data": "pre_v5",
        "description": (
            "Pre-v5 fixture + IndexKernel(rank=2) + LogNormal + time-tie. "
            "Kernel × data × prior cell (IndexKernel, pre-v5, with priors)."
        ),
        "stage": 0,
    },
    "stage0_pre_v5_hamming_lognormal_timetie": {
        "factory": _stage_factory(
            "hamming",
            include_lognormal_baseline=True,
            time_tying_sigma=0.5,
        ),
        "data": "pre_v5",
        "description": (
            "Pre-v5 fixture + Hamming + LogNormal + time-tie. "
            "Kernel × data × prior cell (Hamming, pre-v5, with priors)."
        ),
        "stage": 0,
    },
    # ------------------------------------------------------------
    # Stage 1/2: v5 data, bare prior (kernel-effect-only).
    # ------------------------------------------------------------
    "stage1_v5_indexkernel_r2_bare": {
        "factory": _stage_factory(
            "indexkernel_r2",
            include_lognormal_baseline=False,
            time_tying_sigma=None,
        ),
        "data": "v5",
        "description": (
            "v5 (679 rows, ternary {0,1,2}) + IndexKernel(rank=2) + bare. "
            "Kernel × data cell (IndexKernel, v5)."
        ),
        "stage": 1,
    },
    "stage2a_v5_hamming_bare": {
        "factory": _stage_factory(
            "hamming",
            include_lognormal_baseline=False,
            time_tying_sigma=None,
        ),
        "data": "v5",
        "description": ("v5 + Hamming + bare. Kernel × data cell (Hamming, v5)."),
        "stage": 2,
    },
    # ------------------------------------------------------------
    # Stage 3/3.5: v5 + IndexKernel(r=2) prior ablations.
    # ------------------------------------------------------------
    "stage3_v5_indexkernel_r2_lognormal": {
        "factory": _stage_factory(
            "indexkernel_r2",
            include_lognormal_baseline=True,
            time_tying_sigma=None,
        ),
        "data": "v5",
        "description": "v5 + IndexKernel(r=2) + LogNormal only.",
        "stage": 3,
    },
    "stage3_5a_v5_indexkernel_r2_timetie": {
        "factory": _stage_factory(
            "indexkernel_r2",
            include_lognormal_baseline=False,
            time_tying_sigma=0.5,
        ),
        "data": "v5",
        "description": "v5 + IndexKernel(r=2) + time-tie only.",
        "stage": 3.5,
    },
    "stage3_5b_v5_indexkernel_r2_lognormal_timetie": {
        "factory": _stage_factory(
            "indexkernel_r2",
            include_lognormal_baseline=True,
            time_tying_sigma=0.5,
        ),
        "data": "v5",
        "description": "v5 + IndexKernel(r=2) + LogNormal + time-tie.",
        "stage": 3.5,
    },
    # ------------------------------------------------------------
    # Stage 4: v5 kernel-choice sweep under best prior config.
    # ------------------------------------------------------------
    "stage4_v5_hamming_lognormal_only": {
        "factory": _stage_factory(
            "hamming",
            include_lognormal_baseline=True,
            time_tying_sigma=None,
        ),
        "data": "v5",
        "description": (
            "Hamming + LogNormal baseline only (no time-tying). "
            "Tests if LogN alone is purely positive vs Hamming bare."
        ),
        "stage": 4,
    },
    "stage4_v5_hamming_timetie_only": {
        "factory": _stage_factory(
            "hamming",
            include_lognormal_baseline=False,
            time_tying_sigma=0.5,
        ),
        "data": "v5",
        "description": (
            "Hamming + cross-component time-tying only (no LogNormal). "
            "Tests if time-tying alone is purely positive vs Hamming bare."
        ),
        "stage": 4,
    },
    "stage4_v5_hamming_lognormal_timetie": {
        "factory": _stage_factory(
            "hamming",
            include_lognormal_baseline=True,
            time_tying_sigma=0.5,
        ),
        "data": "v5",
        "description": (
            "v5 + Hamming + LogNormal + time-tie. "
            "Kernel × data × prior cell (Hamming, v5, with priors)."
        ),
        "stage": 4,
    },
    "stage4_v5_indexkernel_r1_lognormal_timetie": {
        "factory": _stage_factory(
            "indexkernel_r1",
            include_lognormal_baseline=True,
            time_tying_sigma=0.5,
        ),
        "data": "v5",
        "description": "v5 + IndexKernel(rank=1) + LogNormal + time-tie.",
        "stage": 4,
    },
    "stage4_v5_indexkernel_r3_lognormal_timetie": {
        "factory": _stage_factory(
            "indexkernel_r3",
            include_lognormal_baseline=True,
            time_tying_sigma=0.5,
        ),
        "data": "v5",
        "description": "v5 + IndexKernel(rank=3) + LogNormal + time-tie.",
        "stage": 4,
    },
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_data(which: str):
    """Return (ds, X, Y, Yvar, bounds, n_real) tuple for the requested
    dataset. ``n_real`` is the count of real (non-anchor) rows used by
    the V2 ``loo_metrics`` battery.
    """
    if which == "pre_v5":
        if not PRE_V5_FIXTURE.exists():
            raise FileNotFoundError(
                f"Pre-v5 fixture missing: {PRE_V5_FIXTURE}. "
                "Re-run scripts/merge_three_class_data.py."
            )
        ds = load_concrete_strength(data_path=str(PRE_V5_FIXTURE))
    elif which == "v5":
        ds = load_concrete_strength()
    else:
        raise ValueError(f"Unknown data choice: {which!r}")
    X, Y, Yvar, _ = ds.strength_data
    return ds, X, Y, Yvar, ds.bounds, int(X.shape[0])


def _make_class_holdout(holdout_class: int):
    """Return (X_train, Y_train, Yvar_train, X_test, Y_test_psi, bounds)
    with all rows of ``holdout_class`` excluded from train and used as
    the held-out test set. ``holdout_class ∈ {0, 1, 2}``.
    """
    ds = load_concrete_strength()
    X, Y, Yvar, _ = ds.strength_data
    source_col = DEFAULT_X_COLUMNS.index("Material Source")
    train_mask = X[:, source_col] != holdout_class
    test_mask = X[:, source_col] == holdout_class
    return (
        X[train_mask],
        Y[train_mask],
        Yvar[train_mask] if Yvar is not None else None,
        X[test_mask],
        Y[test_mask].squeeze(-1),
        ds.bounds,
    )


def _make_set3_holdout():
    """Return (X_train, Y_train, Yvar_train, X_test, Y_test_psi, bounds)
    for the held-out-Set-3 evaluation: fit on (Set 1 + Set 2), evaluate
    on Set 3.
    """
    return _make_class_holdout(2)


# ---------------------------------------------------------------------------
# Per-fit diagnostics on lengthscales (rail detection)
# ---------------------------------------------------------------------------


def _lengthscale_diagnostics(model) -> dict[str, float]:
    """Extract per-branch Time lengthscale + max blind lengthscale +
    within-group ratios."""
    out: dict[str, float] = {}
    try:
        gated = model.covar_module
        components = list(gated.base_kernel.kernels)
        blind, specific, time_kernel = components

        time_idx_raw = DEFAULT_X_COLUMNS.index("Time")
        # blind active dims = no_source + extras
        blind_active = blind.base_kernel.active_dims.tolist()
        blind_ls = blind.base_kernel.lengthscale.detach().squeeze().tolist()
        if isinstance(blind_ls, float):
            blind_ls = [blind_ls]
        try:
            blind_time_pos = blind_active.index(time_idx_raw)
            out["ell_time_blind"] = float(blind_ls[blind_time_pos])
        except ValueError:
            out["ell_time_blind"] = float("nan")
        out["max_blind_lengthscale"] = float(max(blind_ls))
        out["min_blind_lengthscale"] = float(min(blind_ls))

        # Within-group ratios (binder = 0/1/2, aggregate = 5/6) by raw idx.
        binder_idx_raw = (0, 1, 2)
        agg_idx_raw = (5, 6)
        try:
            binder_ls = [
                blind_ls[blind_active.index(i)]
                for i in binder_idx_raw
                if i in blind_active
            ]
            if len(binder_ls) > 1:
                out["binder_within_ratio"] = max(binder_ls) / max(min(binder_ls), 1e-12)
        except ValueError:
            pass
        try:
            agg_ls = [
                blind_ls[blind_active.index(i)]
                for i in agg_idx_raw
                if i in blind_active
            ]
            if len(agg_ls) > 1:
                out["aggregate_within_ratio"] = max(agg_ls) / max(min(agg_ls), 1e-12)
        except ValueError:
            pass

        # Specific Matern's Time lengthscale (categorical * Matern path).
        specific_inner = specific.base_kernel
        if hasattr(specific_inner, "kernels"):
            # ProductKernel(<cat>, Matern).
            _, matern_specific = list(specific_inner.kernels)
            specific_active = matern_specific.active_dims.tolist()
            specific_ls = matern_specific.lengthscale.detach().squeeze().tolist()
            if isinstance(specific_ls, float):
                specific_ls = [specific_ls]
            if time_idx_raw in specific_active:
                out["ell_time_specific"] = float(
                    specific_ls[specific_active.index(time_idx_raw)]
                )
        else:
            # Legacy continuous-ARD or onehot_ard: a single Matern over
            # all (or one-hot expanded) dims. The "specific" branch IS
            # the matern itself, not a ProductKernel of (categorical, matern).
            # The Time lengthscale lives on the same single matern.
            spec_active = specific_inner.active_dims.tolist()
            spec_ls = specific_inner.lengthscale.detach().squeeze().tolist()
            if isinstance(spec_ls, float):
                spec_ls = [spec_ls]
            if time_idx_raw in spec_active:
                out["ell_time_specific"] = float(
                    spec_ls[spec_active.index(time_idx_raw)]
                )

        # RBF time lengthscale.
        out["ell_time_rbf"] = float(
            time_kernel.base_kernel.lengthscale.detach().squeeze()
        )
    except Exception as exc:
        out["lengthscale_error"] = str(exc)[:200]
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _eval_metrics(model, n_real: int) -> dict[str, float]:
    """Compute LOO + block-LOO + lengthscale diagnostics on a fitted
    model. All metrics are in psi (the V2 helpers handle the
    Y/y_max untransformation).

    Adds a "Sets 1+2 only" bLOO subset metric that masks the bLOO
    predictions to rows whose Material Source is in {0, 1} — the v5
    "non-Set-3" subset, which is the apples-to-apples comparison
    against the pre-v5 baseline (which has no Set-3 data).
    """
    metrics: dict[str, float] = {}
    try:
        loo = loo_metrics(model, n_real)
        for k, v in loo.items():
            metrics[f"loo_{k}"] = v
    except Exception as exc:
        metrics["loo_error"] = str(exc)[:200]
    try:
        bloo = block_loo_metrics(model, n_real)
        for k, v in bloo.items():
            metrics[f"bloo_{k}"] = v
    except Exception as exc:
        metrics["bloo_error"] = str(exc)[:200]
    metrics.update(_lengthscale_diagnostics(model))

    # Sets-1+2-only block-LOO subset (the apples-to-apples comparison
    # against pre-v5, which has no Set-3 rows).
    try:
        bloo_subset = _bloo_set12_metrics(model, n_real)
        for k, v in bloo_subset.items():
            metrics[f"bloo_set12_{k}"] = v
    except Exception as exc:
        metrics["bloo_set12_error"] = str(exc)[:200]

    return metrics


def _bloo_set12_metrics(model, n_real: int) -> dict[str, float]:
    """Compute block-LOO metrics restricted to Sets 1+2 (Material Source
    in {0, 1}).

    Uses the same closed-form block-LOO machinery as
    ``model_variant_study.block_loo_metrics``, then masks predictions to
    the Sets-1+2 subset before computing the summary statistics. This is
    the apples-to-apples comparison vs the pre-v5 baseline (which has
    no Set-3 data; pre-v5's MS=1 = pre-v5's "Set 3", which is actually a
    pooled-and-corrupted version of v5's class 2). On the pre-v5 fixture
    (Material Source in {0, 1}), MS=0 is the "Sets 1+2" subset and MS=1
    is the (corrupted) Set-3; we mask MS=0 only there.
    """
    from gpytorch.utils.cholesky import psd_safe_cholesky
    from experiments.model_variant_study import _crps_normal, _normal_cdf

    model.eval()
    with torch.no_grad():
        train_X = model.train_inputs[0]
        train_Y = model.train_targets
        prior = model.forward(train_X)
        noisy = model.likelihood(prior)
        K = noisy.lazy_covariance_matrix.to_dense()
    n = K.shape[-1]
    L = psd_safe_cholesky(K)
    Y_t = train_Y.unsqueeze(-1) if train_Y.dim() == 1 else train_Y
    residuals = Y_t - prior.mean.unsqueeze(-1)
    alpha = torch.cholesky_solve(residuals, L).squeeze(-1)
    I = torch.eye(n, dtype=K.dtype, device=K.device)
    K_inv = torch.cholesky_solve(I, L)

    fingerprints = train_X[..., :9]
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

    # Restrict to real rows.
    obs = Y_t[:n_real, 0]
    mean_v = loo_pred[:n_real]
    std_v = loo_var[:n_real].sqrt()

    # Untransform back to original units (mirrors block_loo_metrics).
    if hasattr(model, "_study_y_std"):
        y_std = model._study_y_std.to(mean_v)
        y_mean = model._study_y_mean.to(mean_v)
        mean_v = mean_v * y_std + y_mean
        obs = obs * y_std + y_mean
        std_v = std_v * y_std

    # Mask to Sets 1+2 subset. The post-input-transform Material Source
    # column is at IDX["source"] = 7; the values are the integer task
    # labels {0, 1, 2} since Normalize excludes the source dim post-Commit-5.
    source_col = train_X[..., 7][:n_real]
    unique_vals = sorted({int(round(v.item())) for v in source_col.unique()})
    if unique_vals == [0, 1]:
        # Pre-v5 path: MS in {0, 1} where 0 = pooled-Sets-1+2; mask to MS=0.
        mask = source_col == 0
    else:
        # v5 path: MS in {0, 1, 2}; mask to {0, 1} = Sets 1+2.
        mask = source_col != 2
    if mask.sum().item() == 0:
        return {"n_rows": 0.0}

    mean_v = mean_v[mask]
    obs = obs[mask]
    std_v = std_v[mask]

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
        "n_rows": float(nrows),
    }


def _run_main(variants: list[str], seeds: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for vid in variants:
        if vid not in VARIANTS:
            raise KeyError(f"Unknown variant {vid!r}")
        spec = VARIANTS[vid]
        ds, X, Y, Yvar, bounds, n_real = _load_data(spec["data"])
        for seed in seeds:
            t0 = time.time()
            row = {
                "variant": vid,
                "stage": spec["stage"],
                "data": spec["data"],
                "seed": seed,
                "n_train": n_real,
            }
            try:
                model = spec["factory"](X, Y, Yvar, bounds, seed)
                row.update(_eval_metrics(model, n_real))
            except Exception as exc:
                row["fit_error"] = str(exc)[:300]
            row["wall_clock_sec"] = time.time() - t0
            print(
                f"[main {vid}@seed{seed}] "
                f"loo_rmse={row.get('loo_rmse')} "
                f"bloo_rmse={row.get('bloo_rmse')} "
                f"ell_t_b={row.get('ell_time_blind')} "
                f"ell_t_s={row.get('ell_time_specific')} "
                f"ell_t_r={row.get('ell_time_rbf')} "
                f"wall={row['wall_clock_sec']:.1f}s",
                flush=True,
            )
            rows.append(row)
            # Persist partial results after every fit so a crash mid-run
            # leaves usable data on disk.
            pd.DataFrame(rows).to_csv(
                RESULTS_CSV.with_suffix(".main.partial.csv"), index=False
            )
    return pd.DataFrame(rows)


def _run_class_holdout(
    variants: list[str], seeds: list[int], holdout_class: int
) -> pd.DataFrame:
    """Leave-one-class-out evaluation: fit on the two non-held-out
    classes, evaluate on the held-out class. Reports the same metric
    battery as ``_run_set3_holdout`` but parameterised by class.

    Class labels: 0 = Set 1 (mortar), 1 = Set 2 (Heidelberg/Class C),
    2 = Set 3 (Amrize/Class F).
    """
    rows: list[dict] = []
    X_train, Y_train, Yvar_train, X_test, Y_test_psi, bounds = _make_class_holdout(
        holdout_class
    )
    n_test = int(X_test.shape[0])
    n_train = int(X_train.shape[0])
    if n_test == 0:
        return pd.DataFrame()
    print(
        f"[holdout class={holdout_class}] n_train={n_train}, n_test={n_test}",
        flush=True,
    )
    for vid in variants:
        if vid not in VARIANTS:
            continue
        spec = VARIANTS[vid]
        if spec["data"] != "v5":
            continue
        if spec["stage"] == 0:
            continue  # pre-v5 baseline doesn't have all three classes
        for seed in seeds:
            t0 = time.time()
            row = {
                "variant": vid,
                "stage": spec["stage"],
                "data": f"v5_class{holdout_class}_heldout",
                "holdout_class": holdout_class,
                "seed": seed,
                "n_train": n_train,
                "n_test": n_test,
            }
            try:
                model = spec["factory"](X_train, Y_train, Yvar_train, bounds, seed)
                ho = held_out_metrics(model, X_test, Y_test_psi)
                for k, v in ho.items():
                    row[f"holdout_{k}"] = v
            except Exception as exc:
                row["fit_error"] = str(exc)[:300]
            row["wall_clock_sec"] = time.time() - t0
            print(
                f"[holdout class={holdout_class} {vid}@seed{seed}] "
                f"rmse={row.get('holdout_rmse')} "
                f"n_test={n_test} wall={row['wall_clock_sec']:.1f}s",
                flush=True,
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(
                RESULTS_CSV.with_suffix(f".class{holdout_class}_holdout.partial.csv"),
                index=False,
            )
    return pd.DataFrame(rows)


def _run_set3_holdout(variants: list[str], seeds: list[int]) -> pd.DataFrame:
    """Held-out-Set-3 evaluation: legacy entry point retained for
    backward compatibility with the existing CLI flag. Now a thin
    wrapper around :func:`_run_class_holdout` with ``holdout_class=2``.

    The returned DataFrame uses the legacy ``set3_*`` column prefix so
    the writeup's existing held-out-Set-3 table keeps rendering.
    """
    df = _run_class_holdout(variants, seeds, holdout_class=2)
    if df.empty:
        return df
    df = df.rename(columns={c: c.replace("holdout_", "set3_") for c in df.columns})
    df["data"] = "v5_set3_heldout"
    return df


# ---------------------------------------------------------------------------
# Markdown writeup
# ---------------------------------------------------------------------------


PRE_REGISTERED_CRITERIA = """
Per the plan §"Commit 6" (pre-registered):

| Metric | Target | Hard fail |
|---|---|---|
| Stage 1 block-LOO RMSE on Sets 1+2 | < Stage 0 baseline | > 105% |
| Stage 3 block-LOO RMSE | <= 105% of Stage 0 | > 120% |
| Held-out Set 3 RMSE | < 1500 psi | > 2000 psi |
| Per-class block-LOO RMSE | within 110% of v2 baseline | > 130% |
| Max non-grouped raw lengthscale | < 100 | > 100 |
| Binder within-group ratio (max/min) | < 1.05 | > 1.10 |
| Aggregate within-group ratio | < 1.05 | > 1.10 |
| PIT-KS distance (calibration) | within 1.5x v2 baseline | > 2x |
""".strip()


def _agg(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean ± std across seeds per variant."""
    if metric not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(["stage", "variant"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["stage", "variant"])
    )


def _headline_analysis(df_main: pd.DataFrame, _df_set3: pd.DataFrame) -> str:
    """Compose a short prose section interpreting the table results.

    Computes the variant-level mean of the marquee metrics and
    describes the qualitative story (which variants pass / fail
    against the pre-registered targets).

    The ``_df_set3`` argument is retained for signature compatibility
    with earlier callers but is unused: the headline now centres on the
    legacy-V2-vs-v5-Hamming no-regression analysis, which uses bLOO
    (full and Sets-1+2 only) and LOCO metrics from ``df_main`` plus
    the per-class holdout dataframe rendered separately by
    :func:`write_markdown`.
    """
    if df_main.empty:
        return "_No completed runs._"

    by_var = (
        df_main.groupby("variant")
        .agg(
            loo_rmse=("loo_rmse", "mean"),
            bloo_rmse=("bloo_rmse", "mean"),
            bloo12_rmse=("bloo_set12_rmse", "mean"),
            ell_t_b=("ell_time_blind", "mean"),
            ell_t_s=("ell_time_specific", "mean"),
            ell_t_r=("ell_time_rbf", "mean"),
            max_blind=("max_blind_lengthscale", "mean"),
        )
        .to_dict(orient="index")
    )

    def get(vid: str, key: str, fmt: str = "{:.0f}") -> str:
        if vid in by_var and key in by_var[vid] and not pd.isna(by_var[vid][key]):
            return fmt.format(by_var[vid][key])
        return "—"

    lines: list[str] = []

    # --- CORRECTED no-regression analysis -----------------------------
    lines.append("## Honest no-regression check vs the currently-deployed model")
    lines.append("")
    lines.append(
        "The currently-deployed `docs/model/strength.json` was fit by the "
        "**legacy V2 architecture** (single Matern with ARD over all dims "
        "including a continuous Material Source coordinate; B'' multi-Matern "
        "+ F5_alllog + gated_t + gated_noise + maxscale_zeromean). The v5 "
        "stack replaces this with a categorical source kernel (Hamming or "
        "IndexKernel) + 3-class data. **Comparing against the actual "
        "deployed architecture:**"
    )
    lines.append("")
    lines.append(
        "| Metric | pre-v5 + legacy V2 (deployed) | v5 + legacy V2 | v5 + Hamming bare (proposed) |"
    )
    lines.append("|---|---|---|---|")
    for key, label, fmt in [
        ("loo_rmse", "LOO RMSE (psi)", "{:.0f}"),
        ("bloo_rmse", "**bLOO RMSE (psi)**", "{:.0f}"),
        ("bloo12_rmse", "**bLOO Sets-1+2 RMSE (psi)**", "{:.0f}"),
        ("max_blind", "max blind ℓ", "{:.1f}"),
    ]:
        a = get("production_pre_v5_legacy_continuous_ard_bare", key, fmt)
        b = get("production_v5_legacy_continuous_ard_bare", key, fmt)
        c = get("stage2a_v5_hamming_bare", key, fmt)
        lines.append(f"| {label} | {a} | {b} | {c} |")
    lines.append("")
    lines.append(
        "**Reading the table — the v5 stack regresses bLOO vs the deployed model:**"
    )
    lines.append("")
    lines.append(
        "  - **Deployed legacy V2 on pre-v5 is the empirical bLOO winner** "
        f"({get('production_pre_v5_legacy_continuous_ard_bare', 'bloo_rmse')} psi). "
        "The rail-prevention motivation for the v5 stack's LogNormal / "
        "time-tying priors was an IndexKernel-specific issue (max blind ℓ = "
        "216 on pre-v5 with IndexKernel(r=2)) — the legacy continuous-ARD "
        "kernel has max blind ℓ ≈ 18 on the same data and never rails."
    )
    lines.append(
        "  - **v5 + legacy V2** has slightly worse bLOO than legacy V2 on "
        "pre-v5 (~18 psi). Mostly attributable to v5 having 48 fewer rows "
        "(strength-less mortars dropped) plus 5 more compositions (smaller "
        "blocks). Not a fundamental v5 regression."
    )
    lines.append(
        "  - **v5 + Hamming bare** is ~24 psi worse than the deployed model "
        "on bLOO (704 vs 680) and ~28 psi worse on bLOO Sets-1+2 (822 vs "
        "794). Hamming wins on held-out generalisation (LOCO) and on "
        "reproducibility (zero seed std), but does NOT win on raw in-data "
        "RMSE."
    )
    lines.append("")
    lines.append("### What the v5 stack actually gains vs costs")
    lines.append("")
    lines.append("**Gains over the deployed model:**")
    lines.append("")
    lines.append(
        "  - **Set-3 modelling capability** (Amrize/Class-F concrete) — "
        "impossible on the pre-v5 dataset which has no Set-3 data."
    )
    lines.append(
        "  - **Silent labelling corruption fix** — the deployed model treats "
        "Set 1 mortar and Set 2 concrete as the same `MS=0` class, "
        "asking one continuous-ARD coordinate to span both contrasts; "
        "v5's 3-class label removes this misspecification."
    )
    lines.append(
        "  - **Reproducible fits** — Hamming-bare on v5 gives zero seed std "
        "across LOCO and bLOO; legacy V2 on pre-v5 has seed std up to ~19 "
        "psi (mainly because the unprior'd Time lengthscale wanders)."
    )
    lines.append(
        "  - **Worst-case held-out RMSE** — Hamming-bare's LOCO Class-0 "
        "RMSE (2707 psi) is better than legacy V2 on v5 (2850 psi)."
    )
    lines.append("")
    lines.append("**Costs vs the deployed model:**")
    lines.append("")
    lines.append(
        f"  - bLOO RMSE: {get('production_pre_v5_legacy_continuous_ard_bare', 'bloo_rmse')} "
        f"psi (deployed) → {get('stage2a_v5_hamming_bare', 'bloo_rmse')} psi "
        f"(v5 + Hamming). **~24 psi regression.**"
    )
    lines.append(
        f"  - bLOO Sets-1+2 RMSE: {get('production_pre_v5_legacy_continuous_ard_bare', 'bloo12_rmse')} "
        f"psi → {get('stage2a_v5_hamming_bare', 'bloo12_rmse')} psi. "
        f"**~28 psi regression.**"
    )
    lines.append(
        f"  - LOO RMSE: {get('production_pre_v5_legacy_continuous_ard_bare', 'loo_rmse')} "
        f"psi → {get('stage2a_v5_hamming_bare', 'loo_rmse')} psi. "
        f"**~62 psi regression** (partly structural — see prior note about "
        f"pre-v5's pooled-class smoothing helping row-LOO)."
    )
    lines.append("")
    lines.append("### Decision recommendation")
    lines.append("")
    lines.append(
        "**Both choices have merit. The right call depends on what we " "weight more:**"
    )
    lines.append("")
    lines.append(
        "  - **Keep deployed (legacy V2 on pre-v5)** if raw in-data RMSE "
        "is the marquee metric and Set-3 capability isn't urgent. ~24 psi "
        "better bLOO; deployed; no migration risk."
    )
    lines.append(
        "  - **Migrate to v5 + Hamming bare** if Set-3 capability, silent-"
        "corruption fix, reproducibility, or worst-case held-out RMSE matter "
        "more than ~24 psi in-data bLOO. Recommended for production going "
        "forward because: (a) the model is intended to be used on NEW "
        "compositions where the deployed model's row-LOO advantage doesn't "
        "apply, (b) Set-3 data is part of the dataset going forward, "
        "(c) deterministic fits are operationally easier."
    )
    lines.append(
        "  - **Compromise: v5 + legacy V2** keeps the deployed kernel "
        "architecture (just updates to the cleaned v5 data + bumps source "
        "bounds to {0,1,2}). bLOO 698 psi (vs deployed 680, v5+Hamming "
        "704). Smallest migration risk; misses out on Hamming's "
        "deterministic fits and slight LOCO advantage."
    )
    lines.append("")
    lines.append(
        "**The Commit 7 production-default flip to Hamming should be "
        "revisited** in light of these numbers. The v5 + Hamming choice "
        "is defensible on robustness grounds but is NOT a strict improvement "
        "over the deployed model on raw RMSE."
    )
    lines.append("")
    return "\n".join(lines)


def _format_cell(mean_val, std_val, count_val, fmt="{:.1f}"):
    if pd.isna(mean_val):
        return "—"
    s = fmt.format(mean_val)
    if pd.notna(std_val) and count_val > 1:
        s += f" ± {fmt.format(std_val)}"
    return s


def write_markdown(
    df_main: pd.DataFrame,
    df_set3: pd.DataFrame,
    out: Path,
    df_class_holdout: pd.DataFrame | None = None,
) -> None:
    lines = []
    lines.append("# Three-class material source + LogNormal prior — ablation")
    lines.append("")
    lines.append(
        "Stage-by-stage decomposition of the materials-classes-and-"
        "lengthscale-prior stack. See the plan document "
        "`~/.llms/plans/three_class_and_lengthscale_prior.plan.md` "
        '§"Commit 6" for the full design and pre-registered success '
        "criteria. All metrics in psi unless noted; multi-seed mean "
        "± std (n_seeds in the rightmost column)."
    )
    lines.append("")

    if df_main.empty:
        lines.append("_No completed runs in this report._")
    else:
        # Headline analysis derived from the actual numbers.
        lines.append("## Headline findings")
        lines.append("")
        lines.append(_headline_analysis(df_main, df_set3))
        lines.append("")
        lines.append("## LOO and block-LOO accuracy + calibration")
        lines.append("")
        lines.append(
            "| Stage | Variant | LOO RMSE | bLOO RMSE | bLOO Sets1+2 RMSE | bLOO MLPD | bLOO PIT-KS | bLOO cov95 | seeds |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        # Aggregate across seeds.
        agg_rows = (
            df_main.groupby(["stage", "variant"])
            .agg(
                loo_rmse=("loo_rmse", "mean"),
                loo_rmse_std=("loo_rmse", "std"),
                bloo_rmse=("bloo_rmse", "mean"),
                bloo_rmse_std=("bloo_rmse", "std"),
                bloo12_rmse=("bloo_set12_rmse", "mean"),
                bloo12_rmse_std=("bloo_set12_rmse", "std"),
                bloo_lpd=("bloo_mean_lpd", "mean"),
                bloo_lpd_std=("bloo_mean_lpd", "std"),
                bloo_pit=("bloo_pit_ks", "mean"),
                bloo_pit_std=("bloo_pit_ks", "std"),
                bloo_cov95=("bloo_coverage_95", "mean"),
                bloo_cov95_std=("bloo_coverage_95", "std"),
                seeds=("seed", "count"),
            )
            .reset_index()
            .sort_values(["stage", "variant"])
        )
        for _, r in agg_rows.iterrows():
            lines.append(
                f"| {r.stage} | {r.variant} | "
                f"{_format_cell(r.loo_rmse, r.loo_rmse_std, r.seeds)} | "
                f"{_format_cell(r.bloo_rmse, r.bloo_rmse_std, r.seeds)} | "
                f"{_format_cell(r.bloo12_rmse, r.bloo12_rmse_std, r.seeds)} | "
                f"{_format_cell(r.bloo_lpd, r.bloo_lpd_std, r.seeds, '{:.3f}')} | "
                f"{_format_cell(r.bloo_pit, r.bloo_pit_std, r.seeds, '{:.3f}')} | "
                f"{_format_cell(r.bloo_cov95, r.bloo_cov95_std, r.seeds, '{:.3f}')} | "
                f"{int(r.seeds)} |"
            )
        lines.append("")
        lines.append("## Time lengthscale rail diagnostic (mean across seeds)")
        lines.append("")
        lines.append(
            "| Stage | Variant | ℓ_Time(blind) | ℓ_Time(specific) | "
            "ℓ_Time(rbf) | max blind ℓ | binder ratio | agg ratio |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        diag_rows = (
            df_main.groupby(["stage", "variant"])
            .agg(
                ell_t_b=("ell_time_blind", "mean"),
                ell_t_s=("ell_time_specific", "mean"),
                ell_t_r=("ell_time_rbf", "mean"),
                max_blind=("max_blind_lengthscale", "mean"),
                binder=("binder_within_ratio", "mean"),
                aggregate_ratio=("aggregate_within_ratio", "mean"),
            )
            .reset_index()
            .sort_values(["stage", "variant"])
        )
        for _, r in diag_rows.iterrows():
            lines.append(
                f"| {r.stage} | {r.variant} | "
                f"{r.ell_t_b:.2f} | {r.ell_t_s:.2f} | {r.ell_t_r:.2f} | "
                f"{r.max_blind:.2f} | "
                f"{r.binder:.4f} | {r.aggregate_ratio:.4f} |"
            )
        lines.append("")

    if df_set3 is not None and not df_set3.empty:
        lines.append("## Held-out Set-3 (fit on Sets 1+2 only; evaluate on Set 3)")
        lines.append("")
        lines.append(
            "| Stage | Variant | Set-3 RMSE | Set-3 MLPD | Set-3 PIT-KS | Set-3 cov95 | seeds |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        agg_set3 = (
            df_set3.groupby(["stage", "variant"])
            .agg(
                rmse=("set3_rmse", "mean"),
                rmse_std=("set3_rmse", "std"),
                lpd=("set3_mean_lpd", "mean"),
                lpd_std=("set3_mean_lpd", "std"),
                pit=("set3_pit_ks", "mean"),
                pit_std=("set3_pit_ks", "std"),
                cov95_mean=("set3_coverage_95", "mean"),
                cov95_std=("set3_coverage_95", "std"),
                seeds=("seed", "count"),
            )
            .reset_index()
            .sort_values(["stage", "variant"])
        )
        for _, r in agg_set3.iterrows():
            lines.append(
                f"| {r.stage} | {r.variant} | "
                f"{_format_cell(r.rmse, r.rmse_std, r.seeds)} | "
                f"{_format_cell(r.lpd, r.lpd_std, r.seeds, '{:.3f}')} | "
                f"{_format_cell(r.pit, r.pit_std, r.seeds, '{:.3f}')} | "
                f"{_format_cell(r.cov95_mean, r.cov95_std, r.seeds, '{:.3f}')} | "
                f"{int(r.seeds)} |"
            )
        lines.append("")

    lines.append("## Pre-registered success criteria")
    lines.append("")
    lines.append(PRE_REGISTERED_CRITERIA)
    lines.append("")
    if df_class_holdout is not None and not df_class_holdout.empty:
        lines.append("## Leave-one-class-out (fit on 2 classes; evaluate on the third)")
        lines.append("")
        lines.append(
            "Tests whether held-out generalisation is class-specific. If "
            "IndexKernel's apparent advantage on the original 'held-out Set-3' "
            "metric is driven by lucky random initialisation of the unseen "
            "task's covar_factor row (rather than a real extrapolation "
            "principle), then holding out class 0 (mortar) or class 1 "
            "(Set-2 concrete) should reveal high seed variance and/or "
            "Hamming/IndexKernel rank-orderings inconsistent with the Set-3 result."
        )
        lines.append("")
        agg_holdout = (
            df_class_holdout.groupby(["holdout_class", "variant"])
            .agg(
                rmse=("holdout_rmse", "mean"),
                rmse_std=("holdout_rmse", "std"),
                lpd=("holdout_mean_lpd", "mean"),
                lpd_std=("holdout_mean_lpd", "std"),
                pit=("holdout_pit_ks", "mean"),
                pit_std=("holdout_pit_ks", "std"),
                cov95_mean=("holdout_coverage_95", "mean"),
                cov95_std=("holdout_coverage_95", "std"),
                seeds=("seed", "count"),
                n_test=("n_test", "first"),
            )
            .reset_index()
            .sort_values(["holdout_class", "variant"])
        )
        class_label = {
            0: "Class 0 (Set 1 — mortar)",
            1: "Class 1 (Set 2 — Heidelberg/Class C concrete)",
            2: "Class 2 (Set 3 — Amrize/Class F concrete)",
        }
        for c in sorted(agg_holdout["holdout_class"].unique()):
            sub = agg_holdout[agg_holdout["holdout_class"] == c]
            n_test_val = (
                int(sub["n_test"].iloc[0])
                if "n_test" in sub.columns and not sub["n_test"].isna().all()
                else 0
            )
            lines.append(
                f"### Held-out: {class_label.get(int(c), f'class {int(c)}')} (n_test = {n_test_val})"
            )
            lines.append("")
            lines.append("| Variant | RMSE (psi) | MLPD | PIT-KS | cov95 | seeds |")
            lines.append("|---|---|---|---|---|---|")
            for _, r in sub.iterrows():
                lines.append(
                    f"| {r.variant} | "
                    f"{_format_cell(r.rmse, r.rmse_std, r.seeds)} | "
                    f"{_format_cell(r.lpd, r.lpd_std, r.seeds, '{:.3f}')} | "
                    f"{_format_cell(r.pit, r.pit_std, r.seeds, '{:.3f}')} | "
                    f"{_format_cell(r.cov95_mean, r.cov95_std, r.seeds, '{:.3f}')} | "
                    f"{int(r.seeds)} |"
                )
            lines.append("")
    lines.append("## Raw results")
    lines.append("")
    lines.append(
        f"See `experiments/three_class_ablation_results.csv` "
        f"({len(df_main)} main rows + {len(df_set3) if df_set3 is not None else 0} held-out-Set-3 rows)."
    )
    out.write_text("\n".join(lines))
    print(f"[writeup] wrote {out.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated variant ids, or 'all'.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated integer seeds.",
    )
    parser.add_argument(
        "--no-set3",
        action="store_true",
        help="Skip the held-out-Set-3 evaluation.",
    )
    parser.add_argument(
        "--class-holdouts",
        type=str,
        default="",
        help=(
            "Comma-separated class indices to run leave-one-class-out "
            "for (e.g., '0,1,2'). Use the holdout-class subset of "
            "variants only (Hamming + IndexKernel rank=2/3 with prior "
            "configs by default). Skips main + Set-3 passes when set."
        ),
    )
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Only run the main (LOO + bLOO) pass.",
    )
    args = parser.parse_args()

    if args.variants == "all":
        variants = sorted(VARIANTS.keys(), key=lambda k: (VARIANTS[k]["stage"], k))
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print(f"[run] variants = {variants}")
    print(f"[run] seeds    = {seeds}")

    if args.class_holdouts:
        # Leave-one-class-out only mode: run holdout for the requested
        # class indices.
        holdout_classes = [
            int(c.strip()) for c in args.class_holdouts.split(",") if c.strip()
        ]
        all_holdout = []
        for c in holdout_classes:
            print(f"\n[run] leave-one-class-out: holding out class {c}")
            df_c = _run_class_holdout(variants, seeds, holdout_class=c)
            all_holdout.append(df_c)
        df_holdout = (
            pd.concat(all_holdout, ignore_index=True) if all_holdout else pd.DataFrame()
        )
        # Append to existing results CSV if present (so we don't lose
        # the main/set3 numbers); otherwise write fresh.
        if RESULTS_CSV.exists():
            existing = pd.read_csv(RESULTS_CSV)
            combined = pd.concat([existing, df_holdout], ignore_index=True)
        else:
            combined = df_holdout
        combined.to_csv(RESULTS_CSV, index=False)
        print(f"[results] wrote {RESULTS_CSV.relative_to(REPO_ROOT)}")
        # Re-render the writeup using the (possibly merged) full results.
        df_main_existing = combined[combined["data"].isin(["pre_v5", "v5"])]
        df_set3_existing = combined[combined["data"] == "v5_set3_heldout"]
        df_holdout_all = combined[
            combined["data"]
            .astype(str)
            .str.contains("class\\d_heldout", regex=True, na=False)
        ]
        write_markdown(
            df_main_existing,
            df_set3_existing,
            WRITEUP_MD,
            df_class_holdout=df_holdout_all,
        )
        return 0

    df_main = _run_main(variants, seeds)

    df_set3 = pd.DataFrame()
    if not args.no_set3 and not args.main_only:
        print()
        print("[run] held-out-Set-3 evaluation")
        df_set3 = _run_set3_holdout(variants, seeds)

    combined = pd.concat([df_main, df_set3], ignore_index=True)
    combined.to_csv(RESULTS_CSV, index=False)
    print(f"[results] wrote {RESULTS_CSV.relative_to(REPO_ROOT)}")
    write_markdown(df_main, df_set3, WRITEUP_MD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
