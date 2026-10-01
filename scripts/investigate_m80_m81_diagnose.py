#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Two-part follow-up to ``investigate_m80_m81_drop.py``:

PART A — Which twin is closer to model predictions?
    Hold out BOTH twins (Mix_80/Mix_81 in pre-v5; M60/M61 in v5) and
    obtain bLOO predictions for each held-out row. Compare the bLOO
    posterior mean against the M80 and M81 strength values at each
    (composition, time) tuple. Whichever value is closer is the
    "more consistent with the rest of the dataset" twin; the other
    is the outlier the GP would not predict.

PART B — Architecture ablations with only ONE twin retained
    Re-run the kernel × dataset grid with two NEW data variants:
    * keep only Mix_80 / M60   (drop the higher-strength twin)
    * keep only Mix_81 / M61   (drop the lower-strength twin)
    Compare bLOO RMSE against {with both twins} and {drop both
    twins}. If conclusions change between "keep M80 only" and
    "keep M81 only", the architecture is sensitive to outliers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from boxcrete import fit_strength_gp, load_concrete_strength  # noqa: E402
from experiments.model_variant_study import block_loo_metrics  # noqa: E402

PRE_V5_FIXTURE = REPO_ROOT / "test" / "fixtures" / "boxcrete_data_pre_v5.csv"
V5_DATA = REPO_ROOT / "data" / "boxcrete_data.csv"
WRITEUP = REPO_ROOT / "experiments" / "M80_M81_TWIN_DIAGNOSIS.md"

# Mix-name pairs in each dataset (low-strength twin, high-strength twin).
# Verified earlier — Mix_80/M60 are the LOW twin; Mix_81/M61 are the HIGH twin.
PRE_V5_LOW = "Mix_80"
PRE_V5_HIGH = "Mix_81"
V5_LOW = "M60"
V5_HIGH = "M61"


def _patched_fit_setup(source_kernel: str):
    """Build patches for the GP fit. Returns (apply, restore) callables."""
    from boxcrete import kernels as _kmod
    from boxcrete import priors as _pmod
    from boxcrete import strength_model as _smod

    original_builder = _kmod.make_gated_strength_kernel_builder
    original_factory = _pmod.within_group_prior

    def _patched_builder(gate_tau=0.05, **kwargs):
        return original_builder(
            gate_tau=gate_tau,
            source_kernel=source_kernel,
            time_tying_sigma=None,
        )

    def _patched_factory(*args, **kwargs):
        kwargs.setdefault("include_lognormal_baseline", False)
        return original_factory(*args, **kwargs)

    def apply():
        _kmod.make_gated_strength_kernel_builder = _patched_builder
        _smod.make_gated_strength_kernel_builder = _patched_builder
        _pmod.within_group_prior = _patched_factory
        _kmod.within_group_prior = _patched_factory

    def restore():
        _kmod.make_gated_strength_kernel_builder = original_builder
        _smod.make_gated_strength_kernel_builder = original_builder
        _pmod.within_group_prior = original_factory
        _kmod.within_group_prior = original_factory

    return apply, restore


def fit_dataset(data_path: Path, source_kernel: str) -> dict[str, float]:
    apply, restore = _patched_fit_setup(source_kernel)
    apply()
    try:
        torch.manual_seed(0)
        ds = load_concrete_strength(data_path=str(data_path))
        X, Y, Yvar, _ = ds.strength_data
        model = fit_strength_gp(
            X=X, Y=Y, Yvar=Yvar, X_bounds=ds.bounds, seed=0
        )
        bloo = block_loo_metrics(model, n_real=X.shape[0])
        return {"n": int(X.shape[0]), "bloo_rmse": float(bloo["rmse"])}
    finally:
        restore()


def make_drop_csv(src_csv: Path, mix_names_to_drop: list[str], suffix: str) -> Path:
    df = pd.read_csv(src_csv)
    df = df[~df["Mix Name"].isin(mix_names_to_drop)].copy()
    out = REPO_ROOT / "experiments" / f"_twin_tmp_{src_csv.stem}_{suffix}.csv"
    df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# PART A: hold both twins out, compare bLOO posterior to M80 and M81
# ---------------------------------------------------------------------------


def part_a_predict_held_out_pair(
    data_path: Path,
    low_name: str,
    high_name: str,
    source_kernel: str,
    label: str,
    out_lines: list[str],
):
    """Train GP on data WITHOUT the twin pair, then predict at the
    twin pair's compositions/times. Compare predictions to actual
    LOW-twin and HIGH-twin strengths."""

    full_df = pd.read_csv(data_path)
    twin_rows = full_df[full_df["Mix Name"].isin([low_name, high_name])].copy()
    twin_rows_sorted = twin_rows.sort_values(["Mix Name", "Time"]).reset_index(drop=True)

    # Build a "training" CSV that drops both twins.
    drop_csv = make_drop_csv(data_path, [low_name, high_name], "predict")

    apply, restore = _patched_fit_setup(source_kernel)
    apply()
    try:
        torch.manual_seed(0)
        ds_train = load_concrete_strength(data_path=str(drop_csv))
        X_tr, Y_tr, Yvar_tr, _ = ds_train.strength_data
        # ``maxscale_zeromean``-style normalisation factor: predictions
        # come back in [0, 1] divided by Y_max of training data. Compute
        # it from the training tensor for the unscaling.
        y_max = float(Y_tr.max().item())
        model = fit_strength_gp(
            X=X_tr, Y=Y_tr, Yvar=Yvar_tr, X_bounds=ds_train.bounds, seed=0
        )
        # Now build X_test for the twin rows. Use the same loader to
        # get correct preprocessing.
        ds_full = load_concrete_strength(data_path=str(data_path))
        X_full, Y_full, _, _ = ds_full.strength_data

        # Identify twin row indices in the full dataset by mix-name.
        # load_concrete_strength preserves row order.
        full_after_drop_nan = pd.read_csv(data_path).dropna(
            subset=["Strength (Mean)"]
        ).reset_index(drop=True)
        twin_idx_low = full_after_drop_nan.index[
            full_after_drop_nan["Mix Name"] == low_name
        ].tolist()
        twin_idx_high = full_after_drop_nan.index[
            full_after_drop_nan["Mix Name"] == high_name
        ].tolist()

        X_low = X_full[twin_idx_low]
        X_high = X_full[twin_idx_high]
        Y_low = Y_full[twin_idx_low]
        Y_high = Y_full[twin_idx_high]

        model.eval()
        with torch.no_grad():
            post_low = model.posterior(X_low)
            post_high = model.posterior(X_high)
            mu_low = post_low.mean.squeeze(-1).cpu().numpy()
            sigma_low = post_low.variance.squeeze(-1).clamp_min(1e-12).sqrt().cpu().numpy()
            mu_high = post_high.mean.squeeze(-1).cpu().numpy()
            sigma_high = post_high.variance.squeeze(-1).clamp_min(1e-12).sqrt().cpu().numpy()

        # Unscale from normalised [0, 1] back to psi.
        mu_low = mu_low * y_max
        sigma_low = sigma_low * y_max
        mu_high = mu_high * y_max
        sigma_high = sigma_high * y_max

        y_low = Y_low.cpu().numpy().ravel()
        y_high = Y_high.cpu().numpy().ravel()
        # X_low and X_high should be at the same composition/time (same fingerprint).
        # Their predictions should be identical (same X). Take mean.
        mu_pair = 0.5 * (mu_low + mu_high)
        sigma_pair = 0.5 * (sigma_low + sigma_high)

        # Scale back from log space. The GP outputs log10(strength) by default
        # when load_concrete_strength applies the log transform. Check whether
        # we need to invert it.
        # Inspection: load_concrete_strength does NOT log-transform by default.
        # mu_pair is in psi units already.

        # Per-time delta:
        rows_per_t = sorted(set(twin_rows_sorted["Time"].tolist()))
        out_lines.append(f"### {label} | source={source_kernel}")
        out_lines.append("")
        out_lines.append(
            f"Trained on n={X_tr.shape[0]} rows (twin pair removed). "
            f"Posterior at twin compositions:"
        )
        out_lines.append("")
        out_lines.append(
            f"| Time | {low_name} actual | {high_name} actual | "
            f"GP μ | GP σ | |μ−{low_name}| | |μ−{high_name}| | closer twin |"
        )
        out_lines.append(
            "|------|---|---|---|---|---|---|---|"
        )
        closer_to_low = 0
        closer_to_high = 0
        sse_low = 0.0
        sse_high = 0.0
        for i, t in enumerate(rows_per_t):
            yl = float(y_low[i])
            yh = float(y_high[i])
            mu = float(mu_pair[i])
            sigma = float(sigma_pair[i])
            d_low = abs(mu - yl)
            d_high = abs(mu - yh)
            sse_low += (mu - yl) ** 2
            sse_high += (mu - yh) ** 2
            closer = low_name if d_low < d_high else high_name
            if d_low < d_high:
                closer_to_low += 1
            else:
                closer_to_high += 1
            out_lines.append(
                f"| {t:g} | {yl:.0f} | {yh:.0f} | {mu:.0f} | {sigma:.0f} | "
                f"{d_low:.0f} | {d_high:.0f} | **{closer}** |"
            )
        rmse_low = float(np.sqrt(sse_low / max(len(rows_per_t), 1)))
        rmse_high = float(np.sqrt(sse_high / max(len(rows_per_t), 1)))
        out_lines.append("")
        out_lines.append(
            f"**Score**: {closer_to_low}/{len(rows_per_t)} timepoints closer to "
            f"{low_name}, {closer_to_high}/{len(rows_per_t)} closer to {high_name}. "
            f"RMSE vs {low_name} = {rmse_low:.0f} psi; "
            f"RMSE vs {high_name} = {rmse_high:.0f} psi."
        )
        out_lines.append("")
    finally:
        restore()
        if drop_csv.exists():
            drop_csv.unlink()


# ---------------------------------------------------------------------------
# PART B: keep-one-twin grid
# ---------------------------------------------------------------------------


def part_b_keep_one_grid(out_lines: list[str]):
    out_lines.append("## Part B — Architecture ablations with one twin retained")
    out_lines.append("")
    out_lines.append(
        "Re-run the bare source-kernel grid with two NEW data variants per "
        "dataset: `keep low only` (drop the high-strength twin) and "
        "`keep high only` (drop the low-strength twin)."
    )
    out_lines.append("")

    pre_keep_low = make_drop_csv(PRE_V5_FIXTURE, [PRE_V5_HIGH], "keep_low")
    pre_keep_high = make_drop_csv(PRE_V5_FIXTURE, [PRE_V5_LOW], "keep_high")
    v5_keep_low = make_drop_csv(V5_DATA, [V5_HIGH], "keep_low")
    v5_keep_high = make_drop_csv(V5_DATA, [V5_LOW], "keep_high")

    grid = [
        ("pre-v5", "keep low only",  pre_keep_low),
        ("pre-v5", "keep high only", pre_keep_high),
        ("v5",     "keep low only",  v5_keep_low),
        ("v5",     "keep high only", v5_keep_high),
    ]
    kernels = ["legacy_continuous_ard", "hamming"]
    out_lines.append(
        "| Dataset | twin variant | source kernel | n | bLOO RMSE (psi) |"
    )
    out_lines.append("|---|---|---|---|---|")
    rows = []
    for ds_label, twin_label, csv_path in grid:
        for kernel in kernels:
            print(f"[fit] {ds_label} | {twin_label} | source={kernel}")
            try:
                res = fit_dataset(csv_path, kernel)
                rows.append((ds_label, twin_label, kernel, res["n"], res["bloo_rmse"]))
                out_lines.append(
                    f"| {ds_label} | {twin_label} | {kernel} | "
                    f"{res['n']} | {res['bloo_rmse']:.0f} |"
                )
            except Exception as exc:
                out_lines.append(
                    f"| {ds_label} | {twin_label} | {kernel} | -- | "
                    f"FAILED: {exc} |"
                )
    for p in [pre_keep_low, pre_keep_high, v5_keep_low, v5_keep_high]:
        if p.exists():
            p.unlink()

    # Reference values from previous run (in M80_M81_DROP_ABLATION.md):
    out_lines.append("")
    out_lines.append("### Reference values (from previous ablation)")
    out_lines.append("")
    out_lines.append(
        "| Dataset | source kernel | both twins | drop both |"
    )
    out_lines.append("|---|---|---|---|")
    out_lines.append("| pre-v5 | legacy_continuous_ard | 680 | 735 |")
    out_lines.append("| pre-v5 | hamming                | 725 | 769 |")
    out_lines.append("| v5     | legacy_continuous_ard | 702 | 736 |")
    out_lines.append("| v5     | hamming                | 738 | 768 |")
    out_lines.append("")


def main() -> int:
    out_lines: list[str] = []
    out_lines.append("# M80/M81 twin diagnosis — which is the outlier?")
    out_lines.append("")
    out_lines.append(
        "The Cement=667, Water=333 mortar pair (Mix_80/Mix_81 in pre-v5; "
        "M60/M61 in v5) records factor-2.3 different cylinder strengths "
        "for bit-identical input columns. Two diagnostics:"
    )
    out_lines.append("")
    out_lines.append(
        "1. **Hold both out, predict, and ask which is closer to the GP's "
        "best guess.** The closer twin is more consistent with the rest of "
        "the dataset; the farther twin is the outlier."
    )
    out_lines.append(
        "2. **Run architecture ablations with one twin retained** to see "
        "whether the kernel-comparison conclusions are sensitive to the "
        "outlier."
    )
    out_lines.append("")
    out_lines.append("## Part A — Which twin is the GP closer to?")
    out_lines.append("")
    for ds_label, path, low, high in [
        ("pre-v5", PRE_V5_FIXTURE, PRE_V5_LOW, PRE_V5_HIGH),
        ("v5",     V5_DATA,        V5_LOW,    V5_HIGH),
    ]:
        for kernel in ["legacy_continuous_ard", "hamming"]:
            print(f"[predict] {ds_label} | source={kernel}")
            part_a_predict_held_out_pair(
                path, low, high, kernel, ds_label, out_lines
            )

    part_b_keep_one_grid(out_lines)

    WRITEUP.write_text("\n".join(out_lines) + "\n")
    print(f"\n[writeup] wrote {WRITEUP.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
