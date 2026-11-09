#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Follow-up tests after the main investigation.

The summary table from investigate_v5_vs_pre_v5.py:

    pre-v5 + legacy V2:                           680
    v5 + legacy V2:                               698
    v5 (drop new rows) + legacy V2:               702
    v5 + 2-class relabel + legacy V2:             701
    v5 (common comps only) + legacy V2:           702

ALL four v5 modifications cluster at 698-702 — none of the standard
hypotheses (3-class, added rows, comp-set difference) explains the gap.

Two more tests:

  H7  Restrict pre-v5 to (comp, temp, time) tuples present in v5;
      refit. If pre-v5(common) ≈ pre-v5(full) ≈ 680 → the 9 extra
      pre-v5 rows aren't the cause. If pre-v5(common) → 700ish then
      they are.

  H8  Yvar / Strength (Std) comparison: for the 638 matching tuples,
      do the variances differ between pre-v5 and v5?
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from boxcrete import fit_strength_gp, load_concrete_strength  # noqa: E402
from experiments.model_variant_study import block_loo_metrics  # noqa: E402

PRE_V5_FIXTURE = REPO_ROOT / "test" / "fixtures" / "boxcrete_data_pre_v5.csv"
V5_DATA = REPO_ROOT / "data" / "boxcrete_data.csv"
WRITEUP = REPO_ROOT / "experiments" / "V5_VS_PRE_V5_INVESTIGATION.md"

COMP_COLS = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
]

OUTPUT: list[str] = []


def emit(msg: str = "") -> None:
    print(msg)
    OUTPUT.append(msg)


def comp_key(row, decimals: int = 2) -> tuple:
    keys = COMP_COLS + ["Temp (C)", "Time"]
    parts = []
    for c in keys:
        v = row.get(c)
        try:
            parts.append(round(float(v), decimals))
        except (TypeError, ValueError):
            parts.append(None)
    return tuple(parts)


def fit_legacy(data_path: Path) -> dict[str, float]:
    """Fit legacy continuous-ARD V2 and return bLOO RMSE."""
    from boxcrete import kernels as _kmod
    from boxcrete import priors as _pmod
    from boxcrete import strength_model as _smod

    original_builder = _kmod.make_gated_strength_kernel_builder
    original_factory = _pmod.within_group_prior

    def _patched_builder(gate_tau=0.05, **kwargs):
        return original_builder(
            gate_tau=gate_tau,
            source_kernel="legacy_continuous_ard",
            time_tying_sigma=None,
        )

    def _patched_factory(*args, **kwargs):
        kwargs.setdefault("include_lognormal_baseline", False)
        return original_factory(*args, **kwargs)

    _kmod.make_gated_strength_kernel_builder = _patched_builder
    _smod.make_gated_strength_kernel_builder = _patched_builder
    _pmod.within_group_prior = _patched_factory
    _kmod.within_group_prior = _patched_factory

    try:
        torch.manual_seed(0)
        ds = load_concrete_strength(data_path=str(data_path))
        X, Y, Yvar, _ = ds.strength_data
        model = fit_strength_gp(
            X=X, Y=Y, Yvar=Yvar, X_bounds=ds.bounds, seed=0
        )
        bloo = block_loo_metrics(model, n_real=X.shape[0])
        return {"n_train": int(X.shape[0]), "bloo_rmse": float(bloo["rmse"])}
    finally:
        _kmod.make_gated_strength_kernel_builder = original_builder
        _smod.make_gated_strength_kernel_builder = original_builder
        _pmod.within_group_prior = original_factory
        _kmod.within_group_prior = original_factory


# ---------------------------------------------------------------------------
# H7: restrict pre-v5 to common tuples
# ---------------------------------------------------------------------------


def hypothesis_7_pre_v5_common(pre_v5: pd.DataFrame, v5: pd.DataFrame) -> None:
    emit("## H7: pre-v5 restricted to (comp, temp, time) tuples present in v5")
    emit("")
    emit(
        "If pre-v5(common) ≈ pre-v5(full) = 680 → the 9 extra pre-v5 rows "
        "aren't the cause. If pre-v5(common) goes up to ~700 → they ARE."
    )
    emit("")
    v5_keys = set(
        v5.dropna(subset=["Strength (Mean)"]).apply(comp_key, axis=1)
    )
    pre_clean = pre_v5.dropna(subset=["Strength (Mean)"]).copy()
    pre_clean["_key"] = pre_clean.apply(comp_key, axis=1)
    pre_common = pre_clean[pre_clean["_key"].isin(v5_keys)].drop(
        columns=["_key"]
    )
    emit(
        f"pre-v5 rows after restricting to v5 tuples: "
        f"{len(pre_common)} (vs original = {len(pre_clean)})"
    )

    tmp_csv = REPO_ROOT / "experiments" / "_pre_v5_common_tmp.csv"
    pre_common.to_csv(tmp_csv, index=False)
    try:
        result = fit_legacy(tmp_csv)
        emit(
            f"pre-v5 (common only) + legacy V2: bLOO RMSE = "
            f"{result['bloo_rmse']:.0f} psi (n={result['n_train']})"
        )
        emit("  (vs pre-v5 + legacy V2: 680 psi)")
        emit("  (vs v5 + legacy V2: 698 psi)")
    except Exception as exc:
        emit(f"FIT FAILED: {exc}")
    finally:
        if tmp_csv.exists():
            tmp_csv.unlink()
    emit("")


# ---------------------------------------------------------------------------
# H8: Strength (Std) / Yvar comparison for matching tuples
# ---------------------------------------------------------------------------


def hypothesis_8_yvar(pre_v5: pd.DataFrame, v5: pd.DataFrame) -> None:
    emit("## H8: Yvar / Strength (Std) comparison for matching tuples")
    emit("")
    pre = pre_v5.dropna(subset=["Strength (Mean)"]).copy()
    v5_clean = v5.dropna(subset=["Strength (Mean)"]).copy()
    pre["_key"] = pre.apply(comp_key, axis=1)
    v5_clean["_key"] = v5_clean.apply(comp_key, axis=1)
    std_col = "Strength (Std)"
    if std_col not in pre.columns or std_col not in v5_clean.columns:
        emit(f"WARNING: '{std_col}' column missing — skipping Yvar comparison.")
        emit(f"   pre-v5 columns: {list(pre.columns)}")
        emit(f"   v5 columns:     {list(v5_clean.columns)}")
        emit("")
        return
    pre_by_key = pre.groupby("_key")[std_col].first()
    v5_by_key = v5_clean.groupby("_key")[std_col].first()
    merged = pd.DataFrame({"pre_std": pre_by_key, "v5_std": v5_by_key}).dropna(
        how="any"
    )
    merged["delta"] = merged["v5_std"] - merged["pre_std"]
    merged["abs_delta"] = merged["delta"].abs()
    emit(f"Common keys with Strength (Std) in both: {len(merged)}")
    if len(merged) == 0:
        emit("")
        return
    emit(f"Mean abs delta:        {merged['abs_delta'].mean():.2f} psi")
    emit(f"Max  abs delta:        {merged['abs_delta'].max():.2f} psi")
    emit(f"#  abs delta > 1 psi:  {(merged['abs_delta'] > 1).sum()}")
    emit(f"#  abs delta > 10 psi: {(merged['abs_delta'] > 10).sum()}")
    emit(f"#  abs delta > 50 psi: {(merged['abs_delta'] > 50).sum()}")
    emit("")
    if (merged["abs_delta"] > 10).sum() > 0:
        emit("Worst Yvar mismatches:")
        worst = merged.nlargest(10, "abs_delta")
        emit("```")
        emit(worst.to_string())
        emit("```")
    emit("")


def main() -> int:
    pre_v5 = pd.read_csv(PRE_V5_FIXTURE)
    v5 = pd.read_csv(V5_DATA)

    emit("# Follow-up tests (H7 + H8)")
    emit("")
    hypothesis_8_yvar(pre_v5, v5)
    hypothesis_7_pre_v5_common(pre_v5, v5)

    # Append to existing writeup.
    existing = WRITEUP.read_text() if WRITEUP.exists() else ""
    WRITEUP.write_text(existing + "\n\n" + "\n".join(OUTPUT))
    print(f"\n[writeup] appended H7+H8 to {WRITEUP.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
