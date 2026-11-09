#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Architecture ablation grid with the high-strength M81/Mix_81 twin
DROPPED from each dataset.

Reuses the metric helpers from ``experiments.three_class_ablation``
(``_eval_metrics`` -> LOO + bLOO + Sets-1+2-bLOO) so the metric
definitions are consistent with the existing comprehensive ablation.

Grid:
    data: pre-v5 (with twins) | pre-v5 (drop M81) | v5 (with twins) | v5 (drop M61)
    source: legacy_continuous_ard | hamming | indexkernel_r2
    prior: bare (no LogN, no time-tying)

Output: ``experiments/M81_DROP_FULL_METRICS.md`` and
``experiments/m81_drop_full_metrics.csv``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from boxcrete import fit_strength_gp, load_concrete_strength  # noqa: E402
from experiments.three_class_ablation import _eval_metrics  # noqa: E402

PRE_V5_FIXTURE = REPO_ROOT / "test" / "fixtures" / "boxcrete_data_pre_v5.csv"
V5_DATA = REPO_ROOT / "data" / "boxcrete_data.csv"
WRITEUP = REPO_ROOT / "experiments" / "M81_DROP_FULL_METRICS.md"
RESULTS_CSV = REPO_ROOT / "experiments" / "m81_drop_full_metrics.csv"

PRE_V5_HIGH = "Mix_81"
V5_HIGH = "M61"


def _patched_fit_setup(source_kernel: str):
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


def fit_and_eval(data_path: Path, source_kernel: str) -> dict[str, float]:
    apply, restore = _patched_fit_setup(source_kernel)
    apply()
    try:
        torch.manual_seed(0)
        ds = load_concrete_strength(data_path=str(data_path))
        X, Y, Yvar, _ = ds.strength_data
        model = fit_strength_gp(
            X=X, Y=Y, Yvar=Yvar, X_bounds=ds.bounds, seed=0
        )
        m = _eval_metrics(model, n_real=X.shape[0])
        m["n_train"] = int(X.shape[0])
        return m
    finally:
        restore()


def make_drop_csv(src_csv: Path, mix_to_drop: str, suffix: str) -> Path:
    df = pd.read_csv(src_csv)
    df = df[df["Mix Name"] != mix_to_drop].copy()
    out = REPO_ROOT / "experiments" / f"_m81drop_tmp_{src_csv.stem}_{suffix}.csv"
    df.to_csv(out, index=False)
    return out


def main() -> int:
    pre_drop = make_drop_csv(PRE_V5_FIXTURE, PRE_V5_HIGH, "drop_high")
    v5_drop = make_drop_csv(V5_DATA, V5_HIGH, "drop_high")

    grid = [
        ("pre-v5", "with twins",   PRE_V5_FIXTURE),
        ("pre-v5", "drop high",    pre_drop),
        ("v5",     "with twins",   V5_DATA),
        ("v5",     "drop high",    v5_drop),
    ]
    kernels = ["legacy_continuous_ard", "hamming", "indexkernel_r2"]

    rows: list[dict] = []
    for ds_label, twin_label, csv_path in grid:
        for kernel in kernels:
            print(f"[fit] {ds_label} | {twin_label} | {kernel}")
            t0 = time.time()
            try:
                m = fit_and_eval(csv_path, kernel)
                row = {
                    "data": ds_label,
                    "twin_variant": twin_label,
                    "source_kernel": kernel,
                    "wall_sec": time.time() - t0,
                }
                row.update(m)
                rows.append(row)
                print(
                    f"   loo_rmse={row.get('loo_rmse'):.0f} "
                    f"bloo_rmse={row.get('bloo_rmse'):.0f} "
                    f"cov95={row.get('bloo_coverage_95', float('nan')):.2f} "
                    f"pit_ks={row.get('bloo_pit_ks', float('nan')):.3f}"
                )
            except Exception as exc:
                rows.append({
                    "data": ds_label,
                    "twin_variant": twin_label,
                    "source_kernel": kernel,
                    "fit_error": str(exc)[:200],
                })
                print(f"   FAILED: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[csv] wrote {RESULTS_CSV.relative_to(REPO_ROOT)}")

    # Build markdown writeup.
    out_lines: list[str] = []
    out_lines.append("# Architecture ablation: drop-high-twin (M81/Mix_81)")
    out_lines.append("")
    out_lines.append(
        "Re-run the source-kernel × dataset grid after dropping the "
        "high-strength twin (Mix_81 in pre-v5, M61 in v5). Metrics are "
        "computed by ``experiments.three_class_ablation._eval_metrics``: "
        "LOO + bLOO + Sets-1+2-only bLOO, each with RMSE, MAE, "
        "coverage_95, PIT-KS, MLPD, CRPS."
    )
    out_lines.append("")

    headers_main = [
        "data", "twin_variant", "source_kernel", "n_train",
        "loo_rmse", "loo_mae", "loo_coverage_95", "loo_pit_ks", "loo_mlpd", "loo_crps",
        "bloo_rmse", "bloo_mae", "bloo_coverage_95", "bloo_pit_ks", "bloo_mlpd", "bloo_crps",
        "bloo_set12_rmse", "bloo_set12_coverage_95", "bloo_set12_pit_ks", "bloo_set12_mlpd",
    ]
    label_map = {
        "data": "data",
        "twin_variant": "twin",
        "source_kernel": "kernel",
        "n_train": "n",
        "loo_rmse": "LOO RMSE",
        "loo_mae": "LOO MAE",
        "loo_coverage_95": "LOO cov95",
        "loo_pit_ks": "LOO PIT-KS",
        "loo_mlpd": "LOO MLPD",
        "loo_crps": "LOO CRPS",
        "bloo_rmse": "bLOO RMSE",
        "bloo_mae": "bLOO MAE",
        "bloo_coverage_95": "bLOO cov95",
        "bloo_pit_ks": "bLOO PIT-KS",
        "bloo_mlpd": "bLOO MLPD",
        "bloo_crps": "bLOO CRPS",
        "bloo_set12_rmse": "S12-bLOO RMSE",
        "bloo_set12_coverage_95": "S12-bLOO cov95",
        "bloo_set12_pit_ks": "S12-bLOO PIT-KS",
        "bloo_set12_mlpd": "S12-bLOO MLPD",
    }

    def fmt(v):
        if v is None:
            return "--"
        if isinstance(v, str):
            return v
        if isinstance(v, float):
            if abs(v) > 50:
                return f"{v:.0f}"
            return f"{v:.3f}"
        return str(v)

    out_lines.append("## Full metrics table")
    out_lines.append("")
    out_lines.append("| " + " | ".join(label_map[h] for h in headers_main) + " |")
    out_lines.append("|" + "|".join(["---"] * len(headers_main)) + "|")
    for row in rows:
        cells = [fmt(row.get(h)) for h in headers_main]
        out_lines.append("| " + " | ".join(cells) + " |")
    out_lines.append("")

    # Summary of deltas (drop high - with twins).
    out_lines.append("## Δ from dropping the high twin (drop − with)")
    out_lines.append("")
    out_lines.append(
        "Negative Δ on RMSE/MAE/PIT-KS = improvement; positive Δ on "
        "coverage_95 = improvement (toward 0.95)."
    )
    out_lines.append("")
    delta_cols = [
        "loo_rmse", "bloo_rmse", "bloo_set12_rmse",
        "loo_coverage_95", "bloo_coverage_95",
        "loo_pit_ks", "bloo_pit_ks",
        "loo_mlpd", "bloo_mlpd",
    ]
    out_lines.append(
        "| data | kernel | "
        + " | ".join(f"Δ {label_map[c]}" for c in delta_cols)
        + " |"
    )
    out_lines.append("|" + "|".join(["---"] * (2 + len(delta_cols))) + "|")
    by = {(r["data"], r.get("twin_variant"), r["source_kernel"]): r for r in rows}
    for ds in ["pre-v5", "v5"]:
        for k in kernels:
            full = by.get((ds, "with twins", k), {})
            drop = by.get((ds, "drop high", k), {})
            cells = []
            for c in delta_cols:
                a = full.get(c)
                b = drop.get(c)
                if a is None or b is None:
                    cells.append("--")
                else:
                    delta = b - a
                    cells.append(f"{delta:+.3f}" if abs(delta) <= 50 else f"{delta:+.0f}")
            out_lines.append(f"| {ds} | {k} | " + " | ".join(cells) + " |")

    WRITEUP.write_text("\n".join(out_lines) + "\n")
    print(f"[writeup] wrote {WRITEUP.relative_to(REPO_ROOT)}")

    for p in [pre_drop, v5_drop]:
        if p.exists():
            p.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
