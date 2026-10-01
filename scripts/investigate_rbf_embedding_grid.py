#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""RBF-embedding kernel ablation: compare against Hamming and IndexKernel.

Grid:
    data:    pre-v5, v5
    source:  hamming, indexkernel_r2,
             rbf_embedding_d1, rbf_embedding_d2, rbf_embedding_d3
    prior:   bare (no LogN, no time-tying)

Metrics: LOO + bLOO + Sets-1+2 bLOO with RMSE, MAE, coverage_95,
PIT-KS, MLPD, CRPS (via experiments.three_class_ablation._eval_metrics).
3 seeds per cell to detect init sensitivity.

Output:
    experiments/RBF_EMBEDDING_VS_HAMMING_VS_INDEXKERNEL.md
    experiments/rbf_embedding_grid.csv
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
WRITEUP = REPO_ROOT / "experiments" / "RBF_EMBEDDING_VS_HAMMING_VS_INDEXKERNEL.md"
RESULTS_CSV = REPO_ROOT / "experiments" / "rbf_embedding_grid.csv"


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


def fit_and_eval(
    data_path: Path, source_kernel: str, seed: int
) -> dict[str, float]:
    apply, restore = _patched_fit_setup(source_kernel)
    apply()
    try:
        torch.manual_seed(seed)
        ds = load_concrete_strength(data_path=str(data_path))
        X, Y, Yvar, _ = ds.strength_data
        model = fit_strength_gp(
            X=X, Y=Y, Yvar=Yvar, X_bounds=ds.bounds, seed=seed
        )
        m = _eval_metrics(model, n_real=X.shape[0])
        m["n_train"] = int(X.shape[0])

        # Snapshot learned class embeddings for RBF variants for
        # interpretability.
        if source_kernel.startswith("rbf_embedding"):
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "RBFEmbeddingKernel":
                    emb = mod.embeddings.detach().cpu().numpy()
                    ell = float(mod.lengthscale.detach().squeeze().item())
                    m["rbf_embedding_lengthscale"] = ell
                    for ci in range(emb.shape[0]):
                        for di in range(emb.shape[1]):
                            m[f"rbf_x_{ci}_{di}"] = float(emb[ci, di])
                    break
        return m
    finally:
        restore()


def main() -> int:
    grid = [
        ("pre-v5", PRE_V5_FIXTURE),
        ("v5",     V5_DATA),
    ]
    kernels = [
        "hamming",
        "indexkernel_r2",
        "rbf_embedding_d1",
        "rbf_embedding_d2",
        "rbf_embedding_d3",
    ]
    seeds = [0, 1, 2]

    rows: list[dict] = []
    for ds_label, csv_path in grid:
        for kernel in kernels:
            for seed in seeds:
                print(f"[fit] {ds_label} | {kernel} | seed={seed}")
                t0 = time.time()
                try:
                    m = fit_and_eval(csv_path, kernel, seed)
                    row = {
                        "data": ds_label,
                        "source_kernel": kernel,
                        "seed": seed,
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
                        "source_kernel": kernel,
                        "seed": seed,
                        "fit_error": str(exc)[:200],
                    })
                    print(f"   FAILED: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[csv] wrote {RESULTS_CSV.relative_to(REPO_ROOT)}")

    # Aggregate (mean, std) per (data, kernel) cell.
    metric_cols = [
        "loo_rmse", "loo_pit_ks", "loo_coverage_95", "loo_crps",
        "bloo_rmse", "bloo_pit_ks", "bloo_coverage_95", "bloo_crps",
        "bloo_set12_rmse", "bloo_set12_pit_ks", "bloo_set12_coverage_95",
    ]
    avail = [c for c in metric_cols if c in df.columns]
    agg = df.groupby(["data", "source_kernel"], dropna=False)[avail].agg(
        ["mean", "std"]
    )

    out: list[str] = []
    out.append("# RBF-embedding source kernel: ablation grid")
    out.append("")
    out.append(
        "Compares the new ``rbf_embedding_d{1,2,3}`` source kernel "
        "(learned per-class embedding + RBF distance in embedding "
        "space) against the existing ``hamming`` and "
        "``indexkernel_r2`` baselines."
    )
    out.append("")
    out.append(
        "**Setup**: bare prior (no LogNormal, no time-tying), full V2 "
        "B''+F5_alllog architecture, 3 seeds per cell, "
        "``experiments.three_class_ablation._eval_metrics`` for "
        "scoring (LOO + bLOO + Sets-1+2-bLOO each with RMSE, MAE, "
        "coverage_95, PIT-KS, MLPD, CRPS)."
    )
    out.append("")

    out.append("## Per-cell mean ± std across 3 seeds")
    out.append("")
    out.append(
        "| data | kernel | LOO RMSE | bLOO RMSE | bLOO cov95 | "
        "bLOO PIT-KS | S12-bLOO RMSE | S12-bLOO PIT-KS |"
    )
    out.append("|---|---|---|---|---|---|---|---|")
    for (data, kernel), sub in agg.iterrows():
        def cell(metric: str) -> str:
            if (metric, "mean") not in sub.index:
                return "--"
            mu = sub[(metric, "mean")]
            sigma = sub[(metric, "std")]
            if pd.isna(mu):
                return "--"
            if abs(mu) < 50:
                return f"{mu:.3f} ± {sigma:.3f}" if not pd.isna(sigma) else f"{mu:.3f}"
            return f"{mu:.0f} ± {sigma:.0f}" if not pd.isna(sigma) else f"{mu:.0f}"

        out.append(
            f"| {data} | {kernel} | "
            f"{cell('loo_rmse')} | {cell('bloo_rmse')} | "
            f"{cell('bloo_coverage_95')} | {cell('bloo_pit_ks')} | "
            f"{cell('bloo_set12_rmse')} | {cell('bloo_set12_pit_ks')} |"
        )
    out.append("")

    # Print per-seed RBF embeddings if present.
    rbf_rows = [r for r in rows if r.get("source_kernel", "").startswith("rbf_embedding")]
    if rbf_rows:
        out.append("## Learned RBF embeddings (best seed per cell)")
        out.append("")
        out.append(
            "Per-class embedding vectors fit by MLE. Class 0 is pinned at the "
            "origin and class 1 lies on the first axis (gauge fix); class 2's "
            "position is free."
        )
        out.append("")
        for ds_label in ["pre-v5", "v5"]:
            for k in ["rbf_embedding_d1", "rbf_embedding_d2", "rbf_embedding_d3"]:
                cell_rows = [
                    r for r in rbf_rows
                    if r.get("data") == ds_label and r.get("source_kernel") == k
                ]
                if not cell_rows:
                    continue
                # Pick best seed by bLOO RMSE.
                cell_rows.sort(key=lambda r: r.get("bloo_rmse", float("inf")))
                best = cell_rows[0]
                d = int(k.split("_d")[-1])
                out.append(f"### {ds_label} / {k} (best seed = {best['seed']})")
                out.append("")
                out.append(
                    f"Lengthscale: ``{best.get('rbf_embedding_lengthscale', float('nan')):.3f}``"
                )
                out.append("")
                out.append("| class | " + " | ".join(f"x_{i}" for i in range(d)) + " |")
                out.append("|---|" + "|".join(["---"] * d) + "|")
                for ci in range(3):
                    coords = [best.get(f"rbf_x_{ci}_{di}", float("nan")) for di in range(d)]
                    out.append(
                        f"| {ci} | "
                        + " | ".join(f"{c:+.3f}" for c in coords)
                        + " |"
                    )
                out.append("")

    WRITEUP.write_text("\n".join(out) + "\n")
    print(f"[writeup] wrote {WRITEUP.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
