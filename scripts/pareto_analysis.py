#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pareto-frontier analysis across all v5 ablation results.

Aggregates the per-cell metrics from every grid we've run and
identifies the Pareto frontier on the two-axis trade-off that
matters for production:

  * x-axis: in-distribution bLOO RMSE  (lower = better)
  * y-axis: held-out Class-2 LOCO RMSE  (lower = better, the most
            important held-out-class test for v5)

A variant is Pareto-optimal if no other variant beats it on both
axes simultaneously. The frontier is the set of Pareto-optimal
variants.

Also produces:

  * weighted LOCO summary (n-weighted across the 3 classes)
  * variant-by-criterion winners table
  * data for a writeup section on "Pareto-optimal architectures"

Input CSVs (uses whatever exists):
  experiments/joint_distance_family_ablation.csv (132-cell, primary)
  experiments/joint_hamming_matern_ablation.csv (48-cell, confirmatory)
  experiments/rbf_embedding_loco.csv (54-cell, LOCO baselines)
  experiments/rbf_embedding_grid.csv (30-cell, in-distribution RBF)
  experiments/three_class_ablation_results.csv (legacy 144-cell)

Output:
  experiments/PARETO_ANALYSIS.md
  experiments/pareto_analysis.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

WRITEUP = REPO_ROOT / "experiments" / "PARETO_ANALYSIS.md"
RESULTS_CSV = REPO_ROOT / "experiments" / "pareto_analysis.csv"

CSVS = [
    REPO_ROOT / "experiments" / "joint_distance_family_ablation.csv",
    REPO_ROOT / "experiments" / "joint_hamming_matern_ablation.csv",
]


def aggregate_per_variant(df: pd.DataFrame) -> pd.DataFrame:
    """For each (source_kernel, phase, holdout_class), compute the
    mean across seeds. Combined into a wide dataframe with one row
    per source_kernel."""
    rows = []
    for vid, vdf in df.groupby("source_kernel"):
        row = {"source_kernel": vid}
        in_dist = vdf[vdf["phase"] == "in_distribution"]
        if len(in_dist) > 0:
            for c in ("loo_rmse", "bloo_rmse", "bloo_pit_ks",
                      "bloo_coverage_95", "bloo_set12_rmse",
                      "bloo_set12_pit_ks"):
                if c in in_dist.columns:
                    row[c] = float(in_dist[c].mean())
        loco = vdf[vdf["phase"] == "loco"]
        for hc in (0, 1, 2):
            sub = loco[loco["holdout_class"] == hc]
            if len(sub) > 0:
                row[f"loco_{hc}_rmse"] = float(sub["holdout_rmse"].mean())
                row[f"loco_{hc}_pit_ks"] = float(sub["holdout_pit_ks"].mean())
                row[f"loco_{hc}_n_test"] = int(sub["n_test"].iloc[0])
        # Weighted LOCO across classes (by n_test).
        loco_means = {}
        n_tests = {}
        for hc in (0, 1, 2):
            if f"loco_{hc}_rmse" in row and f"loco_{hc}_n_test" in row:
                loco_means[hc] = row[f"loco_{hc}_rmse"]
                n_tests[hc] = row[f"loco_{hc}_n_test"]
        if loco_means:
            total_n = sum(n_tests.values())
            # Weighted MEAN-RMSE (not the standard composite MSE, but
            # a useful summary statistic).
            row["weighted_loco_rmse"] = (
                sum(loco_means[hc] * n_tests[hc] for hc in loco_means)
                / total_n
            )
            # Weighted MSE-based RMSE: combine MSEs not RMSEs.
            row["weighted_loco_rmse_mse"] = (
                sum(loco_means[hc] ** 2 * n_tests[hc] for hc in loco_means)
                / total_n
            ) ** 0.5
        rows.append(row)
    return pd.DataFrame(rows).set_index("source_kernel").sort_index()


def find_pareto(
    df: pd.DataFrame,
    minimise: list[str],
) -> pd.DataFrame:
    """Return rows that are not strictly dominated on any of the
    ``minimise`` columns.

    A row r is dominated by row s if s[c] <= r[c] for all c, with
    strict inequality on at least one.
    """
    available = [c for c in minimise if c in df.columns]
    sub = df[available].dropna()
    pareto_idx: list = []
    for i, ri in sub.iterrows():
        dominated = False
        for j, rj in sub.iterrows():
            if i == j:
                continue
            if all(rj[c] <= ri[c] for c in available) and any(
                rj[c] < ri[c] for c in available
            ):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)
    return df.loc[pareto_idx][available]


def main() -> int:
    dfs = []
    for csv in CSVS:
        if csv.exists():
            d = pd.read_csv(csv)
            dfs.append(d)
            print(f"  loaded {csv.relative_to(REPO_ROOT)}: {len(d)} rows")
    if not dfs:
        print("ERROR: no CSVs found.")
        return 1
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nMerged: {len(df)} rows from {len(dfs)} files")

    summary = aggregate_per_variant(df)
    summary.to_csv(RESULTS_CSV)
    print(f"\n[csv] wrote {RESULTS_CSV.relative_to(REPO_ROOT)}")

    out: list[str] = []
    out.append("# Pareto-frontier analysis across all v5 ablation variants")
    out.append("")
    out.append(
        "Aggregates per-cell metrics from all ablation grids we've "
        "run and identifies the Pareto-optimal frontier on the "
        "production-relevant trade-off:"
    )
    out.append("")
    out.append(
        "* **in-distribution bLOO RMSE** (the in-distribution accuracy "
        "metric for BO over existing 3-class compositions)"
    )
    out.append(
        "* **held-out Class-2 (Set-3) LOCO RMSE** (the blind-class "
        "extrapolation metric)"
    )
    out.append("")

    out.append("## Per-variant summary (3-seed means)")
    out.append("")
    out.append(
        "| variant | LOO | bLOO | bLOO PIT-KS | S12-bLOO | "
        "LOCO-0 | LOCO-1 | LOCO-2 | weighted LOCO |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|")

    def fmt(v):
        if pd.isna(v):
            return "--"
        if abs(v) < 50:
            return f"{v:.3f}"
        return f"{v:.0f}"

    for vid in summary.index:
        r = summary.loc[vid]
        out.append(
            f"| `{vid}` | "
            f"{fmt(r.get('loo_rmse'))} | "
            f"{fmt(r.get('bloo_rmse'))} | "
            f"{fmt(r.get('bloo_pit_ks'))} | "
            f"{fmt(r.get('bloo_set12_rmse'))} | "
            f"{fmt(r.get('loco_0_rmse'))} | "
            f"{fmt(r.get('loco_1_rmse'))} | "
            f"{fmt(r.get('loco_2_rmse'))} | "
            f"{fmt(r.get('weighted_loco_rmse'))} |"
        )
    out.append("")

    # Pareto frontier on (bloo_rmse, loco_2_rmse).
    pareto_bloo_loco2 = find_pareto(
        summary, minimise=["bloo_rmse", "loco_2_rmse"]
    )
    out.append("## Pareto frontier on (bLOO RMSE, Class-2 LOCO RMSE)")
    out.append("")
    out.append(
        "Variants on the frontier; no other variant beats them on both "
        "the in-distribution bLOO and the blind Class-2 LOCO metrics."
    )
    out.append("")
    out.append("| rank | variant | bLOO RMSE | LOCO Class-2 RMSE |")
    out.append("|---|---|---|---|")
    sorted_p = pareto_bloo_loco2.sort_values("bloo_rmse")
    for i, (vid, r) in enumerate(sorted_p.iterrows(), start=1):
        out.append(
            f"| {i} | `{vid}` | {r['bloo_rmse']:.0f} | "
            f"{r['loco_2_rmse']:.0f} |"
        )
    out.append("")

    # Pareto frontier on (LOO, weighted_LOCO).
    pareto_loo_loco = find_pareto(
        summary, minimise=["loo_rmse", "weighted_loco_rmse"]
    )
    out.append("## Pareto frontier on (LOO RMSE, weighted LOCO RMSE)")
    out.append("")
    out.append(
        "Lower bound on the achievable LOO vs LOCO trade-off across "
        "all explored kernel architectures."
    )
    out.append("")
    out.append("| rank | variant | LOO RMSE | weighted LOCO RMSE |")
    out.append("|---|---|---|---|")
    sorted_p = pareto_loo_loco.sort_values("loo_rmse")
    for i, (vid, r) in enumerate(sorted_p.iterrows(), start=1):
        out.append(
            f"| {i} | `{vid}` | {r['loo_rmse']:.0f} | "
            f"{r['weighted_loco_rmse']:.0f} |"
        )
    out.append("")

    # Best-by-criterion table.
    out.append("## Best variant by criterion")
    out.append("")
    out.append("| criterion | best variant | value |")
    out.append("|---|---|---|")
    criteria = [
        ("LOO RMSE", "loo_rmse", min),
        ("bLOO RMSE", "bloo_rmse", min),
        ("bLOO PIT-KS", "bloo_pit_ks", min),
        ("S12-bLOO RMSE", "bloo_set12_rmse", min),
        ("LOCO Class-0 RMSE", "loco_0_rmse", min),
        ("LOCO Class-1 RMSE", "loco_1_rmse", min),
        ("LOCO Class-2 RMSE", "loco_2_rmse", min),
        ("weighted LOCO RMSE", "weighted_loco_rmse", min),
    ]
    for name, col, op in criteria:
        if col not in summary.columns:
            continue
        s = summary[col].dropna()
        if len(s) == 0:
            continue
        if op is min:
            idx = s.idxmin()
        else:
            idx = s.idxmax()
        out.append(f"| {name} | `{idx}` | {fmt(s.loc[idx])} |")
    out.append("")

    WRITEUP.write_text("\n".join(out) + "\n")
    print(f"[writeup] wrote {WRITEUP.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
