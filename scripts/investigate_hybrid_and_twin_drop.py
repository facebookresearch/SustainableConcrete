#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pareto-hybrid + twin-drop ablation for the new joint-distance winner.

Part A: additive hybrid kernels combining joint_hamming_matern with
        rbf_embedding_d2 (the two Pareto corners).
Part B: twin-drop study for joint_hamming_matern (drop M81).

Cells: 2 hybrids x 4 contexts x 3 seeds + 1 variant x drop-M81 x 4 x 3
     = 24 + 12 = 36 cells.

Output:
    experiments/HYBRID_AND_TWIN_DROP_ABLATION.md
    experiments/hybrid_and_twin_drop_ablation.csv
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.three_class_ablation import (  # noqa: E402
    _eval_metrics,
    _load_data,
    _make_class_holdout,
    _stage_factory,
)
from experiments.model_variant_study import held_out_metrics  # noqa: E402

WRITEUP = REPO_ROOT / "experiments" / "HYBRID_AND_TWIN_DROP_ABLATION.md"
RESULTS_CSV = (
    REPO_ROOT / "experiments" / "hybrid_and_twin_drop_ablation.csv"
)

# Part A: additive hybrids (compared against the two Pareto corners).
HYBRID_VARIANTS = [
    "additive_joint_hamming_rbf_d2",
    "additive_joint_hamming_nu25_rbf_d2",
]
SEEDS = [0, 1, 2]


def _drop_m81_csv() -> Path:
    """Create a temporary v5 CSV with the M61 (high-strength twin)
    dropped, then return its path. The caller is responsible for
    cleaning it up."""
    src = REPO_ROOT / "data" / "boxcrete_data.csv"
    df = pd.read_csv(src)
    n_before = len(df)
    df = df[df["Mix Name"] != "M61"].copy()
    print(f"  Dropping M61: {n_before} -> {len(df)} rows")
    out = REPO_ROOT / "experiments" / "_v5_drop_m61_tmp.csv"
    df.to_csv(out, index=False)
    return out


def main() -> int:
    rows: list[dict] = []

    # ------------------------------------------------------------
    # Part A: additive hybrids on full v5
    # ------------------------------------------------------------
    print("\n===== Part A: Additive hybrids =====", flush=True)
    print("\nIn-distribution v5", flush=True)
    ds, X, Y, Yvar, bounds, n_real = _load_data("v5")
    for vid in HYBRID_VARIANTS:
        factory = _stage_factory(
            source_kernel=vid,
            include_lognormal_baseline=False,
            time_tying_sigma=None,
        )
        for seed in SEEDS:
            t0 = time.time()
            row = {
                "phase": "in_distribution",
                "data": "v5",
                "source_kernel": vid,
                "seed": seed,
                "n_train": n_real,
            }
            try:
                model = factory(X, Y, Yvar, bounds, seed)
                m = _eval_metrics(model, n_real)
                row.update(m)
            except Exception as exc:
                row["fit_error"] = str(exc)[:200]
            row["wall_sec"] = time.time() - t0
            rows.append(row)
            loo = row.get("loo_rmse", float("nan"))
            bloo = row.get("bloo_rmse", float("nan"))
            print(
                f"  {vid:<42s} seed={seed} loo={loo:.0f} bloo={bloo:.0f} "
                f"({row['wall_sec']:.0f}s)",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    for hc in [0, 1, 2]:
        print(f"\nLOCO class {hc}", flush=True)
        X_tr, Y_tr, Yvar_tr, X_te, Y_te_psi, b = _make_class_holdout(hc)
        n_test = int(X_te.shape[0])
        n_train = int(X_tr.shape[0])
        for vid in HYBRID_VARIANTS:
            factory = _stage_factory(
                source_kernel=vid,
                include_lognormal_baseline=False,
                time_tying_sigma=None,
            )
            for seed in SEEDS:
                t0 = time.time()
                row = {
                    "phase": "loco",
                    "data": "v5",
                    "holdout_class": hc,
                    "source_kernel": vid,
                    "seed": seed,
                    "n_train": n_train,
                    "n_test": n_test,
                }
                try:
                    model = factory(X_tr, Y_tr, Yvar_tr, b, seed)
                    ho = held_out_metrics(model, X_te, Y_te_psi)
                    for k, v in ho.items():
                        row[f"holdout_{k}"] = v
                except Exception as exc:
                    row["fit_error"] = str(exc)[:200]
                row["wall_sec"] = time.time() - t0
                rows.append(row)
                rmse = row.get("holdout_rmse", float("nan"))
                print(
                    f"  {vid:<42s} class={hc} seed={seed} rmse={rmse:.0f} "
                    f"({row['wall_sec']:.0f}s)",
                    flush=True,
                )
                pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    # ------------------------------------------------------------
    # Part B: twin-drop for joint_hamming_matern
    # ------------------------------------------------------------
    print("\n===== Part B: M61 (high-twin) drop for joint_hamming_matern =====",
          flush=True)
    drop_csv = _drop_m81_csv()
    src_csv = REPO_ROOT / "data" / "boxcrete_data.csv"
    backup = REPO_ROOT / "experiments" / "_v5_full_backup_tmp.csv"
    shutil.copy(src_csv, backup)
    try:
        # Swap in the drop-M61 dataset.
        shutil.copy(drop_csv, src_csv)

        # In-distribution + LOCO for joint_hamming_matern on drop-M61.
        ds, X, Y, Yvar, bounds, n_real = _load_data("v5")
        print(f"\nIn-distribution v5 (drop M61): n_train = {n_real}",
              flush=True)
        for seed in SEEDS:
            factory = _stage_factory(
                source_kernel="joint_hamming_matern",
                include_lognormal_baseline=False,
                time_tying_sigma=None,
            )
            t0 = time.time()
            row = {
                "phase": "in_distribution",
                "data": "v5_drop_m61",
                "source_kernel": "joint_hamming_matern_drop_m61",
                "seed": seed,
                "n_train": n_real,
            }
            try:
                model = factory(X, Y, Yvar, bounds, seed)
                m = _eval_metrics(model, n_real)
                row.update(m)
            except Exception as exc:
                row["fit_error"] = str(exc)[:200]
            row["wall_sec"] = time.time() - t0
            rows.append(row)
            loo = row.get("loo_rmse", float("nan"))
            bloo = row.get("bloo_rmse", float("nan"))
            print(
                f"  drop_m61 joint_hamming_matern seed={seed} "
                f"loo={loo:.0f} bloo={bloo:.0f} ({row['wall_sec']:.0f}s)",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

        for hc in [0, 1, 2]:
            print(f"\nLOCO class {hc} (drop M61)", flush=True)
            X_tr, Y_tr, Yvar_tr, X_te, Y_te_psi, b = _make_class_holdout(hc)
            n_test = int(X_te.shape[0])
            n_train = int(X_tr.shape[0])
            for seed in SEEDS:
                factory = _stage_factory(
                    source_kernel="joint_hamming_matern",
                    include_lognormal_baseline=False,
                    time_tying_sigma=None,
                )
                t0 = time.time()
                row = {
                    "phase": "loco",
                    "data": "v5_drop_m61",
                    "holdout_class": hc,
                    "source_kernel": "joint_hamming_matern_drop_m61",
                    "seed": seed,
                    "n_train": n_train,
                    "n_test": n_test,
                }
                try:
                    model = factory(X_tr, Y_tr, Yvar_tr, b, seed)
                    ho = held_out_metrics(model, X_te, Y_te_psi)
                    for k, v in ho.items():
                        row[f"holdout_{k}"] = v
                except Exception as exc:
                    row["fit_error"] = str(exc)[:200]
                row["wall_sec"] = time.time() - t0
                rows.append(row)
                rmse = row.get("holdout_rmse", float("nan"))
                print(
                    f"  drop_m61 joint_hamming_matern class={hc} seed={seed} "
                    f"rmse={rmse:.0f} ({row['wall_sec']:.0f}s)",
                    flush=True,
                )
                pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    finally:
        # Always restore the original CSV.
        shutil.copy(backup, src_csv)
        if drop_csv.exists():
            drop_csv.unlink()
        if backup.exists():
            backup.unlink()

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[csv] wrote {RESULTS_CSV.relative_to(REPO_ROOT)}")

    # ------------------------------------------------------------
    # Markdown writeup
    # ------------------------------------------------------------
    out: list[str] = []
    out.append("# Pareto-hybrid + twin-drop ablation")
    out.append("")

    in_dist = df[df["phase"] == "in_distribution"]
    out.append("## In-distribution v5 (mean ± std across 3 seeds)")
    out.append("")
    out.append(
        "| variant | LOO RMSE | bLOO RMSE | bLOO PIT-KS | bLOO cov95 |"
    )
    out.append("|---|---|---|---|---|")
    agg = in_dist.groupby("source_kernel").agg(
        loo_mean=("loo_rmse", "mean"),
        loo_std=("loo_rmse", "std"),
        bloo_mean=("bloo_rmse", "mean"),
        bloo_std=("bloo_rmse", "std"),
        pit_mean=("bloo_pit_ks", "mean"),
        pit_std=("bloo_pit_ks", "std"),
        cov_mean=("bloo_coverage_95", "mean"),
        cov_std=("bloo_coverage_95", "std"),
    )
    for vid in agg.index:
        r = agg.loc[vid]
        out.append(
            f"| `{vid}` | "
            f"{r['loo_mean']:.0f} ± {r['loo_std']:.1f} | "
            f"{r['bloo_mean']:.0f} ± {r['bloo_std']:.1f} | "
            f"{r['pit_mean']:.3f} ± {r['pit_std']:.4f} | "
            f"{r['cov_mean']:.3f} ± {r['cov_std']:.4f} |"
        )
    out.append("")

    loco_df = df[df["phase"] == "loco"]
    for hc in [0, 1, 2]:
        sub = loco_df[loco_df["holdout_class"] == hc]
        if len(sub) == 0:
            continue
        out.append(f"## LOCO class {hc} (mean ± std across 3 seeds)")
        out.append("")
        n_test = int(sub.iloc[0]["n_test"])
        out.append(f"$n_{{test}} = {n_test}$")
        out.append("")
        out.append("| variant | RMSE | PIT-KS | cov95 |")
        out.append("|---|---|---|---|")
        agg = sub.groupby("source_kernel").agg(
            rmse_mean=("holdout_rmse", "mean"),
            rmse_std=("holdout_rmse", "std"),
            pit_mean=("holdout_pit_ks", "mean"),
            pit_std=("holdout_pit_ks", "std"),
            cov_mean=("holdout_coverage_95", "mean"),
            cov_std=("holdout_coverage_95", "std"),
        )
        for vid in agg.index:
            r = agg.loc[vid]
            out.append(
                f"| `{vid}` | "
                f"{r['rmse_mean']:.0f} ± {r['rmse_std']:.1f} | "
                f"{r['pit_mean']:.3f} ± {r['pit_std']:.4f} | "
                f"{r['cov_mean']:.3f} ± {r['cov_std']:.4f} |"
            )
        out.append("")

    WRITEUP.write_text("\n".join(out) + "\n")
    print(f"[writeup] wrote {WRITEUP.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
