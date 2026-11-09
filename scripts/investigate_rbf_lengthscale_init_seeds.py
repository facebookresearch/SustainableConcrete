#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-seed verification for the new RBF-embedding variants.

The original lengthscale-and-init ablation
(``scripts/investigate_rbf_lengthscale_init_ablation.py``) ran each
variant at seed=0 only. This script reruns the new variants at
seeds=1, 2 to verify determinism (or detect non-determinism) and
solidify the production recommendation.

Variants verified:
    rbf_embedding_d1_fixed_ell
    rbf_embedding_d1_linear_init
    rbf_embedding_d1_linear_init_fixed_ell  (the catastrophic Class-2 LOCO)
    rbf_embedding_d2_fixed_ell
    rbf_embedding_d3_fixed_ell

Per cell: in-distribution v5 fit + 3 LOCO holdouts. 5 variants × 2
seeds × 4 contexts = 40 cells. Each fit ~90s -> ~60 min wall.

Output:
    experiments/rbf_embedding_lengthscale_init_seeds.csv
    experiments/RBF_EMBEDDING_LENGTHSCALE_INIT_SEEDS.md
"""

from __future__ import annotations

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

WRITEUP = (
    REPO_ROOT
    / "experiments"
    / "RBF_EMBEDDING_LENGTHSCALE_INIT_SEEDS.md"
)
RESULTS_CSV = (
    REPO_ROOT / "experiments" / "rbf_embedding_lengthscale_init_seeds.csv"
)

VARIANTS = [
    "rbf_embedding_d1_fixed_ell",
    "rbf_embedding_d1_linear_init",
    "rbf_embedding_d1_linear_init_fixed_ell",
    "rbf_embedding_d2_fixed_ell",
    "rbf_embedding_d3_fixed_ell",
]
EXTRA_SEEDS = [1, 2]


def main() -> int:
    rows: list[dict] = []

    print("\n===== In-distribution v5 (extra seeds) =====", flush=True)
    ds, X, Y, Yvar, bounds, n_real = _load_data("v5")
    for vid in VARIANTS:
        for seed in EXTRA_SEEDS:
            factory = _stage_factory(
                source_kernel=vid,
                include_lognormal_baseline=False,
                time_tying_sigma=None,
            )
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
                f"  {vid:<45s} seed={seed} loo={loo:.0f} bloo={bloo:.0f} "
                f"({row['wall_sec']:.0f}s)",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    for hc in [0, 1, 2]:
        print(f"\n===== LOCO class {hc} (extra seeds) =====", flush=True)
        X_tr, Y_tr, Yvar_tr, X_te, Y_te_psi, b = _make_class_holdout(hc)
        n_test = int(X_te.shape[0])
        n_train = int(X_tr.shape[0])
        for vid in VARIANTS:
            for seed in EXTRA_SEEDS:
                factory = _stage_factory(
                    source_kernel=vid,
                    include_lognormal_baseline=False,
                    time_tying_sigma=None,
                )
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
                    f"  {vid:<45s} class={hc} seed={seed} rmse={rmse:.0f} "
                    f"({row['wall_sec']:.0f}s)",
                    flush=True,
                )
                pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[csv] wrote {RESULTS_CSV.relative_to(REPO_ROOT)}")

    # Combine with seed=0 from the previous CSV so we can produce
    # mean ± std summary tables.
    seed0_csv = REPO_ROOT / "experiments" / "rbf_embedding_lengthscale_init.csv"
    if seed0_csv.exists():
        df0 = pd.read_csv(seed0_csv)
        df0_filtered = df0[df0["source_kernel"].isin(VARIANTS)]
        combined = pd.concat([df0_filtered, df], ignore_index=True)
    else:
        combined = df
    combined_csv = (
        REPO_ROOT
        / "experiments"
        / "rbf_embedding_lengthscale_init_seeds_combined.csv"
    )
    combined.to_csv(combined_csv, index=False)

    out: list[str] = []
    out.append("# RBF-embedding kernel: lengthscale & init — multi-seed verification")
    out.append("")
    out.append(
        "Reruns the new variants at seeds 1 and 2 (seed 0 from the "
        "original ablation is folded in) to verify determinism (or "
        "detect non-determinism) for each (variant, phase) combination."
    )
    out.append("")

    in_dist = combined[combined["phase"] == "in_distribution"]
    if len(in_dist) > 0:
        out.append("## In-distribution v5 (mean ± std across seeds 0, 1, 2)")
        out.append("")
        out.append(
            "| variant | n_seeds | LOO RMSE | bLOO RMSE | bLOO PIT-KS | bLOO cov95 |"
        )
        out.append("|---|---|---|---|---|---|")
        agg = in_dist.groupby("source_kernel").agg(
            n=("seed", "nunique"),
            loo_mean=("loo_rmse", "mean"),
            loo_std=("loo_rmse", "std"),
            bloo_mean=("bloo_rmse", "mean"),
            bloo_std=("bloo_rmse", "std"),
            pit_mean=("bloo_pit_ks", "mean"),
            pit_std=("bloo_pit_ks", "std"),
            cov_mean=("bloo_coverage_95", "mean"),
            cov_std=("bloo_coverage_95", "std"),
        )
        for vid in VARIANTS:
            if vid not in agg.index:
                continue
            r = agg.loc[vid]
            out.append(
                f"| `{vid}` | {int(r['n'])} | "
                f"{r['loo_mean']:.0f} ± {r['loo_std']:.1f} | "
                f"{r['bloo_mean']:.0f} ± {r['bloo_std']:.1f} | "
                f"{r['pit_mean']:.3f} ± {r['pit_std']:.4f} | "
                f"{r['cov_mean']:.3f} ± {r['cov_std']:.4f} |"
            )
        out.append("")

    loco_df = combined[combined["phase"] == "loco"]
    for hc in [0, 1, 2]:
        sub = loco_df[loco_df["holdout_class"] == hc]
        if len(sub) == 0:
            continue
        out.append(
            f"## LOCO class {hc} held out (mean ± std across seeds 0, 1, 2)"
        )
        out.append("")
        n_test = int(sub.iloc[0]["n_test"])
        out.append(f"$n_{{test}} = {n_test}$")
        out.append("")
        out.append(
            "| variant | n_seeds | RMSE | MAE | PIT-KS | cov95 |"
        )
        out.append("|---|---|---|---|---|---|")
        agg = sub.groupby("source_kernel").agg(
            n=("seed", "nunique"),
            rmse_mean=("holdout_rmse", "mean"),
            rmse_std=("holdout_rmse", "std"),
            mae_mean=("holdout_mae", "mean"),
            mae_std=("holdout_mae", "std"),
            pit_mean=("holdout_pit_ks", "mean"),
            pit_std=("holdout_pit_ks", "std"),
            cov_mean=("holdout_coverage_95", "mean"),
            cov_std=("holdout_coverage_95", "std"),
        )
        for vid in VARIANTS:
            if vid not in agg.index:
                continue
            r = agg.loc[vid]
            out.append(
                f"| `{vid}` | {int(r['n'])} | "
                f"{r['rmse_mean']:.0f} ± {r['rmse_std']:.1f} | "
                f"{r['mae_mean']:.0f} ± {r['mae_std']:.1f} | "
                f"{r['pit_mean']:.3f} ± {r['pit_std']:.4f} | "
                f"{r['cov_mean']:.3f} ± {r['cov_std']:.4f} |"
            )
        out.append("")

    WRITEUP.write_text("\n".join(out) + "\n")
    print(f"[writeup] wrote {WRITEUP.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
