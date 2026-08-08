#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Ablation: ``joint_hamming_matern`` (joint feature + categorical
Matern) vs the existing source kernels.

Tests whether replacing the factorised ``ScaleKernel(K_cat × K_features)``
topology with a single ``Matern(sqrt(d^2_feat + alpha * 1[c_i != c_j]))``
closes the +33 psi bLOO architecture gap to ``legacy_continuous_ard``.

If the joint topology is what was costing us, we expect:

  - In-distribution v5 bLOO RMSE for ``joint_hamming_matern`` to land
    near ``legacy_continuous_ard``'s 702 psi (rather than the ~735-738
    psi of the categorical-product variants).
  - LOCO RMSE on Class 2 to remain similar to Hamming (since the
    Hamming-style cross-class penalty is preserved) without the
    catastrophic IndexKernel-style failures.

Grid:
    in-distribution v5: joint_hamming_matern × 3 seeds (alpha learnable)
    LOCO: joint_hamming_matern × 3 holdout classes × 3 seeds

Plus reference baselines (3 seeds each, in-dist + LOCO) for direct
comparison: hamming, legacy_continuous_ard, rbf_embedding_d2.

Output:
    experiments/JOINT_HAMMING_MATERN_ABLATION.md
    experiments/joint_hamming_matern_ablation.csv
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
    REPO_ROOT / "experiments" / "JOINT_HAMMING_MATERN_ABLATION.md"
)
RESULTS_CSV = (
    REPO_ROOT / "experiments" / "joint_hamming_matern_ablation.csv"
)

VARIANTS = [
    "joint_hamming_matern",
    "hamming",
    "legacy_continuous_ard",
    "rbf_embedding_d2",
]
SEEDS = [0, 1, 2]


def main() -> int:
    rows: list[dict] = []

    print("\n===== In-distribution v5 =====", flush=True)
    ds, X, Y, Yvar, bounds, n_real = _load_data("v5")
    for vid in VARIANTS:
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
            pit = row.get("bloo_pit_ks", float("nan"))
            print(
                f"  {vid:<28s} seed={seed} loo={loo:.0f} bloo={bloo:.0f} "
                f"pit_ks={pit:.3f} ({row['wall_sec']:.0f}s)",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    for hc in [0, 1, 2]:
        print(f"\n===== LOCO class {hc} =====", flush=True)
        X_tr, Y_tr, Yvar_tr, X_te, Y_te_psi, b = _make_class_holdout(hc)
        n_test = int(X_te.shape[0])
        n_train = int(X_tr.shape[0])
        for vid in VARIANTS:
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
                pit = row.get("holdout_pit_ks", float("nan"))
                print(
                    f"  {vid:<28s} class={hc} seed={seed} rmse={rmse:.0f} "
                    f"pit_ks={pit:.3f} ({row['wall_sec']:.0f}s)",
                    flush=True,
                )
                pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[csv] wrote {RESULTS_CSV.relative_to(REPO_ROOT)}")

    out: list[str] = []
    out.append("# `joint_hamming_matern` kernel: ablation grid")
    out.append("")
    out.append(
        "Tests whether the joint feature + Hamming-categorical Matern "
        "kernel (a single Matern over a joint distance) closes the "
        "+33 psi bLOO architecture gap to ``legacy_continuous_ard`` "
        "while keeping proper categorical handling."
    )
    out.append("")

    in_dist = df[df["phase"] == "in_distribution"]
    if len(in_dist) > 0:
        out.append("## In-distribution v5 (mean ± std across 3 seeds)")
        out.append("")
        out.append(
            "| variant | LOO RMSE | bLOO RMSE | bLOO PIT-KS | "
            "bLOO cov95 | S12-bLOO RMSE |"
        )
        out.append("|---|---|---|---|---|---|")
        agg = in_dist.groupby("source_kernel").agg(
            loo_mean=("loo_rmse", "mean"),
            loo_std=("loo_rmse", "std"),
            bloo_mean=("bloo_rmse", "mean"),
            bloo_std=("bloo_rmse", "std"),
            pit_mean=("bloo_pit_ks", "mean"),
            pit_std=("bloo_pit_ks", "std"),
            cov_mean=("bloo_coverage_95", "mean"),
            cov_std=("bloo_coverage_95", "std"),
            s12_mean=("bloo_set12_rmse", "mean"),
            s12_std=("bloo_set12_rmse", "std"),
        )
        for vid in VARIANTS:
            if vid not in agg.index:
                continue
            r = agg.loc[vid]
            out.append(
                f"| `{vid}` | "
                f"{r['loo_mean']:.0f} ± {r['loo_std']:.1f} | "
                f"{r['bloo_mean']:.0f} ± {r['bloo_std']:.1f} | "
                f"{r['pit_mean']:.3f} ± {r['pit_std']:.4f} | "
                f"{r['cov_mean']:.3f} ± {r['cov_std']:.4f} | "
                f"{r['s12_mean']:.0f} ± {r['s12_std']:.1f} |"
            )
        out.append("")

    loco_df = df[df["phase"] == "loco"]
    for hc in [0, 1, 2]:
        sub = loco_df[loco_df["holdout_class"] == hc]
        if len(sub) == 0:
            continue
        out.append(
            f"## LOCO class {hc} held out (mean ± std across 3 seeds)"
        )
        out.append("")
        n_test = int(sub.iloc[0]["n_test"])
        out.append(f"$n_{{test}} = {n_test}$")
        out.append("")
        out.append("| variant | RMSE | MAE | PIT-KS | cov95 | CRPS |")
        out.append("|---|---|---|---|---|---|")
        agg = sub.groupby("source_kernel").agg(
            rmse_mean=("holdout_rmse", "mean"),
            rmse_std=("holdout_rmse", "std"),
            mae_mean=("holdout_mae", "mean"),
            mae_std=("holdout_mae", "std"),
            pit_mean=("holdout_pit_ks", "mean"),
            pit_std=("holdout_pit_ks", "std"),
            cov_mean=("holdout_coverage_95", "mean"),
            cov_std=("holdout_coverage_95", "std"),
            crps_mean=("holdout_crps", "mean"),
            crps_std=("holdout_crps", "std"),
        )
        for vid in VARIANTS:
            if vid not in agg.index:
                continue
            r = agg.loc[vid]
            out.append(
                f"| `{vid}` | "
                f"{r['rmse_mean']:.0f} ± {r['rmse_std']:.1f} | "
                f"{r['mae_mean']:.0f} ± {r['mae_std']:.1f} | "
                f"{r['pit_mean']:.3f} ± {r['pit_std']:.4f} | "
                f"{r['cov_mean']:.3f} ± {r['cov_std']:.4f} | "
                f"{r['crps_mean']:.0f} ± {r['crps_std']:.1f} |"
            )
        out.append("")

    WRITEUP.write_text("\n".join(out) + "\n")
    print(f"[writeup] wrote {WRITEUP.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
