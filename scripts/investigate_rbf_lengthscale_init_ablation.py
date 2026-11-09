#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Ablation: lengthscale-fixed and linear-init variants of the
RBFEmbeddingKernel.

User proposal:

  1. Remove the lengthscale: since the embeddings are free parameters
     of the same kernel, the scale degree-of-freedom of x_c is
     redundant with the lengthscale ell. Fixing ell = 1 removes a
     parameter and the loss-surface ridge along that direction.

  2. Linear init for d=1: initialise embeddings at integer class
     labels (0, 1, 2) instead of the equilateral-simplex projection
     (-1, 0, +1). Mimics legacy_continuous_ard's effective init and
     tests whether the +33 psi architecture cost from
     simplex-projected d=1 is partly an optimizer-trapped-in-bad-basin
     effect.

Variants tested:

    rbf_embedding_d1                            (baseline)
    rbf_embedding_d1_fixed_ell                  (fix ell only)
    rbf_embedding_d1_linear_init                (linear init only)
    rbf_embedding_d1_linear_init_fixed_ell      (both)
    rbf_embedding_d2                            (baseline)
    rbf_embedding_d2_fixed_ell                  (fix ell)
    rbf_embedding_d3                            (baseline)
    rbf_embedding_d3_fixed_ell                  (fix ell)
    legacy_continuous_ard                       (target to match on bLOO)
    hamming                                     (current production)

Per cell: in-distribution metrics on v5 (LOO, bLOO, S12-bLOO,
PIT-KS, coverage_95) and held-out-class extrapolation (LOCO RMSE +
PIT-KS for each of the 3 classes). All variants are deterministic
(no random init) so 1 seed each.

Output:
    experiments/RBF_EMBEDDING_LENGTHSCALE_INIT_ABLATION.md
    experiments/rbf_embedding_lengthscale_init.csv
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
    / "RBF_EMBEDDING_LENGTHSCALE_INIT_ABLATION.md"
)
RESULTS_CSV = (
    REPO_ROOT / "experiments" / "rbf_embedding_lengthscale_init.csv"
)

VARIANTS = [
    "rbf_embedding_d1",
    "rbf_embedding_d1_fixed_ell",
    "rbf_embedding_d1_linear_init",
    "rbf_embedding_d1_linear_init_fixed_ell",
    "rbf_embedding_d2",
    "rbf_embedding_d2_fixed_ell",
    "rbf_embedding_d3",
    "rbf_embedding_d3_fixed_ell",
    "legacy_continuous_ard",  # bLOO target
    "hamming",                # current production
]


def main() -> int:
    rows: list[dict] = []

    # -----------------------------
    # In-distribution v5 fits
    # -----------------------------
    print("\n===== In-distribution v5 =====", flush=True)
    ds, X, Y, Yvar, bounds, n_real = _load_data("v5")
    for vid in VARIANTS:
        factory = _stage_factory(
            source_kernel=vid,
            include_lognormal_baseline=False,
            time_tying_sigma=None,
        )
        seed = 0
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
            f"  {vid:<45s} loo={loo:.0f} bloo={bloo:.0f} "
            f"pit_ks={pit:.3f} ({row['wall_sec']:.0f}s)",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    # -----------------------------
    # LOCO held-out-class fits
    # -----------------------------
    for holdout_class in [0, 1, 2]:
        print(f"\n===== LOCO holdout class {holdout_class} =====", flush=True)
        X_tr, Y_tr, Yvar_tr, X_te, Y_te_psi, b = _make_class_holdout(
            holdout_class
        )
        n_test = int(X_te.shape[0])
        n_train = int(X_tr.shape[0])
        for vid in VARIANTS:
            factory = _stage_factory(
                source_kernel=vid,
                include_lognormal_baseline=False,
                time_tying_sigma=None,
            )
            seed = 0
            t0 = time.time()
            row = {
                "phase": "loco",
                "data": "v5",
                "holdout_class": holdout_class,
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
            ho_rmse = row.get("holdout_rmse", float("nan"))
            ho_pit = row.get("holdout_pit_ks", float("nan"))
            print(
                f"  {vid:<45s} class={holdout_class} "
                f"rmse={ho_rmse:.0f} pit_ks={ho_pit:.3f} "
                f"({row['wall_sec']:.0f}s)",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[csv] wrote {RESULTS_CSV.relative_to(REPO_ROOT)}")

    # -----------------------------
    # Markdown writeup
    # -----------------------------
    out: list[str] = []
    out.append("# RBF-embedding kernel: lengthscale & init ablation")
    out.append("")
    out.append(
        "Tests two follow-up proposals: (1) fix the kernel lengthscale "
        "at 1 (since the embedding scale is a redundant DOF); "
        "(2) initialise the d=1 embedding at integer class labels "
        "(0, 1, 2) instead of the equilateral-simplex projection "
        "(-1, 0, +1)."
    )
    out.append("")

    in_dist = df[df["phase"] == "in_distribution"]
    out.append("## In-distribution v5 metrics")
    out.append("")
    out.append(
        "| variant | LOO RMSE | bLOO RMSE | bLOO PIT-KS | bLOO cov95 | "
        "S12-bLOO RMSE |"
    )
    out.append("|---|---|---|---|---|---|")
    for vid in VARIANTS:
        sub = in_dist[in_dist["source_kernel"] == vid]
        if sub.empty:
            continue
        r = sub.iloc[0]
        loo = r.get("loo_rmse", float("nan"))
        bloo = r.get("bloo_rmse", float("nan"))
        pit = r.get("bloo_pit_ks", float("nan"))
        cov = r.get("bloo_coverage_95", float("nan"))
        s12 = r.get("bloo_set12_rmse", float("nan"))
        out.append(
            f"| `{vid}` | {loo:.0f} | {bloo:.0f} | "
            f"{pit:.3f} | {cov:.3f} | {s12:.0f} |"
        )
    out.append("")

    loco_df = df[df["phase"] == "loco"]
    for hc in [0, 1, 2]:
        sub = loco_df[loco_df["holdout_class"] == hc]
        if sub.empty:
            continue
        out.append(f"## LOCO held-out class {hc}")
        out.append("")
        n_test = int(sub.iloc[0]["n_test"])
        out.append(f"$n_{{test}} = {n_test}$")
        out.append("")
        out.append(
            "| variant | RMSE (psi) | MAE (psi) | PIT-KS | cov95 | CRPS (psi) |"
        )
        out.append("|---|---|---|---|---|---|")
        for vid in VARIANTS:
            ssub = sub[sub["source_kernel"] == vid]
            if ssub.empty:
                continue
            r = ssub.iloc[0]
            rmse = r.get("holdout_rmse", float("nan"))
            mae = r.get("holdout_mae", float("nan"))
            pit = r.get("holdout_pit_ks", float("nan"))
            cov = r.get("holdout_coverage_95", float("nan"))
            crps = r.get("holdout_crps", float("nan"))
            out.append(
                f"| `{vid}` | {rmse:.0f} | {mae:.0f} | "
                f"{pit:.3f} | {cov:.3f} | {crps:.0f} |"
            )
        out.append("")

    WRITEUP.write_text("\n".join(out) + "\n")
    print(f"[writeup] wrote {WRITEUP.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
