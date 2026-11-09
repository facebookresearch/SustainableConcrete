#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Leave-one-class-out (LOCO) held-out-class extrapolation for the
new RBFEmbeddingKernel. Mirrors the existing
``experiments.three_class_ablation._run_class_holdout`` infrastructure
so the metrics are directly comparable to the LOCO tables in
``experiments/THREE_CLASS_AND_PRIOR_BENCHMARK.md``.

Grid:
    source kernel: rbf_embedding_d1, rbf_embedding_d2, rbf_embedding_d3
    holdout class: 0 (Set-1 mortar), 1 (Set-2 concrete), 2 (Set-3 concrete)
    seeds: 0, 1, 2

Per cell: fit GP on the two non-held-out classes, evaluate on the
held-out class. The held-out class's embedding x_C has zero gradient
signal during training and stays at its equilateral-simplex
initialisation (deterministic, unlike IndexKernel's random init).

Output:
    experiments/RBF_EMBEDDING_LOCO.md
    experiments/rbf_embedding_loco.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.three_class_ablation import (  # noqa: E402
    _make_class_holdout,
    _stage_factory,
)
from experiments.model_variant_study import held_out_metrics  # noqa: E402

WRITEUP = REPO_ROOT / "experiments" / "RBF_EMBEDDING_LOCO.md"
RESULTS_CSV = REPO_ROOT / "experiments" / "rbf_embedding_loco.csv"


def main() -> int:
    rbf_kernels = [
        "rbf_embedding_d1",
        "rbf_embedding_d2",
        "rbf_embedding_d3",
    ]
    # Also include hamming and indexkernel_r2 for direct comparison
    # in the same script run (uses same _make_class_holdout helper, so
    # same n_test, same X_test, same Y_test psi unscaling).
    baseline_kernels = [
        "hamming",
        "indexkernel_r2",
        "legacy_continuous_ard",
    ]
    seeds = [0, 1, 2]
    holdout_classes = [0, 1, 2]

    rows: list[dict] = []
    for holdout_class in holdout_classes:
        X_train, Y_train, Yvar_train, X_test, Y_test_psi, bounds = (
            _make_class_holdout(holdout_class)
        )
        n_test = int(X_test.shape[0])
        n_train = int(X_train.shape[0])
        print(
            f"\n[holdout class={holdout_class}] n_train={n_train} n_test={n_test}",
            flush=True,
        )
        for kernel in rbf_kernels + baseline_kernels:
            factory = _stage_factory(
                source_kernel=kernel,
                include_lognormal_baseline=False,
                time_tying_sigma=None,
            )
            for seed in seeds:
                t0 = time.time()
                row = {
                    "holdout_class": holdout_class,
                    "source_kernel": kernel,
                    "seed": seed,
                    "n_train": n_train,
                    "n_test": n_test,
                }
                try:
                    model = factory(
                        X_train, Y_train, Yvar_train, bounds, seed
                    )
                    ho = held_out_metrics(model, X_test, Y_test_psi)
                    for k, v in ho.items():
                        row[f"holdout_{k}"] = v
                except Exception as exc:
                    row["fit_error"] = str(exc)[:200]
                row["wall_sec"] = time.time() - t0
                rows.append(row)
                rmse = row.get("holdout_rmse", float("nan"))
                pit = row.get("holdout_pit_ks", float("nan"))
                cov = row.get("holdout_coverage_95", float("nan"))
                print(
                    f"   class={holdout_class} {kernel} seed={seed} "
                    f"rmse={rmse:.0f} pit_ks={pit:.3f} cov95={cov:.2f} "
                    f"({row['wall_sec']:.0f}s)",
                    flush=True,
                )
                # Persist after every cell so partial failures don't
                # lose progress.
                pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[csv] wrote {RESULTS_CSV.relative_to(REPO_ROOT)}")

    # Aggregate per (holdout_class, source_kernel) cell.
    metric_cols = [
        "holdout_rmse",
        "holdout_mae",
        "holdout_pit_ks",
        "holdout_coverage_95",
        "holdout_mean_lpd",
        "holdout_crps",
    ]
    avail = [c for c in metric_cols if c in df.columns]
    agg = df.groupby(["holdout_class", "source_kernel"], dropna=False)[
        avail
    ].agg(["mean", "std"])

    out: list[str] = []
    out.append("# RBFEmbeddingKernel — leave-one-class-out (LOCO) extrapolation")
    out.append("")
    out.append(
        "Compares ``rbf_embedding_d{1,2,3}`` against ``hamming``, "
        "``indexkernel_r2``, and ``legacy_continuous_ard`` on the "
        "held-out-class extrapolation task: fit on the two non-held-out "
        "classes, evaluate on the held-out class. 3 seeds per cell."
    )
    out.append("")
    out.append(
        "Per-kernel held-out-class behaviour (no observations of class C "
        "during training):"
    )
    out.append("")
    out.append("| kernel | how class $C$ is represented at test time |")
    out.append("|---|---|")
    out.append(
        "| `legacy_continuous_ard` | $C$'s numeric coordinate fixed; "
        "extrapolated via learned source ARD lengthscale |"
    )
    out.append(
        "| `hamming` | learned scalar $\\rho$ applied uniformly to all "
        "(seen, $C$) pairs |"
    )
    out.append(
        "| `indexkernel_r2` | $C$'s row of $B$ stays at **random init** "
        "(no gradient) → seed-sensitive |"
    )
    out.append(
        "| **`rbf_embedding_d{k}`** | $x_C$ stays at **equilateral-simplex "
        "init** (no gradient) → deterministic; $K(c_{seen}, C)$ is "
        "fixed at $\\exp(-1 / 2\\ell^2)$ for the learned $\\ell$ |"
    )
    out.append("")

    class_names = {
        0: "Class 0 — Set 1 (mortar)",
        1: "Class 1 — Set 2 (Heidelberg/Class C concrete)",
        2: "Class 2 — Set 3 (Amrize/Class F concrete)",
    }

    def fmt(mu, sigma):
        if pd.isna(mu):
            return "--"
        s = f"{mu:.3f}" if abs(mu) < 50 else f"{mu:.0f}"
        if pd.isna(sigma):
            return s
        sigma_str = (
            f"{sigma:.3f}" if abs(mu) < 50 else f"{sigma:.0f}"
        )
        return f"{s} ± {sigma_str}"

    for holdout_class in holdout_classes:
        out.append(f"## {class_names[holdout_class]} held out")
        out.append("")
        try:
            n_test = int(
                df[df["holdout_class"] == holdout_class]["n_test"].iloc[0]
            )
            out.append(f"$n_{{\\text{{test}}}} = {n_test}$")
            out.append("")
        except Exception:
            pass
        out.append(
            "| kernel | RMSE (psi) | MAE (psi) | MLPD | PIT-KS | "
            "cov95 | CRPS (psi) |"
        )
        out.append("|---|---|---|---|---|---|---|")
        for kernel in rbf_kernels + baseline_kernels:
            try:
                sub = agg.loc[(holdout_class, kernel)]
            except KeyError:
                continue
            cells = []
            for col in [
                "holdout_rmse",
                "holdout_mae",
                "holdout_mean_lpd",
                "holdout_pit_ks",
                "holdout_coverage_95",
                "holdout_crps",
            ]:
                if (col, "mean") not in sub.index:
                    cells.append("--")
                    continue
                cells.append(fmt(sub[(col, "mean")], sub[(col, "std")]))
            out.append(f"| `{kernel}` | " + " | ".join(cells) + " |")
        out.append("")

    out.append("## Headline read")
    out.append("")
    out.append(
        "* All `rbf_embedding_d{1,2,3}` cells should be deterministic "
        "(seed std = 0). If so, the simplex-init blind-extrapolation "
        "design works as theoretically predicted — unlike "
        "`indexkernel_r2` which carries large seed std on Class-0 and "
        "Class-1 holdout."
    )
    out.append(
        "* RBF should match or beat `hamming` because the simplex init's "
        "uniform-equidistant cross-class similarity is structurally "
        "similar to Hamming's single-$\\rho$, but the optimiser is free "
        "to refine the *seen* class embeddings to capture cross-class "
        "structure that Hamming cannot."
    )
    out.append("")
    WRITEUP.write_text("\n".join(out) + "\n")
    print(f"[writeup] wrote {WRITEUP.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
