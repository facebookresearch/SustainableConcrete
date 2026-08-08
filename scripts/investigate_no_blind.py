#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Ablation: include_blind=False for the top source-kernel candidates.

The V2 strength kernel currently has THREE additive branches:

    K = blind_matern(no_source_dims + extras)
      + categorical_source_branch(source_dim, no_source_dims + extras)
      + additive_rbf_time(time_only)

The blind branch is materials-class-INDEPENDENT (an ARD-Matern over
composition features that ignores the source class). This ablation
tests whether the blind branch is load-bearing — i.e. does removing
it materially affect performance? If the source-aware branch alone
can absorb whatever the blind branch was modelling, dropping the
blind branch is a clean simplification (one fewer ScaleKernel and
~10 fewer hyperparameters).

Variants × contexts × seeds: 3 × 4 × 3 = 36 cells.

Variants:
    joint_hamming_matern         (current Occam-optimal production winner)
    rbf_embedding_d2             (Pareto-corner LOCO winner)
    hamming                      (baseline for context)

All run with include_blind=False; compared against the with-blind
results already in experiments/joint_distance_family_ablation.csv.

Output:
    experiments/NO_BLIND_ABLATION.md
    experiments/no_blind_ablation.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from boxcrete import fit_strength_gp  # noqa: E402
from experiments.three_class_ablation import (  # noqa: E402
    _eval_metrics,
    _load_data,
    _make_class_holdout,
)
from experiments.model_variant_study import held_out_metrics  # noqa: E402

WRITEUP = REPO_ROOT / "experiments" / "NO_BLIND_ABLATION.md"
RESULTS_CSV = REPO_ROOT / "experiments" / "no_blind_ablation.csv"

VARIANTS = ["joint_hamming_matern", "rbf_embedding_d2", "hamming"]
SEEDS = [0, 1, 2]


def _no_blind_factory(source_kernel: str):
    """Build a fit factory that uses include_blind=False."""

    def _fit(X, Y, Yvar, X_bounds, seed):
        torch.manual_seed(seed)
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
                include_blind=False,
            )

        def _patched_factory(*args, **kwargs):
            kwargs.setdefault("include_lognormal_baseline", False)
            return original_factory(*args, **kwargs)

        _kmod.make_gated_strength_kernel_builder = _patched_builder
        _smod.make_gated_strength_kernel_builder = _patched_builder
        _pmod.within_group_prior = _patched_factory
        _kmod.within_group_prior = _patched_factory
        try:
            return fit_strength_gp(
                X=X, Y=Y, Yvar=Yvar, X_bounds=X_bounds, seed=seed
            )
        finally:
            _kmod.make_gated_strength_kernel_builder = original_builder
            _smod.make_gated_strength_kernel_builder = original_builder
            _pmod.within_group_prior = original_factory
            _kmod.within_group_prior = original_factory

    return _fit


def main() -> int:
    rows: list[dict] = []

    print("\n===== In-distribution v5 (include_blind=False) =====", flush=True)
    ds, X, Y, Yvar, bounds, n_real = _load_data("v5")
    for vid in VARIANTS:
        factory = _no_blind_factory(vid)
        for seed in SEEDS:
            t0 = time.time()
            row = {
                "phase": "in_distribution",
                "data": "v5",
                "source_kernel": f"{vid}_no_blind",
                "base_source_kernel": vid,
                "include_blind": False,
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
                f"  {vid}_no_blind seed={seed} loo={loo:.0f} bloo={bloo:.0f} "
                f"({row['wall_sec']:.0f}s)",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    for hc in [0, 1, 2]:
        print(f"\n===== LOCO class {hc} (include_blind=False) =====",
              flush=True)
        X_tr, Y_tr, Yvar_tr, X_te, Y_te_psi, b = _make_class_holdout(hc)
        n_test = int(X_te.shape[0])
        n_train = int(X_tr.shape[0])
        for vid in VARIANTS:
            factory = _no_blind_factory(vid)
            for seed in SEEDS:
                t0 = time.time()
                row = {
                    "phase": "loco",
                    "data": "v5",
                    "holdout_class": hc,
                    "source_kernel": f"{vid}_no_blind",
                    "base_source_kernel": vid,
                    "include_blind": False,
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
                    f"  {vid}_no_blind class={hc} seed={seed} rmse={rmse:.0f} "
                    f"({row['wall_sec']:.0f}s)",
                    flush=True,
                )
                pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[csv] wrote {RESULTS_CSV.relative_to(REPO_ROOT)}")

    # Writeup: compare no_blind vs with_blind (the latter pulled from
    # joint_distance_family_ablation.csv if available).
    with_blind_csv = (
        REPO_ROOT / "experiments" / "joint_distance_family_ablation.csv"
    )
    if with_blind_csv.exists():
        df_with = pd.read_csv(with_blind_csv)
        df_with["base_source_kernel"] = df_with["source_kernel"]
        df_with["include_blind"] = True
    else:
        df_with = pd.DataFrame()

    combined = pd.concat([df_with, df], ignore_index=True)

    out: list[str] = []
    out.append("# `include_blind=False` ablation")
    out.append("")
    out.append(
        "Tests whether the materials-class-INDEPENDENT 'blind Matern' "
        "branch is load-bearing in the V2 strength kernel. The full "
        "kernel is:"
    )
    out.append("")
    out.append("```")
    out.append(
        "K = blind_matern(no_source + extras)"
        "  +  categorical_source_branch(source, no_source + extras)"
        "  +  additive_rbf_time(time)"
    )
    out.append("```")
    out.append("")
    out.append(
        "This ablation runs the top source-kernel candidates with the "
        "blind branch removed, leaving only `source-aware + time`. "
        "If the source-aware kernel alone can absorb the blind branch's "
        "load, dropping it is a clean simplification (~10 fewer hyperparameters)."
    )
    out.append("")

    in_dist = combined[combined["phase"] == "in_distribution"]
    if len(in_dist) > 0:
        out.append("## In-distribution v5 (mean across 3 seeds)")
        out.append("")
        out.append(
            "| base kernel | include_blind | LOO RMSE | bLOO RMSE | "
            "bLOO PIT-KS | S12-bLOO RMSE |"
        )
        out.append("|---|---|---|---|---|---|")
        for vid in VARIANTS:
            for inc in [True, False]:
                sub = in_dist[
                    (in_dist["base_source_kernel"] == vid)
                    & (in_dist["include_blind"] == inc)
                ]
                if len(sub) == 0:
                    continue
                loo = sub["loo_rmse"].mean()
                bloo = sub["bloo_rmse"].mean()
                pit = sub["bloo_pit_ks"].mean()
                s12 = sub["bloo_set12_rmse"].mean()
                inc_str = "yes" if inc else "**NO**"
                out.append(
                    f"| `{vid}` | {inc_str} | {loo:.0f} | {bloo:.0f} | "
                    f"{pit:.3f} | {s12:.0f} |"
                )
        out.append("")

    loco_df = combined[combined["phase"] == "loco"]
    for hc in [0, 1, 2]:
        sub_hc = loco_df[loco_df["holdout_class"] == hc]
        if len(sub_hc) == 0:
            continue
        out.append(f"## LOCO class {hc} (mean across 3 seeds)")
        out.append("")
        n_test = int(sub_hc.iloc[0]["n_test"])
        out.append(f"$n_{{test}} = {n_test}$")
        out.append("")
        out.append("| base kernel | include_blind | RMSE | PIT-KS | cov95 |")
        out.append("|---|---|---|---|---|")
        for vid in VARIANTS:
            for inc in [True, False]:
                sub = sub_hc[
                    (sub_hc["base_source_kernel"] == vid)
                    & (sub_hc["include_blind"] == inc)
                ]
                if len(sub) == 0:
                    continue
                rmse = sub["holdout_rmse"].mean()
                pit = sub["holdout_pit_ks"].mean()
                cov = sub["holdout_coverage_95"].mean()
                inc_str = "yes" if inc else "**NO**"
                out.append(
                    f"| `{vid}` | {inc_str} | {rmse:.0f} | {pit:.3f} | "
                    f"{cov:.3f} |"
                )
        out.append("")

    WRITEUP.write_text("\n".join(out) + "\n")
    print(f"[writeup] wrote {WRITEUP.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
