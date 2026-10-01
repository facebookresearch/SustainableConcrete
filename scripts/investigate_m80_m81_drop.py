#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Drop the M80/M81 (= Mix_80/Mix_81 in pre-v5; M60/M61 in v5) twin pair
from both datasets and compare bLOO RMSE across model variations.

Background. M80 and M81 have literally bit-identical input columns in
the collaborator's source file (Cement=667, Water=333, all other
features zero or equal) but factor-2.3 different cylinder strengths.
This row-pair forms a pure-noise data point from the GP's perspective.
This experiment quantifies how much the pair contributes to bLOO RMSE
under each kernel × prior × dataset combination.

Grid:
  - data: pre-v5 (with twins), pre-v5 (no Mix_80/81),
          v5 (with twins), v5 (no M60/61)
  - source kernel: legacy_continuous_ard (deployed V2), hamming
  - prior: bare

Output: prints a summary table; writes
``experiments/M80_M81_DROP_ABLATION.md``.
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
WRITEUP = REPO_ROOT / "experiments" / "M80_M81_DROP_ABLATION.md"

# Mix names to drop in each dataset.
PRE_V5_TWINS = ["Mix_80", "Mix_81"]
V5_TWINS = ["M60", "M61"]


def fit_dataset(
    data_path: Path,
    source_kernel: str,
) -> dict[str, float]:
    """Fit a GP with the specified source-kernel branch + bare prior."""
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
        return {"n": int(X.shape[0]), "bloo_rmse": float(bloo["rmse"])}
    finally:
        _kmod.make_gated_strength_kernel_builder = original_builder
        _smod.make_gated_strength_kernel_builder = original_builder
        _pmod.within_group_prior = original_factory
        _kmod.within_group_prior = original_factory


def make_drop_csv(src_csv: Path, mix_names_to_drop: list[str]) -> Path:
    df = pd.read_csv(src_csv)
    n_before = len(df)
    df = df[~df["Mix Name"].isin(mix_names_to_drop)].copy()
    n_after = len(df)
    print(
        f"   dropped {n_before - n_after} rows for {mix_names_to_drop} "
        f"({n_before} -> {n_after})"
    )
    out = REPO_ROOT / "experiments" / f"_drop_tmp_{src_csv.stem}.csv"
    df.to_csv(out, index=False)
    return out


def main() -> int:
    out_lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg)
        out_lines.append(msg)

    emit("# M80/M81 drop ablation")
    emit("")
    emit(
        "Drop the duplicate-fingerprint mortar pair "
        "(Cement=667, Water=333: Mix_80/Mix_81 in pre-v5; M60/M61 in v5) "
        "from each dataset and compare bLOO RMSE across model variants."
    )
    emit("")
    emit("Source: the pair has bit-identical input columns in the "
         "collaborator's file but factor-2.3 different cylinder strengths.")
    emit("")

    # Build the four data variants.
    print("[prep] making temp CSVs")
    pre_full_csv = PRE_V5_FIXTURE
    pre_drop_csv = make_drop_csv(PRE_V5_FIXTURE, PRE_V5_TWINS)
    v5_full_csv = V5_DATA
    v5_drop_csv = make_drop_csv(V5_DATA, V5_TWINS)

    grid = [
        ("pre-v5", "with twins",    pre_full_csv),
        ("pre-v5", "drop twins",    pre_drop_csv),
        ("v5",     "with twins",    v5_full_csv),
        ("v5",     "drop twins",    v5_drop_csv),
    ]
    kernels = ["legacy_continuous_ard", "hamming"]

    results = []
    emit("## Results")
    emit("")
    emit(
        f"| Dataset | twins | Source kernel | n | bLOO RMSE (psi) |"
    )
    emit("|---|---|---|---|---|")
    for ds_label, twins_label, csv_path in grid:
        for kernel in kernels:
            print(
                f"\n[fit] {ds_label} | {twins_label} | source={kernel}"
            )
            try:
                res = fit_dataset(csv_path, kernel)
                results.append(
                    (ds_label, twins_label, kernel, res["n"], res["bloo_rmse"])
                )
                emit(
                    f"| {ds_label} | {twins_label} | {kernel} | "
                    f"{res['n']} | {res['bloo_rmse']:.0f} |"
                )
            except Exception as exc:
                emit(
                    f"| {ds_label} | {twins_label} | {kernel} | -- | "
                    f"FAILED: {exc} |"
                )

    # Cleanup temp CSVs.
    for p in [pre_drop_csv, v5_drop_csv]:
        if p.exists():
            p.unlink()

    emit("")
    emit("## Per-dataset bLOO delta (drop twins vs keep twins)")
    emit("")
    emit("| Dataset | Source kernel | with twins | drop twins | Δ |")
    emit("|---|---|---|---|---|")
    by_key: dict[tuple[str, str, str], float] = {
        (ds, t, k): r for ds, t, k, _n, r in results
    }
    for ds in ["pre-v5", "v5"]:
        for k in kernels:
            full = by_key.get((ds, "with twins", k))
            drop = by_key.get((ds, "drop twins", k))
            if full is None or drop is None:
                continue
            delta = drop - full
            emit(
                f"| {ds} | {k} | {full:.0f} | {drop:.0f} | "
                f"{delta:+.0f} |"
            )

    WRITEUP.write_text("\n".join(out_lines) + "\n")
    print(f"\n[writeup] wrote {WRITEUP.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
