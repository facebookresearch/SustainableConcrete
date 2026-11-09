#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test pre-v5 with the ORIGINAL (0, 1) Material Source bounds.

The deployed pre-v5 model used CONCRETE_BOUNDS_DICT["Material Source"] = (0, 1).
We bumped it to (0, 2) for v5's 3-class scheme. Re-running pre-v5 with the
bumped (0, 2) bounds compresses pre-v5's MS=1 rows to normalized=0.5 instead
of 1.0 — i.e. the comparison against the 'deployed' baseline has been done
against a non-deployed configuration.

Test BOTH bounds for both datasets to nail down the apples-to-apples bLOO.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from boxcrete import fit_strength_gp, load_concrete_strength  # noqa: E402
from boxcrete.utils import CONCRETE_BOUNDS_DICT  # noqa: E402
from experiments.model_variant_study import block_loo_metrics  # noqa: E402

PRE_V5_FIXTURE = REPO_ROOT / "test" / "fixtures" / "boxcrete_data_pre_v5.csv"
V5_DATA = REPO_ROOT / "data" / "boxcrete_data.csv"


def fit_legacy(data_path: Path, ms_bounds: tuple) -> dict[str, float]:
    from boxcrete import kernels as _kmod
    from boxcrete import priors as _pmod
    from boxcrete import strength_model as _smod

    original_builder = _kmod.make_gated_strength_kernel_builder
    original_factory = _pmod.within_group_prior

    def _patched_builder(gate_tau=0.05, **kwargs):
        return original_builder(
            gate_tau=gate_tau,
            source_kernel="legacy_continuous_ard",
            time_tying_sigma=None,
        )

    def _patched_factory(*args, **kwargs):
        kwargs.setdefault("include_lognormal_baseline", False)
        return original_factory(*args, **kwargs)

    _kmod.make_gated_strength_kernel_builder = _patched_builder
    _smod.make_gated_strength_kernel_builder = _patched_builder
    _pmod.within_group_prior = _patched_factory
    _kmod.within_group_prior = _patched_factory

    bounds = dict(CONCRETE_BOUNDS_DICT)
    bounds["Material Source"] = ms_bounds

    try:
        torch.manual_seed(0)
        ds = load_concrete_strength(data_path=str(data_path), bounds_dict=bounds)
        X, Y, Yvar, _ = ds.strength_data
        model = fit_strength_gp(
            X=X, Y=Y, Yvar=Yvar, X_bounds=ds.bounds, seed=0
        )
        bloo = block_loo_metrics(model, n_real=X.shape[0])
        return {
            "n_train": int(X.shape[0]),
            "bloo_rmse": float(bloo["rmse"]),
        }
    finally:
        _kmod.make_gated_strength_kernel_builder = original_builder
        _smod.make_gated_strength_kernel_builder = original_builder
        _pmod.within_group_prior = original_factory
        _kmod.within_group_prior = original_factory


def main() -> int:
    print("=" * 72)
    print("MATERIAL SOURCE BOUNDS ABLATION (legacy V2 architecture)")
    print("=" * 72)
    print()

    cases = [
        ("pre-v5", PRE_V5_FIXTURE, (0, 1), "deployed bounds"),
        ("pre-v5", PRE_V5_FIXTURE, (0, 2), "current default (bumped)"),
        ("v5",     V5_DATA,        (0, 1), "would clip MS=2 — invalid"),
        ("v5",     V5_DATA,        (0, 2), "current default"),
    ]
    print(f"{'Dataset':<10} {'MS bounds':<12} {'Notes':<30} {'bLOO RMSE':>10}  n")
    print("-" * 72)
    results = []
    for label, path, bounds, note in cases:
        try:
            res = fit_legacy(path, bounds)
            print(
                f"{label:<10} {str(bounds):<12} {note:<30} "
                f"{res['bloo_rmse']:>9.0f}  {res['n_train']}"
            )
            results.append((label, bounds, res["bloo_rmse"], res["n_train"]))
        except Exception as exc:
            print(f"{label:<10} {str(bounds):<12} {note:<30} FAILED: {exc}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
