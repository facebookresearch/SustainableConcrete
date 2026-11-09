#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Last-mile diagnostic: build a 'transcribed' v5 that has byte-identical
row content to pre-v5 (same comp+temp+time+strength+std), only differing
in (a) column names and (b) Material Source labelling. If even THAT fits
to 680 psi, we know the gap is purely in the v5-vs-pre-v5 column
schema/Material Source label. If it fits to ~700, the gap is in the
data we just transcribed.
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


def fit_legacy(data_path: Path) -> dict[str, float]:
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

    try:
        torch.manual_seed(0)
        ds = load_concrete_strength(data_path=str(data_path))
        X, Y, Yvar, _ = ds.strength_data
        model = fit_strength_gp(
            X=X, Y=Y, Yvar=Yvar, X_bounds=ds.bounds, seed=0
        )
        bloo = block_loo_metrics(model, n_real=X.shape[0])
        return {
            "n_train": int(X.shape[0]),
            "bloo_rmse": float(bloo["rmse"]),
            "X_first_two": X[:2].tolist(),
        }
    finally:
        _kmod.make_gated_strength_kernel_builder = original_builder
        _smod.make_gated_strength_kernel_builder = original_builder
        _pmod.within_group_prior = original_factory
        _kmod.within_group_prior = original_factory


def main() -> int:
    pre = pd.read_csv(PRE_V5_FIXTURE)
    v5 = pd.read_csv(V5_DATA)

    print("=" * 72)
    print("COLUMN SCHEMAS")
    print("=" * 72)
    pre_cols = set(pre.columns)
    v5_cols = set(v5.columns)
    print(f"Pre-v5 columns ({len(pre_cols)}):  {sorted(pre_cols)}")
    print(f"v5 columns ({len(v5_cols)}):       {sorted(v5_cols)}")
    print(f"Only in pre-v5: {sorted(pre_cols - v5_cols)}")
    print(f"Only in v5:     {sorted(v5_cols - pre_cols)}")
    print()

    print("=" * 72)
    print("MATERIAL SOURCE DISTRIBUTIONS")
    print("=" * 72)
    print(f"Pre-v5 Material Source value counts:\n"
          f"{pre['Material Source'].value_counts().sort_index()}")
    print()
    print(f"v5 Material Source value counts:\n"
          f"{v5['Material Source'].value_counts().sort_index()}")
    print()

    print("=" * 72)
    print("RECONFIRM BASELINE FITS")
    print("=" * 72)
    pre_res = fit_legacy(PRE_V5_FIXTURE)
    print(f"pre-v5 (full):           bLOO = {pre_res['bloo_rmse']:.0f} psi  "
          f"(n={pre_res['n_train']})")
    v5_res = fit_legacy(V5_DATA)
    print(f"v5 (full):               bLOO = {v5_res['bloo_rmse']:.0f} psi  "
          f"(n={v5_res['n_train']})")
    print()

    print("First 2 rows of X tensor:")
    print(f"   pre-v5: {pre_res['X_first_two']}")
    print(f"   v5:     {v5_res['X_first_two']}")
    print()

    print("=" * 72)
    print("BUILD A 'TRANSCRIBED V5' from pre-v5 content with v5 column schema")
    print("=" * 72)
    # Goal: same Y/Yvar/X as pre-v5, but column names match v5. The key
    # delta from pre-v5 to a v5-format file is: which columns (e.g. is
    # there a 'Mix Name' column) and what its values look like.
    #
    # Strategy: take pre-v5 verbatim, but add any v5-only columns with
    # synthetic values (e.g. fabricate Mix Name from row idx). If the
    # bLOO is still 680, we've shown the column schema doesn't matter.
    transcribed = pre.copy()
    only_in_v5 = sorted(v5_cols - pre_cols)
    print(f"v5-only columns to add: {only_in_v5}")
    if "Mix Name" in only_in_v5:
        # Use canonical names — 1 per unique composition.
        from boxcrete.mix_naming import derive_source_from_mix_name  # noqa: F401
        # Just give every row a unique synthetic mix name keyed off MS.
        # The canonical-name sanity check requires Material Source ∈ {0,1,2}.
        # Pre-v5 has MS ∈ {0, 1}. So this check will not fire (it requires
        # exactly the {0,1,2} 3-class set).
        transcribed["Mix Name"] = "row_" + transcribed.index.astype(str)
    out = REPO_ROOT / "experiments" / "_transcribed_pre_to_v5_tmp.csv"
    transcribed.to_csv(out, index=False)
    try:
        trans_res = fit_legacy(out)
        print(
            f"transcribed (pre-v5 content + v5 column schema): "
            f"bLOO = {trans_res['bloo_rmse']:.0f} psi  "
            f"(n={trans_res['n_train']})"
        )
        print("  (vs pre-v5 = 680, vs v5 = 698)")
    finally:
        if out.exists():
            out.unlink()
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
