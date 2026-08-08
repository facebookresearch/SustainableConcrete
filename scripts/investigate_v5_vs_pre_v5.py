#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Deep investigation: why does v5 + legacy V2 have ~18 psi worse bLOO
than pre-v5 + legacy V2 (680 -> 698 psi)?

Several hypotheses to test systematically:

  H1 (data volume)              v5 has 48 fewer rows than pre-v5
                                (mostly strength-less mortars dropped).
                                Less data -> harder bLOO mechanically.

  H2 (block structure)          v5 has 149 unique compositions vs
                                pre-v5's ~144 (corruption-collisions
                                split). Smaller blocks; more held-out
                                compositions; bLOO is structurally
                                harder.

  H3 (added strength rows)      v5 adds 30 rows from the new
                                collaborator file for
                                (composition, time) tuples not in
                                pre-v5. These could carry extra noise
                                or have systematic differences from
                                the original strength measurements.

  H4 (3-class labelling)        Pre-v5 pools Set 1 mortar with Set 2
                                concrete under MS=0, applying a
                                single Material Source ARD lengthscale.
                                v5 separates them with a 3-class label.
                                If the legacy continuous-ARD treatment
                                of source as an integer coordinate is
                                actually better than the corrected
                                3-class label, that's a model-
                                specification issue, not a v5 fix.

  H5 (corruption-collision split) The 12 Mix_<n> names that pre-v5
                                pools into one block (with 2 distinct
                                recipes) become 24 separate blocks in
                                v5. Possibly the GP fit on pre-v5
                                benefits from the implicit
                                regularisation of treating the 2
                                recipes as same-block (averaging
                                their effective bLOO prediction).

  H6 (data quality issues)      Could there be a real data-quality
                                bug in v5 — e.g. wrong canonical name
                                assigned to some legacy rows, mixing
                                up strength values?

This script runs each test, prints findings, and writes
``experiments/V5_VS_PRE_V5_INVESTIGATION.md``.
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
WRITEUP = REPO_ROOT / "experiments" / "V5_VS_PRE_V5_INVESTIGATION.md"

COMP_COLS = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
]

OUTPUT: list[str] = []


def emit(msg: str = "") -> None:
    print(msg)
    OUTPUT.append(msg)


def comp_key(row, decimals: int = 2) -> tuple:
    """Composition+temp+time fingerprint."""
    keys = COMP_COLS + ["Temp (C)", "Time"]
    parts = []
    for c in keys:
        v = row.get(c)
        try:
            parts.append(round(float(v), decimals))
        except (TypeError, ValueError):
            parts.append(None)
    return tuple(parts)


def comp_only_key(row, decimals: int = 2) -> tuple:
    """Composition fingerprint (no time)."""
    keys = COMP_COLS + ["Temp (C)"]
    parts = []
    for c in keys:
        v = row.get(c)
        try:
            parts.append(round(float(v), decimals))
        except (TypeError, ValueError):
            parts.append(None)
    return tuple(parts)


# ---------------------------------------------------------------------------
# H1 + H2: data volume + block structure
# ---------------------------------------------------------------------------


def hypothesis_1_2_data_volume_block_structure(pre_v5: pd.DataFrame, v5: pd.DataFrame):
    emit("## H1 + H2: data volume and block structure")
    emit("")
    emit(f"Pre-v5 raw rows: {len(pre_v5)}")
    emit(f"v5 raw rows:     {len(v5)}")
    emit(f"Difference:      {len(pre_v5) - len(v5)} (v5 has fewer)")
    emit("")
    pre_strength = pre_v5.dropna(subset=["Strength (Mean)"])
    v5_strength = v5.dropna(subset=["Strength (Mean)"])
    emit(
        f"After dropping NaN-strength rows: pre-v5 = {len(pre_strength)}, "
        f"v5 = {len(v5_strength)}"
    )
    emit("")

    pre_v5_comp = pre_v5.assign(
        _comp=pre_v5.apply(comp_only_key, axis=1)
    )
    v5_comp = v5.assign(_comp=v5.apply(comp_only_key, axis=1))
    pre_unique_comps = set(pre_v5_comp["_comp"].unique())
    v5_unique_comps = set(v5_comp["_comp"].unique())
    emit(
        f"Pre-v5 unique compositions (composition+temp): {len(pre_unique_comps)}"
    )
    emit(
        f"v5 unique compositions (composition+temp): {len(v5_unique_comps)}"
    )
    common = pre_unique_comps & v5_unique_comps
    only_pre = pre_unique_comps - v5_unique_comps
    only_v5 = v5_unique_comps - pre_unique_comps
    emit(f"Common compositions: {len(common)}")
    emit(f"Only in pre-v5:      {len(only_pre)}")
    emit(f"Only in v5:          {len(only_v5)}")
    emit("")
    emit(
        "**Block size statistics** (median rows per composition):"
    )
    pre_block_sizes = pre_v5_comp.groupby("_comp").size()
    v5_block_sizes = v5_comp.groupby("_comp").size()
    emit(
        f"Pre-v5: min={pre_block_sizes.min()}, "
        f"median={pre_block_sizes.median():.1f}, "
        f"max={pre_block_sizes.max()}"
    )
    emit(
        f"v5:     min={v5_block_sizes.min()}, "
        f"median={v5_block_sizes.median():.1f}, "
        f"max={v5_block_sizes.max()}"
    )
    emit("")
    return common, only_pre, only_v5


# ---------------------------------------------------------------------------
# H3: added strength rows from the new collaborator file
# ---------------------------------------------------------------------------


def hypothesis_3_added_rows(pre_v5: pd.DataFrame, v5: pd.DataFrame):
    emit("## H3: added strength rows (new-file compositions/times)")
    emit("")
    pre_keys = set(
        pre_v5.dropna(subset=["Strength (Mean)"])
        .apply(comp_key, axis=1)
    )
    v5_clean = v5.dropna(subset=["Strength (Mean)"]).copy()
    v5_clean["_key"] = v5_clean.apply(comp_key, axis=1)
    v5_only = v5_clean[~v5_clean["_key"].isin(pre_keys)].drop(
        columns=["_key"]
    )
    emit(
        f"Rows in v5 with strength but with NO matching "
        f"(composition+temp+time) in pre-v5: {len(v5_only)}"
    )
    if len(v5_only) > 0:
        emit("")
        emit("Sample:")
        sample = v5_only[
            ["Mix Name", "Material Source", "Time", "Strength (Mean)"]
            + COMP_COLS[:3]
        ].head(10)
        emit("```")
        emit(sample.to_string())
        emit("```")
    emit("")
    return v5_only


# ---------------------------------------------------------------------------
# H6: data quality — strength values for matching (composition, time) tuples
# ---------------------------------------------------------------------------


def hypothesis_6_strength_quality(pre_v5: pd.DataFrame, v5: pd.DataFrame):
    emit("## H6: strength values for matching (composition+temp+time) rows")
    emit("")
    emit(
        "For every (composition, temp, time) tuple present in BOTH datasets, "
        "compare the recorded strength values."
    )
    emit("")
    pre = pre_v5.dropna(subset=["Strength (Mean)"]).copy()
    v5_clean = v5.dropna(subset=["Strength (Mean)"]).copy()
    pre["_key"] = pre.apply(comp_key, axis=1)
    v5_clean["_key"] = v5_clean.apply(comp_key, axis=1)

    pre_by_key = pre.groupby("_key").agg(
        pre_strength=("Strength (Mean)", "first"),
        pre_mix=("Mix Name", "first"),
    )
    v5_by_key = v5_clean.groupby("_key").agg(
        v5_strength=("Strength (Mean)", "first"),
        v5_mix=("Mix Name", "first"),
    )
    merged = pre_by_key.join(v5_by_key, how="inner")
    emit(f"Common keys with strength in both: {len(merged)}")
    if len(merged) == 0:
        return
    merged["delta"] = merged["v5_strength"] - merged["pre_strength"]
    merged["abs_delta"] = merged["delta"].abs()
    n_mismatch_1psi = (merged["abs_delta"] > 1).sum()
    n_mismatch_10psi = (merged["abs_delta"] > 10).sum()
    n_mismatch_100psi = (merged["abs_delta"] > 100).sum()
    emit(f"Strength values that differ between v5 and pre-v5:")
    emit(f"   |Δ| > 1   psi: {n_mismatch_1psi} of {len(merged)}")
    emit(f"   |Δ| > 10  psi: {n_mismatch_10psi}")
    emit(f"   |Δ| > 100 psi: {n_mismatch_100psi}")
    emit("")
    if n_mismatch_10psi > 0:
        emit("Worst mismatches:")
        worst = merged.nlargest(10, "abs_delta")[
            ["pre_mix", "v5_mix", "pre_strength", "v5_strength", "delta"]
        ]
        emit("```")
        emit(worst.to_string())
        emit("```")
    emit("")
    return merged


# ---------------------------------------------------------------------------
# H4: relabel v5 with pre-v5's binary scheme to test the 3-class effect
# ---------------------------------------------------------------------------


def hypothesis_4_relabel(v5: pd.DataFrame):
    """Apply pre-v5's binary labelling (Set 1 + Set 2 -> 0; Set 3 -> 1)
    to the v5 data and run a fit; compare bLOO.
    """
    emit("## H4: 2-class relabelling of v5 data")
    emit("")
    emit(
        "Re-label v5's Material Source from {0,1,2} -> {0,0,1} to mimic "
        "pre-v5's pooling of mortar+Set-2-concrete under MS=0, then fit "
        "with the legacy continuous-ARD architecture. Compare against "
        "v5 + legacy V2 with the original 3-class label."
    )
    emit("")
    # Save and reload with relabelled data via a side-effect path: write
    # a temp CSV, point load_concrete_strength at it. Avoid mutating v5.
    tmp_csv = REPO_ROOT / "experiments" / "_v5_2class_relabel_tmp.csv"
    v5_relabel = v5.copy()
    v5_relabel["Material Source"] = v5_relabel["Material Source"].map(
        {0: 0, 1: 0, 2: 1}
    )
    # Drop ``Mix Name`` so the canonical-name vs Material Source consistency
    # check at load time does not fire (this is a deliberate relabelling).
    if "Mix Name" in v5_relabel.columns:
        v5_relabel = v5_relabel.drop(columns=["Mix Name"])
    v5_relabel.to_csv(tmp_csv, index=False)

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

    # Need bounds-dict accommodating the relabel.
    from boxcrete.utils import CONCRETE_BOUNDS_DICT
    relabel_bounds = dict(CONCRETE_BOUNDS_DICT)
    relabel_bounds["Material Source"] = (0, 1)

    try:
        torch.manual_seed(0)
        ds = load_concrete_strength(
            data_path=str(tmp_csv), bounds_dict=relabel_bounds
        )
        X, Y, Yvar, _ = ds.strength_data
        emit(f"v5 + 2-class relabel: n_train = {X.shape[0]}")
        model = fit_strength_gp(
            X=X, Y=Y, Yvar=Yvar, X_bounds=ds.bounds, seed=0
        )
        bloo = block_loo_metrics(model, n_real=X.shape[0])
        emit(f"v5 + 2-class relabel + legacy V2: bLOO RMSE = {bloo['rmse']:.0f} psi")
        emit(f"  (vs pre-v5 + legacy V2: 680 psi)")
        emit(f"  (vs v5 + 3-class + legacy V2: 698 psi)")
    except Exception as exc:
        emit(f"FIT FAILED: {exc}")
    finally:
        _kmod.make_gated_strength_kernel_builder = original_builder
        _smod.make_gated_strength_kernel_builder = original_builder
        _pmod.within_group_prior = original_factory
        _kmod.within_group_prior = original_factory
        if tmp_csv.exists():
            tmp_csv.unlink()
    emit("")


# ---------------------------------------------------------------------------
# H3 follow-up: drop the new rows, fit, compare
# ---------------------------------------------------------------------------


def hypothesis_3_drop_new_rows(v5: pd.DataFrame, pre_v5: pd.DataFrame):
    emit("## H3 follow-up: drop new strength rows from v5, refit")
    emit("")
    emit(
        "Drop rows in v5 whose (composition, temp, time) tuple does NOT "
        "appear in pre-v5. This isolates the effect of the 30 'new' "
        "strength rows added from the collaborator's updated file."
    )
    emit("")
    pre_keys = set(
        pre_v5.dropna(subset=["Strength (Mean)"]).apply(comp_key, axis=1)
    )
    v5_clean = v5.dropna(subset=["Strength (Mean)"]).copy()
    v5_clean["_key"] = v5_clean.apply(comp_key, axis=1)
    v5_drop_new = v5_clean[v5_clean["_key"].isin(pre_keys)].drop(
        columns=["_key"]
    )
    emit(
        f"v5 rows after dropping new (no-pre-v5-match): "
        f"{len(v5_drop_new)} (vs original v5 = {len(v5_clean)})"
    )

    tmp_csv = REPO_ROOT / "experiments" / "_v5_drop_new_tmp.csv"
    v5_drop_new.to_csv(tmp_csv, index=False)

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
        ds = load_concrete_strength(data_path=str(tmp_csv))
        X, Y, Yvar, _ = ds.strength_data
        emit(f"v5 minus new rows: n_train = {X.shape[0]}")
        model = fit_strength_gp(
            X=X, Y=Y, Yvar=Yvar, X_bounds=ds.bounds, seed=0
        )
        bloo = block_loo_metrics(model, n_real=X.shape[0])
        emit(f"v5 (no-new-rows) + legacy V2: bLOO RMSE = {bloo['rmse']:.0f} psi")
        emit("  (vs pre-v5 + legacy V2: 680 psi)")
        emit("  (vs v5 + legacy V2: 698 psi)")
    except Exception as exc:
        emit(f"FIT FAILED: {exc}")
    finally:
        _kmod.make_gated_strength_kernel_builder = original_builder
        _smod.make_gated_strength_kernel_builder = original_builder
        _pmod.within_group_prior = original_factory
        _kmod.within_group_prior = original_factory
        if tmp_csv.exists():
            tmp_csv.unlink()
    emit("")


# ---------------------------------------------------------------------------
# H5: restrict v5 to compositions also present in pre-v5
# ---------------------------------------------------------------------------


def hypothesis_5_common_only(v5: pd.DataFrame, pre_v5: pd.DataFrame):
    """Refit v5 restricted to the 144 compositions that are also in pre-v5.
    Isolates the effect of the 15 missing pre-v5 compositions and the 1
    extra v5 composition (i.e. the corruption-collision split + dropped
    strength-less mortars combined).
    """
    emit("## H5: v5 restricted to compositions present in pre-v5")
    emit("")
    emit(
        "Drop v5 rows whose composition+temp fingerprint does NOT appear in "
        "pre-v5. This isolates the structural difference (the 15 "
        "compositions that pre-v5 has but v5 doesn't, plus the 1 v5-only "
        "composition that gets dropped here too)."
    )
    emit("")
    pre_comps = set(
        pre_v5.dropna(subset=["Strength (Mean)"]).apply(comp_only_key, axis=1)
    )
    v5_clean = v5.dropna(subset=["Strength (Mean)"]).copy()
    v5_clean["_comp"] = v5_clean.apply(comp_only_key, axis=1)
    v5_common = v5_clean[v5_clean["_comp"].isin(pre_comps)].drop(
        columns=["_comp"]
    )
    emit(
        f"v5 rows after restricting to pre-v5 compositions: "
        f"{len(v5_common)} (vs original v5 = {len(v5_clean)})"
    )

    tmp_csv = REPO_ROOT / "experiments" / "_v5_common_tmp.csv"
    v5_common.to_csv(tmp_csv, index=False)

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
        ds = load_concrete_strength(data_path=str(tmp_csv))
        X, Y, Yvar, _ = ds.strength_data
        emit(f"v5 (common comps only): n_train = {X.shape[0]}")
        model = fit_strength_gp(
            X=X, Y=Y, Yvar=Yvar, X_bounds=ds.bounds, seed=0
        )
        bloo = block_loo_metrics(model, n_real=X.shape[0])
        emit(
            f"v5 (common comps only) + legacy V2: bLOO RMSE = "
            f"{bloo['rmse']:.0f} psi"
        )
        emit("  (vs pre-v5 + legacy V2: 680 psi)")
        emit("  (vs v5 + legacy V2: 698 psi)")
    except Exception as exc:
        emit(f"FIT FAILED: {exc}")
    finally:
        _kmod.make_gated_strength_kernel_builder = original_builder
        _smod.make_gated_strength_kernel_builder = original_builder
        _pmod.within_group_prior = original_factory
        _kmod.within_group_prior = original_factory
        if tmp_csv.exists():
            tmp_csv.unlink()
    emit("")


def main() -> int:
    pre_v5 = pd.read_csv(PRE_V5_FIXTURE)
    v5 = pd.read_csv(V5_DATA)

    emit("# v5 vs pre-v5 bLOO regression — root-cause investigation")
    emit("")
    emit(
        "The deployed legacy V2 architecture has bLOO RMSE 680 psi on "
        "pre-v5 data. The same architecture on v5 data has bLOO 698 psi "
        "(+18 psi worse). This document investigates the cause."
    )
    emit("")

    common, only_pre, only_v5 = hypothesis_1_2_data_volume_block_structure(
        pre_v5, v5
    )
    del common, only_pre, only_v5  # informational; not used in subsequent steps
    hypothesis_3_added_rows(pre_v5, v5)
    hypothesis_6_strength_quality(pre_v5, v5)
    hypothesis_3_drop_new_rows(v5, pre_v5)
    hypothesis_4_relabel(v5)
    hypothesis_5_common_only(v5, pre_v5)

    emit("")
    emit("## Summary table")
    emit("")
    emit(
        "| Configuration | bLOO RMSE (psi) | Notes |\n"
        "|---|---|---|\n"
        "| pre-v5 + legacy V2 | 680 | deployed model |\n"
        "| v5 + legacy V2     | 698 | +18 psi vs deployed |\n"
        "| v5 (drop new rows) + legacy V2 | (see H3 follow-up) | tests added-row effect |\n"
        "| v5 + 2-class relabel + legacy V2 | (see H4) | tests 3-class effect |\n"
        "| v5 (common comps only) + legacy V2 | (see H5) | tests structural-block effect |\n"
    )
    emit("")

    WRITEUP.write_text("\n".join(OUTPUT))
    print(f"\n[writeup] wrote {WRITEUP.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
