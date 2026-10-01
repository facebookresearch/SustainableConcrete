#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Inspect 'duplicate' rows in pre-v5 and v5 — are they truly identical
across all columns, or do replicate measurements (Strength1/2/3) differ?

Earlier H7 showed pre-v5 has 647 strength rows but only 638 unique
(comp+temp+time) keys → 9 'duplicates'. Same for v5 (659 rows, 650
unique). This script checks whether those are bit-exact duplicates or
independent replicate measurements with different individual cylinder
strengths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PRE_V5_FIXTURE = REPO_ROOT / "test" / "fixtures" / "boxcrete_data_pre_v5.csv"
V5_DATA = REPO_ROOT / "data" / "boxcrete_data.csv"

COMP_COLS = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
]


def comp_key(row, decimals: int = 2) -> tuple:
    keys = COMP_COLS + ["Temp (C)", "Time"]
    parts = []
    for c in keys:
        v = row.get(c)
        try:
            parts.append(round(float(v), decimals))
        except (TypeError, ValueError):
            parts.append(None)
    return tuple(parts)


def inspect(label: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    df_strength = df.dropna(subset=["Strength (Mean)"]).copy()
    df_strength["_key"] = df_strength.apply(comp_key, axis=1)
    print(f"Strength rows: {len(df_strength)}")
    print(f"Unique keys:   {df_strength['_key'].nunique()}")

    # Find tuples with multiple rows.
    counts = df_strength.groupby("_key").size()
    multi_keys = counts[counts > 1].index
    print(f"Keys with >1 row: {len(multi_keys)}  "
          f"(total replicate rows: {counts[counts > 1].sum()})")
    if len(multi_keys) == 0:
        return

    # For each multi-row key, print the rows side-by-side.
    detail_cols = [
        "Mix Name",
        "Material Source",
        "Time",
        "Strength (Mean)",
        "Strength (Std)",
        "Strength1 (psi)",
        "Strength2 (psi)",
        "Strength3 (psi)",
        "# of measurements",
    ]
    detail_cols = [c for c in detail_cols if c in df_strength.columns]
    print(f"\nReplicate-row detail (showing first 15 multi-keys):")
    n_truly_identical = 0
    n_distinct_replicates = 0
    n_other = 0
    for k in multi_keys[:15]:
        rows = df_strength[df_strength["_key"] == k][detail_cols]
        # Compare all numeric strength cols across the rows.
        strength_cols_present = [
            c for c in [
                "Strength (Mean)", "Strength (Std)",
                "Strength1 (psi)", "Strength2 (psi)", "Strength3 (psi)",
            ] if c in rows.columns
        ]
        all_strength_identical = True
        for c in strength_cols_present:
            vals = rows[c].dropna().unique()
            if len(vals) > 1:
                all_strength_identical = False
                break
        any_replicate_differs = False
        for c in [
            "Strength1 (psi)", "Strength2 (psi)", "Strength3 (psi)"
        ]:
            if c not in rows.columns:
                continue
            vals = rows[c].dropna().unique()
            if len(vals) > 1:
                any_replicate_differs = True
                break
        flag = "?"
        if all_strength_identical:
            n_truly_identical += 1
            flag = "DUPLICATE"
        elif any_replicate_differs:
            n_distinct_replicates += 1
            flag = "INDEPENDENT REPLICATES"
        else:
            n_other += 1
            flag = "OTHER (mean/std differ but cylinders missing)"
        print(f"\n  Key (Mix={rows.iloc[0]['Mix Name']}, "
              f"MS={int(rows.iloc[0]['Material Source'])}, "
              f"t={rows.iloc[0]['Time']}): {flag}")
        print(rows.to_string(index=False))

    # Now do the full count over ALL multi-keys (not just the printed first 15).
    print(f"\n{'-' * 72}")
    print(f"Summary across ALL {len(multi_keys)} multi-row keys:")
    full_counts = {"identical": 0, "indep_replicates": 0, "other": 0}
    for k in multi_keys:
        rows = df_strength[df_strength["_key"] == k]
        strength_cols_present = [
            c for c in [
                "Strength (Mean)", "Strength (Std)",
                "Strength1 (psi)", "Strength2 (psi)", "Strength3 (psi)",
            ] if c in rows.columns
        ]
        all_identical = True
        for c in strength_cols_present:
            vals = rows[c].dropna().unique()
            if len(vals) > 1:
                all_identical = False
                break
        any_replicate_differs = False
        for c in [
            "Strength1 (psi)", "Strength2 (psi)", "Strength3 (psi)"
        ]:
            if c not in rows.columns:
                continue
            vals = rows[c].dropna().unique()
            if len(vals) > 1:
                any_replicate_differs = True
                break
        if all_identical:
            full_counts["identical"] += 1
        elif any_replicate_differs:
            full_counts["indep_replicates"] += 1
        else:
            full_counts["other"] += 1
    print(f"  bit-identical duplicates:     {full_counts['identical']}")
    print(f"  independent replicates:       {full_counts['indep_replicates']}")
    print(f"  other (means differ, no Sx):  {full_counts['other']}")


def main() -> int:
    pre = pd.read_csv(PRE_V5_FIXTURE)
    v5 = pd.read_csv(V5_DATA)
    inspect("PRE-V5", pre)
    inspect("V5", v5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
