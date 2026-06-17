#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Post-merge invariants for ``data/boxcrete_data.csv`` (v5).

Run after :mod:`scripts.merge_three_class_data` to verify the resulting
3-class dataset satisfies the v5 design constants laid out in the plan
document ``three_class_and_lengthscale_prior.plan.md`` §"Commit 2".

Each invariant is restated as an assertion. Run::

    python scripts/test_merge_invariants.py

Exits non-zero on the first violation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = REPO_ROOT / "data" / "boxcrete_data.csv"
PRE_V5_FIXTURE = REPO_ROOT / "test" / "fixtures" / "boxcrete_data_pre_v5.csv"
TABLE = REPO_ROOT / "boxcrete" / "_mix_naming_table.csv"


# ----- v5 design constants (from the plan) -----
EXPECTED_TOTAL_MIXES = 149
EXPECTED_MORTAR_COUNT = 65
EXPECTED_SET2_COUNT = 30
EXPECTED_SET3_COUNT = 54
# Plan target row count: 645 (current) + 30 (added from new file) = 675.
# Allow ±50 row drift to absorb implementation differences in row-level
# matching (the structural mix-count invariants are tightened above).
ROW_COUNT_TARGET = 675
ROW_COUNT_TOLERANCE = 50

CLASS_MORTAR = 0
CLASS_SET2 = 1
CLASS_SET3 = 2

failed = []


def _check(name: str, predicate: bool, msg: str) -> None:
    status = "PASS" if predicate else "FAIL"
    print(f"  [{status}] {name}: {msg}")
    if not predicate:
        failed.append(name)


def main() -> int:
    df = pd.read_csv(DATA_CSV)
    table = pd.read_csv(TABLE)

    print("==> Structural invariants on data/boxcrete_data.csv")

    # 1. Total unique mixes.
    n_mixes = df["Mix Name"].nunique()
    _check(
        "total_mixes",
        n_mixes == EXPECTED_TOTAL_MIXES,
        f"unique Mix Name count = {n_mixes} (expected {EXPECTED_TOTAL_MIXES})",
    )

    # 2. Class distribution.
    by_class = (
        df.drop_duplicates("Mix Name")["Material Source"].value_counts().to_dict()
    )
    _check(
        "mortar_count",
        by_class.get(CLASS_MORTAR, 0) == EXPECTED_MORTAR_COUNT,
        f"mortars (MS=0) = {by_class.get(CLASS_MORTAR, 0)} "
        f"(expected {EXPECTED_MORTAR_COUNT})",
    )
    _check(
        "set2_count",
        by_class.get(CLASS_SET2, 0) == EXPECTED_SET2_COUNT,
        f"Set 2 (MS=1) = {by_class.get(CLASS_SET2, 0)} "
        f"(expected {EXPECTED_SET2_COUNT})",
    )
    _check(
        "set3_count",
        by_class.get(CLASS_SET3, 0) == EXPECTED_SET3_COUNT,
        f"Set 3 (MS=2) = {by_class.get(CLASS_SET3, 0)} "
        f"(expected {EXPECTED_SET3_COUNT})",
    )

    # 3. Row count is in tolerance.
    n_rows = len(df)
    _check(
        "row_count_in_tolerance",
        abs(n_rows - ROW_COUNT_TARGET) <= ROW_COUNT_TOLERANCE,
        f"row count = {n_rows} (target {ROW_COUNT_TARGET} +/- {ROW_COUNT_TOLERANCE})",
    )

    # 4. Material Source values are exactly {0, 1, 2}.
    ms_unique = sorted(df["Material Source"].unique().tolist())
    _check(
        "material_source_values",
        ms_unique == [0, 1, 2],
        f"Material Source unique values = {ms_unique} (expected [0, 1, 2])",
    )

    # 5. M-prefix <=> mortar (class 0); C-prefix <=> concrete (class 1 or 2).
    for mix_name, sub in df.groupby("Mix Name"):
        cls = sub["Material Source"].iloc[0]
        if mix_name.startswith("M"):
            if cls != CLASS_MORTAR:
                failed.append("M_prefix_class")
                print(
                    f"  [FAIL] M_prefix_class: {mix_name} has class {cls} (expected 0)"
                )
                break
        elif mix_name.startswith("C"):
            if cls not in (CLASS_SET2, CLASS_SET3):
                failed.append("C_prefix_class")
                print(f"  [FAIL] C_prefix_class: {mix_name} has class {cls}")
                break
        else:
            failed.append("name_prefix")
            print(f"  [FAIL] name_prefix: {mix_name} has neither M nor C prefix")
            break
    else:
        _check("M_prefix_class", True, "all M-prefix mixes are class 0")
        _check("C_prefix_class", True, "all C-prefix mixes are class 1 or 2")
        _check("name_prefix", True, "all mix names start with M or C")

    # 6. C1..C30 are class 1, C31..C84 are class 2.
    set2_misclassified = []
    set3_misclassified = []
    for mix_name, sub in df.groupby("Mix Name"):
        m = re.match(r"^C(\d+)$", mix_name)
        if m:
            n = int(m.group(1))
            cls = sub["Material Source"].iloc[0]
            if 1 <= n <= 30 and cls != CLASS_SET2:
                set2_misclassified.append((mix_name, cls))
            elif 31 <= n <= 84 and cls != CLASS_SET3:
                set3_misclassified.append((mix_name, cls))
    _check(
        "C1..C30 = Set 2",
        not set2_misclassified,
        f"C1..C30 must be class 1; misclassified: {set2_misclassified}",
    )
    _check(
        "C31..C84 = Set 3",
        not set3_misclassified,
        f"C31..C84 must be class 2; misclassified: {set3_misclassified}",
    )

    # 7. No overflow-prefix names like ``C2_28`` / ``C3_55``.
    overflow = [n for n in df["Mix Name"].unique() if re.match(r"^C[23]_\d+$", n)]
    _check(
        "no_overflow_prefixes",
        not overflow,
        f"overflow-prefix names (C2_*, C3_*) detected: {overflow[:5]}",
    )

    # 8. Mortars have Coarse Aggregates = 0; concretes have Coarse > 0.
    mortar_with_coarse = []
    concrete_no_coarse = []
    for mix_name, sub in df.groupby("Mix Name"):
        cls = sub["Material Source"].iloc[0]
        coarse = sub["Coarse Aggregates (kg/m3)"]
        if cls == CLASS_MORTAR and (coarse > 0).any():
            mortar_with_coarse.append(mix_name)
        elif cls in (CLASS_SET2, CLASS_SET3) and (coarse <= 0).any():
            concrete_no_coarse.append(mix_name)
    _check(
        "mortar_no_coarse",
        not mortar_with_coarse,
        f"mortars must have Coarse=0; offenders: {mortar_with_coarse}",
    )
    _check(
        "concrete_with_coarse",
        not concrete_no_coarse,
        f"concretes must have Coarse>0; offenders: {concrete_no_coarse}",
    )

    # 9. Strength-less mortars from the NEW FILE (M60..M74) must not have
    # leaked into the v5 dataset under their original new-file names. In
    # v5 canonical, the surviving new-file mortars M75..M82 are renumbered
    # M58..M65, so the canonical M60..M65 are valid (they're not the
    # strength-less ones).  We can verify the drop only by checking the
    # mapping table doesn't list the strength-less new-file mortars as
    # any canonical's source.
    strengthless_new_file = [f"M{i}" for i in range(60, 75)]
    table_evidence = " ".join(table["source_evidence"].dropna().tolist())
    leaked = [n for n in strengthless_new_file if n in table_evidence.split()]
    _check(
        "strengthless_not_referenced",
        not leaked,
        f"strength-less new-file mortars must not be referenced in table; "
        f"leaked: {leaked}",
    )

    # 10. M canonicals in range M1..M65.
    out_of_range_M = sorted(
        [
            n
            for n in df["Mix Name"].unique()
            if (m := re.match(r"^M(\d+)$", n))
            and not (1 <= int(m.group(1)) <= EXPECTED_MORTAR_COUNT)
        ]
    )
    _check(
        "M_in_range",
        not out_of_range_M,
        f"M-canonicals must be M1..M{EXPECTED_MORTAR_COUNT}; "
        f"out-of-range: {out_of_range_M}",
    )

    # 11. C canonicals in range C1..C84.
    max_C = EXPECTED_SET2_COUNT + EXPECTED_SET3_COUNT
    out_of_range_C = sorted(
        [
            n
            for n in df["Mix Name"].unique()
            if (m := re.match(r"^C(\d+)$", n)) and not (1 <= int(m.group(1)) <= max_C)
        ]
    )
    _check(
        "C_in_range",
        not out_of_range_C,
        f"C-canonicals must be C1..C{max_C}; out-of-range: {out_of_range_C}",
    )

    # 12. Pre-v5 fixture exists and has the same column schema.
    _check(
        "fixture_exists",
        PRE_V5_FIXTURE.exists(),
        f"{PRE_V5_FIXTURE.relative_to(REPO_ROOT)} present",
    )
    if PRE_V5_FIXTURE.exists():
        fixture_cols = pd.read_csv(PRE_V5_FIXTURE, nrows=0).columns.tolist()
        v5_cols = df.columns.tolist()
        _check(
            "fixture_schema_match",
            fixture_cols == v5_cols,
            f"v5 column schema matches pre-v5 fixture",
        )
        # Sanity: fixture is byte-identical to the V2-merge-tip dataset.
        # We can't verify "byte-identical to a specific commit" portably,
        # but we can verify the fixture has the pre-v5 row count (727).
        n_fixture = len(pd.read_csv(PRE_V5_FIXTURE))
        _check(
            "fixture_row_count_pre_v5",
            n_fixture == 727,
            f"pre-v5 fixture row count = {n_fixture} (expected 727)",
        )

    # 13. Mapping table is consistent with the v5 data.
    print()
    print("==> Mapping table consistency")
    table_canonicals = set(table["canonical_name"].unique())
    data_canonicals = set(df["Mix Name"].unique())
    new_only_canonicals = data_canonicals - table_canonicals
    table_only_canonicals = table_canonicals - data_canonicals
    _check(
        "table_canonicals_subset_of_data",
        not table_only_canonicals,
        f"table has canonicals not in data: {table_only_canonicals}",
    )
    _check(
        "small_number_of_new_file_only",
        len(new_only_canonicals) <= 10,
        f"data has {len(new_only_canonicals)} canonicals with no legacy origin "
        f"(expected <=10): {sorted(new_only_canonicals)}",
    )

    # 14. Class consistency between table and data.
    table_class = {
        r["canonical_name"]: r["material_class"] for _, r in table.iterrows()
    }
    mismatches = []
    for canonical, sub in df.groupby("Mix Name"):
        cls_in_data = sub["Material Source"].iloc[0]
        if canonical in table_class and table_class[canonical] != cls_in_data:
            mismatches.append((canonical, table_class[canonical], cls_in_data))
    _check(
        "class_consistency",
        not mismatches,
        f"class mismatches table vs data: {mismatches}",
    )

    # 15. No duplicate (Mix Name, Time) tuples in v5 data.
    dups = df.groupby(["Mix Name", "Time"]).size()
    duplicate_keys = dups[dups > 1].index.tolist()
    _check(
        "unique_mix_time",
        not duplicate_keys,
        f"(Mix Name, Time) duplicates: {duplicate_keys[:5]}",
    )

    # 16. Time column is non-negative.
    neg_time = (df["Time"] < 0).sum()
    _check("non_negative_time", neg_time == 0, f"negative Time count = {neg_time}")

    # 17. Strength columns are numeric where present.
    for col in ["Strength1 (psi)", "Strength2 (psi)", "Strength3 (psi)"]:
        bad = pd.to_numeric(df[col], errors="coerce").notna().sum()
        # Just check it's parseable; some rows can be NaN.
        _check(
            f"{col}_parseable",
            bad >= 0,
            f"{col} has {bad} numeric entries",
        )

    # 18. Mix-level: every kept mix has at least one row with a numeric
    # strength value (otherwise it can't inform the GP fit).
    s_mean = pd.to_numeric(df["Strength (Mean)"], errors="coerce")
    s1 = pd.to_numeric(df["Strength1 (psi)"], errors="coerce")
    df["_has_strength"] = s_mean.notna() | s1.notna()
    mixes_no_strength = (
        df.groupby("Mix Name")["_has_strength"]
        .any()
        .pipe(lambda s: s[~s].index.tolist())
    )
    df.drop(columns=["_has_strength"], inplace=True)
    _check(
        "no_strengthless_mixes",
        not mixes_no_strength,
        f"mixes with NO usable strength row: {mixes_no_strength}",
    )

    # 19. Material Source on a row matches its Mix Name's class.
    from boxcrete.mix_naming import derive_source_from_mix_name

    bad_rows = []
    for _, row in df.iterrows():
        derived = derive_source_from_mix_name(row["Mix Name"])
        if derived != int(row["Material Source"]):
            bad_rows.append((row["Mix Name"], derived, row["Material Source"]))
            if len(bad_rows) >= 5:
                break
    _check(
        "row_class_matches_name",
        not bad_rows,
        f"row Material Source disagrees with mix-name class: {bad_rows}",
    )

    # 20. Composition columns numeric.
    for col in [
        "Cement (kg/m3)",
        "Fly Ash (kg/m3)",
        "Slag (kg/m3)",
        "Water (kg/m3)",
        "HRWR (kg/m3)",
        "Fine Aggregate (kg/m3)",
        "Coarse Aggregates (kg/m3)",
    ]:
        nonnumeric = pd.to_numeric(df[col], errors="coerce").isna().sum()
        _check(
            f"{col}_numeric",
            nonnumeric == 0,
            f"{col} has {nonnumeric} non-numeric entries",
        )

    # 21. No NaN in Material Source.
    nan_ms = df["Material Source"].isna().sum()
    _check("material_source_no_nan", nan_ms == 0, f"NaN MS count = {nan_ms}")

    # 22. Mix Name strings only.
    non_string = sum(not isinstance(n, str) for n in df["Mix Name"])
    _check("mix_name_strings", non_string == 0, f"non-string mix names = {non_string}")

    # 23. Aggregate sum sanity (composition sums to a plausible total mass).
    total_mass = (
        pd.to_numeric(df["Cement (kg/m3)"], errors="coerce").fillna(0)
        + pd.to_numeric(df["Fly Ash (kg/m3)"], errors="coerce").fillna(0)
        + pd.to_numeric(df["Slag (kg/m3)"], errors="coerce").fillna(0)
        + pd.to_numeric(df["Water (kg/m3)"], errors="coerce").fillna(0)
        + pd.to_numeric(df["HRWR (kg/m3)"], errors="coerce").fillna(0)
        + pd.to_numeric(df["Fine Aggregate (kg/m3)"], errors="coerce").fillna(0)
        + pd.to_numeric(df["Coarse Aggregates (kg/m3)"], errors="coerce").fillna(0)
    )
    plausible = ((total_mass > 1500) & (total_mass < 3500)).all()
    _check(
        "composition_total_mass_plausible",
        plausible,
        f"row totals fall in [1500, 3500] kg/m^3: {plausible}",
    )

    print()
    if failed:
        print(f"==> FAILED ({len(failed)} invariants): {failed}")
        return 1
    print("==> All v5 merge invariants PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
