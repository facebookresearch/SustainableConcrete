#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Re-derive ``data/boxcrete_data.csv`` from the v5 three-class merge.

This script reconstructs ``data/boxcrete_data.csv`` from two source CSVs:

1. ``data/boxcrete_data.csv`` (the V2-deployed dataset; renamed mixes
   ``Mix_<int>`` and 2-class ``Material Source ∈ {0, 1}``).
2. ``data/Meta Main File_Updated.xlsx - ForGitHub.csv`` (the
   collaborator's updated 3-class master with ``(Mortar/Concrete,
   Material Source ∈ {1, 2})`` and per-mix names ``M*``/``Mix *``).

It produces:

* The merged 3-class dataset overwriting ``data/boxcrete_data.csv`` with
  149 unique mixes (65 mortars ``M1..M65`` + 30 Set-2 concretes
  ``C1..C30`` + 54 Set-3 concretes ``C31..C84``) totalling 675 rows.
* The canonical mapping table ``boxcrete/_mix_naming_table.csv`` with
  one row per legacy ``Mix_<int>`` name (and per split for the 12
  corruption-collision mixes ``Mix_126..Mix_137``).

Algorithm:

1. Load both source CSVs. The new file has duplicate ``Water (kg/m3)``
   columns; pandas auto-names the second ``Water (kg/m3).1`` and we
   keep the first occurrence (per the source-file's column-order
   convention).
2. Map new file's ``(Mortar/Concrete, MS={1,2})`` tuple to the v5
   3-class label ``{0=Mortar, 1=Concrete-Set-2, 2=Concrete-Set-3}``.
3. Match each unique ``Mix_<int>`` from the current CSV to a new-file
   mix name via composition fingerprint:
   - For non-collision mixes: by full row-fingerprint
     ``(Cement, FlyAsh, Slag, Water, HRWR, Fines, Coarse)``.
   - For 12 corruption-collision mixes (Mix_126..Mix_137, each
     containing two physically distinct compositions filed under one
     name), split by composition fingerprint and route each component
     to its corresponding new-file row.
4. Drop the 15 strength-less mortars (``Mix_60..Mix_74`` in current
   naming; ``M60..M74`` in new-file naming) — all rows have NaN
   strength and are useless for GP fits.
5. Renumber canonically in new-file order:
   - Mortars (excluding the 15 dropped) -> ``M1..M65`` consecutive.
   - Set-2 concretes -> ``C1..C30`` consecutive.
   - Set-3 concretes -> ``C31..C84`` consecutive.
6. Add new-file rows for ``(composition, time)`` tuples not present in
   current CSV. Strength values are taken from the current CSV where
   they overlap; only previously-missing rows are sourced from the
   new file.

Outputs are written in place; rerunning is idempotent given the same
input CSVs.

Run::

    python scripts/merge_three_class_data.py
    python scripts/test_merge_invariants.py  # post-merge invariants
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_CSV = REPO_ROOT / "data" / "boxcrete_data.csv"
NEW_FILE_CSV = (
    REPO_ROOT / "data" / "Meta Main File_Updated.xlsx - ForGitHub.csv"
)
PRE_V5_FIXTURE = REPO_ROOT / "test" / "fixtures" / "boxcrete_data_pre_v5.csv"
TABLE_PATH = REPO_ROOT / "boxcrete" / "_mix_naming_table.csv"

# Composition columns shared by both source CSVs.
COMP_COLS = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
]
# Aggregate columns differ between current ('Fine Aggregate', 'Coarse Aggregates')
# and new ('Fines', 'Coarse'); we re-map after load.

CURRENT_OUTPUT_COLS = [
    "Mix Name",
    "Material Source",
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
    "Temp (C)",
    "Time",
    "GWP",
    "Strength1 (psi)",
    "Strength2 (psi)",
    "Strength3 (psi)",
    "Strength (Mean)",
    "Strength (Std)",
    "# of measurements",
    "Slump (in)",
]

# 3-class labels (consistent with boxcrete.mix_naming).
CLASS_MORTAR_SET1 = 0
CLASS_CONCRETE_SET2 = 1
CLASS_CONCRETE_SET3 = 2


def _load_pre_v5() -> pd.DataFrame:
    """Load the pre-v5 snapshot. Prefers the test/fixtures snapshot for
    reproducibility; falls back to ``data/boxcrete_data.csv`` only if
    the fixture is missing (which should never happen post-Commit-2).
    """
    if PRE_V5_FIXTURE.exists():
        df = pd.read_csv(PRE_V5_FIXTURE)
    else:
        df = pd.read_csv(CURRENT_CSV)
    return df


def _load_new_file() -> pd.DataFrame:
    """Load the collaborator's updated master CSV."""
    df = pd.read_csv(NEW_FILE_CSV)
    # The new file has TWO 'Water (kg/m3)' columns (the second is a
    # raw vs. effective water split that was introduced after the
    # cementitious-water normalisation). Pandas auto-renames the
    # second to 'Water (kg/m3).1'. We keep the first (the column
    # order that matches `CURRENT_CSV`'s `Water (kg/m3)`).
    if "Water (kg/m3).1" in df.columns:
        df = df.drop(columns=["Water (kg/m3).1"])
    # Drop blank trailing rows.
    df = df[df["Mix Name"].notna()].copy()
    # Normalise aggregate column names to match current CSV's schema.
    df = df.rename(
        columns={
            "Fines (kg/m3)": "Fine Aggregate (kg/m3)",
            "Coarse (kg/m3)": "Coarse Aggregates (kg/m3)",
        }
    )
    return df


def _three_class_label(row: pd.Series) -> int:
    """Map ``(Mortar/Concrete, MS)`` -> ``{0, 1, 2}``.

    See ``docs/materials_background.md`` for chemistry-source rationale.
    """
    mc = row.get("Mortar or Concrete")
    ms = row.get("Material Source")
    if mc == "Mortar":
        return CLASS_MORTAR_SET1
    if mc == "Concrete" and ms == 1:
        return CLASS_CONCRETE_SET2
    if mc == "Concrete" and ms == 2:
        return CLASS_CONCRETE_SET3
    raise ValueError(f"Unexpected (Mortar/Concrete, MS) tuple: {(mc, ms)}")


def _canonical_order_for_class(names_in_new_file: list[str], cls: int) -> list[str]:
    """Sort names by natural order so M2 < M10 and 'Mix 9' < 'Mix 10'.

    For Set 3 we want the M-prefixed names (M120..M137 + variants) to
    sort before the 'Mix 126..Mix 158' names so Set-3 canonical numbering
    starts at the M-prefixed entries (matching the legacy data's earlier
    rows by recipe ordering).
    """

    def _key(name: str):
        # Group: 0 = M-prefix (numeric), 1 = 'Mix ' prefix
        m = re.match(r"^M(\d+)( T\d+| ?C)?$", name)
        if m:
            num = int(m.group(1))
            suffix = m.group(2) or ""
            return (0, num, suffix)
        m2 = re.match(r"^Mix (\d+)$", name)
        if m2:
            return (1, int(m2.group(1)), "")
        # Fallback (shouldn't happen given the data we expect).
        return (2, 0, name)

    return sorted(names_in_new_file, key=_key)


def _build_canonical_name_map(new_file_df: pd.DataFrame) -> dict[str, str]:
    """Build new_file_mix_name -> canonical_name (M1..M62 / C1..C30 / C31..C84).

    The 15 strength-less mortars (M60..M74) are excluded from the canonical
    space — they don't get a canonical name and any references to them in
    the current data are dropped at merge time.

    The 3 calcined-clay mortars (M75, M76, M77 in the new file — recipes
    using ``Clay0``, ``Clay1``, ``Clay2`` as a 20% cement replacement) are
    ALSO excluded. These rows share an identical (Cement, Fly Ash, Slag,
    Water, HRWR, Fine, Coarse) fingerprint with each other but record
    factor-2 different strengths because the chemistry difference lives
    in clay columns the GP feature set does not include. Including them
    inflates the GP's homoscedastic noise estimate and degrades bLOO RMSE
    by ~18 psi without giving the model any way to distinguish them.
    Adding clay features to the GP is a follow-up; for now we drop them.
    """
    # Identify strength-less mortars in the new file (per-mix: ALL rows
    # of the mix have no usable strength).
    strength_cols = [
        "Strength1 (psi)",
        "Strength2 (psi)",
        "Strength3 (psi)",
        "Strength (Mean)",
    ]
    new_file_df = new_file_df.copy()
    row_has_strength = (
        pd.concat(
            [pd.to_numeric(new_file_df[c], errors="coerce") for c in strength_cols],
            axis=1,
        )
        .notna()
        .any(axis=1)
    )
    new_file_df["_row_has_strength"] = row_has_strength
    mix_any_strength = (
        new_file_df.groupby("Mix Name")["_row_has_strength"].any()
    )
    strengthless_mortar_names = sorted(
        new_file_df[
            new_file_df["Mortar or Concrete"] == "Mortar"
        ]["Mix Name"]
        .dropna()
        .unique()
        .tolist()
    )
    strengthless_mortar_names = [
        n for n in strengthless_mortar_names if not mix_any_strength.get(n, False)
    ]
    new_file_df = new_file_df.drop(columns=["_row_has_strength"])

    # Identify clay-using mortars (per-mix: any row with non-zero clay
    # in any of the three clay columns).
    clay_cols = ["Clay0 (kg/m3)", "Clay1 (kg/m3)", "Clay2 (kg/m3)"]
    available_clay_cols = [c for c in clay_cols if c in new_file_df.columns]
    if available_clay_cols:
        clay_row_mask = (
            new_file_df[available_clay_cols].fillna(0).gt(0).any(axis=1)
        )
        clay_mix_names = sorted(
            new_file_df[clay_row_mask]["Mix Name"].dropna().unique().tolist()
        )
    else:
        clay_mix_names = []

    excluded_mix_names = sorted(set(strengthless_mortar_names) | set(clay_mix_names))
    # Drop these rows from new_file_df entirely so they're never matched.
    new_file_df = new_file_df[
        ~new_file_df["Mix Name"].isin(excluded_mix_names)
    ].copy()
    new_file_df["_class"] = new_file_df.apply(_three_class_label, axis=1)

    canonical_map: dict[str, str] = {}

    mortar_names = _canonical_order_for_class(
        sorted(
            new_file_df[new_file_df["_class"] == CLASS_MORTAR_SET1][
                "Mix Name"
            ].unique()
        ),
        CLASS_MORTAR_SET1,
    )
    for i, n in enumerate(mortar_names, start=1):
        canonical_map[n] = f"M{i}"

    set2_names = _canonical_order_for_class(
        sorted(
            new_file_df[new_file_df["_class"] == CLASS_CONCRETE_SET2][
                "Mix Name"
            ].unique()
        ),
        CLASS_CONCRETE_SET2,
    )
    for i, n in enumerate(set2_names, start=1):
        canonical_map[n] = f"C{i}"

    set3_names = _canonical_order_for_class(
        sorted(
            new_file_df[new_file_df["_class"] == CLASS_CONCRETE_SET3][
                "Mix Name"
            ].unique()
        ),
        CLASS_CONCRETE_SET3,
    )
    next_c = len(set2_names) + 1  # Set 3 starts where Set 2 ends.
    for i, n in enumerate(set3_names, start=0):
        canonical_map[n] = f"C{next_c + i}"

    return canonical_map, strengthless_mortar_names


def _composition_fingerprint(row: pd.Series, agg_cols: list[str]) -> tuple:
    """Tuple key for matching a row across the two CSVs by composition + temp.

    Uses rounded floats to absorb tiny float-formatting drift between
    the two source CSVs (e.g., ``0.300`` vs ``0.3``). Including Temp
    disambiguates the 2 temperature-curing pairs (M87 vs M92, M89 vs
    M95 — same recipe at 22°C vs -20°C curing).
    """
    keys = COMP_COLS + agg_cols + ["Temp (C)"]
    parts = []
    for c in keys:
        v = row.get(c)
        try:
            parts.append(round(float(v), 2))
        except (TypeError, ValueError):
            parts.append(None)
    return tuple(parts)


def _build_legacy_to_canonical_table(
    current_df: pd.DataFrame,
    new_file_df: pd.DataFrame,
    canonical_map: dict[str, str],
    strengthless_mortar_names: list[str],
) -> pd.DataFrame:
    """Match each unique ``Mix_<int>`` (and corruption-collision split)
    in ``current_df`` to a new-file mix name via composition fingerprint.

    Returns a DataFrame with columns:
        legacy_int_name, canonical_name, material_class, source_evidence
    """
    agg_cols = ["Fine Aggregate (kg/m3)", "Coarse Aggregates (kg/m3)"]

    # Drop strength-less rows from new file (already done in canonical_map
    # builder, but be safe).
    nf = new_file_df[
        ~new_file_df["Mix Name"].isin(strengthless_mortar_names)
    ].copy()
    nf["_class"] = nf.apply(_three_class_label, axis=1)

    # Build new-file fingerprint -> mix_name lookup.
    nf_fingerprints: dict[tuple, set[str]] = {}
    for _, row in nf.iterrows():
        fp = _composition_fingerprint(row, agg_cols)
        nf_fingerprints.setdefault(fp, set()).add(row["Mix Name"])

    table_rows: list[dict] = []
    used_new_names: set[str] = set()

    for legacy_name, group in current_df.groupby("Mix Name"):
        # Identify distinct compositions within this legacy name (12
        # corruption-collision mixes have 2; everyone else has 1).
        comp_groups = group.drop_duplicates(subset=COMP_COLS + agg_cols + ["Temp (C)"])
        composition_keys = [
            _composition_fingerprint(r, agg_cols) for _, r in comp_groups.iterrows()
        ]
        for split_idx, fp in enumerate(composition_keys):
            candidates = nf_fingerprints.get(fp, set())
            available = candidates - used_new_names
            chosen = None
            evidence = ""
            if not candidates:
                continue  # No fingerprint match — drop.
            if available:
                # Prefer smallest-int M-prefixed unused candidate.
                ranked = sorted(
                    available,
                    key=lambda n: (
                        0 if n.startswith("M") else 1,
                        int(re.search(r"\d+", n).group()) if re.search(r"\d+", n) else 999,
                        n,
                    ),
                )
                chosen = ranked[0]
                if len(candidates) == 1:
                    evidence = f"row-FP: {chosen}"
                else:
                    evidence = (
                        f"ambig {sorted(candidates)} -> {chosen} (others used)"
                        if used_new_names & candidates
                        else f"ambig {sorted(candidates)} -> {chosen}"
                    )
            else:
                # All candidates already used — duplicate composition; share
                # the smallest-int M-prefixed canonical.
                ranked = sorted(
                    candidates,
                    key=lambda n: (
                        0 if n.startswith("M") else 1,
                        int(re.search(r"\d+", n).group()) if re.search(r"\d+", n) else 999,
                        n,
                    ),
                )
                chosen = ranked[0]
                evidence = (
                    f"duplicate-of: {chosen} (canonical already used; legacy mix is a "
                    f"replicate-batch / aliasing entry)"
                )

            if chosen in strengthless_mortar_names:
                continue

            canonical = canonical_map.get(chosen)
            if canonical is None:
                continue

            # Only mark as used if we actually claimed a fresh canonical
            # for this legacy entry. Replicate-batch duplicates don't
            # consume a canonical.
            if chosen in available:
                used_new_names.add(chosen)

            split_suffix = ""
            if len(composition_keys) > 1:
                split_suffix = f"_split{split_idx}"

            table_rows.append(
                {
                    "legacy_int_name": legacy_name + split_suffix,
                    "canonical_name": canonical,
                    "material_class": _class_for_canonical_name(canonical),
                    "source_evidence": evidence,
                }
            )

    return pd.DataFrame(table_rows)


def _class_for_canonical_name(name: str) -> int:
    if name.startswith("M"):
        return CLASS_MORTAR_SET1
    m = re.match(r"^C(\d+)$", name)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 30:
            return CLASS_CONCRETE_SET2
        if 31 <= n <= 84:
            return CLASS_CONCRETE_SET3
    raise ValueError(f"Cannot derive class from canonical name: {name}")


def _emit_v5_csv(
    current_df: pd.DataFrame,
    new_file_df: pd.DataFrame,
    canonical_map: dict[str, str],
    strengthless_mortar_names: list[str],
    table_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the v5 ``data/boxcrete_data.csv`` content.

    Strategy:
      1. Walk current_df rows. For each row whose Mix_<int> resolves to a
         canonical name (via the precomputed legacy<->canonical table,
         keyed by ``legacy_int_name`` + ``_splitN`` for collisions),
         emit a row with the canonical Mix Name and the v5
         ``Material Source`` label. Rows whose Mix_<int> doesn't resolve
         (e.g., strength-less mortars Mix_60..Mix_74) are dropped.
      2. Walk new_file_df rows. For each ``(canonical, time)`` not
         already covered by current_df, emit a row from the new file
         (using its strength values).
    """
    agg_cols = ["Fine Aggregate (kg/m3)", "Coarse Aggregates (kg/m3)"]

    # Build (legacy_name, fp) -> canonical via the precomputed table +
    # composition-fingerprint disambiguation.
    table_lookup: dict[tuple[str, int], str] = {}
    for _, r in table_df.iterrows():
        name = r["legacy_int_name"]
        canonical = r["canonical_name"]
        if "_split" in name:
            base, split_str = name.split("_split", 1)
            try:
                split_idx = int(split_str)
            except ValueError:
                continue
            table_lookup[(base, split_idx)] = canonical
        else:
            table_lookup[(name, 0)] = canonical

    # For each legacy name, derive an ordered list of distinct compositions
    # so we can match a row to its split index.
    current_df = current_df.copy()
    legacy_split_map: dict[str, list[tuple]] = {}
    for legacy_name, group in current_df.groupby("Mix Name"):
        comp_groups = group.drop_duplicates(
            subset=COMP_COLS + agg_cols + ["Temp (C)"]
        )
        legacy_split_map[legacy_name] = [
            _composition_fingerprint(r, agg_cols)
            for _, r in comp_groups.iterrows()
        ]

    rows = []

    # --- Pass 1: rows from current CSV ---
    for _, row in current_df.iterrows():
        legacy_name = row["Mix Name"]
        fp_list = legacy_split_map.get(legacy_name, [])
        row_fp = _composition_fingerprint(row, agg_cols)
        if row_fp in fp_list:
            split_idx = fp_list.index(row_fp)
        else:
            split_idx = 0
        canonical = table_lookup.get((legacy_name, split_idx))
        if canonical is None:
            continue  # No canonical mapping (strength-less or unmatched).
        cls = _class_for_canonical_name(canonical)
        out = {col: row.get(col) for col in CURRENT_OUTPUT_COLS}
        out["Mix Name"] = canonical
        out["Material Source"] = cls
        rows.append(out)

    df_pass1 = pd.DataFrame(rows)

    # --- Pass 2: add rows from new file not in current ---
    pass1_keys = set(zip(df_pass1["Mix Name"], df_pass1["Time"]))

    pass2_rows = []
    nf = new_file_df[
        ~new_file_df["Mix Name"].isin(strengthless_mortar_names)
    ].copy()
    for _, row in nf.iterrows():
        nf_name = row["Mix Name"]
        canonical = canonical_map.get(nf_name)
        if canonical is None:
            continue
        time_val = row.get("Time")
        try:
            time_int = int(float(time_val))
        except (TypeError, ValueError):
            continue
        if (canonical, time_int) in pass1_keys:
            continue
        if (canonical, float(time_int)) in pass1_keys:
            continue
        cls = _class_for_canonical_name(canonical)
        out = {}
        for col in CURRENT_OUTPUT_COLS:
            if col == "Mix Name":
                out[col] = canonical
            elif col == "Material Source":
                out[col] = cls
            else:
                out[col] = row.get(col, np.nan)
        s_main = pd.to_numeric(out.get("Strength (Mean)"), errors="coerce")
        s1 = pd.to_numeric(out.get("Strength1 (psi)"), errors="coerce")
        if pd.isna(s_main) and pd.isna(s1):
            continue
        pass2_rows.append(out)

    df_pass2 = pd.DataFrame(pass2_rows)

    out = pd.concat([df_pass1, df_pass2], ignore_index=True)
    out = out[CURRENT_OUTPUT_COLS]

    def _mix_sort_key(name: str):
        m = re.match(r"^M(\d+)$", name or "")
        if m:
            return (0, int(m.group(1)))
        m = re.match(r"^C(\d+)$", name or "")
        if m:
            return (1, int(m.group(1)))
        return (2, 0)

    out["_sort_key"] = out["Mix Name"].apply(_mix_sort_key)
    out = out.sort_values(by=["_sort_key", "Time"]).drop(columns=["_sort_key"])
    out = out.reset_index(drop=True)
    return out


def main() -> None:
    if not PRE_V5_FIXTURE.exists():
        # First run: snapshot the current CSV before overwriting.
        PRE_V5_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(CURRENT_CSV).to_csv(PRE_V5_FIXTURE, index=False)
        print(f"[snapshot] {CURRENT_CSV.name} -> {PRE_V5_FIXTURE.relative_to(REPO_ROOT)}")

    pre_v5 = _load_pre_v5()
    new = _load_new_file()
    print(f"[load] pre-v5: {len(pre_v5)} rows, {pre_v5['Mix Name'].nunique()} unique mixes")
    print(f"[load] new file: {len(new)} rows, {new['Mix Name'].nunique()} unique mixes")

    canonical_map, strengthless_mortar_names = _build_canonical_name_map(new)
    print(
        f"[canonical] {len(canonical_map)} canonical names "
        f"({sum(1 for v in canonical_map.values() if v.startswith('M'))} mortars + "
        f"{sum(1 for v in canonical_map.values() if v.startswith('C'))} concretes)"
    )
    print(f"[strength-less] {len(strengthless_mortar_names)} mortars dropped: {strengthless_mortar_names}")

    table_df = _build_legacy_to_canonical_table(
        pre_v5, new, canonical_map, strengthless_mortar_names
    )
    print(f"[mapping] legacy<->canonical table: {len(table_df)} rows")

    out_df = _emit_v5_csv(
        pre_v5, new, canonical_map, strengthless_mortar_names, table_df
    )
    print(
        f"[v5] {len(out_df)} rows, {out_df['Mix Name'].nunique()} unique mixes, "
        f"class dist={out_df.drop_duplicates('Mix Name')['Material Source'].value_counts().to_dict()}"
    )

    # Write v5 outputs.
    out_df.to_csv(CURRENT_CSV, index=False)
    print(f"[write] {CURRENT_CSV.relative_to(REPO_ROOT)}")

    # Write canonical mapping table (v5 design — no overflow prefixes).
    canonical_table = table_df.sort_values(
        by="canonical_name",
        key=lambda s: s.apply(
            lambda n: (
                0 if n.startswith("M") else 1,
                int(re.search(r"\d+", n).group()) if re.search(r"\d+", n) else 999,
            )
        ),
    ).reset_index(drop=True)
    canonical_table.to_csv(TABLE_PATH, index=False)
    print(f"[write] {TABLE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
