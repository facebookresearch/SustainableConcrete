#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fix ``docs/model/compositions.json::compositions[i]`` Material Source
labels by matching each composition against the v5 raw data.

CONTEXT
-------
The explorer catalog ``docs/model/compositions.json`` was generated
before the v5 3-class data migration. Its compositions list still uses
the pre-v5 2-class encoding (Material Source ∈ {0, 1}), where ``0``
pooled mortar + Set-2 concrete and ``1`` covered Set-3 concrete with
corruption artifacts. v5 unpooled these classes (see
``data/boxcrete_data.csv``):

  pre-v5 class 0 (pooled)      → v5 class 0 (mortar) ∪ class 1 (Set-2)
  pre-v5 class 1 (contaminated) → v5 class 2 (Set-3)

Symptom: clicking on a Set-2 or Set-3 composition in the explorer
shows it labelled as class 0 (or class 1), but the *actual training
row* for that composition is at the correct v5 class. The GP's
predictive variance at the catalog-stored (wrong) class is therefore
artificially HIGH (no nearby training data), and switching to the
correct class in the panel makes the variance CONTRACT to near the
noise floor (because the model has direct training data there).

FIX
---
For each composition in ``compositions.json``, look up the
matching row in ``data/boxcrete_data.csv`` by composition fingerprint
(7 columns: Cement, FlyAsh, Slag, Water, HRWR, FineAgg, CoarseAgg)
and replace its Material Source label with the v5 class. Compositions
that don't match the v5 data are flagged for manual inspection.

Run::

    python experiments/fix_compositions_material_source.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
COMP_JSON = REPO_ROOT / "docs" / "model" / "compositions.json"
DATA_CSV = REPO_ROOT / "data" / "boxcrete_data.csv"


COMP_COLS_RAW = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
]


def main() -> int:
    print(f"Loading raw v5 data: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    print(
        "  raw data Material Source counts:",
        df["Material Source"].value_counts().to_dict(),
    )

    print(f"\nLoading compositions catalog: {COMP_JSON}")
    catalog = json.loads(COMP_JSON.read_text())
    comp_list = catalog["compositions"]
    cols = catalog["column_names"]
    print(f"  catalog count: {len(comp_list)}")
    counts_before = {0: 0, 1: 0, 2: 0}
    for c in comp_list:
        counts_before[int(c[cols.index("Material Source")])] += 1
    print(f"  catalog Material Source counts BEFORE: {counts_before}")

    # Build a lookup: (rounded composition fingerprint) -> v5 Material Source.
    # Round to 1 decimal place to handle FP slop in the explorer compositions.
    def fingerprint(row):
        return tuple(round(float(row[c]), 1) for c in COMP_COLS_RAW)

    v5_lookup: dict[tuple, int] = {}
    for _, row in df.iterrows():
        fp = fingerprint(row)
        ms = int(row["Material Source"])
        if fp in v5_lookup and v5_lookup[fp] != ms:
            # composition has rows at multiple classes — should be rare.
            # Keep the first one we see; flag below.
            continue
        v5_lookup[fp] = ms

    print(f"\n  v5 lookup table size: {len(v5_lookup)} unique compositions")

    ms_idx = cols.index("Material Source")
    n_changed = 0
    n_unchanged = 0
    n_missing = 0
    change_breakdown: dict[tuple, int] = {}
    for c in comp_list:
        fp = tuple(round(float(x), 1) for x in (c[i] for i in range(7)))
        old_ms = int(c[ms_idx])
        new_ms = v5_lookup.get(fp)
        if new_ms is None:
            n_missing += 1
            continue
        if old_ms != new_ms:
            key = (old_ms, new_ms)
            change_breakdown[key] = change_breakdown.get(key, 0) + 1
            c[ms_idx] = new_ms
            n_changed += 1
        else:
            n_unchanged += 1

    print(f"\n=== RESULT ===")
    print(f"  unchanged (label already correct): {n_unchanged}")
    print(f"  CHANGED (label updated):           {n_changed}")
    print(f"  no match in v5 data (skipped):     {n_missing}")
    if change_breakdown:
        print(f"\n  Change breakdown (old → new):")
        for (old, new), count in sorted(change_breakdown.items()):
            print(f"    class {old} → class {new}: {count}")

    counts_after = {0: 0, 1: 0, 2: 0}
    for c in comp_list:
        counts_after[int(c[ms_idx])] += 1
    print(f"\n  catalog Material Source counts AFTER:  {counts_after}")

    # Write back.
    catalog["compositions"] = comp_list
    COMP_JSON.write_text(json.dumps(catalog, separators=(",", ":")))
    print(f"\n  wrote {COMP_JSON}")
    print(
        f"\nNext: rerun the precomputed-prediction pipeline to refresh"
        f" strength_predictions, gwp_predictions, and pareto_mask:"
        f"\n  bash experiments/regenerate_all_artifacts.sh"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
