# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the v5 canonical-mix-naming module and table."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from boxcrete.mix_naming import (
    CLASS_CONCRETE_SET2,
    CLASS_CONCRETE_SET3,
    CLASS_MORTAR_SET1,
    MixEntry,
    NUM_MATERIAL_CLASSES,
    _class_from_canonical,
    all_mix_entries,
    canonical_name,
    derive_source_from_mix_name,
    get_mix_entry,
    legacy_name,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = REPO_ROOT / "boxcrete" / "_mix_naming_table.csv"
DATA_CSV_PATH = REPO_ROOT / "data" / "boxcrete_data.csv"

# v6 design constants (BOxCrete_All authoritative data):
#   Class 0 mortars   M1..M69   (69)
#   Class 1 concretes C1..C27   (27, Set 2)
#   Class 2 concretes C28..C80  (53, Set 3)
EXPECTED_MORTAR_COUNT = 69
EXPECTED_SET2_COUNT = 27
EXPECTED_SET3_COUNT = 53
EXPECTED_CANONICAL_COUNT = (
    EXPECTED_MORTAR_COUNT + EXPECTED_SET2_COUNT + EXPECTED_SET3_COUNT
)
# Some canonicals have no legacy origin (new mortars added directly from
# the v5 collaborator file). These don't appear in the legacy<->canonical
# table, so the table is allowed to have fewer rows than the canonical
# count.
MAX_NEW_FILE_ONLY_CANONICALS = 10


def test_class_id_constants():
    assert CLASS_MORTAR_SET1 == 0
    assert CLASS_CONCRETE_SET2 == 1
    assert CLASS_CONCRETE_SET3 == 2
    assert NUM_MATERIAL_CLASSES == 3


def test_table_unique_canonical_names():
    canonicals = [e.canonical_name for e in all_mix_entries()]
    duplicates = [c for c, n in Counter(canonicals).items() if n > 1]
    assert not duplicates, f"Duplicate canonical names: {duplicates}"


def test_table_unique_legacy_names():
    legacies = [e.legacy_int_name for e in all_mix_entries()]
    duplicates = [n for n, c in Counter(legacies).items() if c > 1]
    assert not duplicates, f"Duplicate legacy names: {duplicates}"


def test_table_count_within_expected_range():
    entries = list(all_mix_entries())
    # Lower bound: every legacy name in current data must map to a
    # canonical (after dropping the 15 strength-less mortars). The legacy
    # table has at least (149 unique current Mix_<int> - 15 strength-less)
    # but with corruption-collision splits it can exceed this. The hard
    # lower bound is "all 149 canonicals minus a small number of new-file-
    # only additions".
    assert (
        EXPECTED_CANONICAL_COUNT - MAX_NEW_FILE_ONLY_CANONICALS
        <= len(entries)
        <= EXPECTED_CANONICAL_COUNT + 30
    ), (
        f"Table size {len(entries)} not in expected range "
        f"[{EXPECTED_CANONICAL_COUNT - MAX_NEW_FILE_ONLY_CANONICALS}, "
        f"{EXPECTED_CANONICAL_COUNT + 30}]"
    )


def test_class_distribution_in_table():
    by_class = Counter(e.material_class for e in all_mix_entries())
    # The table only contains entries with a legacy origin, so it may
    # under-count mortars (the 3 new-file-only mortars don't appear).
    # Set 2 & Set 3 are expected to be exact since all those concretes
    # come from current data.
    assert by_class[CLASS_CONCRETE_SET2] == EXPECTED_SET2_COUNT
    assert by_class[CLASS_CONCRETE_SET3] == EXPECTED_SET3_COUNT
    assert (
        EXPECTED_MORTAR_COUNT - MAX_NEW_FILE_ONLY_CANONICALS
        <= by_class[CLASS_MORTAR_SET1]
        <= EXPECTED_MORTAR_COUNT
    )


def test_no_overflow_prefixes():
    """Plan §"Commit 2" rejects the C2_/C3_/M70+ overflow-prefix design.

    Canonicals must be the consecutive ``M<int>`` / ``C<int>`` form.
    """
    overflow_re = re.compile(r"^C[23]_\d+$")
    for entry in all_mix_entries():
        assert not overflow_re.match(
            entry.canonical_name
        ), f"Overflow-prefix canonical found: {entry.canonical_name}"


def test_canonical_form_is_M_or_C_int():
    for entry in all_mix_entries():
        assert re.match(
            r"^[MC]\d+$", entry.canonical_name
        ), f"Canonical must match 'M<int>' or 'C<int>': {entry.canonical_name}"


def test_M_canonicals_in_range():
    for entry in all_mix_entries():
        m = re.match(r"^M(\d+)$", entry.canonical_name)
        if m:
            n = int(m.group(1))
            assert 1 <= n <= EXPECTED_MORTAR_COUNT, (
                f"Mortar canonical out of range M1..M{EXPECTED_MORTAR_COUNT}: "
                f"{entry.canonical_name}"
            )
            assert entry.material_class == CLASS_MORTAR_SET1


def test_C_canonicals_in_range_with_correct_class():
    for entry in all_mix_entries():
        m = re.match(r"^C(\d+)$", entry.canonical_name)
        if m:
            n = int(m.group(1))
            if 1 <= n <= EXPECTED_SET2_COUNT:
                assert (
                    entry.material_class == CLASS_CONCRETE_SET2
                ), f"C1..C{EXPECTED_SET2_COUNT} must be class 1 (Set 2): {entry}"
            elif EXPECTED_SET2_COUNT < n <= EXPECTED_SET2_COUNT + EXPECTED_SET3_COUNT:
                assert entry.material_class == CLASS_CONCRETE_SET3, (
                    f"C{EXPECTED_SET2_COUNT + 1}.."
                    f"C{EXPECTED_SET2_COUNT + EXPECTED_SET3_COUNT} "
                    f"must be class 2 (Set 3): {entry}"
                )
            else:
                pytest.fail(
                    f"Concrete canonical out of v5 range "
                    f"C1..C{EXPECTED_SET2_COUNT + EXPECTED_SET3_COUNT}: "
                    f"{entry.canonical_name}"
                )


def test_round_trip_legacy_to_canonical():
    for entry in all_mix_entries():
        # canonical_name(legacy) -> canonical
        assert canonical_name(entry.legacy_int_name) == entry.canonical_name
        # legacy_name(canonical) -> some legacy (may differ for splits;
        # at minimum must round-trip back through canonical_name).
        round_tripped_legacy = legacy_name(entry.canonical_name)
        assert canonical_name(round_tripped_legacy) == entry.canonical_name


@pytest.mark.parametrize("i", list(range(1, EXPECTED_MORTAR_COUNT + 1, 7)))
def test_canonical_M_decodes_to_class_0(i):
    name = f"M{i}"
    assert derive_source_from_mix_name(name) == CLASS_MORTAR_SET1


@pytest.mark.parametrize("i", list(range(1, EXPECTED_SET2_COUNT + 1, 5)))
def test_canonical_C_set2_decodes_to_class_1(i):
    name = f"C{i}"
    assert derive_source_from_mix_name(name) == CLASS_CONCRETE_SET2


@pytest.mark.parametrize(
    "i",
    list(
        range(
            EXPECTED_SET2_COUNT + 1,
            EXPECTED_SET2_COUNT + EXPECTED_SET3_COUNT + 1,
            7,
        )
    ),
)
def test_canonical_C_set3_decodes_to_class_2(i):
    name = f"C{i}"
    assert derive_source_from_mix_name(name) == CLASS_CONCRETE_SET3


def test_unknown_canonical_raises():
    with pytest.raises(ValueError):
        derive_source_from_mix_name("XYZ123")
    with pytest.raises(ValueError):
        derive_source_from_mix_name(f"M{EXPECTED_MORTAR_COUNT + 1}")
    with pytest.raises(ValueError):
        derive_source_from_mix_name(f"C{EXPECTED_SET2_COUNT + EXPECTED_SET3_COUNT + 1}")


def test_get_mix_entry_returns_namedtuple():
    e = next(iter(all_mix_entries()))
    assert isinstance(e, MixEntry)
    assert isinstance(e.canonical_name, str)
    assert isinstance(e.legacy_int_name, str)
    assert isinstance(e.material_class, int)
    assert isinstance(e.source_evidence, str)


def test_coarse_aggregate_invariant_against_data_csv():
    """Mortars (class 0) must have Coarse Aggregates = 0 in every row;
    Concretes (class 1, 2) must have Coarse Aggregates > 0.
    """
    df = pd.read_csv(DATA_CSV_PATH)
    for cls, sub in df.groupby("Material Source"):
        coarse = sub["Coarse Aggregates (kg/m3)"]
        if cls == CLASS_MORTAR_SET1:
            assert (
                coarse == 0
            ).all(), "Class 0 (mortar) must have Coarse Aggregates = 0 in every row"
        else:
            assert (
                coarse > 0
            ).all(), (
                f"Class {cls} (concrete) must have Coarse Aggregates > 0 in every row"
            )


def test_class_assignment_agrees_between_table_and_data():
    """Each canonical name's material_class in the table must match
    the ``Material Source`` value used for its rows in
    ``data/boxcrete_data.csv``.
    """
    df = pd.read_csv(DATA_CSV_PATH)
    table_class = {e.canonical_name: e.material_class for e in all_mix_entries()}
    for mix_name, sub in df.groupby("Mix Name"):
        cls_in_data = sub["Material Source"].unique().tolist()
        assert (
            len(cls_in_data) == 1
        ), f"Mix {mix_name} has multiple Material Source values: {cls_in_data}"
        derived_cls = derive_source_from_mix_name(mix_name)
        assert int(cls_in_data[0]) == derived_cls, (
            f"Material Source mismatch for {mix_name}: "
            f"data={cls_in_data[0]}, derived={derived_cls}"
        )
        if mix_name in table_class:
            assert table_class[mix_name] == derived_cls


# --- Error / edge-case branch coverage -------------------------------------
def test_derive_source_rejects_non_string():
    with pytest.raises(TypeError):
        derive_source_from_mix_name(123)  # type: ignore[arg-type]


def test_derive_source_legacy_name_now_unsupported():
    # v6 is canonical-only; a legacy ``Mix_<int>`` name is not canonical
    # and is rejected with ValueError.
    with pytest.raises(ValueError):
        derive_source_from_mix_name("Mix_9999")


def test_class_from_canonical_rejects_non_canonical():
    with pytest.raises(ValueError):
        _class_from_canonical("ZZZ123")


def test_canonical_name_unknown_legacy_raises():
    with pytest.raises(KeyError):
        canonical_name("Mix_9999")


def test_legacy_name_unknown_canonical_raises():
    with pytest.raises(KeyError):
        legacy_name("M9999")


def test_get_mix_entry_canonical_and_unknown():
    entry = get_mix_entry("M1")
    assert isinstance(entry, MixEntry)
    assert entry.canonical_name == "M1"
    with pytest.raises(KeyError):
        get_mix_entry("ZZZ123")
