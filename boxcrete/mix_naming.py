#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Canonical mix naming and 3-class material-source decoding (v5).

The BOxCrete dataset spans **three** physically distinct material sets
(see ``docs/materials_background.md`` for full chemistry-source detail):

  * **Class 0 — Set 1 — Mortar**: canonical names ``M1..M69``.
    Defining property: ``Coarse Aggregates = 0 kg/m^3``.
    Cement = Amrize 1L (Bloomsdale, MO); fly ash = **Class C**
    (Ozinga, WI); slag = Grade 100 (Ozinga, China); fine
    aggregate = Masonary Sand (Prairie Material, IL); HRWR = Chryso
    Adva Cast 530.

  * **Class 1 — Set 2 — Concrete (Heidelberg cement, Class C fly ash)**:
    canonical names ``C1..C27``. Defining property: nonzero coarse
    aggregate; cement = Heidelberg 1L (Mitchell, IN); fly ash =
    **Class C** (Eco Material, MO); fine aggregate = Concrete Sand
    (Prairie Material, IL); coarse aggregate = Limestone (Vulcan
    Kankakee, IL); HRWR = Chryso Adva Cast 593.

  * **Class 2 — Set 3 — Concrete (Amrize cement, Class F fly ash)**:
    canonical names ``C28..C80``. Defining property: nonzero coarse
    aggregate; cement = Amrize 1L (Bloomsdale, MO); fly ash =
    **Class F** (Eco Material, ND); fine aggregate = Concrete Sand
    (Amrize Elk River, MN); coarse aggregate = #6 + #89 gravel
    (Amrize Empire, MN); HRWR = Sika ViscoCrete 1000.

Naming convention (v5; consecutive integers — no overflow prefixes):

  * ``M<int>``    : mortar mix; M1..M69. All Class 0.
  * ``C<int>``    : concrete mix.
    - ``C1..C27``   are Class 1 (Set 2).
    - ``C28..C80``  are Class 2 (Set 3).
  * ``Mix_<int>`` : legacy auto-generated naming used between commits
    ``611f146`` (2026-05-07) and the ``three-class-source`` branch.
    Resolved to canonical via the bundled
    :data:`_mix_naming_table.csv`.

The v5 design deliberately drops the overflow-prefix tier (``C2_``,
``C3_``, ``M70+``) used by the prior overflow-prefix WIP. After dropping
the 15 strength-less mortars from the new-file collaborator dataset
(``M60..M74`` — all NaN strength) and splitting the 12 corruption-
collision mixes (``Mix_126..Mix_137``; each contains two physically
distinct compositions filed under one name) into their constituent
parts, the canonical name space cleanly contains exactly 149 mixes
that admit consecutive integer numbering within each material class.

The :func:`derive_source_from_mix_name` helper accepts both legacy and
canonical forms and returns the class id ``{0, 1, 2}``.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterator, NamedTuple

# Public class-id constants (align with ``Material Source`` column values
# in the v5-relabeled ``data/boxcrete_data.csv``).
CLASS_MORTAR_SET1 = 0
"""Class 0 — Mortar (Set 1). Defining property: Coarse Aggregates = 0 kg/m^3."""

CLASS_CONCRETE_SET2 = 1
"""Class 1 — Concrete (Set 2; Heidelberg cement, Class C fly ash)."""

CLASS_CONCRETE_SET3 = 2
"""Class 2 — Concrete (Set 3; Amrize cement, Class F fly ash)."""

NUM_MATERIAL_CLASSES = 3
"""Total number of material classes (``{0, 1, 2}``)."""

# Canonical numbering boundaries (BOxCrete_All authoritative labels):
#   Class 0 mortars    M1..M69   (all M<int>)
#   Class 1 concretes  C1..C27   (Set 2)
#   Class 2 concretes  C28..C80  (Set 3)
_SET2_C_MAX = 27
_SET3_C_MAX = 80
_MORTAR_M_MAX = 69

_TABLE_PATH = Path(__file__).resolve().parent / "_mix_naming_table.csv"


class MixEntry(NamedTuple):
    """One row of the canonical mix-naming table."""

    legacy_int_name: str
    """Legacy ``Mix_<int>`` name (possibly with a ``_split0``/``_split1``
    suffix for corruption-collision mixes whose single legacy name covers
    two physically distinct compositions)."""

    canonical_name: str
    """v5 canonical name: ``M1..M69`` (mortar) or ``C1..C80`` (concrete)."""

    material_class: int
    """3-class label ``{0, 1, 2}`` matching ``Material Source`` in the v5
    ``data/boxcrete_data.csv``."""

    source_evidence: str
    """Free-text provenance: ``"row-FP: M12"`` for direct fingerprint
    matches; ``"ambig [...]"`` for ties resolved by smallest-int M-prefix;
    ``"duplicate-of: ..."`` for replicate-batch legacy entries whose
    canonical was already claimed by an earlier sibling."""


@lru_cache(maxsize=1)
def _load_mix_table() -> tuple[MixEntry, ...]:
    """Read the bundled ``_mix_naming_table.csv``.

    Cached; the CSV is read at most once per process.
    """
    if not _TABLE_PATH.exists():  # pragma: no cover - bundled data file always present
        raise FileNotFoundError(
            f"Canonical mix-naming table not found at {_TABLE_PATH}. "
            "It ships with the boxcrete package; a missing file indicates "
            "an incomplete install."
        )
    entries: list[MixEntry] = []
    with _TABLE_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(
                MixEntry(
                    legacy_int_name=row["legacy_int_name"],
                    canonical_name=row["canonical_name"],
                    material_class=int(row["material_class"]),
                    source_evidence=row.get("source_evidence", ""),
                )
            )
    return tuple(entries)


@lru_cache(maxsize=1)
def _canonical_to_entry() -> dict[str, MixEntry]:
    """``canonical_name -> MixEntry`` lookup. Cached."""
    return {e.canonical_name: e for e in _load_mix_table()}


@lru_cache(maxsize=1)
def _legacy_to_entry() -> dict[str, MixEntry]:
    """``legacy_int_name -> MixEntry`` lookup (including ``_splitN``
    suffixed entries for corruption-collision mixes). Cached.
    """
    return {e.legacy_int_name: e for e in _load_mix_table()}


_CANONICAL_M_RE = re.compile(r"^M(\d+)$")
_CANONICAL_C_RE = re.compile(r"^C(\d+)$")


def _class_from_canonical(name: str) -> int:
    """Derive the 3-class label from a canonical ``M<int>`` / ``C<int>`` name.

    Does not consult the bundled table; pure structural parse.
    """
    m = _CANONICAL_M_RE.match(name)
    if m:
        n = int(m.group(1))
        if 1 <= n <= _MORTAR_M_MAX:
            return CLASS_MORTAR_SET1
        raise ValueError(
            f"Canonical mortar name out of range: {name} "
            f"(expected M1..M{_MORTAR_M_MAX})"
        )
    m = _CANONICAL_C_RE.match(name)
    if m:
        n = int(m.group(1))
        if 1 <= n <= _SET2_C_MAX:
            return CLASS_CONCRETE_SET2
        if _SET2_C_MAX < n <= _SET3_C_MAX:
            return CLASS_CONCRETE_SET3
        raise ValueError(
            f"Canonical concrete name out of range: {name} "
            f"(expected C1..C{_SET3_C_MAX})"
        )
    raise ValueError(
        f"Not a canonical mix name (expected 'M<int>' or 'C<int>'): {name!r}"
    )


def derive_source_from_mix_name(name: str) -> int:
    """Return the 3-class material label ``{0, 1, 2}`` for a canonical
    ``M<int>`` / ``C<int>`` mix name (decoded structurally).

    Raises ``TypeError`` if ``name`` is not a string and ``ValueError`` if
    it is not a canonical mix name.
    """
    if not isinstance(name, str):
        raise TypeError(f"Mix name must be a string, got {type(name).__name__}")
    if _CANONICAL_M_RE.match(name) or _CANONICAL_C_RE.match(name):
        return _class_from_canonical(name)
    raise ValueError(
        f"Mix name does not match canonical 'M<int>'/'C<int>' pattern: {name!r}"
    )


def canonical_name(legacy: str) -> str:
    """Convert a legacy ``Mix_<int>`` (optionally ``_splitN``) to its v5
    canonical ``M<int>`` / ``C<int>`` name.

    Raises ``KeyError`` if the legacy name has no canonical equivalent
    (e.g., for strength-less mortars ``Mix_60..Mix_74`` which were
    dropped during the v5 merge).
    """
    legacy_map = _legacy_to_entry()
    if legacy in legacy_map:
        return legacy_map[legacy].canonical_name
    raise KeyError(f"Legacy name {legacy!r} not found in canonical mapping table")


def legacy_name(canonical: str) -> str:
    """Reverse-lookup the legacy name for a canonical ``M<int>`` / ``C<int>``.

    Returns the *first* matching legacy entry (the smallest split index
    for corruption-collision mixes). Raises ``KeyError`` for canonicals
    that are new-file-only additions (no legacy origin).
    """
    canonical_map = _canonical_to_entry()
    if canonical not in canonical_map:
        raise KeyError(
            f"Canonical name {canonical!r} has no legacy origin "
            "(likely a new mix added directly from the v5 collaborator file)"
        )
    return canonical_map[canonical].legacy_int_name


def get_mix_entry(name: str) -> MixEntry:
    """Return the :class:`MixEntry` for ``name`` (canonical or legacy)."""
    canonical_map = _canonical_to_entry()
    if name in canonical_map:
        return canonical_map[name]
    raise KeyError(f"Mix name not found: {name!r}")


def all_mix_entries() -> Iterator[MixEntry]:
    """Iterate over all rows of the bundled canonical mapping table."""
    yield from _load_mix_table()


__all__ = [
    "CLASS_MORTAR_SET1",
    "CLASS_CONCRETE_SET2",
    "CLASS_CONCRETE_SET3",
    "NUM_MATERIAL_CLASSES",
    "MixEntry",
    "all_mix_entries",
    "canonical_name",
    "derive_source_from_mix_name",
    "get_mix_entry",
    "legacy_name",
]
