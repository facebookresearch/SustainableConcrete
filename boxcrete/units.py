#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit conversion utilities.

Display of model outputs. Only strength and slump follow the unit
system setting:
- Strength: psi (imperial) ↔ MPa (metric), factor = 0.006895
- Slump: inches (imperial) ↔ mm (metric), factor = 25.4

All other quantities are displayed in their native units regardless of
the unit system setting:
- Composition: kg/m³
- GWP: kg CO₂/m³
- Cost: USD/m³
- Temperature: °C

Conversion of model inputs. The models are fit on compositions in kg/m³
and temperatures in °C. Mix designs recorded in imperial units can be
brought into (and out of) that form with:
- Mass per volume: lb/yd³ ↔ kg/m³ (``convert_density``)
- Temperature: °F ↔ °C (``convert_temperature``)
- HRWR dosage: fl oz/cwt of binder ↔ kg/m³ (``convert_hrwr_dosage``)
- A whole table of mixes at once (``convert_composition``)
"""

from __future__ import annotations

import os
import re
from enum import Enum

import pandas as pd

# Conversion factors
PSI_TO_MPA = 0.006895
INCHES_TO_MM = 25.4
# 1 lb = 0.45359237 kg and 1 yd³ = 0.764554857984 m³, both exact.
LB_PER_YD3_TO_KG_PER_M3 = 0.45359237 / 0.764554857984
# One hundredweight (cwt, 100 lb) of binder in kg, and one US fluid ounce in L.
_KG_PER_CWT = 45.359237
_L_PER_FL_OZ = 0.0295735295625
# Assumed density of liquid HRWR in g/mL (= kg/L), typical of
# polycarboxylate-based products.
DEFAULT_HRWR_DENSITY = 1.10

_BINDER_COLUMNS = ("Cement (kg/m3)", "Fly Ash (kg/m3)", "Slag (kg/m3)")
_HRWR_COLUMN = "HRWR (kg/m3)"
# Units of composition columns as (imperial, metric) pairs. The first
# spelling of each is the one used in converted column names; the others
# are accepted on input, after ``_normalize_unit``.
_MASS_UNITS = (("lb/yd3", "lbs/yd3", "pcy"), ("kg/m3",))
_TEMPERATURE_UNITS = (("F", "degF"), ("C", "degC"))
_DOSAGE_UNITS = ("oz/cwt", "floz/cwt")
# A column name that ends in a unit in parentheses, e.g. "Cement (kg/m3)".
_COLUMN_WITH_UNIT = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")


class UnitSystem(Enum):
    """Unit system for display of strength and slump values."""

    IMPERIAL = "imperial"
    METRIC = "metric"


# Module-level default from environment variable (default: imperial)
_unit_str = os.environ.get("BOXCRETE_UNITS", "imperial").lower()
DEFAULT_UNIT_SYSTEM = (
    UnitSystem.METRIC if _unit_str == "metric" else UnitSystem.IMPERIAL
)

# Display scale factors: multiply model-native values (psi, inches) by these
# to get values in the configured display unit system.
STRENGTH_DISPLAY_SCALE = PSI_TO_MPA if DEFAULT_UNIT_SYSTEM == UnitSystem.METRIC else 1.0
SLUMP_DISPLAY_SCALE = INCHES_TO_MM if DEFAULT_UNIT_SYSTEM == UnitSystem.METRIC else 1.0


def convert_strength(value: float, to: UnitSystem) -> float:
    """Convert a strength value to the target unit system.

    Input is assumed to be in the *other* unit system. If ``to`` is METRIC,
    the input is assumed to be in psi and converted to MPa. If ``to`` is
    IMPERIAL, the input is assumed to be in MPa and converted to psi.

    Args:
        value: Strength value to convert.
        to: Target unit system.

    Returns:
        Converted strength value.
    """
    if to == UnitSystem.METRIC:
        return value * PSI_TO_MPA
    else:
        return value / PSI_TO_MPA


def convert_slump(value: float, to: UnitSystem) -> float:
    """Convert a slump value to the target unit system.

    Input is assumed to be in the *other* unit system. If ``to`` is METRIC,
    the input is assumed to be in inches and converted to mm. If ``to`` is
    IMPERIAL, the input is assumed to be in mm and converted to inches.

    Args:
        value: Slump value to convert.
        to: Target unit system.

    Returns:
        Converted slump value.
    """
    if to == UnitSystem.METRIC:
        return value * INCHES_TO_MM
    else:
        return value / INCHES_TO_MM


def strength_label(unit_system: UnitSystem | None = None) -> str:
    """Return the strength unit label for the given unit system.

    Args:
        unit_system: Unit system. Defaults to ``DEFAULT_UNIT_SYSTEM``.

    Returns:
        ``"psi"`` for imperial, ``"MPa"`` for metric.
    """
    if unit_system is None:
        unit_system = DEFAULT_UNIT_SYSTEM
    return "psi" if unit_system == UnitSystem.IMPERIAL else "MPa"


def slump_label(unit_system: UnitSystem | None = None) -> str:
    """Return the slump unit label for the given unit system.

    Args:
        unit_system: Unit system. Defaults to ``DEFAULT_UNIT_SYSTEM``.

    Returns:
        ``"in"`` for imperial, ``"mm"`` for metric.
    """
    if unit_system is None:
        unit_system = DEFAULT_UNIT_SYSTEM
    return "in" if unit_system == UnitSystem.IMPERIAL else "mm"


def convert_density(value: float, to: UnitSystem) -> float:
    """Convert a mass per unit volume, e.g. the amount of cement in a mix.

    Input is assumed to be in the *other* unit system. If ``to`` is METRIC,
    the input is assumed to be in lb/yd³ and converted to kg/m³. If ``to`` is
    IMPERIAL, the input is assumed to be in kg/m³ and converted to lb/yd³.

    Args:
        value: Mass per unit volume to convert.
        to: Target unit system.

    Returns:
        Converted mass per unit volume.

    Raises:
        ValueError: If ``to`` is not a unit system.
    """
    to = UnitSystem(to)
    if to == UnitSystem.METRIC:
        return value * LB_PER_YD3_TO_KG_PER_M3
    else:
        return value / LB_PER_YD3_TO_KG_PER_M3


def convert_temperature(value: float, to: UnitSystem) -> float:
    """Convert a temperature to the target unit system.

    Input is assumed to be in the *other* unit system. If ``to`` is METRIC,
    the input is assumed to be in °F and converted to °C. If ``to`` is
    IMPERIAL, the input is assumed to be in °C and converted to °F.

    Args:
        value: Temperature to convert.
        to: Target unit system.

    Returns:
        Converted temperature.

    Raises:
        ValueError: If ``to`` is not a unit system.
    """
    to = UnitSystem(to)
    if to == UnitSystem.METRIC:
        return (value - 32.0) * 5.0 / 9.0
    else:
        return value * 9.0 / 5.0 + 32.0


def convert_hrwr_dosage(
    value: float,
    binder: float,
    to: UnitSystem,
    density: float = DEFAULT_HRWR_DENSITY,
) -> float:
    """Convert a HRWR amount between a dosage rate and a mass per volume.

    Admixtures are commonly recorded as a dosage in fluid ounces per
    hundredweight (100 lb) of binder, while the models take the mass of
    HRWR per unit volume of concrete. If ``to`` is METRIC, the input is
    assumed to be in fl oz/cwt and converted to kg/m³. If ``to`` is
    IMPERIAL, the input is assumed to be in kg/m³ and converted to
    fl oz/cwt.

    Args:
        value: HRWR amount to convert.
        binder: Total binder (cement + fly ash + slag) of the mix in kg/m³,
            regardless of ``to``.
        to: Target unit system.
        density: Density of the liquid HRWR in g/mL.

    Returns:
        Converted HRWR amount.

    Raises:
        ValueError: If ``to`` is not a unit system.
    """
    to = UnitSystem(to)
    # kg of HRWR per kg of binder, for a dosage of 1 fl oz/cwt
    mass_ratio_per_dosage = _L_PER_FL_OZ * density / _KG_PER_CWT
    if to == UnitSystem.METRIC:
        return value * mass_ratio_per_dosage * binder
    else:
        return value / (mass_ratio_per_dosage * binder)


def _normalize_unit(unit: str) -> str:
    """Reduces the spellings of a unit to one, e.g. ``"Lb / yd³"`` to ``"lb/yd3"``."""
    unit = unit.lower().replace("³", "3").replace("°", "")
    return re.sub(r"[\s^]", "", unit)


def _split_unit(column: object) -> tuple[str, str] | None:
    """Splits ``"Cement (kg/m3)"`` into ``("Cement", "kg/m3")``, with the unit
    normalized. Returns ``None`` for a column name without a unit."""
    match = _COLUMN_WITH_UNIT.match(column) if isinstance(column, str) else None
    if match is None:
        return None
    return match.group(1), _normalize_unit(match.group(2))


def convert_composition(
    df: pd.DataFrame,
    to: UnitSystem = UnitSystem.METRIC,
    hrwr_density: float = DEFAULT_HRWR_DENSITY,
) -> pd.DataFrame:
    """Convert the composition columns of a table of mixes between unit systems.

    Columns are recognized by the unit in parentheses at the end of their
    name and are renamed to carry the new unit, so that a table recorded in
    imperial units comes out with the column names the models expect::

        "Cement (lb/yd3)" <-> "Cement (kg/m3)"
        "Temp (F)"        <-> "Temp (C)"

    Common spellings of the units are accepted, ignoring case and spaces:
    ``lb/yd3``, ``lbs/yd3``, ``lb/yd^3``, ``lb/yd³`` and ``pcy``; ``kg/m3``,
    ``kg/m^3`` and ``kg/m³``; ``F``, ``°F`` and ``degF``; ``C``, ``°C`` and
    ``degC``. A column whose unit is spelled any other way is not recognized
    and is returned unchanged, like all columns that are not part of the
    composition (e.g. mix name, time, strength).

    When converting to METRIC, a ``"HRWR (oz/cwt)"`` dosage column is also
    converted to ``"HRWR (kg/m3)"``, using the total binder of each mix.
    Converting to IMPERIAL always yields ``"HRWR (lb/yd3)"``; use
    ``convert_hrwr_dosage`` to obtain a dosage instead.

    Example::

        df = pd.read_csv("mixes_in_lb_per_yd3.csv")
        data = load_concrete_strength(convert_composition(df))

    Args:
        df: A DataFrame with one mix per row, in the *other* unit system.
        to: Target unit system. Defaults to METRIC, the units of the models.
        hrwr_density: Density of the liquid HRWR in g/mL, used only to
            convert a ``"HRWR (oz/cwt)"`` column.

    Returns:
        A new DataFrame with converted and renamed composition columns, in
        the original column order.

    Raises:
        ValueError: If ``to`` is not a unit system, if ``df`` has no column
            in the source unit system, if two columns would end up with the
            same name (e.g. a HRWR given both as a mass and as a dosage, or a
            column that already exists in the target units), or if a HRWR
            dosage is given without all of the binder columns.
    """
    to = UnitSystem(to)
    to_metric = to == UnitSystem.METRIC
    source, target = (0, 1) if to_metric else (1, 0)
    converters = (
        (_MASS_UNITS, convert_density),
        (_TEMPERATURE_UNITS, convert_temperature),
    )

    out = df.copy()
    renames: dict[str, str] = {}
    dosage_column = None
    for col in df.columns:
        split = _split_unit(col)
        if split is None:
            continue
        stem, unit = split
        for units, convert in converters:
            if unit in [_normalize_unit(u) for u in units[source]]:
                out[col] = convert(df[col], to)
                renames[col] = f"{stem} ({units[target][0]})"
        if to_metric and stem.lower() == "hrwr" and unit in _DOSAGE_UNITS:
            dosage_column = col
            renames[col] = _HRWR_COLUMN

    if not renames:
        raise ValueError(
            "convert_composition: found no columns to convert to "
            f"{to.value} units. Expected column names ending in a unit such as "
            f"'({_MASS_UNITS[source][0]})' or '({_TEMPERATURE_UNITS[source][0]})'; "
            f"got {list(df.columns)}."
        )
    new_columns = [renames.get(col, col) for col in df.columns]
    duplicates = sorted({col for col in new_columns if new_columns.count(col) > 1})
    if duplicates:
        raise ValueError(
            "convert_composition: more than one column would be named "
            f"{duplicates} after conversion. Remove or rename the columns of "
            f"{list(df.columns)} that describe the same quantity."
        )
    out = out.rename(columns=renames)

    if dosage_column is not None:
        missing = [name for name in _BINDER_COLUMNS if name not in out.columns]
        if missing:
            raise ValueError(
                f"convert_composition: converting '{dosage_column}' requires "
                f"the total binder, but {missing} are missing after conversion."
            )
        out[_HRWR_COLUMN] = convert_hrwr_dosage(
            out[_HRWR_COLUMN],
            binder=out[list(_BINDER_COLUMNS)].sum(axis=1),
            to=to,
            density=hrwr_density,
        )
    return out
