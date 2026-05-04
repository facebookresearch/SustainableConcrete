#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit conversion utilities for strength and slump display.

Only strength and slump are converted between unit systems:
- Strength: psi (imperial) ↔ MPa (metric), factor = 0.006895
- Slump: inches (imperial) ↔ mm (metric), factor = 25.4

All other quantities remain in their native units regardless of the
unit system setting:
- Composition: kg/m³
- GWP: kg CO₂/m³
- Cost: USD/m³
- Temperature: °C
"""

from __future__ import annotations

import os
from enum import Enum

# Conversion factors
PSI_TO_MPA = 0.006895
INCHES_TO_MM = 25.4


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
