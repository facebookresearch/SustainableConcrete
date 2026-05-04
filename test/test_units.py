#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from boxcrete.units import (
    convert_slump,
    convert_strength,
    DEFAULT_UNIT_SYSTEM,
    slump_label,
    strength_label,
    UnitSystem,
)


class TestStrengthConversion:
    def test_psi_to_mpa(self):
        result = convert_strength(1000.0, to=UnitSystem.METRIC)
        assert pytest.approx(result, rel=1e-4) == 6.895

    def test_mpa_to_psi(self):
        result = convert_strength(6.895, to=UnitSystem.IMPERIAL)
        assert pytest.approx(result, rel=1e-3) == 1000.0

    def test_round_trip(self):
        original = 5000.0
        mpa = convert_strength(original, to=UnitSystem.METRIC)
        back = convert_strength(mpa, to=UnitSystem.IMPERIAL)
        assert pytest.approx(back, rel=1e-10) == original


class TestSlumpConversion:
    def test_inches_to_mm(self):
        result = convert_slump(4.0, to=UnitSystem.METRIC)
        assert pytest.approx(result, rel=1e-10) == 101.6

    def test_mm_to_inches(self):
        result = convert_slump(101.6, to=UnitSystem.IMPERIAL)
        assert pytest.approx(result, rel=1e-10) == 4.0

    def test_round_trip(self):
        original = 7.5
        mm = convert_slump(original, to=UnitSystem.METRIC)
        back = convert_slump(mm, to=UnitSystem.IMPERIAL)
        assert pytest.approx(back, rel=1e-10) == original


class TestLabels:
    def test_strength_label_imperial(self):
        assert strength_label(UnitSystem.IMPERIAL) == "psi"

    def test_strength_label_metric(self):
        assert strength_label(UnitSystem.METRIC) == "MPa"

    def test_slump_label_imperial(self):
        assert slump_label(UnitSystem.IMPERIAL) == "in"

    def test_slump_label_metric(self):
        assert slump_label(UnitSystem.METRIC) == "mm"


class TestDefault:
    def test_default_unit_system(self):
        # Default (without BOXCRETE_UNITS env var) should be IMPERIAL
        assert DEFAULT_UNIT_SYSTEM == UnitSystem.IMPERIAL
