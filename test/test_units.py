#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import pytest
from boxcrete.units import (
    convert_composition,
    convert_density,
    convert_hrwr_dosage,
    convert_slump,
    convert_strength,
    convert_temperature,
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


class TestDensityConversion:
    def test_lb_per_yd3_to_kg_per_m3(self):
        # 1 lb/yd³ = 0.5933 kg/m³
        result = convert_density(1000.0, to=UnitSystem.METRIC)
        assert pytest.approx(result, rel=1e-4) == 593.28

    def test_kg_per_m3_to_lb_per_yd3(self):
        # Same factor as ``massFactor`` in docs/units.mjs.
        result = convert_density(1.0, to=UnitSystem.IMPERIAL)
        assert pytest.approx(result, abs=1e-4) == 1.6856

    def test_round_trip(self):
        original = 564.0
        metric = convert_density(original, to=UnitSystem.METRIC)
        back = convert_density(metric, to=UnitSystem.IMPERIAL)
        assert pytest.approx(back, rel=1e-10) == original


class TestTemperatureConversion:
    @pytest.mark.parametrize(
        "fahrenheit, celsius", [(32.0, 0.0), (68.0, 20.0), (212.0, 100.0)]
    )
    def test_known_points(self, fahrenheit, celsius):
        to_metric = convert_temperature(fahrenheit, to=UnitSystem.METRIC)
        assert pytest.approx(to_metric, abs=1e-10) == celsius
        to_imperial = convert_temperature(celsius, to=UnitSystem.IMPERIAL)
        assert pytest.approx(to_imperial, abs=1e-10) == fahrenheit


class TestHrwrDosageConversion:
    def test_oz_per_cwt_to_kg_per_m3(self):
        # 400 kg/m³ of binder is 8.818 cwt per m³; at 5 fl oz/cwt that is
        # 44.09 fl oz = 1.304 L, which weighs 1.434 kg at 1.10 g/mL.
        result = convert_hrwr_dosage(5.0, binder=400.0, to=UnitSystem.METRIC)
        assert pytest.approx(result, rel=1e-3) == 1.434

    def test_density_scales_mass(self):
        light = convert_hrwr_dosage(
            5.0, binder=400.0, to=UnitSystem.METRIC, density=1.0
        )
        heavy = convert_hrwr_dosage(
            5.0, binder=400.0, to=UnitSystem.METRIC, density=1.2
        )
        assert pytest.approx(heavy / light, rel=1e-10) == 1.2

    def test_round_trip(self):
        original = 7.5
        metric = convert_hrwr_dosage(original, binder=350.0, to=UnitSystem.METRIC)
        back = convert_hrwr_dosage(metric, binder=350.0, to=UnitSystem.IMPERIAL)
        assert pytest.approx(back, rel=1e-10) == original


def _imperial_mixes():
    return pd.DataFrame(
        {
            "Mix Name": ["A", "B"],
            "Cement (lb/yd3)": [500.0, 400.0],
            "Fly Ash (lb/yd3)": [100.0, 0.0],
            "Slag (lb/yd3)": [0.0, 200.0],
            "Water (lb/yd3)": [270.0, 280.0],
            "HRWR (oz/cwt)": [5.0, 0.0],
            "Temp (F)": [68.0, 50.0],
            "Time": [28, 28],
            "Strength (Mean)": [5000.0, 6000.0],
        }
    )


class TestCompositionConversion:
    def test_imperial_to_metric(self):
        df = _imperial_mixes()
        out = convert_composition(df)
        assert list(out.columns) == [
            "Mix Name",
            "Cement (kg/m3)",
            "Fly Ash (kg/m3)",
            "Slag (kg/m3)",
            "Water (kg/m3)",
            "HRWR (kg/m3)",
            "Temp (C)",
            "Time",
            "Strength (Mean)",
        ]
        assert out["Cement (kg/m3)"].tolist() == pytest.approx([296.64, 237.31], 1e-4)
        assert out["Temp (C)"].tolist() == pytest.approx([20.0, 10.0])
        # 600 lb/yd³ of binder is 355.97 kg/m³; see TestHrwrDosageConversion.
        expected = convert_hrwr_dosage(5.0, binder=355.966, to=UnitSystem.METRIC)
        assert out["HRWR (kg/m3)"].tolist() == pytest.approx([expected, 0.0], 1e-5)
        # columns without a unit to convert are untouched
        for col in ["Mix Name", "Time", "Strength (Mean)"]:
            assert out[col].tolist() == df[col].tolist()

    def test_does_not_modify_input(self):
        df = _imperial_mixes()
        convert_composition(df)
        pd.testing.assert_frame_equal(df, _imperial_mixes())

    def test_round_trip(self):
        df = _imperial_mixes().drop(columns="HRWR (oz/cwt)")
        df["HRWR (lb/yd3)"] = [2.5, 0.0]
        metric = convert_composition(df, to=UnitSystem.METRIC)
        back = convert_composition(metric, to=UnitSystem.IMPERIAL)
        pd.testing.assert_frame_equal(back, df)

    def test_metric_to_imperial_keeps_hrwr_as_mass(self):
        metric = convert_composition(_imperial_mixes())
        out = convert_composition(metric, to=UnitSystem.IMPERIAL)
        assert "HRWR (lb/yd3)" in out.columns
        assert "HRWR (oz/cwt)" not in out.columns

    def test_hrwr_density(self):
        default = convert_composition(_imperial_mixes())
        dense = convert_composition(_imperial_mixes(), hrwr_density=2.2)
        ratio = dense["HRWR (kg/m3)"][0] / default["HRWR (kg/m3)"][0]
        assert pytest.approx(ratio, rel=1e-10) == 2.0

    def test_no_convertible_columns_raises(self):
        # already metric, so there is nothing to convert to metric
        metric = convert_composition(_imperial_mixes())
        with pytest.raises(ValueError, match="no columns to convert"):
            convert_composition(metric)

    def test_collision_raises(self):
        df = _imperial_mixes()
        df["Cement (kg/m3)"] = 300.0
        with pytest.raises(ValueError, match="more than one column"):
            convert_composition(df)

    def test_hrwr_as_mass_and_dosage_raises(self):
        df = _imperial_mixes()
        df["HRWR (lb/yd3)"] = [2.5, 0.0]
        with pytest.raises(ValueError, match=r"\['HRWR \(kg/m3\)'\]"):
            convert_composition(df)

    @pytest.mark.parametrize(
        "mass, temperature, dosage",
        [
            ("(lbs/yd3)", "(°F)", "(fl oz/cwt)"),
            ("(lb/yd^3)", "(degF)", "(OZ/CWT)"),
            ("(Lb / yd³) ", "(f) ", "(oz / cwt)"),
            ("(pcy)", "(F)", "(oz/cwt)"),
        ],
    )
    def test_unit_spellings(self, mass, temperature, dosage):
        expected = convert_composition(_imperial_mixes())
        spelled = _imperial_mixes().rename(
            columns=lambda c: c.replace("(lb/yd3)", mass)
            .replace("(F)", temperature)
            .replace("(oz/cwt)", dosage)
        )
        pd.testing.assert_frame_equal(convert_composition(spelled), expected)

    def test_metric_unit_spellings(self):
        expected = convert_composition(
            convert_composition(_imperial_mixes()), to=UnitSystem.IMPERIAL
        )
        spelled = convert_composition(_imperial_mixes()).rename(
            columns=lambda c: c.replace("(kg/m3)", "(kg/m³)").replace("(C)", "(°C)")
        )
        out = convert_composition(spelled, to=UnitSystem.IMPERIAL)
        pd.testing.assert_frame_equal(out, expected)

    def test_non_string_column_names(self):
        df = _imperial_mixes()
        df[0] = [1.0, 2.0]
        out = convert_composition(df)
        assert out[0].tolist() == [1.0, 2.0]

    def test_unit_system_given_by_value(self):
        expected = convert_composition(_imperial_mixes(), to=UnitSystem.METRIC)
        out = convert_composition(_imperial_mixes(), to="metric")
        pd.testing.assert_frame_equal(out, expected)

    @pytest.mark.parametrize(
        "convert",
        [
            lambda to: convert_composition(_imperial_mixes(), to=to),
            lambda to: convert_density(1.0, to=to),
            lambda to: convert_temperature(1.0, to=to),
            lambda to: convert_hrwr_dosage(1.0, binder=400.0, to=to),
        ],
    )
    def test_invalid_unit_system_raises(self, convert):
        with pytest.raises(ValueError, match="not a valid UnitSystem"):
            convert("Metric units")

    def test_hrwr_dosage_without_binder_raises(self):
        df = _imperial_mixes().drop(columns="Slag (lb/yd3)")
        with pytest.raises(ValueError, match="requires the total binder"):
            convert_composition(df)

    def test_loads_into_dataset(self):
        from boxcrete.utils import load_concrete_strength

        df = _imperial_mixes()
        df["Fine Aggregate (lb/yd3)"] = [1300.0, 1250.0]
        df["Coarse Aggregates (lb/yd3)"] = [1700.0, 1750.0]
        df["Material Source"] = [1, 1]
        df["GWP"] = [250.0, 200.0]
        df["Strength (Std)"] = [100.0, 120.0]
        data = load_concrete_strength(convert_composition(df))
        assert data.X.shape == (2, 10)
        assert not data.X.isnan().any()


class TestLabels:
    def test_strength_label_imperial(self):
        assert strength_label(UnitSystem.IMPERIAL) == "psi"

    def test_strength_label_metric(self):
        assert strength_label(UnitSystem.METRIC) == "MPa"

    def test_strength_label_default(self):
        # Covers the None → DEFAULT_UNIT_SYSTEM fallback branch
        assert strength_label() == "psi"

    def test_slump_label_imperial(self):
        assert slump_label(UnitSystem.IMPERIAL) == "in"

    def test_slump_label_metric(self):
        assert slump_label(UnitSystem.METRIC) == "mm"

    def test_slump_label_default(self):
        # Covers the None → DEFAULT_UNIT_SYSTEM fallback branch
        assert slump_label() == "in"


class TestDefault:
    def test_default_unit_system(self):
        # Default (without BOXCRETE_UNITS env var) should be IMPERIAL
        assert DEFAULT_UNIT_SYSTEM == UnitSystem.IMPERIAL
