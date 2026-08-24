#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the strength break-screening transforms (COV + monotonicity).

These transforms operate on the RAW break-level data (all three
``Strength{1,2,3} (psi)`` cylinders preserved) and recompute the
derived ``Strength (Mean)``/``Strength (Std)``/``# of measurements``
columns the strength GP actually consumes. Keeping the raw breaks in
the CSV and screening at load time makes the screening rule
reproducible and ablatable (see THREE_CLASS_AND_PRIOR_BENCHMARK.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from boxcrete.data_cleaning import (
    apply_cov_screen,
    apply_monotonicity_screen,
    cov_percent,
)

BREAK_COLS = ["Strength1 (psi)", "Strength2 (psi)", "Strength3 (psi)"]


def _row(mix, time, b1, b2, b3, source=0):
    return {
        "Mix Name": mix,
        "Material Source": source,
        "Time": time,
        "Strength1 (psi)": b1,
        "Strength2 (psi)": b2,
        "Strength3 (psi)": b3,
        "Strength (Mean)": np.nan,
        "Strength (Std)": np.nan,
        "# of measurements": np.nan,
    }


# ---------------------------------------------------------------------------
# cov_percent
# ---------------------------------------------------------------------------
def test_cov_percent_uses_sample_std():
    # ddof=1 sample SD of [100, 110, 120] is 10; mean 110 -> 9.0909%
    assert cov_percent([100.0, 110.0, 120.0]) == pytest.approx(100 * 10 / 110)


def test_cov_percent_ignores_nan():
    assert cov_percent([100.0, 120.0, np.nan]) == pytest.approx(
        100 * np.std([100.0, 120.0], ddof=1) / 110.0
    )


# ---------------------------------------------------------------------------
# apply_cov_screen
# ---------------------------------------------------------------------------
def test_cov_screen_keeps_all_three_when_under_threshold():
    df = pd.DataFrame([_row("M1", 1, 100.0, 101.0, 102.0)])
    out = apply_cov_screen(df, threshold=10.0)
    assert len(out) == 1
    assert out.iloc[0]["# of measurements"] == 3
    assert out.iloc[0]["Strength (Mean)"] == pytest.approx(101.0)


def test_cov_screen_trims_worst_break_and_keeps_two():
    # [1670, 5706, 6047]: COV ~44% (>10). Removing 1670 (furthest from mean)
    # leaves [5706, 6047], COV ~4.1% (<=10) -> keep 2.
    df = pd.DataFrame([_row("C20", 14, 6047.0, 1670.0, 5706.0)])
    out = apply_cov_screen(df, threshold=10.0)
    assert len(out) == 1
    assert out.iloc[0]["# of measurements"] == 2
    assert out.iloc[0]["Strength (Mean)"] == pytest.approx(np.mean([5706.0, 6047.0]))


def test_cov_screen_drops_row_when_still_over_threshold_after_trim():
    # Widely scattered: no pair is within 10% COV -> drop the whole row.
    df = pd.DataFrame([_row("Bad", 1, 100.0, 500.0, 5000.0)])
    out = apply_cov_screen(df, threshold=10.0)
    assert len(out) == 0


def test_cov_screen_recomputes_std_as_sample_std():
    df = pd.DataFrame([_row("M1", 1, 100.0, 110.0, 120.0)])
    out = apply_cov_screen(df, threshold=100.0)  # keep all 3
    assert out.iloc[0]["Strength (Std)"] == pytest.approx(
        np.std([100.0, 110.0, 120.0], ddof=1)
    )


def test_cov_screen_preserves_raw_breaks():
    df = pd.DataFrame([_row("C20", 14, 6047.0, 1670.0, 5706.0)])
    out = apply_cov_screen(df, threshold=10.0)
    # Raw breaks are untouched even though the mean is from 2 cylinders.
    assert out.iloc[0]["Strength1 (psi)"] == 6047.0
    assert out.iloc[0]["Strength2 (psi)"] == 1670.0
    assert out.iloc[0]["Strength3 (psi)"] == 5706.0


def test_cov_screen_leaves_fewer_than_three_breaks_untouched():
    # Only two cylinders cast; nothing to trim, keep as-is.
    df = pd.DataFrame([_row("M1", 1, 100.0, 200.0, np.nan)])
    out = apply_cov_screen(df, threshold=10.0)
    assert len(out) == 1
    assert out.iloc[0]["# of measurements"] == 2


# ---------------------------------------------------------------------------
# apply_monotonicity_screen
# ---------------------------------------------------------------------------
def test_monotonicity_removes_later_age_below_earlier():
    df = pd.DataFrame(
        [
            _row("M1", 1, 100.0, 100.0, 100.0),
            _row("M1", 3, 200.0, 200.0, 200.0),
            _row("M1", 5, 150.0, 150.0, 150.0),  # regresses below 3-day
            _row("M1", 28, 300.0, 300.0, 300.0),
        ]
    )
    df = apply_cov_screen(df, threshold=100.0)
    out = apply_monotonicity_screen(df)
    times = sorted(out["Time"].tolist())
    assert times == [1, 3, 28]  # 5-day removed


def test_monotonicity_keeps_strictly_increasing_curve():
    df = pd.DataFrame(
        [
            _row("M1", 1, 100.0, 100.0, 100.0),
            _row("M1", 3, 200.0, 200.0, 200.0),
            _row("M1", 28, 300.0, 300.0, 300.0),
        ]
    )
    df = apply_cov_screen(df, threshold=100.0)
    out = apply_monotonicity_screen(df)
    assert len(out) == 3


def test_monotonicity_is_per_mix():
    df = pd.DataFrame(
        [
            _row("M1", 1, 100.0, 100.0, 100.0),
            _row("M1", 3, 200.0, 200.0, 200.0),
            _row("M2", 1, 500.0, 500.0, 500.0),
            _row("M2", 3, 400.0, 400.0, 400.0),  # M2 regresses, M1 does not
        ]
    )
    df = apply_cov_screen(df, threshold=100.0)
    out = apply_monotonicity_screen(df)
    assert set(zip(out["Mix Name"], out["Time"])) == {
        ("M1", 1),
        ("M1", 3),
        ("M2", 1),
    }
