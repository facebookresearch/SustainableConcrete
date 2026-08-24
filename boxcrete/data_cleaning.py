#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Break-level screening transforms for the concrete strength dataset.

The raw dataset (``data/boxcrete_data.csv``) stores every measured
cylinder break in ``Strength{1,2,3} (psi)``. These transforms screen
those raw breaks at load time and recompute the derived columns the
strength GP consumes — ``Strength (Mean)``, ``Strength (Std)`` (sample
SD, ddof=1) and ``# of measurements`` — WITHOUT mutating the raw break
columns. Keeping the raw values in the CSV and screening in code makes
each screening rule reproducible, parameterisable, and ablatable.

Two screens, matching the collaborator's protocol:

1. **COV screen** (``apply_cov_screen``): for each test with three
   breaks, if the coefficient of variation (COV = sample SD / mean)
   exceeds ``threshold`` percent, drop the single break furthest from
   the mean and recompute on the remaining two. If still over
   threshold, drop the whole (mix, age) row.

2. **Monotonicity screen** (``apply_monotonicity_screen``): within a
   mix, ordered by curing age, remove any later-age result whose
   (post-COV) mean strength falls below an earlier age — concrete
   strength should not decrease with age.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BREAK_COLS = ["Strength1 (psi)", "Strength2 (psi)", "Strength3 (psi)"]
DEFAULT_MEAN_COL = "Strength (Mean)"
DEFAULT_STD_COL = "Strength (Std)"
DEFAULT_N_COL = "# of measurements"
DEFAULT_MIX_COL = "Mix Name"
DEFAULT_TIME_COL = "Time"


def cov_percent(breaks) -> float:
    """Coefficient of variation (%) of a set of breaks: 100 * sample SD / mean.

    NaNs are ignored. Uses the sample standard deviation (``ddof=1``) to
    match the ``Strength (Std)`` convention in the dataset. Returns
    ``inf`` for a non-positive mean (degenerate sentinel rows) so the
    caller treats them as failing any finite threshold.
    """
    vals = np.asarray([b for b in breaks if pd.notna(b)], dtype=float)
    mean = vals.mean()
    if mean <= 0:
        return float("inf")
    return 100.0 * vals.std(ddof=1) / mean


def _screen_breaks(values: list[float], threshold: float) -> list[float] | None:
    """Apply the COV protocol to the non-null breaks of one test.

    Returns the list of breaks to keep, or ``None`` if the row should be
    dropped entirely. Rows with fewer than three breaks are returned
    unchanged (nothing to trim).
    """
    if len(values) < 3:
        return values
    if cov_percent(values) <= threshold:
        return values
    arr = np.asarray(values, dtype=float)
    furthest = int(np.argmax(np.abs(arr - arr.mean())))
    pair = np.delete(arr, furthest).tolist()
    if cov_percent(pair) <= threshold:
        return pair
    return None


def apply_cov_screen(
    df: pd.DataFrame,
    threshold: float = 10.0,
    break_cols: list[str] = DEFAULT_BREAK_COLS,
    mean_col: str = DEFAULT_MEAN_COL,
    std_col: str = DEFAULT_STD_COL,
    n_col: str = DEFAULT_N_COL,
) -> pd.DataFrame:
    """Screen each row by within-test COV and recompute derived columns.

    The raw ``break_cols`` are preserved untouched; only ``mean_col``,
    ``std_col`` and ``n_col`` are recomputed from the retained breaks.
    Rows whose scatter cannot be brought under ``threshold`` are dropped.
    """
    kept_rows = []
    for idx, row in df.iterrows():
        values = [float(row[c]) for c in break_cols if pd.notna(row[c])]
        kept = _screen_breaks(values, threshold)
        if kept is None:
            continue
        new_row = row.copy()
        arr = np.asarray(kept, dtype=float)
        new_row[mean_col] = arr.mean()
        new_row[std_col] = arr.std(ddof=1) if len(arr) > 1 else 0.0
        new_row[n_col] = len(arr)
        kept_rows.append(new_row)
    if not kept_rows:
        return df.iloc[0:0].copy()
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def apply_monotonicity_screen(
    df: pd.DataFrame,
    mix_col: str = DEFAULT_MIX_COL,
    time_col: str = DEFAULT_TIME_COL,
    mean_col: str = DEFAULT_MEAN_COL,
) -> pd.DataFrame:
    """Remove later-age results whose mean falls below an earlier age.

    Within each mix, rows are ordered by ``time_col`` and a running
    maximum of ``mean_col`` is tracked; any row below the running max of
    strictly earlier ages is dropped (concrete strength should not
    regress with curing age).
    """
    keep_mask = pd.Series(True, index=df.index)
    for _, group in df.groupby(mix_col, sort=False):
        running_max = -np.inf
        for idx in group.sort_values(time_col).index:
            mean = df.loc[idx, mean_col]
            if mean < running_max:
                keep_mask[idx] = False
            else:
                running_max = mean
    return df[keep_mask].reset_index(drop=True)
