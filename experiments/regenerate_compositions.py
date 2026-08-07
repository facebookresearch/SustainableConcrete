#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Rebuild ``docs/model/compositions.json`` (the explorer candidate catalog).

Ported from the deleted ``scripts/export_model.py::export_compositions``
(recoverable at git ``711ac20``) and adapted to the current 3-class stack.

The catalog drives the explorer's scatter plot and slider bounds. This
script rebuilds it from scratch from the *current* ``data/boxcrete_data.csv``
so all three Material Source classes (0/1/2) appear and ``slider_bounds``
matches the deployed model.

What this script populates directly (Python):
  - ``column_names``      : composition columns (without Time).
  - ``compositions``      : unique 9-dim candidate compositions
                            (7 ingredients + Material Source + Temp).
  - ``cost_predictions``  : model-space (-cost) predictions from the
                            class-agnostic ``DEFAULT_COST_COEFFICIENTS``.
  - ``observations``      : per-candidate real ``[time, strength]`` points.
  - ``slider_bounds``     : per-column min/max from the data.
  - ``n_compositions``    : candidate count.

What this script leaves as placeholders (overwritten by the JS step
``experiments/regenerate_compositions_strength_predictions.mjs``, which
runs the deployed JS model):
  - ``strength_predictions`` : {"1": [...], "28": [...]}
  - ``gwp_predictions``      : [...]
  - ``pareto_mask``          : [...]

Run via ``bash experiments/regenerate_all_artifacts.sh`` so the JS step
fills in the strength/GWP/pareto fields afterward.
"""

from __future__ import annotations

import json
import os

import torch

from boxcrete.utils import (
    DEFAULT_COST_COEFFICIENTS,
    load_concrete_strength,
    make_linear_coefficients,
    REPO_DIR,
)

OUTPUT_DIR = os.path.join(REPO_DIR, "docs", "model")

# Days at which the JS step recomputes strength predictions. Must match the
# keys the ``.mjs`` regenerator expects (it reads these keys to know which
# days to predict).
STRENGTH_DAYS = [1, 28]


def to_list(t: torch.Tensor) -> list:
    """Convert a tensor to a nested Python list."""
    return t.detach().cpu().tolist()


def build_compositions(data) -> dict:
    """Build the candidate catalog dict from the loaded dataset."""
    # Unique candidate compositions (without Time); already 3-class.
    X_gwp, _, _, _ = data.gwp_data
    compositions = X_gwp.detach()
    n = compositions.shape[0]

    col_names = data.X_columns[:-1]  # drop Time

    # Cost is class-agnostic. The cost model predicts -cost (minimize cost →
    # maximize -cost), so mirror that sign here: cost_pred = -(comp · means).
    cost_means, _ = make_linear_coefficients(col_names, DEFAULT_COST_COEFFICIENTS)
    cost_preds = -(compositions @ cost_means)

    # Per-column slider bounds from the data range. Material Source min/max
    # becomes {0, 2} automatically now that class 2 is present.
    slider_bounds = {}
    for i, col in enumerate(col_names):
        slider_bounds[col] = {
            "min": compositions[:, i].min().item(),
            "max": compositions[:, i].max().item(),
        }

    # Per-candidate real (time, strength) observations, grouped by the
    # composition index they match (for overlaying on the strength curve).
    X_str, Y_str, _, _ = data.strength_data
    X_comps_str = X_str[:, :-1]  # compositions without time
    X_time_str = X_str[:, -1]  # time column
    observations: dict[str, list] = {}
    for i in range(X_str.shape[0]):
        comp = X_comps_str[i]
        dists = (compositions - comp).abs().sum(dim=-1)
        match_idx = dists.argmin().item()
        if dists[match_idx].item() < 1e-3:  # close enough match
            key = str(match_idx)
            observations.setdefault(key, []).append(
                [X_time_str[i].item(), Y_str[i, 0].item()]
            )

    # Placeholders overwritten by the JS regenerator step.
    strength_placeholder = {str(day): [0.0] * n for day in STRENGTH_DAYS}

    return {
        "column_names": col_names,
        "compositions": to_list(compositions),
        "gwp_predictions": [0.0] * n,
        "cost_predictions": to_list(cost_preds),
        "strength_predictions": strength_placeholder,
        "pareto_mask": [False] * n,
        "slider_bounds": slider_bounds,
        "n_compositions": n,
        "observations": observations,
    }


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data …")
    data = load_concrete_strength()

    print("Building candidate catalog …")
    params = build_compositions(data)

    ms_idx = params["column_names"].index("Material Source")
    classes = sorted({round(c[ms_idx]) for c in params["compositions"]})
    print(
        f"  {params['n_compositions']} candidate compositions; "
        f"Material Source classes present: {classes}"
    )

    path = os.path.join(OUTPUT_DIR, "compositions.json")
    with open(path, "w") as f:
        json.dump(params, f, separators=(",", ":"))
    print(
        f"Wrote {path} ({os.path.getsize(path) / 1024:.0f} KB). "
        "strength_predictions/gwp_predictions/pareto_mask are placeholders; "
        "run the JS step to fill them in."
    )


if __name__ == "__main__":
    main()
