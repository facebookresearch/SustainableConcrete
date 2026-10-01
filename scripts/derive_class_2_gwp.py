#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Derive ``DEFAULT_GWP_COEFFICIENTS`` from per-class rows in
``data/boxcrete_data.csv`` via per-class least-squares regression.

Plan reference: ``three_class_and_lengthscale_prior.plan.md`` §"Commit 3".

Method:

  GWP_i = sum_j (EF_j * mass_ij)

For each row of class ``c``, the GWP column gives the total kg CO_2 / m^3
and the per-component masses give the regression matrix. Ordinary
least-squares yields per-component emission factors (EF_j) and their
standard errors. Run::

    python scripts/derive_class_2_gwp.py

The script prints a Python dict literal that can be pasted into
``boxcrete/utils.py::DEFAULT_GWP_COEFFICIENTS``. Re-run the derivation
after any change to ``data/boxcrete_data.csv`` rows for that class.

The derivation runs for class 0, 1, and 2 in v5 (the previous deployed
``DEFAULT_GWP_COEFFICIENTS[0]`` was fit on pooled
Mortar+Set-2-Concrete data — the v5 unpool means that fit no longer
matches the data for the now-pure-mortar class 0; re-derive to recover
the per-class linearity invariant assert in
``test/test_utils.py::test_gwp_linearity``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = REPO_ROOT / "data" / "boxcrete_data.csv"

INGREDIENT_COLS = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
]


def main() -> None:
    df = pd.read_csv(DATA_CSV)

    for cls in (0, 1, 2):
        sub = df[df["Material Source"] == cls].copy()
        sub = sub.drop_duplicates(subset=INGREDIENT_COLS).copy()
        sub["GWP"] = pd.to_numeric(sub["GWP"], errors="coerce")
        sub = sub.dropna(subset=["GWP"]).copy()

        print(
            f"[class {cls}] {len(sub)} unique-composition rows with GWP data"
        )

        X = sub[INGREDIENT_COLS].to_numpy(dtype=float)
        y = sub["GWP"].to_numpy(dtype=float)

        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        n, d = X.shape
        if rank < d:
            print(
                "[WARN] X is rank-deficient (rank={} < d={}); "
                "coefficients are unstable.".format(rank, d)
            )
        y_pred = X @ beta
        resid = y - y_pred
        dof = max(n - rank, 1)
        sigma2 = float((resid**2).sum() / dof)
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        r2 = 1.0 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()

        print(
            f"[class {cls}] R^2 = {r2:.6f}, sigma = {np.sqrt(sigma2):.4f} kg CO2/m^3"
        )
        print(
            f"[class {cls}] dof = {dof}, n = {n}, d = {d}, rank = {rank}"
        )
        # Header explaining the chemistry source.
        chem_label = {
            0: "Set 1 (Amrize cement / Class C fly ash mortar)",
            1: "Set 2 (Heidelberg cement / Class C fly ash concrete)",
            2: "Set 3 (Amrize cement / Class F fly ash concrete)",
        }[cls]
        print()
        print(f"    {cls}: {{  # Material Source {cls} - {chem_label}")
        for col, b, s in zip(INGREDIENT_COLS, beta, se):
            print(f'        "{col}": ({b:.6f}, {s:.6f}),')
        print("    },")
        print()


if __name__ == "__main__":
    main()
