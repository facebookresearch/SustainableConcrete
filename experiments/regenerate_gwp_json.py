#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regenerate ``docs/model/gwp.json`` from
``boxcrete.utils.DEFAULT_GWP_COEFFICIENTS``.

The deployed GWP predictor is a per-class linear model
``GWP = sum_j c_j * x_j``. The JS predictor (``docs/gp.mjs::predictGWP``)
reads the coefficients from ``docs/model/gwp.json``.

Pre-v5 ``gwp.json`` only had classes 0 and 1. v5 introduces class 2
(Set-3 / Amrize Class-F concrete) via the data merge; the per-class
coefficients are derived from the v5 rows in ``DEFAULT_GWP_COEFFICIENTS``
(see ``scripts/derive_class_2_gwp.py``). This script regenerates
``gwp.json`` to expose all three classes to the JS port.

Note: the JS sign convention is that ``means[i] = -coefficient_i``
because the model predicts ``-GWP`` (a value to be maximised).
``variances[i] = stderr_i ** 2``.

Run::

    python experiments/regenerate_gwp_json.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from boxcrete.utils import DEFAULT_GWP_COEFFICIENTS  # noqa: E402

OUT = REPO_ROOT / "docs" / "model" / "gwp.json"

# Column order must match the JSON's ``column_names`` field and the JS
# predictor's expectations. The Material Source column is included with
# zero coefficient (it's the class label itself, not a quantity).
COLUMN_NAMES = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
    "Material Source",
    "Temp (C)",
]

# class_dim is the position of Material Source in the augmented vector
# the JS predictor receives (i.e., the same as the strength model's
# source_dim).
CLASS_DIM = 7


def main() -> int:
    coefficients_out: dict[str, dict[str, list[float]]] = {}
    for class_id, per_class in DEFAULT_GWP_COEFFICIENTS.items():
        means: list[float] = []
        variances: list[float] = []
        for col in COLUMN_NAMES:
            if col == "Material Source" or col == "Temp (C)":
                # Source class and temperature don't contribute to GWP.
                means.append(-0.0)
                variances.append(0.0)
                continue
            mean, stderr = per_class.get(col, (0.0, 0.0))
            # JS convention: means[i] = -coefficient (model predicts -GWP).
            means.append(-float(mean))
            variances.append(float(stderr) ** 2)
        coefficients_out[str(class_id)] = {"means": means, "variances": variances}

    output = {
        "coefficients": coefficients_out,
        "class_dim": CLASS_DIM,
        "column_names": COLUMN_NAMES,
    }
    OUT.write_text(json.dumps(output, separators=(",", ":")))
    print(f"[regenerate-gwp] wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"[regenerate-gwp]   classes: {sorted(coefficients_out.keys())}")
    for class_id, coeffs in sorted(coefficients_out.items()):
        print(
            f"[regenerate-gwp]   class {class_id}: "
            f"means={['{:.4f}'.format(m) for m in coeffs['means']]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
