#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Re-run only Part A of investigate_m80_m81_diagnose.py with the bug fix."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.investigate_m80_m81_diagnose import (  # noqa: E402
    part_a_predict_held_out_pair,
    PRE_V5_FIXTURE,
    PRE_V5_LOW,
    PRE_V5_HIGH,
    V5_DATA,
    V5_LOW,
    V5_HIGH,
)

WRITEUP = REPO_ROOT / "experiments" / "M80_M81_TWIN_DIAGNOSIS_PARTA.md"


def main() -> int:
    out: list[str] = []
    out.append("# M80/M81 twin diagnosis — Part A (corrected)")
    out.append("")
    out.append(
        "Hold both twins out, fit, and ask which is closer to the GP's "
        "best guess. Posterior is unscaled from `[0, 1]` (the V2 "
        "`maxscale_zeromean` outcome transform) back to psi."
    )
    out.append("")
    for ds_label, path, low, high in [
        ("pre-v5", PRE_V5_FIXTURE, PRE_V5_LOW, PRE_V5_HIGH),
        ("v5",     V5_DATA,        V5_LOW,    V5_HIGH),
    ]:
        for kernel in ["legacy_continuous_ard", "hamming"]:
            print(f"[predict] {ds_label} | source={kernel}")
            part_a_predict_held_out_pair(
                path, low, high, kernel, ds_label, out
            )
    WRITEUP.write_text("\n".join(out) + "\n")
    print(f"\n[writeup] wrote {WRITEUP.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
