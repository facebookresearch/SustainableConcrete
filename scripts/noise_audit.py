#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Noise audit: GP fitted likelihood noise vs measurement Yvar.

For each of the top fitted source kernels, compare:
  - GP's fitted likelihood noise (sigma_gp^2) — what the model
    "absorbs" as homoscedastic noise.
  - The data's measurement variance (median Yvar) — what the
    Strength(Std) column says the true noise is.

If sigma_gp >> sqrt(median Yvar), the GP is absorbing structure as
noise. If sigma_gp ≈ sqrt(median Yvar), the noise model is calibrated.
Also splits by class to detect heteroscedasticity.

This is a free analysis — we just inspect fitted parameters on
already-trained models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.three_class_ablation import (  # noqa: E402
    _load_data,
    _stage_factory,
)

WRITEUP = REPO_ROOT / "experiments" / "NOISE_AUDIT.md"


def main() -> int:
    # Load v5 data and inspect Yvar (measurement variance) by class.
    ds, X, Y, Yvar, bounds, n_real = _load_data("v5")
    print(f"n_train = {n_real}")
    print(f"Y shape: {Y.shape}, Yvar shape: {Yvar.shape}")
    print()

    # Per-class Yvar summary. Yvar is in the SAME units as Y (psi^2);
    # sqrt(Yvar) is in psi directly. No Y_max scaling needed for the
    # data column (only the GP's fitted likelihood.noise is in
    # normalized space and needs the Y_max scale).
    source_col = 7  # _SOURCE_DIM
    classes = X[:, source_col].round().long()
    print("=== Measurement noise (sqrt(Yvar)) by class ===")
    yvar_per_class = {}
    for c in sorted(classes.unique().tolist()):
        mask = classes == c
        yv = Yvar[mask].squeeze()
        sigma_meas = yv.sqrt()
        print(
            f"  Class {c}: n={mask.sum().item()}, "
            f"sqrt(Yvar) median={sigma_meas.median().item():.0f} psi, "
            f"mean={sigma_meas.mean().item():.0f} psi, "
            f"p90={sigma_meas.quantile(0.9).item():.0f} psi"
        )
        yvar_per_class[c] = sigma_meas.median().item()

    # Y range for context (in original psi space).
    print()
    Y_max = float(Y.max().item())
    print(f"Y_max = {Y_max:.0f} psi (maxscale_zeromean normalizer)")
    print()

    # Fit each variant and inspect noise.
    variants = ["joint_hamming_matern", "rbf_embedding_d2", "hamming"]
    rows: list[dict] = []
    for vid in variants:
        print(f"\n=== Fitting {vid} for noise inspection ===", flush=True)
        factory = _stage_factory(
            source_kernel=vid,
            include_lognormal_baseline=False,
            time_tying_sigma=None,
        )
        model = factory(X, Y, Yvar, bounds, seed=0)

        # The likelihood's noise is in NORMALIZED Y-space (after
        # the maxscale_zeromean outcome transform). To compare to
        # raw measurement noise (in psi), multiply by Y_max.
        likelihood_noise_normalized = float(
            model.likelihood.noise.detach().squeeze().item()
        )
        sigma_gp_normalized = likelihood_noise_normalized ** 0.5
        sigma_gp_psi = sigma_gp_normalized * Y_max

        # Note: the GP likelihood noise is added on top of Yvar.
        # The fitted likelihood.noise is in NORMALIZED Y-space, so
        # multiplying by Y_max gives psi-units sigma_gp.
        # Yvar (from _load_data) is already in psi^2 — no scaling
        # needed for that column.
        median_sqrt_yvar_psi = float(Yvar.median().sqrt().item())

        # Per-class fitted noise: the GP has a single sigma_gp shared
        # across classes (homoscedastic). So we just report it once
        # but compare against per-class measurement noise.

        print(f"  Likelihood noise (normalized): {likelihood_noise_normalized:.6e}")
        print(f"  sigma_gp (normalized): {sigma_gp_normalized:.6e}")
        print(f"  sigma_gp (psi): {sigma_gp_psi:.0f}")
        print(f"  Median sqrt(Yvar) (psi): {median_sqrt_yvar_psi:.0f}")
        print(f"  Ratio sigma_gp / median sqrt(Yvar): "
              f"{sigma_gp_psi / median_sqrt_yvar_psi:.2f}")

        rows.append({
            "kernel": vid,
            "sigma_gp_normalized": sigma_gp_normalized,
            "sigma_gp_psi": sigma_gp_psi,
            "median_sqrt_Yvar_psi": median_sqrt_yvar_psi,
            "ratio_gp_to_yvar": sigma_gp_psi / median_sqrt_yvar_psi,
        })

    print("\n\n=== SUMMARY ===")
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    # Write the markdown writeup.
    out = []
    out.append("# Noise audit: GP fitted likelihood noise vs measurement variance")
    out.append("")
    out.append(
        "Inspects each fitted model's likelihood noise σ_gp against "
        "the data's measurement noise (median sqrt(Yvar)) from "
        "Strength(Std). If σ_gp >> sqrt(Yvar), the GP is absorbing "
        "data structure as homoscedastic noise; if σ_gp ≈ sqrt(Yvar), "
        "the noise model matches the data."
    )
    out.append("")
    out.append("## Measurement noise by class (data)")
    out.append("")
    out.append("| class | n | median sqrt(Yvar) (psi) |")
    out.append("|---|---|---|")
    for c, m in yvar_per_class.items():
        out.append(f"| {int(c)} | {int((classes == c).sum().item())} | {m:.0f} |")
    out.append("")
    out.append("## GP fitted noise vs median measurement noise")
    out.append("")
    out.append(
        "| kernel | σ_gp (psi) | median sqrt(Yvar) (psi) | "
        "ratio σ_gp / sqrt(Yvar) |"
    )
    out.append("|---|---|---|---|")
    for r in rows:
        out.append(
            f"| `{r['kernel']}` | {r['sigma_gp_psi']:.0f} | "
            f"{r['median_sqrt_Yvar_psi']:.0f} | "
            f"{r['ratio_gp_to_yvar']:.2f} |"
        )
    out.append("")
    out.append("## Interpretation")
    out.append("")
    out.append(
        "* σ_gp ≈ 0: GP noise is negligible vs measurement noise. "
        "Yvar is doing all the work. No headroom from a heteroscedastic "
        "likelihood — the data already encodes per-row noise."
    )
    out.append(
        "* σ_gp ≫ sqrt(Yvar): GP is absorbing extra noise on top of "
        "the measurement noise. Likely batch-level variance, "
        "compositional fingerprint collisions, or model "
        "misspecification. A heteroscedastic per-class noise could "
        "help if the extra variance is class-dependent."
    )
    out.append(
        "* σ_gp ≈ sqrt(Yvar): GP's noise model matches the data's "
        "measurement noise. Noise is well-calibrated; further "
        "calibration improvements need to come from the kernel."
    )
    out.append("")
    WRITEUP.write_text("\n".join(out) + "\n")
    print(f"\n[writeup] wrote {WRITEUP.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
