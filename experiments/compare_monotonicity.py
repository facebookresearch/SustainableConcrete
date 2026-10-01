"""Compare strength-curve monotonicity across model variants.

For each variant, fit on full data, predict curves at 64 log-spaced
times in [0.5, 28] days for every composition in compositions.json,
and report:
  - frac_decreasing: fraction of compositions with any decreasing interval
  - frac_oscillating: fraction with > 2 second-difference sign changes
  - max_dropdown_psi: worst single-step strength drop
  - mean_total_variation: average sum of |Δslope|

Usage:
  python experiments/compare_monotonicity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from boxcrete.utils import load_concrete_strength  # noqa: E402

# pyrefly: ignore [missing-import]
from model_variant_study import VARIANTS  # noqa: E402

DOCS = HERE.parent / "docs" / "model"
compositions = json.loads((DOCS / "compositions.json").read_text())["compositions"]

T_MIN, T_MAX = 0.5, 28.0
N_TIMES = 64
times = np.exp(np.linspace(np.log(T_MIN), np.log(T_MAX), N_TIMES))


def diagnose(variant_name: str) -> dict:
    print(f"  Fitting {variant_name} …", flush=True)
    data = load_concrete_strength()
    X, Y, Yvar, bounds = data.strength_data
    fit_fn = VARIANTS[variant_name]
    model, _ = fit_fn(X, Y, Yvar, bounds, seed=0)
    model.eval()

    # Compute block-LOO RMSE for trade-off evaluation.
    # pyrefly: ignore [missing-import]
    from model_variant_study import block_loo_metrics

    bl_metrics = block_loo_metrics(
        model,
        n_real=X.shape[0],
        n_composition_dims=9,
    )
    bloo_rmse_unscaled = bl_metrics.get("rmse", float("nan"))
    # block_loo_metrics returns RMSE in the model's outcome-transformed
    # space ([0, 1] for max-scale models). Multiply by y_max to get psi.
    # (When y_max <= 1, this is essentially a pass-through.)
    bloo_rmse = bloo_rmse_unscaled  # already in psi for our models

    # Build query: 144 compositions × 64 times = 9216 points.
    # Use the full 10-dim raw input (composition + time).
    n_comp = len(compositions)
    pred_grid = np.zeros((n_comp, N_TIMES))
    with torch.no_grad():
        for c_idx, comp in enumerate(compositions):
            full = torch.tensor(
                [comp + [t] for t in times.tolist()],
                dtype=torch.double,
            )
            posterior = model.posterior(full)
            mean = posterior.mean.squeeze().cpu().numpy()
            # Un-scale (Y_scaled * y_max for max-scale models)
            y_max = float(getattr(model, "_study_y_std", 1.0))
            y_mean = float(getattr(model, "_study_y_mean", 0.0))
            pred_grid[c_idx, :] = mean * y_max + y_mean

    # Compute monotonicity stats.
    slopes = np.diff(pred_grid, axis=1)  # [n_comp, N_TIMES-1]
    second_diffs = np.diff(slopes, axis=1)  # [n_comp, N_TIMES-2]

    # Decreasing intervals
    min_slope_per_comp = slopes.min(axis=1)
    n_with_drop = int(np.sum(min_slope_per_comp < -1.0))
    max_dropdown = float(np.abs(min_slope_per_comp.min()))

    # Inflections via sign changes (ignore tiny noise via threshold)
    EPS = 1e-3
    sd = np.where(np.abs(second_diffs) < EPS, 0.0, second_diffs)
    sd_sign = np.sign(sd)
    # Replace zeros with previous sign to make sign-change detection consistent
    sign_changes = np.zeros(n_comp, dtype=int)
    for c in range(n_comp):
        prev = 0
        for v in sd_sign[c]:
            if v != 0:
                if prev != 0 and v != prev:
                    sign_changes[c] += 1
                prev = v
    n_oscillating = int(np.sum(sign_changes > 2))
    mean_inflections = float(sign_changes.mean())

    total_variation = float(np.abs(second_diffs).sum())
    mean_tv = total_variation / n_comp

    return {
        "variant": variant_name,
        "block_loo_rmse_psi": bloo_rmse,
        "frac_decreasing": n_with_drop / n_comp,
        "frac_oscillating_gt2": n_oscillating / n_comp,
        "max_dropdown_psi": max_dropdown,
        "mean_inflections": mean_inflections,
        "mean_total_variation_psi": mean_tv,
    }


def main():
    candidates = [
        # Baselines
        "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_only",  # F5_alllog (current champion: 672, monotone)
        "B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+block_loo_only",  # F5_no_log_mat (best block-LOO but oscillating)
        # Monotonicity hinge sweep on F5_no_log_mat (the oscillating one) — directly targets the violation
        "B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+block_loo_only+mono10.0",
        "B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+block_loo_only+mono100.0",
        "B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+block_loo_only+mono1000.0",
        "B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+block_loo_only+mono10000.0",
        "B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+block_loo_only+mono100000.0",
        "B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+block_loo_only+mono1000000.0",
        # Smoothness sweep on F5_no_log_mat (already known to harm block-LOO at large lambda)
        "B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+block_loo_only+smooth1.0",
        "B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+block_loo_only+smooth100.0",
        "B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+block_loo_only+smooth1000.0",
        # Combined (smoothness + monotonicity) — best of both?
        "B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+block_loo_only+smooth100.0+mono1000.0",
        "B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+block_loo_only+smooth100.0+mono10000.0",
        "B''+F5_no_log_mat+gated_t+gated_noise+maxscale_zeromean+block_loo_only+smooth1000.0+mono10000.0",
        # Sanity: monotonicity on the already-monotone F5_alllog
        "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_only+mono100.0",
        "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_only+mono10000.0",
    ]
    print("Comparing strength-curve monotonicity across variants …\n")
    rows = []
    for name in candidates:
        try:
            rows.append(diagnose(name))
        except Exception as exc:
            print(f"  FAILED for {name}: {exc}")
            rows.append({"variant": name, "error": str(exc)})

    # Print table
    print()
    print(
        f"{'variant':80s}  {'BLOO':>5s}  {'%dec':>6s}  {'%osc':>6s}  {'maxDrop':>8s}  {'<infl>':>6s}  {'<TV>':>6s}"
    )
    print("-" * 145)
    for r in rows:
        if "error" in r:
            print(f"{r['variant']:80s}  ERROR: {r['error']}")
            continue
        print(
            f"{r['variant']:80s}  "
            f"{r['block_loo_rmse_psi']:5.0f}  "
            f"{100*r['frac_decreasing']:5.1f}%  "
            f"{100*r['frac_oscillating_gt2']:5.1f}%  "
            f"{r['max_dropdown_psi']:7.0f}  "
            f"{r['mean_inflections']:6.2f}  "
            f"{r['mean_total_variation_psi']:6.0f}"
        )

    # Save results
    out_path = HERE / "monotonicity_results.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
