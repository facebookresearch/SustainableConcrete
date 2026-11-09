#!/usr/bin/env python3
"""Plot the model improvement journey from production baseline → champion.

This is an OSS-friendly educational artefact. It runs five model variants in
sequence — each adding one more design choice on top of the previous — and
produces a chart showing the cumulative descent of block-LOO RMSE (the
realistic extrapolation metric).

The five stages mirror the markdown's TL;DR decomposition (see
``experiments/STRENGTH_GP_BENCHMARK.md``):

  0. ``baseline``           Production starting point: single Matern-5/2
                            ARD over all 10 raw dims + additive RBF on time
                            + scalar Gaussian noise (with day-zero anchors).
  1. ``B''+F0``             B''-style multi-component kernel (blind +
                            source-specific Materns + RBF on time).
  2. ``B''+F3``             + 3 chemistry-ratio features (W/B, SCM frac,
                            HRWR/binder).
  3. ``B''+F5``             + 4 more chemistry features (W/C, Coarse/Fine,
                            Aggregate/Paste, maturity).
  4. ``B''+F5_alllog``      Distribution-aware log-transforms applied to
                            all five heavy-tailed features (HRWR/binder,
                            W/C, Coarse/Fine, Agg/Paste, maturity_robust).

Both single-row LOO and block-LOO are reported at each stage so the reader
sees how the two metrics diverge in places (hint: stage 0 → stage 1).

Run:
    python experiments/plot_improvement_journey.py

The fit results are cached in ``experiments/improvement_journey_cache.json``.
Re-running uses the cache and re-renders only the plot.

Output: ``experiments/improvement_journey.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "experiments"))

from boxcrete.utils import load_concrete_strength  # noqa: E402

# pyrefly: ignore [missing-import]
from model_variant_study import (  # noqa: E402
    VARIANTS,
    block_loo_metrics,
    loo_metrics,
    phantom_anchor_metrics,
)

# (label_for_axis, variant_name, "what's added in this stage" for annotations)
STAGES: list[tuple[str, str, str]] = [
    # Stages 1-2: current production-landscape baselines (anchored —
    # in-training constraint near-zero, but OOT constraint hugely
    # violated; not shippable for users submitting unseen mixes).
    (
        "Pre-shrinkage baseline\n(single Matern + log-time\n+ scalar noise,\nno within-group prior)",
        "baseline_no_prior",
        "",
    ),
    (
        "Production baseline\n(+ within-group\nshrinkage prior)",
        "baseline",
        "+ within-group\nshrinkage prior on\nbinder & aggregate ℓ",
    ),
    # Stage 3: Drop anchors, install the unified "physics constraint at t=0"
    # step: gated kernel + gated noise. Both vanish at t=0 so the FULL
    # predictive distribution (mean AND variance) collapses to (0, 0)
    # there, replacing anchor pseudo-observations with a structural
    # prior modification. This is the "first shippable model":
    # phantom-RMSE = 0 everywhere AND zero predictive variance.
    (
        "+ gated kernel + gated noise\n(structural physics:\n0 mean AND 0 var at t=0  →  shippable!)",
        "single_matern+F0+gated_t+gated_noise+maxscale_zeromean+prior+rbf_t",
        "drop anchors, add\nh(t) = 1 - exp(-t/0.05)\non BOTH kernel and noise\n+ Y/y_max + ZeroMean",
    ),
    # Stages 4-5: feature engineering on the gated foundation. Add features
    # BEFORE the source-specific Matern decomposition because the
    # source-specific component is empirically **net-negative at F0**
    # (+23 psi block-LOO regression — kernel needs features to discover
    # what's actually source-specific) but **net-positive at F5_alllog**
    # (−33 psi). Keeping the journey monotonic requires this ordering.
    (
        "+ F3 (chemistry features)\n(W/B + SCM\n+ HRWR/binder)",
        "single_matern+F3+gated_t+gated_noise+maxscale_zeromean+prior+rbf_t",
        "+ 3 chemistry features",
    ),
    (
        "+ F5_alllog\n(+ W/C, C/F, A/P,\nlog-maturity, all log)",
        "single_matern+F5_alllog+gated_t+gated_noise+maxscale_zeromean+prior+rbf_t",
        "+ 4 features\n+ log-transform 5\nheavy-tailed features",
    ),
    # Stage 6 (CHAMPION): source-specific Matern decomposition on top of
    # the gated single-Matern + features foundation. The deployed model.
    # We deliberately do NOT add a "+ block-LOO+priors training" stage
    # here even though it lowers full-data block-LOO by 8 psi — at full
    # data it regresses single-row LOO RMSE by 55 psi (533 → 588) for a
    # marginal block-LOO gain. Holistic metric trade-off favors MLL+priors.
    # The block_loo_only objective remains valuable at small data
    # (see learning curves) but is NOT the production default.
    (
        "+ source-specific Matern\n(Multi-Matern B''  →  champion)",
        "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean",
        "blind + source-specific\nMaterns (RBF(t) preserved)",
    ),
]

CACHE_FILE = REPO_DIR / "experiments" / "improvement_journey_cache.json"
OUT_FILE = REPO_DIR / "experiments" / "improvement_journey.png"
OUT_FILE_PDF = REPO_DIR / "experiments" / "improvement_journey.pdf"


def _fit_and_score(variant: str) -> dict[str, float]:
    """Fit a variant on the full dataset and return RMSE + MAE for both
    single-row LOO and block-LOO, plus phantom-anchor metrics quantifying
    physics-constraint violation at t=0 (both at training compositions
    and at random out-of-training compositions sampled in the bound box)."""
    data = load_concrete_strength()
    X, Y, Yvar, bounds = data.strength_data
    fit_fn = VARIANTS[variant]
    model, n_real = fit_fn(X, Y, Yvar, bounds, seed=0)
    loo = loo_metrics(model, n_real)
    block = block_loo_metrics(model, n_real)
    # Skip phantom metrics for variants that don't stash _study_X_train_raw
    # (e.g., the original `baseline` and `baseline_no_prior` use
    # `fit_strength_gp` which doesn't set this attribute).
    try:
        phantom = phantom_anchor_metrics(
            model,
            n_real,
            out_of_training_n=144,
            out_of_training_seed=0,
            bounds=bounds,
        )
        phantom_in_rmse = float(phantom["rmse"])
        phantom_oot_rmse = float(phantom.get("oot_rmse", float("nan")))
    except (ValueError, RuntimeError):
        # Compute the phantom-anchor RMSE manually for variants that don't
        # stash X_train_raw — fall back to using the dataset directly.
        phantom_in_rmse, phantom_oot_rmse = _compute_phantom_via_full_data(
            model,
            X,
            bounds,
        )
    return {
        "loo_rmse": float(loo["rmse"]),
        "loo_mae": float(loo["mae"]),
        "block_loo_rmse": float(block["rmse"]),
        "block_loo_mae": float(block["mae"]),
        "block_loo_cov95": float(block["coverage_95"]),
        "phantom_in_rmse": phantom_in_rmse,
        "phantom_oot_rmse": phantom_oot_rmse,
    }


def _compute_phantom_via_full_data(
    model,
    X_full,
    bounds,
) -> tuple[float, float]:
    """Phantom-anchor RMSE for models that don't stash _study_X_train_raw
    (e.g., the production-baseline `fit_strength_gp` path). We compute it
    directly using the full dataset's compositions.

    Returns (in_training_rmse, out_of_training_rmse).
    """
    # In-training: each unique composition + t=0
    fingerprint = X_full[..., :9]
    unique_comp, _ = torch.unique(fingerprint, dim=0, return_inverse=True)
    n_unique = unique_comp.shape[0]
    zero_time = torch.zeros(
        n_unique,
        1,
        dtype=unique_comp.dtype,
        device=unique_comp.device,
    )
    X_phantom = torch.cat([unique_comp, zero_time], dim=-1)

    # OOT: random compositions in bound box, t=0
    gen = torch.Generator(device=X_full.device).manual_seed(0)
    lo = bounds[0].to(X_full)
    hi = bounds[1].to(X_full)
    n_oot = 144
    u = torch.rand(
        n_oot,
        lo.shape[-1],
        dtype=X_full.dtype,
        device=X_full.device,
        generator=gen,
    )
    X_oot = lo + u * (hi - lo)
    X_oot[..., 9] = 0.0

    model.eval()
    with torch.no_grad():
        # The legacy production `fit_strength_gp` uses a `Standardize`
        # outcome transform internally, so model.posterior() returns
        # un-standardised psi predictions directly.
        try:
            mean_in = model.posterior(X_phantom).mean.squeeze(-1)
            mean_oot = model.posterior(X_oot).mean.squeeze(-1)
        except (TypeError, RuntimeError):
            return float("nan"), float("nan")
    return (
        float(mean_in.pow(2).mean().sqrt().item()),
        float(mean_oot.pow(2).mean().sqrt().item()),
    )


def _load_or_run(stages: list[tuple[str, str, str]]) -> dict[str, dict[str, float]]:
    cache: dict[str, dict[str, float]] = {}
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            cache = {}
    # Re-fit any stage whose cached entry is missing fields the plotter needs
    # (e.g. MAE was added later). This auto-migrates an old cache cleanly.
    required = {
        "loo_rmse",
        "loo_mae",
        "block_loo_rmse",
        "block_loo_mae",
        "phantom_in_rmse",
        "phantom_oot_rmse",
    }
    for _, name, _ in stages:
        cached = cache.get(name, {})
        if not required.issubset(cached):
            print(f"Fitting {name}...")
            cache[name] = _fit_and_score(name)
            CACHE_FILE.write_text(json.dumps(cache, indent=2))
    return cache


def _plot(stages: list[tuple[str, str, str]], results: dict[str, dict[str, float]]):
    n = len(stages)
    block_rmse = np.array([results[s[1]]["block_loo_rmse"] for s in stages])
    block_mae = np.array([results[s[1]]["block_loo_mae"] for s in stages])
    loo_rmse = np.array([results[s[1]]["loo_rmse"] for s in stages])
    loo_mae = np.array([results[s[1]]["loo_mae"] for s in stages])
    phantom_in = np.array(
        [results[s[1]].get("phantom_in_rmse", np.nan) for s in stages]
    )
    phantom_oot = np.array(
        [results[s[1]].get("phantom_oot_rmse", np.nan) for s in stages]
    )
    labels = [s[0] for s in stages]
    descriptions = [s[2] for s in stages]
    x = np.arange(n)

    # Two stacked subplots — separates the "errors at actual measurements"
    # story (block-LOO + single-row LOO) from the "physics-constraint
    # violation" story (phantom-anchor at t=0). They share the x-axis but
    # have very different y-ranges and complementary narratives.
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(13, 13.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2.2]},
    )

    # Visual encoding (same across subplots):
    #   Color = held-out unit (Block-LOO = dark blue; single-row LOO = orange;
    #                         physics constraint = dark green)
    #   Style = metric (RMSE = solid, MAE / OOT = dashed/dotted)
    BLOCK_COLOR = "#1f4e79"  # dark blue (the realistic extrapolation metric)
    LOO_COLOR = "#bb6643"  # dark orange (the within-curve interpolation metric)
    PHYS_COLOR = "#2d6a3e"  # dark green (the physics constraint metric)

    # ---- Top subplot: errors at actual strength measurements ----
    top_series = [
        # (values, color, linestyle, linewidth, markersize, label, label_position)
        (
            block_rmse,
            BLOCK_COLOR,
            "-",
            2.4,
            9,
            "Block-LOO RMSE  (held-out compositions \u2014 the realistic metric)",
            "top",
        ),
        (block_mae, BLOCK_COLOR, "--", 2.0, 8, "Block-LOO MAE", "bot"),
        (
            loo_rmse,
            LOO_COLOR,
            "-",
            1.6,
            7,
            "Single-row LOO RMSE  (held-out rows \u2014 confounded by time-curve leakage)",
            "top",
        ),
        (loo_mae, LOO_COLOR, "--", 1.4, 6, "Single-row LOO MAE", "bot"),
    ]
    for vals, color, ls, lw, ms, label, _pos in top_series:
        ax_top.plot(
            x,
            vals,
            color=color,
            linestyle=ls,
            linewidth=lw,
            marker="o",
            markersize=ms,
            label=label,
            alpha=0.95,
        )

    _annotate_per_stage_labels(ax_top, top_series, x, n)

    # Inter-stage Δ annotations on the headline series (Block-LOO RMSE).
    y_top_top = max(block_rmse.max(), loo_rmse.max())
    arrow_y_top = y_top_top + 75
    for i in range(1, n):
        delta = block_rmse[i] - block_rmse[i - 1]
        sign = "\u2212" if delta < 0 else "+"
        color = "#0a8a3a" if delta < 0 else "#cc5500"
        ax_top.text(
            (i - 1 + i) / 2,
            arrow_y_top,
            f"{sign}{abs(delta):.0f} psi",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=color,
        )
        # Description text. Use a white background so the data line
        # doesn't obscure short descriptions in the high-RMSE early
        # stages (where the line is close to arrow_y_top).
        ax_top.text(
            (i - 1 + i) / 2,
            arrow_y_top - 38,
            descriptions[i],
            ha="center",
            va="top",
            fontsize=8.5,
            color="#444",
            style="italic",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="none",
                alpha=0.92,
            ),
            zorder=5,
        )

    block_drop = block_rmse[0] - block_rmse[-1]
    block_pct = (block_rmse[-1] / block_rmse[0] - 1) * 100
    # Phantom-anchor RMSE drops from a non-trivial value (typically
    # ~14,953 psi OOT for the pre-shrinkage baseline) to exactly 0 at
    # the gated stages — a 100% drop, since the constraint is structural.
    phantom_oot_start = phantom_oot[0]
    phantom_oot_end = phantom_oot[-1]
    phantom_pct = (
        100.0
        if phantom_oot_start > 0 and phantom_oot_end == 0
        else (phantom_oot_end / max(phantom_oot_start, 1e-12) - 1) * 100 * -1
    )
    ax_top.set_ylabel("Error at actual strength measurements (psi)", fontsize=11.5)
    ax_top.set_title(
        "Concrete strength GP \u2014 model improvement journey\n"
        f"Block-LOO RMSE: {block_rmse[0]:.0f} \u2192 {block_rmse[-1]:.0f} psi  "
        f"({block_drop:.0f} psi, {block_pct:+.1f}%)\n"
        f"Phantom-anchor RMSE at t=0 (out-of-training): "
        f"{phantom_oot_start:.0f} \u2192 {phantom_oot_end:.0f} psi  "
        f"(\u2212{phantom_pct:.0f}%)",
        fontsize=12.5,
        fontweight="bold",
        pad=20,
    )
    # Place the legend OUTSIDE the data area (anchored just below the
    # subplot's lower-x edge) so it doesn't overlap the lines.
    ax_top.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        fontsize=9.5,
        framealpha=0.92,
        borderaxespad=0.0,
    )
    ax_top.grid(axis="y", alpha=0.25, linestyle=":")
    y_min_top = min(loo_mae.min(), block_mae.min()) - 80
    y_max_top = arrow_y_top + 50
    ax_top.set_ylim(max(0, y_min_top), y_max_top)

    # ---- Bottom subplot: physics-constraint violations ----
    # Phantom RMSEs span 0 to ~15,000 psi across stages; the dramatic
    # part is the *drop to 0* at the gated stage. Use a log y-axis so
    # the entire range is visible AND the drop to ~0 is striking
    # (line falls off the bottom). Floor values at 0.3 psi for display
    # so log(0) doesn't blow up; per-stage labels show actual values.
    PHANTOM_DISPLAY_FLOOR = 0.3
    phantom_in_clipped = np.maximum(phantom_in, PHANTOM_DISPLAY_FLOOR)
    phantom_oot_clipped = np.maximum(phantom_oot, PHANTOM_DISPLAY_FLOOR)

    bot_series = [
        (
            phantom_in_clipped,
            phantom_in,
            PHYS_COLOR,
            "-",
            2.0,
            9,
            "Constraint violation at observed compositions  (RMSE at t=0 across 144 unique training mixes)",
            "top",
        ),
        (
            phantom_oot_clipped,
            phantom_oot,
            PHYS_COLOR,
            "--",
            1.8,
            8,
            "Constraint violation at unobserved compositions  (RMSE at t=0 across 144 random mixes in bound box)",
            "bot",
        ),
    ]
    for vals_disp, _vals_actual, color, ls, lw, ms, label, _pos in bot_series:
        ax_bot.plot(
            x,
            vals_disp,
            color=color,
            linestyle=ls,
            linewidth=lw,
            marker="o",
            markersize=ms,
            label=label,
            alpha=0.95,
        )

    # Per-stage labels showing ACTUAL values (not clipped). On a log axis,
    # we use additive pixel offsets for the "above marker / below marker"
    # placement.
    BOT_LABEL_OFFSET_PT = 16  # bigger than the top-subplot offset since the
    # log axis compresses visually-close values
    for i in range(n):
        v_in_actual = phantom_in[i]
        v_oot_actual = phantom_oot[i]
        v_in_disp = phantom_in_clipped[i]
        v_oot_disp = phantom_oot_clipped[i]
        # Default placement: in-training above its marker, OOT below.
        # When the OOT value is at the display floor (i.e., 0 in physical
        # units, after gating), placing the label below would overlap the
        # x-axis. Flip it above instead so the "0" stays readable.
        oot_at_floor = v_oot_actual < 1.0
        for v_actual, v_disp, color, dy_sign in [
            (v_in_actual, v_in_disp, PHYS_COLOR, +1),
            (
                v_oot_actual,
                v_oot_disp,
                PHYS_COLOR,
                +1 if oot_at_floor else -1,
            ),
        ]:
            label_text = f"{v_actual:.0f}" if v_actual >= 1.0 else f"{v_actual:.1f}"
            ax_bot.annotate(
                label_text,
                xy=(x[i], v_disp),
                xytext=(0, dy_sign * BOT_LABEL_OFFSET_PT),
                textcoords="offset points",
                ha="center",
                va="bottom" if dy_sign > 0 else "top",
                fontsize=8.5,
                fontweight="bold",
                color=color,
                alpha=0.95,
                # White background so the "0" label doesn't visually
                # blur into the x-axis tick line / tick labels below.
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                ),
                zorder=5,
            )

    ax_bot.set_yscale("log")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(labels, fontsize=9.5)
    ax_bot.set_ylabel(
        "Constraint-violation RMSE at t=0 (psi, log scale)", fontsize=11.5
    )
    # Place the legend OUTSIDE the data area (anchored just below the
    # subplot's lower-x edge) so it doesn't overlap the lines.
    ax_bot.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=1,
        fontsize=9.5,
        framealpha=0.92,
        borderaxespad=0.0,
    )
    ax_bot.grid(axis="y", alpha=0.25, linestyle=":", which="both")
    # Y-range: from just below the floor up to slightly above max OOT,
    # with a bit of extra headroom so per-stage labels don't get clipped.
    y_min_bot = PHANTOM_DISPLAY_FLOOR * 0.4
    y_max_bot = max(phantom_oot.max(), phantom_in.max()) * 3.0
    ax_bot.set_ylim(y_min_bot, y_max_bot)

    # Shared footnote
    fig.text(
        0.01,
        0.012,
        "Top:  Block-LOO holds out *all* time-points of one composition at a time "
        "(realistic extrapolation evaluation; see \u00a72 of the benchmark study).\n"
        "      Single-row LOO holds out one measurement at random \u2014 leaves the rest "
        "of each composition's time curve in training and over-rewards models that overfit time curves.\n"
        "Bottom: Phantom-anchor RMSE at t=0 measures how well the model satisfies the physics constraint f(x, 0) = 0. "
        "Two flavours: at observed (training) compositions, and at random unobserved compositions in the bound box.\n"
        "        Anchor pseudo-observations enforce locally (small in-training error, but "
        "huge OOT error). The gated kernel enforces structurally (both errors drop to ~0).\n"
        "Feature importance ranking (leave-one-out from F5_alllog, see \u00a76.3): "
        "log_maturity_robust (\u221284 psi if dropped) > log_hrwr_binder (\u221276) > "
        "log_agg_paste (\u221266) > wb_ratio (\u221264) > scm_frac (\u221238) > "
        "log_wc_ratio (\u221233) > log_coarse_fine (\u221228). All 7 features remain necessary.",
        fontsize=8,
        style="italic",
        color="#555",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.20, hspace=0.15)
    fig.savefig(OUT_FILE, dpi=120, bbox_inches="tight")
    # Also save a vector PDF for high-quality zoom (no rasterized
    # fuzziness in the descriptions or per-point labels).
    fig.savefig(OUT_FILE_PDF, bbox_inches="tight")
    print(f"Saved to {OUT_FILE}")
    print(f"Saved to {OUT_FILE_PDF}")


def _annotate_per_stage_labels(ax, series, x, n):
    """Per-stage value labels for ALL series at EACH stage on the given
    axes. Two same-position series close in y are nudged horizontally."""
    NEAR_THRESHOLD = 50.0  # psi (slightly larger so labels don't visually merge)
    LABEL_OFFSET_PT = 16  # pixel offset above/below marker
    top_idx = [k for k, s in enumerate(series) if s[6] == "top"]
    bot_idx = [k for k, s in enumerate(series) if s[6] == "bot"]
    for i in range(n):
        for grp_idx, dy_sign in ((top_idx, +1), (bot_idx, -1)):
            if not grp_idx:
                continue
            grp_sorted = sorted(grp_idx, key=lambda k: series[k][0][i])
            v_lo = series[grp_sorted[0]][0][i]
            v_hi = series[grp_sorted[-1]][0][i]
            close = (v_hi - v_lo) < NEAR_THRESHOLD
            for rank, k in enumerate(grp_sorted):
                vals, color = series[k][0], series[k][1]
                dx, dy = 0, dy_sign * LABEL_OFFSET_PT
                ha = "center"
                if close and len(grp_sorted) >= 2:
                    dx = -26 if rank == 0 else 26
                    dy = dy_sign * 8
                    ha = "right" if rank == 0 else "left"
                ax.annotate(
                    f"{vals[i]:.0f}",
                    xy=(x[i], vals[i]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=ha,
                    va="bottom" if dy_sign > 0 else "top",
                    fontsize=8.5,
                    fontweight="bold",
                    color=color,
                    alpha=0.95,
                )


def main():
    # Verify all stages are registered
    for _, name, _ in STAGES:
        if name not in VARIANTS:
            raise RuntimeError(f"Variant {name!r} not registered — fix the stage list.")

    results = _load_or_run(STAGES)
    print()
    print(
        f"{'Stage':<35} {'LOO RMSE':>10} {'LOO MAE':>10} {'Block RMSE':>12} {'Block MAE':>11}"
    )
    print("-" * 80)
    for label, name, _ in STAGES:
        r = results[name]
        print(
            f"{label.replace(chr(10), ' ').strip():<35} "
            f"{r['loo_rmse']:>10.1f} {r['loo_mae']:>10.1f} "
            f"{r['block_loo_rmse']:>12.1f} {r['block_loo_mae']:>11.1f}"
        )
    print()
    _plot(STAGES, results)


if __name__ == "__main__":
    main()
