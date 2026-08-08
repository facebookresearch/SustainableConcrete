"""Plot v1 vs v2 learning curves.

Trains both models at multiple data sizes (n_train compositions ∈ {25, 50,
100, 144}) and plots block-LOO RMSE as a function of n_train, with error
bars over multiple subset_seed draws.

This visualizes the **sample-efficiency improvement** from v1 (legacy
production OSS model) to v2 (current champion). The largest gains show up
at small n_train, where v2's block-LOO + priors training is most effective.

Outputs:
    experiments/learning_curve_v1_vs_v2.png
    experiments/learning_curve_v1_vs_v2_cache.json   (cached fits)

Usage:
    python experiments/plot_learning_curves_v1_vs_v2.py
"""

# pyrefly: ignore [missing-import]
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "experiments"))

# pyrefly: ignore [missing-import]
from boxcrete.utils import load_concrete_strength  # noqa: E402

# pyrefly: ignore [missing-import]
from model_variant_study import VARIANTS, block_loo_metrics  # noqa: E402

# v1 = legacy production OSS model (single Matern + within-group prior +
# anchors + log-time + scalar noise). The closest registered variant is
# the un-gated baseline with within-group shrinkage prior.
V1_VARIANT = "baseline"

# v2 = current deployed champion (multi-Matern + gated kernel + gated
# noise + block-LOO+priors single-stage training).
V2_VARIANT = "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_only"

# Compositions sampled at each subset size; full data has n_compositions=144.
SUBSET_SIZES = [25, 50, 100, 144]
N_SEEDS_PER_SIZE = 3
SUBSET_SEEDS = list(range(N_SEEDS_PER_SIZE))

CACHE_FILE = REPO_DIR / "experiments" / "learning_curve_v1_vs_v2_cache.json"
OUT_FILE = REPO_DIR / "experiments" / "learning_curve_v1_vs_v2.png"


def _fit_block_loo(variant: str, subset_n: int | None, subset_seed: int) -> float:
    """Fit ``variant`` on a (possibly subsampled) data set and return
    block-LOO RMSE in psi."""
    data = load_concrete_strength()
    X, Y, Yvar, bounds = data.strength_data
    if subset_n is not None and subset_n < 144:
        # Subsample by composition (matches the deployment regime).
        # Reuse the same composition-fingerprint logic the variant study
        # uses for ``--subset_n``: hash-stable per (subset_seed, n).
        rng = np.random.default_rng(subset_seed)
        # Composition fingerprint = first 9 input dims (cement..source..temp).
        fingerprints = X[:, :9].numpy()
        unique_fps, inverse = np.unique(fingerprints, axis=0, return_inverse=True)
        n_unique = unique_fps.shape[0]
        keep_n = min(subset_n, n_unique)
        keep_idxs = rng.choice(n_unique, size=keep_n, replace=False)
        keep_mask = np.isin(inverse, keep_idxs)
        X_sub = X[keep_mask]
        Y_sub = Y[keep_mask]
        Yvar_sub = Yvar[keep_mask] if Yvar is not None else None
    else:
        X_sub, Y_sub, Yvar_sub = X, Y, Yvar

    fit_fn = VARIANTS[variant]
    model, _ = fit_fn(X_sub, Y_sub, Yvar_sub, bounds, seed=0)
    bl_metrics = block_loo_metrics(model, n_real=X_sub.shape[0], n_composition_dims=9)
    rmse_psi = float(bl_metrics.get("rmse", float("nan")))
    return rmse_psi


def _load_or_run() -> dict:
    """Load cached learning-curve results, filling in missing combinations
    by re-fitting. Cache key: ``f"{variant}::{n}::{seed}"``."""
    cache: dict = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())

    needed_keys = []
    for variant in [V1_VARIANT, V2_VARIANT]:
        for n in SUBSET_SIZES:
            seeds = [0] if n == 144 else SUBSET_SEEDS  # full data is deterministic
            for seed in seeds:
                key = f"{variant}::{n}::{seed}"
                if key not in cache:
                    needed_keys.append((variant, n, seed, key))

    n_needed = len(needed_keys)
    print(f"Computing {n_needed} learning-curve points (cached: {len(cache)})…")
    for i, (variant, n, seed, key) in enumerate(needed_keys, 1):
        print(f"  [{i}/{n_needed}] {variant} | n_comp={n} | seed={seed} …", flush=True)
        cache[key] = _fit_block_loo(variant, n, seed)
        # Persist incrementally so an interruption doesn't lose work.
        CACHE_FILE.write_text(json.dumps(cache, indent=2))

    return cache


def _plot(cache: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=120)

    colors = {V1_VARIANT: "#888", V2_VARIANT: "#0a6"}
    labels = {
        V1_VARIANT: "v1 (legacy OSS production)\nSingle Matern + anchors + scalar noise",
        V2_VARIANT: "v2 (current champion)\nMulti-Matern + gated kernel + gated noise\n+ block-LOO+priors training",
    }

    for variant in [V1_VARIANT, V2_VARIANT]:
        means, stds, ns = [], [], []
        for n in SUBSET_SIZES:
            seeds = [0] if n == 144 else SUBSET_SEEDS
            vals = [cache.get(f"{variant}::{n}::{seed}") for seed in seeds]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if not vals:
                continue
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)
            ns.append(n)
        ns = np.array(ns)
        means = np.array(means)
        stds = np.array(stds)
        ax.errorbar(
            ns,
            means,
            yerr=stds,
            marker="o",
            markersize=8,
            linewidth=2,
            capsize=4,
            color=colors[variant],
            label=labels[variant],
        )
        # Annotate the value next to each point for readability
        for n, m in zip(ns, means):
            ax.annotate(
                f"{m:.0f}",
                (n, m),
                textcoords="offset points",
                xytext=(8, 6 if variant == V1_VARIANT else -16),
                fontsize=9,
                color=colors[variant],
                fontweight="bold",
            )

    ax.set_xlabel("Number of training compositions ($n$)", fontsize=11)
    ax.set_ylabel("Block-LOO RMSE (psi)\n← lower is better", fontsize=11)
    ax.set_title(
        "Sample-efficiency improvement: v1 → v2\n"
        "(error bars: ±1 std over 3 random subset seeds)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xscale("log")
    ax.set_xticks(SUBSET_SIZES)
    ax.set_xticklabels([str(n) for n in SUBSET_SIZES])
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # Add gap annotations between v1 and v2 at each n
    for n in SUBSET_SIZES:
        seeds = [0] if n == 144 else SUBSET_SEEDS
        v1_vals = [cache.get(f"{V1_VARIANT}::{n}::{s}") for s in seeds]
        v2_vals = [cache.get(f"{V2_VARIANT}::{n}::{s}") for s in seeds]
        v1_vals = [v for v in v1_vals if v and np.isfinite(v)]
        v2_vals = [v for v in v2_vals if v and np.isfinite(v)]
        if v1_vals and v2_vals:
            v1_mean = np.mean(v1_vals)
            v2_mean = np.mean(v2_vals)
            delta = v2_mean - v1_mean
            mid_y = (v1_mean + v2_mean) / 2
            ax.annotate(
                f"{delta:+.0f} psi",
                xy=(n, mid_y),
                xytext=(20, 0),
                textcoords="offset points",
                fontsize=9,
                fontstyle="italic",
                color="#444",
                arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5),
            )

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"\nSaved learning curve to {OUT_FILE}")


def main():
    cache = _load_or_run()
    _plot(cache)

    # Print summary table for the writeup.
    print("\n=== Block-LOO RMSE (psi) by n_train ===")
    print(f"{'n':>5}  {'v1':>10}  {'v2':>10}  {'Δ':>10}  {'speedup':>10}")
    print("-" * 55)
    for n in SUBSET_SIZES:
        seeds = [0] if n == 144 else SUBSET_SEEDS
        v1_vals = [cache[f"{V1_VARIANT}::{n}::{s}"] for s in seeds]
        v2_vals = [cache[f"{V2_VARIANT}::{n}::{s}"] for s in seeds]
        v1m = np.mean(v1_vals)
        v2m = np.mean(v2_vals)
        delta = v2m - v1m
        ratio = v1m / v2m
        print(f"{n:>5}  {v1m:>10.0f}  {v2m:>10.0f}  {delta:>+10.0f}  {ratio:>10.2f}×")


if __name__ == "__main__":
    main()
