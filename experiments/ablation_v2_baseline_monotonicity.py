"""Baseline measurement: V2 architecture (legacy_continuous_ard source kernel)
non-monotonicity metric on both v5 and pre-v5 data.

Purpose: provide a reference point for the v5 + joint_hamming_matern
non-monotonicity numbers reported in
``experiments/ABLATION_TIME_KERNEL.md``. The user asked: was V2
itself afflicted by similar non-monotonicity? If yes, the issue is
not specific to v5 / joint_hamming_matern. If no, then the new
kernel is at fault.

Comparisons (each refit from scratch with seed=0):

  v5 + joint_hamming_matern (production)
    -- this is the "current" non-monotonicity number; ~22% drop frac.
  v5 + legacy_continuous_ard (V2 source kernel on v5 data)
    -- isolates the effect of the source-kernel choice given the
       v5 data.
  pre-v5 + legacy_continuous_ard (V2 production architecture exactly)
    -- this is "what the explorer used to ship with"; the baseline
       the user is asking for.
  pre-v5 + joint_hamming_matern (joint kernel on pre-v5 data)
    -- isolates the data effect from the architecture effect.

Run:
    python experiments/ablation_v2_baseline_monotonicity.py
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from boxcrete import (  # noqa: E402
    compute_loo_cv,
    fit_strength_gp,
    load_concrete_strength,
)
from boxcrete.features import IDX  # noqa: E402
import boxcrete.kernels as _kernels  # noqa: E402

NUM_DENSE_TIMES = 100
N_CATALOG_SAMPLE = 144
PRE_V5 = REPO / "test" / "fixtures" / "boxcrete_data_pre_v5.csv"


def _curve_monotonicity_for_model(model, X_train, time_idx, Y_max):
    n_total = X_train.shape[0]
    n_eval = min(N_CATALOG_SAMPLE, n_total)
    sample_idx = (
        torch.linspace(0, n_total - 1, n_eval, dtype=torch.long)
        .clamp(0, n_total - 1)
        .unique()
    )
    times = torch.logspace(
        torch.log10(torch.tensor(0.04)),
        torch.log10(torch.tensor(28.0)),
        NUM_DENSE_TIMES,
        dtype=torch.double,
    )
    n_with_drop = 0
    max_drop_psi = 0.0
    with torch.no_grad():
        for i in sample_idx.tolist():
            row = X_train[i].clone()
            test_X = row.unsqueeze(0).expand(NUM_DENSE_TIMES, -1).clone()
            test_X[:, time_idx] = times
            post = model.posterior(test_X.unsqueeze(0))
            mean_psi = (post.mean.detach().squeeze() * Y_max).tolist()
            drops = [mean_psi[k - 1] - mean_psi[k] for k in range(1, NUM_DENSE_TIMES)]
            md = max([0.0] + drops)
            if md > 1.0:
                n_with_drop += 1
            if md > max_drop_psi:
                max_drop_psi = md
    return n_with_drop / len(sample_idx), max_drop_psi


def _loo_rmse_psi(model, Y_max):
    _preds_mean, _preds_var, residuals = compute_loo_cv(model)
    return float(residuals.pow(2).mean().sqrt().item()) * Y_max


def _bloo_rmse_psi(model, X, Y_max):
    n = X.shape[0]
    key_cols = list(range(7)) + [7, 8]
    keys = X[:, key_cols].round(decimals=2)
    seen = {}
    group_of = []
    for i in range(n):
        k = tuple(keys[i].tolist())
        if k not in seen:
            seen[k] = len(seen)
        group_of.append(seen[k])
    group_of = torch.tensor(group_of)
    _preds_mean, _preds_var, residuals = compute_loo_cv(model)
    G = max(set(group_of.tolist())) + 1
    sq = torch.zeros(G, dtype=torch.double)
    cnt = torch.zeros(G, dtype=torch.long)
    for i in range(n):
        sq[group_of[i]] += residuals[i].pow(2)
        cnt[group_of[i]] += 1
    return float((sq / cnt.clamp(min=1)).mean().sqrt().item()) * Y_max


def _load_data(dataset: str):
    """Return X, Y, Yvar, bounds for either v5 (default) or pre-v5."""
    if dataset == "v5":
        data = load_concrete_strength()
        return data.strength_data
    elif dataset == "pre-v5":
        # Use load_concrete_strength's data_path argument to load the
        # pre-v5 fixture through the canonical loader (which handles
        # NaN filtering, bounds extraction, etc.).
        data = load_concrete_strength(data_path=str(PRE_V5))
        return data.strength_data
    else:
        raise ValueError(f"unknown dataset: {dataset}")


def _fit(dataset: str, source_kernel: str):
    """Fit the strength GP with the given source_kernel on the given
    dataset. Returns (model, X, Y_max). Overrides source_kernel by
    wrapping make_gated_strength_kernel_builder."""
    torch.manual_seed(0)
    X, Y, Yvar, bounds = _load_data(dataset)

    import boxcrete.strength_model as _smm

    saved_builder = _smm.make_gated_strength_kernel_builder
    saved_default = _kernels.DEFAULT_SOURCE_KERNEL

    def patched_builder(*, gate_tau: float = _smm.GATE_TAU, **kwargs):
        return saved_builder(
            gate_tau=gate_tau,
            source_kernel=source_kernel,
            **kwargs,
        )

    try:
        _smm.make_gated_strength_kernel_builder = patched_builder
        _kernels.DEFAULT_SOURCE_KERNEL = source_kernel
        model = fit_strength_gp(X=X, Y=Y, Yvar=Yvar, X_bounds=bounds, seed=0)
    finally:
        _smm.make_gated_strength_kernel_builder = saved_builder
        _kernels.DEFAULT_SOURCE_KERNEL = saved_default
    return model, X, Y, float(Y.max().item())


def main() -> int:
    rows = [
        (
            "v5",
            "joint_hamming_matern",
            "v5 + joint_hamming_matern (current production)",
        ),
        ("v5", "legacy_continuous_ard", "v5 + V2 architecture"),
        (
            "pre-v5",
            "legacy_continuous_ard",
            "pre-v5 + V2 architecture (deployed V2 production)",
        ),
        ("pre-v5", "joint_hamming_matern", "pre-v5 + joint_hamming_matern"),
    ]
    print(
        f"{'configuration':>55} | {'LOO':>7} | {'bLOO':>7} | "
        f"{'%drop':>6} | {'max ss-drop':>11} | {'n':>4} | {'fit-s':>5}"
    )
    print("-" * 120)
    for dataset, source_kernel, label in rows:
        t0 = _time.time()
        try:
            model, X, _Y, Y_max = _fit(dataset, source_kernel)
        except Exception as e:
            print(f"{label[:55]:>55} | FAIL: {type(e).__name__}: {e}")
            continue
        t1 = _time.time()
        try:
            loo = _loo_rmse_psi(model, Y_max)
            bloo = _bloo_rmse_psi(model, X, Y_max)
            time_idx = IDX["time"]
            frac, md = _curve_monotonicity_for_model(model, X, time_idx, Y_max)
        except Exception as e:
            print(f"{label[:55]:>55} | METRICS FAIL: {type(e).__name__}: {e}")
            continue
        print(
            f"{label[:55]:>55} | {loo:>7.0f} | {bloo:>7.0f} | "
            f"{frac * 100:>5.1f}% | {md:>11.1f} | {X.shape[0]:>4} | "
            f"{t1 - t0:>5.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
