"""Ablation: time-kernel structure variants.

Investigates the user's hypothesis that the additive RBF-on-time
component is the source of strength-curve non-monotonicity in the
v5 + joint_hamming_matern model. The V2 benchmark showed RBF(t)
contributed +20 psi block-LOO when paired with F5_alllog features
on the LEGACY continuous-source kernel — but this ablation was
never re-run on the v5 + joint_hamming_matern architecture, where
the joint kernel may already capture composition × time interaction
more effectively.

Variants compared (each refit from scratch):

  rbf_time        — production default. ScaleKernel(RBF(t)).
  matern32_time   — ScaleKernel(Matern_3/2(t)). Less smooth, more
                    abrupt-decay time correlation.
  matern52_time   — ScaleKernel(Matern_5/2(t)). Smoother than RBF
                    at long range, sharper at short.
  no_time        — drop the additive time-kernel component
                    entirely; let blind+specific Materns handle time
                    via their ARD time dim.
  linear_time     — ScaleKernel(LinearKernel(t)) (i.e., dot-product
                    kernel on time). Posterior mean is monotonic in
                    t for non-negative observed strength values
                    because the kernel is positive on all t pairs
                    and the sign of the predictive contribution
                    follows the sign of (alpha · t · t_train).

For each variant: report LOO RMSE, bLOO RMSE, % catalog
compositions with any drop > 1 psi over t in [0.04, 28] d, max
single-step drop in psi.

Run:
    python experiments/ablation_time_kernel.py
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import torch
from gpytorch.kernels import (
    LinearKernel,
    MaternKernel,
    RBFKernel,
    ScaleKernel,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import boxcrete.kernels as _kernels  # noqa: E402
from boxcrete import (  # noqa: E402
    compute_loo_cv,
    fit_strength_gp,
    load_concrete_strength,
)
from boxcrete.features import IDX  # noqa: E402
from boxcrete.kernels import LogTransformedInterval  # noqa: E402

NUM_DENSE_TIMES = 100
N_CATALOG_SAMPLE = 144


def _make_time_kernel_factory(variant: str):
    """Return a function that constructs the time-only branch given d_in.
    Mirrors the signature of ``boxcrete.kernels.additive_time_kernel``."""
    if variant == "rbf_time":
        # Production default — exactly the existing additive_time_kernel.
        return _kernels.additive_time_kernel
    if variant == "matern32_time":

        def factory(d_in: int) -> ScaleKernel:
            return ScaleKernel(
                MaternKernel(
                    nu=1.5,
                    active_dims=torch.tensor([d_in - 1]),
                    ard_num_dims=1,
                    lengthscale_constraint=LogTransformedInterval(
                        1e-2, 1e3, initial_value=1.0
                    ),
                ),
                outputscale_constraint=LogTransformedInterval(
                    1e-2, 1e2, initial_value=1.0
                ),
            )

        return factory
    if variant == "matern52_time":

        def factory(d_in: int) -> ScaleKernel:
            return ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    active_dims=torch.tensor([d_in - 1]),
                    ard_num_dims=1,
                    lengthscale_constraint=LogTransformedInterval(
                        1e-2, 1e3, initial_value=1.0
                    ),
                ),
                outputscale_constraint=LogTransformedInterval(
                    1e-2, 1e2, initial_value=1.0
                ),
            )

        return factory
    if variant == "linear_time":

        def factory(d_in: int) -> ScaleKernel:
            return ScaleKernel(
                LinearKernel(active_dims=torch.tensor([d_in - 1])),
                outputscale_constraint=LogTransformedInterval(
                    1e-2, 1e2, initial_value=1.0
                ),
            )

        return factory
    if variant == "no_time":
        # Returns a zero-output ScaleKernel so the additive branch
        # contributes nothing. Use a ScaleKernel with very small
        # outputscale clamp to make it effectively zero (a true
        # `no-op` requires a structural change to make_gated_strength_kernel_builder
        # which is invasive; near-zero outputscale is functionally
        # equivalent for fit-time and predict-time).
        def factory(d_in: int) -> ScaleKernel:
            sk = ScaleKernel(
                RBFKernel(
                    active_dims=torch.tensor([d_in - 1]),
                    ard_num_dims=1,
                    lengthscale_constraint=LogTransformedInterval(
                        1e-2, 1e3, initial_value=1.0
                    ),
                ),
                outputscale_constraint=LogTransformedInterval(
                    1e-12, 1e-10, initial_value=1e-11
                ),
            )
            return sk

        return factory
    raise ValueError(f"Unknown variant: {variant}")


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


def _loo_rmse_psi(model, Y_max):
    _preds_mean, _preds_var, residuals = compute_loo_cv(model)
    return float(residuals.pow(2).mean().sqrt().item()) * Y_max


def _fit_with_time_kernel(variant: str):
    torch.manual_seed(0)
    data = load_concrete_strength()
    X, Y, Yvar, bounds = data.strength_data

    saved_factory = _kernels.additive_time_kernel
    factory = _make_time_kernel_factory(variant)
    try:
        _kernels.additive_time_kernel = factory
        model = fit_strength_gp(X=X, Y=Y, Yvar=Yvar, X_bounds=bounds, seed=0)
    finally:
        _kernels.additive_time_kernel = saved_factory
    return model, X, Y, float(Y.max().item())


def main() -> int:
    variants = [
        "rbf_time",  # production
        "matern52_time",
        "matern32_time",
        "linear_time",
        "no_time",
    ]
    print(
        f"{'variant':>15} | {'LOO RMSE':>10} | {'bLOO RMSE':>10} | "
        f"{'% drop':>7} | {'max ss-drop':>11} | {'fit-s':>6}"
    )
    print("-" * 88)
    for v in variants:
        t0 = _time.time()
        try:
            model, X, _Y, Y_max = _fit_with_time_kernel(v)
        except Exception as e:
            print(f"{v:>15} | FAIL: {e!r}")
            continue
        t1 = _time.time()
        loo = _loo_rmse_psi(model, Y_max)
        bloo = _bloo_rmse_psi(model, X, Y_max)
        time_idx = IDX["time"]
        frac_drop, max_drop = _curve_monotonicity_for_model(model, X, time_idx, Y_max)
        print(
            f"{v:>15} | {loo:>10.1f} | {bloo:>10.1f} | "
            f"{frac_drop * 100:>6.1f}% | {max_drop:>11.1f} | {t1 - t0:>6.1f}"
        )

    print()
    print("Trade-off analysis: prefer a time-kernel variant that")
    print("(i) produces fewer non-monotonic curves AND")
    print("(ii) doesn't regress LOO/bLOO RMSE materially vs RBF baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
