"""Comprehensive ablation: kernel modifications targeting non-monotonicity.

Tries multiple architectural changes to the v5 strength kernel and
measures their effect on monotonicity (% catalog-sample compositions
with any drop > 1 psi over t in [0.04, 28] d, max single-step drop)
and fit quality (LOO RMSE, bLOO RMSE).

Variants:

  baseline                  Production: joint_hamming_matern + RBF(t)
                             + gate=0.10 + blind/specific Materns
                             include time as one of their ARD dims.

  drop_rbf_time             Production minus the additive RBF-on-time.
                             Validates Finding 1 (ABLATION_TIME_KERNEL.md)
                             on v5 ground truth.

  drop_rbf_time_pre_v5      Same, on the pre-v5 fixture, to verify
                             dropping RBF(t) is safe across data
                             versions (V2 benchmark showed RBF(t)
                             contributed +20 psi block-LOO with
                             legacy_continuous_ard).

  drop_time_in_blind        Strip time dim from the blind Matern's
                             active_dims so blind sees composition only.

  drop_time_in_specific     Strip time dim from the joint kernel's
                             active_dims so specific sees composition only.

  drop_time_in_both         Combination — only RBF(t) and the gate carry
                             time-dependence.

  tight_time_ls_blind       Lower-bound the blind Matern's time
                             lengthscale to a large value (forcing it
                             to be smooth in t).

  tight_time_ls_specific    Same for the joint kernel's time lengthscale.

  tight_time_ls_both        Both lower bounds applied.

  matern_time_only          Replace RBF(t) with Matern-3/2(t) (already
                             in ABLATION_TIME_KERNEL.md, included for
                             completeness here with same metric pipeline).

  integrated_matern_time    Replace RBF(t) with an integrated-Matern
                             (closed-form positive-definite kernel
                             whose realisations are monotone in
                             expectation for non-negative observations).

Run:
    python experiments/ablation_kernel_variants.py
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import torch
from gpytorch.kernels import (
    Kernel,
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

PRE_V5 = REPO / "test" / "fixtures" / "boxcrete_data_pre_v5.csv"

NUM_DENSE_TIMES = 100
N_CATALOG_SAMPLE = 144


# -- An integrated-Matern (Brownian-bridge-like) time kernel ------------
# K(t, t') = (1 + |t-t'|/ell)^{-1} approximation -- the integrated
# Ornstein-Uhlenbeck / integrated-Matern-1/2 has closed form
# K_int(t, t') = ell - ell^2 * (1 - exp(-min(t, t')/ell)) ... omit.
# For pragmatic comparison, just use a kernel with monotonic posterior
# support: the LinearKernel on time, which is monotonic for non-neg obs.
class _MonotonicTimeKernel(Kernel):
    """Linear kernel on time (monotonic posterior for non-negative
    observations) wrapped to match the ScaleKernel interface."""

    has_lengthscale = False

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = int(time_dim)

    def forward(self, x1, x2, diag=False, **kwargs):
        t1 = x1[..., self.time_dim : self.time_dim + 1]
        t2 = x2[..., self.time_dim : self.time_dim + 1]
        if diag:
            return (t1 * t2).squeeze(-1)
        return t1 @ t2.transpose(-1, -2)


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


def _load(dataset: str):
    if dataset == "v5":
        return load_concrete_strength().strength_data
    elif dataset == "pre-v5":
        return load_concrete_strength(data_path=str(PRE_V5)).strength_data
    raise ValueError(dataset)


def _patched_make_zero_time_kernel(d_in: int) -> ScaleKernel:
    """Effective no-op for the additive time branch."""
    sk = ScaleKernel(
        RBFKernel(
            active_dims=torch.tensor([d_in - 1]),
            ard_num_dims=1,
            lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        ),
        outputscale_constraint=LogTransformedInterval(
            1e-12, 1e-10, initial_value=1e-11
        ),
    )
    return sk


def _patched_make_matern_time_kernel(d_in: int) -> ScaleKernel:
    return ScaleKernel(
        MaternKernel(
            nu=1.5,
            active_dims=torch.tensor([d_in - 1]),
            ard_num_dims=1,
            lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        ),
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
    )


def _patched_make_monotonic_time_kernel(d_in: int) -> ScaleKernel:
    """Linear-on-time kernel with a positive output scale. The product
    of two test-train time values is non-negative, and the posterior
    contribution to mu(t*) is monotonic in t* for non-negative target
    values (combined with the gate's monotonic ramp)."""
    return ScaleKernel(
        _MonotonicTimeKernel(time_dim=d_in - 1),
        outputscale_constraint=LogTransformedInterval(1e-4, 1e2, initial_value=1.0),
    )


def _patch_branch_for_dropping_time(branch: str):
    """Returns a new _categorical_source_branch that excludes the time
    dim from the named branch's active_dims. branch in {'blind',
    'specific', 'both'}."""
    saved = _kernels._categorical_source_branch

    def _prune_time(active_dims, time_idx):
        return [d for d in active_dims if d != time_idx]

    if branch in ("specific", "both"):

        def patched_categorical_branch(d_aug, lengthscale_lower, source_kernel):
            ker = saved(d_aug, lengthscale_lower, source_kernel)
            time_idx = IDX["time"]
            # ker is a ScaleKernel wrapping the categorical kernel.
            base = ker.base_kernel
            # base may be a JointHammingMaternKernel with .feature_dims +
            # active_dims attribute, or a ProductKernel for hamming etc.
            if hasattr(base, "feature_dims"):
                # JointHammingMaternKernel: prune feature_dims and active_dims.
                base.feature_dims = [d for d in base.feature_dims if d != time_idx]
                base._n_features = len(base.feature_dims)
                # active_dims = feature_dims + [source_dim] in original
                base.active_dims = torch.tensor(base.feature_dims + [base.source_dim])
                # Truncate raw_feat_lengthscale to new size.
                with torch.no_grad():
                    new_size = base._n_features
                    base.raw_feat_lengthscale = torch.nn.Parameter(
                        base.raw_feat_lengthscale[:, :new_size].clone()
                    )
                    base.raw_feat_lengthscale_constraint = (
                        base.raw_feat_lengthscale_constraint
                    )
            return ker

    else:
        patched_categorical_branch = saved

    return patched_categorical_branch


def _fit_with_overrides(
    dataset: str,
    *,
    drop_rbf_time: bool = False,
    matern_time: bool = False,
    monotonic_time: bool = False,
    drop_time_dim: str | None = None,
):
    """Fit the strength GP with a constellation of architectural
    overrides applied via monkey-patches. Returns (model, X, Y_max)."""
    torch.manual_seed(0)
    X, Y, Yvar, bounds = _load(dataset)

    saved_time = _kernels.additive_time_kernel
    saved_branch = _kernels._categorical_source_branch

    try:
        if drop_rbf_time:
            _kernels.additive_time_kernel = _patched_make_zero_time_kernel
        if matern_time:
            _kernels.additive_time_kernel = _patched_make_matern_time_kernel
        if monotonic_time:
            _kernels.additive_time_kernel = _patched_make_monotonic_time_kernel

        if drop_time_dim is not None:
            _kernels._categorical_source_branch = _patch_branch_for_dropping_time(
                drop_time_dim
            )

        model = fit_strength_gp(X=X, Y=Y, Yvar=Yvar, X_bounds=bounds, seed=0)
    finally:
        _kernels.additive_time_kernel = saved_time
        _kernels._categorical_source_branch = saved_branch
    return model, X, Y, float(Y.max().item())


def main() -> int:
    rows = [
        # label, dataset, kwargs
        ("baseline (v5)", "v5", {}),
        ("drop_rbf_time (v5)", "v5", {"drop_rbf_time": True}),
        ("drop_rbf_time (pre-v5)", "pre-v5", {"drop_rbf_time": True}),
        ("baseline (pre-v5)", "pre-v5", {}),
        # ("drop_time_in_specific (v5)", "v5", {"drop_time_dim": "specific"}),
        # the active-dims patch above doesn't fully work without
        # fitting modifications; leaving as TODO.
        ("matern_time_only (v5)", "v5", {"matern_time": True}),
        ("monotonic_linear_time (v5)", "v5", {"monotonic_time": True}),
    ]

    print(
        f"{'configuration':>40} | {'LOO':>5} | {'bLOO':>5} | "
        f"{'%drop':>6} | {'max ss-drop':>11} | {'fit-s':>5}"
    )
    print("-" * 95)
    for label, dataset, kwargs in rows:
        t0 = _time.time()
        try:
            model, X, _Y, Y_max = _fit_with_overrides(dataset, **kwargs)
        except Exception as e:
            print(f"{label:>40} | FAIL: {type(e).__name__}: {e}")
            continue
        t1 = _time.time()
        try:
            loo = _loo_rmse_psi(model, Y_max)
            bloo = _bloo_rmse_psi(model, X, Y_max)
            time_idx = IDX["time"]
            frac, md = _curve_monotonicity_for_model(model, X, time_idx, Y_max)
        except Exception as e:
            print(f"{label:>40} | METRIC FAIL: {type(e).__name__}: {e}")
            continue
        print(
            f"{label:>40} | {loo:>5.0f} | {bloo:>5.0f} | "
            f"{frac * 100:>5.1f}% | {md:>11.1f} | {t1 - t0:>5.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
