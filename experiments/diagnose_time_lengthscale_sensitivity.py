"""Diagnostic: post-fit perturbation of time lengthscales.

Take the production v5 + joint_hamming_matern model, then artificially
inflate / deflate the time-dim lengthscales in (a) the blind Matern
ARD vector, (b) the specific (JointHammingMatern) lengthscale vector,
and (c) the additive RBF(t) lengthscale. Re-evaluate monotonicity on
the same dense time grid.

Purpose: if forcing the kernel to be "flatter in time" (longer
lengthscales) reduces non-monotonicity, then the issue is *in the
kernel-of-time component* and a tighter prior on time lengthscales
during fit would be the right mitigation. If it doesn't (or makes
things worse), the issue is elsewhere — likely in the alpha-vector
sign structure that comes from the GP marginal likelihood at the
fitted hyperparameters.

This is a *post-fit* diagnostic, NOT a refit. Doesn't require the
~80-second fit per variant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from boxcrete import fit_strength_gp, load_concrete_strength  # noqa: E402
from boxcrete.features import IDX  # noqa: E402

NUM_DENSE_TIMES = 100


def _walk_kernels(module):
    yield module
    for c in module.children():
        yield from _walk_kernels(c)


def _curve_monotonicity(model, X, time_idx, Y_max, n_eval=144):
    n_total = X.shape[0]
    n_eval = min(n_eval, n_total)
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
            row = X[i].clone()
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


def _patch_time_lengthscales(model, multiplier: float):
    """Multiply time-direction lengthscales in all matern/rbf kernels
    that have a time-dim active. Returns count of modified kernels."""
    from gpytorch.kernels import MaternKernel, RBFKernel
    from boxcrete.kernels import JointHammingMaternKernel

    time_idx = IDX["time"]
    n_changed = 0
    n_changed

    for sub in _walk_kernels(model):
        if isinstance(sub, JointHammingMaternKernel):
            # raw_feat_lengthscale shape (1, n_features); time_idx is in
            # feature_dims if present.
            if time_idx not in sub.feature_dims:
                continue
            time_pos_in_feats = sub.feature_dims.index(time_idx)
            with torch.no_grad():
                old_ell = float(sub.lengthscale[0, time_pos_in_feats].item())
                new_ell = old_ell * multiplier
                # Set the underlying raw via inverse_transform.
                ls = sub.lengthscale.clone()
                ls[0, time_pos_in_feats] = new_ell
                sub.raw_feat_lengthscale.copy_(
                    sub.raw_feat_lengthscale_constraint.inverse_transform(ls)
                )
            n_changed += 1
        elif isinstance(sub, MaternKernel):
            ad = sub.active_dims
            if ad is None or time_idx not in ad.tolist():
                continue
            time_pos = ad.tolist().index(time_idx)
            with torch.no_grad():
                old_ell = float(sub.lengthscale[0, time_pos].item())
                new_ell = old_ell * multiplier
                ls = sub.lengthscale.clone()
                ls[0, time_pos] = new_ell
                sub.raw_lengthscale.copy_(
                    sub.raw_lengthscale_constraint.inverse_transform(ls)
                )
            n_changed += 1
        elif isinstance(sub, RBFKernel):
            ad = sub.active_dims
            if ad is None or time_idx not in ad.tolist():
                continue
            # RBF(t) is scalar (1D)
            with torch.no_grad():
                old_ell = float(sub.lengthscale[0, 0].item())
                new_ell = old_ell * multiplier
                ls = sub.lengthscale.clone()
                ls[0, 0] = new_ell
                sub.raw_lengthscale.copy_(
                    sub.raw_lengthscale_constraint.inverse_transform(ls)
                )
            n_changed += 1
    return n_changed


def main() -> int:
    torch.manual_seed(0)
    print("Fitting production model (v5 + joint_hamming_matern + tau=0.10)...")
    data = load_concrete_strength()
    X, Y, Yvar, bounds = data.strength_data
    Y_max = float(Y.max().item())
    model = fit_strength_gp(X=X, Y=Y, Yvar=Yvar, X_bounds=bounds, seed=0)

    time_idx = IDX["time"]
    base_frac, base_max = _curve_monotonicity(model, X, time_idx, Y_max)
    print(
        f"\nBaseline: {base_frac * 100:.1f}% drop, max ss-drop " f"{base_max:.1f} psi"
    )

    # Save current lengthscale state so we can restore between variants.
    state = {n: p.detach().clone() for n, p in model.named_parameters()}

    print(f"\n{'time-ell × multiplier':>25} | {'%drop':>6} | " f"{'max ss-drop':>11}")
    print("-" * 60)
    for mult in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
        # Restore
        for n, p in model.named_parameters():
            p.data.copy_(state[n])
        _ = _patch_time_lengthscales(model, mult)
        frac, md = _curve_monotonicity(model, X, time_idx, Y_max)
        print(f"{mult:>25.2f} | {frac * 100:>5.1f}% | {md:>11.1f}")

    print()
    print("Interpretation:")
    print("  if higher mult -> lower %drop, the kernel-of-time IS the issue")
    print("    (a tighter prior / lower bound on time lengthscales would help).")
    print("  if mult has no effect, the issue is elsewhere")
    print("    (alpha vector sign structure, gate behavior, or composition")
    print("     extrapolation regions). In that case, only architectural")
    print("     changes (mean function, monotonicity penalty, or UI clamp)")
    print("     can help.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
