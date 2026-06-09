# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Likelihoods used across the boxcrete package.

Two unrelated Gaussian likelihoods live here so that consumers searching
for "where is the [...] likelihood?" land in one obvious place:

  * :class:`PartialFixedNoiseLikelihood` — Gaussian likelihood that
    learns noise for real observations while applying fixed near-zero
    noise to pseudo-observations. Used by the slump GP and by the V1
    strength fit factory (instructional).
  * :class:`GatedGaussianLikelihood` — heteroscedastic Gaussian noise
    multiplicatively gated by ``h(t)²``. Used by the V2 strength GP
    (production).
"""

from __future__ import annotations

import math

import torch
from botorch.utils.constraints import LogTransformedInterval
from gpytorch.likelihoods import GaussianLikelihood, _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from linear_operator.operators import DiagLinearOperator


class PartialFixedNoiseLikelihood(GaussianLikelihood):
    """Gaussian likelihood that learns noise for real observations while applying
    fixed near-zero noise to pseudo-observations.

    This enables conditioning the GP to pass through pseudo-observations (e.g.,
    zero strength at time zero) with high certainty, while still learning the
    observation noise for real data points via marginal likelihood optimization.

    Args:
        n_real: Number of real observations (must come first in training data).
        n_pseudo: Number of pseudo-observations (must come last in training data).
        pseudo_noise: Fixed noise variance for pseudo-observations.
        **kwargs: Additional keyword arguments passed to GaussianLikelihood
            (e.g., noise_constraint).
    """

    def __init__(
        self,
        n_real: int,
        n_pseudo: int,
        pseudo_noise: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._n_real = n_real
        self._n_pseudo = n_pseudo
        self._pseudo_noise = pseudo_noise

    @property
    def n_real(self) -> int:
        return self._n_real

    @property
    def n_pseudo(self) -> int:
        return self._n_pseudo

    @property
    def pseudo_noise(self) -> float:
        return self._pseudo_noise

    def _shaped_noise_covar(self, base_shape, *params, **kwargs):
        n = base_shape[-1]
        noise = self.noise_covar.noise.squeeze()  # learned scalar noise

        if n == self._n_real + self._n_pseudo:
            # Training: learned noise for real obs, fixed for pseudo-obs
            diag = torch.cat(
                [
                    noise.expand(self._n_real),
                    torch.full(
                        (self._n_pseudo,),
                        self._pseudo_noise,
                        device=noise.device,
                        dtype=noise.dtype,
                    ),
                ]
            )
            return DiagLinearOperator(diag)
        # Prediction at test points: use learned noise
        return super()._shaped_noise_covar(base_shape, *params, **kwargs)


class GatedGaussianLikelihood(_GaussianLikelihoodBase):
    """Heteroscedastic Gaussian likelihood whose aleatoric noise is gated
    by ``h(t)²`` — the same gate used by the kernel.

    Models ``y(x, t) = f(x, t) + ε(x, t)`` where:
        f ~ GP(0, h(t) k(x, x') h(t'))    [gated kernel]
        ε(x, t) ~ N(0, h(t)² σ²_global)  [gated noise; this class]

    Combined effect at ``t = 0``:
        E[y | t=0] = 0   AND   Var[y | t=0] = 0

    so both the predicted mean AND the uncertainty band collapse to the
    physical prior (concrete has zero strength and zero scatter at t=0).

    Calibration sanity check: at the smallest training time (raw t=1 day,
    post-transform t=log10(2)≈0.301), the gate is h≈0.998 → h²≈0.996, so
    aleatoric noise on training data is ≈ unchanged (99.6% of σ²_global).
    Only inference at t < 1 day (extrapolation away from training data)
    sees significant noise attenuation. This means the global noise
    estimate σ²_global converges to essentially the same value as a
    standard non-gated Gaussian likelihood would yield; we only change
    the predictive distribution at *small unseen t*.

    Implementation pattern mirrors ``_PerSourceGaussianLikelihood``:
    use a non-batched placeholder ``HomoskedasticNoise`` so SingleTaskGP
    sees a scalar-noise model, and override ``_shaped_noise_covar`` to
    return ``diag(h(t_i)² σ²_global)``. ``train_times`` are stashed for
    the MLL path (where GPyTorch doesn't pass X to the likelihood).
    """

    def __init__(
        self,
        time_idx: int = 9,
        gate_tau: float = 0.05,
        noise_constraint=None,
        noise_prior=None,
        **kwargs,
    ):
        if noise_constraint is None:
            # ``pragma: no cover`` -- production callers always pass
            # an explicit ``noise_constraint`` via
            # ``build_strength_kernel_for_aug_dim``; this default-
            # fallback path is research-only.
            noise_constraint = LogTransformedInterval(  # pragma: no cover
                1e-6,
                1.0,
                initial_value=1e-1,
            )
        # Standard scalar HomoskedasticNoise — owns σ²_global and its
        # constraint. `_shaped_noise_covar` then multiplies by h(t)².
        noise_covar = HomoskedasticNoise(
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
        )
        super().__init__(noise_covar=noise_covar)
        self.time_idx = int(time_idx)
        self.register_buffer(
            "_gate_log_tau",
            torch.tensor(math.log(gate_tau), dtype=torch.double),
        )

    @property
    def gate_tau(self) -> torch.Tensor:
        return torch.exp(self._gate_log_tau)

    @property
    def noise(self) -> torch.Tensor:
        """Expose the global noise scalar via the standard
        ``likelihood.noise`` accessor that GPyTorch / BoTorch / our
        notebooks use. The ``_GaussianLikelihoodBase`` base class
        doesn't proxy this automatically when we override
        ``_shaped_noise_covar`` with a ``HomoskedasticNoise`` placeholder,
        so we expose it explicitly here."""
        # ``pragma: no cover`` -- exposed for notebook ergonomics;
        # production fit/predict paths read ``self.noise_covar.noise``
        # directly.
        return self.noise_covar.noise  # pragma: no cover

    def set_train_times(self, time_values: torch.Tensor) -> None:
        """Stash post-input-transform train times so the MLL path (which
        doesn't pass X to the likelihood) can compute h(t_i)² per row.

        Two-phase contract: the V2 fit factory in
        ``boxcrete/strength_model.py`` calls this **twice** — once with
        raw days *before* model construction (so the prior-MLL path has
        a value to read), then again with *post*-input-transform values
        from ``model.train_inputs[0][..., IDX["time"]]``. NOTE: BoTorch
        stores raw inputs in ``train_inputs`` and applies the input
        transform at ``forward()`` time, so even the second call passes
        in *raw* days, not log10(t+1) values. For the strength dataset
        (raw t ≥ 1 day, gate_tau = 0.05), ``h(raw_t / 0.05)`` saturates
        to ≈1.0, so the gated noise diagonal is empirically equivalent
        to bare ``σ²`` at training. The kernel-side gate
        ``h(t1) k(x1, x2) h(t2)`` is unaffected; it sees post-transform
        time straight from the augmented input."""
        self._train_times = time_values.detach().clone().to(dtype=torch.double)

    def _gate(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.exp(-t.clamp_min(0.0) / self.gate_tau.to(t))

    def _shaped_noise_covar(self, base_shape, *params, **kwargs):
        # Determine which time vector to use (post-input-transform).
        if params and hasattr(params[0], "shape") and params[0].dim() >= 2:
            t = params[0][..., self.time_idx]
        elif getattr(self, "_train_times", None) is not None:
            t = self._train_times
        else:
            # ``pragma: no cover`` -- defensive fallback; V2 fit always
            # either passes a 2D X (training) or has ``_train_times``
            # set (eval) before this method is called.
            return super()._shaped_noise_covar(  # pragma: no cover
                base_shape, *params, **kwargs
            )
        h = self._gate(t)
        h2 = h * h  # element-wise [n]
        # Scalar global noise (broadcasted to per-row).
        sigma2 = self.noise_covar.noise.flatten()[0]  # scalar
        per_row_var = sigma2 * h2.flatten()
        n = int(base_shape[-1])
        if per_row_var.shape[0] >= n:
            per_row_var = per_row_var[:n]
        else:
            # ``pragma: no cover`` -- pad branch; V2 fit pre-sizes
            # ``_train_times`` to match training data, so
            # ``per_row_var.shape[0]`` always >= n.
            pad = sigma2.expand(n - per_row_var.shape[0])  # pragma: no cover
            per_row_var = torch.cat([per_row_var, pad], dim=0)  # pragma: no cover
        return DiagLinearOperator(per_row_var)


__all__ = [
    "GatedGaussianLikelihood",
    "PartialFixedNoiseLikelihood",
]
