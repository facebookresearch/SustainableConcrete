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

    Implementation pattern: use a non-batched placeholder
    ``HomoskedasticNoise`` so SingleTaskGP
    sees a scalar-noise model, and override ``_shaped_noise_covar`` to
    return ``diag(h(t_i)² σ²_global)``. ``train_times`` are stashed for
    the MLL path (where GPyTorch doesn't pass X to the likelihood).
    """

    def __init__(
        self,
        time_idx: int = 9,
        gate_tau: float = 0.1,
        noise_constraint=None,
        noise_prior=None,
        **kwargs,
    ):
        if noise_constraint is None:
            # ``pragma: no cover`` -- production callers always pass
            # an explicit ``noise_constraint`` via
            # ``fit_strength_gp``; this default-
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
        (raw t ≥ 1 day, gate_tau = 0.10), ``h(raw_t / 0.10)`` saturates
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
    "PerClassGatedGaussianLikelihood",
]


class PerClassGatedGaussianLikelihood(_GaussianLikelihoodBase):
    r"""Per-class heteroscedastic gated Gaussian likelihood.

    Like :class:`GatedGaussianLikelihood`, but with **C separate
    learnable noise variances** :math:`\sigma_c^2` (one per categorical
    class):

    .. math::
        \mathrm{Var}[\,y(x_i, t_i)\,] = h(t_i)^2 \cdot \sigma_{c(i)}^2

    where :math:`c(i)` is the class label of row :math:`i` (read from
    ``X[..., source_idx]``). The time gate :math:`h(t)` is unchanged.

    Motivated by the noise audit (see ``experiments/NOISE_AUDIT.md``):
    the homoscedastic ``GatedGaussianLikelihood`` fits :math:`\sigma_gp
    \approx 370` psi globally, which is ~5× larger than Set-3's actual
    measurement noise (~76 psi). Per-class :math:`\sigma_c` lets the
    optimiser separately calibrate each class's noise floor.

    Args:
        num_classes: number of categorical classes (typically 3 for v5).
        source_idx: index of the source-class column in the input
            tensor (typically 7 = ``IDX["source"]``).
        time_idx: index of the time column. Default 9 = ``IDX["time"]``.
        gate_tau: time-gate timescale (same as ``GatedGaussianLikelihood``).
        noise_constraint: optional GPyTorch constraint applied to each
            per-class :math:`\sigma_c`.
        noise_prior: optional GPyTorch prior applied to the per-class
            noise tensor (shape ``(num_classes,)``).
    """

    def __init__(
        self,
        num_classes: int,
        source_idx: int,
        time_idx: int = 9,
        gate_tau: float = 0.05,
        noise_constraint=None,
        noise_prior=None,
        **kwargs,
    ):
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if noise_constraint is None:
            noise_constraint = LogTransformedInterval(1e-6, 1.0, initial_value=1e-1)
        # Placeholder HomoskedasticNoise so _GaussianLikelihoodBase
        # is happy. We don't use it; per-class noise is stored
        # separately and assembled in _shaped_noise_covar.
        noise_covar = HomoskedasticNoise(noise_constraint=noise_constraint)
        super().__init__(noise_covar=noise_covar)
        self.num_classes = int(num_classes)
        self.source_idx = int(source_idx)
        self.time_idx = int(time_idx)
        self.register_buffer(
            "_gate_log_tau",
            torch.tensor(math.log(gate_tau), dtype=torch.double),
        )
        # Per-class noise: register one parameter of shape (C,).
        # Initialise to log(0.1) ≈ -2.30 so softplus/exp maps to ~0.1
        # — same neighbourhood as GatedGaussianLikelihood's init.
        init_log = math.log(0.1)
        self.register_parameter(
            "raw_per_class_noise",
            torch.nn.Parameter(
                torch.full((num_classes,), init_log, dtype=torch.double)
            ),
        )
        self.register_constraint("raw_per_class_noise", noise_constraint)
        if noise_prior is not None:
            self.register_prior(
                "per_class_noise_prior",
                noise_prior,
                lambda m: m.per_class_noise,
            )

    @property
    def gate_tau(self) -> torch.Tensor:
        return torch.exp(self._gate_log_tau)

    @property
    def per_class_noise(self) -> torch.Tensor:
        """Returns per-class noise variances (shape ``(C,)``)."""
        return self.raw_per_class_noise_constraint.transform(self.raw_per_class_noise)

    @property
    def noise(self) -> torch.Tensor:
        """Mean of per-class noise — for backward compatibility with
        APIs that read ``likelihood.noise`` (e.g. notebooks)."""
        return self.per_class_noise.mean().unsqueeze(0)

    def set_train_inputs(self, X: torch.Tensor) -> None:
        """Stash the full training input tensor so the MLL path
        (which doesn't pass X to the likelihood) can compute per-row
        class indices AND time gating."""
        self._train_X = X.detach().clone().to(dtype=torch.double)
        self._train_times = (
            X[..., self.time_idx].detach().clone().to(dtype=torch.double)
        )

    def set_train_times(self, time_values: torch.Tensor) -> None:
        """Backward-compat alias for two-phase MLL contract; only
        time values are stashed (used when classes are not yet
        available). Prefer :meth:`set_train_inputs` once X is built."""
        self._train_times = time_values.detach().clone().to(dtype=torch.double)

    def _gate(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.exp(-t.clamp_min(0.0) / self.gate_tau.to(t))

    def _shaped_noise_covar(self, base_shape, *params, **kwargs):
        # Determine which (t, c) vectors to use.
        if params and hasattr(params[0], "shape") and params[0].dim() >= 2:
            X = params[0]
            t = X[..., self.time_idx]
            c = X[..., self.source_idx].round().long().clamp(0, self.num_classes - 1)
        elif getattr(self, "_train_X", None) is not None:
            X = self._train_X
            t = X[..., self.time_idx]
            c = X[..., self.source_idx].round().long().clamp(0, self.num_classes - 1)
        else:
            # Fallback: no class info — use mean per-class noise as
            # scalar (matches GatedGaussianLikelihood behaviour).
            return super()._shaped_noise_covar(base_shape, *params, **kwargs)

        h = self._gate(t)
        h2 = h * h  # element-wise [n]
        per_class_var = self.per_class_noise  # shape (C,)
        # Look up sigma^2_c for each row's class.
        sigma2_row = per_class_var[c.flatten()]
        per_row_var = sigma2_row * h2.flatten()
        n = int(base_shape[-1])
        if per_row_var.shape[0] >= n:
            per_row_var = per_row_var[:n]
        else:  # pragma: no cover - defensive padding
            mean_var = per_class_var.mean()
            pad = mean_var.expand(n - per_row_var.shape[0])
            per_row_var = torch.cat([per_row_var, pad], dim=0)
        return DiagLinearOperator(per_row_var)
