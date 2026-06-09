# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Kernels for the V2 strength GP.

Public extension surface for users / variant authors who want to compose
the same building blocks the deployed V2 strength GP uses:

  * :class:`TimeGatedKernel` — wraps any GPyTorch kernel with a
    multiplicative ``h(t_i) · k(x_i, x_j) · h(t_j)`` gate.
  * :func:`ard_matern_with_within_group_prior` — Matern-5/2 ARD wrapped
    in a ScaleKernel with the within-group prior installed (the prior
    factory itself, :func:`boxcrete.priors.within_group_prior`, lives
    in :mod:`boxcrete.priors`).
  * :func:`additive_time_kernel` — standalone additive RBF on the time
    column.
  * :func:`build_strength_kernel_for_aug_dim` — the V2 strength GP's
    ``blind_matern + source_specific_matern + additive_rbf_time``
    decomposition.
  * :func:`make_gated_strength_kernel_builder` — produces a
    ``(d_aug) -> Kernel`` factory that wraps the strength kernel in
    :class:`TimeGatedKernel`.

The :class:`GatedGaussianLikelihood` (paired with the gated kernel)
lives in :mod:`boxcrete.likelihoods` alongside the package's other
likelihoods.

These are re-exported by :mod:`boxcrete` so that ``from boxcrete import
TimeGatedKernel`` works.
"""

from __future__ import annotations

import math

import torch
from botorch.utils.constraints import LogTransformedInterval
from gpytorch.kernels import (
    Kernel,
    MaternKernel,
    RBFKernel,
    ScaleKernel,
)

from boxcrete.features import IDX
from boxcrete.priors import (
    WithinGroupShrinkagePrior,
    within_group_prior,
)
from boxcrete.utils import DEFAULT_X_COLUMNS

_SOURCE_DIM = IDX["source"]
_N_RAW_DIMS = len(DEFAULT_X_COLUMNS)


def additive_time_kernel(d_in: int) -> ScaleKernel:
    """Standalone additive RBF on the last dim (time)."""
    return ScaleKernel(
        RBFKernel(
            active_dims=torch.tensor([d_in - 1]),
            ard_num_dims=1,
            lengthscale_constraint=LogTransformedInterval(1e-2, 1e3, initial_value=1.0),
        ),
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
    )


def ard_matern_with_within_group_prior(
    *,
    ard_num_dims: int,
    active_dims: torch.Tensor | None,
    prior: WithinGroupShrinkagePrior,
    initial_outputscale: float = 1.0,
    lengthscale_lower: float = 1e-2,
) -> ScaleKernel:
    """ScaleKernel(Matern-5/2 ARD) with within-group prior installed."""
    base = MaternKernel(
        nu=2.5,
        ard_num_dims=ard_num_dims,
        active_dims=active_dims,
        lengthscale_constraint=LogTransformedInterval(
            lengthscale_lower, 1e3, initial_value=1.0
        ),
        lengthscale_prior=prior,
    )
    return ScaleKernel(
        base,
        outputscale_constraint=LogTransformedInterval(
            1e-2, 1e2, initial_value=initial_outputscale
        ),
    )


class TimeGatedKernel(Kernel):
    """Multiplicatively gates a base kernel by a time-dependent function
    `h(t)` so that ``k_gated(x_i, x_j) = h(t_i) * k_base(x_i, x_j) * h(t_j)``.

    With ``h(0) = 0``, this **structurally enforces** ``f(x, 0) = 0`` (in
    expectation AND in posterior, with prior variance 0 at t=0). No
    day-zero anchor pseudo-observations are needed; the constraint is
    built into the prior. For ``t ≥ 1`` (post-input-transform value
    ≥ ~0.21), ``h(t) ≈ 1``, so the gated kernel is essentially the
    base kernel and real-data fit is preserved.

    The transition function ``h(s) = 1 - exp(-s / tau)`` (where ``s`` is
    the post-input-transform time) gives ``h(0) = 0`` exactly and
    saturates as ``s ≫ tau``. A small ``tau`` (e.g. 0.05) makes the
    gate near-1 at t=1 (post-transform ~0.21 with Normalize, ~0.30
    without), so training data is essentially unaffected.

    Args:
        base_kernel: any GPyTorch kernel.
        time_idx: index of the time column in the kernel's input
            (post-input-transform).
        gate_tau: fixed transition timescale, registered as a buffer
            (``raw_log_tau``) — **not** a learnable parameter. A
            learnable ``tau`` was explored during V2 development but
            produced no measurable block-LOO RMSE improvement and
            destabilised L-BFGS-B; the buffer path is the only one
            we ship.

    See markdown ``STRENGTH_GP_ANCHORS_STUDY.md`` §5 (Tier 3 idea 8)
    for the design rationale.
    """

    has_lengthscale = False  # delegates to base_kernel

    def __init__(
        self,
        base_kernel: Kernel,
        time_idx: int,
        gate_tau: float = 0.05,
    ):
        # No ``**kwargs`` passthrough: TimeGatedKernel does not need any
        # of GPyTorch's generic Kernel kwargs (``ard_num_dims``,
        # ``batch_shape``, ``active_dims``, lengthscale priors/constraints
        # — all of those are properties of ``base_kernel``). Forbidding
        # them at the constructor turns typos like the previously-removed
        # ``gate_learnable=True`` into hard ``TypeError`` instead of
        # silent swallowing. ``test/test_kernel_layout.py`` pins this
        # rejection.
        super().__init__()
        self.base_kernel = base_kernel
        self.time_idx = int(time_idx)
        # ``tau`` is frozen as a buffer (not a Parameter) for the V2
        # production fit. Parameterising it as a learnable scalar was
        # explored during V2 development but produced no measurable
        # block-LOO RMSE improvement and added an inner local optimum
        # that destabilised L-BFGS-B; the buffer path is the only one
        # we ship.
        log_tau_init = math.log(gate_tau)
        self.register_buffer(
            "raw_log_tau",
            torch.tensor(log_tau_init, dtype=torch.double),
        )

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.raw_log_tau)

    def _h(self, t: torch.Tensor) -> torch.Tensor:
        """Gating function ``h(t) = 1 - exp(-t / tau)``; clamped to t >= 0
        for safety (post-input-transform should already be non-negative)."""
        return 1.0 - torch.exp(-t.clamp_min(0.0) / self.tau.to(t))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        K = self.base_kernel.forward(
            x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params
        )
        h1 = self._h(x1[..., self.time_idx])
        h2 = self._h(x2[..., self.time_idx])
        if diag:
            # K shape: [..., n] — element-wise multiply by h1, h2 (same shape)
            # ``diag=True`` kernel-eval branch; BoTorch's
            # ``posterior(...).variance`` computes the full covariance
            # and extracts the diagonal, so kernel.forward is never
            # called with diag=True in the production fit path.
            return K * h1 * h2  # pragma: no cover
        # K shape: [..., n1, n2]; multiply by h1[...,n1,1] and h2[...,1,n2]
        return K * h1.unsqueeze(-1) * h2.unsqueeze(-2)


def build_strength_kernel_for_aug_dim(
    d_aug: int,
    lengthscale_lower: float = 1e-2,
) -> torch.nn.Module:
    """Build the V2 strength kernel adapted to the augmented input dim
    (raw composition + appended engineered features).

    Returns an additive composition of three subkernels::

        blind_matern(no_source_dims + extras)
            + source_specific_matern(all_orig_dims + extras)
            + additive_rbf_time(time_only)

    The within-group prior (Cement/FA/Slag tied, Fine/Coarse Aggregate
    tied) is installed on BOTH Matern subkernels but operates only on
    the original (raw) feature dims (everything in
    ``boxcrete.utils.DEFAULT_X_COLUMNS`` except ``IDX["source"]``) — the
    appended engineered features get free lengthscales (the prior
    doesn't apply to engineered ratios).

    The ``source_specific`` Matern sees all ``len(DEFAULT_X_COLUMNS)``
    raw dims including Material Source (``IDX["source"]``) so it can
    learn per-source corrections; the ``blind`` Matern excludes Material
    Source so it captures the source-agnostic part of the response surface.

    ``lengthscale_lower`` (default 1e-2) controls the lengthscale
    lower constraint. The HRWR/binder ablation found that some
    engineered features rail at this bound under the default; pass
    ``1e-4`` to give the optimiser more room.
    """
    no_source_dims = [i for i in range(_N_RAW_DIMS) if i != _SOURCE_DIM]
    all_orig_dims = list(range(_N_RAW_DIMS))
    extra_dims = list(range(_N_RAW_DIMS, d_aug))  # appended feature indices

    blind = ard_matern_with_within_group_prior(
        ard_num_dims=len(no_source_dims) + len(extra_dims),
        active_dims=torch.tensor(no_source_dims + extra_dims),
        prior=within_group_prior(
            d_in=_N_RAW_DIMS,
            source_dim=_SOURCE_DIM,
            num_extras=len(extra_dims),
        ),
        initial_outputscale=1.0,
        lengthscale_lower=lengthscale_lower,
    )
    specific = ard_matern_with_within_group_prior(
        ard_num_dims=_N_RAW_DIMS + len(extra_dims),
        active_dims=torch.tensor(all_orig_dims + extra_dims),
        prior=within_group_prior(d_in=_N_RAW_DIMS, num_extras=len(extra_dims)),
        initial_outputscale=0.5,
        lengthscale_lower=lengthscale_lower,
    )
    return blind + specific + additive_time_kernel(_N_RAW_DIMS)


def make_gated_strength_kernel_builder(
    gate_tau: float = 0.05,
):
    """Returns a `(d_aug) -> Kernel` builder that produces the V2
    strength kernel (see :func:`build_strength_kernel_for_aug_dim`)
    wrapped in a :class:`TimeGatedKernel`. The gate makes the prior
    variance at t=0 exactly zero, structurally enforcing f(x, 0) = 0
    without the need for day-zero anchor pseudo-observations.

    The `time_idx` for the gate is the time dim (``IDX["time"]``), which is
    where the time column sits in the post-input-transform vector. After
    the engineered input transform, t=0 maps to 0 in this column (true
    for both the `Normalize`-on-time and `skip_time_in_normalize` paths).
    """

    def _builder(d_aug: int) -> torch.nn.Module:
        base = build_strength_kernel_for_aug_dim(d_aug)
        return TimeGatedKernel(
            base,
            time_idx=IDX["time"],
            gate_tau=gate_tau,
        )

    return _builder


__all__ = [
    "TimeGatedKernel",
    "additive_time_kernel",
    "ard_matern_with_within_group_prior",
    "build_strength_kernel_for_aug_dim",
    "make_gated_strength_kernel_builder",
]
