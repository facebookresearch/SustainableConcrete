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
NUM_MATERIAL_CLASSES = 3
"""3-class Material Source dimensionality. Synced with
:data:`boxcrete.mix_naming.NUM_MATERIAL_CLASSES`."""

DEFAULT_SOURCE_KERNEL = "joint_hamming_matern"
"""Default categorical source-kernel topology used by
:func:`build_strength_kernel_for_aug_dim`. Production: a single Matérn
over a joint feature-plus-Hamming distance (one learnable
categorical-penalty α) — see :class:`JointHammingMaternKernel`.

The kernel ``K(z_i, z_j) = σ² · M_{3/2}(√(Σ_f Δx_f² / ℓ_f² + α · 1[c_i ≠ c_j]))``
combines per-feature ARD over composition/time features with a
single learnable Hamming penalty on the Material Source class —
the simplest acceptable Pareto-optimal architecture identified by
the ablation suite (full study and registered alternatives in the
companion research commit's
``experiments/THREE_CLASS_AND_PRIOR_BENCHMARK.md``).

Production supports only ``"joint_hamming_matern"``. Additional
research / ablation source-kernel topologies (e.g. categorical-product
IndexKernel, RBF-embedding, additive hybrids) are registered in the
research commit, which extends ``_SUPPORTED_SOURCE_KERNELS`` and the
``_categorical_source_branch`` dispatch.
"""

_SUPPORTED_SOURCE_KERNELS = ("joint_hamming_matern",)


class JointHammingMaternKernel(Kernel):
    r"""Single Matern kernel over a joint feature + categorical distance.

    Defines a joint squared distance combining ARD-scaled continuous
    feature distances with a categorical penalty:

    .. math::
        d^2(z_i, z_j) = \sum_{f} \frac{(x_{i,f} - x_{j,f})^2}{\ell_f^2}
        + \alpha \cdot d^2_{\text{cat}}(c_i, c_j)

    where the categorical squared distance is one of:

    * **``categorical_mode="hamming"``** (default):
      :math:`d^2_{\text{cat}}(c_i, c_j) = \mathbb{1}[c_i \ne c_j]`.
    * **``categorical_mode="chain"``**:
      :math:`d^2_{\text{cat}}(c_i, c_j) = (c_i - c_j)^2`. Encodes a
      natural ordering of the classes (e.g. mortar → Set-2 → Set-3
      along binder chemistry); cross-class similarity decays
      geometrically with the integer class-distance.

    Then applies Matern (default Matern_3/2; smoothness is selectable
    via the ``nu`` parameter):

    .. math::
        K(z_i, z_j) = M_\nu\!\big(\sqrt{d^2(z_i, z_j)}\big).
    """

    has_lengthscale = False  # we manage feature lengthscales ourselves

    def __init__(
        self,
        feature_dims: list[int],
        source_dim: int,
        nu: float = 1.5,
        ard_num_dims: int | None = None,  # unused; kept for API compat
        active_dims: torch.Tensor | None = None,
        lengthscale_constraint=None,
        lengthscale_prior=None,
        alpha_constraint=None,
        alpha_initial_value: float = 1.0,
        categorical_mode: str = "hamming",
        **kwargs,
    ) -> None:
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError(f"nu must be in {{0.5, 1.5, 2.5}}; got {nu}")
        if categorical_mode not in ("hamming", "chain"):
            raise ValueError(
                "categorical_mode must be 'hamming' or 'chain'; "
                f"got {categorical_mode!r}"
            )
        if lengthscale_constraint is None:
            lengthscale_constraint = LogTransformedInterval(
                1e-2, 1e3, initial_value=1.0
            )
        if alpha_constraint is None:
            alpha_constraint = LogTransformedInterval(
                1e-3, 1e3, initial_value=alpha_initial_value
            )
        # Pass ``ard_num_dims=None`` so GPyTorch's ARD-num-dims check
        # in ``Kernel.__call__`` doesn't fire (our input includes both
        # feature dims AND the source-class dim, which is more dims
        # than our feature ARD).
        super().__init__(
            ard_num_dims=None,
            active_dims=active_dims,
            **kwargs,
        )
        del ard_num_dims  # explicitly unused
        self.feature_dims = list(feature_dims)
        self.source_dim = int(source_dim)
        self.nu = float(nu)
        self.categorical_mode = categorical_mode
        self._n_features = len(feature_dims)

        # Per-feature lengthscale managed explicitly (not via the
        # ``has_lengthscale`` machinery) since GPyTorch's ARD check
        # collides with our mixed feature+categorical input layout.
        self.register_parameter(
            name="raw_feat_lengthscale",
            parameter=torch.nn.Parameter(torch.zeros(1, self._n_features)),
        )
        self.register_constraint("raw_feat_lengthscale", lengthscale_constraint)
        with torch.no_grad():
            init_ell = torch.full(
                (1, self._n_features),
                float(getattr(lengthscale_constraint, "initial_value", 1.0) or 1.0),
            )
            self.raw_feat_lengthscale.copy_(
                self.raw_feat_lengthscale_constraint.inverse_transform(init_ell)
            )

        # Learnable categorical-penalty alpha (positive scalar).
        self.register_parameter(
            name="raw_alpha",
            parameter=torch.nn.Parameter(torch.zeros(1)),
        )
        self.register_constraint("raw_alpha", alpha_constraint)
        with torch.no_grad():
            self.raw_alpha.copy_(
                self.raw_alpha_constraint.inverse_transform(
                    torch.tensor(alpha_initial_value)
                )
            )

        # Optional within-group / shrinkage prior on the per-feature
        # lengthscales. Mirrors what GPyTorch's standard ``MaternKernel``
        # does via the ``lengthscale_prior=`` kwarg, but since we manage
        # ``raw_feat_lengthscale`` ourselves, we register it manually.
        # The prior is evaluated on the post-transform ``self.lengthscale``
        # value (shape ``(1, n_features)``) — the same contract that
        # ``WithinGroupShrinkagePrior`` and ``ComposedLengthscalePrior``
        # in ``boxcrete.priors`` expect.
        if lengthscale_prior is not None:
            self.register_prior(
                "feat_lengthscale_prior",
                lengthscale_prior,
                lambda m: m.lengthscale,
            )

    @property
    def lengthscale(self) -> torch.Tensor:
        """Per-feature ARD lengthscales, shape (1, n_features)."""
        return self.raw_feat_lengthscale_constraint.transform(self.raw_feat_lengthscale)

    @property
    def alpha(self) -> torch.Tensor:
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    def _matern(self, d: torch.Tensor) -> torch.Tensor:
        """Apply Matern_nu to a (joint) distance tensor."""
        if self.nu == 0.5:  # pragma: no cover
            return torch.exp(-d)
        if self.nu == 1.5:
            sqrt3_d = math.sqrt(3.0) * d
            return (1.0 + sqrt3_d) * torch.exp(-sqrt3_d)
        if self.nu == 2.5:  # pragma: no cover
            sqrt5_d = math.sqrt(5.0) * d
            return (1.0 + sqrt5_d + sqrt5_d.pow(2) / 3.0) * torch.exp(-sqrt5_d)
        raise NotImplementedError(
            f"Matern nu={self.nu} not supported."
        )  # pragma: no cover

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor:
        if last_dim_is_batch:  # pragma: no cover
            raise NotImplementedError(
                "JointHammingMaternKernel does not support last_dim_is_batch."
            )
        # The active_dims slice has already been applied by the
        # base Kernel forward; ``x1`` and ``x2`` are restricted to
        # ``self.active_dims`` columns. We need to map our absolute
        # ``feature_dims`` / ``source_dim`` to positions within
        # ``self.active_dims``.
        if self.active_dims is None:
            feature_local = self.feature_dims
            source_local = self.source_dim
        else:
            absolute = self.active_dims.tolist()
            try:
                feature_local = [absolute.index(f) for f in self.feature_dims]
                source_local = absolute.index(self.source_dim)
            except ValueError as exc:  # pragma: no cover
                raise ValueError(
                    "feature_dims/source_dim not all in active_dims"
                ) from exc

        # ARD-scaled continuous-feature contribution to squared distance.
        feat1 = x1[..., feature_local]
        feat2 = x2[..., feature_local]
        # ``self.lengthscale`` has shape (1, ard_num_dims).
        ell = self.lengthscale.squeeze(0)
        scaled1 = feat1 / ell
        scaled2 = feat2 / ell
        if diag:  # pragma: no cover
            d2_feat = (scaled1 - scaled2).pow(2).sum(dim=-1)
        else:
            d2_feat = (scaled1.unsqueeze(-2) - scaled2.unsqueeze(-3)).pow(2).sum(dim=-1)

        # Categorical squared distance contribution.
        c1 = x1[..., source_local].round().long()
        c2 = x2[..., source_local].round().long()
        if self.categorical_mode == "hamming":
            if diag:  # pragma: no cover
                d2_cat = (c1 != c2).to(d2_feat.dtype)
            else:
                d2_cat = (c1.unsqueeze(-1) != c2.unsqueeze(-2)).to(d2_feat.dtype)
        else:  # "chain"  # pragma: no cover
            # Squared difference of integer labels; class-distance grows
            # quadratically with the number of class-steps along the
            # natural ordering 0 < 1 < 2 < ... < (C - 1).
            if diag:
                d2_cat = (c1 - c2).to(d2_feat.dtype).pow(2)
            else:
                d2_cat = (c1.unsqueeze(-1) - c2.unsqueeze(-2)).to(d2_feat.dtype).pow(2)

        d2_joint = d2_feat + self.alpha * d2_cat
        d_joint = d2_joint.clamp_min(1e-12).sqrt()
        return self._matern(d_joint)


def _categorical_source_branch(
    d_aug: int,
    lengthscale_lower: float,
    source_kernel: str,
) -> ScaleKernel:
    """Construct the source-aware sub-kernel of the strength GP.

    Production registers a single source-kernel topology
    ``"joint_hamming_matern"`` — a single Matern kernel over a joint
    feature-plus-Hamming distance with one learnable categorical
    penalty α. See :class:`JointHammingMaternKernel` for the math.

    Research / ablation alternative topologies (categorical-product
    IndexKernel, RBF-embedding, additive hybrids, etc.) are added in
    the companion research commit.
    """
    if source_kernel != "joint_hamming_matern":
        raise ValueError(  # pragma: no cover
            f"Unsupported source_kernel: {source_kernel!r}. "
            f"Expected one of {_SUPPORTED_SOURCE_KERNELS}."
        )

    no_source_dims = [i for i in range(_N_RAW_DIMS) if i != _SOURCE_DIM]
    extra_dims = list(range(_N_RAW_DIMS, d_aug))
    feature_dims = no_source_dims + extra_dims
    joint = JointHammingMaternKernel(
        feature_dims=feature_dims,
        source_dim=_SOURCE_DIM,
        nu=1.5,
        categorical_mode="hamming",
        ard_num_dims=len(feature_dims),
        active_dims=torch.tensor(feature_dims + [_SOURCE_DIM]),
        lengthscale_constraint=LogTransformedInterval(
            lengthscale_lower, 1e3, initial_value=1.0
        ),
        lengthscale_prior=within_group_prior(
            d_in=_N_RAW_DIMS,
            source_dim=_SOURCE_DIM,
            num_extras=len(extra_dims),
        ),
        alpha_initial_value=1.0,
    )
    return ScaleKernel(
        joint,
        outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
    )


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
        gate_tau: float = 0.10,
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
    source_kernel: str = DEFAULT_SOURCE_KERNEL,
) -> torch.nn.Module:
    """Build the V2 strength kernel adapted to the augmented input dim
    (raw composition + appended engineered features).

    Returns an additive composition of three subkernels::

        blind_matern(no_source_dims + extras)
            + categorical_source_branch(source_dim, no_source_dims + extras)
            + additive_rbf_time(time_only)

    Production source kernel is ``"joint_hamming_matern"`` (a single
    Matern kernel over a joint feature-plus-Hamming distance with one
    learnable categorical penalty α). The within-group prior
    (Cement/FA/Slag tied, Fine/Coarse Aggregate tied) is installed on
    the ``blind`` Matern's lengthscales and on the joint kernel's
    feature lengthscales.

    ``lengthscale_lower`` (default 1e-2) controls the lengthscale lower
    constraint. The HRWR/binder ablation found that some engineered
    features rail at this bound under the default; pass ``1e-4`` to
    give the optimiser more room.

    Args:
        d_aug: post-feature-append input dim.
        lengthscale_lower: ARD lengthscale lower constraint.
        source_kernel: categorical source-kernel topology
            (see :func:`_categorical_source_branch`).
    """
    no_source_dims = [i for i in range(_N_RAW_DIMS) if i != _SOURCE_DIM]
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
    specific = _categorical_source_branch(
        d_aug=d_aug,
        lengthscale_lower=lengthscale_lower,
        source_kernel=source_kernel,
    )
    time_branch = additive_time_kernel(_N_RAW_DIMS)

    return blind + specific + time_branch


def make_gated_strength_kernel_builder(
    gate_tau: float = 0.10,
    source_kernel: str = DEFAULT_SOURCE_KERNEL,
):
    """Returns a `(d_aug) -> Kernel` builder that produces the V2
    strength kernel (see :func:`build_strength_kernel_for_aug_dim`)
    wrapped in a :class:`TimeGatedKernel`. The gate makes the prior
    variance at t=0 exactly zero, structurally enforcing f(x, 0) = 0
    without the need for day-zero anchor pseudo-observations.

    Args:
        gate_tau: time-gate timescale; see :class:`TimeGatedKernel`.
        source_kernel: categorical source-kernel topology; see
            :func:`_categorical_source_branch` for supported values.
            Default is :data:`DEFAULT_SOURCE_KERNEL`.

    The `time_idx` for the gate is the time dim (``IDX["time"]``), which is
    where the time column sits in the post-input-transform vector. After
    the engineered input transform, t=0 maps to 0 in this column (true
    for both the `Normalize`-on-time and `skip_time_in_normalize` paths).
    """

    def _builder(d_aug: int) -> torch.nn.Module:
        base = build_strength_kernel_for_aug_dim(
            d_aug,
            source_kernel=source_kernel,
        )
        return TimeGatedKernel(
            base,
            time_idx=IDX["time"],
            gate_tau=gate_tau,
        )

    return _builder


__all__ = [
    "DEFAULT_SOURCE_KERNEL",
    "JointHammingMaternKernel",
    "NUM_MATERIAL_CLASSES",
    "TimeGatedKernel",
    "additive_time_kernel",
    "ard_matern_with_within_group_prior",
    "build_strength_kernel_for_aug_dim",
    "make_gated_strength_kernel_builder",
]
