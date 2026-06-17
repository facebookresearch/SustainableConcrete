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
from botorch.models.kernels import CategoricalKernel
from botorch.utils.constraints import LogTransformedInterval
from gpytorch.kernels import (
    IndexKernel,
    Kernel,
    MaternKernel,
    ProductKernel,
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
categorical-penalty α) — see :class:`JointHammingMaternKernel` and
``experiments/THREE_CLASS_AND_PRIOR_BENCHMARK.md`` for the full
ablation.

* The kernel ``K(z_i, z_j) = σ² · M_{3/2}(√(Σ_f Δx_f² / ℓ_f² + α · 1[c_i ≠ c_j]))``
  combines per-feature ARD over composition/time features with a
  single learnable Hamming penalty on the Material Source class —
  the simplest acceptable Pareto-optimal architecture identified by
  the ablation suite.
* Strictly dominates the previous ``hamming`` default on every
  in-distribution metric (LOO 510 → 495, bLOO 738 → 717, PIT-KS
  0.039 → 0.030), passes pre-registered LOCO acceptance on every
  held-out class, and is seed-deterministic.

Override via the ``source_kernel`` parameter of
:func:`build_strength_kernel_for_aug_dim` /
:func:`make_gated_strength_kernel_builder` for ablation study sub-variants
(``hamming``, ``indexkernel_r1``, ``indexkernel_r2``, ``indexkernel_r3``,
``onehot_ard``)."""

_SUPPORTED_SOURCE_KERNELS = (
    "indexkernel_r1",
    "indexkernel_r2",
    "indexkernel_r3",
    "hamming",
    "onehot_ard",
    "legacy_continuous_ard",
    "rbf_embedding_d1",
    "rbf_embedding_d2",
    "rbf_embedding_d3",
    # Joint feature + categorical Matern (one kernel, not a product):
    # K = Matern_3/2(sqrt(d^2_feat + alpha * d^2_cat)) with learnable
    # per-feature ARD lengthscales and a learnable categorical-penalty
    # alpha. Closes the joint-vs-product architectural gap to
    # legacy_continuous_ard while preserving proper categorical
    # handling.
    "joint_hamming_matern",
    "joint_chain_matern",
    "joint_hamming_matern_nu05",
    "joint_hamming_matern_nu25",
    # Joint kernel with learnable per-class embeddings replacing the
    # Hamming/chain categorical penalty. ``embedding_dim``-suffix
    # controls expressivity. d=1 is the natural categorical sibling
    # of legacy_continuous_ard (with learnable class-coordinate scalar
    # instead of integer label).
    "joint_embedding_matern_d1",
    "joint_embedding_matern_d2",
    "joint_embedding_matern_d3",
    # Additive hybrid: ScaleKernel(joint_hamming_matern) +
    # ScaleKernel(rbf_embedding_d2_product). Combines the joint
    # kernel topology (in-distribution accuracy) with the categorical
    # embedding kernel (Class-2 LOCO extrapolation). Tests whether
    # an additive composition recovers BOTH Pareto corners
    # simultaneously.
    "additive_joint_hamming_rbf_d2",
    "additive_joint_hamming_nu25_rbf_d2",
)


class RBFEmbeddingKernel(Kernel):  # pragma: no cover
    """Categorical kernel with a learned per-class embedding + RBF.

    Each of the ``num_classes`` source labels is associated with a
    free embedding vector ``x_c \\in R^{embedding_dim}``, and the
    inter-class covariance is computed as

        ``K(c_i, c_j) = exp(- ||x_{c_i} - x_{c_j}||^2 / (2 * ell^2))``.

    Embeddings and lengthscale are co-optimized with the rest of the
    kernel hyperparameters via marginal likelihood.

    Comparison with existing source kernels:

    * **Hamming** is the limiting case where embeddings sit on the
      vertices of an (n-1)-simplex with unit spacing and ``ell -> 0``
      collapses cross-class similarity to zero.
    * **IndexKernel-rank-r** parameterises ``K = B B^T`` where rows of
      ``B`` are class embeddings and similarity is the inner product.
      The RBF formulation here uses *distances* in embedding space,
      which gives strictly positive cross-class similarities in
      ``(0, 1]`` and is invariant to translations / scale-of-embedding
      (the latter absorbed into the lengthscale).

    Initialisation: embeddings are placed at the rows of an equilateral
    simplex in ``R^{embedding_dim}`` (Hamming-equivalent starting point).
    For ``embedding_dim < num_classes - 1`` the simplex is projected
    down (which warm-starts a low-rank embedding by collapsing nearby
    class differences along projected dims).

    Identifiability: the kernel is invariant to global translations
    and rotations of the embeddings. We pin ``x_0`` at the origin and
    ``x_1[1:]`` at zero to fix the gauge for ``embedding_dim >= 2``.
    For ``embedding_dim == 1`` we pin ``x_0 = 0`` and let ``x_1`` be
    free (sign-symmetry remains but does not affect the kernel value).
    """

    has_lengthscale = True

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 2,
        active_dims: torch.Tensor | None = None,
        learn_lengthscale: bool = True,
        init_strategy: str = "simplex",
        **kwargs,
    ) -> None:
        """Initialise the RBF-embedding kernel.

        Args:
            num_classes: number of categorical classes (e.g. 3 for v5).
            embedding_dim: dimension of the learnable embedding space.
            active_dims: which dim of the input tensor carries the class
                label. Forwarded to ``Kernel.__init__``.
            learn_lengthscale: if False, fix the lengthscale at 1 and
                disable its gradient. The kernel then becomes
                ``K(c_i, c_j) = exp(-||x_{c_i} - x_{c_j}||^2 / 2)``,
                with the embedding scale absorbing what the lengthscale
                would otherwise control. Removes a redundant
                degree of freedom (the scale-vs-lengthscale ridge in
                the loss surface). Default True for backward
                compatibility.
            init_strategy: one of ``"simplex"`` (default;
                equilateral-simplex projection) or ``"linear"``
                (embeddings initialised at integer class labels along
                the first axis, mimicking the legacy continuous-ARD
                treatment of source as a numeric coordinate). Only
                ``"simplex"`` is supported for ``embedding_dim >= 2``;
                ``"linear"`` is most meaningful at d=1.
        """
        super().__init__(
            ard_num_dims=None,
            active_dims=active_dims,
            lengthscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
            **kwargs,
        )
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        if init_strategy not in ("simplex", "linear"):
            raise ValueError(
                f"init_strategy must be 'simplex' or 'linear', got {init_strategy!r}"
            )
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.learn_lengthscale = learn_lengthscale
        self.init_strategy = init_strategy

        # ``num_free`` = num embedding entries that are NOT pinned.
        # Class 0 is always pinned at the origin (embedding_dim free
        # entries removed). For embedding_dim >= 2, class 1's last
        # (embedding_dim - 1) coords are pinned at 0 (an additional
        # rotation lock) — class 1 lives on the first axis only.
        if embedding_dim >= 2:
            num_pinned = embedding_dim + (embedding_dim - 1)
        else:
            num_pinned = embedding_dim  # only translation lock
        num_free_entries = num_classes * embedding_dim - num_pinned
        # Embedding warm start. Two strategies:
        #
        # ``"simplex"`` (default): rows of an equilateral simplex
        #     in R^{num_classes-1} projected to R^{embedding_dim}.
        #     For d >= num_classes-1 this is exact (Hamming-equivalent
        #     at ell = 1/sqrt(2)); for d < num_classes-1, project via
        #     SVD on the standard simplex.
        #
        # ``"linear"``: each class c is initialised at coordinate
        #     (c, 0, 0, ..., 0) along the first embedding axis. Mimics
        #     the legacy_continuous_ard treatment of source as an
        #     ordered numeric coordinate (0, 1, 2, ...). Most useful
        #     at d=1 for testing whether the simplex projection's
        #     non-monotone d=1 init (-1, 0, +1) traps the optimizer in
        #     a different basin than the legacy (0, 1, 2) init.
        if init_strategy == "linear":
            init = torch.zeros(num_classes, embedding_dim)
            init[:, 0] = torch.arange(num_classes, dtype=init.dtype)
            # Pin x_0 at origin: subtract row 0 from every row. (For
            # linear init this is a no-op since x_0 = 0 already.)
            init = init - init[0:1]
        else:
            init = self._equilateral_simplex_init(num_classes, embedding_dim)
        pin_mask = self._build_pin_mask(num_classes, embedding_dim)
        # Initialise the trainable free entries from the chosen init at
        # the unpinned positions — NOT zero (else the parameter starts
        # at the origin which collapses cross-class similarity to 1).
        free_init = init[~pin_mask]
        assert free_init.numel() == num_free_entries, (
            f"free init size {free_init.numel()} != "
            f"num_free_entries {num_free_entries}"
        )
        self.register_parameter(
            name="raw_embedding_free",
            parameter=torch.nn.Parameter(free_init.clone()),
        )
        # The init buffer carries pinned entries (zeros at gauge-fixing
        # positions) plus the chosen init values at free positions.
        # Only the pinned entries are read at forward time.
        self.register_buffer("_embedding_init", init)
        self.register_buffer("_pin_mask", pin_mask)

        # Optionally fix the lengthscale at 1 (i.e. disable its
        # gradient). Equivalent to letting the embedding scale absorb
        # the lengthscale degree of freedom, which removes the
        # ell ↔ |x| redundancy in the loss surface.
        if not learn_lengthscale:
            self.raw_lengthscale.requires_grad_(False)

    @staticmethod
    def _equilateral_simplex_init(num_classes: int, embedding_dim: int) -> torch.Tensor:
        """Equilateral simplex in ``R^{num_classes-1}``, projected to
        ``R^{embedding_dim}`` via the leading eigenvectors of its Gram
        matrix.
        """
        # Standard simplex: e_i in R^{num_classes}. Centre + drop one
        # coord -> equilateral simplex in R^{num_classes - 1}.
        e = torch.eye(num_classes)
        centred = e - e.mean(dim=0, keepdim=True)
        # SVD-project to embedding_dim dims (leading components).
        u, s, _ = torch.linalg.svd(centred, full_matrices=False)
        keep = min(embedding_dim, u.shape[-1])
        proj = u[:, :keep] * s[:keep].unsqueeze(0)
        # Normalise so that pairwise distances are ~1 (matches Hamming
        # at ell = 1 / sqrt(2)).
        scale = float((proj[1] - proj[0]).pow(2).sum().sqrt().clamp_min(1e-9).item())
        proj = proj / scale
        if embedding_dim > keep:
            pad = torch.zeros(num_classes, embedding_dim - keep)
            proj = torch.cat([proj, pad], dim=-1)
        # Pin x_0 to origin: subtract row 0 from every row.
        proj = proj - proj[0:1]
        if embedding_dim >= 2:
            # Rotate so that x_1 lies on the first axis: rotate by the
            # inverse of the rotation that sends x_1 / ||x_1|| to e_1.
            v1 = proj[1].clone()
            n1 = float(v1.norm().clamp_min(1e-9).item())
            if n1 > 1e-9:
                e1 = torch.zeros_like(v1)
                e1[0] = 1.0
                # Householder reflection sending v1 / n1 -> e1.
                u_ref = v1 / n1 - e1
                u_norm = float(u_ref.norm().clamp_min(1e-9).item())
                if u_norm > 1e-9:
                    u_ref = u_ref / u_norm
                    proj = proj - 2.0 * (proj @ u_ref).unsqueeze(-1) * u_ref
        return proj

    @staticmethod
    def _build_pin_mask(num_classes: int, embedding_dim: int) -> torch.Tensor:
        """Bool mask of shape (num_classes, embedding_dim) where True =
        pinned (not optimised).
        """
        mask = torch.zeros(num_classes, embedding_dim, dtype=torch.bool)
        # Class 0 fully pinned at the origin.
        mask[0, :] = True
        # Class 1's last (embedding_dim - 1) coords pinned at 0 to
        # remove rotation freedom (only when embedding_dim >= 2).
        if embedding_dim >= 2:
            mask[1, 1:] = True
        return mask

    @property
    def embeddings(self) -> torch.Tensor:
        """Materialise the (num_classes, embedding_dim) embedding
        matrix: pinned entries take their init values; free entries
        come from ``raw_embedding_free``.
        """
        emb = self._embedding_init.clone().to(self.raw_embedding_free)
        # Scatter free params into the unpinned positions.
        free_mask = ~self._pin_mask
        emb = emb.clone()
        # Use indexing to write — preserves the autograd graph.
        emb[free_mask] = emb[free_mask].detach() * 0.0 + self.raw_embedding_free
        return emb

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor:
        if last_dim_is_batch:
            raise NotImplementedError(
                "RBFEmbeddingKernel does not support last_dim_is_batch."
            )
        # x1, x2 contain integer-valued class labels (potentially
        # post-Normalize float). Cast to long.
        idx1 = x1.squeeze(-1).round().long().clamp_(0, self.num_classes - 1)
        idx2 = x2.squeeze(-1).round().long().clamp_(0, self.num_classes - 1)
        emb = self.embeddings  # (num_classes, embedding_dim)
        e1 = emb[idx1]  # (..., n1, embedding_dim)
        e2 = emb[idx2]  # (..., n2, embedding_dim)
        ell = self.lengthscale.squeeze(-1)  # scalar; .squeeze for any
        if diag:
            d2 = (e1 - e2).pow(2).sum(dim=-1)
        else:
            diff = e1.unsqueeze(-2) - e2.unsqueeze(-3)
            d2 = diff.pow(2).sum(dim=-1)
        return torch.exp(-0.5 * d2 / ell.pow(2))


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


class JointEmbeddingMaternKernel(Kernel):  # pragma: no cover
    r"""Joint feature + learned-class-embedding Matern kernel.

    Generalises ``JointHammingMaternKernel`` by replacing the
    Hamming/chain categorical penalty with a learnable per-class
    embedding:

    .. math::
        d^2(z_i, z_j) = \sum_{f} \frac{(x_{i,f} - x_{j,f})^2}{\ell_f^2}
        + \|\mathbf{x}_{c_i} - \mathbf{x}_{c_j}\|^2,
        \quad K(z_i, z_j) = M_\nu\!\big(\sqrt{d^2(z_i, z_j)}\big).

    The class embeddings :math:`\mathbf{x}_c \in \mathbb{R}^{embedding\_dim}`
    are co-optimised with the feature lengthscales. The embedding scale
    absorbs what ``alpha`` would do in the Hamming/chain variants
    (no separate alpha parameter).

    Gauge fixing: ``x_0`` is pinned at the origin (translation lock);
    for ``embedding_dim >= 2``, ``x_1[1:]`` is pinned at zero (rotation
    lock). Free entries are stored in a flat ``raw_embedding_free``
    parameter and scattered into the full matrix at forward time.

    At ``embedding_dim = 1``, this kernel is the natural categorical
    generalisation of ``legacy_continuous_ard``: it treats the source
    coordinate as a learnable per-class scalar with the same joint
    ARD-Matern topology that legacy uses for source-as-numeric.

    Args:
        feature_dims, source_dim, nu, active_dims: as in
            :class:`JointHammingMaternKernel`.
        num_classes: number of categorical classes.
        embedding_dim: dimensionality of the learnable per-class
            embedding (typically 1, 2, or 3 for 3-class data).
        init_strategy: ``"simplex"`` (default; equilateral-simplex
            warm start, equidistant pairwise classes) or ``"linear"``
            (init at integer class labels along axis 0).
    """

    has_lengthscale = False  # manage our own feature lengthscales

    def __init__(
        self,
        feature_dims: list[int],
        source_dim: int,
        num_classes: int,
        embedding_dim: int = 1,
        nu: float = 1.5,
        active_dims: torch.Tensor | None = None,
        lengthscale_constraint=None,
        lengthscale_prior=None,
        init_strategy: str = "simplex",
        **kwargs,
    ) -> None:
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError(f"nu must be in {{0.5, 1.5, 2.5}}; got {nu}")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        if init_strategy not in ("simplex", "linear"):
            raise ValueError(
                f"init_strategy must be 'simplex' or 'linear'; got {init_strategy!r}"
            )
        if lengthscale_constraint is None:
            lengthscale_constraint = LogTransformedInterval(
                1e-2, 1e3, initial_value=1.0
            )
        super().__init__(
            ard_num_dims=None,
            active_dims=active_dims,
            **kwargs,
        )
        self.feature_dims = list(feature_dims)
        self.source_dim = int(source_dim)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.nu = float(nu)
        self.init_strategy = init_strategy
        self._n_features = len(feature_dims)

        # Per-feature ARD lengthscales (managed manually, like
        # JointHammingMaternKernel, to avoid GPyTorch's ARD check).
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

        # Optional within-group / shrinkage prior on the per-feature
        # lengthscales. See JointHammingMaternKernel for the contract.
        if lengthscale_prior is not None:
            self.register_prior(
                "feat_lengthscale_prior",
                lengthscale_prior,
                lambda m: m.lengthscale,
            )

        # Class embeddings: same gauge-fix scheme as RBFEmbeddingKernel.
        if self.embedding_dim >= 2:
            num_pinned = self.embedding_dim + (self.embedding_dim - 1)
        else:
            num_pinned = self.embedding_dim
        num_free_entries = self.num_classes * self.embedding_dim - num_pinned

        if init_strategy == "linear":
            init = torch.zeros(self.num_classes, self.embedding_dim)
            init[:, 0] = torch.arange(self.num_classes, dtype=init.dtype)
            init = init - init[0:1]
        else:
            init = RBFEmbeddingKernel._equilateral_simplex_init(
                self.num_classes, self.embedding_dim
            )
        pin_mask = RBFEmbeddingKernel._build_pin_mask(
            self.num_classes, self.embedding_dim
        )
        free_init = init[~pin_mask]
        assert free_init.numel() == num_free_entries, (
            f"free init size {free_init.numel()} != "
            f"num_free_entries {num_free_entries}"
        )
        self.register_parameter(
            name="raw_embedding_free",
            parameter=torch.nn.Parameter(free_init.clone()),
        )
        self.register_buffer("_embedding_init", init)
        self.register_buffer("_pin_mask", pin_mask)

    @property
    def lengthscale(self) -> torch.Tensor:
        return self.raw_feat_lengthscale_constraint.transform(self.raw_feat_lengthscale)

    @property
    def embeddings(self) -> torch.Tensor:
        emb = self._embedding_init.clone().to(self.raw_embedding_free)
        free_mask = ~self._pin_mask
        emb = emb.clone()
        emb[free_mask] = emb[free_mask].detach() * 0.0 + self.raw_embedding_free
        return emb

    def _matern(self, d: torch.Tensor) -> torch.Tensor:
        if self.nu == 0.5:
            return torch.exp(-d)
        if self.nu == 1.5:
            sqrt3_d = math.sqrt(3.0) * d
            return (1.0 + sqrt3_d) * torch.exp(-sqrt3_d)
        if self.nu == 2.5:
            sqrt5_d = math.sqrt(5.0) * d
            return (1.0 + sqrt5_d + sqrt5_d.pow(2) / 3.0) * torch.exp(-sqrt5_d)
        raise NotImplementedError(f"Matern nu={self.nu} not supported.")

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor:
        if last_dim_is_batch:
            raise NotImplementedError(
                "JointEmbeddingMaternKernel does not support last_dim_is_batch."
            )
        if self.active_dims is None:
            feature_local = self.feature_dims
            source_local = self.source_dim
        else:
            absolute = self.active_dims.tolist()
            feature_local = [absolute.index(f) for f in self.feature_dims]
            source_local = absolute.index(self.source_dim)

        # Continuous-feature contribution.
        feat1 = x1[..., feature_local]
        feat2 = x2[..., feature_local]
        ell = self.lengthscale.squeeze(0)
        scaled1 = feat1 / ell
        scaled2 = feat2 / ell
        if diag:
            d2_feat = (scaled1 - scaled2).pow(2).sum(dim=-1)
        else:
            d2_feat = (scaled1.unsqueeze(-2) - scaled2.unsqueeze(-3)).pow(2).sum(dim=-1)

        # Categorical contribution: ||x_{c_i} - x_{c_j}||^2 in the
        # learned embedding space.
        c1 = x1[..., source_local].round().long().clamp_(0, self.num_classes - 1)
        c2 = x2[..., source_local].round().long().clamp_(0, self.num_classes - 1)
        emb = self.embeddings  # (num_classes, embedding_dim)
        e1 = emb[c1]  # (..., n1, embedding_dim)
        e2 = emb[c2]  # (..., n2, embedding_dim)
        if diag:
            d2_cat = (e1 - e2).pow(2).sum(dim=-1)
        else:
            d2_cat = (e1.unsqueeze(-2) - e2.unsqueeze(-3)).pow(2).sum(dim=-1)

        d2_joint = d2_feat + d2_cat
        d_joint = d2_joint.clamp_min(1e-12).sqrt()
        return self._matern(d_joint)


def _categorical_source_branch(
    d_aug: int,
    lengthscale_lower: float,
    source_kernel: str,
) -> ScaleKernel:
    """Construct the source-aware sub-kernel of the strength GP.

    Returns ``ScaleKernel(<categorical_source> * MaternKernel(no_source + extras))``,
    a Bonilla et al. (2008) intrinsic-coregionalization-model factorisation:
    the Matern shape is shared across all 3 sources; the categorical
    factor modulates inter-source amplitude and correlation.

    Supported ``source_kernel`` topologies:

      ``indexkernel_r{1,2,3}``  ``gpytorch.kernels.IndexKernel(num_tasks=3, rank=r)``.
                                Free task-covar params: ``3*r + 3``. Evaluates
                                ``k(i, j) = (B B^T + diag(v))_{i, j}``.

      ``hamming``               ``botorch.models.kernels.CategoricalKernel(
                                ard_num_dims=1)``.
                                Free params: 1 (single learned correlation
                                between distinct classes; ``rho = exp(-1/ell)``).

      ``onehot_ard``            One-hot encode the source dim into 3 binary
                                indicator dims and ARD-Matern over them. Free
                                params: 3 lengthscales. The ``onehot_ard`` path
                                does NOT use a categorical kernel — it treats
                                source as 3 continuous binary cols (a baseline
                                for "is the categorical kernel actually buying
                                anything?").

      ``legacy_continuous_ard``  Pre-Commit-5 baseline: a single ARD-Matern over
                                ALL raw dims (INCLUDING Material Source) plus
                                the appended engineered features. Source is
                                treated as a continuous coordinate with one
                                ARD lengthscale, exactly as the deployed
                                production V2 model does. Used as the "current
                                productionised architecture" anchor in
                                three-class ablation.

    The Matern half spans the non-source raw dims and the appended
    engineered features (except ``legacy_continuous_ard`` which spans
    all raw dims).
    """
    if source_kernel not in _SUPPORTED_SOURCE_KERNELS:  # pragma: no cover
        raise ValueError(
            f"Unsupported source_kernel: {source_kernel!r}. "
            f"Expected one of {_SUPPORTED_SOURCE_KERNELS}."
        )

    no_source_dims = [i for i in range(_N_RAW_DIMS) if i != _SOURCE_DIM]
    extra_dims = list(range(_N_RAW_DIMS, d_aug))

    if source_kernel == "additive_joint_hamming_rbf_d2":  # pragma: no cover
        # Additive hybrid: K_total = ScaleKernel(joint_hamming_matern)
        # + ScaleKernel(rbf_embedding_d2 product). Each summand has
        # its own outputscale and lengthscales; the optimizer can
        # rebalance between in-distribution (joint topology) and
        # LOCO Class-2 (categorical-embedding) contributions.
        branch_a = _categorical_source_branch(
            d_aug, lengthscale_lower, "joint_hamming_matern"
        )
        branch_b = _categorical_source_branch(
            d_aug, lengthscale_lower, "rbf_embedding_d2"
        )
        return branch_a + branch_b

    if source_kernel == "additive_joint_hamming_nu25_rbf_d2":  # pragma: no cover
        # Additive hybrid using the Matern_5/2 joint variant (the
        # in-distribution corner of the Pareto frontier).
        branch_a = _categorical_source_branch(
            d_aug, lengthscale_lower, "joint_hamming_matern_nu25"
        )
        branch_b = _categorical_source_branch(
            d_aug, lengthscale_lower, "rbf_embedding_d2"
        )
        return branch_a + branch_b

    if (
        source_kernel == "joint_hamming_matern"
        or source_kernel.startswith("joint_hamming_matern_nu")
        or source_kernel == "joint_chain_matern"
    ):
        # Joint kernel: Matern(sqrt(d^2_feat + alpha * d^2_cat)).
        # The categorical_mode and Matern smoothness are parsed from
        # the variant name suffix.
        feature_dims = no_source_dims + extra_dims
        if source_kernel == "joint_chain_matern":  # pragma: no cover
            categorical_mode = "chain"
            nu = 1.5
        elif source_kernel == "joint_hamming_matern_nu05":  # pragma: no cover
            categorical_mode = "hamming"
            nu = 0.5
        elif source_kernel == "joint_hamming_matern_nu25":  # pragma: no cover
            categorical_mode = "hamming"
            nu = 2.5
        else:  # "joint_hamming_matern"
            categorical_mode = "hamming"
            nu = 1.5
        joint = JointHammingMaternKernel(
            feature_dims=feature_dims,
            source_dim=_SOURCE_DIM,
            nu=nu,
            categorical_mode=categorical_mode,
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

    if source_kernel.startswith("joint_embedding_matern_d"):  # pragma: no cover
        # Parse ``joint_embedding_matern_d{embedding_dim}``.
        feature_dims = no_source_dims + extra_dims
        embedding_dim = int(source_kernel[len("joint_embedding_matern_d") :])
        joint = JointEmbeddingMaternKernel(
            feature_dims=feature_dims,
            source_dim=_SOURCE_DIM,
            num_classes=NUM_MATERIAL_CLASSES,
            embedding_dim=embedding_dim,
            nu=1.5,
            active_dims=torch.tensor(feature_dims + [_SOURCE_DIM]),
            lengthscale_constraint=LogTransformedInterval(
                lengthscale_lower, 1e3, initial_value=1.0
            ),
            lengthscale_prior=within_group_prior(
                d_in=_N_RAW_DIMS,
                source_dim=_SOURCE_DIM,
                num_extras=len(extra_dims),
            ),
        )
        return ScaleKernel(
            joint,
            outputscale_constraint=LogTransformedInterval(1e-2, 1e2, initial_value=1.0),
        )

    if source_kernel == "legacy_continuous_ard":  # pragma: no cover
        # Pre-Commit-5 V2 production architecture: a single ARD-Matern over
        # all raw dims including Material Source as a continuous coordinate.
        # This is the model deployed in docs/model/strength.json before the
        # v5 categorical-kernel migration; here it serves as the
        # "current-production-on-current-data" baseline for the
        # three-class ablation's no-regression check.
        all_orig_dims = list(range(_N_RAW_DIMS))
        return ard_matern_with_within_group_prior(
            ard_num_dims=_N_RAW_DIMS + len(extra_dims),
            active_dims=torch.tensor(all_orig_dims + extra_dims),
            prior=within_group_prior(d_in=_N_RAW_DIMS, num_extras=len(extra_dims)),
            initial_outputscale=0.5,
            lengthscale_lower=lengthscale_lower,
        )

    if source_kernel == "onehot_ard":  # pragma: no cover
        # In the one-hot baseline the source dim is expanded to 3 binary
        # cols via an input transform UPSTREAM (in strength_model.py).
        # Here we simply ARD-Matern across (no_source + 3 indicator + extras).
        # The input-transform contract: the 3 indicator cols replace the
        # source dim in-place (so positions 7..9 become indicator[0..2]
        # and the rest of the raw cols shift by +2). The kernel's
        # ``active_dims`` reflect the post-transform layout.
        d_after = _N_RAW_DIMS - 1 + NUM_MATERIAL_CLASSES + len(extra_dims)
        active = list(range(d_after))
        matern = ard_matern_with_within_group_prior(
            ard_num_dims=d_after,
            active_dims=torch.tensor(active),
            prior=None,  # one-hot dims aren't grouped; let MLL fit
            initial_outputscale=0.5,
            lengthscale_lower=lengthscale_lower,
        )
        return matern

    matern = ard_matern_with_within_group_prior(  # pragma: no cover
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
    if source_kernel == "hamming":  # pragma: no cover
        cat = CategoricalKernel(
            ard_num_dims=1,
            active_dims=torch.tensor([_SOURCE_DIM]),
        )
    elif source_kernel.startswith("rbf_embedding_d"):  # pragma: no cover
        # Parse ``rbf_embedding_d{dim}``. Production exposes only
        # the simplex-init + learnable-lengthscale form;
        # ``_fixed_ell`` / ``_linear_init`` variants are accessible
        # as constructor kwargs on the RBFEmbeddingKernel class for
        # research / ablation purposes.
        embedding_dim = int(source_kernel.split("_d")[-1])
        cat = RBFEmbeddingKernel(
            num_classes=NUM_MATERIAL_CLASSES,
            embedding_dim=embedding_dim,
            active_dims=torch.tensor([_SOURCE_DIM]),
        )
    else:  # pragma: no cover
        # indexkernel_r{1, 2, 3}
        rank = int(source_kernel.split("_r")[-1])
        cat = IndexKernel(
            num_tasks=NUM_MATERIAL_CLASSES,
            rank=rank,
            active_dims=torch.tensor([_SOURCE_DIM]),
        )

    return ScaleKernel(  # pragma: no cover
        ProductKernel(cat, matern.base_kernel),
        outputscale_constraint=matern.raw_outputscale_constraint,
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
    time_tying_sigma: float | None = None,
    include_blind: bool = True,
) -> torch.nn.Module:
    """Build the V2 strength kernel adapted to the augmented input dim
    (raw composition + appended engineered features).

    Returns an additive composition of three subkernels::

        blind_matern(no_source_dims + extras)
            + categorical_source_branch(source_dim, no_source_dims + extras)
            + additive_rbf_time(time_only)

    The source-aware sub-kernel is a categorical-times-Matern product:

        ScaleKernel(<categorical_source> * MaternKernel(no_source + extras))

    where ``<categorical_source>`` is one of ``IndexKernel(num_tasks=3,
    rank=r)``, ``CategoricalKernel`` (Hamming), or one-hot ARD —
    selected via the ``source_kernel`` parameter. The default
    ``"indexkernel_r2"`` expresses two latent task-space axes (mortar
    vs. concrete; Class-C-vs-Class-F fly ash) which match the dataset's
    natural chemistry hierarchy.

    Pre-v5 (the deployed V2 model) treated Material Source as a
    continuous ARD coordinate of the source-specific Matern. The v5
    refactor moves to a categorical kernel because the dataset truly
    has three discrete sources with three distinct chemistries; one
    ARD coordinate trying to span both (Set 1 ↔ Set 2) and
    (Set 1+2 ↔ Set 3) variation is ill-defined. See the plan §"Commit 5"
    and ``experiments/THREE_CLASS_AND_PRIOR_BENCHMARK.md`` for the
    Stage-2 ablation that picks the default rank.

    The within-group prior (Cement/FA/Slag tied, Fine/Coarse Aggregate
    tied) is installed on the ``blind`` Matern's lengthscales and on
    the categorical branch's Matern half (operating only on the
    original raw feature dims; the appended engineered features get
    free lengthscales).

    ``lengthscale_lower`` (default 1e-2) controls the lengthscale
    lower constraint. The HRWR/binder ablation found that some
    engineered features rail at this bound under the default; pass
    ``1e-4`` to give the optimiser more room.

    Args:
        d_aug: post-feature-append input dim.
        lengthscale_lower: ARD lengthscale lower constraint.
        source_kernel: categorical source-kernel topology
            (see :func:`_categorical_source_branch`).
        time_tying_sigma: optional cross-component soft-tying of the
            three Time lengthscales (blind Matern, source-specific
            Matern, additive RBF) via a
            :class:`boxcrete.priors.CrossComponentLengthscalePrior`.
            ``None`` (default) disables tying. Typical permissive value
            is 0.5 (admits ~e^0.5 ≈ 1.65× spread between sub-components).
            Stage-3.5 in the v5 ablation; see plan discussion.
            The ``onehot_ard`` source-kernel path skips tying because
            its specific branch is a bare Matern over one-hot-expanded
            source cols — no specific Matern with a Time-active-dim
            to tie to.
    """
    no_source_dims = [i for i in range(_N_RAW_DIMS) if i != _SOURCE_DIM]
    extra_dims = list(range(_N_RAW_DIMS, d_aug))  # appended feature indices

    if include_blind:
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

    if (  # pragma: no cover
        include_blind
        and time_tying_sigma is not None
        and source_kernel
        not in (
            "onehot_ard",
            "legacy_continuous_ard",
        )
    ):
        # Wire up the cross-component soft-tying on Time lengthscales.
        # blind_matern.active_dims = no_source_dims + extras, so Time
        # (raw idx _TIME_DIM_RAW) sits at position
        # no_source_dims.index(_TIME_DIM_RAW). The specific branch
        # (categorical * Matern) has the same Matern active_dims layout.
        from boxcrete.priors import CrossComponentLengthscalePrior

        time_pos_in_blind = no_source_dims.index(IDX["time"])
        blind_matern = blind.base_kernel  # MaternKernel
        # The specific kernel is ScaleKernel(ProductKernel(<cat>, Matern))
        specific_inner_matern = specific.base_kernel.kernels[1]
        rbf_kernel = time_branch.base_kernel  # RBFKernel

        cross_prior = CrossComponentLengthscalePrior(
            lengthscale_getters=[
                # blind Matern's Time lengthscale
                lambda bm=blind_matern, i=time_pos_in_blind: bm.lengthscale[..., i],
                # specific Matern's Time lengthscale (same active_dims layout)
                lambda sm=specific_inner_matern, i=time_pos_in_blind: sm.lengthscale[
                    ..., i
                ],
                # rbf_time's lengthscale (already 1D since active_dims=[time])
                lambda rk=rbf_kernel: rk.lengthscale,
            ],
            sigma=time_tying_sigma,
            attached_dim=rbf_kernel.lengthscale.numel(),
        )
        # Attach to the rbf kernel (smallest tensor, single-element).
        # Registering on exactly one kernel ensures GPyTorch's MLL counts
        # the cross-component penalty exactly once.
        rbf_kernel.register_prior(
            "cross_component_time_prior",
            cross_prior,
            lambda m: m.lengthscale,
        )

    if include_blind:
        return blind + specific + time_branch
    return specific + time_branch  # pragma: no cover


def make_gated_strength_kernel_builder(
    gate_tau: float = 0.10,
    source_kernel: str = DEFAULT_SOURCE_KERNEL,
    time_tying_sigma: float | None = None,
    include_blind: bool = True,
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
        time_tying_sigma: optional cross-component time-lengthscale
            tying; see :func:`build_strength_kernel_for_aug_dim`.

    The `time_idx` for the gate is the time dim (``IDX["time"]``), which is
    where the time column sits in the post-input-transform vector. After
    the engineered input transform, t=0 maps to 0 in this column (true
    for both the `Normalize`-on-time and `skip_time_in_normalize` paths).
    """

    def _builder(d_aug: int) -> torch.nn.Module:
        base = build_strength_kernel_for_aug_dim(
            d_aug,
            source_kernel=source_kernel,
            time_tying_sigma=time_tying_sigma,
            include_blind=include_blind,
        )
        return TimeGatedKernel(
            base,
            time_idx=IDX["time"],
            gate_tau=gate_tau,
        )

    return _builder


__all__ = [
    "DEFAULT_SOURCE_KERNEL",
    "JointEmbeddingMaternKernel",
    "JointHammingMaternKernel",
    "NUM_MATERIAL_CLASSES",
    "RBFEmbeddingKernel",
    "TimeGatedKernel",
    "additive_time_kernel",
    "ard_matern_with_within_group_prior",
    "build_strength_kernel_for_aug_dim",
    "make_gated_strength_kernel_builder",
]
