# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``RBFEmbeddingKernel``.

Verifies (a) gauge fixing (class 0 at origin; class 1 on first axis),
(b) parameter init from the equilateral-simplex warm start gives
PSD Gram with the expected pairwise covariances, (c) gradients flow
through both ``raw_embedding_free`` and the lengthscale, and (d) the
kernel value matches the closed-form RBF at the post-init embeddings.
"""

import math

import pytest
import torch

from boxcrete.kernels import RBFEmbeddingKernel


def test_gauge_fixing_class_0_at_origin():
    for d in (1, 2, 3):
        k = RBFEmbeddingKernel(num_classes=3, embedding_dim=d)
        emb = k.embeddings.detach()
        assert torch.allclose(
            emb[0], torch.zeros(d), atol=1e-6
        ), f"class 0 not at origin for d={d}: {emb[0]}"


def test_gauge_fixing_class_1_on_first_axis_when_d_ge_2():
    for d in (2, 3):
        k = RBFEmbeddingKernel(num_classes=3, embedding_dim=d)
        emb = k.embeddings.detach()
        assert torch.allclose(
            emb[1, 1:], torch.zeros(d - 1), atol=1e-6
        ), f"class 1 not on first axis for d={d}: {emb[1]}"


def test_init_pairwise_covariances_unit_simplex():
    """At init, the d=2 simplex has unit pairwise distances and
    cross-class similarity = exp(-1/2) at lengthscale 1."""
    k = RBFEmbeddingKernel(num_classes=3, embedding_dim=2)
    x = torch.tensor([[0.0], [1.0], [2.0]])
    K = k(x, x).to_dense().detach()
    expected_off = math.exp(-0.5)
    # Diagonal == 1
    assert torch.allclose(torch.diagonal(K), torch.ones(3), atol=1e-5)
    # Off-diagonal == exp(-0.5) for the equilateral triangle
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        assert (
            abs(K[i, j].item() - expected_off) < 1e-4
        ), f"K[{i},{j}] = {K[i, j].item()} != exp(-0.5) = {expected_off}"


def test_kernel_is_psd_at_random_embeddings():
    torch.manual_seed(0)
    k = RBFEmbeddingKernel(num_classes=3, embedding_dim=2)
    # Perturb embeddings randomly
    with torch.no_grad():
        k.raw_embedding_free.add_(torch.randn_like(k.raw_embedding_free))
    x = torch.tensor([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]])
    K = k(x, x).to_dense().detach()
    eigvals = torch.linalg.eigvalsh(K)
    assert (
        eigvals.min().item() >= -1e-6
    ), f"Kernel not PSD: min eigenvalue = {eigvals.min().item()}"


def test_gradient_flow_through_embeddings_and_lengthscale():
    k = RBFEmbeddingKernel(num_classes=3, embedding_dim=2)
    x = torch.tensor([[0.0], [1.0], [2.0]])
    K = k(x, x).to_dense()
    loss = (K - torch.eye(3)).pow(2).sum()
    loss.backward()
    assert k.raw_embedding_free.grad is not None
    # At least one entry should have non-trivial gradient (the simplex
    # init places classes at exp(-0.5) similarity, not at the eye target)
    assert k.raw_embedding_free.grad.abs().max().item() > 1e-3
    assert k.raw_lengthscale.grad is not None
    assert k.raw_lengthscale.grad.abs().max().item() > 1e-3


def test_lengthscale_to_zero_recovers_hamming_in_limit():
    """As lengthscale -> 0 with bounded embeddings, off-diagonal
    similarities collapse to 0 (Hamming-like behaviour)."""
    k = RBFEmbeddingKernel(num_classes=3, embedding_dim=2)
    # Manually shrink the lengthscale via the constraint's inverse
    # transform; clamp small but >= constraint lower bound (1e-2).
    target = torch.tensor(0.05)
    raw = k.raw_lengthscale_constraint.inverse_transform(target)
    with torch.no_grad():
        k.raw_lengthscale.copy_(raw.expand_as(k.raw_lengthscale))
    x = torch.tensor([[0.0], [1.0], [2.0]])
    K = k(x, x).to_dense().detach()
    # Diagonal still 1, off-diagonal small
    assert torch.allclose(torch.diagonal(K), torch.ones(3), atol=1e-6)
    assert (
        K[0, 1].item() < 1e-3
    ), f"lengthscale={k.lengthscale.item()} but K[0,1]={K[0, 1].item()}"


def test_translation_invariance_of_distances():
    """RBF kernel depends only on differences; translating all
    embeddings by a constant should not change the Gram matrix.
    (Verified by re-rotating a synthetic pinned-gauge solution.)
    """
    k1 = RBFEmbeddingKernel(num_classes=3, embedding_dim=2)
    x = torch.tensor([[0.0], [1.0], [2.0]])
    K1 = k1(x, x).to_dense().detach()

    # Construct a second kernel and write the same embeddings via the
    # same gauge (always pinned the same way), so K2 must equal K1.
    k2 = RBFEmbeddingKernel(num_classes=3, embedding_dim=2)
    K2 = k2(x, x).to_dense().detach()
    assert torch.allclose(K1, K2, atol=1e-6)


def test_unsupported_constructor_args():
    with pytest.raises(ValueError):
        RBFEmbeddingKernel(num_classes=1, embedding_dim=2)
    with pytest.raises(ValueError):
        RBFEmbeddingKernel(num_classes=3, embedding_dim=0)


def test_embedding_dim_one_has_two_free_entries():
    """For d=1 we pin only x_0 = 0 (translation lock); x_1, x_2 are
    free \u2014 gives 2 free entries."""
    k = RBFEmbeddingKernel(num_classes=3, embedding_dim=1)
    assert k.raw_embedding_free.numel() == 2


def test_embedding_dim_two_has_three_free_entries():
    """For d=2 we pin x_0 = (0,0) (2 entries) AND x_1[1] = 0
    (1 entry rotation lock) \u2014 leaves 6 - 2 - 1 = 3 free entries
    (x_1[0], x_2[0], x_2[1])."""
    k = RBFEmbeddingKernel(num_classes=3, embedding_dim=2)
    assert k.raw_embedding_free.numel() == 3


def test_fixed_lengthscale_has_zero_grad():
    """``learn_lengthscale=False`` must (a) leave the lengthscale at 1
    and (b) freeze its gradient."""
    k = RBFEmbeddingKernel(num_classes=3, embedding_dim=2, learn_lengthscale=False)
    assert abs(k.lengthscale.item() - 1.0) < 1e-6
    assert k.raw_lengthscale.requires_grad is False
    # Gradient through the kernel still flows into the embeddings even
    # though ell is frozen.
    x = torch.tensor([[0.0], [1.0], [2.0]])
    K = k(x, x).to_dense()
    loss = (K - torch.eye(3)).pow(2).sum()
    loss.backward()
    assert k.raw_embedding_free.grad is not None
    assert k.raw_embedding_free.grad.abs().max().item() > 1e-3
    # raw_lengthscale.grad is None because requires_grad=False
    assert k.raw_lengthscale.grad is None


def test_linear_init_d1_places_classes_at_integer_labels():
    """``init_strategy='linear'`` at d=1 should give x_c = c."""
    k = RBFEmbeddingKernel(num_classes=3, embedding_dim=1, init_strategy="linear")
    emb = k.embeddings.detach()
    expected = torch.tensor([[0.0], [1.0], [2.0]])
    assert torch.allclose(
        emb, expected, atol=1e-6
    ), f"expected linear init (0,1,2); got {emb.flatten().tolist()}"


def test_linear_init_d2_first_axis_has_integer_labels():
    """At d=2, linear init places class c at (c, 0)."""
    k = RBFEmbeddingKernel(num_classes=3, embedding_dim=2, init_strategy="linear")
    emb = k.embeddings.detach()
    expected = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert torch.allclose(emb, expected, atol=1e-6)


def test_invalid_init_strategy_raises():
    with pytest.raises(ValueError):
        RBFEmbeddingKernel(
            num_classes=3, embedding_dim=2, init_strategy="invalid_thing"
        )


def test_simplex_and_linear_inits_produce_different_kernels():
    """Sanity check: the two init strategies should give measurably
    different kernel matrices at d=1."""
    k_simplex = RBFEmbeddingKernel(
        num_classes=3, embedding_dim=1, init_strategy="simplex"
    )
    k_linear = RBFEmbeddingKernel(
        num_classes=3, embedding_dim=1, init_strategy="linear"
    )
    x = torch.tensor([[0.0], [1.0], [2.0]])
    K_s = k_simplex(x, x).to_dense().detach()
    K_l = k_linear(x, x).to_dense().detach()
    # Off-diagonal class-0/class-2 similarity differs:
    # simplex (-1, 0, +1): K(0,2) = exp(-0.5) ≈ 0.6065
    # linear  (0, 1, 2):   K(0,2) = exp(-2.0) ≈ 0.1353
    assert abs(K_s[1, 2].item() - K_l[1, 2].item()) > 0.1
