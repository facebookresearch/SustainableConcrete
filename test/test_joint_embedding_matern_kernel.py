# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``JointEmbeddingMaternKernel``.

Verifies the joint feature + learnable-class-embedding Matern kernel:
gauge fixing (class 0 at origin; for d>=2, class 1 on first axis),
PSD-ness, gradient flow through both feature lengthscales and class
embeddings, and that linear vs simplex init produce different kernels.
"""

import pytest
import torch

from boxcrete.kernels import JointEmbeddingMaternKernel


def _make_kernel(d=2, nu=1.5, init_strategy="simplex"):
    return JointEmbeddingMaternKernel(
        feature_dims=[0, 1],
        source_dim=2,
        num_classes=3,
        embedding_dim=d,
        nu=nu,
        active_dims=torch.tensor([0, 1, 2]),
        init_strategy=init_strategy,
    )


def _make_input(n=4, classes=3):
    torch.manual_seed(0)
    x = torch.randn(n, 3)
    x[:, 2] = torch.randint(0, classes, (n,)).to(x.dtype)
    return x


def test_gauge_class_0_pinned_at_origin():
    for d in (1, 2, 3):
        k = JointEmbeddingMaternKernel(
            feature_dims=[0, 1],
            source_dim=2,
            num_classes=3,
            embedding_dim=d,
            active_dims=torch.tensor([0, 1, 2]),
        )
        emb = k.embeddings.detach()
        assert torch.allclose(emb[0], torch.zeros(d), atol=1e-6)


def test_gauge_class_1_on_first_axis_when_d_ge_2():
    for d in (2, 3):
        k = JointEmbeddingMaternKernel(
            feature_dims=[0, 1],
            source_dim=2,
            num_classes=3,
            embedding_dim=d,
            active_dims=torch.tensor([0, 1, 2]),
        )
        emb = k.embeddings.detach()
        assert torch.allclose(emb[1, 1:], torch.zeros(d - 1), atol=1e-6)


def test_diagonal_is_one():
    k = _make_kernel(d=2)
    x = _make_input(n=5)
    K = k(x, x).to_dense().detach()
    assert torch.allclose(torch.diagonal(K), torch.ones(5), atol=1e-5)


def test_kernel_is_psd():
    k = _make_kernel(d=2)
    x = _make_input(n=8)
    K = k(x, x).to_dense().detach()
    eigvals = torch.linalg.eigvalsh(K + 1e-6 * torch.eye(K.shape[0]))
    assert eigvals.min().item() >= -1e-5


def test_gradient_flows_through_embeddings_and_lengthscale():
    k = _make_kernel(d=2)
    x = _make_input(n=4)
    K = k(x, x).to_dense()
    loss = (K - torch.eye(4)).pow(2).sum()
    loss.backward()
    assert k.raw_feat_lengthscale.grad is not None
    assert k.raw_feat_lengthscale.grad.abs().max().item() > 1e-4
    assert k.raw_embedding_free.grad is not None
    assert k.raw_embedding_free.grad.abs().max().item() > 1e-4


def test_linear_init_d1_places_classes_at_integer_labels():
    k = _make_kernel(d=1, init_strategy="linear")
    emb = k.embeddings.detach()
    expected = torch.tensor([[0.0], [1.0], [2.0]])
    assert torch.allclose(emb, expected, atol=1e-6)


def test_linear_init_d2_first_axis_has_integer_labels():
    k = _make_kernel(d=2, init_strategy="linear")
    emb = k.embeddings.detach()
    expected = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert torch.allclose(emb, expected, atol=1e-6)


def test_simplex_and_linear_init_differ_at_d1():
    k_simplex = _make_kernel(d=1, init_strategy="simplex")
    k_linear = _make_kernel(d=1, init_strategy="linear")
    # Same-feature, different-class points to isolate the
    # categorical contribution.
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]
    )
    K_s = k_simplex(x, x).to_dense().detach()
    K_l = k_linear(x, x).to_dense().detach()
    # K(0, 2): simplex has class 2 at +1 (distance 1), linear at +2
    # (distance 2). So K(0, 2) should differ.
    assert abs(K_s[0, 2].item() - K_l[0, 2].item()) > 0.05


def test_invalid_init_strategy_raises():
    with pytest.raises(ValueError):
        _make_kernel(d=2, init_strategy="invalid")


def test_invalid_nu_raises():
    with pytest.raises(ValueError):
        JointEmbeddingMaternKernel(
            feature_dims=[0],
            source_dim=1,
            num_classes=2,
            embedding_dim=1,
            nu=3.0,
            active_dims=torch.tensor([0, 1]),
        )
