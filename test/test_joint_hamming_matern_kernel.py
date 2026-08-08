# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``JointHammingMaternKernel``.

Verifies the joint feature + Hamming-categorical Matern kernel:
(a) is PSD on synthetic inputs, (b) recovers ARD-Matern when alpha is
zero, (c) attenuates cross-class pairs as alpha grows, (d) supports
gradient flow through both lengthscales and alpha, and (e) handles
the strength-GP layout (feature_dims + source_dim) correctly via
active_dims.
"""

import pytest
import torch

from boxcrete.kernels import JointHammingMaternKernel


def _make_kernel(d=2, nu=1.5, alpha_init=1.0):
    return JointHammingMaternKernel(
        feature_dims=list(range(d)),
        source_dim=d,
        nu=nu,
        ard_num_dims=d,
        active_dims=torch.tensor(list(range(d)) + [d]),
        alpha_initial_value=alpha_init,
    )


def _make_input(n=5, d=2, classes=3):
    torch.manual_seed(0)
    x = torch.randn(n, d + 1)
    # Last dim is class label — round to integer in [0, classes-1]
    x[:, d] = torch.randint(0, classes, (n,)).to(x.dtype)
    return x


def test_diagonal_is_one_at_zero_distance():
    k = _make_kernel(d=2)
    x = _make_input(n=4, d=2)
    K = k(x, x).to_dense().detach()
    assert torch.allclose(torch.diagonal(K), torch.ones(4), atol=1e-5)


def test_kernel_is_psd():
    k = _make_kernel(d=3)
    x = _make_input(n=6, d=3)
    K = k(x, x).to_dense().detach()
    eigvals = torch.linalg.eigvalsh(K + 1e-6 * torch.eye(K.shape[0]))
    assert eigvals.min().item() >= -1e-5


def test_alpha_zero_recovers_ard_matern():
    """At alpha = 0, the kernel ignores class labels — same-class and
    different-class rows with equal feature distances should give the
    same kernel value."""
    k = JointHammingMaternKernel(
        feature_dims=[0, 1],
        source_dim=2,
        nu=1.5,
        ard_num_dims=2,
        active_dims=torch.tensor([0, 1, 2]),
        alpha_initial_value=1e-2,  # near-zero (but inside constraint range)
    )
    # Two rows with same features but different classes:
    x = torch.tensor(
        [[1.0, 2.0, 0.0], [1.0, 2.0, 1.0]]
    )
    K = k(x, x).to_dense().detach()
    # Same features → same-feature-distance contribution; categorical
    # adds 1e-2 (negligible). K[0,1] should be close to K[0,0] = 1.
    assert K[0, 1].item() > 0.95


def test_alpha_large_decouples_classes():
    """At very large alpha, cross-class similarity should collapse
    toward 0 even when features are identical."""
    k = JointHammingMaternKernel(
        feature_dims=[0, 1],
        source_dim=2,
        nu=1.5,
        ard_num_dims=2,
        active_dims=torch.tensor([0, 1, 2]),
        alpha_initial_value=100.0,  # interior of (1e-3, 1e3)
    )
    x = torch.tensor(
        [[1.0, 2.0, 0.0], [1.0, 2.0, 1.0]]
    )
    K = k(x, x).to_dense().detach()
    # Cross-class with equal features: alpha=100, d^2 = 100, d = 10,
    # Matern_3/2(sqrt(3) * 10) ~ 1e-7.
    assert K[0, 1].item() < 1e-3


def test_same_class_pairs_match_ard_matern():
    """Same-class rows with the same feature distance must produce
    identical kernel values regardless of which class they share —
    the categorical term is zero for same-class pairs."""
    k = _make_kernel(d=2, alpha_init=5.0)
    # Two same-class pairs with identical feature differences:
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # class 0
            [1.0, 0.0, 0.0],  # class 0, dx = 1
            [0.0, 0.0, 1.0],  # class 1
            [1.0, 0.0, 1.0],  # class 1, dx = 1
        ]
    )
    K = k(x, x).to_dense().detach()
    # K[0,1] (class 0 pair, dx=1) and K[2,3] (class 1 pair, dx=1)
    # must be equal.
    assert abs(K[0, 1].item() - K[2, 3].item()) < 1e-6


def test_gradient_flow_through_lengthscale_and_alpha():
    k = _make_kernel(d=2)
    x = _make_input(n=4, d=2)
    K = k(x, x).to_dense()
    loss = (K - torch.eye(4)).pow(2).sum()
    loss.backward()
    assert k.raw_feat_lengthscale.grad is not None
    assert k.raw_feat_lengthscale.grad.abs().max().item() > 1e-4
    assert k.raw_alpha.grad is not None
    assert k.raw_alpha.grad.abs().max().item() > 1e-4


def test_active_dims_remap_works_with_strength_gp_layout():
    """In the strength-GP, source_dim is at position 7 of a 10-dim
    raw input; the kernel's active_dims maps to non-source dims plus
    the source dim. Verify the kernel correctly identifies which
    column carries the class label after the active_dims slicing."""
    feature_dims = [0, 1, 2, 3, 4, 5, 6, 8, 9]  # all non-source dims
    source_dim = 7
    active = torch.tensor(feature_dims + [source_dim])
    k = JointHammingMaternKernel(
        feature_dims=feature_dims,
        source_dim=source_dim,
        ard_num_dims=len(feature_dims),
        active_dims=active,
    )
    torch.manual_seed(0)
    x = torch.randn(3, 10)
    x[:, source_dim] = torch.tensor([0.0, 1.0, 2.0])  # 3 distinct classes
    K = k(x, x).to_dense().detach()
    # All distinct classes → off-diagonals all attenuated by alpha.
    # Diagonals must be 1 regardless.
    assert torch.allclose(torch.diagonal(K), torch.ones(3), atol=1e-5)


def test_invalid_nu_raises():
    with pytest.raises(ValueError):
        JointHammingMaternKernel(
            feature_dims=[0],
            source_dim=1,
            nu=3.0,
            ard_num_dims=1,
            active_dims=torch.tensor([0, 1]),
        )


def test_alpha_property_is_positive():
    k = _make_kernel(d=2, alpha_init=2.5)
    assert k.alpha.item() > 0
    # The constraint is monotone, so alpha at init is alpha_initial_value
    # (within numerical tolerance).
    assert abs(k.alpha.item() - 2.5) < 1e-4


def test_matern_smoothness_options():
    for nu in (0.5, 1.5, 2.5):
        k = JointHammingMaternKernel(
            feature_dims=[0],
            source_dim=1,
            nu=nu,
            ard_num_dims=1,
            active_dims=torch.tensor([0, 1]),
        )
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        K = k(x, x).to_dense().detach()
        # Diagonal == 1
        assert abs(K[0, 0].item() - 1.0) < 1e-5
        # Off-diagonal in (0, 1)
        assert 0 < K[0, 1].item() < 1


def test_chain_categorical_mode_distances():
    """In chain mode, cross-class squared distance is (i - j)^2.
    K(0, 2) at distance 2 should be smaller than K(0, 1) at distance 1."""
    k = JointHammingMaternKernel(
        feature_dims=[0],
        source_dim=1,
        nu=1.5,
        active_dims=torch.tensor([0, 1]),
        categorical_mode="chain",
        alpha_initial_value=1.0,
    )
    # Same feature value, different class labels — only the
    # categorical-distance term contributes.
    x = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]
    )
    K = k(x, x).to_dense().detach()
    # K(0, 1) > K(0, 2) because |0 - 1|^2 = 1 < 4 = |0 - 2|^2.
    assert K[0, 1].item() > K[0, 2].item() + 0.05


def test_chain_mode_matches_hamming_for_adjacent_classes():
    """For adjacent classes (i, j with |i - j| = 1), chain and Hamming
    give the same kernel value (both have d^2_cat = 1)."""
    k_chain = JointHammingMaternKernel(
        feature_dims=[0], source_dim=1, nu=1.5,
        active_dims=torch.tensor([0, 1]),
        categorical_mode="chain", alpha_initial_value=1.0,
    )
    k_ham = JointHammingMaternKernel(
        feature_dims=[0], source_dim=1, nu=1.5,
        active_dims=torch.tensor([0, 1]),
        categorical_mode="hamming", alpha_initial_value=1.0,
    )
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    K_c = k_chain(x, x).to_dense().detach()
    K_h = k_ham(x, x).to_dense().detach()
    assert abs(K_c[0, 1].item() - K_h[0, 1].item()) < 1e-6


def test_invalid_categorical_mode_raises():
    with pytest.raises(ValueError):
        JointHammingMaternKernel(
            feature_dims=[0], source_dim=1, nu=1.5,
            active_dims=torch.tensor([0, 1]),
            categorical_mode="invalid_mode",
        )
