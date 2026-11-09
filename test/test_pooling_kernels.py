#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the partial-pooling / structured 3-class source kernels:

  * ``indexkernel_r2_pooled[_t<tau>]`` - IndexKernel + task-pooling prior.
  * ``indexkernel_r2_shrunk``          - IndexKernel + residual-outputscale shrinkage.
  * ``joint_hamming_matern_pooled``    - joint kernel + alpha-shrinkage (pooling).
  * ``joint_{chain,hamming}_matern_regprior`` - joint kernel + weakly-informative alpha prior.
  * ``fixed_chem_task``                - FIXED chemistry-informed task covariance (no learnable task params).
"""

from __future__ import annotations

import torch

from boxcrete.kernels import (
    DEFAULT_X_COLUMNS,
    _SOURCE_DIM,
    _chemistry_task_covar,
    build_strength_kernel_for_aug_dim,
)
from boxcrete.priors import TaskPoolingPrior


def _random_X(n=12, d_aug=None):
    d_aug = d_aug or len(DEFAULT_X_COLUMNS)
    X = torch.rand(n, d_aug, dtype=torch.float64)
    X[:, _SOURCE_DIM] = torch.randint(0, 3, (n,)).to(torch.float64)
    return X


def _prior_names(kernel):
    return {name for name, *_ in kernel.named_priors()}


def _build(sk):
    return build_strength_kernel_for_aug_dim(len(DEFAULT_X_COLUMNS), source_kernel=sk)


def _forward_ok(k):
    X = _random_X()
    K = k(X).to_dense()
    assert K.shape == (X.shape[0], X.shape[0])


def test_pooled_kernel_builds_and_registers_prior():
    k = _build("indexkernel_r2_pooled")
    assert any("pooling" in n for n in _prior_names(k))
    _forward_ok(k)


def test_pooled_kernel_tau_suffix_builds():
    for sk in ("indexkernel_r2_pooled_t0.1", "indexkernel_r2_pooled_t0.5"):
        k = _build(sk)
        assert any("pooling" in n for n in _prior_names(k))
        _forward_ok(k)


def test_shrunk_kernel_builds_and_registers_prior():
    k = _build("indexkernel_r2_shrunk")
    assert any("shrink" in n for n in _prior_names(k))
    _forward_ok(k)


def test_joint_hamming_matern_pooled_registers_alpha_prior():
    k = _build("joint_hamming_matern_pooled")
    assert any("alpha_pooling" in n for n in _prior_names(k))
    _forward_ok(k)


def test_chain_regprior_registers_alpha_prior():
    k = _build("joint_chain_matern_regprior")
    assert any("alpha_regprior" in n for n in _prior_names(k))
    _forward_ok(k)


def test_fixed_chem_task_builds_and_is_psd():
    k = _build("fixed_chem_task")
    _forward_ok(k)
    T = _chemistry_task_covar()
    assert float(torch.linalg.eigvalsh(T).min()) > -1e-9
    # mortar (class 0) is the outlier: the two concretes are more similar.
    assert float(T[1, 2]) > float(T[0, 1])


def test_task_pooling_prior_penalises_spread_rows():
    prior = TaskPoolingPrior(num_tasks=3, rank=2, tau=0.3)
    identical = torch.zeros(3, 2, dtype=torch.float64)
    spread = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    assert prior.log_prob(identical).sum() > prior.log_prob(spread).sum()
    assert torch.isclose(
        prior.log_prob(identical).sum(),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-9,
    )
