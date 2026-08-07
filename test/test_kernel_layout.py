# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Kernel-layout regression tests.

The V2 strength GP has a specific nested kernel structure:

    SingleTaskGP.covar_module
      = TimeGatedKernel
          .base_kernel
          = AdditiveKernel
              .kernels
              = [
                  ScaleKernel(matern_blind).base_kernel.lengthscale,
                  ScaleKernel(ProductKernel(CategoricalKernel, matern_specific)),
                  ScaleKernel(rbf_time).base_kernel.lengthscale,
              ]

The production source kernel is ``hamming`` (``CategoricalKernel x Matern``),
so the "specific" branch's ``.base_kernel`` is a ``ProductKernel`` whose
Matern factor carries the continuous ARD lengthscales (over every dim
EXCEPT the source column) and whose ``CategoricalKernel`` factor carries a
single source lengthscale.

Code that introspects fitted models for diagnostics — e.g.,
``experiments/regenerate_strength_json.py`` (the canonical writer of
``docs/model/strength.json``) — must walk this exact path.
A previous diagnostic script walked the wrong path
(``covar_module.kernels[0]``, skipping the gate's ``base_kernel``)
and caught the resulting ``AttributeError`` with a bare
``except Exception:``, silently producing an empty stability report.

These tests exist to make any future refactor of the kernel composition
fail loudly at the structural level rather than at the reporting level.
"""

from __future__ import annotations

import unittest

import torch

from boxcrete import fit_strength_gp, load_concrete_strength
from boxcrete.features import F5_ALLLOG_FEATURES
from boxcrete.kernels import TimeGatedKernel
from boxcrete.utils import DEFAULT_X_COLUMNS


def _matern_lengthscale_of(scale_kernel):
    """Return the continuous ARD lengthscale tensor of a sub-kernel.

    For blind / rbf_time the ScaleKernel wraps the Matern/RBF directly. For
    the hamming ``specific`` branch it wraps ``ProductKernel(Categorical,
    Matern)``; return the Matern factor's lengthscale (the factor that has a
    multi-dim lengthscale).
    """
    inner = scale_kernel.base_kernel
    ls = getattr(inner, "lengthscale", None)
    if ls is not None:
        return ls
    # ProductKernel: pick the factor whose lengthscale spans the most dims.
    best = None
    for factor in inner.kernels:
        fls = getattr(factor, "lengthscale", None)
        if fls is not None and (best is None or fls.numel() > best.numel()):
            best = fls
    return best


class TestKernelLayout(unittest.TestCase):
    """Asserts the V2 strength GP's kernel introspection contract."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch.manual_seed(0)
        data = load_concrete_strength()
        X, Y, Yvar, bounds = data.strength_data
        # Use the fast-test affordance — structural correctness is what
        # we're testing, not converged-fit numerics.
        cls.model = fit_strength_gp(
            X=X, Y=Y, Yvar=Yvar, X_bounds=bounds, seed=0, max_optimizer_iter=20
        )

    def test_top_level_is_time_gated_kernel(self):
        """``covar_module`` is the gate; bare ``.kernels`` is wrong."""
        self.assertIsInstance(self.model.covar_module, TimeGatedKernel)

    def test_gate_wraps_an_additive_sum(self):
        """``base_kernel`` is the AdditiveKernel sum of (blind, specific, rbf)."""
        base = self.model.covar_module.base_kernel
        # AdditiveKernel exposes .kernels as a ModuleList. We don't import
        # the class here to avoid coupling the test to a private GPyTorch
        # API; the duck-typed contract is that .kernels is iterable of 3.
        self.assertTrue(
            hasattr(base, "kernels"),
            "covar_module.base_kernel should expose .kernels (AdditiveKernel)",
        )
        self.assertEqual(
            len(base.kernels),
            3,
            "expected exactly 3 sub-kernels (blind, specific, rbf_time)",
        )

    def test_each_sub_kernel_is_a_scale_kernel_with_lengthscale(self):
        """Each ``base.kernels[i]`` is a ``ScaleKernel``. blind and rbf_time
        expose ``.base_kernel.lengthscale`` directly; the ``hamming``
        specific branch's ``.base_kernel`` is a ``ProductKernel`` whose
        Matern factor carries the ARD lengthscale — the path used by
        ``regenerate_strength_json.py`` and downstream tooling."""
        base = self.model.covar_module.base_kernel
        for i, sub in enumerate(base.kernels):
            self.assertTrue(
                hasattr(sub, "base_kernel"),
                f"sub-kernel {i} should expose .base_kernel (ScaleKernel)",
            )
            ls = _matern_lengthscale_of(sub)
            self.assertIsInstance(
                ls, torch.Tensor, f"sub-kernel {i} has no Matern/RBF lengthscale tensor"
            )

    def test_canonical_introspection_path_yields_three_lengthscale_tensors(self):
        """End-to-end smoke: the exact path used by tooling produces
        non-empty per-sub-kernel lengthscales of the right number of
        dimensions (matching the production model)."""
        base = self.model.covar_module.base_kernel
        # Order: [blind_matern, specific(=Product(Categorical, Matern)),
        # rbf_time] — used by the canonical writer at
        # ``experiments/regenerate_strength_json.py``.
        blind_ls = _matern_lengthscale_of(base.kernels[0]).flatten()
        specific_ls = _matern_lengthscale_of(base.kernels[1]).flatten()
        rbf_ls = _matern_lengthscale_of(base.kernels[2]).flatten()
        # Derive expected ARD dim counts from the same constants the
        # production fit uses, so adding a feature to ``F5_ALLLOG_FEATURES``
        # or a column to ``DEFAULT_X_COLUMNS`` doesn't fire this test
        # opaquely — the test failure (if any) will then come from a
        # legitimate kernel-structure change, not a stale hardcoded constant.
        d_aug = len(DEFAULT_X_COLUMNS) + len(F5_ALLLOG_FEATURES)
        # Both blind and specific Matern span all aug dims EXCEPT the source
        # column (the hamming CategoricalKernel factor handles source);
        # rbf time is a single-dim kernel on time only.
        self.assertEqual(blind_ls.numel(), d_aug - 1)
        self.assertEqual(specific_ls.numel(), d_aug - 1)
        self.assertEqual(rbf_ls.numel(), 1)

    def test_short_path_through_covar_module_kernels_does_not_exist(self):
        """Anti-test: the SHORT path ``covar_module.kernels[0]`` is wrong
        (the previous-generation diagnostic-script bug this regression test
        exists to prevent). The gate is a single kernel, not a container
        — ``ScaleKernel``-shaped attribute access via ``.kernels`` should
        not work at the top level."""
        # We intentionally call hasattr instead of accessing directly so a
        # future GPyTorch upgrade that adds a top-level ``.kernels`` shim
        # surfaces a deliberate test failure asking the introspector to
        # decide which path to walk.
        self.assertFalse(
            hasattr(self.model.covar_module, "kernels"),
            "TimeGatedKernel should not directly expose .kernels — walk "
            ".base_kernel.kernels instead. If GPyTorch ever shims this, "
            "review tooling that walks the kernel tree.",
        )

    def test_init_rejects_unknown_kwargs(self):
        """``TimeGatedKernel.__init__`` should reject unknown keyword
        arguments rather than silently swallowing them. ``gate_learnable``
        used to be a real kwarg controlling whether ``tau`` was learnable;
        when we removed it (the V2 strength GP always uses a frozen tau
        buffer), an earlier ``**kwargs`` passthrough silently accepted
        ``gate_learnable=True`` calls and dropped the value on the floor.
        This test pins the explicit-arg-list contract so a typo on any
        former kwarg fires a hard ``TypeError`` instead."""
        from gpytorch.kernels import RBFKernel

        with self.assertRaises(TypeError):
            # pyrefly: ignore [unexpected-keyword] - the invalid kwarg is
            # the point of this test (we assert it raises at runtime).
            TimeGatedKernel(RBFKernel(), time_idx=0, gate_learnable=True)
        # Sanity: the legitimate construction path still works.
        TimeGatedKernel(RBFKernel(), time_idx=0, gate_tau=0.05)


if __name__ == "__main__":
    unittest.main()
