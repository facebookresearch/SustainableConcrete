# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke-test the boxcrete public API.

This catches the bug class where ``__all__`` advertises a name that
isn't actually defined in the package (we shipped
``"fit_strength_gp_v2"`` in ``__all__`` for two weeks while the symbol
itself didn't exist; ``from boxcrete import *`` would have caught it).

Each name in ``boxcrete.__all__`` must:
  1. Resolve via ``getattr(boxcrete, name)`` (i.e., be importable).
  2. Be a non-None object.
"""

from __future__ import annotations

import unittest


class TestPublicAPI(unittest.TestCase):
    def test_star_import_succeeds(self):
        # Star-import must succeed — i.e., every name in __all__ resolves.
        ns: dict = {}
        exec("from boxcrete import *", ns)  # noqa: S102
        # Touch a couple of canonical names so the test fails loudly if
        # the package is somehow loaded but missing core entry points.
        self.assertTrue(callable(ns["fit_strength_gp"]))
        self.assertTrue(callable(ns["fit_slump_gp"]))

    def test_all_names_resolve(self):
        import boxcrete

        for name in boxcrete.__all__:
            with self.subTest(name=name):
                self.assertTrue(
                    hasattr(boxcrete, name),
                    f"`boxcrete.__all__` advertises {name!r} but the "
                    f"name is not defined on the package.",
                )
                obj = getattr(boxcrete, name)
                self.assertIsNotNone(
                    obj,
                    f"`boxcrete.{name}` resolved to None.",
                )

    def test_fit_strength_gp_is_callable(self):
        from boxcrete import fit_strength_gp

        self.assertTrue(callable(fit_strength_gp))

    def test_load_pretrained_strength_gp_is_callable(self):
        from boxcrete import load_pretrained_strength_gp

        self.assertTrue(callable(load_pretrained_strength_gp))

    def test_strength_v2_module_is_gone(self):
        # Sanity-check the rename: the old name must NOT resolve.
        with self.assertRaises((ImportError, ModuleNotFoundError)):
            __import__("boxcrete.strength_v2")

    def test_new_module_layout_present(self):
        """Positively assert the V2 module split is intact. The reorg
        introduced 8 submodules; if any get accidentally removed or
        renamed, this fires before the symptom shows up downstream."""
        for mod in (
            "boxcrete.concrete_model",
            "boxcrete.slump_model",
            "boxcrete.strength_model",
            "boxcrete.strength_model_legacy",
            "boxcrete.kernels",
            "boxcrete.features",
            "boxcrete.likelihoods",
            "boxcrete.priors",
        ):
            with self.subTest(module=mod):
                __import__(mod)


if __name__ == "__main__":
    unittest.main()
