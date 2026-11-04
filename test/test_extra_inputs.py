# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for extra inputs appended after Time in the strength GP.

``fit_strength_gp`` accepts columns after the 10 ``DEFAULT_X_COLUMNS``. The
kernels already give dims ``[10, d_aug)`` free ARD lengthscales; these tests pin
the parts that must agree with them: the input transform has to find Time by
name, and block-LOO has to keep a strength curve's ages together.

The extras here are synthetic and constant along each curve:

* ``informative`` is the composition's mean strength plus a little noise;
* ``noise`` is unrelated to strength.
"""

from __future__ import annotations

import unittest

import torch

from boxcrete import fit_strength_gp, load_concrete_strength
from boxcrete.block_loo import _block_fingerprints, block_loo_loss, train_block_loo
from boxcrete.features import F5_ALLLOG_FEATURES, IDX
from boxcrete.utils import DEFAULT_X_COLUMNS

_N_RAW = len(DEFAULT_X_COLUMNS)
# Index of the first extra in the blind Matern's lengthscales, which span the
# non-source raw dims (9 of them) and then the extras.
_FIRST_EXTRA_LENGTHSCALE = _N_RAW - 1


def _curve_level_extras(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """``[n, 2]`` columns (informative, noise), constant along each curve."""
    generator = torch.Generator().manual_seed(0)
    _, inverse = torch.unique(X[:, : IDX["time"]], dim=0, return_inverse=True)
    n_curves = int(inverse.max()) + 1
    totals = torch.zeros(n_curves, dtype=Y.dtype).index_add_(0, inverse, Y[:, 0])
    means = totals / torch.bincount(inverse, minlength=n_curves).to(Y.dtype)
    informative = means / means.max() + 0.02 * torch.randn(
        n_curves, generator=generator, dtype=Y.dtype
    )
    noise = torch.randn(n_curves, generator=generator, dtype=Y.dtype)
    return torch.stack([informative[inverse], noise[inverse]], dim=-1)


def _with_extras(
    X: torch.Tensor, bounds: torch.Tensor, extras: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    extra_bounds = torch.stack([extras.amin(dim=0), extras.amax(dim=0)])
    return torch.cat([X, extras], dim=-1), torch.cat([bounds, extra_bounds], dim=-1)


def _blind_lengthscales(model) -> torch.Tensor:
    blind = model.covar_module.base_kernel.kernels[0]
    return blind.base_kernel.lengthscale.detach().squeeze()


class TestDefaultInputsUnchanged(unittest.TestCase):
    """With the 10 default columns the new code paths must be no-ops."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.X, cls.Y, cls.Yvar, cls.bounds = load_concrete_strength().strength_data

    def test_block_fingerprints_are_the_composition_dims(self):
        fingerprints = _block_fingerprints(self.X, IDX["time"])
        self.assertTrue(torch.equal(fingerprints, self.X[..., : IDX["time"]]))

    def test_time_is_still_the_only_log_transformed_column(self):
        model = fit_strength_gp(
            X=self.X,
            Y=self.Y,
            Yvar=self.Yvar,
            X_bounds=self.bounds,
            max_optimizer_iter=0,
        )
        self.assertEqual(model.input_transform["log"].indices.tolist(), [IDX["time"]])
        d_aug = _N_RAW + len(F5_ALLLOG_FEATURES)
        skipped = set(range(d_aug)) - set(
            model.input_transform["normalize"].indices.tolist()
        )
        self.assertEqual(skipped, {IDX["time"], IDX["source"]})


class TestBlockFingerprintsWithExtras(unittest.TestCase):
    """Which rows are held out together. No model needed."""

    def setUp(self):
        # Two curves with the same composition, each measured at two ages,
        # separated only by the extra input in the last column.
        composition = [float(i) for i in range(IDX["time"])]
        self.X = torch.tensor(
            [
                [*composition, 1.0, 5.0],
                [*composition, 28.0, 5.0],
                [*composition, 1.0, 6.0],
                [*composition, 28.0, 6.0],
            ],
            dtype=torch.double,
        )

    def _block_sizes(self, X: torch.Tensor, n_composition_dims: int) -> list[int]:
        _, counts = torch.unique(
            _block_fingerprints(X, n_composition_dims), dim=0, return_counts=True
        )
        return counts.tolist()

    def test_ages_of_one_curve_stay_together(self):
        self.assertEqual(self._block_sizes(self.X, IDX["time"]), [2, 2])

    def test_extra_that_changes_along_a_curve_splits_it(self):
        """The documented caveat: per-row extras reduce block-LOO to row LOO."""
        X = self.X.clone()
        X[:, -1] = torch.arange(4, dtype=X.dtype)
        self.assertEqual(self._block_sizes(X, IDX["time"]), [1, 1, 1, 1])

    def test_time_never_defines_a_block_when_fewer_dims_are_requested(self):
        """``n_composition_dims`` counts leading dims; Time stays excluded even
        when the dim right after them is not Time."""
        fingerprints = _block_fingerprints(self.X, IDX["temp"])
        self.assertEqual(fingerprints.shape[-1], IDX["temp"] + 1)
        self.assertEqual(self._block_sizes(self.X, IDX["temp"]), [2, 2])


class TestFitWithExtras(unittest.TestCase):
    """End-to-end fits on the public data with two synthetic extras."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.X, cls.Y, _, bounds = load_concrete_strength().strength_data
        cls.extras = _curve_level_extras(cls.X, cls.Y)
        cls.X_extra, cls.bounds_extra = _with_extras(cls.X, bounds, cls.extras)

    def _untrained_model(self):
        return fit_strength_gp(
            X=self.X_extra,
            Y=self.Y,
            X_bounds=self.bounds_extra,
            max_optimizer_iter=0,
        )

    def test_transform_treats_time_and_extras_correctly(self):
        model = self._untrained_model()
        transform = model.input_transform
        self.assertEqual(transform["log"].indices.tolist(), [IDX["time"]])
        d_aug = self.X_extra.shape[-1] + len(F5_ALLLOG_FEATURES)
        skipped = set(range(d_aug)) - set(transform["normalize"].indices.tolist())
        self.assertEqual(skipped, {IDX["time"], IDX["source"]})

        with torch.no_grad():
            out = transform(self.X_extra)
        self.assertEqual(out.shape[-1], d_aug)
        # Time is log10(t + 1), exactly as with the default inputs.
        expected_time = torch.log10(self.X_extra[..., IDX["time"]] + 1.0)
        self.assertTrue(torch.allclose(out[..., IDX["time"]], expected_time))
        # Negative extras must not reach the time column's log.
        self.assertLess(float(self.extras.min()), -1.0)
        self.assertFalse(out.isnan().any())
        # Extras are normalised to [0, 1] by their bounds.
        extras_out = out[..., _N_RAW : self.X_extra.shape[-1]]
        self.assertAlmostEqual(float(extras_out.min()), 0.0)
        self.assertAlmostEqual(float(extras_out.max()), 1.0)

    def test_bounds_narrower_than_x_are_rejected(self):
        """The natural mistake: extras in X, the dataset's bounds unchanged.
        With exactly 2 extras ``Normalize`` would accept them and apply them to
        the wrong columns, so the fit has to refuse."""
        with self.assertRaisesRegex(ValueError, "one column per column of X"):
            fit_strength_gp(
                X=self.X_extra,
                Y=self.Y,
                X_bounds=self.bounds_extra[..., :_N_RAW],
                max_optimizer_iter=0,
            )

    def test_bounds_are_derived_for_extras_when_omitted(self):
        model = fit_strength_gp(X=self.X_extra, Y=self.Y, max_optimizer_iter=0)
        with torch.no_grad():
            out = model.input_transform(self.X_extra)
        extras_out = out[..., _N_RAW : self.X_extra.shape[-1]]
        self.assertAlmostEqual(float(extras_out.min()), 0.0)
        self.assertAlmostEqual(float(extras_out.max()), 1.0)

    def test_each_extra_gets_its_own_lengthscale(self):
        model = self._untrained_model()
        n_extra = self.extras.shape[-1]
        expected = _N_RAW - 1 + n_extra + len(F5_ALLLOG_FEATURES)
        self.assertEqual(_blind_lengthscales(model).numel(), expected)

    def test_predictions_are_finite_with_extras(self):
        model = self._untrained_model()
        with torch.no_grad():
            posterior = model.posterior(self.X_extra[:5])
        self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))
        self.assertTrue(posterior.mean.isfinite().all())
        self.assertTrue(posterior.variance.isfinite().all())

    def test_block_loo_runs_and_improves_with_extras(self):
        """Curve-level extras leave one block per composition, and the training
        objective can be evaluated and reduced through the extras' lengthscales.
        """
        model = self._untrained_model()
        model.train()
        n = self.X_extra.shape[0]
        with torch.no_grad():
            before = block_loo_loss(model, n).item()
        n_blocks = torch.unique(
            _block_fingerprints(model.train_inputs[0], IDX["time"]), dim=0
        ).shape[0]
        n_compositions = torch.unique(self.X[:, : IDX["time"]], dim=0).shape[0]
        self.assertEqual(n_blocks, n_compositions)

        after = train_block_loo(model, n, max_iter=5)
        self.assertTrue(torch.isfinite(torch.tensor(after)))
        self.assertLess(after, before)

    def test_informative_extra_gets_a_shorter_lengthscale_than_noise(self):
        model = fit_strength_gp(
            X=self.X_extra,
            Y=self.Y,
            X_bounds=self.bounds_extra,
            max_optimizer_iter=50,
        )
        lengthscales = _blind_lengthscales(model)
        informative = lengthscales[_FIRST_EXTRA_LENGTHSCALE].item()
        noise = lengthscales[_FIRST_EXTRA_LENGTHSCALE + 1].item()
        # A shorter lengthscale means the kernel relies on that input more. The
        # factor leaves room for BLAS drift across hosts; see the fit logs in
        # the pull request for the measured values.
        self.assertLess(informative * 2.0, noise)


if __name__ == "__main__":
    unittest.main()
