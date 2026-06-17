# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for :class:`boxcrete.priors.ComposedLengthscalePrior`."""

from __future__ import annotations

import math

import pytest
import torch

from boxcrete.priors import (
    ComposedLengthscalePrior,
    WithinGroupShrinkagePrior,
    _AGGREGATE_LENGTHSCALE_GROUP,
    _BINDER_LENGTHSCALE_GROUP,
    _LENGTHSCALE_SHRINKAGE_SIGMA,
    _LOGNORMAL_BASELINE_SCALE,
    _lognormal_baseline_loc,
    within_group_prior,
)


@pytest.fixture
def production_groups():
    return [
        (_BINDER_LENGTHSCALE_GROUP, _LENGTHSCALE_SHRINKAGE_SIGMA),
        (_AGGREGATE_LENGTHSCALE_GROUP, _LENGTHSCALE_SHRINKAGE_SIGMA),
    ]


def _manual_within_total(x_flat: torch.Tensor, groups) -> torch.Tensor:
    log_x = x_flat.log()
    total = torch.zeros((), dtype=x_flat.dtype)
    for grp, sigma in groups:
        if len(grp) < 2:
            continue
        gl = log_x[list(grp)]
        total = total - 0.5 * ((gl - gl.mean()) ** 2).sum() / (sigma**2)
    return total


def _manual_lognormal_logpdf(x: torch.Tensor, loc: float, scale: float) -> torch.Tensor:
    log_x = x.log()
    return (
        -0.5 * ((log_x - loc) / scale) ** 2
        - log_x
        - math.log(scale * math.sqrt(2 * math.pi))
    )


def test_log_prob_shape_matches_input(production_groups):
    prior = ComposedLengthscalePrior(production_groups, dim=10)
    x = torch.full((1, 10), 1.5, dtype=torch.double)
    out = prior.log_prob(x)
    assert out.shape == x.shape


def test_log_prob_sum_equals_lognormal_plus_within_group(production_groups):
    """Plan §"Commit 4" verification: sum equals
    ``lognormal_full + shrinkage_total``."""
    prior = ComposedLengthscalePrior(production_groups, dim=10)
    x = torch.tensor(
        [[1.5, 2.0, 1.8, 0.9, 1.1, 2.5, 2.3, 1.0, 1.0, 1.0]],
        dtype=torch.double,
    )
    composed = prior.log_prob(x).sum()
    loc = _lognormal_baseline_loc(10)
    lognormal_total = _manual_lognormal_logpdf(x, loc, _LOGNORMAL_BASELINE_SCALE).sum()
    within_total = _manual_within_total(x.flatten(), production_groups)
    expected = lognormal_total + within_total
    assert torch.isclose(composed, expected, atol=1e-6)


def test_log_prob_gradient_flows_through_both_terms(production_groups):
    prior = ComposedLengthscalePrior(production_groups, dim=10)
    x = torch.full((1, 10), 1.5, dtype=torch.double, requires_grad=True)
    loss = prior.log_prob(x).sum()
    loss.backward()
    assert x.grad is not None
    # Every element should have a non-zero gradient (LogNormal contribution).
    assert (x.grad.abs() > 1e-8).all()


def test_factory_default_returns_composed():
    prior = within_group_prior(d_in=10)
    assert isinstance(prior, ComposedLengthscalePrior)


def test_factory_opt_out_returns_within_group_only():
    prior = within_group_prior(d_in=10, include_lognormal_baseline=False)
    assert isinstance(prior, WithinGroupShrinkagePrior)
    assert not isinstance(prior, ComposedLengthscalePrior)


def test_factory_handles_source_dim_exclusion():
    """``source_dim`` is excluded from the input; group indices are
    remapped to skip it. With the default ``DEFAULT_X_COLUMNS`` ordering
    the binder ``{0,1,2}`` and aggregate ``{5,6}`` groups are unchanged
    because ``source_dim=7`` sits AFTER the groups.
    """
    prior = within_group_prior(d_in=10, source_dim=7, include_lognormal_baseline=True)
    assert isinstance(prior, ComposedLengthscalePrior)
    # Effective dim is 9 (10 minus the source column).
    assert prior._dim == 9
    assert prior._groups_with_sigma[0][0] == (0, 1, 2)
    assert prior._groups_with_sigma[1][0] == (5, 6)


def test_lognormal_baseline_loc_matches_botorch_default():
    # BoTorch ARD-Matern default: sqrt(2) + 0.5 * ln(d).
    for d in (1, 9, 10, 17):
        assert _lognormal_baseline_loc(d) == pytest.approx(
            math.sqrt(2.0) + 0.5 * math.log(d)
        )


def test_lognormal_baseline_scale_is_sqrt_3():
    assert _LOGNORMAL_BASELINE_SCALE == pytest.approx(math.sqrt(3.0))


def test_composed_prior_pulls_lengthscales_to_sqrt_d_mode(production_groups):
    """Sanity: argmax of the LogNormal density (dlog_prob/dx = 0) is at
    sqrt(d) for the default loc/scale (mode formula:
    ``mode = exp(loc - scale^2)``; with loc = sqrt(2) + 0.5 ln d and
    scale = sqrt(3), mode = exp(sqrt(2) - 3) * sqrt(d)). Verify the
    LogNormal half of the composed prior's gradient at lengthscale =
    sqrt(d)·exp(sqrt(2)-3) is approximately zero (excluding the
    within-group contribution at uniform input).
    """
    d = 10
    target_mode = math.exp(math.sqrt(2.0) - 3.0) * math.sqrt(d)
    # Use uniform lengthscales so the within-group penalty contributes 0.
    x = torch.full((1, d), target_mode, dtype=torch.double, requires_grad=True)
    prior = ComposedLengthscalePrior(production_groups, dim=d)
    prior.log_prob(x).sum().backward()
    # Each element's gradient should be small (we're at the LogNormal mode
    # AND the within-group penalty is 0 at uniform input).
    assert x.grad.abs().max().item() < 1e-3


def test_composed_prior_within_group_dominates_over_lognormal_when_groups_disagree(
    production_groups,
):
    """If we set up a configuration where group members disagree wildly,
    the within-group penalty should produce a much larger negative
    contribution than the LogNormal baseline."""
    d = 10
    prior = ComposedLengthscalePrior(production_groups, dim=d)
    # Disagreement: cement = 1, fly_ash = 100 (factor 100 difference).
    x_disagree = torch.full((1, d), 1.0, dtype=torch.double)
    x_disagree[0, 1] = 100.0  # fly ash way larger
    x_agree = torch.full((1, d), 1.0, dtype=torch.double)
    lp_disagree = prior.log_prob(x_disagree).sum().item()
    lp_agree = prior.log_prob(x_agree).sum().item()
    # Disagreement should be MUCH less likely.
    assert lp_disagree < lp_agree - 100.0


def test_composed_prior_state_dict_roundtrip_no_aliased_storage_error(
    production_groups,
):
    """The pre-instantiation contiguous-tensor pattern (matching
    :class:`WithinGroupShrinkagePrior`) protects against PyTorch 2.12+'s
    ``load_state_dict`` aliased-storage error.
    """
    prior = ComposedLengthscalePrior(production_groups, dim=10)
    state = {k: v.clone() for k, v in prior.state_dict().items()}
    prior.load_state_dict(state)  # should not raise
