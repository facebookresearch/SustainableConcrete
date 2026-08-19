"""Tests for :mod:`boxcrete.bo_reference`.

The load-bearing test here is :func:`test_expected_hvi_matches_botorch`. Every
other check confirms the reference is self-consistent; that one confirms it is
*right*, by pinning the closed form to BoTorch's analytic
``ExpectedHypervolumeImprovement``.
"""

import json
import math
import pathlib

import numpy as np
import pytest
import torch
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.testing import MockModel, MockPosterior

from boxcrete.bo_reference import (
    expected_hvi,
    expected_improvement,
    extend_cholesky_block,
    hypervolume_2d,
    pareto_staircase,
)

REF_X, REF_Y = 200.0, 5000.0
OBS_X = np.array([100.0, 150.0])
OBS_Y = np.array([6000.0, 8000.0])


def _botorch_ehvi(mu, sd, g, obs_x, obs_y, ref_x=REF_X, ref_y=REF_Y):
    """Analytic EHVI from BoTorch, with x as a (near) deterministic outcome.

    BoTorch maximises both objectives and treats ``ref_point`` as a lower bound
    on both, so we pass ``(-x, y)`` against ``(-ref_x, ref_y)``. The known
    objective gets a negligible variance rather than exactly zero, which keeps
    the analytic form well conditioned.
    """
    Y = torch.tensor([[-x, y] for x, y in zip(obs_x, obs_y)], dtype=torch.double)
    ref = torch.tensor([-ref_x, ref_y], dtype=torch.double)
    partitioning = NondominatedPartitioning(ref_point=ref, Y=Y)
    mean = torch.tensor([[-g, mu]], dtype=torch.double).unsqueeze(0)
    var = torch.tensor([[1e-18, sd**2]], dtype=torch.double).unsqueeze(0)
    acq = ExpectedHypervolumeImprovement(
        model=MockModel(MockPosterior(mean=mean, variance=var)),
        ref_point=ref.tolist(),
        partitioning=partitioning,
    )
    return float(acq(torch.zeros(1, 1, 1, dtype=torch.double)))


class TestParetoStaircase:
    def test_empty(self):
        assert pareto_staircase(np.array([]), np.array([])) == []

    def test_keeps_genuine_tradeoff(self):
        front = pareto_staircase(np.array([100.0, 150.0]), np.array([5000.0, 8000.0]))
        assert front == [(100.0, 5000.0), (150.0, 8000.0)]

    def test_drops_dominated_point(self):
        front = pareto_staircase(np.array([100.0, 150.0]), np.array([8000.0, 5000.0]))
        assert front == [(100.0, 8000.0)]

    def test_tie_in_x_keeps_higher_y(self):
        front = pareto_staircase(np.array([100.0, 100.0]), np.array([5000.0, 8000.0]))
        assert front == [(100.0, 8000.0)]


class TestHypervolume:
    def test_single_point(self):
        assert hypervolume_2d(np.array([150.0]), np.array([8000.0]), REF_X, REF_Y) == (
            50.0 * 3000.0
        )

    def test_two_step_staircase(self):
        assert hypervolume_2d(OBS_X, OBS_Y, REF_X, REF_Y) == 200000.0

    def test_zero_outside_reference_box(self):
        hv = hypervolume_2d(np.array([250.0]), np.array([4000.0]), REF_X, REF_Y)
        assert hv == 0.0
        assert not math.isnan(hv)

    def test_matches_botorch_hypervolume(self):
        from botorch.utils.multi_objective.hypervolume import Hypervolume

        Y = torch.tensor([[-x, y] for x, y in zip(OBS_X, OBS_Y)], dtype=torch.double)
        hv = Hypervolume(ref_point=torch.tensor([-REF_X, REF_Y], dtype=torch.double))
        assert hv.compute(Y) == pytest.approx(
            hypervolume_2d(OBS_X, OBS_Y, REF_X, REF_Y), rel=1e-12
        )


class TestExpectedImprovement:
    def test_deterministic_branch(self):
        assert expected_improvement(5000.0, 0.0, 3000.0) == 2000.0
        assert expected_improvement(2000.0, 0.0, 3000.0) == 0.0

    def test_at_threshold(self):
        assert expected_improvement(3000.0, 500.0, 3000.0) == pytest.approx(
            500.0 / math.sqrt(2.0 * math.pi)
        )

    def test_never_negative(self):
        assert expected_improvement(1000.0, 200.0, 9000.0) >= 0.0


class TestExpectedHVI:
    @pytest.mark.parametrize(
        "mu, sd, g",
        [(7000.0, 1500.0, 120.0), (5200.0, 2000.0, 90.0), (9000.0, 800.0, 175.0)],
    )
    def test_expected_hvi_matches_botorch(self, mu, sd, g):
        """The claim the whole acquisition function rests on."""
        front = pareto_staircase(OBS_X, OBS_Y)
        ours = expected_hvi(mu, sd, g, front, REF_X, REF_Y)
        theirs = _botorch_ehvi(mu, sd, g, OBS_X, OBS_Y)
        assert ours == pytest.approx(theirs, rel=1e-9)

    def test_reduces_to_deterministic_hvi(self):
        front = pareto_staircase(OBS_X, OBS_Y)
        base = hypervolume_2d(OBS_X, OBS_Y, REF_X, REF_Y)
        exact = (
            hypervolume_2d(
                np.append(OBS_X, 120.0), np.append(OBS_Y, 9000.0), REF_X, REF_Y
            )
            - base
        )
        assert expected_hvi(9000.0, 0.0, 120.0, front, REF_X, REF_Y) == pytest.approx(
            exact
        )

    def test_zero_beyond_reference_x(self):
        front = pareto_staircase(OBS_X, OBS_Y)
        assert expected_hvi(20000.0, 500.0, 250.0, front, REF_X, REF_Y) == 0.0

    def test_empty_front_is_the_bare_reference_box(self):
        got = expected_hvi(8000.0, 1000.0, 150.0, [], REF_X, REF_Y)
        want = 50.0 * expected_improvement(8000.0, 1000.0, REF_Y)
        assert got == pytest.approx(want)

    def test_ignores_front_points_outside_the_box(self):
        clean = pareto_staircase(OBS_X, OBS_Y)
        dirty = pareto_staircase(
            np.array([100.0, 150.0, 260.0, 80.0]),
            np.array([6000.0, 8000.0, 12000.0, 100.0]),
        )
        assert expected_hvi(9000.0, 600.0, 130.0, dirty, REF_X, REF_Y) == pytest.approx(
            expected_hvi(9000.0, 600.0, 130.0, clean, REF_X, REF_Y)
        )


class TestExtendCholeskyBlock:
    @pytest.mark.parametrize("n, b", [(6, 1), (6, 3), (10, 5)])
    def test_matches_a_direct_factorisation(self, n, b):
        torch.manual_seed(n * 100 + b)
        m = n + b
        G = torch.randn(m, m, dtype=torch.double)
        K = G @ G.T + m * torch.eye(m, dtype=torch.double)
        L = torch.linalg.cholesky(K[:n, :n])
        L_b, L_nn = extend_cholesky_block(L, K[:n, n:], K[n:, n:])
        full = torch.zeros(m, m, dtype=torch.double)
        full[:n, :n] = L
        full[n:, :n] = L_b.T
        full[n:, n:] = L_nn
        assert torch.allclose(full @ full.T, K, atol=1e-8)


def test_golden_fixture_is_in_sync_with_this_module():
    """The committed fixture must still describe what the reference computes.

    Guards against the fixture and the reference drifting apart -- if this
    fails, re-run ``experiments/regenerate_bo_golden.py``.
    """
    path = pathlib.Path(__file__).parent / "fixtures" / "bo_golden.json"
    golden = json.loads(path.read_text())
    front = pareto_staircase(
        np.array(golden["observed_x"]), np.array(golden["observed_y"])
    )
    for case in golden["ehvi_cases"]:
        got = expected_hvi(
            case["mu"],
            case["sd"],
            case["g"],
            front,
            golden["ref_x"],
            golden["ref_y"],
        )
        assert got == pytest.approx(case["ehvi"], rel=1e-12)
    for case in golden["hypervolume_cases"]:
        got = hypervolume_2d(
            np.array(case["xs"]), np.array(case["ys"]), golden["ref_x"], golden["ref_y"]
        )
        assert got == pytest.approx(case["hv"], rel=1e-12)
