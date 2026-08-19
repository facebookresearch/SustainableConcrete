"""Python reference for the browser's multi-objective BO demo.

``docs/bo.mjs`` runs the demo client-side. This module is the authority it is
checked against: the same quantities, written independently in Python, and
cross-validated here against BoTorch. ``experiments/regenerate_bo_golden.py``
freezes the outputs into ``test/fixtures/bo_golden.json``, which the JS suite
asserts against, so a future edit to either implementation that changes a
number has to be deliberate.

Convention, matching the website's scatter plot: **minimise x** (GWP or cost,
known exactly from a linear model) and **maximise y** (strength, the GP
posterior). The reference point is therefore an upper bound on x and a lower
bound on y. BoTorch maximises everything, so the bridge is ``obj = (-x, y)``
with ``ref = (-refX, refY)``; hypervolume is invariant under that reflection.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "pareto_staircase",
    "hypervolume_2d",
    "expected_improvement",
    "expected_hvi",
    "extend_cholesky_block",
]


def pareto_staircase(xs: np.ndarray, ys: np.ndarray) -> list[tuple[float, float]]:
    """Non-dominated staircase, ascending in x and therefore in y.

    Sorted by ascending x then *descending* y: on a tie in x the weaker point
    is strictly dominated, and taking the stronger one first lets the
    running-maximum filter drop it.
    """
    order = sorted(range(len(xs)), key=lambda i: (xs[i], -ys[i]))
    front: list[tuple[float, float]] = []
    best = -math.inf
    for i in order:
        if ys[i] > best:
            front.append((float(xs[i]), float(ys[i])))
            best = ys[i]
    return front


def hypervolume_2d(xs: np.ndarray, ys: np.ndarray, ref_x: float, ref_y: float) -> float:
    """Dominated hypervolume bounded by ``(ref_x, ref_y)``.

    Exactly zero, never NaN, when nothing clears the reference point -- the
    real situation early in a run under ``CONCRETE_REFERENCE_POINT``.
    """
    keep = [i for i in range(len(xs)) if xs[i] <= ref_x and ys[i] >= ref_y]
    front = pareto_staircase(np.asarray([xs[i] for i in keep]), [ys[i] for i in keep])
    total = 0.0
    for k, (x, y) in enumerate(front):
        x_hi = front[k + 1][0] if k + 1 < len(front) else ref_x
        total += (x_hi - x) * (y - ref_y)
    return total


def expected_improvement(mu: float, sd: float, threshold: float) -> float:
    """``E[max(0, Y - threshold)]`` for ``Y ~ N(mu, sd**2)``."""
    gap = mu - threshold
    if sd <= 0:
        return max(0.0, gap)
    z = gap / sd
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * math.erfc(-z / math.sqrt(2.0))
    return sd * pdf + gap * cdf


def expected_hvi(
    mu: float,
    sd: float,
    g: float,
    front: list[tuple[float, float]],
    ref_x: float,
    ref_y: float,
) -> float:
    """Closed-form expected hypervolume improvement.

    Only y is uncertain, so with x fixed at ``g`` the improvement is
    piecewise-linear in y -- one piece per staircase segment right of ``g`` --
    and the expectation passes straight through the sum::

        E[HVI] = sum over segments of  width * EI(mu, sd; segment ceiling)

    Exact and O(P). No quadrature, no sampling.
    """
    if g > ref_x:
        return 0.0
    total = 0.0
    cursor = g
    ceiling = ref_y
    for x, y in front:
        if x > ref_x or y < ref_y:
            continue
        if x > g:
            total += (x - cursor) * expected_improvement(mu, sd, ceiling)
            cursor = x
        ceiling = max(ceiling, y)
    total += (ref_x - cursor) * expected_improvement(mu, sd, ceiling)
    return total


def extend_cholesky_block(
    L: Tensor, K_na: Tensor, K_nn: Tensor
) -> tuple[Tensor, Tensor]:
    """Grow a Cholesky factor by a block of new rows.

    ``L_b = L^-1 K_na`` and ``L_nn = chol(K_nn - L_b^T L_b)``, so the full
    factor is ``[[L, 0], [L_b^T, L_nn]]``. This is the O(n^2 b) step that lets
    the demo condition on a new mix without an O(n^3) refit.

    Args:
        L: ``[n, n]`` lower-triangular factor of the current Gram matrix.
        K_na: ``[n, b]`` cross-covariance between current and new rows.
        K_nn: ``[b, b]`` self-covariance of the new rows, noise already added.
    """
    L_b = torch.linalg.solve_triangular(L, K_na, upper=False)
    schur = K_nn - L_b.T @ L_b
    return L_b, torch.linalg.cholesky(schur)
