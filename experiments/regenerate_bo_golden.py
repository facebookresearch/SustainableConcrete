"""Regenerate ``test/fixtures/bo_golden.json``.

Freezes the Python reference's expected-hypervolume-improvement and
hypervolume outputs, cross-validated against BoTorch, so ``docs/bo.mjs`` can be
pinned to them from the JS suite. Run from the repo root after any deliberate
change to :mod:`boxcrete.bo_reference`::

    python -m experiments.regenerate_bo_golden

``-m`` rather than ``python experiments/regenerate_bo_golden.py``: the latter
puts ``experiments/`` on ``sys.path`` instead of the repo root, so ``boxcrete``
resolves through whatever ``pip install -e .`` last pointed at. In a git
worktree that is the *primary* checkout, and the script would silently import a
different tree's ``boxcrete`` than the one being edited. ``-m`` puts the
current directory first, so the checkout you are standing in is the one that
gets used.

Every EHVI case is checked against BoTorch's analytic
``ExpectedHypervolumeImprovement`` before it is written, so a fixture that
lands in the tree has already been validated against an outside authority.
"""

import json
import pathlib

import numpy as np
import torch
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.testing import MockModel, MockPosterior

from boxcrete.bo_reference import expected_hvi, hypervolume_2d, pareto_staircase

# boxcrete.CONCRETE_REFERENCE_POINT is [-200, 1000, 5000]; the 28-day slice in
# natural (un-negated) units is GWP <= 200 kg CO2e/m3 and strength >= 5000 psi.
REF_X, REF_Y = 200.0, 5000.0

# Absolute floor for the BoTorch cross-check, ~14 orders below the hypervolume
# scale this problem works at.
ABS_TOL = 1e-9

OBSERVED_X = [100.0, 150.0, 175.0]
OBSERVED_Y = [6000.0, 8000.0, 8200.0]

# (mu, sd, g, botorch_rtol). The tolerance is per-case because BoTorch's
# analytic EHVI is being driven through a MockPosterior whose "deterministic"
# objective actually carries variance 1e-18. Where a candidate's g coincides
# exactly with an observed point's x, that sliver of variance straddles a
# box-decomposition boundary and BoTorch loses about seven digits.
#
# Established, not assumed: at g=175.0 (which is an observed x) the closed form
# and BoTorch differ by 1.0e-7, while independent scipy quadrature of
# HVI(y) * N(y; mu, sd) agrees with the closed form to 3.4e-16. Nudging g to
# 174.9 or 175.1 restores exact BoTorch agreement. So the closed form is right
# and the mock is the imprecise party. The boundary case is kept -- it is worth
# testing -- with a tolerance that reflects the mock's real accuracy there.
EHVI_CASES = [
    (7000.0, 1500.0, 120.0, 1e-9),
    (5200.0, 2000.0, 90.0, 1e-9),
    (9000.0, 800.0, 175.0, 1e-6),  # g sits exactly on an observed x
    (5500.0, 100.0, 130.0, 1e-9),  # several sigma under the ceiling: must vanish
    (12000.0, 50.0, 60.0, 1e-9),  # dominates everything: near-deterministic
    (8000.0, 3000.0, 199.0, 1e-9),  # hard against the reference bound
    (20000.0, 500.0, 250.0, 1e-9),  # outside the box entirely
]

HV_CASES = [
    ([], []),
    ([150.0], [8000.0]),
    ([100.0, 150.0], [6000.0, 8000.0]),
    ([100.0, 150.0, 180.0], [6000.0, 8000.0, 5500.0]),  # includes a dominated point
    ([250.0, 300.0], [4000.0, 4500.0]),  # all outside the box
    ([100.0, 100.0], [5000.0, 8000.0]),  # tie in x
]


def botorch_ehvi(mu, sd, g, obs_x, obs_y):
    """Analytic EHVI from BoTorch, treating x as a deterministic outcome."""
    Y = torch.tensor([[-x, y] for x, y in zip(obs_x, obs_y)], dtype=torch.double)
    ref = torch.tensor([-REF_X, REF_Y], dtype=torch.double)
    partitioning = NondominatedPartitioning(ref_point=ref, Y=Y)
    mean = torch.tensor([[-g, mu]], dtype=torch.double).unsqueeze(0)
    var = torch.tensor([[1e-18, sd**2]], dtype=torch.double).unsqueeze(0)
    acq = ExpectedHypervolumeImprovement(
        model=MockModel(MockPosterior(mean=mean, variance=var)),
        ref_point=ref.tolist(),
        partitioning=partitioning,
    )
    return float(acq(torch.zeros(1, 1, 1, dtype=torch.double)))


def main():
    front = pareto_staircase(np.array(OBSERVED_X), np.array(OBSERVED_Y))

    ehvi_cases = []
    for mu, sd, g, rtol in EHVI_CASES:
        ours = expected_hvi(mu, sd, g, front, REF_X, REF_Y)
        theirs = botorch_ehvi(mu, sd, g, OBSERVED_X, OBSERVED_Y)
        # Relative tolerance alone is the wrong yardstick deep in the tail. At
        # mu=5500, sd=100, g=130 the candidate is 5 sigma under the ceiling and
        # EHVI is ~1e-4, where two different erf implementations differ by
        # 6.7e-13 absolute -- i.e. 6e-9 relative, but utterly negligible against
        # a hypervolume scale of ~1e5. So a value only has to clear ONE of the
        # two tolerances.
        gap = abs(ours - theirs)
        if gap > rtol * max(abs(theirs), 1e-12) and gap > ABS_TOL:
            raise SystemExit(
                f"Refusing to write the fixture: closed form {ours} disagrees with "
                f"BoTorch {theirs} at mu={mu} sd={sd} g={g} "
                f"(gap {gap:.3e}, rtol {rtol}, atol {ABS_TOL})."
            )
        ehvi_cases.append(
            {
                "mu": mu,
                "sd": sd,
                "g": g,
                "ehvi": ours,
                "botorch_ehvi": theirs,
                "botorch_rtol": rtol,
            }
        )

    hv_cases = [
        {
            "xs": xs,
            "ys": ys,
            "hv": hypervolume_2d(np.array(xs), np.array(ys), REF_X, REF_Y),
        }
        for xs, ys in HV_CASES
    ]

    payload = {
        "_comment": (
            "Generated by experiments/regenerate_bo_golden.py. Convention: "
            "minimise x (GWP), maximise y (strength); ref_x is an upper bound "
            "and ref_y a lower bound. Every ehvi value was checked against "
            "BoTorch's analytic ExpectedHypervolumeImprovement before writing."
        ),
        "ref_x": REF_X,
        "ref_y": REF_Y,
        "observed_x": OBSERVED_X,
        "observed_y": OBSERVED_Y,
        "ehvi_cases": ehvi_cases,
        "hypervolume_cases": hv_cases,
    }

    out = pathlib.Path(__file__).resolve().parents[1] / "test" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bo_golden.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {path} ({len(ehvi_cases)} EHVI cases, {len(hv_cases)} HV cases).")


if __name__ == "__main__":
    main()
