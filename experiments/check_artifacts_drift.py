# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-architecture-portable replacement for ``git diff --exit-code docs/model/``.

The original byte-level idempotency check assumes that two consecutive runs
of ``bash experiments/regenerate_all_artifacts.sh`` on different architectures
produce bit-identical JSON output. In practice they don't: the V2 strength
GP fit goes through scipy's L-BFGS-B, which depends on BLAS rounding;
different CPU architectures (Apple Silicon via Docker amd64-emulation,
GitHub-runner native x86_64, Intel/AMD desktops) land in different local
optima of the multi-modal MLL surface and emit lengthscales that can differ
by 2x while predicting nearly the same surface. See the docstring of
``test/test_lengthscale_identifiability.py::test_committed_strength_matches_fresh_fit``
for the full discussion.

This script replaces the byte-level diff with a numerical-tolerance check
that catches the failure modes the original test cared about:

  * stale artifacts: committed strength.json / test_vectors.json /
    compositions.json predictions disagree with what the current model code
    produces.
  * model-architecture changes that flip the prediction surface
    (kernel structure changes, prior changes, feature engineering changes).

while tolerating cross-architecture optimizer divergence:

  * lengthscale drift (basins differ): allowed within a 10x ratio band.
  * prediction drift (predictions are stable across basins): allowed
    within ``max(250 psi, 5% rel)`` per element. Same gauge as
    ``test_committed_strength_matches_fresh_fit``.

Usage:
    python experiments/check_artifacts_drift.py \\
        --committed-dir /tmp/committed_docs_model \\
        --fresh-dir docs/model

The CI workflow saves the committed copies before running regen, then
points this script at both directories.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Tolerances. Keep these in sync with
# test/test_lengthscale_identifiability.py and
# test/test_lengthscales_v2.mjs.
# ---------------------------------------------------------------------------

# Lengthscales: a 10x ratio band tolerates cross-arch basin divergence
# while still catching gross model-architecture changes (which typically
# perturb lengthscales by orders of magnitude).
LENGTHSCALE_RATIO_LO = 0.1
LENGTHSCALE_RATIO_HI = 10.0
LENGTHSCALE_ATOL = 1e-3  # absorb divide-by-near-zero on prior-railed dims

# Predictions: max(250 psi, 5% relative). Sized empirically against the
# cross-runner / cross-architecture drift envelope observed across this
# repo's CI saga (worst case ~230 psi absolute / ~4.3% relative on
# certain GitHub-hosted ubuntu-latest VM hosts). Different physical
# Azure hosts have slightly different CPU rounding, which feeds into
# L-BFGS-B and occasionally lands the multi-modal MLL fit in a
# different local optimum even with float64 throughout. The 250 psi
# absolute floor handles low-magnitude predictions cleanly; the 5%
# relative ceiling still catches structural model changes (which
# typically shift OOT predictions by 10-50%+).
PREDICTION_PSI_FLOOR = 250.0
PREDICTION_RTOL = 0.05

# Variances: looser than means because variance is on a different scale
# (psi^2). A 1% rtol on a variance of 200_000 psi^2 = 2000 psi^2 absolute,
# which is well above the empirical worst-case cross-arch drift.
VARIANCE_PSI2_FLOOR = 2_000.0
VARIANCE_RTOL = 0.05

# Outputscales / noise / scalar hyperparameters: same 10x ratio band as
# lengthscales — they're internal model parameters that drift with
# basin, not predictions.
SCALAR_PARAM_RATIO_LO = 0.1
SCALAR_PARAM_RATIO_HI = 10.0


# ---------------------------------------------------------------------------
# Failure / pass collection. We accumulate every drift, print a single
# summary at the end, and exit non-zero if any element exceeded its
# tolerance. This matches the UX of ``git diff --exit-code`` (one
# summary, non-zero exit).
# ---------------------------------------------------------------------------


class DriftCollector:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.checks: int = 0

    def assert_within_ratio(
        self,
        committed: float,
        fresh: float,
        *,
        lo: float,
        hi: float,
        atol: float,
        label: str,
    ) -> None:
        self.checks += 1
        # Use signed positivity guard: lengthscales / outputscales are
        # positive by construction, but FP near-zero values can divide
        # awkwardly. The ``atol`` shoulder absorbs that.
        ratio = (fresh + atol) / (committed + atol)
        if ratio < lo or ratio > hi:
            self.failures.append(
                f"  {label}: committed={committed:.6g}, fresh={fresh:.6g}, "
                f"ratio={ratio:.4f} (allowed band [{lo}, {hi}])"
            )

    def assert_prediction_close(
        self,
        committed: float,
        fresh: float,
        *,
        floor: float,
        rtol: float,
        unit: str = "psi",
        label: str,
    ) -> None:
        self.checks += 1
        diff = abs(fresh - committed)
        delta = max(floor, rtol * abs(committed))
        if diff > delta:
            rel = diff / max(abs(committed), 1.0)
            self.failures.append(
                f"  {label}: committed={committed:.4f}, fresh={fresh:.4f}, "
                f"abs={diff:.2f} {unit} ({rel * 100:.2f}% rel; tol="
                f"max({floor:.4g}, {rtol * 100:.1f}%)={delta:.2f})"
            )

    def assert_variance_close(
        self,
        committed: float,
        fresh: float,
        *,
        floor: float = VARIANCE_PSI2_FLOOR,
        rtol: float = VARIANCE_RTOL,
        unit: str = "psi^2",
        label: str,
    ) -> None:
        self.checks += 1
        diff = abs(fresh - committed)
        delta = max(floor, rtol * abs(committed))
        if diff > delta:
            rel = diff / max(abs(committed), 1.0)
            self.failures.append(
                f"  {label}: committed={committed:.4f}, fresh={fresh:.4f}, "
                f"abs={diff:.2f} {unit} ({rel * 100:.2f}% rel; tol="
                f"max({floor:.4g}, "
                f"{rtol * 100:.1f}%)={delta:.2f})"
            )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict:
    with open(path) as fp:
        return json.load(fp)


def _zip_lists(
    a: Sequence[float], b: Sequence[float], label: str
) -> Iterable[Tuple[int, float, float, str]]:
    if len(a) != len(b):
        raise ValueError(
            f"{label}: list length mismatch (committed={len(a)}, "
            f"fresh={len(b)}). Schema-level change detected — "
            "regenerate both committed copies and the test."
        )
    for i, (x, y) in enumerate(zip(a, b)):
        yield i, x, y, f"{label}[{i}]"


# ---------------------------------------------------------------------------
# Per-artifact checks.
# ---------------------------------------------------------------------------


def check_strength_json(committed: dict, fresh: dict, drift: DriftCollector) -> None:
    """Compare hyperparameters in ``docs/model/strength.json``."""
    # Schema sanity (same shape across the two copies).
    for key in ("kernel_kind", "noise_kind", "y_scaling", "mean_kind"):
        if committed.get(key) != fresh.get(key):
            drift.failures.append(
                f"  strength.json[{key}]: committed={committed.get(key)!r}, "
                f"fresh={fresh.get(key)!r} (schema-level change — regenerate)"
            )
    # Lengthscales: 10x ratio band.
    for sub in ("matern_blind", "matern_specific"):
        c_ls = committed[sub]["lengthscales"]
        f_ls = fresh[sub]["lengthscales"]
        for _i, c, f, lbl in _zip_lists(
            c_ls, f_ls, f"strength.json[{sub}.lengthscales]"
        ):
            drift.assert_within_ratio(
                c,
                f,
                lo=LENGTHSCALE_RATIO_LO,
                hi=LENGTHSCALE_RATIO_HI,
                atol=LENGTHSCALE_ATOL,
                label=lbl,
            )
        # Outputscales: 10x ratio band.
        drift.assert_within_ratio(
            committed[sub]["outputscale"],
            fresh[sub]["outputscale"],
            lo=SCALAR_PARAM_RATIO_LO,
            hi=SCALAR_PARAM_RATIO_HI,
            atol=1e-9,
            label=f"strength.json[{sub}.outputscale]",
        )
    # RBF-time lengthscale + outputscale.
    drift.assert_within_ratio(
        committed["rbf_time"]["lengthscale"],
        fresh["rbf_time"]["lengthscale"],
        lo=SCALAR_PARAM_RATIO_LO,
        hi=SCALAR_PARAM_RATIO_HI,
        atol=1e-9,
        label="strength.json[rbf_time.lengthscale]",
    )
    drift.assert_within_ratio(
        committed["rbf_time"]["outputscale"],
        fresh["rbf_time"]["outputscale"],
        lo=SCALAR_PARAM_RATIO_LO,
        hi=SCALAR_PARAM_RATIO_HI,
        atol=1e-9,
        label="strength.json[rbf_time.outputscale]",
    )
    # Noise: 10x ratio band (same rationale as lengthscales).
    drift.assert_within_ratio(
        committed["noise"],
        fresh["noise"],
        lo=SCALAR_PARAM_RATIO_LO,
        hi=SCALAR_PARAM_RATIO_HI,
        atol=1e-9,
        label="strength.json[noise]",
    )


def check_test_vectors_json(
    committed: dict, fresh: dict, drift: DriftCollector
) -> None:
    """Compare baked predictions in ``docs/model/test_vectors.json``."""
    c_tv = committed.get("test_vectors", [])
    f_tv = fresh.get("test_vectors", [])
    if len(c_tv) != len(f_tv):
        drift.failures.append(
            f"  test_vectors.json: vector count differs "
            f"(committed={len(c_tv)}, fresh={len(f_tv)}). Schema-level "
            "change — regenerate."
        )
        return
    for i, (cv, fv) in enumerate(zip(c_tv, f_tv)):
        # Predictions: tight psi tolerance.
        drift.assert_prediction_close(
            cv["expected_mean"],
            fv["expected_mean"],
            floor=PREDICTION_PSI_FLOOR,
            rtol=PREDICTION_RTOL,
            label=f"test_vectors[{i}].expected_mean",
        )
        # Variances: looser psi^2 tolerance.
        drift.assert_variance_close(
            cv["expected_variance"],
            fv["expected_variance"],
            label=f"test_vectors[{i}].expected_variance",
        )
        # Per-day strength predictions, if present.
        for day, c_day in cv.get("strength", {}).items():
            f_day = fv.get("strength", {}).get(day)
            if f_day is None:
                drift.failures.append(
                    f"  test_vectors[{i}].strength[{day}]: missing in fresh"
                )
                continue
            drift.assert_prediction_close(
                c_day["mean"],
                f_day["mean"],
                floor=PREDICTION_PSI_FLOOR,
                rtol=PREDICTION_RTOL,
                label=f"test_vectors[{i}].strength[{day}].mean",
            )
            drift.assert_variance_close(
                c_day["variance"],
                f_day["variance"],
                label=f"test_vectors[{i}].strength[{day}].variance",
            )


def check_compositions_json(
    committed: dict, fresh: dict, drift: DriftCollector
) -> None:
    """Compare baked strength_predictions in ``docs/model/compositions.json``."""
    c_sp = committed.get("strength_predictions", {})
    f_sp = fresh.get("strength_predictions", {})
    days = sorted(set(c_sp.keys()) | set(f_sp.keys()))
    for day in days:
        if day not in c_sp or day not in f_sp:
            drift.failures.append(
                f"  compositions.json[strength_predictions][{day}]: "
                f"missing on one side"
            )
            continue
        for _i, c, f, lbl in _zip_lists(
            c_sp[day],
            f_sp[day],
            f"compositions.json[strength_predictions[{day}]]",
        ):
            drift.assert_prediction_close(
                c,
                f,
                floor=PREDICTION_PSI_FLOOR,
                rtol=PREDICTION_RTOL,
                label=lbl,
            )


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def check_slump_json(committed: dict, fresh: dict, drift: DriftCollector) -> None:
    """Compare hyperparameters in ``docs/model/slump.json``."""
    # Data-derived fields must match bit-for-bit: they do not depend on where
    # the non-convex MLL fit lands, so any change is a real semantic change
    # (and docs/slump.mjs hardcodes column indices consistent with them).
    for key in (
        "schema_version",
        "kernel_kind",
        "mean_kind",
        "d_in",
        "d_aug",
        "n_train",
        "supported_source_classes",
        "source_dim_raw",
        "raw_feature_names",
        "normalize_lower",
        "normalize_upper",
    ):
        if committed.get(key) != fresh.get(key):
            drift.checks += 1
            drift.failures.append(
                f"  slump.json[{key}]: committed={committed.get(key)!r}, "
                f"fresh={fresh.get(key)!r}"
            )

    c_ls = committed.get("lengthscales", [])
    f_ls = fresh.get("lengthscales", [])
    if len(c_ls) != len(f_ls):
        drift.checks += 1
        drift.failures.append(
            f"  slump.json[lengthscales]: length {len(c_ls)} vs {len(f_ls)}"
        )
    else:
        for i, (c, f) in enumerate(zip(c_ls, f_ls)):
            drift.assert_within_ratio(
                c,
                f,
                lo=LENGTHSCALE_RATIO_LO,
                hi=LENGTHSCALE_RATIO_HI,
                atol=1e-9,
                label=f"slump.json[lengthscales][{i}]",
            )

    for key in ("noise", "y_mean", "y_std"):
        drift.assert_within_ratio(
            committed.get(key, 0.0),
            fresh.get(key, 0.0),
            lo=SCALAR_PARAM_RATIO_LO,
            hi=SCALAR_PARAM_RATIO_HI,
            atol=1e-9,
            label=f"slump.json[{key}]",
        )


def check_slump_test_vectors_json(
    committed: dict, fresh: dict, drift: DriftCollector
) -> None:
    """Compare golden posteriors in ``docs/model/slump_test_vectors.json``.

    Slump is in INCHES, so the psi-scaled defaults would be vacuous here:
    posterior variances land around 2-11 in^2 against a 2000 psi^2 floor.
    The floors below are set from the data — slump spans 0.8-11.1 in with a
    fitted noise std near 1.5 in, so 0.1 in on the mean (and 0.25 in^2, about
    0.5 in on the sigma scale) catches a genuinely stale export while
    tolerating cross-architecture basin divergence.
    """
    c_tv = committed.get("test_vectors", [])
    f_tv = fresh.get("test_vectors", [])
    if len(c_tv) != len(f_tv):
        drift.checks += 1
        drift.failures.append(
            f"  slump_test_vectors.json: {len(c_tv)} vs {len(f_tv)} vectors"
        )
        return
    for i, (cv, fv) in enumerate(zip(c_tv, f_tv)):
        drift.assert_prediction_close(
            cv["expected_mean"],
            fv["expected_mean"],
            floor=0.1,
            rtol=0.05,
            unit="in",
            label=f"slump_test_vectors[{i}].expected_mean",
        )
        drift.assert_variance_close(
            cv["expected_variance"],
            fv["expected_variance"],
            floor=0.25,
            rtol=0.10,
            unit="in^2",
            label=f"slump_test_vectors[{i}].expected_variance",
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--committed-dir", required=True, type=Path)
    p.add_argument("--fresh-dir", required=True, type=Path)
    args = p.parse_args()

    drift = DriftCollector()

    for fname, checker in [
        ("strength.json", check_strength_json),
        ("test_vectors.json", check_test_vectors_json),
        ("compositions.json", check_compositions_json),
        ("slump.json", check_slump_json),
        ("slump_test_vectors.json", check_slump_test_vectors_json),
    ]:
        c_path = args.committed_dir / fname
        f_path = args.fresh_dir / fname
        if not c_path.exists() or not f_path.exists():
            # A file present in --fresh-dir but absent from --committed-dir
            # means the workflow forgot to snapshot a newly-added artifact,
            # which would silently give that artifact zero drift coverage.
            # That is a wiring bug, not a benign absence.
            if f_path.exists() and not c_path.exists():
                print(
                    f"::error::{fname} exists in --fresh-dir but was not "
                    "snapshotted into --committed-dir; add it to the "
                    "'Save committed docs/model/' workflow step."
                )
                return 1
            print(
                f"::warning::skipping {fname}: missing on at least one side "
                f"(committed={c_path.exists()}, fresh={f_path.exists()})"
            )
            continue
        committed = _load(c_path)
        fresh = _load(f_path)
        checker(committed, fresh, drift)

    print(f"check_artifacts_drift: ran {drift.checks} numerical comparisons.")
    if drift.failures:
        print()
        print(
            "::error::Committed docs/model/* artifacts disagree with the "
            "freshly-regenerated copy beyond the cross-architecture-portable "
            "tolerance band. This usually means the model code changed but "
            "the JSON wasn't regenerated. Run:"
        )
        print("::error::  bash experiments/regenerate_all_artifacts.sh")
        print("::error:: and commit the resulting docs/model/ changes.")
        print()
        print(f"{len(drift.failures)} drift(s) detected:")
        for line in drift.failures:
            print(line)
        return 1
    print("::notice::All artifacts agree with committed copy within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
