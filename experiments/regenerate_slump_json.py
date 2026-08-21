# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Export the slump GP to docs/model/slump.json for the web explorer.

Writes two files from a SINGLE fit (the MLL objective is non-convex, so two
runs land on different optima and the golden vectors would not match the
shipped model):

  * docs/model/slump.json              — kernel ingredients for docs/slump.mjs
  * docs/model/slump_test_vectors.json — Python-reference posteriors

Like the strength exporter, this deliberately does NOT ship the Cholesky
factor or alpha: JS rebuilds K and factors it, so both sides reach the same
posterior from the same kernel evaluations instead of inheriting Python's
factorization.

Usage: python experiments/regenerate_slump_json.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean

from boxcrete.slump_model import fit_slump_gp
from boxcrete.utils import load_concrete_strength, SLUMP_Y_COLUMNS

SEED = 0
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "model" / "slump.json"
OUT_VECTORS = REPO / "docs" / "model" / "slump_test_vectors.json"

# Material Source classes with slump observations. Mortar (class 0) has none:
# workability there is measured by flow table, not a slump cone. The explorer
# renders "n/a" for classes outside this list rather than extrapolating.
SUPPORTED_SOURCE_CLASSES = [1, 2]

RAW_FEATURE_NAMES = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
    "Material Source",
    "Temp (C)",
]

SOURCE_DIM_RAW = 7
HRWR_DIM_RAW = 4


def main() -> None:
    torch.manual_seed(SEED)

    print("[regenerate] Loading slump data …")
    dataset = load_concrete_strength(Y_columns=SLUMP_Y_COLUMNS)
    slump = dataset.slump_data
    if slump is None:
        raise RuntimeError("slump_data is None; check the 'Slump (in)' column.")
    X, Y, Yvar, _bounds = slump
    print(f"[regenerate]   X={tuple(X.shape)} Y={tuple(Y.shape)}")

    # Compare the full column LIST, not just its length: docs/slump.mjs
    # hardcodes I_CEMENT=0 / I_FLYASH=1 / I_SLAG=2 / I_HRWR=4 and this file
    # hardcodes source_dim_raw=7, so a reordering of DEFAULT_X_COLUMNS would
    # silently corrupt the derived feature and the mortar gate while still
    # producing a parity-passing artifact (both sides regenerate the same
    # wrong thing). X_columns[:-1] drops "Time", which slump_data strips.
    if list(dataset.X_columns[:-1]) != RAW_FEATURE_NAMES:
        raise RuntimeError(
            "RAW_FEATURE_NAMES is out of sync with DEFAULT_X_COLUMNS: "
            f"expected {RAW_FEATURE_NAMES}, got {list(dataset.X_columns[:-1])}. "
            "docs/slump.mjs hardcodes column indices and slump.json hardcodes "
            "source_dim_raw=7; both must be revisited with this list."
        )
    if RAW_FEATURE_NAMES.index("Material Source") != SOURCE_DIM_RAW:
        raise RuntimeError("Material Source is no longer at index 7.")
    if RAW_FEATURE_NAMES.index("HRWR (kg/m3)") != HRWR_DIM_RAW:
        raise RuntimeError("HRWR is no longer at index 4.")

    observed = sorted({int(v) for v in X[:, SOURCE_DIM_RAW].tolist()})
    if observed != SUPPORTED_SOURCE_CLASSES:
        raise RuntimeError(
            f"Slump source classes changed: expected {SUPPORTED_SOURCE_CLASSES}, "
            f"found {observed}. The explorer's mortar gate in docs/ui.mjs and the "
            "supported_source_classes field must be revisited together."
        )

    print("[regenerate] Fitting slump GP …")
    model = fit_slump_gp(X, Y, Yvar)
    # eval() triggers _set_transformed_inputs(), which is what makes
    # train_inputs[0] the POST-input-transform tensor we export below.
    model.eval()

    # --- Refuse to export an architecture the JS port cannot reproduce. ---
    # docs/slump.mjs implements exactly: ConstantMean + ARD RBF (outputscale 1)
    # + Standardize + Normalize + AppendDerivedFeatures. If BoTorch's
    # SingleTaskGP defaults move, this fails loudly here rather than shipping
    # a silently-wrong artifact.
    if not isinstance(model.covar_module, RBFKernel):
        raise RuntimeError(
            "docs/slump.mjs implements ARD RBF; model uses "
            f"{type(model.covar_module).__name__}. Update both together."
        )
    if hasattr(model.covar_module, "outputscale"):
        raise RuntimeError(
            "Kernel gained an outputscale (ScaleKernel). Export it and teach "
            "docs/slump.mjs to read params.outputscale from the artifact."
        )
    if not isinstance(model.mean_module, ConstantMean):
        raise RuntimeError(
            "docs/slump.mjs implements ConstantMean; model uses "
            f"{type(model.mean_module).__name__}."
        )
    if not isinstance(model.outcome_transform, Standardize):
        raise RuntimeError(
            "docs/slump.mjs implements Standardize; model uses "
            f"{type(model.outcome_transform).__name__}."
        )

    with torch.no_grad():
        lengthscales = model.covar_module.lengthscale.detach().flatten().tolist()
        noise = float(model.likelihood.noise.detach().flatten()[0])
        mean_constant = float(model.mean_module.constant.detach().flatten()[0])
        y_mean = float(model.outcome_transform.means.detach().flatten()[0])
        y_std = float(model.outcome_transform.stdvs.detach().flatten()[0])
        norm = model.input_transform["normalize"]
        lower = norm.bounds[0].detach().tolist()
        upper = norm.bounds[1].detach().tolist()
        train_X_aug = model.train_inputs[0].detach()  # [n, 10], post-transform
        train_Y_std = model.train_targets.detach()  # [n], standardized

    d_aug = train_X_aug.shape[-1]
    print(f"[regenerate]   d_in={X.shape[-1]} d_aug={d_aug} n={train_X_aug.shape[0]}")
    print(f"[regenerate]   noise={noise:.6f} mean_constant={mean_constant:.6f}")
    print(f"[regenerate]   y_mean={y_mean:.4f} in  y_std={y_std:.4f} in")

    out = {
        "schema_version": 1,
        "model_name": "slump_gp_v1",
        "comment": (
            "Slump GP for the web explorer. BoTorch SingleTaskGP defaults: "
            "ConstantMean + ARD RBF (no ScaleKernel, so outputscale is 1) + "
            "Standardize(1) outcome transform. Inputs are the 9 composition "
            "dims (no Time) with the HRWR/binder ratio appended by "
            "boxcrete.features.AppendDerivedFeatures, then min-max normalized. "
            "Trained only on Material Source 1 and 2 — mortar (class 0) has no "
            "slump measurements. Generated by "
            "experiments/regenerate_slump_json.py; do not edit by hand."
        ),
        "d_in": int(X.shape[-1]),
        "d_aug": int(d_aug),
        "n_train": int(train_X_aug.shape[0]),
        "raw_feature_names": RAW_FEATURE_NAMES,
        "engineered_feature_names": ["hrwr_binder_clamped"],
        "source_dim_raw": SOURCE_DIM_RAW,
        "supported_source_classes": SUPPORTED_SOURCE_CLASSES,
        "normalize_lower": lower,
        "normalize_upper": upper,
        "kernel_kind": "rbf_ard",
        "lengthscales": lengthscales,
        "outputscale": 1.0,
        "mean_kind": "constant",
        "mean_constant": mean_constant,
        "noise": noise,
        "variance_includes_aleatoric": True,
        "y_scaling": "standardize",
        "y_mean": y_mean,
        "y_std": y_std,
        "y_units": "in",
        "X_train": train_X_aug.tolist(),
        "Y_train": train_Y_std.tolist(),
    }
    OUT.write_text(json.dumps(out) + "\n")
    print(f"[regenerate] Wrote {OUT.relative_to(REPO)}")

    _write_test_vectors(model, X)


def _write_test_vectors(model, X: torch.Tensor) -> None:
    """Golden posteriors from the SAME fit, for test/test_js_slump.mjs."""
    torch.manual_seed(SEED + 1)
    n_random = 24

    lo, hi = X.min(dim=0).values, X.max(dim=0).values
    rand = lo + (hi - lo) * torch.rand(n_random, X.shape[-1], dtype=X.dtype)
    # Material Source is categorical: snap to an observed class.
    rand[:, SOURCE_DIM_RAW] = torch.tensor(
        [SUPPORTED_SOURCE_CLASSES[i % 2] for i in range(n_random)], dtype=X.dtype
    )
    # Include every training point too — these must reproduce near-exactly.
    probes = torch.cat([X, rand], dim=0)

    with torch.no_grad():
        # observation_noise=True matches variance_includes_aleatoric in the
        # artifact and the +noise term in docs/slump.mjs.
        post = model.posterior(probes, observation_noise=True)
        means = post.mean.flatten().tolist()
        variances = post.variance.flatten().tolist()

    vectors = [
        {
            "input": probes[i].tolist(),
            "expected_mean": means[i],
            "expected_variance": variances[i],
        }
        for i in range(probes.shape[0])
    ]
    OUT_VECTORS.write_text(
        json.dumps(
            {
                "comment": (
                    "Golden slump posteriors (inches, observation_noise=True) from "
                    "the same fit that produced slump.json. Regenerate both together "
                    "via experiments/regenerate_slump_json.py."
                ),
                "model_name": "slump_gp_v1",
                "test_vectors": vectors,
            }
        )
        + "\n"
    )
    print(
        f"[regenerate] Wrote {OUT_VECTORS.relative_to(REPO)} "
        f"({len(vectors)} vectors)"
    )


if __name__ == "__main__":
    main()
