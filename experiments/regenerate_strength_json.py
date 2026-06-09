#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regenerate ``docs/model/strength.json`` AND ``docs/model/test_vectors.json``.

Production model: ``B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean``
(see ``STRENGTH_GP_BENCHMARK.md`` §6.12 and Appendix A "Anchors study").

This script:
1. Loads the public 647-row strength dataset.
2. Fits the V2 strength GP via ``boxcrete.fit_strength_gp``.
3. Extracts all parameters needed for in-browser inference into a flat
   JSON schema consumed by the updated ``docs/gp.mjs``.
4. Writes ``docs/model/strength.json`` and ``docs/model/test_vectors.json``
   from the same fitted model in a single process (the JS-Python
   equivalence test ``test/test_js_gp.mjs`` requires both files to come
   from the same fit; see ``docs/model/README.md``).

Run via ``bash experiments/regenerate_all_artifacts.sh`` to also refresh
``docs/model/compositions.json`` + run the JS test suite. Direct
``python experiments/regenerate_strength_json.py`` is fine if you only
need the two model artifacts.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch

# Path setup so we can import from experiments/.
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from boxcrete.utils import DEFAULT_X_COLUMNS, load_concrete_strength  # noqa: E402

# F5_alllog feature names — order matters; matches Python's
# F5_ALLLOG_FEATURES in boxcrete.features (single source of truth).
from boxcrete.features import (  # noqa: E402
    F5_ALLLOG_FEATURES,
    GATE_TAU as _PROD_GATE_TAU,
)

# pyrefly: ignore [missing-import]
# ``_SOURCE_DIM`` is the index of the "Material Source" column in the
# default X column ordering; previously imported from
# ``experiments/model_variant_study.py`` but that file is not part of
# the production release. Inlined here as a single-line computation
# off the canonical ``DEFAULT_X_COLUMNS`` list.
_SOURCE_DIM = DEFAULT_X_COLUMNS.index("Material Source")

# pyrefly: ignore [missing-import]
from boxcrete.features import IDX  # noqa: E402

# ----- Constants --------------------------------------------------------

CHAMPION_NAME = "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean"
# Kernel + noise gate constants. ``GATE_TAU`` is the canonical value
# imported from ``boxcrete.features``. ``NOISE_GATE_TAU`` is held locally
# so a future variant could detune the noise gate from the kernel gate;
# today both are equal.
GATE_TAU = _PROD_GATE_TAU
NOISE_GATE_TAU = _PROD_GATE_TAU
# Log-time offset baked into the V2 ``_get_v2_input_transform`` chain
# (``AffineInputTransform`` adds 1 to the time column before ``Log10``).
# Single source of truth for the regen script; emitted as
# ``log_time_offset`` in the JSON and consumed by ``_gated_aleatoric_psi2``
# below.
LOG_TIME_OFFSET = 1.0
TIME_DIM_IN_AUG = 9  # post-input-transform time dim (0-indexed)

CHAMPION_FEATURES = F5_ALLLOG_FEATURES

# Repo-relative paths so the script runs on any clone.
_REPO = Path(__file__).resolve().parent.parent
_MODEL_DIR = _REPO / "docs" / "model"

OUTPUT_PATH = _MODEL_DIR / "strength.json"
TEST_VECTORS_PATH = _MODEL_DIR / "test_vectors.json"
# State-dict path consumed by the Python-canonical
# ``boxcrete.load_pretrained_strength_gp`` loader. The .pt and the .json
# above are written from the SAME fitted model in this script; the
# parity between them is asserted by
# ``test/test_pretrained_loader_fidelity.py``. The .json is what
# ``docs/gp.mjs`` and ``docs/gp_v2_fast.mjs`` consume for in-browser
# inference; the .pt is what Python uses for fast skip-the-fit
# deserialisation via ``torch.load`` + ``model.load_state_dict``.
STATE_DICT_PATH = _MODEL_DIR / "strength_model.pt"


def main() -> None:
    print(f"[regenerate] V2 strength GP variant: {CHAMPION_NAME}")
    print("[regenerate] Loading dataset…")
    data = load_concrete_strength()
    X, Y, Yvar, bounds = data.strength_data
    n_real = X.shape[0]
    print(f"[regenerate]   n={n_real} rows, d={X.shape[-1]}")

    print("[regenerate] Fitting V2 strength GP via boxcrete.fit_strength_gp…")
    from boxcrete import fit_strength_gp

    model = fit_strength_gp(X=X, Y=Y, Yvar=Yvar, X_bounds=bounds, seed=0)
    n_real_out = X.shape[0]
    assert n_real_out == n_real
    model.eval()
    print(f"[regenerate]   fit complete; y_max = {float(model._study_y_std):.2f}")

    # --- Extract input-transform parameters ---
    # The chain is: append features -> log_offset (time only) -> log10
    # (time only) -> normalize. We store the Normalize bounds and the
    # log_time_offset; the engineered-feature recipe is implicit in the
    # JS code (matched to Python by name via ``engineered_feature_names``).
    input_tf = model.input_transform
    # Walk the chain once to locate the Normalize step (we need its
    # ``bounds`` and ``indices``). The other input-transform steps are
    # parameter-free (``Log10``) or have a fixed offset baked into the
    # JSON below as ``log_time_offset = 1.0``.
    normalize = None
    norm_indices = None
    for sub in input_tf.modules():
        if type(sub).__name__ == "Normalize" and sub is not input_tf:
            normalize = sub
            if hasattr(sub, "indices") and sub.indices is not None:
                norm_indices = sub.indices.tolist()
            break
    assert (
        normalize is not None
    ), "input transform is missing a Normalize step; cannot extract bounds"
    norm_bounds_raw = normalize.bounds.detach().clone()
    print(
        f"[regenerate]   normalize.bounds shape (raw): {tuple(norm_bounds_raw.shape)}"
    )
    if norm_indices is not None and len(norm_indices) != int(norm_bounds_raw.shape[-1]):
        # Edge case: bounds full-d but indices partial — should not happen
        norm_indices = list(range(int(norm_bounds_raw.shape[-1])))

    # Determine d_aug from the kernel's specific Matern (which spans all dims).
    # Find the kernel module and inspect its specific component.
    # We'll do this fully below; for now use the maximum size.
    if norm_indices is not None:
        # Reconstruct full d_aug bounds with identity at non-indexed dims.
        d_aug_inferred = (
            max(norm_indices) + 1 if norm_indices else int(norm_bounds_raw.shape[-1])
        )
        # The time dim is at TIME_DIM_IN_AUG (=9) and is missing from
        # indices when skip_time_in_normalize=True.
        if TIME_DIM_IN_AUG not in norm_indices:
            d_aug_inferred = max(d_aug_inferred, TIME_DIM_IN_AUG + 1)
        # Final d_aug: 10 raw + 7 features = 17. Use this as the source of truth.
        d_aug = 10 + len(F5_ALLLOG_FEATURES)
        full_lower = [0.0] * d_aug
        full_upper = [1.0] * d_aug
        # Map: idx_in_norm_indices -> norm_bounds column k
        for k, dim in enumerate(norm_indices):
            full_lower[dim] = float(norm_bounds_raw[0, k])
            full_upper[dim] = float(norm_bounds_raw[1, k])
        norm_lower_list = full_lower
        norm_upper_list = full_upper
        print(
            f"[regenerate]   reconstructed full {d_aug}-dim bounds "
            f"(time dim {TIME_DIM_IN_AUG} = identity [0,1] = no-op)"
        )
    else:
        d_aug = int(norm_bounds_raw.shape[-1])
        norm_lower_list = norm_bounds_raw[0].tolist()
        norm_upper_list = norm_bounds_raw[1].tolist()

    # --- Extract kernel hyperparameters ---
    # The champion's covar_module is _TimeGatedKernel wrapping a sum:
    # blind_matern + source-specific_matern + rbf(t).
    gated = model.covar_module  # _TimeGatedKernel
    base = gated.base_kernel  # AdditiveKernel: blind + specific + rbf
    # Iterate base.kernels — should be 3 components.
    components = list(base.kernels)
    assert len(components) == 3, f"Expected 3 base kernels, got {len(components)}"
    blind_kernel, specific_kernel, time_kernel = components
    # Each has a ScaleKernel wrapper around an ARD Matern (or RBF for time).
    blind_lengthscales = (
        blind_kernel.base_kernel.lengthscale.detach().squeeze().tolist()
    )
    blind_outputscale = float(blind_kernel.outputscale.detach())
    blind_active_dims = blind_kernel.base_kernel.active_dims.tolist()
    specific_lengthscales = (
        specific_kernel.base_kernel.lengthscale.detach().squeeze().tolist()
    )
    specific_outputscale = float(specific_kernel.outputscale.detach())
    specific_active_dims = specific_kernel.base_kernel.active_dims.tolist()
    rbf_lengthscale = float(time_kernel.base_kernel.lengthscale.detach().squeeze())
    rbf_outputscale = float(time_kernel.outputscale.detach())
    rbf_active_dims = time_kernel.base_kernel.active_dims.tolist()
    print(
        f"[regenerate]   blind: outputscale={blind_outputscale:.4f} "
        f"active_dims={blind_active_dims} "
        f"len(lengthscales)={len(blind_lengthscales)}"
    )
    print(
        f"[regenerate]   specific: outputscale={specific_outputscale:.4f} "
        f"active_dims={specific_active_dims} "
        f"len(lengthscales)={len(specific_lengthscales)}"
    )
    print(
        f"[regenerate]   rbf(t): lengthscale={rbf_lengthscale:.4f} "
        f"outputscale={rbf_outputscale:.4f} "
        f"active_dims={rbf_active_dims}"
    )
    # --- Noise ---
    # `GaussianLikelihood` exposes `noise` directly; custom likelihoods
    # (like _GatedGaussianLikelihood) only have `noise_covar.noise`.
    if hasattr(model.likelihood, "noise"):
        noise = float(model.likelihood.noise.detach().squeeze())
    else:
        noise = float(model.likelihood.noise_covar.noise.detach().flatten()[0])
    print(f"[regenerate]   noise (in scaled space): {noise:.6f}")

    # --- Y scaling: y_max for un-scaling ---
    y_max = float(model._study_y_std)
    # In the max-scale path, y_mean is 0.
    y_mean_offset = (
        float(model._study_y_mean) if hasattr(model, "_study_y_mean") else 0.0
    )

    # --- Extract training targets in scaled (Y / y_max) space ---
    # We deliberately do NOT pre-compute or serialize the Cholesky factor
    # `L` or the kernel-solve vector `alpha`. Doing so couples the JS path's
    # numerics to whichever Cholesky implementation Python happens to use
    # (e.g., manual `chol(K + 1e-6·I)` here vs. BoTorch's adaptive
    # `psd_safe_cholesky` inside `model.posterior()`). That coupling caused
    # a ~1e-3 relative variance drift between the JS in-browser inference
    # and `test_vectors.json` (which was generated from `posterior()`).
    #
    # Instead we ship only the kernel ingredients (X_train, kernel
    # hyperparameters, noise, gate_tau, Y_train) and let the JS side
    # rebuild K and run its own Cholesky in `initStrengthModel`. JS and
    # Python now reach the same K from the same kernel evaluations, so
    # the two paths agree to FP-summation order (~1e-13), well below the
    # `test/test_js_gp.mjs` tolerance.
    print("[regenerate] Extracting training targets (Y_train, scaled)…")
    with torch.no_grad():
        train_X_aug = model.train_inputs[0]  # [n, d_aug] post-input-transform
        train_Y_scaled = model.train_targets  # [n] in scaled space (Y / y_max)
    print(
        f"[regenerate]   Y_train range (scaled): "
        f"[{train_Y_scaled.min():.4f}, {train_Y_scaled.max():.4f}]"
    )

    # --- Assemble output JSON ---
    out = {
        "schema_version": 2,
        "model_name": CHAMPION_NAME,
        "comment": (
            "V2 strength GP. Architecture: gated multi-Matern + 7 engineered "
            "features + Y/y_max scaling + ZeroMean + block-LOO HP refinement. "
            "Block-LOO RMSE 665 psi at full data, phantom-anchor RMSE = 0 by "
            "construction. See experiments/STRENGTH_GP_BENCHMARK.md for the "
            "full study."
        ),
        # Dataset metadata
        "n_train": n_real,
        "n_real": n_real,
        "d_in": int(X.shape[-1]),  # raw input dim (10)
        "d_aug": d_aug,  # post-feature-append (17)
        "raw_feature_names": [
            "Cement",
            "Fly Ash",
            "Slag",
            "Water",
            "HRWR",
            "Fine Aggregate",
            "Coarse Aggregates",
            "Material Source",
            "Temp",
            "Time",
        ],
        "engineered_feature_names": CHAMPION_FEATURES,
        "time_dim_raw": int(IDX["time"]),  # 9
        "source_dim_raw": int(_SOURCE_DIM),  # 7
        "time_dim_aug": TIME_DIM_IN_AUG,  # also 9 (Append doesn't move time)
        # Input transform: Normalize bounds (post log+features), log time offset
        "log_time_offset": LOG_TIME_OFFSET,
        "normalize_lower": norm_lower_list,
        "normalize_upper": norm_upper_list,
        # Multi-Matern (B'') kernel
        "kernel_kind": "gated_multi_matern_rbf",
        "gate_tau": GATE_TAU,
        "matern_blind": {
            "active_dims": blind_active_dims,
            "lengthscales": blind_lengthscales,
            "outputscale": blind_outputscale,
        },
        "matern_specific": {
            "active_dims": specific_active_dims,
            "lengthscales": specific_lengthscales,
            "outputscale": specific_outputscale,
        },
        "rbf_time": {
            "active_dims": rbf_active_dims,
            "lengthscale": rbf_lengthscale,
            "outputscale": rbf_outputscale,
        },
        "noise": noise,
        # Noise gating: when "gated", the JS computes per-row aleatoric
        # variance as h(t)² * noise (with t = post-input-transform time
        # at index time_dim_aug). This makes the FULL predictive distribution
        # collapse to (0, 0) at t=0 — physically faithful for concrete.
        "noise_kind": "gated",
        "noise_gate_tau": NOISE_GATE_TAU,
        # When `noise_kind == "gated"`, gp_v2_fast.mjs returns variances
        # that already include the aleatoric component, and ui.mjs's
        # computeStds treats the returned variances as total (does NOT
        # add noise again). The flag below tells ui.mjs how to interpret.
        "variance_includes_aleatoric": True,
        # Y scaling: max-scale (Y_scaled = Y / y_max, no mean subtraction)
        "y_scaling": "max_scale",
        "y_max": y_max,
        "y_mean": y_mean_offset,
        # Mean function: ZeroMean
        "mean_kind": "zero",
        # Training data and targets — JS rebuilds K + alpha at init time.
        "X_train": train_X_aug.detach().cpu().tolist(),  # [n, d_aug]
        "Y_train": train_Y_scaled.detach().cpu().tolist(),  # [n] in scaled space
    }

    print(
        "[regenerate] Writing test vectors (multi-format) for JS-side "
        "regression checks\u2026"
    )
    # 32 random training rows, picked uniformly across the dataset
    rng = torch.Generator().manual_seed(0)
    n_tests = 32
    indices = torch.randperm(n_real, generator=rng)[:n_tests]
    test_vectors = []
    # Standard test days for the legacy `test_js_gp.mjs` format.
    standard_days = [1, 7, 28]

    def _gated_aleatoric_psi2(t_raw: float) -> float:
        """Total-variance contract (variance_includes_aleatoric=True):
        the JS path returns ``(latent_var + h(t)²·σ²) · y_max²``. Mirror
        that here so test_vectors.json's ``expected_variance`` matches
        the JS-side total. ``h`` evaluates at the post-input-transform
        time (raw t→ log10(t+1) then identity-normalize for the time dim
        per ``skip_time_in_normalize=True``)."""
        t_post = math.log10(t_raw + LOG_TIME_OFFSET)
        h = 1.0 - math.exp(-max(t_post, 0.0) / NOISE_GATE_TAU)
        return h * h * noise * (y_max**2)

    for idx in indices.tolist():
        x_raw_full = X[idx].clone()
        comp = x_raw_full[: IDX["time"]].tolist()
        time_val = float(x_raw_full[IDX["time"]])
        # Per-vector flat prediction (new schema)
        with torch.no_grad():
            x_pred = X[idx : idx + 1].clone()
            posterior = model.posterior(x_pred)
            mean_pred = float(posterior.mean.item()) * y_max + y_mean_offset
            var_pred = float(posterior.variance.item()) * (
                y_max**2
            ) + _gated_aleatoric_psi2(time_val)
        # Per-day predictions (legacy format expected by test_js_gp.mjs)
        strength_per_day = {}
        for d in standard_days:
            with torch.no_grad():
                x_d = X[idx : idx + 1].clone()
                x_d[..., IDX["time"]] = float(d)
                pos_d = model.posterior(x_d)
                m_d = float(pos_d.mean.item()) * y_max + y_mean_offset
                v_d = float(pos_d.variance.item()) * (
                    y_max**2
                ) + _gated_aleatoric_psi2(float(d))
            strength_per_day[str(d)] = {"mean": m_d, "variance": v_d}
        test_vectors.append(
            {
                "composition": comp,
                "input": comp,  # legacy alias
                "time": time_val,
                "expected_mean": mean_pred,
                "expected_variance": var_pred,
                "strength": strength_per_day,  # legacy per-day format
            }
        )
    # Add 5 t=0 constraint-check vectors.
    for i in range(min(5, n_real)):
        x_raw_full = X[i].clone()
        comp = x_raw_full[: IDX["time"]].tolist()
        with torch.no_grad():
            x_pred = X[i : i + 1].clone()
            x_pred[..., IDX["time"]] = 0.0
            posterior = model.posterior(x_pred)
            mean_pred = float(posterior.mean.item()) * y_max + y_mean_offset
            # At t=0, h(0) = 0, so aleatoric is zero by construction —
            # but still call the helper for symmetry / clarity.
            var_pred = float(posterior.variance.item()) * (
                y_max**2
            ) + _gated_aleatoric_psi2(0.0)
        # Per-day predictions (legacy format expected by test_js_gp.mjs).
        # Even for the t=0 constraint-check vectors, include the per-day
        # dict so the legacy test can iterate over them uniformly.
        strength_per_day = {}
        for d in standard_days:
            with torch.no_grad():
                x_d = X[i : i + 1].clone()
                x_d[..., IDX["time"]] = float(d)
                pos_d = model.posterior(x_d)
                m_d = float(pos_d.mean.item()) * y_max + y_mean_offset
                v_d = float(pos_d.variance.item()) * (
                    y_max**2
                ) + _gated_aleatoric_psi2(float(d))
            strength_per_day[str(d)] = {"mean": m_d, "variance": v_d}
        test_vectors.append(
            {
                "composition": comp,
                "input": comp,
                "time": 0.0,
                "expected_mean": mean_pred,
                "expected_variance": var_pred,
                "strength": strength_per_day,
                "note": "t=0 physics constraint check",
            }
        )

    # --- Write out ---
    print(f"[regenerate] Writing {OUTPUT_PATH}…")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"[regenerate]   wrote {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")

    print(f"[regenerate] Writing {TEST_VECTORS_PATH}…")
    with open(TEST_VECTORS_PATH, "w") as f:
        json.dump(
            {
                "strength_days": [1, 7, 28],  # legacy field used by test_js_gp.mjs
                "test_vectors": test_vectors,
            },
            f,
            indent=2,
        )
    print(f"[regenerate]   wrote {len(test_vectors)} test vectors")

    print(f"[regenerate] Writing {STATE_DICT_PATH}…")
    torch.save(model.state_dict(), STATE_DICT_PATH)
    print(
        f"[regenerate]   wrote {os.path.getsize(STATE_DICT_PATH) / 1024:.1f} KB "
        "state_dict (Python-canonical loader)"
    )

    print("[regenerate] DONE.")


if __name__ == "__main__":
    main()
