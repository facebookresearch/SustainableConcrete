#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Per-class heteroscedastic noise ablation.

Replaces ``GatedGaussianLikelihood`` (single global sigma^2) with
``PerClassGatedGaussianLikelihood`` (3 separate sigma_c^2 — one per
material class) and re-evaluates the top source kernels.

Motivated by the noise audit finding (experiments/NOISE_AUDIT.md):
the homoscedastic GP fits sigma_gp ~ 370 psi, ~5x larger than Set-3's
actual measurement noise (~76 psi). Letting MLE find separate
sigma_c for each class could:

  * Improve Class-2 LOCO PIT-KS (currently 0.30 across kernels)
  * Improve Class-2 LOCO RMSE (currently 849-1113 psi)
  * Improve calibration on Class-1 (PIT-KS 0.30-0.40)

Grid: 1 kernel (joint_hamming_matern) x homoscedastic vs per-class
likelihood x 4 contexts (in-dist + 3 LOCO) x 3 seeds = 24 cells.
Wall time ~ 36 min at 90s/cell.

The script monkey-patches ``boxcrete.strength_model.GatedGaussianLikelihood``
during the fit to use the per-class variant.

Output:
    experiments/PER_CLASS_NOISE_ABLATION.md
    experiments/per_class_noise_ablation.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from boxcrete import fit_strength_gp  # noqa: E402
from boxcrete.features import IDX  # noqa: E402
from experiments.three_class_ablation import (  # noqa: E402
    _eval_metrics,
    _load_data,
    _make_class_holdout,
)
from experiments.model_variant_study import held_out_metrics  # noqa: E402

WRITEUP = REPO_ROOT / "experiments" / "PER_CLASS_NOISE_ABLATION.md"
RESULTS_CSV = REPO_ROOT / "experiments" / "per_class_noise_ablation.csv"

SOURCE_KERNEL = "joint_hamming_matern"
SEEDS = [0, 1, 2]


def _fit_homoscedastic(X, Y, Yvar, bounds, seed):
    """Standard fit with GatedGaussianLikelihood (homoscedastic)."""
    torch.manual_seed(seed)
    from boxcrete import kernels as _kmod
    from boxcrete import priors as _pmod
    from boxcrete import strength_model as _smod

    original_builder = _kmod.make_gated_strength_kernel_builder
    original_factory = _pmod.within_group_prior

    def _patched_builder(gate_tau=0.05, **kwargs):
        return original_builder(
            gate_tau=gate_tau,
            source_kernel=SOURCE_KERNEL,
            time_tying_sigma=None,
        )

    def _patched_factory(*args, **kwargs):
        kwargs.setdefault("include_lognormal_baseline", False)
        return original_factory(*args, **kwargs)

    _kmod.make_gated_strength_kernel_builder = _patched_builder
    _smod.make_gated_strength_kernel_builder = _patched_builder
    _pmod.within_group_prior = _patched_factory
    _kmod.within_group_prior = _patched_factory
    try:
        return fit_strength_gp(
            X=X, Y=Y, Yvar=Yvar, X_bounds=bounds, seed=seed
        )
    finally:
        _kmod.make_gated_strength_kernel_builder = original_builder
        _smod.make_gated_strength_kernel_builder = original_builder
        _pmod.within_group_prior = original_factory
        _kmod.within_group_prior = original_factory


def _fit_per_class(X, Y, Yvar, bounds, seed):
    """Fit with PerClassGatedGaussianLikelihood (per-class sigma_c)."""
    torch.manual_seed(seed)
    from boxcrete import kernels as _kmod
    from boxcrete import priors as _pmod
    from boxcrete import strength_model as _smod
    from boxcrete import likelihoods as _likmod

    original_builder = _kmod.make_gated_strength_kernel_builder
    original_factory = _pmod.within_group_prior
    original_likelihood = _smod.GatedGaussianLikelihood

    def _patched_builder(gate_tau=0.05, **kwargs):
        return original_builder(
            gate_tau=gate_tau,
            source_kernel=SOURCE_KERNEL,
            time_tying_sigma=None,
        )

    def _patched_factory(*args, **kwargs):
        kwargs.setdefault("include_lognormal_baseline", False)
        return original_factory(*args, **kwargs)

    # Wrap the per-class likelihood with an X-snapshot mechanism: at
    # construction time, the V2 fit factory calls
    # ``likelihood.set_train_times(X[..., IDX["time"]])``, but our
    # per-class likelihood ALSO needs the source-column. We patch
    # ``set_train_times`` to ALSO snapshot the full X (via a closure
    # over the X passed to fit_strength_gp).
    captured_X = {"X": None}

    class _PatchedPerClassLikelihood(_likmod.PerClassGatedGaussianLikelihood):
        def __init__(self, *args, **kwargs):
            # Match the V2 factory's call signature
            # (it passes time_idx, gate_tau, noise_constraint).
            super().__init__(
                num_classes=3,
                source_idx=IDX["source"],
                time_idx=kwargs.get("time_idx", IDX["time"]),
                gate_tau=kwargs.get("gate_tau", 0.05),
                noise_constraint=kwargs.get("noise_constraint"),
            )

        def set_train_times(self, time_values):
            # The V2 factory passes raw days here. We need full X for
            # per-class lookup, which is stored in captured_X.
            super().set_train_times(time_values)
            if captured_X["X"] is not None:
                self.set_train_inputs(captured_X["X"])

    _kmod.make_gated_strength_kernel_builder = _patched_builder
    _smod.make_gated_strength_kernel_builder = _patched_builder
    _pmod.within_group_prior = _patched_factory
    _kmod.within_group_prior = _patched_factory
    _smod.GatedGaussianLikelihood = _PatchedPerClassLikelihood
    captured_X["X"] = X
    try:
        model = fit_strength_gp(
            X=X, Y=Y, Yvar=Yvar, X_bounds=bounds, seed=seed
        )
        # After fit, also set train inputs on the fitted likelihood
        # so eval uses the per-row class lookup.
        train_inputs = model.train_inputs[0]
        model.likelihood.set_train_inputs(train_inputs)
        return model
    finally:
        _kmod.make_gated_strength_kernel_builder = original_builder
        _smod.make_gated_strength_kernel_builder = original_builder
        _pmod.within_group_prior = original_factory
        _kmod.within_group_prior = original_factory
        _smod.GatedGaussianLikelihood = original_likelihood


def main() -> int:
    rows: list[dict] = []

    print("\n===== In-distribution v5 =====", flush=True)
    ds, X, Y, Yvar, bounds, n_real = _load_data("v5")
    for noise_mode, fit_fn in (
        ("homoscedastic", _fit_homoscedastic),
        ("per_class", _fit_per_class),
    ):
        for seed in SEEDS:
            t0 = time.time()
            row = {
                "phase": "in_distribution",
                "data": "v5",
                "source_kernel": SOURCE_KERNEL,
                "noise_mode": noise_mode,
                "seed": seed,
                "n_train": n_real,
            }
            try:
                model = fit_fn(X, Y, Yvar, bounds, seed)
                m = _eval_metrics(model, n_real)
                row.update(m)
                # Inspect fitted per-class noise (if applicable).
                if noise_mode == "per_class":
                    pcn = model.likelihood.per_class_noise.detach()
                    Y_max = float(Y.max().item())
                    for c in range(3):
                        sigma_c_psi = float(pcn[c].sqrt().item() * Y_max)
                        row[f"sigma_c{c}_psi"] = sigma_c_psi
            except Exception as exc:
                row["fit_error"] = str(exc)[:200]
            row["wall_sec"] = time.time() - t0
            rows.append(row)
            loo = row.get("loo_rmse", float("nan"))
            bloo = row.get("bloo_rmse", float("nan"))
            print(
                f"  {SOURCE_KERNEL} {noise_mode} seed={seed} "
                f"loo={loo:.0f} bloo={bloo:.0f} ({row['wall_sec']:.0f}s)",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    for hc in [0, 1, 2]:
        print(f"\n===== LOCO class {hc} =====", flush=True)
        X_tr, Y_tr, Yvar_tr, X_te, Y_te_psi, b = _make_class_holdout(hc)
        n_test = int(X_te.shape[0])
        n_train = int(X_tr.shape[0])
        for noise_mode, fit_fn in (
            ("homoscedastic", _fit_homoscedastic),
            ("per_class", _fit_per_class),
        ):
            for seed in SEEDS:
                t0 = time.time()
                row = {
                    "phase": "loco",
                    "data": "v5",
                    "holdout_class": hc,
                    "source_kernel": SOURCE_KERNEL,
                    "noise_mode": noise_mode,
                    "seed": seed,
                    "n_train": n_train,
                    "n_test": n_test,
                }
                try:
                    model = fit_fn(X_tr, Y_tr, Yvar_tr, b, seed)
                    ho = held_out_metrics(model, X_te, Y_te_psi)
                    for k, v in ho.items():
                        row[f"holdout_{k}"] = v
                except Exception as exc:
                    row["fit_error"] = str(exc)[:200]
                row["wall_sec"] = time.time() - t0
                rows.append(row)
                rmse = row.get("holdout_rmse", float("nan"))
                pit = row.get("holdout_pit_ks", float("nan"))
                print(
                    f"  {SOURCE_KERNEL} {noise_mode} class={hc} seed={seed} "
                    f"rmse={rmse:.0f} pit_ks={pit:.3f} ({row['wall_sec']:.0f}s)",
                    flush=True,
                )
                pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[csv] wrote {RESULTS_CSV.relative_to(REPO_ROOT)}")

    # Writeup.
    out: list[str] = []
    out.append("# Per-class heteroscedastic noise ablation")
    out.append("")
    out.append(
        f"Source kernel: `{SOURCE_KERNEL}` (current Occam-optimal production "
        f"winner). Compares homoscedastic Gaussian likelihood (single global "
        f"sigma_gp) against per-class heteroscedastic likelihood "
        f"(3 separate sigma_c, one per material class)."
    )
    out.append("")

    in_dist = df[df["phase"] == "in_distribution"]
    out.append("## In-distribution v5 (mean across 3 seeds)")
    out.append("")
    out.append(
        "| noise mode | LOO RMSE | bLOO RMSE | bLOO PIT-KS | bLOO cov95 |"
    )
    out.append("|---|---|---|---|---|")
    for mode in ["homoscedastic", "per_class"]:
        sub = in_dist[in_dist["noise_mode"] == mode]
        if len(sub) == 0:
            continue
        loo = sub["loo_rmse"].mean()
        bloo = sub["bloo_rmse"].mean()
        pit = sub["bloo_pit_ks"].mean()
        cov = sub["bloo_coverage_95"].mean()
        out.append(
            f"| {mode} | {loo:.0f} | {bloo:.0f} | {pit:.3f} | {cov:.3f} |"
        )
    out.append("")

    # Per-class learned sigma_c.
    pcn = in_dist[in_dist["noise_mode"] == "per_class"]
    if len(pcn) > 0 and "sigma_c0_psi" in pcn.columns:
        out.append("### Learned per-class noise (best seed)")
        out.append("")
        best = pcn.iloc[0]
        out.append("| class | sigma_c (psi, learned) | sigma_meas (psi, data) |")
        out.append("|---|---|---|")
        meas = {0: 185, 1: 146, 2: 76}
        for c in range(3):
            sigma_learned = best.get(f"sigma_c{c}_psi", float("nan"))
            out.append(
                f"| {c} | {sigma_learned:.0f} | {meas[c]} |"
            )
        out.append("")

    loco_df = df[df["phase"] == "loco"]
    for hc in [0, 1, 2]:
        sub = loco_df[loco_df["holdout_class"] == hc]
        if len(sub) == 0:
            continue
        out.append(f"## LOCO class {hc} (mean across 3 seeds)")
        out.append("")
        n_test = int(sub.iloc[0]["n_test"])
        out.append(f"$n_{{test}} = {n_test}$")
        out.append("")
        out.append("| noise mode | RMSE | PIT-KS | cov95 | CRPS |")
        out.append("|---|---|---|---|---|")
        for mode in ["homoscedastic", "per_class"]:
            ssub = sub[sub["noise_mode"] == mode]
            if len(ssub) == 0:
                continue
            rmse = ssub["holdout_rmse"].mean()
            pit = ssub["holdout_pit_ks"].mean()
            cov = ssub["holdout_coverage_95"].mean()
            crps = ssub["holdout_crps"].mean()
            out.append(
                f"| {mode} | {rmse:.0f} | {pit:.3f} | "
                f"{cov:.3f} | {crps:.0f} |"
            )
        out.append("")

    WRITEUP.write_text("\n".join(out) + "\n")
    print(f"[writeup] wrote {WRITEUP.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
