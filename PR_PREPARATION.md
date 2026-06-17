# PR preparation plan — `joint_hamming_matern` production migration

This document tracks the cleanup work needed to ship the
`joint_hamming_matern` source-kernel migration as a focused PR,
following the precedent set by the V2 strength GP PR
(`#26 v2-strength-gp`, public commit `73cff6b21e11`).

## Status snapshot

* **40 non-public commits** on top of the V2 merge.
* **93 / 93 tests pass** after pre-flight fix to `test/test_kernel_layout.py`
  (commit `cca2ab80f71a`) which handled the new ProductKernel and
  joint-kernel layouts.
* **No production-blocker code issues**. The remaining work is
  packaging (commit squash + file cleanup + production-default flip
  + JSON regen + gp.mjs update).

## Phase A — pre-flight fixes ✅ DONE

* `cca2ab80f71a` fixed `test_kernel_layout.py` for the
  ProductKernel + joint-kernel structures (was pre-existing breakage
  from the v5 categorical migration; not from recent work).

## Phase B — proposed 6-commit squash

| commit | content | source commits to fold |
|---|---|---|
| 1. `data: v5 three-class dataset (canonical naming + corruption-collision splits + clay-mortar drop)` | `data/boxcrete_data.csv`, `test/fixtures/boxcrete_data_pre_v5.csv`, naming infrastructure, class-2 GWP | `7d7fc0bff104`, `545ef301afcf`, `5103a9d2e957`, fold in `scripts/merge_three_class_data.py`, `scripts/derive_class_2_gwp.py`, `scripts/test_merge_invariants.py`, `test/test_mix_naming.py`, `boxcrete/mix_naming.py`, `boxcrete/_mix_naming_table.csv` |
| 2. `boxcrete: 3-class Material Source categorical kernel framework` | Categorical kernel switch, within-group prior framework | `6d360fa025ad`, `814b4eae4794`, `23bb80732310`, `c1bfc4918e73`, `e930f32fb146`, related fixes + `test/test_composed_prior.py` |
| 3. `boxcrete: add JointHammingMaternKernel + joint-distance kernel family` | New kernels and their unit tests | `261e5f0a7d82`, `311e08646a61`, `7b222d73e774` (additive hybrids), `cf62d1c128b8` (RBF embedding extensions), `d03516d6dbcf` (RBFEmbeddingKernel), `79e89f8fa5ab` (`include_blind` flag) |
| 4. `boxcrete: flip production default to joint_hamming_matern` | The actual production change | NEW: `DEFAULT_SOURCE_KERNEL = "joint_hamming_matern"` + `regenerate_strength_json.py` update + `docs/model/strength.json` regen + `docs/gp.mjs` update |
| 5. `experiments: research infrastructure` | Three-class ablation evaluation framework, regenerate script | `bab29005acdc`, `98b15bc2a4b6`, `6fbf7458ba0d`, `2434bbcbe36d`, `346550783cf2`, `41d43257a8d3`, `8c83273f02d8`, `6ed49083abee`, `6060b41edf70` |
| 6. `experiments: 3-class benchmark writeup` | Public writeup | All `experiments/THREE_CLASS_AND_PRIOR_BENCHMARK.md` commits + `cca2ab80f71a` (test layout fix) |

**Execution** (after confirming the scope):

```sh
# Save current state as recovery point
sl bookmark pr-prep-backup

# Reorder + fold per the plan above. Use `sl fold --from <commit>` to
# combine adjacent commits into the target.
# Example for commit-3:
sl fold -r 'd03516d6dbcf::311e08646a61' \
  --message "boxcrete: add JointHammingMaternKernel + joint-distance kernel family"
# Continue for the other 5 logical commits.
```

## Phase C — items deferred (NOT in PR)

These are kept locally / on the research branch but excluded from the
PR — mirroring how the V2 PR deferred its ablation infrastructure:

| category | files | rationale |
|---|---|---|
| **Investigation scripts** (16 files) | `scripts/investigate_*.py`, `scripts/noise_audit.py`, `scripts/pareto_analysis.py` | Experimental tooling, not load-bearing for production |
| **Auto-generated writeups** (~15 files) | `experiments/{RBF_*,JOINT_*,HYBRID_*,M80_*,M81_*,NOISE_*,NO_BLIND_*,PARETO_*,PER_CLASS_*,V5_VS_*}.md` | Per-experiment detail; redundant with main `THREE_CLASS_AND_PRIOR_BENCHMARK.md` |
| **Auto-generated CSVs** (~12 files) | `experiments/*.csv` | Raw per-cell metrics |
| **Research-only likelihood** | `PerClassGatedGaussianLikelihood` class in `boxcrete/likelihoods.py` | Negative result — should be **removed from `likelihoods.py`** before PR (covered by `de64ce6257c2`'s writeup but not production code) |
| **Research-only kernel flag** | `include_blind=False` parameter on `build_strength_kernel_for_aug_dim` and `make_gated_strength_kernel_builder` | Research flag, default `True` is production; **safe to remove the flag** (always-True semantics) |
| **Research-only kernel variants** | `rbf_embedding_d{1,2,3}_fixed_ell`, `rbf_embedding_d1_linear_init`, `rbf_embedding_d1_linear_init_fixed_ell` (in `_SUPPORTED_SOURCE_KERNELS` + the dispatch logic in `_categorical_source_branch`) | Negative or research-only variants. Branch logic in `_categorical_source_branch` handles them via suffix parsing; **safe to remove** |
| Research-only kernel classes (KEEP per current call) | `RBFEmbeddingKernel`, `JointEmbeddingMaternKernel`, additive `additive_joint_hamming_*` variants | Pareto-corners or close ablation alternatives. Referenced in the main writeup. **Keep as registered ablation variants** — they don't pollute the production code surface; each is one branch in the `_categorical_source_branch` switch. |

## Phase D — production-default flip (must do before PR review)

This is the actual public-facing change:

1. **`DEFAULT_SOURCE_KERNEL = "joint_hamming_matern"`** in `boxcrete/kernels.py` (currently still `"hamming"`).
2. **Update `experiments/regenerate_strength_json.py`** to walk the `JointHammingMaternKernel` introspection path:
   * Read `kernel.raw_feat_lengthscale` (per-feature ARD) via `kernel.lengthscale` property
   * Read `kernel.raw_alpha` via `kernel.alpha` property
   * Emit a new `source_kernel_kind = "joint_hamming_matern"` JSON case alongside the existing `hamming` / `indexkernel` / `onehot` cases.
3. **Regenerate `docs/model/strength.json`**:
   ```sh
   bash experiments/regenerate_all_artifacts.sh
   ```
4. **Update `docs/gp.mjs`** to read the new joint-kernel parameters
   and compute the joint Matern distance at inference time. The
   JS port currently has separate paths for Matern-on-continuous and
   categorical-rho; for the joint kernel, the JS-side distance
   computation needs:
   ```js
   const d2 = sum_f((x_f - x_test_f) / ell_f)**2 + alpha * (c !== c_test ? 1 : 0)
   const k = matern32(Math.sqrt(d2))
   ```
5. **Run JS-Python equivalence test**: `bash test/test_js_gp.mjs` —
   verifies the JS port produces identical predictions to the
   Python model.

## Phase E — deep code review checklist

Before opening the PR, audit:

* [ ] `boxcrete/kernels.py` — Remove `_fixed_ell` / `_linear_init` RBF variants from `_SUPPORTED_SOURCE_KERNELS` and `_categorical_source_branch` dispatch (research-only).
* [ ] `boxcrete/kernels.py` — Remove `include_blind` parameter (research-only, default `True` is production).
* [ ] `boxcrete/likelihoods.py` — Remove `PerClassGatedGaussianLikelihood` (research-only, negative result).
* [ ] `experiments/THREE_CLASS_AND_PRIOR_BENCHMARK.md` — Update any references to deferred experimental writeups so they don't dangle.
* [ ] `experiments/three_class_ablation.py` + `model_variant_study.py` — Verify they don't reference removed variants/likelihoods.
* [ ] `test/` — `test_joint_hamming_matern_kernel.py`, `test_joint_embedding_matern_kernel.py`, `test_rbf_embedding_kernel.py` all pass; no references to removed flags.
* [ ] `docs/model/strength.json` regenerated.
* [ ] `docs/gp.mjs` updated and consistent with `strength.json`.
* [ ] `test/test_js_gp.mjs` passes.
* [ ] `bash experiments/regenerate_all_artifacts.sh` runs end-to-end without errors.

## Phase F — PR description draft

> **Switch the production strength GP from `hamming` to `joint_hamming_matern` source kernel.**
>
> The v5 strength dataset has a corrected 3-class Material Source label
> (mortar / Set-2 / Set-3) where the deployed pre-v5 model pooled the
> first two classes and contaminated the third. The `joint_hamming_matern`
> kernel handles this 3-class categorical correctly while preserving the
> joint kernel topology that the deployed model's continuous-source-ARD
> formulation has — closing 33 of the 36 psi bLOO architectural gap that
> the simpler `hamming` (product-kernel) topology leaves on the table.
>
> Strictly dominates the previous `hamming` default on every
> in-distribution metric (LOO 510 → 495, bLOO 738 → 717, PIT-KS 0.039 →
> 0.030), and within pre-registered LOCO acceptance criteria
> on all three held-out classes.
>
> Full ablation: `experiments/THREE_CLASS_AND_PRIOR_BENCHMARK.md`.
> The research detail (per-experiment ablation writeups, investigation
> scripts, ablation CSVs) is deferred to a follow-up research repo
> following the V2 PR's pattern.

## Outstanding open questions for the PR author

1. **Should `RBFEmbeddingKernel` and `JointEmbeddingMaternKernel` ship as registered ablation variants, or be deferred entirely?**
   * Current proposal: **ship as registered** (Pareto-corner / close-alternative; cited in writeup; ~700 lines total). Removing them would require redacting the writeup's "what didn't work" section. Resolution: **keep**.
2. **Should the `joint_chain_matern` / `_nu05` / `_nu25` smoothness variants ship?**
   * Cheap surface (one branch each in the switch). Resolution: **keep** for completeness of the ablation family.
3. **Should the additive hybrid variants ship?**
   * They're Pareto-improving but Occam-rejected. Resolution: **keep** behind their explicit `additive_*` prefix, so they're never accidentally instantiated.
4. **PR target branch?** Recommend opening against `main` post-rebase on top of the V2 merge.
