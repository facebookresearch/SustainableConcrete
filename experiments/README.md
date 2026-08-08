# `experiments/`

Research scripts, training/evaluation pipeline, and benchmark writeup
for the strength GP. Distinct from `docs/` (the public web explorer)
and `scripts/` (production CLI utilities).

## Quick start

```bash
# Re-train the deployed champion + regenerate all artifacts + run all tests
bash experiments/regenerate_all_artifacts.sh
```

## Contents

### Documentation

| File | Purpose |
|---|---|
| `STRENGTH_GP_BENCHMARK.md` | Full architecture study writeup. **Start here.** TL;DR + lessons learned + section-by-section narrative across 250+ tested variants. Appendix A folds in the companion anchors-vs-gating study. |

### Pipeline

| File | Purpose |
|---|---|
| `regenerate_all_artifacts.sh` | **Single-command full pipeline**: re-train → re-export → regenerate compositions → run all tests. Idempotent. |
| `regenerate_strength_json.py` | Train champion variant (Python), export model parameters to `docs/model/strength.json`. |
| `regenerate_compositions_strength_predictions.mjs` | Recompute precomputed strength predictions in `docs/model/compositions.json` after re-training. |
| `augment_test_vectors_with_gwp_cost.mjs` | Add GWP/cost predictions to `docs/model/test_vectors.json` for cross-language regression testing. |

### Variant catalog

| File | Purpose |
|---|---|
| `model_variant_study.py` | All ~250 historically-tested model variants, registered in a `VARIANTS` dispatch dict. Each variant is a small kernel-builder + fit-adapter closure. CLI entrypoint for running any subset of variants on the full or subsetted data. |

### Diagnostics & visualization

| File | Purpose |
|---|---|
| `compare_monotonicity.py` | Compare candidate variants on physical-realism metrics (% decreasing intervals, % oscillating, max dropdown). Used for §6.13 of the benchmark. |
| `plot_improvement_journey.py` | Renders `improvement_journey.png` showing the cumulative block-LOO RMSE gain stage-by-stage from production baseline to deployed champion. |

## Reproduction recipes

### The deployed champion

```bash
python experiments/model_variant_study.py --variants \
    "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_only" \
    --seeds 0
```

Block-LOO RMSE: **672 psi** at full data.

### The §4 architecture decomposition (the journey)

```bash
python experiments/model_variant_study.py --variants \
    baseline_no_prior baseline \
    "single_matern+F0+gated_t+maxscale_zeromean+prior+rbf_t" \
    "single_matern+F3+gated_t+maxscale_zeromean+prior+rbf_t" \
    "single_matern+F5_alllog+gated_t+maxscale_zeromean+prior+rbf_t" \
    "B''+F5_alllog+gated_t+maxscale_zeromean" \
    "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean" \
    "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_only" \
    --seeds 0
python experiments/plot_improvement_journey.py
```

### Subset learning curves (the small-data regime where block_loo_only shines)

```bash
for n in 25 50 100; do
  for s in 0 1 2; do
    python experiments/model_variant_study.py --variants \
        "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean" \
        "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_only" \
        --seeds 0 --subset_n $n --subset_seed $s --holdout_unit composition
  done
done
```

### Curve monotonicity diagnostic

```bash
python experiments/compare_monotonicity.py
```

Tests the deployed champion plus alternative regularizer settings.
Reports `(BLOO, %dec, %osc, maxDrop)` per variant — caught the
F5_no_log_mat regression that shipped briefly in 2026-05-16.

## CSV results files

| File | Schema |
|---|---|
| `strength_gp_benchmark_results.csv` | 108 per-seed metric rows from the original kernel/noise/feature sweep (LOO-only) |
| `strength_gp_subset_loo_vs_heldout.csv` | 27 paired LOO + held-out rows from the row-level subset learning curves |
| `strength_gp_subset_composition_holdout.csv` | 27 paired LOO + held-out rows from the composition-level subset learning curves (the realistic extrapolation regime) |
| `strength_gp_block_loo_full.csv` | 7 paired single-row LOO + block-LOO rows from the closed-form leave-one-composition-out evaluation at full data |

## See also

- **Skill file**: `~/.llms/skills/strength_gp_architecture_optimization.md`
  documents how to re-run the full architecture sweep when training data
  changes, plus the common pitfalls we encountered.
- **Tests**: `test/test_*.{py,mjs}` — run via `bash experiments/regenerate_all_artifacts.sh`.
- **Production library**: `boxcrete/` — the model and utilities that
  the experiment code wraps.
