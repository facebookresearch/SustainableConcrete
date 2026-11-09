# `experiments/mix_narratives/`

**Research-only — not part of the production release.**

How `docs/model/mix_analyses.json` (the per-mix prose shown in the
explorer's insight panel) is produced.

The 2-class to 3-class merge renumbered the candidate catalog, so the
previous narratives could not be carried over: only 31 of the 144 old
entries were still index-aligned to the same composition, and 70 indices
had changed `Material Source` label. Restoring them by index would have
attached descriptions to the wrong mixes. The catalog was therefore
re-authored from scratch for all 149 entries.

## Pipeline

```
python experiments/mix_narratives/build_mix_facts.py
python experiments/mix_narratives/author_mix_analyses.py
```

1. **`build_mix_facts.py`** joins `docs/model/compositions.json` (index-keyed,
   no mix names) back to `data/boxcrete_data.csv` by exact composition
   match, recovering the canonical mix name for every catalog index. It
   emits `mix_facts.json`: composition, derived mix-design metrics
   (binder, SCM fraction, w/b, w/c, HRWR dosage, aggregate split, paste
   mass), the measured strength trajectory, GWP, cost, Pareto status, and
   class-relative percentiles. The join is exact — worst L1 mismatch is
   0.0 across all 149 entries, with no unmatched mixes.

2. **`author_mix_analyses.py`** composes each entry from two parts:
   * a factual opening built entirely from `mix_facts.json`, so no figure
     in the prose can drift from the shipped catalog;
   * a hand-authored `ANALYSIS` string per mix — the interpretation that
     cannot be derived mechanically (which designed series the mix belongs
     to, what single variable it isolates, how it compares to its
     siblings, what its behaviour demonstrates).

   It deliberately does **not** restate the measured strength points, the
   embodied-carbon figure, or Pareto status. The explorer already renders
   all three: observations are overlaid on the strength curve
   (`docs/ui.mjs`), GWP/cost/W-B have their own readouts, and Pareto
   membership is a pill above the insight panel. Repeating them only
   lengthened every entry. Where one of those facts carries an argument
   (M54 having the lowest mortar carbon, C6 earning its Pareto slot on
   efficiency), it lives in that mix's hand-authored `ANALYSIS` instead.

## Why the claim checker exists

Prose superlatives ("the strongest mix in the dataset", "the lowest
embodied carbon of any mortar") rot silently when the dataset is
re-merged. `author_mix_analyses.py::check_claims` asserts all 27 such
claims against `mix_facts.json` and fails the build if any is
contradicted.

This is not hypothetical — during authoring it caught seven false
statements, including:

  * `M63` described as the lowest-carbon mortar when `M54` and `M65` are
    lower;
  * `C62`/`C63` described as sharing a replacement level when only their
    binder content matches;
  * `C28` called low-carbon while sitting above its class median;
  * `C1`-`C4` described as an aggregate-*content* series when total
    aggregate is roughly constant (~1,780 kg/m3) and only the coarse/fine
    split varies — and their strengths are non-monotonic in coarse
    fraction, so the spread is batch variability rather than a gradation
    effect.

## Regenerating

`experiments/regenerate_all_artifacts.sh` deliberately does **not** run
these scripts, and also does not run `docs/generate_mix_analyses.py`
(whose templated fallback output would overwrite the authored prose).
Re-run this pipeline only when the catalog itself changes; any new mix
needs a new hand-authored `ANALYSIS` entry, and the build will fail with
`Missing hand-authored analysis for: [...]` until it has one.
