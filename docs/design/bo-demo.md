# BO mode — retrospective Bayesian-optimization replay

Written against `1f57ed0` (branch `feature/bo-demo`, forked from `18d9792`).

## Status

Feature-complete and self-consistent; **not yet reviewed or landed**. All seven phases
including the UI wiring are done — the site loads `bo.mjs` and `bo_view.mjs` and the header
carries a "Run Optimization" button.

Known gaps are listed under [What's left](#whats-left). See the PR description for the
current test results; do not assume green.

## Problem

The explorer shows a static catalog. It cannot answer *"would a multi-objective optimizer
have found the good mixes, and how much faster than picking at random?"* — the question that
justifies using BO for concrete design at all.

## Approach

A **retrospective replay**, not a live optimizer. Hyperparameters stay frozen at the
full-data fit and every outcome comes from a measurement already in the dataset. This is
stated in the UI (`docs/index.html:218-222`) because otherwise the comparison looks rigged.

Per iteration the engine scores every unacquired catalog mix by expected hypervolume
improvement, acquires the best, conditions the GP on that mix's observation rows, and
records the hypervolume of what has actually been measured so far. A random arm replays the
same process picking uniformly, over 200 restarts, to give a p10–p90 band.

### Objectives

Two, always: **minimise** GWP *or* cost (whichever the scatter's X toggle shows) and
**maximise** strength at day 1 or 28. GWP and cost are alternatives on one axis, not a third
objective. The x-objective is deterministic (linear models, known exactly); only y is
Gaussian. Three-objective fronts are out of scope.

Reference point (`bo.mjs:23-28`) mirrors `boxcrete.CONCRETE_REFERENCE_POINT` and
`CONCRETE_COST_THRESHOLD`. `referenceFor(xAxis, day)` **throws rather than defaulting** —
a silently-wrong reference point makes every hypervolume number meaningless with no visible
symptom.

### Why EHVI is closed-form here

Because x is known exactly at the candidate's value `g`, hypervolume improvement is
piecewise-linear in y, so the expectation passes straight through the sum over Pareto
segments:

```
E[HVI] = Σ_segments  width · EI(mu, sd; segment_ceiling)
```

Exact, `O(P)` per candidate, no quadrature and no sampling (`bo.mjs:195-234`).

### Why a hand-rolled erfc

`bo.mjs:131-145` uses Numerical Recipes' 24-coefficient Chebyshev form, **not** Abramowitz &
Stegun 7.1.26. A&S is accurate to ~1.5e-7 *absolutely*, but Φ(−5) ≈ 2.87e-7 — so at the tail
that is a ~50% *relative* error and a candidate several σ below the ceiling would get a
meaningless acquisition value. The chosen form holds ~1e-15 relative out to z = −7.

### Why incremental Cholesky

A refit is `O(n³)` per iteration; extending the factor is `O(n²b)`. Keeping `w = L⁻¹y` and
`V = L⁻¹K_test` incrementally puts an iteration at roughly 0.5 MFLOP at n=670 — cheap enough
for the main thread, so **no web worker and no Lighthouse TBT regression**.

### Subset-model strength curve (Phase 4.5)

In BO mode the strength panel is re-pointed at the just-acquired mix using a GP conditioned
on **only the rows acquired so far** — same frozen hyperparameters, different conditioning
set. Leaving it on the full-data model would show a confident band next to a scatter driven
by a handful of observations, which is actively contradictory rather than merely a missed
opportunity (`bo.mjs:871-877`).

## The fairness contract

The most important thing on this branch. Three guarantees, each with a named test
(`test_js_bo.mjs:1118-1194`):

1. **Both arms start from an identical seed set.** `randomArmTraces` takes `seedMixes` and
   excludes them from its draw pool. At iteration 0 the p10 and p90 bands must equal the BO
   hypervolume exactly — zero spread across all 200 restarts. Giving each arm its own random
   seeds confounds the comparison; an earlier throwaway prototype did exactly that and made
   BO look better than it is.

2. **Both arms are scored on measured outcomes, never predictions.** The model chooses what
   to acquire; it never grades itself. Pinned by replaying a fixed acquisition sequence twice
   — once against the real GP, once against a model poisoned with `Y_train.map(() => 0.5)` —
   and asserting the hypervolume traces are identical. The test notes *"no coverage metric
   catches this"*, which is the branch's own argument for why 100% coverage is
   necessary-but-not-sufficient.

3. **The random arm cannot reach the GP even by accident** — asserted structurally, via
   `!/strengthParams/.test(randomArmTraces.toString())`.

## Key invariants

| Invariant | Pinned by |
|---|---|
| EHVI reduces to exact HVI as sd→0 | `test_js_bo.mjs:381-451` (1e-9) |
| EHVI agrees with Monte Carlo | same, N=200k fixed-seed, 1% relative |
| Far-tail Φ accuracy at z=−5, −7 | `test_js_bo.mjs:312-379` (1e-13) |
| JS ↔ Python fixture | `test_js_bo.mjs:1087`, `1e-9·max(\|ehvi\|,1)` |
| Python ↔ BoTorch | `test_bo_reference.py:119`, rel 1e-9 |
| Separate `kernelBlock` never diverges from `gp.mjs` | `test_js_bo.mjs:576-597` |
| 100% line/branch/function coverage of `bo.mjs`, `bo_view.mjs` | `make test-js-bo` |

Golden vectors are generated by `python -m experiments.regenerate_bo_golden`, validated
against `botorch...ExpectedHypervolumeImprovement`. The script **refuses to write the
fixture** if any case disagrees, so anything committed has already passed an outside
authority.

## Traps

- **Run the generator with `-m`.** `python experiments/regenerate_bo_golden.py` puts
  `experiments/` on `sys.path` instead of the repo root, so `boxcrete` resolves through
  whatever `pip install -e .` last pointed at — **in a git worktree that is the primary
  checkout**, silently importing a different tree than the one you are editing.

- **`L` is row-major lower-triangular with leading dimension `ld` = capacity; the `[n×b]`
  blocks are column-major** (matching `gp_v2_fast.mjs`). Mixing these up is the easiest bug
  to introduce here.

- **Jitter is added to the accumulator, never written back into `A`**, so retries cannot
  compound. Rank-deficient blocks are real, not hypothetical: acquiring a mix whose
  observations duplicate ones already conditioned on produces exactly that.

- **Node only reports coverage for files it actually loaded.** A module no test imports is
  invisible and vacuously passes at 100%. `Makefile:162-168` greps the report for each module
  name to make that failure loud.

- **`CI=1` in the e2e target is load-bearing.** `playwright.config.ts` keys worker and retry
  count off it. Measured on a clean checkout: without it, 9 failed / 85 passed from pure
  timing flakiness; with it, 0 failed / 94 passed on the same commit.

- **Guarantee 3 is a source-text assertion** and will break under any minifier or transpiler
  that renames the `strengthParams` parameter. It also passes vacuously if someone renames
  the variable while still threading the model through.

- **Phase 3.3 was reversed.** The plan called for extracting kernel evaluation from
  `gp_v2_fast.mjs` for reuse. Instead `bo.mjs:328-425` builds a separate `kernelBlock`,
  because `gp_v2_fast`'s hot loop is deliberately inlined and its header records measured
  timings for that inlining. The parity test is the mitigation. The commit subject still says
  "Phase 3.2/3.3", which is misleading.

## What's left

- **Node 22 bump is cross-cutting.** This branch moves all four workflows from `'20'` to
  `'22'` (`js-sync`, `e2e`, `lighthouse`, `model-artifacts-coherence`); `origin/main` is still
  on 20. Required because `--test-coverage-*` threshold flags need Node ≥ 22.8. Coordinate
  this with whatever lands first.
- Rebase onto `main`. This branch forked at `18d9792`; expect conflicts in `Makefile`,
  `docs/index.html`, `docs/style.css`, `docs/ui.mjs` — the files every in-flight explorer
  branch touches.
- `docs/style.css` changes for `.bo-*` were not independently reviewed during this writeup.
- Verify against `main`'s current four-project Playwright matrix (desktop, mobile,
  mobile-webkit, desktop-webkit). This branch predates `mobile-webkit`/`desktop-webkit`
  coverage being routine.

## How to verify

```bash
npm ci
make test-js            # legacy JS suite, then delegates to test-js-bo
make test-js-bo         # BO engine + view, with the 100% coverage gate
make test-e2e           # CI=1 desktop + mobile
python -m pytest test/test_bo_reference.py -q
python -m experiments.regenerate_bo_golden   # note the -m
```
