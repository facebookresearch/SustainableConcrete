# Slump prediction in the explorer readout strip

Written against `5edb2a1` (branch `feature/explorer-slump-prediction`, forked from
`18d9792`).

## Status

Feature-complete. Verified at the time of authoring against `18d9792` with 176/176 JS parity
assertions, 270 Python tests at 100% coverage, and the then-current Playwright matrix.
**Those results predate `main`'s current state** — see [What's left](#whats-left) for two
specific staleness issues. See the PR description for current results.

## Problem

The strength-curve readout strip showed GWP, Cost, and W/B. W/B (water-to-binder) is a model
*input* ratio sitting next to two predicted outcomes. Replacing it with the slump prediction
makes all three numbers measured outcome properties, and surfaces the slump model — which
existed in `boxcrete` but was invisible on the site.

## Approach

`experiments/regenerate_slump_json.py` exports the fitted slump GP as **kernel ingredients**
(`X_train`, hyperparameters, noise, `Y_train`) rather than a precomputed Cholesky factor or
`α`. `docs/slump.mjs` rebuilds `K` and factors it itself, so both sides reach the posterior
from the same kernel evaluations instead of JS inheriting Python's factorization. Same
discipline as the strength exporter.

The exporter and its 85 golden posteriors come from a **single fit** — the MLL objective is
non-convex, so two runs land on different optima and the golden vectors would not match the
shipped model.

### Why a separate module rather than extending `gp.mjs`

`gp.mjs::transformInput` unconditionally log-scales a time dimension and `kernel()`
unconditionally applies a time gate plus an RBF-time term. Slump is measured pre-cure and has
no time dimension. Threading schema flags through the strength model's 60 fps hot path —
which four existing tests pin — to share ~50 lines is a bad trade. The linear algebra *is*
shared: `cholesky` and `solveTriangularLower` are imported.

### Mortar renders "n/a"

Slump has 61 observations, **all from Material Source 1 and 2 and none from 0 (mortar)** —
the `Normalize` bounds on that dim are literally `[1, 2]`. This is physically correct: mortar
workability is measured with a flow table, not a slump cone. Rather than extrapolate a
stationary GP into a region with zero support, the readout shows `n/a` with an explanatory
tooltip. 69 of the explorer's 149 catalog mixes are mortar, so this path is common, not an
edge case.

## Key invariants

| Invariant | Pinned by |
|---|---|
| JS posterior matches Python to 1e-6 | `test/test_js_slump.mjs`, 176 assertions (worst observed 6.7e-15) |
| Exported architecture is what the JS implements | four `isinstance` guards in the exporter |
| Column order has not drifted | full `X_columns` list compared, not just `len()` |
| Only sources 1 and 2 are supported | `supported_source_classes` + the exporter's observed-class assertion |
| Slump units convert in→mm | `test/test_js_units.mjs` |
| Mortar shows `n/a`, unit toggle rescales the value | `test/e2e/readouts-strip.spec.ts` |
| Artifacts do not silently go stale | `check_artifacts_drift.py`, prediction-tolerance not bit-equality |

## Traps

- **The model is ARD RBF, not Matérn.** `slump_model.py`'s docstring claimed "single-Matern".
  BoTorch's `SingleTaskGP` default for this input is an **ARD `RBFKernel` with no
  `ScaleKernel`** (so `outputscale` is exactly 1), plus `ConstantMean` and `Standardize(1)` —
  not the strength model's `ZeroMean` and max-scaling. The docstring was corrected, and the
  exporter now hard-fails on any of the four if BoTorch's defaults move.

- **`hrwr_binder` differs between the two models.** Slump uses `hrwr / max(binder, 1.0)`
  (`boxcrete.features.AppendDerivedFeatures`); the strength registry uses
  `hrwr / (binder + 1.0)`. They disagree by ~0.3% on real mixes. `docs/slump.mjs` implements
  its own copy deliberately, and `test/test_js_slump.mjs` pins the difference **in both
  directions** so neither can drift into the other. Do not "deduplicate" them.

- **Slump is the one quantity stored in imperial (inches).** Its unit factors therefore run
  inverted relative to mass/GWP/cost: metric multiplies by 25.4, imperial is the no-op.

- **The unit toggle never recomputed readout values** — it only relabelled. This was a
  *pre-existing* bug affecting GWP (1.69×) and Cost; latent enough to go unnoticed, but at
  slump's 25.4× it would have rendered `6.8 mm` for a 6.8 in prediction. Fixed at the root
  with one guarded `updateReadouts()` call rather than adding a third readout to the broken
  path.

- **`test_js_units.mjs` has an exit gate at the bottom.** Assertions appended after the
  `// --- Summary ---` block run after `process.exit(1)` has already been evaluated and can
  never fail the suite. Its `assertClose` also scored `undefined` as a pass until it was
  hardened with a `Number.isFinite` check.

- **The mobile/desktop CSS was measured, not assumed.** Three readouts do not fit on one line
  at 412px, so mobile hides both ±2σ suffixes *and* the GWP unit (the widest token, 79px, and
  static). The slump and cost units stay because slump's flips mm↔in with the toggle, so a
  bare number would be ambiguous by 25.4×. At 1280px desktop the items wrapped mid-value, so
  `@media (max-width: 1439px)` hides the slump ±2σ.

  **`white-space: nowrap` was tried and rejected.** It gave a clean single row from
  1100–1600px with no overflow, but perturbed layout enough to trigger an extra scatter-canvas
  redraw, failing two `canvas-scheduling` invariants (12/12 pass without it, 10/12 with).
  Not worth destabilizing the redraw scheduler for a cosmetic guard.

- **Slump is the noisiest target**: LOO R² ≈ 0.336 (`test/test_models.py`), versus > 0.90 for
  strength. The slider bounds also far exceed slump's training support in most dimensions
  (Slag 0–1198 vs 0–367; HRWR 0–13.3 vs 0–4.63). The stationary RBF with outputscale 1 means
  the far-field posterior reverts to ~6.9 in with ±2σ ≈ ±6.6 in. That is honest, and the ±2σ
  display is load-bearing rather than decorative.

## What's left

Two staleness issues introduced by `main` moving underneath this branch (`f57d82b`):

1. **The regenerated visual baseline is obsolete.** This branch updated
   `home-desktop-desktop-linux.png`; `main` deleted the `home-*` snapshots and replaced the
   suite with `dashboard-*`, `composition-*`, `references-*`, `tradeoffs-*`. On rebase, drop
   this branch's `home-*.png` and regenerate against `main`'s suite. Baselines are
   Linux-rendered and the spec `test.skip`s on macOS, so this needs Docker
   (`mcr.microsoft.com/playwright:v<version>-noble`) or a CI dispatch.

2. **Only two of four Playwright projects were run.** `main` now has `desktop`, `mobile`,
   `mobile-webkit`, `desktop-webkit`. This branch was verified against the first two only, so
   the readout layout is unverified on WebKit.

Also on rebase: `ux/concrete-explorer-guided-insights` adds
`test/e2e/dashboard-helpers.ts::openMobileComposition`, which does exactly what this branch's
`readouts-strip.spec.ts` hand-rolls. If that branch lands first, switch to the helper.

Expect conflicts in `Makefile`, `docs/style.css`, `test/e2e/README.md`,
`test/e2e/readouts-strip.spec.ts`.

Optional follow-ups, deliberately out of scope: a slump axis in the scatter/Pareto view would
need a story for the 46% of catalog mixes where slump is undefined.

## How to verify

```bash
npm ci
node test/test_js_slump.mjs      # expect "All 176 slump assertions passed."
make test-js
make lint && make test-py        # 100% coverage gate
npx playwright test readouts-strip
python experiments/regenerate_slump_json.py   # regenerates BOTH json artifacts
```

`slump.json` and `slump_test_vectors.json` must always be regenerated **together** — they
come from one fit, and `test_js_slump.mjs` compares them at rtol 1e-6.
