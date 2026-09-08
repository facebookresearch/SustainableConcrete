# Recipe-similarity encoding in the Pareto scatter

Written against `02b1a2b` (branch `feature/pareto-similarity-encoding`, forked from
`18d9792`).

## Status

Feature-complete: metric, rendering, toggle, unit tests, and a 14-invariant e2e spec.
**Not yet reviewed or landed.** See the PR description for current test results.

## Problem

The Pareto scatter plots GWP-or-cost against strength. Two points sitting next to each other
may be chemically unrelated mixes that merely happen to land in the same place. A user
holding a composition cannot tell which of the nearby frontier mixes are **realistic
substitutions** for what they have.

## Approach

Fade each catalog point's **fill** by how unlike the selected recipe it is, while keeping the
**outline** at full opacity in the point's own hue. Hue and radius keep their existing Pareto
meaning.

Keeping the outline solid means position and contrast never degrade — a point can go hollow
but never invisible, which a plain alpha fade cannot promise (`docs/ui.mjs:2120-2127`).

### The metric is the GP's own kernel

`docs/similarity.mjs` computes the **normalized composition kernel of the production strength
GP** — the model's believed correlation between two mixes, in [0,1]:

```js
const kb = matern52ActiveDims(z1, z2, b.active_dims, b.lengthscales, b.outputscale);
const ks = matern52ActiveDims(z1, z2, s.active_dims, s.lengthscales, s.outputscale)
         * hammingFactor(src1, src2, params);
return (kb + ks) / (b.outputscale + s.outputscale);
```

Normalizing by the sum of outputscales is exact because Matérn-5/2 self-similarity *is* the
outputscale, so `s(x,x) === 1`.

**Why not Euclidean or cosine.** The GP's lengthscales are *learned* — it already knows that
Temp matters less than Cement — and it treats Material Source as an unordered categorical via
a Hamming factor rather than a number. A hand-rolled metric would need hand-chosen dimension
weights and would wrongly make source `0→2` twice as far as `0→1`. Nothing here is hand-tuned
except the presentation mapping.

**The time branch is deliberately excluded.** Two mixes are always compared at the same curing
day, so the `h(t)` gate cancels in the normalization and `RBF_time` reduces to its outputscale
— a constant 0.4578 of the 0.5245 total. Folding it in would compress every similarity into
[0.907, 1.000], rendering as a uniform plot rather than an obviously broken one. A unit test
asserts `min < 0.40` specifically to catch that mistake.

### Contrast stretch

Raw catalog similarities span [0.267, 1.0] but with an interquartile range of only
[0.637, 0.828] — mapped straight to alpha, every point would render ~70% solid, i.e. no
visible encoding. So (`similarity.mjs:29-34, 75-79`):

```
t = clamp((s − 0.25) / (1 − 0.25));   alpha = 0.06 + 0.94 · t^2.0
```

`SIMILARITY_FLOOR = 0.25` (empirical catalog min is 0.267), `SIMILARITY_GAMMA = 2.0`
(1.0 leaves everything solid, 3.0 leaves everything hollow), `FILL_ALPHA_MIN = 0.06`. This
spreads a rendered frame across roughly [0.18, 0.95], and a test asserts the spread across a
real frame exceeds 0.5 — that assertion is the actual anti-regression guard on the gamma.

### Caching

Two-level, mirroring the existing `_paretoCache` (`docs/ui.mjs:1923-1946`). Keyed on
`scatterDay | composition`. The `sims` array invalidates on any composition or day change,
but the expensive `ctx` (the transformed catalog) is **retained across composition changes**
and rebuilt only when the curing day changes. So a slider drag runs 149 kernel evaluations
and zero input transforms. Recomputing all 149 similarities costs ~11 µs; rebuilding the
context costs ~0.04 ms.

The reference is the **committed** composition, deliberately not the hover preview —
re-centring on hover would churn the whole field whenever the cursor crossed a point.

`sims === null` is the single "encoding inactive" signal that `drawScatter` branches on.

## Key invariants

Unit tests in `test/test_js_similarity.mjs` (hand-rolled `check()`, exit gate must stay last):

| Invariant | Note |
|---|---|
| `s(x,x) === 1`, symmetric, all 149×149 finite in [0,1] | |
| **`min < 0.40`** | catches the RBF_time-folded-in regression |
| source `0→1` equals `0→2` | Hamming, not ordinal |
| strictly decreasing under growing Cement perturbation | |
| Temp is part of the metric | explicit product requirement |
| **reconstructs `gp.mjs::kernel()` to < 1e-10** | the kernel-parity pin |
| rendered alpha spread across a frame > 0.5 | guards the gamma choice |
| top-5 nearest neighbours of mix #0 === `[0,16,22,19,15]` | exact ordinal equality |
| off-catalog reference stays finite | a dragged slider sits between catalog points |

The golden neighbour list is an **inline constant**, not a fixture file, and is a rank test
rather than a numeric one — insensitive to FP noise, maximally sensitive to lengthscale
changes. Regenerate deliberately, and review the diff, if the strength model is retrained.

The 14 e2e invariants are enumerated in `test/e2e/README.md`, per that file's house rule that
every spec maps to at least one named invariant.

## Traps

- **Run parallel worktrees on different ports.** `playwright.config.ts` reads
  `BOXCRETE_TEST_PORT` (default 4173) and `reuseExistingServer` is on locally, so a server
  left running by another checkout is silently reused — and the suite then tests *that*
  checkout's `docs/`, which presents as inexplicable assertion failures. Use
  `BOXCRETE_TEST_PORT=4183 npm run test:e2e`. This mechanism was added on this branch, almost
  certainly after being bitten by it.

- **The metric is coupled to the shipped model artifact.** Retraining the strength GP changes
  similarities, the golden neighbour list, and possibly the calibrated floor. The kernel-parity
  test is what stops `similarity.mjs` and `gp.mjs` drifting apart.

- **Filtered-out points are categorically different from dissimilar ones.** They are drawn
  first, in grey, and excluded from the similarity reorder. Exclusion must never read as
  "far away".

- **The toggle handler must call `invalidateCanvases(CANVAS_SCATTER)`, not `drawScatter()`.**
  Canvas renderers stay owned by `requestAnimationFrame`, which `test/e2e/canvas-frame-probe.ts`
  asserts.

- **Dangling reference.** `similarity.mjs:69-70` cites "the plan's gamma table" for the 1.0 /
  3.0 alternatives. That plan is not in the repo — the repo gitignores `*.plan.md`
  (`.gitignore:72`), so it was scratch and is now lost. The two rejected gamma values are
  recorded in the constants' comments, which is the surviving record.

## What's left

- Rebase onto `main` (forked at `18d9792`). Expect conflicts in `Makefile`,
  `docs/index.html`, `docs/style.css`, `docs/ui.mjs`, `playwright.config.ts`.
- Verify against `main`'s current four-project Playwright matrix; the e2e spec is scoped
  `desktop` only, so WebKit behaviour of the alpha encoding is unverified.
- Similarity state is in-memory only, deliberately not persisted — revisit only if the axis
  and unit toggles start persisting too.

## How to verify

```bash
npm ci
node test/test_js_similarity.mjs
make test-js
BOXCRETE_TEST_PORT=4183 npx playwright test test/e2e/pareto-similarity.spec.ts --project=desktop
```
