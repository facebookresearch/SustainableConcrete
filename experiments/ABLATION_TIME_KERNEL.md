# Time-kernel structure ablation

## Setup

The user hypothesised that the additive `RBF(t)` time-only kernel
component might be the source of strength-curve non-monotonicity in
the v5 + `joint_hamming_matern` model. The V2 benchmark
(experiments/STRENGTH_GP_BENCHMARK.md §4.1) showed `RBF(t)`
contributes +20 psi block-LOO when paired with F5_alllog features on
the *legacy continuous-source* kernel — but this ablation was never
re-run on the v5 + `joint_hamming_matern` architecture.

This ablation asks: with the *new* kernel architecture, does the
choice of time-only kernel still matter? Does dropping it help
monotonicity? Does swapping for a different kernel type help?

## Variants

| variant | time-only branch |
|---|---|
| `rbf_time` (production) | ScaleKernel(RBF(t)) |
| `matern52_time` | ScaleKernel(Matern_5/2(t)) |
| `matern32_time` | ScaleKernel(Matern_3/2(t)) |
| `linear_time` | ScaleKernel(LinearKernel(t)) (monotonic posterior for non-neg data) |
| `no_time` | outputscale clamped to 1e-12 (effectively dropped) |

Each refit from scratch with seed=0 on v5 data.

Run: `python experiments/ablation_time_kernel.py`

## Results

| variant | LOO RMSE | bLOO RMSE | % mix w/ drop | max single-step drop |
|---|---|---|---|---|
| **rbf_time** (production) | 501.3 | 505.9 | 22.2% | 29.2 psi |
| matern52_time | 501.0 | 505.7 | 22.9% | 29.0 psi |
| matern32_time | 501.8 | 506.5 | 22.2% | 29.2 psi |
| linear_time | 502.4 | 506.9 | 24.3% | 29.9 psi |
| **no_time** (drop entirely) | 502.1 | 506.7 | 22.2% | 29.1 psi |

## Headline finding: the time-only kernel doesn't matter

**All five variants give essentially identical metrics** on v5 +
`joint_hamming_matern`: LOO within 1.4 psi, bLOO within 1.2 psi,
non-monotonicity floor at 22-24%, max single-step drop within
0.9 psi.

Notably, **dropping the time-only kernel entirely (`no_time`)
preserves all metrics** within noise. This is a substantial
departure from the V2 finding (where `RBF(t)` contributed +20 psi
block-LOO).

**Interpretation**: with the joint kernel architecture, the
composition × time interaction is captured *inside* the
`JointHammingMaternKernel` and the blind Matern's ARD time dim.
The additive RBF-on-time channel is **redundant** — there's no
"smooth time correction" left for it to model after the joint
kernel is in place.

## Implication for the non-monotonicity bug

The user's hypothesis (that `RBF(t)` causes non-monotonicity in
the early-hour region) is **falsified** by this ablation:
removing the additive time kernel does **not** reduce the drop
fraction at all (22.2% in both `rbf_time` and `no_time`).

**The source of non-monotonicity is therefore in the
composition × time interaction inside the blind+specific Materns**,
not in the additive RBF-on-time. Specifically:

- The blind Matern is ARD over (composition + time) — it's a single
  `Matern_5/2(d)` where `d = √(Σ_dim Δ_dim² / ℓ_dim²)`. The
  composition contributions at each test time `t*` weight different
  training points differently (different feature distances reweighted
  by the Matern's ARD lengthscales).
- The joint Hamming Matern adds an `α·𝟙[c_i ≠ c_j]` term inside the
  same Matern radial basis, so it has the same composition × time
  coupling.
- For feature-extrapolated test compositions, these
  composition-weighted-residual contributions can oscillate as t*
  sweeps, even though no individual training point's kernel is
  non-monotonic in t*.

## Implication for follow-up fixes

This ablation rules out the cheapest fix (drop or swap the
time-only kernel). The remaining options to reduce the 22% floor
are architectural:

1. **Decompose the blind/specific Materns** so time is not in the
   same ARD as composition. Replace
   `Matern((composition, time))` with
   `Matern(composition) × MonotonicKernel(time)`. Loses ARD
   flexibility on time; needs a refit-and-validate cycle.
2. **Soft monotonicity penalty in the MLE** on `∂μ/∂t` at sampled
   `(z, t)` points. Already proposed.
3. **Hierarchical mean function** (Verhulst / Gompertz / Bazant)
   pulls bulk of the time trajectory into a deterministic
   monotonic mean; GP residuals are smaller magnitude and unlikely
   to flip sign.
4. **UI-level isotonic projection** (display-only, no metric
   impact).

## Why is `RBF(t)` no longer needed?

In the V2 era (legacy_continuous_ard source kernel), the source
dim was treated as a 1D continuous coordinate. The blind+specific
Materns had no direct mechanism to model time-only structure
without entangling it with composition; the additive `RBF(t)`
provided a clean "smooth time correction" channel.

The joint_hamming_matern adds the categorical penalty
`α·𝟙[c≠c']` *inside* the radial basis, decoupling categorical
structure from continuous distance. This frees the
composition-direction lengthscales to fit composition-only
patterns, while the time direction in the same Matern can fit
time-only patterns. The additive `RBF(t)` becomes redundant.

Could we drop `RBF(t)` from the production kernel? Yes, with
zero metric cost on v5. We'd need to verify on pre-v5 data
(which still showed RBF(t) helping by +20 psi in V2 benchmark)
to make sure dropping it is safe across data versions —
but it's an attractive parsimony win.

**Open follow-up**: re-fit pre-v5 with the joint_hamming_matern
+ no `RBF(t)`, verify bLOO is preserved. If yes, simplify the
production kernel by removing the additive time branch.
