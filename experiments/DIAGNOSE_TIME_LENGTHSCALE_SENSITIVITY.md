# Time-lengthscale post-fit sensitivity diagnostic

## Question

Does the kernel-of-time decomposition (the time-direction lengthscales
in the blind Matern, the joint Hamming Matern, and the additive RBF on
time) cause the v5 non-monotonicity? If yes, a tighter prior /
larger lower bound on those lengthscales during fit would mitigate
the issue.

## Method

Take the production v5 + joint_hamming_matern model, then artificially
**multiply all three time-direction lengthscales** (blind Matern's
time-dim ARD entry, JointHammingMaternKernel's time-dim ARD entry,
and additive RBF(t)'s lengthscale) by `m ∈ {0.1, 0.5, 1, 2, 5, 10, 100}`.
Re-evaluate monotonicity on the same dense time grid (no refit).

This is a *post-fit* perturbation, not a refit. Caveat: at higher
multipliers the (composition, time) covariance regions the data was
fit to no longer match the kernel, but the test isolates whether the
kernel-of-time *shape* is responsible for the non-monotonicity.

Run: `python experiments/diagnose_time_lengthscale_sensitivity.py`

## Results

| time-ell × multiplier | % drop | max single-step drop |
|---|---|---|
| 0.1× | 94.4% | 1253.4 psi |
| 0.5× | 27.1% | 56.9 psi |
| **1.0× (baseline)** | **22.2%** | **29.2 psi** |
| 2.0× | 24.3% | 24.5 psi |
| 5.0× | 23.6% | 24.1 psi |
| 10× | 22.9% | 24.1 psi |
| 100× | 22.9% | 24.1 psi |

## Findings

### 1. Short time lengthscales catastrophically destroy monotonicity

At multiplier 0.1× (time-direction lengthscales ÷ 10), drop
fraction explodes to **94.4 %** with max single-step drops of
**1253 psi**. The kernel-of-time becomes spiky in t and the
weighted-residual oscillation amplifies dramatically.

This is consistent with the gate-tau ablation finding that
`tau=0.02` (sharper gate, less dampening) gave 42 % drops:
**short time-direction "scales" — whether through gate or kernel —
are bad for monotonicity**.

### 2. Long time lengthscales DO NOT eliminate non-monotonicity

At multiplier 100× (time-direction lengthscales effectively
infinite — kernel becomes constant in t), drop fraction stays
at **22.9 %**. Max single-step drop **only** reduces from 29.2 to
24.1 psi (a 17 % improvement).

If the kernel-of-time were the dominant cause, multiplier 100×
should drive drops near 0 % (since the kernel becomes flat in t,
making μ(t*) ≈ h(t*) · const, which is monotonic by h's shape).

That this DOES NOT happen is the **smoking gun**: the residual
~22 % non-monotonicity floor is **not from the kernel-of-time
decomposition**.

### 3. Where DOES the residual non-monotonicity come from?

Candidate causes (to investigate further if pursuing the floor):

* **Composition-direction kernel × gate interaction.** Even with
  time-flat kernels, the gate `h(t*)` factor multiplied with
  `h(t_i)` per training row produces per-row weights that vary
  slightly (training data has `t_i ∈ {1, 3, 7, 28} d`, giving
  `h(t_i) ∈ {0.95, 0.999, 1.0, 1.0}` at τ=0.10). The mixed-sign
  `α` vector multiplied with these slightly-varying h(t_i) weights
  could produce small t*-direction signal that, combined with
  feature-extrapolation, looks non-monotonic.

* **Noise / likelihood structure.** The `GatedGaussianLikelihood`
  also gates the noise by `h(t)²`. K_lik diagonals at smaller h(t_i)
  values get less noise dampening, which shifts α weights.

* **Numerical artifacts.** At extreme multipliers the kernel is
  numerically nearly constant; small-magnitude float drift could
  be reported as "drops > 1 psi" even when the true posterior is
  flat-in-t.

A more careful experiment (refit at locked time lengthscales,
post-hoc check of α vector signs at extrapolated test points)
would isolate which.

## Implication for mitigation strategies

This diagnostic **rules out** the cheapest model-level mitigation
(tighten the time lengthscale prior). Even an effectively-infinite
time lengthscale leaves us at 22.9 % non-monotonic curves.

The remaining viable mitigations, in order of effort:

1. **UI-level isotonic projection** (display monotonization).
   Zero metric impact, fixes the user-visible issue completely.
   No model change needed. **Recommended as the immediate fix.**

2. **Hierarchical Verhulst / Gompertz mean function.** Replace
   the GP's role: bulk of the strength curve comes from a
   parameterized monotonic mean function whose parameters are
   themselves modeled by a small GP over composition. The GP
   residuals are then small-magnitude corrections that are
   unlikely to flip sign.

3. **Soft monotonicity penalty in the MLE objective.** Sample
   `(z, t)` pairs at fit time, penalize negative `∂μ/∂t`. May or
   may not be effective; the diagnostic above suggests the
   non-monotonicity is built into the structure of α at the
   fitted hyperparameters, so a kernel-level penalty might help
   only marginally.

4. **Replace kernel structure entirely with monotonic-by-
   construction time kernel** (integrated GP, Riihimäki-Vehtari
   virtual derivative observations). Substantial research effort.

Tracked as research follow-ups; **not blocking the production PR**
which already improves monotonicity from V2's 27.8 % to 22.2 %.

## Summary

| ablation | finding |
|---|---|
| `ABLATION_GATE_TAU.md` | τ=0.05 → 0.10 reduces drops 33 % → 22 %, free LOO improvement. |
| `ABLATION_TIME_KERNEL.md` | RBF/Matern/Linear/no-time choice for additive time-only kernel doesn't matter. |
| `ABLATION_V2_BASELINE.md` | V2 production has 27.8 % drops; v5+joint reduces to 22.2 %. |
| **this diagnostic** | **Time lengthscale magnitude doesn't matter at long range; floor is from elsewhere.** |

The 22 % non-monotonicity floor is **architectural** (multi-Matern +
GatedKernel + feature-extrapolation), shared with V2, and
**not** addressable by tweaking the time-direction kernel structure.
Reducing it requires either an output-side intervention (UI clamp)
or a structural prior change (mean function or virtual derivative
observations). Both are research follow-ups.
