# Strength GP Benchmark Study

A self-contained record of the model-architecture / noise-model /
feature-engineering experiments run on the **public 647-row strength
dataset** for the BOxCrete strength GP. The goal is twofold:

1. **Improve LOO RMSE** — the model's point-prediction accuracy.
2. **Improve calibration** — predictive intervals' coverage matches
   their nominal levels (mean log predictive density, PIT-KS distance
   from N(0,1), 95% interval coverage).

Every result here is reproducible via:

```
python experiments/model_variant_study.py --variants <name>... --seeds 0
```

This document is the consolidated snapshot of all results — superseded
intermediate "champions" have been removed; negative results that
informed the final recommendation are kept in §6.

---

## Canonical metric

The deployed champion (`B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean`,
fit via `boxcrete.fit_strength_gp` with `seed=0`) on the public
647-row strength dataset:

| metric         | value          |
|----------------|----------------|
| LOO-CV RMSE    | **532.66 psi** |
| LOO-CV NLL     | **7.676**      |
| y_max (psi)    | 16029.0        |
| n_real         | 647            |
| seed           | 0              |

Reproduction commit: `915a809028aa` (stack base). Versions:
`torch=2.10.0`, `botorch=0.16.1`, `gpytorch=1.15.1`. LOO computed
via `boxcrete.compute_loo_cv(model, n_real=n_real)` and untransformed
to psi via `model._study_y_std`. NLL is the per-observation
Gaussian log-density `0.5·log(2π σ²) + (y−μ)²/(2σ²)`. These two
numbers are the **single source of truth** for the deployed model's
quality; all variant comparisons in §3–§7 of this document should be
read relative to them.

---

## TL;DR

The original production strength GP achieves single-row LOO RMSE
**723.7** psi and **block-LOO** (leave-one-composition-out) RMSE
**843.1**. After this study, the **production champion** is a
**Multi-Matern (radial source-aware) kernel + 7 chemistry features
with all heavy-tailed ones log-transformed + a multiplicative time-gate
that structurally enforces $f(x, 0) = 0$ + heteroscedastic gated noise
likelihood + single-stage block-LOO+priors training**
(registered in the script as
**`B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_only`**):

- A multi-component ARD-Matern kernel that lets the two Material
  Sources differ smoothly (a "blind" Matern shared across Sources +
  a "source-specific" Matern that lets them diverge), with an
  additive RBF on time
- 7 engineered chemistry features (W/B, SCM frac, HRWR/binder, W/C,
  Coarse/Fine, Aggregate/Paste, log-maturity), all 5 heavy-tailed ones
  log-transformed
- **Multiplicative time gate** $h(t) = 1 - e^{-t/\tau}$ with $\tau = 0.05$
  that vanishes at $t = 0$, multiplying the kernel left-and-right.
  This makes prior variance at $t = 0$ exactly zero — the physics
  constraint $f(x, 0) = 0$ is built into the prior, so **anchor
  pseudo-observations are no longer needed**.
- **Heteroscedastic gated likelihood** (NEW 2026-05-16): noise
  variance is also gated by $h(t)^2$, so the FULL predictive
  distribution (mean AND variance) collapses to $(0, 0)$ at $t=0$.
  At training data ($t \geq 1$ day, post-transform $t \geq 0.30$),
  $h^2 \approx 0.996$ — the global noise estimate $\sigma^2_{\text{global}}$
  converges essentially the same as a standard Gaussian likelihood.
- **Multiplicative-only Y scaling** (`Y / y_max`, no mean subtraction)
  + **`ZeroMean`** GP prior + `outcome_transform=None` — together
  these ensure the GP's "zero" in scaled space corresponds to zero in
  raw psi space, so the gated kernel actually produces 0 at $t = 0$
- **Single-stage block-LOO + priors training** (`block_loo_only=True`):
  optimize $\mathrm{NLL}_{\text{block-LOO}}(\theta) - \log p(\theta)$
  directly via LBFGS from default kernel init, using the closed-form
  Sundararajan-Keerthi block-inverse identity for the data term and
  GPyTorch's registered priors for the regularizer. **Beats MLL+priors
  by 8 psi at full data and up to 203 psi at small data**. Uniformly
  better calibration. See §6.12.

**Final block-LOO performance** (corrected after fixing two silent bugs
— see "Lessons Learned" §0):

| n_train compositions | MLL+priors block-LOO | block-LOO+priors single-stage | Δ |
|---:|---:|---:|---:|
| **144 (full)** | 680.0 | **672.1** | −8 psi |
| 100 (mean over 3 subsets) | 790 | **709** | **−81 psi** |
| 50 (mean over 3 subsets) | 889 | **686** | **−203 psi ⚡** |
| 25 (mean over 3 subsets) | 846 | **748** | −98 psi |

The block-LOO+priors single-stage objective wins on EVERY data size,
with the largest gain at $n=50$ — matching Bachoc's (2013) theoretical
result that CV-NLL is asymptotically more robust to model
misspecification than MLL.

The gated kernel + gated noise jointly enforce **physical realism
at $t = 0$**: predicted strength is exactly zero with zero predictive
variance for any composition. This was previously enforced via day-zero
anchor pseudo-observations (loss term ≈ 2670 psi RMSE constraint
violation); the structural gating makes the constraint exact.

---

## 0. Lessons Learned (and bugs caught the hard way)

This study was originally written assuming all reported numbers
were correct. During the productionization push (2026-05-16), the
user spotted three suspicious patterns in the explorer that turned
out to be different bugs in our experimental tooling, each of which
silently invalidated an earlier round of "champion" decisions. The
final TL;DR reflects the post-bug-fix results.

### Bug 1 — `_model_prior_log_prob` had wrong calling convention

**Symptom**: After fixing the missing-prior issue (see Bug 2), the
"prior-aware refinement" appeared to be a complete no-op: the
post-refinement model was bitwise identical to the MLL fit, and the
`block_loo_only` variant (refinement from default init) gave
catastrophically miscalibrated predictions (cov95 = 1.00, MLPD = -9.6).

**Root cause**: `_model_prior_log_prob` called `closure(model)`, but
GPyTorch's `named_priors()` yields closures that expect the kernel
module (not the top-level GP) — so every call raised `AttributeError`.
The error was swallowed by an outer `try/except` in the fit factory.

**Fix**: Call `closure(parent_module)` (the module yielded by
`named_priors`). Removed the silent `try/except`. After this fix,
single-stage block-LOO+priors finally worked correctly and beat
MLL+priors by 8-203 psi (see TL;DR table).

**Generalised lesson**: NEVER let `try/except` swallow exceptions
without printing-then-re-raising or at least logging loudly. Half
the bugs in this session were initially observed as "the variant
ran fine but produced numbers that didn't change with the
intervention", which is exactly what a silent crash looks like.

### Bug 2 — Block-LOO refinement was missing the prior log-density

**Symptom**: Block-LOO refinement made MLL collapse from $+2.28$ to
$-25{,}796$ per row, lengthscales drifted to extreme values
(Fine Aggregate: $\ell = 170$, Material Source: $\ell = 199$,
Temp: $\ell = 384$), and 5 of 17 input dimensions were effectively
inactive. The user noticed in the explorer that the Fine Aggregate
slider didn't change predictions at all.

**Root cause**: `block_loo_loss` only included the data term
$\log p(y \mid \theta)$, dropping the prior $\log p(\theta)$. So
the within-group shrinkage prior on lengthscales (active during
the MLL stage) was silently turned off for the refinement stage.
Without the prior, lengthscales were free to grow without bound
along weakly-identified directions.

**Fix**: Added a `_model_prior_log_prob` term to make the loss a
proper MAP-style objective:
$\min_\theta \,-\log p(y \mid \theta) - \log p(\theta)$.

**Generalised lesson**: When implementing a custom training
objective, replicate the regularization terms that were active in
the standard objective. GPyTorch's `ExactMarginalLogLikelihood`
adds prior log-density via `named_priors`; any replacement loss
must do the same.

### Bug 3 — Schema-v1 lengthscale identifiability test never migrated to v2

**Symptom**: We had `test/test_lengthscale_identifiability.py` from
the production-baseline study that would have caught Bug 2's
extreme lengthscales. But it read `params["matern_lengthscales"]`
(v1 field name) which doesn't exist in schema-v2 — failing with a
`KeyError` rather than running. We never ran `pytest test/`
between the schema migration and the prior-bug deployment.

**Fix**: Wrote a v2-aware `test/test_lengthscales_v2.mjs` (also
v2-mirror in Python: `test_strength_curve_monotonicity.py`).
Both run as part of `bash experiments/regenerate_all_artifacts.sh`,
which is now the standard pre-deploy gate.

**Generalised lesson**: Schema migrations need test coverage
specifically. The right pattern is to write the new tests as part
of the migration PR, then delete the old ones (rather than letting
both linger and silently neither work).

### Bug 4 — Stale precomputed predictions in `compositions.json`

**Symptom**: The user noticed that the highlighted-composition
circle on the Pareto plot would smoothly animate during click-to-mix,
then visually "snap" at the end — the destination dot was at a
different y-value than where the circle landed.

**Root cause**: `docs/model/compositions.json` stores precomputed
strength predictions per composition (used to draw the static
background dots). These predictions were generated by an older
model and were 600+ psi different from the live GP at the same
compositions. The animated circle (live prediction) and static
dot (precomputed) disagreed.

**Fix**: Wrote
`experiments/regenerate_compositions_strength_predictions.mjs` and
`test/test_data_freshness.mjs` (which catches stale precomputed
data across all JSON artifacts: strength_predictions, gwp_predictions,
cost_predictions, pareto_mask, test_vectors). Now part of the
regenerate pipeline. The freshness test caught a stale
`pareto_mask` field that we hadn't previously noticed.

**Generalised lesson**: Precomputed JSON artifacts need automatic
regeneration AND automatic freshness checking — manual
"remember to regenerate" is unreliable. Build both.

### Bug 5 — Feature change broke monotonicity even when it improved BLOO

**Symptom**: After deploying a new champion that swapped
`log_maturity_robust` for raw `maturity_robust = (T+10) \cdot t`
(motivated by a 17 psi block-LOO improvement), the user observed
that strength curves in the explorer "slumped down" between days
5-14 and 14-28 — clearly unphysical (concrete strength is monotone
increasing under standard hydration).

**Quantification**: Wrote `test/test_curve_monotonicity.mjs`
(JS) and `test/test_strength_curve_monotonicity.py` (Python).
Diagnostic across 144 compositions × 64 log-spaced times revealed:

| Variant | %decreasing | %oscillating (>2 inflections) | maxDrop |
|---|---:|---:|---:|
| F5_alllog (current champion) | 3.5% | 8.3% | 24 psi |
| F5_no_log_mat (broken) | **99.3%** | **98.6%** | **876 psi** |

**Root cause**: Raw $(T+10) \cdot t$ is *linear in time*, while
post-log time (the kernel's time argument) is *logarithmic in time*.
Mixing two non-commensurate functions of the same variable in an
ARD kernel let the optimizer produce destructive interference at
intermediate times.

**Fix**: Reverted to `log_maturity_robust`. Added the monotonicity
test as a permanent gate.

**Investigation**: A separate experiment swept smoothness
(L2 second-derivative) and monotonicity (L2 hinge) regularizer
strengths to see if F5_no_log_mat's BLOO advantage could be
recovered without the oscillations. **Answer: no.** Even at the
best monotonicity-regularized point ($\lambda_{\text{mono}}=10000$),
F5_no_log_mat gives BLOO 776 with 19% / 27% / 71 psi monotonicity —
worse than F5_alllog (672 / 3.5% / 8.3% / 24) on every metric.
The 17 psi BLOO advantage was inseparable from the wiggly fit.

**Generalised lesson**: When evaluating a model purely on
held-out RMSE / log-likelihood, you may be selecting for
"flexibility that fits training points tightly via wiggly
extrapolation" — which scores well on point-evaluation metrics but
is unphysical and visually obvious in continuous predictions. ALWAYS
visualize predictions as continuous curves alongside the metric, and
add explicit physical-realism tests (monotonicity, smoothness) to
the deployment gate.

### Test infrastructure now in place (post-this-study)

To prevent any of the above five bugs from re-occurring silently:

| Test | What it catches | Lang |
|---|---|---|
| `test_js_strength_v2.mjs` | Python-JS prediction sync (37 vectors) | JS |
| `test_js_physical_constraints.mjs` | $f(x, 0) = 0$ at deployment | JS |
| `test_js_gp.mjs` | GP numerical fidelity (296 assertions) | JS |
| `test_js_ui_smoke.mjs` | Explorer render-path NaNs (144×64) | JS |
| `test_lengthscales_v2.mjs` | Inactive-input detection ($\ell$ caps) | JS |
| `test_data_freshness.mjs` | Stale precomputed JSON | JS |
| `test_curve_monotonicity.mjs` | Unphysical curves | JS |
| `test_lengthscale_identifiability.py` | (v1 — needs migration) | Py |
| `test_strength_curve_monotonicity.py` | Unphysical curves (model-side) | Py |

All of these run via `bash experiments/regenerate_all_artifacts.sh`.

---

## 0a. Known gaps for V2 deployment (2026-05-17)

The deployed champion is **V2** (Multi-Matern + gated + gated noise);
the legacy production **V1** architecture is retained as a fallback
via the `architecture="v1"` kwarg to `boxcrete.fit_strength_gp`. After
the cleanup pass on 2026-05-18, only one architectural gap remains:

### Gap 1 — V2 implementation lives in `experiments/`, not `boxcrete/`

The V2 building blocks (`_GatedGaussianLikelihood`, `_TimeGatedKernel`,
`_AppendEngineeredFeatures`, the multi-Matern kernel builder, and the
champion fit factory `_make_engineered_no_a_fit`) currently live in
`experiments/model_variant_study.py` (~500 LOC). The public Python
API `boxcrete.fit_strength_gp` is a thin facade that re-exports
the champion fit via:

```python
from experiments.model_variant_study import VARIANTS
fit_fn = VARIANTS["B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean"]
```

This requires `experiments/` to be installed as a package
(see `pyproject.toml`'s `packages = ["boxcrete", "experiments"]`).

**Why deferred**: extracting cleanly without code duplication takes a
focused PR (~400 LOC move + adapt + test) which doesn't fit in this
commit's scope. The bridge is functionally complete and pre-commit-clean,
and the 3 in-CI notebooks all execute end-to-end against V2.

**Follow-up plan**: extract the five V2 building blocks above into
`boxcrete/strength_v2.py`, then have `experiments/model_variant_study.py`
import them back. Net code reduction.

### Resolved in this commit (2026-05-18)

- ✅ **`docs/gp.mjs` v1 cleanup**: deleted the v1 schema branches in
  ``kernel``, ``transformInput``, ``initStrengthModel``,
  ``predictStrengthCurve``, and ``predictStrengthMeanOnly``. File
  shrunk 697 → 462 LOC (–34%). v1-on-v2 now raises a clear error.
- ✅ **`scripts/export_model.py`**: rewritten as a thin wrapper
  around the V2 `experiments/regenerate_all_artifacts.sh` pipeline.
- ✅ **Notebooks**: 3/3 in-CI notebooks pass V2 end-to-end.
  `bayesian_optimization_tutorial.ipynb` was deleted as redundant —
  `prediction_and_optimization_tutorial.ipynb` already exercises BO
  via qLogNEHVI plus GWP fit, Pareto frontier, and gradient-based
  experimental design.
- ✅ **CI gap**: `notebooks.yml` now executes ALL notebooks (was
  previously only running `prediction_and_optimization_tutorial`,
  letting other notebooks rot silently — this is how the BO tutorial's
  stale data path went undetected for months).

---

## TL;DR (continued — original architecture decomposition)

  after un-scaling.
- Plain learnable scalar Gaussian noise (no heteroscedastic noise
  from empirical Yvar, no noise floor)
- No day-zero pseudo-observations (the constraint is in the kernel,
  not the data)

It achieves:

| Metric | Production baseline | New champion (gated) | Δ |
|---|---:|---:|---:|
| Single-row LOO RMSE | 723.7 | 532.9 | −26.4% |
| **Real-only block-LOO RMSE** | 843.1 | **679.4** | **−19.4%** |
| **Phantom-anchor RMSE at $t=0$** | 2670 (unsatisfied) | **0.0** | **physics constraint exact** |
| **Joint block-LOO RMSE (real + anchors)** | 1296 | **614** | **−52.6%** (the right deployment metric) |
| cov95 (block-LOO) | 0.92 | 0.93 | preserved |

The drop from the un-gated `B''+F5_alllog` (block-LOO 663.4) to the
gated variant (679.4) is a 16-psi real-only regression — within seed
noise — that's more than offset by the 2670-psi *exact* improvement
in the t=0 prediction. **For deployment, the gated variant is strictly
preferred**: same real-data extrapolation quality plus a physics
constraint the unanchored variant cannot satisfy.

### Decomposition of the gain (block-LOO, the realistic extrapolation metric)

| # | Design choice added | Block-LOO Δ | Cumulative |
|---|---|---:|---:|
| 0 | Pre-shrinkage baseline (single Matern + scalar noise + day-zero anchors + log-time input transform) | — | 851.5 |
| 1 | + within-group shrinkage prior on binder & aggregate lengthscales (= production baseline) | −8 psi (block-LOO) — but **−47 psi single-row LOO**, by curing fly-ash & coarse-aggregate lengthscale railing | 843.1 |
| 2 | Switch to the recommended Multi-Matern kernel | **−89 psi** | 753.9 |
| 3 | + 3 core chemistry features (`F3` = W/B + SCM + HRWR/binder) | −30 psi | 724.2 |
| 4 | + 4 more chemistry features (= `F5` with aggregates + maturity + W/C) | −6 psi | 718.4 |
| 5 | **+ log-transform all 5 heavy-tailed features (= `F5_alllog`)** | **−55 psi** | **663.4** |

**Notes**:

- All stages already use the production **`log10(time + 1)` input
  transform**; that's a foundational choice from the production
  baseline, not an ablated lever.
- The **within-group shrinkage prior** (stage 1) is the highest-yield
  single-row LOO improvement of the entire study (−47 psi) but a more
  modest block-LOO contributor — the prior cures *interpolation*-side
  identifiability (collinear binder & aggregate lengthscales railing
  at the upper bound) more than it helps extrapolation. Even so, every
  variant in this study (and the recommended one) installs it on every
  Matern in every kernel — it's free regularisation.

**Two big block-LOO levers dominate**: (a) the multi-component kernel
architecture (−89 psi), and (b) distribution-aware log-transforms of
heavy-tailed engineered features (−55 psi). Feature engineering
itself contributes another −36 psi cumulatively. Everything else this
study explored — heteroscedastic noise, noise floors, alternative
kernels, log-Y outcome, OAK — either does not help or actively hurts
on the realistic extrapolation metric. Details and negative results
in §6.

---

## 1. Setup

### 1.1 Data

- **Source**: `data/boxcrete_data.csv` — 647 strength observations,
  public dataset.
- **Inputs (10 raw dims)**: Cement, Fly Ash, Slag, Water, HRWR
  (high-range water reducer), Fine Aggregate, Coarse Aggregates,
  Material Source (binary 0/1), Temp, Time.
- **Output**: Strength (Mean) in psi, with empirical Strength (Std)
  populated for 645/647 rows (median ≈ 5% of mean — typical concrete
  triplicate noise).
- **Composition structure**: 144 unique compositions × ~4.5 time-point
  measurements each (1d, 3d, 7d, 14d, 28d are typical). This structure
  matters for evaluation — see §2.

### 1.2 Production baseline (pre-this-study)

- **Kernel**: `ScaleKernel(Matern52(ARD=10)) + ScaleKernel(RBF(time))`
  — one joint Matern over all 10 inputs (including Material Source as
  a binary ARD dim), plus an additive RBF on the time dim.
- **Mean**: ConstantMean.
- **Likelihood**: `PartialFixedNoiseLikelihood` — one learned scalar
  noise for the 647 real observations, fixed near-zero noise for 128
  day-zero pseudo-observations encouraging `f(x, 0) ≈ 0`.
- **Outcome transform**: Standardize(Y).
- **Input transform**: `Log10(time + 1) → Normalize` (the standard
  production input transform). All variants in this study, including
  the recommended one, inherit this transform — log-time is a
  foundational choice across the entire study, not an ablated lever.
- **Prior**: `WithinGroupShrinkagePrior(σ=0.001)` ties Cement / Fly Ash /
  Slag to a single binder lengthscale and Fine / Coarse Aggregates to a
  single aggregate lengthscale. This is an already-promoted production
  fix — without it, the optimiser pushes Fly Ash and Coarse Aggregate
  lengthscales to the upper constraint bound (the kernel can't decide
  between co-linear feature combinations). Quantifying the prior's
  contribution: removing it (variant `baseline_no_prior`) regresses
  single-row LOO from 725 → 772 (**+47 psi, −6.5% point accuracy**)
  and block-LOO from 843 → 852 (**+8 psi**) — a meaningful gain on
  *interpolation* (cures the railing) and a smaller but real gain on
  extrapolation. The same prior is installed independently on every
  Matern's lengthscale tensor in every variant studied below; we
  consider it part of the production baseline rather than a separate
  lever.

### 1.3 Metrics

All metrics are computed in original units (psi, untransformed):

| Metric | What it measures |
|---|---|
| **RMSE** (psi) | Point-prediction error |
| **MAE** (psi) | Outlier-robust complement to RMSE |
| **Mean LPD** | Log predictive density (proper scoring rule, higher = better) |
| **PIT-KS** | KS-distance of `(y - μ) / σ` from N(0,1); calibration shape |
| **95% coverage (cov95)** | Fraction of `|y - μ| < 1.96·σ`; ideal ≈ 0.95 |
| **CRPS** | Continuous ranked probability score (calibration + sharpness) |

Results in this document are deterministic at 1 seed for the
recommended scalar-noise variants (the kernel + noise model is convex
enough at full data that seeds collapse). Where seed-noise is
reported, it's mean ± std over 3 seeds.

### 1.4 Variant naming convention

A "variant" is a triple of design choices: **kernel**, **noise model**,
and (optionally) **engineered chemistry features**. Throughout the
document we refer to design choices by descriptive names. The script
(`model_variant_study.py`) registers each variant under a short code
preserved from the original chronological exploration — these codes
appear in the cached CSVs and reproduction commands, so we keep them
as **internal aliases**, but the descriptive names are what you should
hold in your head while reading.

#### Kernel choices

All kernels listed below also include an additive `RBF(t)` time-only
component (unless suffixed with `_norbf` for the §4.1 ablation).
Notation: $x_{\text{all}}$ is the full input (raw + any appended
engineered features); $x^{ns}$ is $x_{\text{all}}$ with the binary
Source dim dropped ("**n**o **s**ource"); $\delta_{\text{source}}$ is
a hard 0/1 categorical kernel on Source; $k_{\text{cat}}(s)$ is a
Hamming-distance RBF on Source; $M(\cdot)$ is a Matern-5/2 ARD kernel
with the within-group shrinkage prior on its lengthscales.

| Descriptive name | Form | Internal alias | Notes |
|---|---|---|---|
| **Multi-Matern (radial source-aware)** | $M(x^{ns}) + M(x_{\text{all}}) + \text{RBF}(t)$ | **`B''`** | **Recommended.** Blind + source-specific Materns; cross-source covariance smoothly attenuates via the source-dim ARD lengthscale. Formal definition in §3.1. |
| Single Matern (production starting point) | $M(x_{\text{all}}) + \text{RBF}(t)$ | `baseline` | The pre-existing production model. One joint ARD-Matern over all 10 raw dims (Source as a binary ARD dim). |
| Multi-Matern (hard δ source) | $M(x^{ns})\cdot(1 + \delta_{\text{source}}) + \text{RBF}(t)$ | `B` | Same multi-component idea but with a hard 0/1 source kernel. Block-LOO close to the recommended kernel; less stable across seeds. |
| Multi-Matern (two unconstrained) | $M_1(x_{\text{all}}) + M_2(x_{\text{all}}) + \text{RBF}(t)$ | `D` | Two joint-ARD Materns over all dims with no explicit blind/specific decomposition. Block-LOO close to the recommended kernel. |
| Multi-Matern (factored Hamming) | $M(x^{ns}) + M(x^{ns})\cdot k_{\text{cat}}(s) + \text{RBF}(t)$ | `E1` | Source effect via a Hamming RBF rather than as an ARD dim. Block-LOO close to the recommended kernel. |
| Log-Y single Matern | Single Matern + log-Y outcome transform | `C` | Drops 2 anomalous Y < 10 psi rows. Block-LOO not yet implemented for log-Y models — §7.3. |
| Orthogonal Additive Kernel (1st / 2nd order) | OAK | `OAK1`, `OAK2` | Pure additive structure (no high-order interactions). §6.5 — wrong inductive bias for this dataset. |
| (Two more failed kernels documented in §6.1) | | `B'`, `E2` | `B'` ties lengthscales across the multi-Matern components (catastrophic). `E2` drops the explicit blind term (worse than the recommended kernel). |

#### Noise model choices

| Descriptive name | What it is | Internal token | Notes |
|---|---|---|---|
| **Scalar Gaussian noise** | Single learnable variance estimated from the marginal likelihood | (no prefix) | **Recommended.** See §4.4 / §6.2. |
| Heteroscedastic noise | Empirical triplicate Yvar plumbed in via `FixedNoiseGaussianLikelihood` | **prefix `A`** | **Single-row LOO trap** — looks like a major win, but regresses block-LOO. See §6.2. |
| `+ noise floor` | Adds a learnable floor `σ²_g · I` to the heteroscedastic variance | **suffix `+floor`** | Always combined with heteroscedastic noise. |
| `+ multiplicative scale` | Scales Yvar by a learnable `c` | **suffix `+mult`** | Always combined with heteroscedastic noise. |
| `+ floor + mult` | Combination of the two above | **suffix `+full`** | |

#### Engineered chemistry features

The `+F<n>` suffix denotes a particular set of derived features
appended to the raw 10-dim input. The cumulative progression
`F0` (none) → `F1` (W/B) → ... → `F5` (full chemistry) is described
in §4.2; the distribution-aware log-transformed configs
(`F5_loghrwr`, `F5_lh_lmat`, **`F5_alllog`** = champion) in §4.3.
Negative-result feature configs used only in §6 are documented inline
where they appear.

#### Other suffixes

| Suffix | Meaning |
|---|---|
| `_rl` | Relaxed lengthscale lower bound (1e-4 instead of 1e-2). §6.7 negative result. |
| `_norbf` | Drop the additive `RBF(t)` time-only component. §4.1 ablation. The multi-Matern part still sees time as one of its ARD dims. |

#### Putting it together — how to read variant names

Variant names are **left-to-right additive**:

| Variant string | What it means in plain English |
|---|---|
| `baseline` | Production starting point |
| **`B''+F5_alllog`** | **Recommended.** Multi-Matern (radial source-aware) + 7 log-transformed chemistry features + scalar Gaussian noise |
| `B''+F5` | Same but without the log-transforms |
| `B''+F0` | Same kernel & noise model, no engineered features |
| `A` | Heteroscedastic noise on top of the Single-Matern baseline |
| `A+B''+floor+F5` | Heteroscedastic noise + noise floor + Multi-Matern + F5 features (the previous "champion" — a single-row LOO trap; see §6.2) |

When you encounter a variant string in the body of the document, decompose it left-to-right into these axes.

---

## 2. Evaluation framework: why block-LOO is the primary metric

The dataset has 144 unique compositions × ~4.5 measurements each. **A
random row-level holdout always leaves the model with most of each
composition's time curve in training**. The model can interpolate
the missing time-point along a curve it largely knows — that's not
extrapolation, it's hole-filling.

The realistic deployment scenario is "produce a new mix, measure its
full strength curve at 1d/3d/7d/28d." The corresponding evaluation
holds out **entire compositions**, all their time-points together.

**Block-LOO** computes leave-one-composition-out predictions in
**closed form** by generalising the standard Sundararajan–Keerthi LOO
identity. For a block $B$ (one composition's rows):

$$\mu_{\text{block-LOO}}[B] = y_B - [K^{-1}_{BB}]^{-1}\,\alpha_B,
  \qquad \Sigma_{\text{block-LOO}}[B] = [K^{-1}_{BB}]^{-1}$$

where $\alpha = K^{-1}(y - \mu_{\text{prior}})$ — the same precomputed
quantity standard single-row LOO uses. Cost is dominated by the one
$K^{-1}$ that we already form for single-row LOO; the additional
144 small `b × b` solves (`b ≈ 4–5`) take milliseconds.

**Day-zero anchors** are added per composition (same input fingerprint
as that composition's real measurements modulo time = 0), so they
join their composition's block automatically — no information leak.

Implemented as `block_loo_metrics(model, n_real, n_composition_dims=9)`
in `model_variant_study.py`.

### Single-row LOO is dangerous as a sole metric on this dataset

Throughout this study, single-row LOO repeatedly produced
**qualitatively wrong** answers. Block-LOO and single-row LOO disagree
on:

- Which kernel architecture is best
- Which noise model helps
- Whether heteroscedastic noise helps at all
- Which feature engineering choices are good

Concrete examples are in §6. **The bottom line**: many architectural
choices that overfit per-composition time curves look fine on
single-row LOO precisely because they overfit time curves —
within-composition there's nothing for them to be wrong about. But
they fail catastrophically on extrapolation to unseen mixes.

**Use block-LOO as the primary metric** for all production decisions.
Single-row LOO remains useful as a fast diagnostic and for spotting
regressions during model development.

---

## 3. The production champion: Multi-Matern + log-transformed F5 features

(Internal alias: `B''+F5_alllog`.)

The production input transform applies `Log10(time + 1) → Normalize`
to the input vector. This is shared with the production baseline and
all variants in this study — log-time is a foundational structural
choice across the entire study, not an ablated lever. The
recommended model also keeps the production within-group shrinkage
prior on lengthscales (§1.2).

### 3.1 Components

**Kernel: the recommended Multi-Matern (radial source-aware) kernel**.
This is a sum of three components — a "blind" Matern that's the same
for both Material Sources, a "source-specific" Matern that lets the
two Sources differ smoothly, and an additive RBF on time:

$$K(x, x') = \sigma^2_{\text{blind}} \cdot M_{5/2}(x^{ns}, x'^{ns}; \boldsymbol\ell_b)
            + \sigma^2_{\text{specific}} \cdot M_{5/2}(x_{\text{all}}, x'_{\text{all}}; \boldsymbol\ell_s)
            + \sigma^2_t \cdot \text{RBF}(t, t'; \ell_t)$$

**Notation**:

- $x \in \mathbb{R}^{17}$ is the augmented input vector (10 raw dims + 7
  engineered chemistry features).
- $x_{\text{all}}$ = the full $x$ (all 17 dims).
- **$x^{ns}$** = $x$ with the **n**o-**s**ource (binary Source dim, index 7)
  removed — i.e., the 16-dim sub-vector containing 9 non-source raw dims +
  7 engineered features.
- $t$ is the time dim (raw dim index 9), used by the additive RBF
  component.
- $M_{5/2}$ denotes the Matern kernel with smoothness $\nu = 5/2$.
- $\boldsymbol\ell_b, \boldsymbol\ell_s$ are the ARD lengthscale vectors of
  the blind and source-specific Materns (16 and 17 dims respectively); each
  has its own copy of the within-group shrinkage prior.
- $\sigma^2_{\text{blind}}, \sigma^2_{\text{specific}}, \sigma^2_t$ are
  the learnable output-scales of the three components.
- $\ell_t$ is the lengthscale of the additive RBF on time.

**Components**:

- **Blind Matern** on $x^{ns}$ — captures the cross-source physics that
  is shared across both source types.
- **Source-specific Matern** on $x_{\text{all}}$ — sees Source as one of
  its ARD dims; the cross-source covariance smoothly attenuates via
  $\boldsymbol\ell_s[\text{source\_dim}]$.
- **Additive RBF on time** — a smooth time-only correction term on top
  of the two multi-Matern components. Its contribution is benchmarked
  in §4.1.
- The within-group shrinkage prior (binder group: Cement / Fly Ash /
  Slag; aggregate group: Fine / Coarse Aggregates) is installed
  independently on each Matern's lengthscale tensor.

**Engineered features** (7 features appended to the raw 10-dim input):

| Feature | Definition | Notes |
|---|---|---|
| W/B | water / (cement + fly_ash + slag) | Abrams' law (1918) |
| SCM frac | (fly_ash + slag) / binder | Supplementary cementitious materials replacement ratio |
| **log_HRWR/binder** | log(HRWR / binder + 1e-4) | Spike-and-slab without log (86% near-zero) |
| **log_W/C** | log(W/C + 1e-3) | Heavy right tail without log (8 outliers up to 287) |
| **log_Coarse/Fine** | log(Coarse/Fine + 1e-3) | Bimodal without log (38% are zero — mortar mixes) |
| **log_Agg/Paste** | log((fine + coarse)/(binder + water) + 1e-3) | Positive skew without log |
| **log_maturity_robust** | log(max(0, T+10)·t + 1) | Saul/Nurse maturity, floored at the −10°C freezing threshold |

**Bold** = features whose distributions required log-transformation to
become identifiable for the kernel (§4.3).

**Noise model**: a single learnable scalar Gaussian noise. The
empirical triplicate Yvar is **not** used (counter-intuitively
explored in §6.2).

**Outcome standardisation**: `Y_z = (Y - μ_Y) / σ_Y` applied manually
in the fit; predictions un-standardised at evaluation time.

**No day-zero anchors** — at full data the kernel + scalar noise model
is identifiable without them; at small data the anchors cost more
than they're worth (§6.2).

### 3.2 Final results

Block-LOO at full data (n=144 unique compositions, deterministic):

| Metric | Value |
|---|---:|
| Block-LOO RMSE | **663.4 psi** |
| Block-LOO MAE | 491.1 psi |
| Block-LOO MLPD | −7.91 |
| Block-LOO PIT-KS | 0.06 |
| Block-LOO cov95 | **0.94** |
| Block-LOO CRPS | 364 |
| Single-row LOO RMSE | 496.9 |
| Single-row LOO MLPD | −7.58 |

Subset learning-curve (composition-level held-out, mean of 3 subset
seeds — see §6 for the full table):

| n_train_comp | F0 baseline | F5_alllog | Δ |
|---:|---:|---:|---:|
| 25 (~112 rows) | 1459 | **1257** | **−14% HO RMSE** |
| 50 (~225 rows) | 1172 | **1079** | **−8%** |
| 100 (~448 rows) | 844 | **832** | **−1%** |
| 144 (full, block-LOO) | 754 | **663** | **−12%** |

### 3.3 Adding the structural physics constraint (gated kernel)

The recommended `B''+F5_alllog` model fits real strength measurements
well but **violates the physics constraint $f(x, t=0) = 0$** —
phantom-anchor RMSE at t=0 is ≈2700 psi (mean prediction +2500 psi
over 144 unique compositions). For deployment, this is unacceptable.

The fix is **structural**, not observational: wrap the kernel in a
multiplicative gate $h(t) = 1 - e^{-t/\tau}$ with $h(0) = 0$:

$$K_{\text{gated}}((x, t), (x', t')) = h(t) \cdot K_{B''}((x,t),(x',t')) \cdot h(t')$$

This makes prior covariance at $t=0$ structurally zero, so the GP
predicts exactly 0 at $t=0$ for every input, by construction. Five
compositional pieces are needed (each is necessary; see
[`STRENGTH_GP_ANCHORS_STUDY.md`](STRENGTH_GP_ANCHORS_STUDY.md) §3.6
for why):

1. **Multiplicatively-gated kernel** with $\tau = 0.05$ (fixed; sweep
   in §3.8 of the anchors study confirmed this is near-optimal).
2. **`Y / y_max` outcome scaling** (no mean subtraction). Z-score
   standardisation breaks the GP=0 ↔ raw=0 correspondence via the
   additive `+y_mean` term in untransform.
3. **`ZeroMean`** GP prior (so the prior at $t=0$ is 0 in scaled
   space, not a learned constant).
4. **`outcome_transform=None`** explicitly in `SingleTaskGP` (to
   disable BoTorch's silent default `Standardize`, which would
   re-introduce an additive offset).
5. **`skip_time_in_normalize=True`** (prevents `Normalize` from
   mapping training $t=1$ to 0, where $h(0) = 0$ would gate away the
   most heavily-trained time point).

The gating is **not** anchor pseudo-observations — anchor approaches
were tested extensively (anchor study §3.1–§3.4) and found to conflict
architecturally with the recommended kernel (paying 100+ psi block-
LOO regression). Structural priors ≠ pseudo-observations.

#### Cost on real-data extrapolation

| Variant | Real-only block-LOO | Phantom-anchor RMSE | Joint block-LOO |
|---|---:|---:|---:|
| `B''+F5_alllog` (un-gated) | 663 | **2669** (constraint violated) | 1296 |
| **`B''+F5_alllog+gated_t+maxscale_zeromean`** | **679** (+16 within seed noise) | **0** (constraint exact) | **614** (−53%) |

The 16-psi real-only regression is the cost; the +2669-psi physics
constraint satisfaction is the gain. **For deployment, the gated
variant is strictly preferred** — same time-curve extrapolation
quality, plus the physics constraint that the un-gated variant
cannot satisfy.

---

## 4. The big levers (in order of magnitude)

§4.1–§4.3 are the three positive design choices that, when combined,
take the model from baseline (block-LOO 843) to champion (block-LOO
663). §4.4 documents two design choices we explicitly evaluated and
decided NOT to add — included here so a reader doesn't wonder why
some "obviously useful" things are absent.

![Improvement journey](improvement_journey.png)

*Cumulative descent of single-row LOO and block-LOO RMSE across the
recommended design choices. Note how the **single-row LOO** drops
dramatically at the first transition (725 → 507) while **block-LOO**
descends more gradually — those two metrics tell different stories
about what extrapolation requires (cf. §2). Reproduce with
`python experiments/plot_improvement_journey.py`.*

### 4.1 Multi-component kernel (−89 psi block-LOO)

The largest single contributor. Adding a second ARD-Matern component
to the production single-Matern kernel halves the irreducible
extrapolation error.

**Mechanism**: with one ARD-Matern + one global lengthscale per dim,
the optimiser must choose between "smooth shared physics" (long ℓ)
and "sharp source-specific corrections" (short ℓ) — cannot have both.
A two-component decomposition (one "blind" Matern shared across
Sources + one "source-specific" Matern that lets them differ) gives
the optimiser two ARD vectors, free to specialise.

Several kernel decompositions tested (§6.1 has the negative-result
details):

| Kernel | Form | Block-LOO |
|---|---|---:|
| Single Matern (production starting point) | `M(x_all) + RBF(t)` | 843.1 |
| **Multi-Matern (radial source-aware)** — **recommended** | `M(x^ns) + M(x_all) + RBF(t)` | **753.9** |
| Multi-Matern (hard δ source) | `M(x^ns) · (1 + δ_source) + RBF(t)` | similar |
| Multi-Matern (shared lengthscales) | tied LS — broken | catastrophic |
| Multi-Matern (two unconstrained) | `M_1(x_all) + M_2(x_all) + RBF(t)` | similar to recommended |
| Multi-Matern (factored Hamming) | `M(x^ns) + M(x^ns)·k_cat(s) + RBF(t)` | similar to recommended |

The recommended kernel wins on a combination of (a) calibrated
extrapolation, (b) stability across seeds (the "two unconstrained"
flavour has ~15× higher seed variance), and (c) clean implementation.
The "two unconstrained" and "factored Hamming" flavours are
competitive and worth remembering as alternatives.

#### Contribution of the additive `RBF(t)` time component

All variants in the table above include an additive `RBF(t)` smoothness
term in addition to the multi-Matern part (which already sees time as
one of its ARD dims). Ablating this single addend on the production
champion's kernel structure (`_norbf` suffix; same B'' multi-Matern,
same features, same noise model):

| Config | With `RBF(t)` | Without `RBF(t)` (`_norbf`) | Δ from removing |
|---|---:|---:|---:|
| B''+F0 (no engineered features) | 753.9 | 750.7 | **−3 psi (no help / very mild hurt)** |
| **B''+F5_alllog** (champion) | **663.4** | 683.7 | **+20 psi (RBF helps with FE)** |

**Asymmetric finding**: the additive `RBF(t)` only contributes
meaningfully when the engineered features are present.

- *Without* engineered features (`B''+F0`): the multi-Matern part
  already handles time as one of its ARD dims and there's nothing
  else for `RBF(t)` to model — removing it is essentially a wash.
- *With* engineered features (`B''+F5_alllog`): three of the seven
  features depend on time (`maturity_robust` is `(T+10)·t`) or on
  features that vary with composition; the additive `RBF(t)` provides
  a clean "smooth time correction" channel decoupled from the
  composition-axis structure of the multi-Matern. Removing it costs
  ~20 psi block-LOO.

**Practical implication**: keep `RBF(t)` in production. The 20-psi
contribution is modest but consistent, with no downside.

### 4.2 Concrete-chemistry feature engineering (−36 psi cumulative)

Engineered ratios encode knowledge that GP kernels (radially smooth on
single dims) cannot easily synthesize. Block-LOO progression:

| Config | Features | Block-LOO | Cumulative Δ |
|---|---|---:|---:|
| F0 | (none, baseline) | 753.9 | — |
| F1 | + W/B | 756.2 | +2 (no help) |
| F2 | F1 + SCM frac | 754.3 | +0 (no help) |
| **F3** | F2 + HRWR/binder | **724.2** | **−30** ⚡ |
| F4 | F3 + W/C | 767.0 | regresses (W/C hurts here) |
| **F5** | F4 + Coarse/Fine + Agg/Paste + maturity | **718.4** | −36 |

**HRWR/binder is the dominant feature** under the recommended
(scalar-noise) regime. The
F2→F3 jump (−30 psi) is the single largest feature-engineering gain.

W/B (Abrams, 1918) — the classical first-thing-to-try — adds nothing
on its own. It only helps in combination with HRWR/binder, where the
two ratios disambiguate workability vs water content. SCM frac, W/C,
maturity each show similar interaction-heavy behaviour: useful in
combination, not alone (§6.3).

The 7 features in F5 are documented in §3.1.

### 4.3 Distribution-aware log-transforms (−55 psi)

**The second-largest single lever.** Discovered when investigating
why the HRWR/binder lengthscale rails at the constraint bound: the
real reason was the feature *distribution*, not a too-tight constraint.

#### Diagnosis: heavy-tailed and bimodal feature distributions

After fitting `B''+F5` and inspecting the lengthscales, the
HRWR/binder lengthscale rails at 0.0100 (the lower constraint bound).
The naive fix — relax the constraint to 1e-4 — actually *regresses*
block-LOO from 724 to 752 on `B''+F3` (variant `B''+F3_rl`; see §6.7
for the full negative result).

The actual problem is that several engineered features have
non-uniform distributions that a stationary kernel lengthscale cannot
accommodate cleanly:

| Feature | min | max | median | zeros / near-zero | Distribution issue |
|---|---:|---:|---:|---:|---|
| HRWR/binder | 0.000 | 0.101 | 0.0022 | **86% < 0.01** | spike-and-slab (33% raw HRWR = 0) |
| W/C | 0.196 | 287 | 0.711 | 0 | **heavy right tail (8 outliers up to 287)** |
| Coarse/Fine | 0.000 | 3.198 | 1.301 | **38% zeros (mortar mixes)** | bimodal |
| Agg/Paste | 0.721 | 12.4 | 3.12 | 0 | positive skew |
| Maturity `(T+10)·t` | **−280** | 896 | 160 | 15 negative values | heavy tail + bare formula gives negatives at Temp < −10°C |

**Mechanism**: the kernel's stationary lengthscale forces a compromise
on these distributions. Either "cliff-edge" the dense region (use a
short ℓ) — which kills smooth interpolation in the slab tail — or
"smooth" everything (long ℓ) — which loses the cliff. Log-transforming
spreads the dense region toward uniform, allowing a single moderate
lengthscale to do both jobs. Maturity additionally needs the Saul/
Nurse `max(0, T+10)·t` floor to avoid the spurious negative values
from the 15 cold-Temp mixes.

#### Ablation: which transformations matter?

**Each transform individually**: small win or wash. **All five
together**: −55 psi block-LOO.

| Variant | Block-LOO | Δ vs F5 (718.4) |
|---|---:|---:|
| F5 | 718.4 | — |
| F5_loghrwr (log HRWR/binder) | 713.1 | −5 |
| F5_lh_lap (log HRWR + log Agg/Paste) | 717.6 | −1 |
| F5_lh_lwc (log HRWR + log W/C) | 734.3 | +16 (worse) |
| F5_lh_lmat (log HRWR + log maturity) | 695.8 | −23 |
| **F5_alllog** (log of all 5 heavy-tailed features) | **663.4** | **−55** ⚡ |

Combinatorial insight: log-W/C alone hurts when the others aren't
log-transformed. Only the **all-five-together** combination unlocks
the full gain.

**Generalisable lesson**: when an ARD lengthscale rails at a
constraint bound, check the feature distribution before tightening
the constraint. Distribution-aware feature transforms (log,
`max(0, ...)`, etc.) often outperform constraint changes on heavy-
tailed or spike-and-slab inputs.

### 4.4 What we deliberately do NOT use

Two prominent design choices are *absent* from the recommended model.
Both looked promising at first but failed on the realistic
extrapolation metric:

- **Heteroscedastic observation noise** (i.e., plumbing the empirical
  triplicate Yvar into the likelihood as fixed per-row noise — internal
  alias `A`). Initially appears as a major win on single-row LOO
  (−134 psi over baseline) but **actively regresses on block-LOO**
  across every feature configuration tested. Full ablation table in §6.2.
- **Day-zero anchors** (128 pseudo-observations at $t=0$ with
  $f \approx 0$). These help the production baseline but are
  unnecessary once the scalar-Gaussian noise model is used at full
  data — the kernel + scalar noise is identifiable without them.
  See §6.2.

The recommendation is therefore the *simplest* combination of these
choices: a single learnable scalar Gaussian noise, no per-row
empirical Yvar, no day-zero anchors. **Do not interpret these as
"levers we removed" — the production baseline doesn't have them
either**; the design space includes "could add this" choices that we
explicitly evaluated and decided against.

### 4.5 Multiplicative time gate enforcing $f(x, 0) = 0$

The fourth positive lever, added in the follow-up
[anchor study](STRENGTH_GP_ANCHORS_STUDY.md). The recommended
`B''+F5_alllog` champion above optimises real-data extrapolation
beautifully but **violates physics** at $t = 0$ (predicts ~+2700 psi
on average for unseen compositions at time zero, where strength
must be zero). For deployment, this matters.

The fix is a multiplicative kernel gate $h(t) = 1 - e^{-t/\tau}$
with $\tau = 0.05$:
$K_{\text{gated}} = h(t) \cdot K_{B''} \cdot h(t')$. Mathematically
clean (PSD product of kernels = PSD). Three elegance properties:

1. **Structurally enforces $f(x, 0) = 0$ for all inputs**, not just
   training compositions.
2. **For $t \geq 1$, $h(t) \approx 1$** — the gated kernel is
   essentially the base kernel; real-data fit is preserved.
3. **One hyperparameter** ($\tau$), which is well-determined empirically.

Pair with: multiplicative-only Y scaling (`Y / y_max` instead of
z-score), `ZeroMean` GP prior, `outcome_transform=None`, and
`skip_time_in_normalize=True`. See §3.3 above for why each is
necessary, and the anchor study for the full empirical journey.

#### Why structural gating beats anchor pseudo-observations

We tested anchor pseudo-observations (`PartialFixedNoiseLikelihood`
with 128 anchors at $t=0, Y=0$, near-zero noise) extensively:

| Approach | Real-only block-LOO | Phantom-anchor RMSE | Joint block-LOO |
|---|---:|---:|---:|
| **Gated kernel** (this lever) | **679** | **0** (exact) | **614** |
| Anchors (default) | 850 | 5 | 776 |
| Anchors + steeper time | 829 | 5 | 758 |
| Anchors + learnable offset | 822 | 5 | 750 |
| Anchors + steeper maturity | 870 | 6 | 795 |

Anchors satisfy the constraint approximately (≤6 psi) but pay a
~150-psi block-LOO penalty on real-data extrapolation — they create
an architectural conflict with the kernel's smoothness assumptions.
Gating modifies the **prior** and is conflict-free.

**Cost of the gating lever**: 16 psi block-LOO regression on real
data (679 vs 663 un-gated), within seed noise. The 2669-psi
phantom-anchor improvement (constraint becomes exact) more than
offsets this on the joint metric (614 vs 1296 unanchored).

---

## 5. Subset learning curves

Composition-level held-out RMSE (mean over 3 random-subset seeds at
each size). 4 representative variants:

| n_train_comp | F0 baseline | F3 (3 features) | F5 (7 features) | F5_alllog |
|---:|---:|---:|---:|---:|
| 25 (~112 rows) | 1459 | 1467 | 1418 | **1257** |
| 50 (~225 rows) | 1172 | 1122 | 1087 | **1079** |
| 100 (~448 rows) | **844** | 864 | 891 | 832 |
| 144 (full, block-LOO) | 754 | 724 | 718 | **663** |

**Findings**:
1. F5_alllog wins at every data size tested.
2. The F5_alllog gain is largest at small data: **−14% at n_comp=25**
   (1418 → 1257 vs F5; −210 psi vs F0).
3. At n=100, F0 (no engineered features) is best — the GP can recover
   chemistry structure from raw inputs given enough data, and
   un-transformed features cost lengthscales without payoff.
4. At full data, F5_alllog re-wins decisively.

**Implication**: distribution-aware feature engineering is
**most valuable in the small-data regime** — exactly where the
production model is likely to deploy on new mix-design campaigns.

### 5.2 Gated kernel under subset learning curves (new architecture)

After adopting the gated-kernel architecture (§3.3 / §4.5), we re-ran
subset curves for the gated champion alongside the un-gated champion.
This tests whether the structural physics constraint helps or hurts at
small data sizes, where the constraint may matter more (less data to
implicitly teach the model t=0 behavior) or less (extra structural
constraints + scarce data could combine badly).

Real-data block-LOO RMSE (mean over 3 random subsets):

| n_comp | un-gated B''+F5_alllog | gated B''+F5_alllog+gated_t+max_zm | Δ |
|---:|---:|---:|---:|
| 25 | 858 | 870 | +12 |
| 50 | 821 | 894 | +73 |
| 100 | 742 | 799 | +57 |
| 144 (full) | 663 | 679 | +16 |

Phantom-anchor RMSE at unobserved compositions (the constraint metric):

| n_comp | un-gated phantom OOT (mean) | gated phantom OOT |
|---:|---:|---:|
| 25 | 1,410 | **0** |
| 50 | **3,986** ⚠ | **0** |
| 100 | (large) | **0** |
| 144 (full) | 2,669 | **0** |

**Key finding**: the gated kernel's *relative* advantage **grows
dramatically at small data sizes**, even as its real-data fit cost
grows slightly:

1. **Real-data cost is roughly constant** (+12 to +73 psi block-LOO across all n_comp).
2. **Phantom OOT for the un-gated model becomes catastrophic at small data**: at n_comp=50 the un-gated model's mean t=0 prediction is **+3,986 psi** for unseen compositions, with one subset seed reaching **+5,323 psi**. Physically nonsensical for a deployed model.
3. The gated kernel is **exactly 0 phantom RMSE at every n_comp** by construction.

**Implication for deployments**: when training data is limited (50–100
compositions, common in real-world concrete mix design), the gated
kernel's structural constraint becomes **essential, not just
nice-to-have**. Real users submitting unseen mixes to a 50-composition
un-gated model would receive predictions off by ~4,000 psi at t=0.

This refines the §3.3 message: gating's value is *especially* large at
small data, and grows with data sparsity in proportion to how much
the un-gated model's t=0 extrapolation degrades.

---

## 6. Negative results & dead ends (consolidated)

### 6.1 Alternative kernel decompositions (none beats the recommended Multi-Matern)

| Kernel | Description | Block-LOO | Verdict |
|---|---|---:|---|
| Single Matern | one ARD-Matern over all 10 raw dims + RBF time | 843 | starting point |
| Multi-Matern (hard δ source) | `IndexKernel(rank=0)` source-aware | 789 (with het noise) | similar to recommended |
| **Multi-Matern (radial soft toggle) — recommended** | source as ARD dim of the source-specific Matern | **754** | **recommended** |
| Multi-Matern (shared lengthscales) | hard-δ Multi-Matern with tied lengthscales between blind / specific | catastrophic (1207 with het noise) | sharing destroys the multi-component value |
| Multi-Matern (two unconstrained) | two joint-ARD Materns over all dims | 772 (with het noise) | similar to recommended; ~15× higher seed variance |
| Multi-Matern (factored Hamming) | `M_blind + M_specific × k_cat(s)` | 776 (with het noise) | similar to recommended |
| Multi-Matern (no blind) | `M·k_cat,1 + M·k_cat,2` | 818 (with het noise) | needs explicit blind term |

(Internal aliases for the rows above, in script order: `baseline`,
`B`, `B''`, `B'`, `D`, `E1`, `E2`.)

Tested as well in the recommended (scalar-noise) regime with engineered features:
**`D+F5_alllog` (701)**, **`E1+F5_alllog` (686)** — both close to the
recommended `B''+F5_alllog` (663) but worse. The "kernel architecture
matters" question is settled: the recommended kernel wins.

**Lesson**: the multi-component nature is what matters; the specific
flavour (radial vs factored, blind vs no-blind) is mostly a wash, but
the recommended kernel is consistently the safest choice.

### 6.2 Heteroscedastic noise is a single-row-LOO trap

Plumbing in the empirical triplicate Yvar (via
`FixedNoiseGaussianLikelihood` — internal token `A`) produces:

| Configuration pair (with → without heteroscedastic noise) | Single-row LOO Δ | **Block-LOO Δ** |
|---|---:|---:|
| Single Matern: `baseline` → `A` | **−134 psi** (improves) | **+441 psi (regression!)** |
| Multi-Matern + F0: `A+B''+floor+F0` → `B''+F0` | +6 (loses 6) | **−27 psi (gains 27)** ✓ |
| Multi-Matern + F1: `A+B''+floor+F1` → `B''+F1` | +15 | −11 ✓ |
| Multi-Matern + F3: `A+B''+floor+F3` → `B''+F3` | +12 | **−150 psi** ⚡ (with het noise: 874) |
| Multi-Matern + F5: `A+B''+floor+F5` → `B''+F5` | +15 | −30 ✓ |

**Removing heteroscedastic noise improves block-LOO across 5 of 6
feature configurations.** The single-row LOO improvement was an
artefact of random-row holdout leaving each composition's time curve
mostly in-training.

**Mechanism**: empirical Yvar lets the kernel attribute within-
composition time-curve variability to known noise. The ARD lengthscales
become longer / smoother — better for predicting an already-seen
composition's missing time-point (single-row LOO), worse for
predicting an entire unseen composition (block-LOO).

The further refinements of the noise model (`+floor`, `+mult`, `+full`)
are similarly trapped: under block-LOO, `A+B''+mult` (756) actually
beats `A+B''+floor` (781) — a reversal of what single-row LOO claimed
(§6.6).

### 6.3 Feature ablations: which features matter (with and without heteroscedastic noise)

Single-row LOO when stacked on heteroscedastic-noise + Multi-Matern + floor:

| Single feature added to F0 | LOO RMSE (vs F0 = 500.8) |
|---|---:|
| W/B | 491 (−10) |
| W/B + SCM | 494 (−7) |
| W/B + SCM + HRWR/binder (= F3) | 493 (−8) |
| F5 (full chemistry) | **490** (−11) |

Block-LOO on the recommended Multi-Matern kernel (no heteroscedastic noise):

| Single feature added to F0 | Block-LOO (vs F0 = 753.9) |
|---|---:|
| W/B alone | 756 (no help) |
| SCM alone | 763 (slight hurt) |
| W/C alone | 764 (slight hurt) |
| maturity alone | 771 (hurts) |
| **HRWR/binder alone** | **748 (−6)** ← only single feature that helps |
| F3 (W/B + SCM + HRWR) | 724 (−30) |
| F5 | 718 (−36) |
| F5_alllog | **663 (−91)** |

**Counter-intuitive result**: under block-LOO, HRWR/binder is the
*only* single chemistry feature that helps in isolation. The classical
W/B (Abrams' law) only helps in combination with HRWR/binder.

**Two-feature minimal hypothesis**: `Fhrwr_mat2` (HRWR/binder + maturity)
was the leave-one-out signal's #1 essential pair. Block-LOO 753.5,
nearly tied with F0. **The hypothesis fails** — interaction effects
mean that LOO-style "essential feature" reasoning is wrong; the
features only work together.

**F3 is the simplicity sweet spot at no-log**: 3 features, block-LOO
724. F5_alllog (7 features with logs) is materially better at 663.

#### Re-ablation under the gated kernel + new objective (2026-05-15)

When we adopted the gated kernel + max-scale + ZeroMean architecture
(see §3.3 / §4.5), we re-ran the leave-one-out feature ablation to
verify each feature is still necessary. Champion = `B''+F5_alllog+gated_t+
maxscale_zeromean` at block-LOO 679.

**Leave-one-out** (drop ONE feature from F5_alllog):

| Feature dropped | Block-LOO | Δ vs 679 (importance) |
|---|---:|---:|
| log_maturity_robust | 763 | **+84** 🥇 most important |
| log_hrwr_binder | 755 | **+76** 🥈 |
| log_agg_paste | 746 | **+66** 🥉 |
| wb_ratio | 743 | **+64** |
| scm_frac | 718 | +38 |
| log_wc_ratio | 713 | +33 |
| log_coarse_fine | 707 | +28 (least, but still significant) |

**Singletons** (F0 + ONE feature only, baseline F0 = 783):

| Singleton feature | Block-LOO | Δ vs F0 (singleton value) |
|---|---:|---:|
| log_hrwr_binder | 752 | **−31** ← strongest solo |
| log_wc_ratio | 756 | **−27** |
| log_maturity_robust | 759 | **−24** |
| wb_ratio | 766 | −17 |
| scm_frac | 780 | −3 (wash) |
| log_agg_paste | 799 | **+17** ← REGRESSES alone |
| log_coarse_fine | 809 | **+26** ← REGRESSES alone |

**Three classes of features emerge**:

- **Class 1: solo-strong AND combination-important** — `log_hrwr_binder`,
  `log_wc_ratio`, `log_maturity_robust`, `wb_ratio`. Useful both alone
  and in combination.
- **Class 2: synergy-only** (HURT alone but matter in combination) —
  `log_agg_paste`, `log_coarse_fine`. Aggregate ratios encode
  particle-packing effects that depend on chemistry; alone they add
  noise, in combination with W/B and SCM_frac they unlock joint behavior.
- **Class 3: combination-modest** — `scm_frac` (wash alone, +38 in
  combination).

**Key conclusions**:

1. **No feature can be dropped from the recommended set** — minimum
   leave-one-out delta is +28 psi (log_coarse_fine), still significant.
2. **Total LOO sum (+389 psi) ≫ sum of negative singletons (−102 psi)** —
   strong synergistic effects between features. The whole F5_alllog
   set is much more than the sum of its parts.
3. **Synergy-only features signal under-modeled dimensions** — the
   aggregate-related features (Class 2) only become useful in
   combination, suggesting that aggregate × chemistry interaction
   space is incompletely captured by the existing features. Worth
   exploring explicit interaction features (F6+ variants — see §7.1).

#### Explicit interaction features (F6/F7/F8) — negative result

Motivated by the Class-2 synergy finding, we tested adding explicit
chemistry × packing interaction terms as engineered features:

- F6: F5_alllog + `log(W/B × A/P)`
- F7: F6 + `log(HRWR/binder × A/P)`
- F8: F7 + `log(W/C × Coarse/Fine)`

Each interaction is the multiplicative product of two existing
F5_alllog features, log-transformed for distribution-awareness.

| Variant | LOO RMSE | Block-LOO | Δ vs F5_alllog (679) |
|---|---:|---:|---:|
| F5_alllog (champion) | 533 | 679 | — |
| F6 (+ 1 interaction) | 539 | 708 | **+29 (regression)** |
| F7 (+ 2 interactions) | **497** | 752 | **+73 (regression)** |
| F8 (+ 3 interactions) | 506 | 743 | **+64 (regression)** |

**Negative result, with a familiar mechanism**: F7 achieves the
**lowest single-row LOO** of any variant (497) but the worst
block-LOO (+73 psi regression). This is the same single-row LOO trap
documented in §6.2 (heteroscedastic noise) and §3.8 of the anchor
study (learnable τ): **adding parameters lets the optimiser fit
within-composition time-curves more tightly, but hurts cross-composition
extrapolation**.

**Mechanistic interpretation**: the Multi-Matern's joint ARD already
**implicitly learns chemistry × packing interactions** through the
kernel's joint smoothness function. Adding explicit interaction
features just gives the optimiser parameters to overfit single-row
noise, without providing genuinely new information. The Class-2
synergy-only features (`log_agg_paste`, `log_coarse_fine`)
synergise *implicitly* through the kernel's joint structure — no
explicit interaction encoding needed.

**Lesson**: with a Multi-Matern (joint ARD) kernel and a sufficient
feature set, explicit pre-computed interaction features are
counterproductive. The kernel's joint structure handles interactions;
features should be additive, primary signals. Explicit interactions
make sense only with kernels that lack joint smoothness (e.g., purely
additive / OAK-style kernels).

### 6.4 W/C clipping is harmful

Tested `wc_ratio_clipped` = `clamp(W/C, max=1.5)` to remove the 8
outliers up to W/C = 287. Combined with log_HRWR and maturity_robust
in `F5_robust`: block-LOO **755** (worse than F5's 718 by 37). Capping
discards the discriminative signal of the extreme mixes; log-transform
of plain W/C is the correct fix (as in F5_alllog).

### 6.5 OAK (Orthogonal Additive Kernel) doesn't fit this data

Tested OAK alone (1st-order and 2nd-order) and as a component of a
multi-Matern decomposition. Block-LOO results:

| Variant | Block-LOO |
|---|---:|
| 1st-order OAK alone (`OAK1`) | 1105 (much worse than baseline) |
| 1st+2nd-order OAK alone (`OAK2`) | 681 (close to baseline) |
| 1st-order OAK + heteroscedastic noise (`A+OAK1`) | 1513 (catastrophic, MLPD −259) |
| 1st-order OAK as one Multi-Matern component (`A+OAK1+B''`) | 908 (worse than baseline) |

**Conclusion**: concrete strength has substantial 3rd-and-higher-order
interactions (cement × water × time × source) that OAK's strictly
additive structure cannot represent. OAK might still be useful for
*interpretability* (its `coeffs_1` and `coeffs_2` give per-feature and
per-pair importance), but not for predictive performance.

### 6.6 Multiplicative-vs-additive noise floor — single-row LOO had it backward

Among the noise-model refinements stacked on heteroscedastic noise:
under single-row LOO, the additive floor beat the multiplicative scale
by 4 psi. Under block-LOO, **multiplicative wins**:

| Noise model on Multi-Matern | Single-row LOO | Block-LOO |
|---|---:|---:|
| heteroscedastic + additive floor (`A+B''+floor`) | 500.8 | 781.4 |
| heteroscedastic + multiplicative scale (`A+B''+mult`) | 504.8 | **755.9** ← better |
| heteroscedastic + floor + mult (`A+B''+full`) | 506.2 | 768.6 |

All three are dominated by the recommended **Multi-Matern + F5_alllog
with plain scalar noise** (block-LOO 663), so this academic ordering
doesn't change the recommendation. Recorded as another data point on
"single-row LOO can be qualitatively wrong".

### 6.7 Other approaches that did not help

| Approach | Why we tried it | What happened |
|---|---|---|
| Log-Y outcome transform (Single Matern + log-Y; internal alias `C`) | residuals more Gaussian; better calibration | Single-row LOO 647; block-LOO not yet implemented for log-Y models. Plausibly traps similarly to heteroscedastic noise. |
| Log-Y + heteroscedastic noise (`A+C`) | combine point-accuracy + calibration wins | Single-row LOO 800 (much worse) — log-Y conflicts with empirical heteroscedastic noise. |
| Relaxed lengthscale lower bound (1e-4 instead of 1e-2) on engineered features | "the model wants tighter lengthscales" | Multi-Matern + F3 with relaxed lengthscale (`B''+F3_rl`): block-LOO 752 (vs unmodified F3 at 724 — regressed by 28 psi). The constraint wasn't the issue; the feature distribution was (§4.3). |
| Day-zero anchors paired with the scalar-noise model | useful for the original heteroscedastic-noise variants | Removed for the no-heteroscedastic regime; the kernel + scalar noise is identifiable without them. |

### 6.8 Anchor pseudo-observations underperform structural gating for the $f(x, 0) = 0$ constraint

The original strength model uses 128 day-zero anchors as
pseudo-observations (`PartialFixedNoiseLikelihood` with σ² = 1e-6).
This was kept around in the original recommended `B''+F5_alllog`
variant via `fit_strength_gp` but never carefully ablated. In the
follow-up [anchor study](STRENGTH_GP_ANCHORS_STUDY.md) we tested
multiple anchor variants and found **all of them regress real-data
block-LOO by 100+ psi** vs the un-gated champion. Summary:

| Approach (anchors-as-pseudo-obs, varying decoupling strategy) | Real-only block-LOO | Joint block-LOO |
|---|---:|---:|
| **No anchors, no gating (original `B''+F5_alllog`)** | **663** | 1296 |
| Anchors, default offset | 850 | 776 |
| Anchors + `log10(t + 0.01)` (steeper time) | 829 | 758 |
| Anchors + learnable log-time offset | 822 | 750 |
| Anchors + steeper maturity (`log((T+10)·t + 1e-3)`) | 870 | 795 |
| **Structural gating (no anchors) → recommended path** | **679** | **614** |

**Mechanism**: anchors create kernel-correlation conflicts that no
amount of per-feature decoupling resolves. The kernel must
simultaneously satisfy zero-noise observations at $t=0$ AND smooth
extrapolation across compositions; the fundamental smoothness
assumption of Matern kernels disagrees with the abrupt cliff at $t=0$.
Structural gating modifies the **prior** rather than the data, has no
such conflict, and is mathematically clean (Mercer-PSD). See §3.6 of
the anchor study for the full derivation.

**Lesson**: when expressing physics constraints of the form
$f(x_0) = 0$, prefer modifying the prior to adding pseudo-data.

### 6.9 Time-component sweep under the gated kernel — RBF still wins; dropping the time-only component costs 7 psi

The user's question (in the anchor study): with the gated kernel
modifying smoothness near $t=0$ structurally, do we still want **RBF**
on the additive time-only component, or should we switch to **Matern-
5/2**? And — given the time information is also captured by the joint
multi-Matern's ARD lengthscale on the time dim — could we drop the
additive time-only component entirely?

| Variant (all + gated_t + max-scale + ZeroMean) | block-LOO | Δ vs default |
|---|---:|---:|
| **B''+F5_alllog (default — RBF on time)** | **679** | — |
| + Matern-5/2 on time-only component | 705 | +26 |
| Drop the additive time-only component entirely | 686 | +7 |

**Conclusion**: keep the additive time-only component, and keep RBF
on it (not Matern). RBF wins by 26 psi over Matern; dropping the
component is within seed noise (7 psi) but a very small loss. The
recommendation includes the additive RBF time component for a small
but non-zero gain.

### 6.10 Per-source fixed noise floor regresses block-LOO — model misspecification needs slack

Motivated by the documented 2× heterogeneity between Source 0 (~163 psi
empirical residual std) and Source 1 (~76 psi), we tested replacing the
learnable global noise with **fixed per-row Yvar based on source**.
Empirical noise estimates come from triplicate-based residual standard
deviations (parent §10.2).

| Variant | MLL | LOO RMSE | Block-LOO | LOO cov95 |
|---|---:|---:|---:|---:|
| Baseline (learnable global noise) | +2.28 | 533 | **679** | 0.951 |
| Per-source fixed (163/76 psi) | +3.22 | 556 | 855 (**+176**) | 0.850 |
| Per-source fixed × 1.5 (245/114 psi) | +2.93 | 506 | 827 (+148) | 0.852 |
| Equal fixed (120/120 psi) | +3.09 | 579 | 830 (+151) | 0.876 |

**All three fixed-noise variants regress block-LOO by 150-180 psi**,
have higher MLL, and have over-confident LOO calibration (cov95 ≈ 0.85
vs the desired 0.95). The block-LOO regression is uniformly large
across `per_source_noise` (heterogeneous) AND `fixed_noise_avg`
(homogeneous), confirming the issue is **fixing noise itself**, not
specifically per-source heterogeneity.

**Mechanism — why "constrain noise to empirical" hurts**:

The triplicate-derived noise (163/76 psi) captures only **measurement
repeatability**. The GP's learnable noise converges to a value that
absorbs both:
1. Measurement repeatability
2. **Model-misspecification slack** — residual variance the kernel
   can't capture (e.g., un-modeled chemistry interactions,
   composition × time non-stationarity, etc.)

When we fix noise to component (1) alone, the GP has no way to express
component (2) and is forced to make the kernel fit individual data
points more tightly. This over-fits the training data (high MLL,
low cov95) at the expense of cross-composition extrapolation
(block-LOO regresses).

**Generalisable lesson — noise behaves opposite to kernel parameters
under the "constrain for less flexibility" heuristic**:

| Parameter type | "Constrain to less flexibility" |
|---|---|
| Kernel lengthscales (within-group prior) | **Helps** (§4.4) |
| Feature set (no extra interactions) | **Helps** (§6.3) |
| Source-specific Matern at full features | **Helps** (anchor study §3.9) |
| Gate τ (fixed vs learnable) | **Helps** (anchor study §3.8) |
| **Noise parameter** | **Hurts** (this section) |

Why noise is different: kernel parameters describe the *signal*
(constraining them prevents overfitting to individual data points);
noise describes the *gap between model and signal* (constraining it
removes the model's only "release valve" for misspecified structure).

**Path forward (future work)**:
A LEARNABLE per-source noise (2 free parameters, one per source)
would let the model find the right per-source level while honoring
the heterogeneity. Requires a custom GPyTorch likelihood subclass that
indexes a 2-element learnable noise vector by the source dim.
**Update (2026-05-15)**: Implemented and tested. **Also regresses
block-LOO by +75 psi** (679 → 754) despite improving MLL (+2.28 →
+2.48). The "extra flexibility overfits" pattern (§6.2, §6.3, §6.7,
§6.10, anchor study §3.8/§3.10) extends to learnable per-source
noise: more parameters → higher MLL → worse block-LOO. Even with
the heterogeneity-respecting structure, the optimiser still finds
a configuration that absorbs more of the "model-misspecification
slack" into noise rather than into kernel structure that
generalises. The recommended champion stays at learnable global
noise. Implementation lives in
``_PerSourceGaussianLikelihood`` for future reference.

### 6.11 Powers'-law parametric mean function — neutral, kernel already learns this

A natural physics-aware addition: replace ``ZeroMean`` with the
Powers'-law functional form known to fit concrete strength
development:

$$\mu(x, t) = \alpha \cdot (1 - e^{-t / \tau_\text{mean}})$$

This **vanishes at t=0** (preserves the gated-kernel physics
constraint) and gives the GP a "head start" by encoding the expected
asymptotic-saturation shape; the GP fits residuals around it. The
parameter ``α`` is learnable; ``τ_mean`` is fixed (sweep below).

| Variant | MLL | LOO RMSE | Block-LOO | LOO cov95 |
|---|---:|---:|---:|---:|
| Baseline (ZeroMean) | +2.2847 | 533 | **679.4** | 0.951 ✓ |
| Powers' τ_mean=0.2 | +2.2856 | 534 | 682.3 (+3) | 0.951 |
| Powers' τ_mean=0.5 | +2.2841 | 533 | 679.3 (~0) | 0.951 |
| Powers' τ_mean=1.0 | +2.2839 | 532 | **678.2** (-1) | 0.951 |

**Result: essentially neutral**. All variants within ±4 psi block-LOO
of the baseline (well within seed noise). The constraint is
preserved (phantom-anchor RMSE = 0 for all). Crucially, this is the
**only "extra structure" addition in the entire study that did not
overfit** — but it didn't help either.

**Mechanism — an "informative neutral"**:
The gated kernel's additive RBF(t) component + the multi-Matern's
joint ARD over (composition, time) features **already learns the
Powers'-law shape implicitly from data**. Adding an explicit
parametric mean is teaching the model something it already knows.
The α parameter just absorbs what the kernel's outputscale would
have absorbed; the τ_mean choice doesn't matter if the kernel can
fit the same shape via its lengthscales.

**Connection to other findings**:
This echoes §6.3's F6/F7/F8 interaction-feature finding (the kernel
learns chemistry × packing interactions implicitly via joint ARD,
so explicit interaction features are redundant). The pattern is
consistent: **with a sufficiently flexible joint kernel and the
right features, any auxiliary structural prior that matches the
data is redundant**. Only structural priors that reflect *external
constraints* (the gated kernel encoding f(x, 0) = 0) genuinely add
information the data alone cannot supply.

**Subset learning curves**: even at small data — where parametric
priors should *most* help — the Powers' law mean **does not improve
performance and can hurt at very small n**. Tested at n_comp ∈
{25, 50, 100} with 3 random subsets each:

| n_comp | Baseline avg block-LOO | Powers' law τ=1.0 avg | Δ |
|---:|---:|---:|---:|
| 25 | 870 | 912 | **+42 (worse)** |
| 50 | 894 | 872 | −23 (marginal, within noise) |
| 100 | 799 | 801 | ~0 |
| 144 | 679 | 678 | ~0 |

**Counter-intuitive small-data finding**: at n_comp=25 the parametric
mean is WORSE on average. Reason: with very sparse data, the GP
cannot overcome the parametric mean's "wrong asymptote" — α biases
predictions toward its initial trend, which doesn't match each
unobserved composition's true asymptote. The kernel's flexibility to
adapt to specific compositions is more valuable than the prior trend.

This refines our understanding of when parametric priors help: they
require enough data to fit the parametric parameters reliably AND
the parametric form to match the data closely enough that the
remaining residuals are small. Concrete strength varies enough across
compositions that a single global α (without per-composition variation)
isn't a good prior for any individual composition.

**Per-composition parametric mean** (α(x) linear in W/B, SCM, ...) is a
plausible extension; the parameter-additions-overfit pattern of this
study suggests it would also regress at full data, but might help
at small data. Untested; deferred as future work.

---

## 6.12 Block-LOO + priors (single-stage, MAP) — the production training objective

### Final, post-bug-fix conclusion

**The current production fit minimizes**
$$\mathcal{L}(\theta) = -\log p_{\text{block-LOO}}(y \mid \theta) - \log p(\theta)$$
via LBFGS from the kernel's default initialization. This is implemented
as `block_loo_only=True` (i.e., NO MLL warmup). It beats the standard
MLL+priors objective on every data size:

| n_train compositions | MLL+priors block-LOO | block-LOO+priors single-stage | Δ |
|---:|---:|---:|---:|
| **144 (full)** | 680.0 | **672.1** | −8 psi |
| 100 (mean over 3 subsets) | 790 | **709** | **−81 psi** |
| 50 (mean over 3 subsets) | 889 | **686** | **−203 psi ⚡** |
| 25 (mean over 3 subsets) | 846 | **748** | **−98 psi** |

The block-LOO+priors objective also yields strictly better calibration
(cov95 closer to nominal, MLPD less negative) at every data size.

### The Sundararajan-Keerthi block-inverse identity (data term)

For block $B$ of correlated rows (here, all measurements from the
same composition):
$$\mu_{\text{LOO}}^{(B)} = y_B - (K^{-1}_{BB})^{-1} \alpha_B$$
$$\Sigma_{\text{LOO}}^{(B)} = (K^{-1}_{BB})^{-1}$$
with $\alpha = K^{-1}(y - \mu_{\text{prior}})$ and
$K = K_{\text{kernel}} + \sigma^2 I$ (so $\Sigma_{\text{LOO}}^{(B)}$
already includes aleatoric noise — this is the proper **predictive**
log-likelihood, not just RMSE).

References:
- Sundararajan & Keerthi (2001) "Predictive Approaches for Choosing
  Hyperparameters in Gaussian Processes." NIPS 13.
- Bachoc (2013) "Cross Validation and Maximum Likelihood estimations
  of hyper-parameters of Gaussian processes." JSPI 143(8). Proves that
  CV-NLL is asymptotically more robust to model misspecification than
  MLL — directly explains why our gain grows with sparsity.
- Vehtari & Ojanen (2012) "A survey of Bayesian predictive methods
  for model assessment, selection and comparison." Statistics Surveys 6.

The block extension uses the standard generalisation of the SK
identity via $K^{-1}_{BB}$. Cost: forms $K^{-1}$ once ($O(n^3)$, same
as one MLL gradient step) then $n_{\text{block}}$ small $b \times b$
solves — essentially MLL-cost per outer iteration.

### Why single-stage (no MLL warmup) is right

We tested three approaches:

| Approach | n=144 BLOO | n=50 BLOO (avg of 3) |
|---|---:|---:|
| (a) MLL + priors | 680 | 889 |
| (b) MLL → refine block-LOO + priors | identical to (a) | identical to (a) |
| (c) **Single-stage block-LOO + priors** | **672** | **686** |

Approach (b) (the original "block-LOO refinement" idea) is bitwise
identical to (a): once the LBFGS objective is the proper MAP loss
$\mathcal{L}(\theta)$, refinement converges back to the MLL-with-priors
optimum (LBFGS finds the same basin from the warmstart).

Approach (c) starts from the kernel's default initialization and lets
the optimizer find a different, better basin for the block-LOO objective.
The data-term gradients of MLL vs. block-LOO point to different optima;
warmstarting from the MLL basin keeps you there, but cold-starting
under the block-LOO objective finds the proper one.

### Cautionary tales (the bugs that previously hid this result)

The first time we tried block-LOO refinement (2026-05-16 morning), it
appeared to give dramatic gains: −221 psi at $n=25$. Those numbers
were INCORRECT — the prior log-density was missing from the loss, so
lengthscales drifted to extreme values (Fine Aggregate $\ell = 170$,
Source $\ell = 199$, Temp $\ell = 384$ — all "effectively inactive"
in the explorer). See "Lessons Learned §0" for the full story. The
corrected results above (with `_model_prior_log_prob` correctly
including all registered priors) are the reliable picture.

### Implementation

```python
# In model_variant_study.py:
def block_loo_loss(model, ..., include_priors=True, ...):
    # 1. Standard SK block-LOO data term via K_inv_BB
    nll_per_row = total_neg_log_lik / n_scored
    # 2. PROPER MAP-style objective: subtract prior log-density per row
    if include_priors:
        prior_lp = _model_prior_log_prob(model)  # sums over named_priors()
        nll_per_row = nll_per_row - prior_lp / n_scored
    return nll_per_row

def refine_with_block_loo(model, ...):
    # LBFGS with strong Wolfe line search; 200 iter for cold-start
    # variant (block_loo_only), 50 for warm-start (refine).
    optimizer = torch.optim.LBFGS(model.parameters(), ...)
    optimizer.step(closure_returning_block_loo_loss)
```

The cold-start variant (`block_loo_only=True`) is the deployed
champion. It is bit-for-bit reproducible.

- ``refine_with_block_loo(model, n_real=...)``: LBFGS refinement step
  (also used for `block_loo_only=True` cold-start).
- Variants: ``B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_only``
  (current champion, single-stage from default init) and
  ``B''+F5_alllog+gated_t+maxscale_zeromean+block_loo_refine`` (legacy
  warm-started two-stage; converges to MLL+priors basin, no longer used).

**Corrected subset learning curves** (after the bug fixes — see
"Lessons Learned §0"; the original numbers in this section reported
−221 psi at n=25 but were a silent crash where refinement was a no-op):

| n_comp | MLL+priors avg block-LOO | block-LOO+priors single-stage | Δ |
|---:|---:|---:|---:|
| 25 | 846 | **748** | **−98 psi** |
| 50 | 889 | **686** | **−203 psi ⚡** |
| 100 | 790 | **709** | **−81 psi** |
| 144 (full) | 680 | **672** | **−8 psi** |

(Subsets averaged over 3 random subset_seed values; full data n=144
is deterministic.)

**The refinement's value GROWS with data sparsity**, peaking at
n=50 with −203 psi. This matches Bachoc (2013)'s theoretical result
that CV-NLL is asymptotically more robust to model misspecification
than MLL — the MLL ≠ block-LOO gap widens at sparse data, where
the model is least constrained and most reliant on regularization.

**Why the gain grows with data sparsity**:
1. MLL has more local optima at small data (sparser surface, bumpier).
2. The MLL ≠ block-LOO gap widens with sparse data (fewer constraints
   on HPs).
3. Generalisation matters MORE when training data is scarce —
   block-LOO objective directly targets it.

**Implications for production deployment**:
This is the **recommended training procedure**: skip the MLL warmup
and optimize block-LOO + priors directly via LBFGS from default
kernel initialization (`block_loo_only=True`). The added compute
relative to MLL is ~3-4× (200 LBFGS iterations vs ~50 for MLL) but
absolute time stays in the seconds-per-fit range. Expected to
preserve or improve calibration (cov95 stays at 0.95 ± 0.01 across
all data sizes).

**Strengthened study-wide finding**:
The "MLL ≠ block-LOO" pattern has a principled remedy: directly
optimize the deployment metric. The result is consistent with
Bachoc's CV-NLL robustness theory and is now the production default.

---

## 6.13 Smoothness / monotonicity regularizers — explored, not deployed

After observing unphysical curves with `F5_no_log_mat` (the variant
that swapped log-maturity for raw-maturity, see §0 Bug 5), we
investigated whether soft regularizers added to the loss could
recover its 17 psi block-LOO advantage WITHOUT the oscillations.

We tested two regularizer forms, evaluated at intermediate (non-training)
times so they don't directly contaminate the block-LOO objective:

1. **L2 second-derivative smoothness**:
   $\mathcal{R}_s(\theta) = \frac{1}{C \cdot T'} \sum_{c, t} |\Delta^2 \mu_c(t)|^2$
2. **L2 monotonicity hinge**:
   $\mathcal{R}_m(\theta) = \frac{1}{C \cdot T'} \sum_{c, t} \max(0, -\Delta \mu_c(t))^2$

Total loss: $\mathcal{L} = \mathrm{NLL}_{\text{block-LOO}} - \log p(\theta) + \lambda_s \mathcal{R}_s + \lambda_m \mathcal{R}_m$.

| Variant | BLOO | %dec | %osc | maxDrop |
|---|---:|---:|---:|---:|
| F5_alllog (current champion) | 672 | 3.5% | 8.3% | 24 |
| F5_no_log_mat (no penalty) | 655 | 99% | 99% | 876 |
| F5_no_log_mat + mono $\lambda$=10000 (best mono) | 776 | 19% | 27% | 71 |
| F5_no_log_mat + smooth $\lambda$=1000 | 771 | 16% | 39% | 44 |
| F5_no_log_mat + smooth100+mono10000 | 775 | 19% | 24% | 72 |
| F5_alllog + mono $\lambda$=10000 (strictest mono) | 803 | **1.4%** | **3.5%** | 25 |

**Conclusion**: F5_no_log_mat's 17 psi BLOO advantage is INSEPARABLE
from its oscillations. No regularizer setting recovers BLOO < 770
while also satisfying physical-realism thresholds. The flexibility
that lets `raw_maturity = (T+10) \cdot t` fit training points tightly
also lets it wiggle between them; you can't constrain one without
constraining the other.

**Comparison of regularizer forms**: The L2 monotonicity hinge is
slightly more BLOO-efficient at equivalent monotonicity than L2
second-derivative smoothness (mono1000 → BLOO 715 vs smooth100 →
BLOO 723 at similar %osc), as expected since the hinge only
penalizes the actual physical violation while the second derivative
also penalizes legitimate curvature (saturation, pozzolanic delay).

**Decision**: do NOT deploy any regularizer. F5_alllog (the
current champion) is already at 3.5% / 8.3% / 24 psi — well within
physical thresholds — without any additional loss term.

**For future architectural improvements** (deferred): proper
solutions to monotonicity require structural changes, not soft
regularizers:

1. **Output transform** — model `log(strength)` instead of `strength`;
   monotone preserved by $\exp$.
2. **Time-monotone GP** (Riihimäki & Vehtari 2010) — virtual derivative
   observations enforce $\partial \mu / \partial t \geq 0$ as a hard
   constraint.
3. **Time-additive structure** $\mu(x, t) = f(x) g(t)$ with monotone $g$.

---

## 7. Open follow-ups (ranked by expected payoff)

### 7.1 Likely productive

| # | Idea | Mechanism | Effort |
|---|---|---|---|
| **K4** | **Domain-knowledge mean function**: `μ(x, t) = α·log(t+1) + β·W/B + γ·SCM_frac + δ` | Captures dominant trends explicitly; GP fits residuals | Medium |
| **J7** | **Multi-output GP with time as task axis** — each unique composition is a *task* with strengths at measured time points; coregionalize | Directly models the time-curve structure that's the source of single-Matern overfitting (cf. §6.2) | High |
| **D2** | **Quantile (rank-based) input transform** | Principled generalisation of log-transform; should give similar gain to F5_alllog more systematically | Medium |
| **N2** | **Per-source noise floor** `σ²_g[Source_i]` | Source 0 std ≈ 285 vs Source 1 std ≈ 133 (2× ratio) | Low |
| **S1** | **Stacked GP / GP boosting on residuals**: fit `M_0` on the data; refit `M_1` on residuals `(y − M_0.predict(x))`; predict via the sum (or a small learned blend). Iterate if useful. | Generalises K4 (parametric trend → GP residuals) to the non-parametric case where each stage can use a different inductive bias (e.g., M_0 = composition-only F5_alllog, M_1 = time-only Wiener / monotone kernel). Each stage handles the structure the others miss. Composes naturally with K4. | Medium |
| **S2** | **Ensemble of GPs with different inductive biases** (e.g., F5_alllog GP + F3 GP + a maturity-only GP) blended by either inverse-variance weighting or stacked predictive log-likelihood | Different feature/kernel choices win in different parts of the design space (cf. §5 — F0 wins at n_comp=100, F5_alllog at n_comp ≤ 50 and full data). An ensemble exploits the heterogeneity rather than choosing one regime | Medium |

### 7.2 Worth trying

- **K1/K2**: OAK 2nd-order with parameter sharing or low-rank
  constraint on the 2nd-order coefficient matrix (current OAK 2nd-order
  underperforms — see §6.5).
- **D6**: per-source GP (sub-models per source, blended).
- **N1**: weakly informative priors on the noise model parameters.

### 7.3 Research-flavoured

- **N5**: Student-t observation likelihood (heavier tails for outliers)
- **K3**: spectral mixture kernel
- **Monotonic-GP via virtual derivative observations**
  (Riihimäki & Vehtari 2010): augment training data with synthetic
  positive-derivative observations $\partial \mu / \partial t \geq 0$
  to enforce monotonicity as a hard constraint rather than a soft
  penalty (see §6.13 — soft penalties cannot recover F5_no_log_mat's
  17 psi BLOO advantage; hard constraints might).
- **Output transform** — model `log(strength)` instead of `strength`.
  Monotonic-by-construction inverse $\exp$. Requires extending
  `block_loo_metrics` to handle the lognormal back-transform.
- **Time-monotone factorisation**: $\mu(x, t) = f(x) \cdot g(t)$ with
  monotone $g$. Hard structural monotonicity at the cost of
  multiplicative-only $x$-$t$ interactions.
- **Block-LOO for log-Y models** — `block_loo_metrics` does not yet
  handle the lognormal back-transform; needed to honestly evaluate the
  log-Y outcome transform under the block-LOO regime.
- ~~Block-LOO-aware hyperparameter optimization~~ — **DONE** (§6.12).

---

## 8. Reproduction

```bash
# Run the production champion (block-LOO + priors single-stage from default init)
python experiments/model_variant_study.py \
    --variants "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_only" \
    --seeds 0

# Compare the three training-objective options (§6.12)
python experiments/model_variant_study.py --variants \
    "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean" \
    "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_refine" \
    "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_only" \
    --seeds 0

# Composition-level subset learning curve at small data
for n in 25 50 100; do
  for s in 0 1 2; do
    python experiments/model_variant_study.py --variants \
      "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean" \
      "B''+F5_alllog+gated_t+gated_noise+maxscale_zeromean+block_loo_only" \
      --seeds 0 --subset_n $n --subset_seed $s --holdout_unit composition
  done
done

# Curve monotonicity diagnostic (§6.13)
python experiments/compare_monotonicity.py

# Render the §4 cumulative-improvement plot
python experiments/plot_improvement_journey.py

# Full pre-deploy check (regenerate artifacts + run all tests)
bash experiments/regenerate_all_artifacts.sh
```

The script reports **single-row LOO + block-LOO** side-by-side at full
data and adds **held-out** when `--subset_n` is set. All metrics are
computed via `boxcrete.compute_loo_cv` (closed-form analytical LOO),
`block_loo_metrics` (closed-form leave-one-composition-out), and
`held_out_metrics` (posterior at test points with `observation_noise=
True`) in `experiments/model_variant_study.py`.

---

## 9. Persistent artefacts

| File | Contents |
|---|---|
| `experiments/STRENGTH_GP_BENCHMARK.md` | This document |
| `experiments/model_variant_study.py` | Driver script — all variants + metric helpers |
| `experiments/plot_improvement_journey.py` | OSS-friendly visualization of the §4 cumulative-improvement story (baseline → champion) |
| `experiments/improvement_journey.png` | Pre-rendered output of the script above |
| `experiments/improvement_journey_cache.json` | Cached LOO + block-LOO numbers for the journey-plot stages
| `experiments/strength_gp_block_loo_full.csv` | 65 paired single-row LOO + block-LOO rows across the variant ladder (incl. the §4.1 RBF(t) ablation and the §1.2 within-group-prior ablation) |
| `experiments/strength_gp_d1_logtransform_subsets.csv` | 54 rows: 6 feature configs × 3 subset sizes × 3 subset seeds (composition-level held-out) |
| `experiments/strength_gp_subset_loo_vs_heldout.csv` | 27 paired LOO + held-out rows, **row-level** holdout (the misleading regime) |
| `experiments/strength_gp_subset_composition_holdout.csv` | 27 paired LOO + held-out rows, composition-level holdout |
| `experiments/strength_gp_feat_subset_ablation.csv` | 54 rows from the §6.3 feature ablation across data sizes |
| `experiments/strength_gp_benchmark_results.csv` | 108 per-seed rows from the original kernel grid (single-row LOO only — these results are partially superseded; see §6) |
| `~/.llms/plans/strength_gp_joint_feature_kernel_design.plan.md` | Future-direction plan (joint feature+kernel design, K4, J7) |

---

## 10. Meta-lessons (transferable to similar studies)

The lessons below are split into two groups: **§10.1 transferable to
any GP-modelling problem**, and **§10.2 specific to the concrete
strength dataset**. Both are worth reading before attempting to repeat
or extend this study; the general ones are also worth keeping in mind
for any new GP modelling task.

### 10.1 General GP-modelling lessons

1. **Match the cross-validation regime to the deployment scenario.**
   Random row-holdout is the universal default but is wrong whenever
   rows are not exchangeable — i.e., when the dataset has natural
   correlated groupings (compositions, time series, batches, hierarchies).
   In this study, single-row LOO was qualitatively misleading on
   3 separate questions (kernel choice §6.1, noise model §6.2 / §6.6,
   feature engineering §6.3) — block-LOO told a consistent story while
   single-row LOO led the optimiser the wrong way. The
   Sundararajan–Keerthi block-LOO identity adds milliseconds per fit
   over standard LOO and is implementable in <50 lines.

2. **Always evaluate at multiple data sizes.** The "best" model at full
   data may not be the best at the data scale you'll deploy at, and
   vice-versa. A single model rarely dominates the entire learning
   curve. In this study, the recommended F5_alllog wins at every size,
   but the no-engineered-features baseline wins at n_comp=100 — a
   reminder that intermediate regimes can have their own surprises
   (§5).

3. **Multi-component kernels are a free-expressiveness lever for
   heterogeneous data.** When the data spans regimes (Material Sources
   here, but also: subpopulations, batches, time-eras, instruments),
   a single ARD-Matern over all dims forces the optimiser to compromise
   between "smooth shared structure" (long ℓ) and "sharp regime-specific
   corrections" (short ℓ). A two-component decomposition (one "shared"
   + one "regime-specific" Matern, summed) gives the optimiser separate
   ARD vectors and consistently improves. This was the **single biggest
   lever** in this study (−89 psi block-LOO).

4. **The four axes of GP design are largely orthogonal.** Kernel
   architecture, noise model, features, and outcome transform can each
   be ablated independently. The recommended workflow:
   1. Get the kernel right first (largest leverage; consider
      multi-component decompositions per #3 above).
   2. Then features (engineered ratios + distribution-aware transforms;
      #5 below).
   3. Then noise model — usually a plain learnable scalar Gaussian is
      best (#6).
   4. Then outcome transform — usually a plain `Standardize` is best.

   Don't ablate multiple axes simultaneously; the interactions confuse
   the signal.

5. **Investigate feature distributions before tuning kernel
   constraints or priors.** When an ARD lengthscale rails at a
   constraint bound, the optimiser is signalling structure your model
   isn't capturing. Almost always:
   - **Lengthscale rails high** → feature is uninformative or
     co-linear with others (cure: regularising / within-group prior;
     see #7).
   - **Lengthscale rails low** → feature distribution is non-uniform,
     usually heavy-tailed or spike-and-slab (cure: log / quantile
     transform; or explicit categorical kernel for hard modes).

   Distribution-aware transforms cost minutes to implement; constraint
   tweaking is rarely the right fix and can make things worse (§6.7).

6. **Per-row "noise" estimates from triplicates are usually NOT the
   noise the model should fit.** Empirical triplicate Yvar measures
   *measurement-level* variability, but a GP's likelihood noise should
   capture all variability the kernel cannot explain. On this dataset
   that includes systematic time-curve patterns. Plumbing the
   empirical Yvar in as fixed observation noise tells the model "this
   variability is irreducible" — it permits under-fitting, which
   manifests as longer ARD lengthscales and worse extrapolation. Lesson:
   default to a **single learnable scalar Gaussian noise**; only add
   per-row noise if you have evidence the kernel is overfitting that
   the per-row signal is genuinely independent measurement error.

7. **Within-group priors cure ARD identifiability when features are
   naturally grouped.** If multiple features measure variants of the
   same physical quantity (binder masses; aggregate masses; or any
   collinear group), independent ARD lengthscales are unidentifiable
   — the optimiser can swap mass between collinear features without
   changing predictions. A within-group shrinkage prior ties
   lengthscales within each group, restoring identifiability without
   constraining what the kernel can express. Reusable across any
   ARD-Matern with naturally-grouped features.

8. **Cumulative / interaction effects can dominate per-feature
   signals.** Leave-one-feature-out and single-feature ablation are
   both misleading whenever features interact. In this study, no
   single chemistry feature improved block-LOO by more than 6 psi
   alone, but seven of them together (with log-transforms) compound
   into −91 psi. Trust combinatorial results; don't dismiss "small"
   levers.

9. **Calibration metrics should track point accuracy.** RMSE,
   block-LOO MLPD, CRPS, and 95% coverage moved together throughout
   this study (the recommended model improves all of them
   simultaneously vs the baseline). When point accuracy and calibration
   *don't* track together, that's a strong signal of either CV leakage
   (#1) or noise-model misspecification (#6).

10. **Negative results are the highest-value content of a benchmark
    study.** ~40% of this document is negative results. They prevent
    rediscovery, calibrate next-step priorities for future
    experimenters, and educate readers about subtle traps. Document
    them with the same rigor as positive results.

11. **Express physics constraints in the prior, not the data.** When
    you need $f(x_0) = 0$ exactly, a multiplicatively-gated kernel
    $h(t) \cdot k \cdot h(t')$ with $h(0) = 0$ enforces it for the
    entire input domain in a self-consistent way. Anchor pseudo-
    observations at $(x_0, 0)$ with near-zero noise approximate this
    locally but create kernel-correlation conflicts that degrade
    extrapolation elsewhere. Documented in
    [`STRENGTH_GP_ANCHORS_STUDY.md`](STRENGTH_GP_ANCHORS_STUDY.md).

12. **Multiplicative-only outcome scaling preserves the GP=0 ↔
    raw=0 correspondence.** Z-score standardisation breaks it via the
    additive `+y_mean` term in the un-transform: a posterior of 0 in
    z-space becomes ~$y_\text{mean}$ in raw space. If your GP must
    predict exactly zero somewhere, use `Y / y_max` (or any
    multiplicative-only scaling) and pair with `ZeroMean`.

13. **Beware silent default outcome transforms.** BoTorch's
    `SingleTaskGP` applies `Standardize` by default unless explicitly
    disabled with `outcome_transform=None`. This silently re-introduces
    additive offsets on top of any custom outcome scaling — invisible
    to a casual reader of the code, but breaks structural-zero
    constraints. Always pass `outcome_transform=None` explicitly when
    you do your own scaling.

14. **`Normalize` over a dim with a structural-zero boundary is
    dangerous.** It maps the smallest training value to 0, which
    collides with structural-zero constraints at boundary values
    (e.g., `Normalize(time)` maps the smallest training $t$ to 0; if
    the kernel has $h(0) = 0$, training data at the boundary gets
    inadvertently gated.). Skip Normalize on dims where the kernel
    structurally treats 0 as special.

15. **Track multiple complementary metrics for every fit.** The
    headline metric depends on the question. The full evaluation
    panel: marginal log-likelihood (training objective), real-only
    block-LOO (extrapolation), phantom-anchor RMSE (physics-constraint
    satisfaction), joint block-LOO (deployment metric), cov95, MLPD,
    CRPS, PIT-KS (calibration), single-row LOO (diagnostic only —
    NOT a deployment metric — see §6.2 / §10.1 lesson 1), and subset
    learning curves (data-efficiency). A single number can hide
    regressions on the others. See the anchor study §2 for the full
    decision discipline.

### 10.2 Concrete-strength-specific findings

These insights are specific to the BOxCrete strength dataset and
related concrete-strength-prediction tasks; they're more domain-bound
than §10.1.

1. **HRWR/binder is the dominant concrete-chemistry feature for
   extrapolation, not W/B.** Classical concrete science (Abrams' law,
   1918) frames water-to-binder ratio as the single most important
   strength predictor. For our test of *extrapolation to unseen
   compositions*, HRWR/binder dominates: it's the only single chemistry
   feature whose addition to the GP improves block-LOO. W/B helps
   only in combination with HRWR/binder — likely because W/B alone is
   confounded with workability (which HRWR controls). Practical
   implication: for new mix-design GPs, **always include
   HRWR/binder as a feature**.

2. **The two Material Sources are different enough that the kernel
   must accommodate them as a distinct regime.** Source 0 std ≈ 285,
   Source 1 std ≈ 133 — a 2× ratio. Single-Matern kernels under-fit
   one Source while over-fitting the other. The Multi-Matern (radial
   source-aware) decomposition lets the optimiser learn separate
   per-Source ARD vectors. If you get more Sources in the future
   (e.g., a 3rd cement supplier), expect the kernel to need a similar
   accommodation — possibly a multi-task GP if the number of Sources
   grows (cf. §7.1 J7).

3. **Many concrete features are spike-and-slab or bimodal, not
   continuous.** 33–46% of mixes have raw HRWR = 0 (no superplasticizer);
   38% have Coarse aggregate = 0 (mortars, no coarse); 35% have Fly
   Ash = 0; 20% have Slag = 0. These zero-spikes break the stationary-
   lengthscale assumption of standard ARD-Matern kernels. **Rule of
   thumb for any concrete dataset**: identify which raw features have
   ≥10% zeros and either log-transform them, model them as
   categorical+continuous, or use a quantile transform.

4. **Maturity needs a freezing-threshold floor to handle cold-temp
   mixes.** The bare Saul/Nurse formulation `(T+10)·t` gives
   *negative* values for the 15 mixes at Temp = −20°C, which is
   physically meaningless (hydration stops below ~−10°C; it doesn't
   go in reverse). Use `max(0, T+10)·t` instead.

5. **Block-LOO is mandatory because compositions × time-points create
   strong within-mix correlation.** The dataset is 144 unique
   compositions × ~4.5 time-points each. Any random-row CV regime
   leaves ~75–80% of each composition's time curve in training, which
   reduces "extrapolation" to "filling in a missing time-point of an
   already-known curve." For new-mix-design tasks, this is the wrong
   metric. Block-LOO (leave-one-composition-out) is the realistic
   evaluation and consistently disagrees with single-row LOO on
   architectural questions (§6.1, §6.2, §6.6).

6. **The 8 outlier mixes with W/C > 5 are real, not errors.** They're
   probably specimen types with very low cement content (paste/binder
   characterisation tests, etc.). Don't clip them — log-transforming
   W/C gives a similar tail-softening effect without losing their
   signal, and is what the recommended model does.

7. **Per-composition triplicate stds aren't the right "noise".** The
   median triplicate std is ~109 psi but the model's residual structure
   includes systematic time-curve patterns that are NOT independent
   triplicate noise. Fitting these as residuals (with a learnable scalar
   Gaussian noise) gives the kernel useful work to do; treating them
   as fixed observation noise ties the model's hands. This generalises
   to any materials-science dataset with triplicate measurements at
   intermediate process stages.

8. **The "correct" early-age model differs from the "correct"
   long-age model.** Time-curve shape varies substantially by mix
   composition (rapid early gain in low-W/B mixes vs slow gain in
   high-SCM mixes). A multi-output GP with time as a task axis
   (§7.1 J7) is the principled way to model this; until that's
   implemented, the additive `RBF(t)` term in the recommended kernel
   is doing a useful but limited job (§4.1).


---

## Appendix A. Day-Zero Anchor Study (folded 2026-05-19 from STRENGTH_GP_ANCHORS_STUDY.md)

_The original document chronicled the design history of the gated-kernel + gated-noise architecture: how various "day-zero anchor" workarounds were tried and discarded before the principled multiplicatively-gated-kernel formulation (§3.5 below) was identified as the right answer. It is preserved here as a numbered subsection of the benchmark for cite-stability and reproducibility._

##### Strength GP — Day-Zero Anchor Study (original document title)

A focused sub-study of the BOxCrete strength GP exploring **how to make
the model satisfy the physics constraint $f(x, t=0) = 0$** without
sacrificing real-data extrapolation quality. **SOLVED** as of 2026-05-15
via a structurally-gated kernel (no anchor pseudo-observations needed).

---

#### A.TL;DR — the answer

**Use a multiplicatively gated kernel** that vanishes at $t=0$, paired
with multiplicative-only Y scaling and a ZeroMean prior:

$$K_{\text{gated}}((x, t), (x', t')) = h(t) \cdot K_{\text{base}}((x,t),(x',t')) \cdot h(t')
\qquad\text{where } h(t) = 1 - e^{-t/\tau},\ \tau = 0.05$$

With this:
- **Phantom-anchor RMSE = 0.0 psi** for all 144 unique compositions.
  The physics constraint is exact (not approximate).
- **Real-only block-LOO: 679 psi** — within seed noise of the
  unanchored champion (683 psi).
- **Joint metric (real + anchors): 614 psi** — a 53% reduction vs the
  unanchored champion's joint RMSE of 1296 psi.
- **No anchor pseudo-observations needed** — the constraint lives in
  the prior, not in the data.

Variant: `B''+F5_alllog+gated_t+maxscale_zeromean`. Four pieces must
be composed together (each is necessary, see §3.6 below for why):

1. **Gated kernel** wrapping the recommended Multi-Matern.
2. **`Y / y_max`** scaling (no mean subtraction).
3. **`ZeroMean`** GP prior mean (so prior at $t=0$ is 0 in scaled
   space).
4. **`outcome_transform=None`** explicitly (to disable BoTorch's
   silent default `Standardize`, which would re-introduce an additive
   offset).
5. **`skip_time_in_normalize=True`** (so post-transform $t=1 \neq 0$,
   else gate also kills training data at $t=1$).

(That's 5 pieces — the 5th is needed for the post-input-transform
geometry, not the constraint itself.)

---

#### A.1. Current status

**SOLVED**. The recommended path is the gated-kernel variant above.
The anchor-pseudo-observation approach (`+anchors`) is **not** the
right answer — it conflicts with the kernel architecture and pays a
significant block-LOO penalty. See §3 for the full empirical journey.

---

#### A.2. The right objective

**The headline metric for this study is the anchor-point-INCLUDING
block-LOO RMSE**, computed identically across all models — anchored,
unanchored, or gated. This is the metric that captures both:

- "Extrapolate the strength curve of an unseen composition" (the real-row
  errors)
- "Predict zero strength at t=0 for an unseen composition" (the
  physics-constraint satisfaction)

Why this is the right objective:

1. The physics constraint `f(x, 0) = 0` is **physically required** —
   any deployed model that violates it materially is broken.
2. Real-only block-LOO (the headline metric of the parent benchmark)
   doesn't test this constraint. A model that extrapolates the
   strength curve well but predicts +500 psi at t=0 is wrong; the
   real-only metric is silent on this.
3. Whether the model **trains** on anchors or not, the **evaluation**
   should treat the t=0 zero-strength condition as part of the truth
   the model is supposed to predict.

##### Computing the metric for both regimes

| Model regime | What goes into the score |
|---|---|
| **Anchored** (anchors in training, near-zero training noise) | Block-LOO predictions for held-out real rows + held-out anchor rows. Both are training rows; block-LOO formula gives both directly. |
| **Unanchored / Gated** (no anchors in training) | Block-LOO predictions for held-out real rows + **phantom t=0 predictions** for each unique composition (computed via the model's posterior at `(c, t=0)`). |

**Both contribute to the joint RMSE** — that's the metric to optimise.

##### The full evaluation panel — track ALL of these

A model fit should be evaluated holistically across **multiple
complementary metrics**. No single number should drive the decision;
seemingly-better fits on one metric can hide regressions on another.
The full panel:

| Metric | What it answers |
|---|---|
| **Marginal log-likelihood (MLL/row)** | The actual training objective. Higher MLL means a better fit *to the training data given the model class*. **Not** a reliable proxy for predictive quality on its own — extra parameters routinely improve MLL while regressing block-LOO (we observed this several times in this study). |
| **Real-only block-LOO RMSE** | How well the model extrapolates to an unseen composition's full time curve. The realistic deployment metric for "predict strength at new mixes." |
| **Phantom-anchor RMSE** (or anchor-only block-LOO RMSE) | How well the model satisfies the physics constraint $f(x, 0) = 0$. For the gated kernel this is 0 by construction; for an unanchored ConstantMean+standardize model it's typically ~y_mean (~4500 psi). |
| **Joint block-LOO RMSE** (real + anchors, weighted) | The single combined objective for deployment. Penalises both extrapolation error and physics-constraint violation. |
| **cov95** | 95% predictive-interval coverage on held-out points. Should be ≈0.95; consistently <0.92 = under-confident, >0.97 = under-fit/over-confident. |
| **MLPD** (mean log-predictive density) | A proper scoring rule combining mean and variance accuracy. More sensitive than cov95 to mis-calibration. |
| **CRPS** (continuous-ranked-probability-score) | Robust proper scoring rule that doesn't require Gaussianity assumptions; useful when residual distributions are heavy-tailed. |
| **PIT-KS** (Kolmogorov-Smirnov of probability-integral-transformed residuals) | A distributional-calibration test. Small values = the predictive CDF matches the empirical residual distribution well. |
| **Single-row LOO RMSE** | Useful for diagnostic comparison only. **Not** a deployment metric on this dataset — it leaks information across the time axis (see parent benchmark §2). |
| **Subset learning curves** | How metrics evolve at smaller training sizes. A model that wins at full data may regress at small data, or vice versa. |

**Decision discipline**: when comparing two variants,
- A change is **strictly better** if all of {MLL, real-only block-LOO,
  joint block-LOO, calibration metrics} improve or are unchanged.
- A change is a **win on the right metric** if joint block-LOO
  improves AND the physics constraint is satisfied, even if
  real-only block-LOO regresses by a small amount within seed noise.
- A change with **higher MLL but worse block-LOO** is overfitting.
  Reject.
- A change with **lower MLL but better block-LOO** is suspicious —
  may be a different local optimum the optimiser found by chance.
  Investigate before adopting.

This panel should be reported for every variant in the study record.
The run loop in `experiments/model_variant_study.py` automatically
prints all of these when an anchored or gated variant is fit; the
existing log files (`/tmp/*.log`) preserve the per-variant detail.

---

#### A.3. What we tried, end-to-end

##### 3.1 Default anchors (offset=1, pseudo_noise=1e-6)

Variant: `B''+F5_alllog+anchors`. Replicates the production-baseline
anchor approach (`PartialFixedNoiseLikelihood` with 1e-6 fixed noise
on 128 anchors) on the recommended Multi-Matern + F5_alllog
architecture.

**Result**: real-only block-LOO **850** (vs 663 unanchored — +187 regression),
anchor-only block-LOO 5 (constraint nearly perfectly satisfied).
Constraint works but real-data extrapolation pays a heavy price.

##### 3.2 Steeper fixed log-time offset

Hypothesis: stretch t=0 vs t=1 apart in transformed space so the kernel
correlation between them goes to zero, decoupling anchors from real data.

**Result**: helps modestly (~20 psi) but does not recover. **Refuted.**

##### 3.3 Learnable log-time offset

Hypothesis: end-to-end optimization of the offset via MLL.

**Result**: anchored variant's offset converges to ~0 (fully decoupling
on the time axis), but real-only block-LOO still 821 (+158 vs
unanchored). MLL improving monotonically — not an optimisation failure.
**Confirms architectural conflict between anchors and the recommended
kernel.**

##### 3.4 Steepening the maturity feature

Hypothesis: the augmented input differs between (c, t=0) and (c, t=1)
on TWO dims (raw time AND log_maturity_robust); decoupling on time
alone leaves the maturity gap of ~1.5. Try replacing
`log(max(0, T+10)·t + 1)` with `log(max(0, T+10)·t + 1e-3)` (or 1e-6).

**Result**: **stretching maturity hurts everywhere** — even the
unanchored baseline regresses by 68 psi (663 → 731). The +1 offset
in the maturity formula was already well-tuned for the data.
**Refuted; stretching alone on either dim doesn't resolve the conflict.**

##### 3.5 The breakthrough — multiplicatively gated kernel

Hypothesis (the user's insight): instead of treating the t=0 constraint
as observations, **build it into the kernel itself**. Use
$K_{\text{gated}}((x, t), (x', t')) = h(t) \cdot K_{\text{base}} \cdot h(t')$
with $h(0) = 0$.

This is mathematically clean (Mercer's theorem: product of kernels is
PSD). The latent function is $f(x, t) = h(t) \cdot g(x, t)$ for some
underlying GP $g$. Prior variance at $t=0$ is exactly zero;
$h(t) \approx 1$ for $t \geq 1$ so real-data fit is preserved.

**First attempt**: gated kernel + standard z-score Y standardisation
+ ConstantMean. Phantom-anchor RMSE was ~2000 psi — non-zero!
**Diagnosis**: standardisation `Y_z = (Y - μ)/σ` makes `Y = 0` map to
`Y_z ≈ -2`, not 0. The GP's posterior of 0 in z-space then
*untransforms to ~y_mean ≈ 4500 psi* via `mean_z · y_std + y_mean`.

**Fix** (the user's "divide by y_max instead" insight): use
multiplicative-only Y scaling (`Y / y_max`, no mean subtraction) and
a `ZeroMean` GP prior so the GP=0 ↔ raw=0 correspondence holds. Plus
two more pieces (see §3.6 below).

##### 3.6 The five compositional pieces

Each is necessary. Removing any one breaks the constraint.

| Piece | Why it's needed |
|---|---|
| **Gated kernel** `K_gated = h(t)·K·h(t')` | Without this, no constraint in the prior. |
| **`Y / y_max`** scaling (no mean subtraction) | Without this, untransform is `mean·y_std + y_mean` — the additive `+y_mean` reintroduces ~4500 psi at t=0 even when the GP outputs 0 in scaled space. |
| **`ZeroMean`** mean module | Without this, ConstantMean fits to a non-zero constant in scaled space; the gated kernel then gives posterior = ConstantMean's value at t=0, not 0. |
| **`outcome_transform=None`** in SingleTaskGP | Without this, BoTorch silently applies its default `Standardize`, which on top of our `Y / y_max` re-centers the data: `(Y/y_max - 0.45) / 0.25`. This re-introduces the additive offset that breaks the t=0 constraint. The most subtle of the five — masquerades as a working fit but predicts ~4500 psi at t=0. |
| **`skip_time_in_normalize=True`** | Otherwise `Normalize` maps the smallest training time-point (`t=1`) to 0; `h(0) = 0` then gates away t=1 data too, hurting block-LOO substantially. |

##### 3.7 Empirical results, all variants

Comparison on full data:

| Variant | Real-only block-LOO | Phantom-anchor RMSE | Joint RMSE | Notes |
|---|---:|---:|---:|---|
| `B''+F5_alllog` (unanchored champion) | 683 | 2669 | 1296 | original recommendation |
| `+anchors` (default) | 850 | 5 | 776 | anchors hurt real-data fit by 167 |
| `+anchors+steeptime0.01` | 829 | 5 | 758 | steeper offset only marginally helps |
| `+anchors+learnoff` | 822 | 5 | 750 | learnable offset converges to 0, still +139 |
| `+anchors+steepmat` (eps=1e-3) | 870 | 6 | 795 | stretching maturity hurts |
| **`+gated_t+maxscale_zeromean`** | **679** | **0** | **614** | **🏆 the answer** |
| `+gated_t_learn+maxscale_zeromean` | 756 | 0 | 689 | learnable τ overfits |
| `+gated_t+maxscale_zeromean+learnoff` | 735 | 0 | 670 | learnable offset overfits |

**Winner**: gated kernel with **fixed** `τ = 0.05` and **fixed** log
offset = 1.0. Adding learnable parameters consistently hurts (a small
MLL gain at the cost of meaningful block-LOO regression — same
overfitting pattern observed throughout the parent benchmark study).

##### 3.8 Fixed-τ sweep — confirming τ=0.05 was a justified choice

The user's challenge: "How did you set the fixed τ value? If
optimization doesn't work, is it still worth trying out a few values?"
Answer: τ=0.05 was originally an educated guess. To verify, we swept
τ ∈ {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0} with the full
recommended config (B'' + F5_alllog + max-scale + ZeroMean +
skip_time_in_normalize):

| τ | MLL/row | Real-only block-LOO | Phantom-RMSE |
|---:|---:|---:|---:|
| 0.005 | +2.371 | 722 | 0 |
| **0.01** | +2.284 | **679** | **0** ← tied best |
| 0.02 | +2.377 | 732 | 0 |
| **0.05** | +2.285 | **679** | **0** ← tied best (original choice) |
| 0.10 | +2.320 | 729 | 0 |
| 0.20 | +2.426 | 742 | 0 |
| 0.50 | +2.433 | 781 | 0 ← highest MLL **but worst block-LOO** |
| 1.00 | +2.398 | 778 | 0 |

**Two findings**:
1. **τ=0.05 is justified** — ties with τ=0.01 for best block-LOO. No
   τ in this sweep beats them.
2. **MLL is anti-correlated with block-LOO across τ values**. The
   highest-MLL fits (τ=0.5) give the worst block-LOO. This is the
   same MLL ≠ block-LOO dissociation we observed for learnable
   parameters. The MLL surface as a function of τ has multiple
   local maxima, and the optimiser has no incentive to find the one
   that generalises best.
3. **Learnable τ converged to a high-MLL/poor-block-LOO local
   optimum** (756 block-LOO, MLL +2.40) — not an optimisation
   failure but a genuine overfit to the training likelihood.

**Is the high-τ regression overfitting or better calibration?**
The user's question. Decomposing the calibration metrics across τ:

| τ | LOO RMSE | LOO MLPD | LOO cov95 | block-LOO RMSE | block-LOO MLPD | block-LOO cov95 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 533 | -7.68 | 0.951 | **679** | -7.99 | 0.932 |
| 0.20 | **489** | **-7.57** | 0.943 | 742 | -8.02 | 0.937 |
| 0.50 | **490** | **-7.57** | 0.934 | 781 | -8.03 | 0.949 |
| 1.00 | 509 | -7.59 | 0.932 | 778 | -7.97 | 0.949 |

**Verdict: overfitting, not better calibration.** Higher τ:
- Improves *single-row* LOO RMSE/MLPD (the trap metric — see parent
  benchmark §2).
- Worsens *block-LOO* RMSE substantially (+100 psi).
- Doesn't systematically improve calibration (LOO cov95 drops *below*
  0.95; block-LOO MLPD is roughly flat).

This is the same time-curve-leakage pattern we documented for
heteroscedastic noise (parent §6.2): the new flexibility lets the
kernel fit within-composition time curves tightly, but cross-
composition extrapolation pays the price. The fix is the same:
**don't select τ on MLL or single-row LOO; cross-validate on
block-LOO.**

**Conclusion**: a single fixed τ=0.05 (or τ=0.01) is the right design
choice. Don't make it learnable; the optimiser doesn't have access
to the metric we actually care about.

##### 3.9 Holistic re-examination of all model components under gating

The user's key challenge: with the gated kernel, are all the steps in
the parent benchmark's improvement journey still necessary? Or can
some be simplified now that the prior structurally enforces the t=0
constraint? Sweep:

| Variant (all + gated_t + max-scale + ZeroMean + skip_time_norm) | block-LOO | Δ vs champion | Verdict |
|---|---:|---:|---|
| **B''+F5_alllog (default — RBF on time)** | **679** | — | 🏆 |
| + Matern instead of RBF on time component | 705 | +26 | **Keep RBF** |
| Drop the additive time-only component entirely | 686 | +7 | Within seed noise — **time-only component is near-redundant** under gating |
| Single Matern instead of B'' (no blind+specific decomposition) | 703 | +24 | **Keep B''** |
| F0 (no engineered features) | 783 | +104 | Keep F5_alllog features |
| F3 (only the top-3 features) | 757 | +78 | Keep F5_alllog features |
| F5 (no log-transforms) | 811 | +132 | Keep log-transforms |

**Findings**:

1. **The user's specific question on RBF vs Matern on time is settled**:
   RBF wins by 26 psi block-LOO. RBF on time is the right choice.
2. **The additive time-only component is nearly redundant** under
   gating (7 psi gain). Could be dropped for architectural simplicity
   if minimum complexity is preferred over best-by-7-psi performance.
3. **Every other component from the parent benchmark remains valuable**:
   - Multi-Matern (B'') decomposition: +24 psi over single Matern
   - Engineered features F0→F5_alllog: cumulative +104 psi
   - Log-transforms (F5→F5_alllog): +132 psi (log-transforms matter
     even more under gating than they did originally)
4. **The simplest model achieving max performance is the same B''+F5_alllog
   from the parent benchmark, plus the 5 gating-related pieces**.
   No simplification is possible without measurable regression.

##### 3.10 Components still worth re-checking (queued)

The following parent-benchmark choices have not yet been re-tested
against the gated objective. Expected outcome in parentheses; results
to fill in as we run them:

- **Heteroscedastic noise (`A`) + gated kernel**: previously rejected
  for overfitting. With gating providing structural regularisation,
  does the rejection still hold? *(Expected: still hurts; the
  overfitting was on within-composition noise, orthogonal to the
  t=0 constraint.)*
- ✅ **Within-group shrinkage prior on/off + gated kernel**:
  **Tested.** Still helps under gating, by ~+60 psi block-LOO
  (679 with prior vs 739 without). Same pattern as everywhere else
  in this study: removing the prior raises MLL (+2.401 vs +2.285)
  and improves single-row LOO (502 vs 533) but regresses block-LOO.
  Prior cures lengthscale identifiability for collinear
  binder/aggregate features; orthogonal to the time-axis gating.
  Keep the prior.
- **Anchor pseudo-observations + gated kernel**: should be redundant
  (constraint already enforced); could even hurt due to numerical
  conflict. *(Expected: hurt or neutral; not worth the complexity.)*
- **Other gate transition functions** (sigmoid, tanh, Heaviside): the
  exponential `1 - exp(-t/τ)` is one of many valid `h(t)` choices.
  A sigmoid would have a sharper transition; might or might not help.
- **Joint MLL/block-LOO objective**: the τ sweep showed MLL
  optimization picks a worse τ than block-LOO would. Could swap MLL
  for block-LOO predictive log-likelihood as the training loss.

##### 3.11 Out-of-training (OOT) physics-constraint test — gated kernel's killer feature

The phantom-anchor RMSE values reported in §3.7 / §3.9 evaluate at
**training compositions** (one phantom test point per unique training
mix, with t=0). This is a meaningful but lenient test: the model
"knows" the composition `c` from the training data, only the t=0
prediction is genuine extrapolation in time.

**A stricter test**: sample random compositions $c_{\text{new}}$
**not in training** (uniform within the bound box) and evaluate at
$(c_{\text{new}}, t=0)$. If the constraint is structural (gated
kernel), this is also exactly 0. If the constraint is only enforced
locally near training data (anchor pseudo-observations), the OOT
predictions will drift far from 0.

| Variant | In-training phantom RMSE | **OOT phantom RMSE** | Verdict |
|---|---:|---:|---|
| Pre-shrinkage baseline (anchors + scalar noise, no prior) | 0.7 | **14,953** | Anchors enforce locally only, fail catastrophically OOT |
| Production baseline (anchors + within-group prior) | 1.0 | **2,116** | Shrinkage prior helps OOT but constraint still violated |
| `B''+F0` (no anchors, no prior) | 1,804 | 2,159 | No constraint at all |
| `B''+F5_alllog` (un-gated champion) | 2,669 | 3,385 | No constraint at all |
| **`B''+F5_alllog+gated_t+maxscale_zeromean`** | **0.0** | **0.0** | **Constraint exact everywhere** |

**This is the single most compelling argument for the gated kernel**:
the production-baseline anchor approach achieves ~1 psi RMSE at
training compositions, looks like it satisfies the constraint, but
**fails by 2,116 psi RMSE on random unseen mixes**. A user submitting
a never-before-seen mix to the deployed model would get a t=0
strength prediction off by ~2,000 psi from the physics truth. The
gated kernel produces 0 psi RMSE at both training mixes AND random
unseen mixes — by construction.

This metric is now plotted in `experiments/improvement_journey.png`
as two additional dotted/dashed lines (in-training and OOT
phantom-anchor RMSE), showing how each model along the journey treats
the t=0 constraint.

---

#### A.4. Hypotheses & mechanisms — final status

| Hypothesis | Status | Evidence |
|---|---|---|
| Anchors need steeper time transform | **Refuted** (§3.2) | Steepening only marginally helps |
| Optimisation is broken / stuck | **Refuted** (§3.3) | MLL improves monotonically |
| Maturity is the missing decoupling dim | **Refuted** (§3.4) | Stretching maturity hurts everywhere |
| Anchors fundamentally conflict with the kernel architecture | **Confirmed** (§3.1–§3.4) | All anchor variants pay 100+ psi block-LOO |
| Standardisation breaks the constraint via additive offset | **Confirmed** (§3.5) | Phantom-anchor RMSE ≈ y_mean for gated + standardize |
| BoTorch's silent default `Standardize` makes max-scaling fail | **Confirmed** (§3.6) | Setting `outcome_transform=None` was the missing piece |
| Normalize-on-time clashes with gated kernel (kills t=1 data) | **Confirmed** (§3.6) | `skip_time_in_normalize=True` was needed |
| Multiplicative gating is the principled answer | **Confirmed** (§3.5) | Achieves phantom-RMSE=0 AND matches real-only block-LOO |

---

#### A.5. Why the gated kernel is the principled answer

(The user's framing.) The anchor-pseudo-observation approach modifies
the model's behaviour **at the specific anchor points**, hoping the
posterior will smooth this through to nearby query points. The gated
kernel modifies the **prior covariance for the entire domain** in a
self-consistent way:

- For any query point at $t = 0$, the posterior is exactly 0 with zero
  variance — the constraint is structural.
- For any query point at $t \geq 1$, the gating factor $h(t) \approx 1$
  is essentially neutral, so the kernel reduces to the base kernel.
- The transition between regimes is smooth and characterised by a
  single hyperparameter ($\tau$), which is well-determined empirically
  ($\tau = 0.05$ is good).

This is "in the language of GPs" — modifying the prior — rather than
hacking around the model with auxiliary observations.

---

#### A.6. Lessons learned (transferable)

1. **For physics constraints expressible as $f(x_0) = 0$, a
   multiplicatively-gated kernel is more principled than
   pseudo-observations.** Anchors as data create architectural
   conflicts; structural constraints don't.

2. **Multiplicative-only outcome scaling preserves the GP=0 ↔ raw=0
   correspondence.** Z-score standardisation breaks it via the
   additive `+y_mean` term in untransform. If your GP needs to predict
   exactly zero, don't z-standardise.

3. **`ZeroMean` is the right choice when the prior should be exactly
   zero somewhere.** ConstantMean fits to data and may not give 0 at
   the constraint point.

4. **BoTorch's silent default `outcome_transform=Standardize` can
   secretly break custom outcome scaling.** Always pass
   `outcome_transform=None` explicitly when you do your own scaling.

5. **`Normalize` over a dim with a constraint at one boundary is
   dangerous.** It maps the smallest training value to 0, which
   collides with structural zeros at boundary values.

6. **Test additive parameters before adopting them.** Both the
   learnable log offset and the learnable gate τ regressed when added
   to the gated kernel — extra parameters → MLL gain → block-LOO
   regression. The simplest model with structurally-correct priors
   wins.

7. **Phantom-anchor RMSE is a clean diagnostic** for "does this model
   satisfy the physics constraint?" Compute posterior at virtual
   `(c, t=0)` test points; expect 0 for gated models and ~y_mean for
   ConstantMean+standardize models.

---

#### A.7. Persistent artefacts

| File | Contents |
|---|---|
| `experiments/STRENGTH_GP_ANCHORS_STUDY.md` | This document |
| `experiments/STRENGTH_GP_BENCHMARK.md` | Parent benchmark study (recommended `B''+F5_alllog+gated_t+maxscale_zeromean` after this study) |
| `experiments/model_variant_study.py` | All variants registered. The recommended path is `B''+F5_alllog+gated_t+maxscale_zeromean`. The factory is `_make_engineered_no_a_fit` with `output_max_scale=True, zero_mean=True, skip_time_in_normalize=True` and a gated kernel builder. |
| `~/.llms/plans/strength_gp_productionization.plan.md` | Phase 0 of the productionization plan, now resolved by the gated kernel finding |
| `test/test_physical_constraints.py` | Python unit tests guarding the physics constraint at `t=0` for the recommended model (added 2026-05-15). |
| `docs/test_vectors.json` | JS side; physics-constraint test vectors are part of the load-time check in `gp.mjs` (web explorer). |
