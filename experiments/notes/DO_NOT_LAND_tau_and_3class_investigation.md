# DO-NOT-LAND — Investigation notes: 3-class explorer + gate-τ parity

> **This file is intentionally in a separate "do-not-land" commit.** It records
> the experiments, measurements, and reasoning behind the accompanying landable
> changes so we can reconstruct *why* we made each decision, without polluting
> the main history with scratch. Drop this commit before landing; keep it on the
> branch for reference.

Repo: `SustainableConcrete` · Branch: `current_pr_work` · Base: `d4f64a6`
("Productionize 3-class Hamming strength GP").

---

## 0. TL;DR of decisions

| Decision | Outcome | Where it lives (landable) |
|---|---|---|
| Explorer 3-class support | Rebuild catalog from current data (149 candidates, classes {0:69, 1:27, 2:53}); 3-way Source A/B/C selector; class-aware insights; smooth click transitions | `docs/ui.mjs`, `docs/generate_mix_analyses.py`, `experiments/regenerate_compositions.py`, `experiments/regenerate_all_artifacts.sh`, regenerated `docs/model/*.json` |
| `lengthscale-identifiability` test failure | Model was correct; tests were pre-v5-stale. Both Matern subkernels exclude the categorical source dim → 16 lengthscales each. Fixed tests to read `active_dims`. | `test/e2e/lengthscale-identifiability.spec.ts`, `test/test_lengthscale_identifiability.py` |
| Gate `GATE_TAU` | Finalized **0.10** (was 0.05 on `main`; an in-progress bump to 0.2 broke JS↔Python parity). 0.10 is the best point that keeps EXACT parity. | `boxcrete/features.py` (+ kernel/likelihood defaults & comments) |

---

## 1. Explorer 3-class support

**Problem.** Model + `gwp.json` were already 3-class, but the explorer was stuck at 2:
`compositions.json` had Source ∈ {0,1} only (stale, built from pre-swap 2-class data by
a since-deleted `scripts/export_model.py`), `ui.mjs` hardcoded a binary toggle, and
click-to-mix linearly interpolated the *categorical* Material Source dim → mid-animation
prediction collapse (fractional class fed to the Hamming kernel).

**What we did.**
- New builder `experiments/regenerate_compositions.py` (ported from `git 711ac20:scripts/export_model.py::export_compositions`, adapted to the current stack). Rebuilds the catalog from `load_concrete_strength().gwp_data`. Result: **149 candidates**, `slider_bounds["Material Source"] = {0,2}`, classes **{0:69, 1:27, 2:53}**. Cost computed from `DEFAULT_COST_COEFFICIENTS` (class-agnostic, model predicts −cost). Strength/GWP/pareto left as placeholders that the existing `.mjs` step overwrites.
- Wired builder into `regenerate_all_artifacts.sh` before the `.mjs` strength/gwp/pareto step, then `generate_mix_analyses.py`.
- `ui.mjs`: replaced the 2-button toggle with a loop over distinct catalog source classes (labels {0:"Source A",1:"Source B",2:"Source C"}, fallback [0,1,2]); `setComposition` loops all `.toggle-btn`; class-aware materials insights (Set 1/2/3 supplier descriptions); **click-discontinuity fix** — `animateToComposition` no longer lerps `COL_MS`, it routes source changes through the existing `triggerMaterialSourceTransition` curve-blend and pins the categorical dim.
- `generate_mix_analyses.py`: added the class-2 ("Source C") narrative branch.

**Naming caveat (flag for data owner).** Supplied description lists Set 3 as C28–C54,
but `mix_naming.py` puts class 2 = C28–C80. We treated class 2 = Set 3 (Amrize) for all
class-2 mixes (assuming C55–C80 are more Set-3 batches).

---

## 2. `lengthscale-identifiability` — root cause & fix

**Symptom.** `test/e2e/lengthscale-identifiability.spec.ts` failed:
`matern_specific: expected 17 lengthscales, got 16` — present on a clean `HEAD` checkout
too (pre-existing).

**Root cause.** The test encoded the **pre-v5** architecture (a continuous-ARD source
coordinate, so `matern_specific` spanned all 17 augmented dims). The v5 3-class migration
replaced that with a **categorical Hamming source kernel**, so Material Source (aug dim 7)
is excluded from **both** Matern subkernels → each has 16 lengthscales. Confirmed by the
served artifact: both subkernels' `active_dims = [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16]`.
The passing Python freshly-fit test already asserted the 16-dim reality; the committed-JSON
Python test passed only by luck (rail-checked by position without a length assert →
silently mislabeled dims; would spuriously fail if an engineered feature ever railed).

**Fix (correct behavior, made robust).** Both tests now map lengthscales → feature names
via the served **`active_dims`** and assert `len(lengthscales)==len(active_dims)`. This is
self-describing and won't rot if the dim layout changes again.

---

## 3. Gate-τ parity investigation (the main experiment)

### 3.1 Context
`main` shipped `GATE_TAU=0.05`. An in-progress change bumped it to **0.2** (documented
rationale: reduce early-time, t<1 day, negative overshoot in the data-free window). But
regenerating artifacts at 0.2 made the JS↔Python freshness/parity tests fail — the
"τ=0.2 parity floor."

### 3.2 Root cause (measured)
The JS explorer (`gp.mjs::initStrengthModel`) and Python (`gp.posterior`, baking
`test_vectors.json`) **each independently Cholesky-factorize** the 670×670 training kernel.

Measured on the τ=0.2 fit (`/tmp/measure_cond.py`):
- `cond(K_lik) = 1.85e5`, `eig_min = 9.45e-4` (> 0 → **PD**), `eig_max = 174.8`, noise σ² = 9.45e-4.
- `linear_operator` default `cholesky_jitter` (double) = **1e-8**, but psd_safe_cholesky
  tries jitter-free first and **succeeds** (K is PD) → **both sides use zero jitter.**
- `alpha` sensitivity: a diagonal perturbation of `2.66e-9` shifts `alpha` by `~1.0e-3`
  (scales ~linearly with the perturbation ⇒ amplification ≈ `cond`).

So the drift is **not** a jitter bug — it's cross-implementation float64 non-reproducibility
(torch/LAPACK vs the JS triple-loop Cholesky) amplified by `cond ≈ 2e5`. Raising τ grows the
fitted outputscale (compensating for the damped early-time gate, `h(t=1)≈0.65` at τ=0.2 vs
`≈0.99` at τ=0.05), which worsens conditioning. The drift is therefore **concentrated at
early times** (all failures at t ≤ 3 days; **zero at day 28**).

### 3.3 The τ sweep (deterministic: edit `GATE_TAU`, fresh regen each time)

| τ | `test_js_gp` (rtol=1e-4, atol=1e-2) **← true parity gate** | freshness (mean, 1 psi) | early overshoot: min pred, t<1d | bLOO loss (lower=better) | monotonicity worst dropdown | outputscale (specific) |
|---|---|---|---|---|---|---|
| 0.05 | **296/296 ✅** | ✅ | −1956.8 psi | −1.6512 | 0.0 psi | 0.0068 |
| **0.10** | **296/296 ✅** | ✅ | −1302.0 psi | −1.6509 | 0.0 psi | 0.0077 |
| 0.15 | ❌ 222/296 (9 mean, 28 var fails; max abs 1.9e2 psi² var / ~0.5 psi mean) | ✅ (mean drift <1 psi) | −975.3 psi | **−1.6529 (best)** | 5.8 psi | 0.0086 |
| 0.20 | ❌ 220/296 (32 mean, 6 var fails; max mean drift 3.51 psi) | ❌ (3.51 psi) | −808.9 psi | −1.6528 | 7.6 psi | 0.0087 |

**Key reads.**
- **The true parity gate is `test_js_gp` (rtol=1e-4/atol=1e-2 psi + variance), not freshness
  (mean-only, 1 psi abs).** Freshness passed at 0.15 while `test_js_gp` failed — freshness is
  too coarse to be the parity guard. (This tripped us once: an early recommendation of 0.15
  was based on freshness alone and was wrong.)
- Parity holds **exactly only for τ ≤ 0.10**.
- Overshoot shrinks monotonically with τ, **diminishing returns**: 0.05→0.10 cuts it ~33%
  (−1957→−1302); the big remaining gains (0.15, 0.2) are where parity breaks.
- bLOO is essentially **flat** across the sweep (−1.6509…−1.6529, within noise).
- Monotonicity dropdown grows with τ but all pass (<100 psi threshold).

### 3.4 Options considered
1. **τ=0.10** *(chosen)* — best parity-clean point. Full suite green, no code/test/tolerance
   changes, ~33% overshoot reduction vs 0.05, bLOO tied.
2. **τ=0.15/0.2 + relax `test_js_gp`/freshness tolerances** — more overshoot reduction + best
   bLOO, but deliberately weakens the JS-port guardrail. Drift is physically negligible
   (<0.5 psi mean at 0.15; ~3.5 psi at 0.2; variance <0.1%), so defensible, but we preferred
   not to loosen a correctness guard.
3. **Ship Python `alpha` in `strength.json`** — makes the *mean* exact by construction, but
   **does not** fix `test_js_gp` because the *variance* channel (`k*ᵀK⁻¹k*`) also drifts and
   needs `L`. Also **not** a load-time win (see §3.5).
4. **Match jitter** — moot: both sides already use zero jitter (K is PD); the divergence is
   backend fp, not jitter.

### 3.5 Load-time note (correcting an earlier claim)
Measured `initStrengthModel + initWASM` ≈ **61–72 ms** (n=670), dominated by building K and
the O(n³) Cholesky.
- Shipping **`alpha`** (670 numbers) fixes the mean but the variance path still needs `L`, so
  the O(n³) factorization stays → **no load-time win.**
- Shipping **`L`** would skip the factorization but is n² ≈ **448,900 floats (~3.6 MB raw)**
  vs the current 184 KB `strength.json` → download cost likely **exceeds** the ~60 ms compute
  saving, especially on mobile. So "ship alpha/L for faster load" is **not** a clear win.

### 3.6 Reproduce
```bash
# per-τ: edit boxcrete/features.py GATE_TAU, then
python experiments/regenerate_strength_json.py
node   experiments/augment_test_vectors_with_gwp_cost.mjs
python experiments/regenerate_compositions.py
node   experiments/regenerate_compositions_strength_predictions.mjs
node   test/test_js_gp.mjs            # ← true parity gate
node   test/test_data_freshness.mjs   # coarse mean-only check
```
Conditioning probe: `/tmp/measure_cond.py` (fits τ=0.2, prints cond(K), jitter sensitivity).
Overshoot probe: load `strength.json`+`compositions.json`, min of `predictStrengthCurve`
mean over t∈(0.02,1.0] across all catalog compositions.

---

## 4. Final verification (τ=0.10, landable state)
- **Python:** 272 passed, **100% coverage**.
- **JS:** all green — `test_js_gp` 296/296, freshness 6/6, monotonicity 0.0 psi, physics 37/37, units, parity.
- **e2e (Playwright):** 72 passed, 0 failed (desktop + mobile); Lighthouse assertions pass.
- Served locally: `http://127.0.0.1:4173/model/strength.json` → `gate_tau=0.1`; `compositions.json` → 149 candidates, MS bounds {0,2}, 3 classes.

## 5. Follow-ups / open items
- If more early-window damping is wanted (τ≥0.15), pair it with a parity fix — either relax
  `test_js_gp` to a physically-motivated tolerance, or make the JS+Python factorizations
  bit-identical (e.g., a shared fixed jitter and matched summation), then re-measure.
- Confirm the class-2 = Set 3 (C28–C80) naming assumption with the data owner.
- τ=0.2 parity-floor decision from the original plan ("Option B") is resolved here by choosing τ=0.10.
