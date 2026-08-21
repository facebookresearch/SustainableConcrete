VERDICT: NEEDS_REVISION

# Adversarial review — Steps 0-6 of `plans/2026-08-21-explorer-slump-readout.md`

## Summary Assessment

I fitted the slump GP, ran the plan's exporter verbatim, wrote the plan's `docs/slump.mjs`
verbatim, and ran the plan's parity test verbatim. **The GP math is correct.** Every posterior
claim in the plan holds to floating-point noise: worst relative error over the 85 golden vectors
was `6.7e-15` (mean) and `7.2e-15` (variance). I additionally generated 43 extrapolation probes
from `docs/model/compositions.json::slider_bounds` (i.e. the regime the readout actually operates
in, far outside the training hull) and parity held at `3.5e-14`. No NaNs, no sign errors, no
transposition errors, no missing mean term, no noise-ordering error.

That said, the plan is not landable as written. There is one **critical** defect that makes
Step 6's test permanently incapable of failing (verified by execution, exit code 0 with three
failing assertions), a second latent hole in the same test's helper, a guard gap in the exporter
that contradicts its own error message, a wrong acceptance number in Step 4b, and a `black`
failure in Step 1b.

Scope: Steps 0-6 only, as requested. Steps 7-16 were read for context but not reviewed.

**Environment**: `botorch 0.18.1`, `gpytorch 1.15.2`, `torch 2.13.0`, Python 3.12.

---

## Critical Issues (must fix)

### C1. Step 6a — appending to the end of `test/test_js_units.mjs` makes the test unable to fail

`/Users/sebastianament/Code/SustainableConcrete-slump/test/test_js_units.mjs` ends with a
summary-and-exit block at lines 139-143:

```js
// --- Summary ---
console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) {
  process.exit(1);
}
```

The plan says "Append to `test/test_js_units.mjs`" and the question of whether that is safe was
raised explicitly. It is not. If the block is appended after line 143, the new assertions run
*after* the summary has printed and *after* the only `process.exit(1)` has been evaluated (at
which point `failed === 0`, so it does not fire). I ran exactly this:

```
67 passed, 0 failed
✗ metric slump label: expected "mm", got undefined
✗ imperial slump label: expected "in", got undefined
✗ imperial slumpFactor (no-op): expected 1, got undefined
EXIT CODE = 0
```

Consequences:
* Step 6b ("Run — must fail") does **not** fail. The TDD red step is fake.
* `make test-js` / the `js-sync` CI job would report this file as green forever, even after a
  future regression deletes `slumpFactor`.
* The printed pass/fail totals are wrong (they exclude the new assertions).

**Exact fix**: insert the new assertions *before* the summary block, i.e. immediately after
line 137 (`assertEqual(sliderUnitLabel("Temp (C)", "imperial"), "°F", "label temp imperial");`)
and before line 139 (`// --- Summary ---`). Change Step 6a's wording from "Append to" to
"Insert into `test/test_js_units.mjs` immediately before the `// --- Summary ---` block at
line 139 (appending after it would place the assertions past `process.exit`)."

Verified with the corrected placement: 4 failures, `EXIT CODE = 1`. Proper red.

---

### C2. Step 6a — `assertClose` silently passes on `undefined`, so two of the five new assertions are no-ops

`test_js_units.mjs:34-46`:

```js
function assertClose(actual, expected, name) {
  const absErr = Math.abs(actual - expected);      // NaN when actual === undefined
  const denom = Math.abs(expected) > ATOL ? Math.abs(expected) : 1;
  const relErr = absErr / denom;                   // NaN
  if (relErr > RTOL && absErr > ATOL) { ... }      // NaN > x is false -> takes the else
  else { passed++; }                               // counted as a PASS
}
```

In the run above, only **3** of the 5 new assertions reported a failure. The two `assertClose`
ones —
`assertClose(UNITS.metric.slumpFactor, 25.4, ...)` and
`assertClose(6 * UNITS.metric.slumpFactor, 152.4, ...)` — were **counted as passes** against
`undefined`. So even with C1 fixed, the plan's chosen assertion for the actual conversion factor
does not test anything.

**Exact fix**, two parts:

1. In Step 6a, change the factor assertion to an exact comparison (verified in Node:
   `25.4 === 25.4` is `true`, so `assertEqual` is safe here):

```js
assertEqual(UNITS.metric.slumpFactor, 25.4, "metric slumpFactor (in -> mm)");
```

   Keep the derived check as `assertClose` — `6 * 25.4 === 152.39999999999998`, so `assertEqual`
   would be wrong there:

```js
assertClose(6 * UNITS.metric.slumpFactor, 152.4, "6 in reads as 152.4 mm");
```

2. Harden the shared helper so this class of bug cannot recur (add as a Step 6a sub-item):

```js
function assertClose(actual, expected, name) {
  if (!Number.isFinite(actual)) {
    failed++;
    console.error(`✗ ${name}: expected ${expected}, got non-finite ${actual}`);
    return;
  }
  ...
}
```

With (1) applied I measured 4 real failures / exit 1 in the red phase, versus 3 / exit 0 as
written.

---

### C3. Step 1 — the exporter's raw-column guard checks only the *count*, contradicting its own error message

Plan lines 234-238:

```python
    if X.shape[-1] != len(RAW_FEATURE_NAMES):
        raise RuntimeError(
            f"Expected {len(RAW_FEATURE_NAMES)} raw dims, got {X.shape[-1]}. "
            "RAW_FEATURE_NAMES is out of sync with DEFAULT_X_COLUMNS."
        )
```

The message claims to detect `RAW_FEATURE_NAMES` drifting from `DEFAULT_X_COLUMNS`, but the
check is `len() != len()`. Any *reordering* of `boxcrete.utils.DEFAULT_X_COLUMNS` passes this
check and then silently corrupts everything downstream:

* `source_dim_raw: 7` and the `X[:, 7]` source-class assertion point at the wrong column;
* `augmentSlumpInput` in `docs/slump.mjs` hardcodes `I_CEMENT=0, I_FLYASH=1, I_SLAG=2, I_HRWR=4`
  and would compute the derived feature from the wrong columns;
* `predictSlump` would still return finite, plausible-looking numbers, and the parity test would
  still pass (both sides regenerate from the same wrong artifact).

The plan spends four `isinstance` assertions guarding the *kernel* architecture ("read it off the
model and hard-fail on anything else, never assume") but leaves the column contract — the thing
the JS port actually hardcodes — unguarded. `dataset.X_columns` exists and is available; I
confirmed it returns
`['Cement (kg/m3)', 'Fly Ash (kg/m3)', 'Slag (kg/m3)', 'Water (kg/m3)', 'HRWR (kg/m3)',
'Fine Aggregate (kg/m3)', 'Coarse Aggregates (kg/m3)', 'Material Source', 'Temp (C)', 'Time']`.

**Exact fix** — replace the length check in Step 1a with:

```python
    if list(dataset.X_columns[:-1]) != RAW_FEATURE_NAMES:
        raise RuntimeError(
            "RAW_FEATURE_NAMES is out of sync with DEFAULT_X_COLUMNS: "
            f"expected {RAW_FEATURE_NAMES}, got {list(dataset.X_columns[:-1])}. "
            "docs/slump.mjs hardcodes column indices (I_CEMENT=0, I_FLYASH=1, "
            "I_SLAG=2, I_HRWR=4) and slump.json hardcodes source_dim_raw=7; "
            "both must be revisited with this list."
        )
    assert RAW_FEATURE_NAMES.index("Material Source") == 7
    assert RAW_FEATURE_NAMES.index("HRWR (kg/m3)") == 4
```

(`dataset.X_columns[:-1]` drops `"Time"`, which `slump_data` already strips.)

---

### C4. Step 4b — the expected assertion count is wrong (172 vs 173)

Plan line 740: `# expect: "All 172 slump assertions passed."`

I ran the plan's `test_js_slump.mjs` verbatim against the plan's `slump.mjs` verbatim and the
artifacts the plan's exporter produced:

```
worst rel err mean: 6.684e-15  var: 7.158e-15
All 173 slump assertions passed.
```

85 vectors × 2 = 170, plus 2 from the `hrwr_binder` block (one `assertClose`, one `else passed++`),
plus 1 from the zero-binder block = **173**.

This matters because Step 4b is written as an acceptance criterion; an executor seeing 173 will go
hunting for a phantom extra assertion.

**Exact fix**: change Step 4b to `# expect: "All 173 slump assertions passed."`

---

### C5. Step 1b — `black --check` fails on the Step 1a source as written

Plan lines 287-288 use three spaces before the inline comment:

```python
        train_X_aug = model.train_inputs[0].detach()   # [n, 10], post-transform
        train_Y_std = model.train_targets.detach()     # [n], standardized
```

Running the repo's pinned formatter (`black==26.3.1`, `line-length = 88`) on the extracted block:

```
-        train_X_aug = model.train_inputs[0].detach()   # [n, 10], post-transform
-        train_Y_std = model.train_targets.detach()     # [n], standardized
+        train_X_aug = model.train_inputs[0].detach()  # [n, 10], post-transform
+        train_Y_std = model.train_targets.detach()  # [n], standardized
would reformat
1 file would be reformatted.
```

Step 1b explicitly runs `black --check` and would fail. (`flake8` with the repo's
`FLAKE8_SELECT` list is clean — 0 errors.)

**Exact fix**: use two spaces before both inline comments in Step 1a.

---

## Suggestions

**S1. Step 0a's second search string is line-wrapped in the source and will not match literally.**
`boxcrete/slump_model.py:10-11` reads:

```
time-independent (slump is measured pre-cure), uses a single Matern
kernel rather than the multi-Matern + gated-time decomposition, and
```

The plan asks to replace `uses a single Matern kernel rather than the multi-Matern + gated-time
decomposition` — one line. Note the newline between `Matern` and `kernel` in the step text, or
split into two edits. (The first replacement, of line 6, matches exactly. The three inserted
docstring lines are 77/79/74 chars, all under the repo's `max-line-length = 88` — no E501 risk.)

**S2. Make the `model.eval()` dependency explicit in the exporter.** The plan's comment
`# [n, 10], post-transform` is correct only because `model.eval()` runs first.
`botorch/models/gp_regression.py:183-184` passes the **raw** `train_X` to `ExactGP.__init__`;
`train_inputs` only becomes the 10-dim transformed tensor because `botorch.models.model.Model.eval()`
(`model.py:246-249`) calls `_set_transformed_inputs()` → `input_transform.preprocess_transform(...)`
→ `set_train_data(...)`. Move `model.eval()` above the extraction block or add, inside the
`torch.no_grad()` block:

```python
        if train_X_aug.shape[-1] != X.shape[-1] + 1:
            raise RuntimeError(
                "train_inputs are not post-input-transform "
                f"({tuple(train_X_aug.shape)}); model.eval() must run before "
                "reading them (BoTorch stores raw train_X until eval())."
            )
        if train_X_aug.shape[-1] != len(lengthscales):
            raise RuntimeError("X_train width disagrees with ard_num_dims.")
```

Today Step 2b's shell assert catches this, but only after the artifact has been written.

**S3. The golden vectors never leave the training box, which is not where the readout lives.**
`_write_test_vectors` draws probes from `X.min(dim=0)` … `X.max(dim=0)`. The explorer's sliders go
far outside that (`slider_bounds`: Slag `0-1198.67` vs training `0-367`; HRWR `0-13.33` vs `0-4.63`;
Fine Agg `423-2357` vs `423-992`). I generated 43 probes from `slider_bounds` plus three corner
cases (all-zero binder, all-max, all-zero) and verified parity independently at `3.5e-14`, so the
math is fine — but the *shipped* test would not have caught a regression there. Worth adding ~8
slider-bound probes and the all-zero-binder corner to `probes`. For reference, in that regime the
model is at its prior: mean ∈ [6.831, 6.932] in, variance ∈ [10.885, 10.901] in² (2σ ≈ 6.6 in),
which corroborates the plan's fact #6 and Step 16c-5.

**S4. Match the strength exporter's JSON serialization.**
`experiments/regenerate_strength_json.py:465` uses `json.dump(out, f, separators=(",", ":"))`
for the model artifact and `indent=2` for the vectors file. The plan uses bare `json.dumps(out)`.
Measured on the real artifact: 10,944 bytes compact vs 11,696 default. Cosmetic, but the plan
claims to carry the strength exporter's conventions "verbatim in spirit".

**S5. Step 5 should also annotate `log_hrwr_binder`.** `docs/feature_registry.mjs:42` has the same
`+ 1.0` divergence as line 41 and is the next thing a reader reaches for.

**S6. `normalize()`'s zero-span branch diverges from Python.** `span === 0 ? 0 : ...` returns 0
where BoTorch's `Normalize` would divide by zero and produce ±inf. This is unreachable in practice
because `boxcrete.utils.derive_bounds_from_X` widens any column narrower than `1e-8` to
`[min, min + 1.0]` before `Normalize` ever sees it — say so in the comment, otherwise it reads like
a deliberate behavioural difference between the two ports.

**S7. The variance clamp cannot do what its comment says.** In `predictSlump` the clamp runs after
`varStd += params.noise`, so with `noise ≈ 0.2444` a slightly-negative latent term is absorbed
silently rather than clamped. If the intent is "the latent variance must not go negative", clamp
before adding noise:

```js
  let varStd = outputscale;
  for (let i = 0; i < n; i++) varStd -= v[i] * v[i];
  if (varStd < 0) varStd = 0;
  if (params.variance_includes_aleatoric) varStd += params.noise;
```

**S8. `initSlumpModel` has no PSD/NaN guard, unlike `initStrengthModel`.** `docs/gp.mjs:367-396`
wraps `cholesky` in a jitter-escalation loop with an explicit `Number.isFinite(L[i][i])` check;
the plan's `initSlumpModel` has neither, and the naive `cholesky` at `gp.mjs:243` returns silent
`NaN`s from `Math.sqrt` of a negative pivot. Risk is genuinely low — I measured
`cond(K) = 6.8e4`, `cond(K + σ²I) = 65` at the fitted `σ² = 0.2444`, and `4.7e4` even at the
`LogTransformedInterval(1e-4, 1.0)` noise floor — but a three-line finiteness check after
`cholesky(K)` costs nothing and turns a silent all-`NaN` readout into a loud error.

**S9. `source_dim_raw` is exported but never consumed, and the derived-feature indices are
hardcoded in JS.** Either have `docs/slump.mjs` / `docs/ui.mjs` read `params.source_dim_raw`, or
drop the field. Better still, export the four indices the derived feature depends on
(`{"cement": 0, "fly_ash": 1, "slag": 2, "hrwr": 4}`) and have `augmentSlumpInput` read them —
that would make C3 structurally impossible rather than assertion-guarded.

**S10. The parity test should pin that the two artifacts came from the same fit.** Both files carry
`model_name`; the test never compares them. One line in Step 3a prevents comparing a fresh
`slump.json` against a stale `slump_test_vectors.json` (the exact failure the exporter docstring
warns about):

```js
if (params.model_name !== golden.model_name) {
  throw new Error(
    `slump.json (${params.model_name}) and slump_test_vectors.json ` +
    `(${golden.model_name}) disagree — regenerate both together.`
  );
}
```

**S11. The parity tolerance has ~8 orders of magnitude of slack.** `RTOL = 1e-6` against a measured
worst-case of `7.2e-15` in-hull and `3.5e-14` extrapolated. `1e-9` would still be safe and would
actually catch a real regression (e.g. an off-by-one lengthscale) instead of waving it through.

**S12. (Step 14, out of scope but caused by Step 1.)** `_write_test_vectors` draws probes with
`torch.rand` after `torch.manual_seed(SEED + 1)`. The drift checker the plan describes compares
`expected_mean` / `expected_variance` positionally, which is only meaningful if the probe *inputs*
are identical between the committed and freshly-generated files. Have `check_slump_test_vectors_json`
compare the `input` arrays exactly before comparing predictions, so an RNG change fails as
"inputs differ" rather than as a bogus prediction drift.

---

## Verified Claims

All of the following were checked by running code, not by reading. Fit performed with
`torch.manual_seed(0)`; reproduced across three independent processes with identical
`noise = 0.244367`, `mean_constant = 0.026112`, `y_mean = 6.8098`, `y_std = 2.9598`.

### The posterior math in `docs/slump.mjs` (Step 4) — **correct**

| Claim | Verdict | Evidence |
|---|---|---|
| GPyTorch `RBFKernel` forward is `exp(-0.5 · Σ((x-x')/ℓ)²)` | **TRUE** | `k(a,b) = 0.00012443605451765055`; manual `exp(-0.5·Σ z²) = 0.00012443605451765055`. Bit-identical. |
| `k(x, x) == outputscale == 1.0` exactly | **TRUE** | `float(k(a,a)) == 1.0`. Matters because `initSlumpModel` writes `K[i][i] = outputscale + noise` and `predictSlump` starts `varStd = outputscale`; both are exact. |
| `model.train_targets` is in STANDARDIZED space | **TRUE** | shape `[61]`, mean `-1.97e-16`, std `1.0000000`; raw `Y` mean/std `6.8098 / 2.9598`. |
| `model.likelihood.noise` is in standardized space | **TRUE** | `posterior(obs_noise=True).variance − posterior(obs_noise=False).variance = 2.1407127406448145` for every probe, exactly `noise · y_std² = 0.244367 · 2.9598² = 2.1407127406448145`. |
| `ConstantMean` constant is in standardized space | **TRUE** | `0.026112`; the manual reconstruction below matches only with the constant applied pre-destandardize. |
| `α = K⁻¹(y − m)` with `μ* = m + k*ᵀα` is the right GPyTorch/BoTorch convention | **TRUE** | Manual reconstruction using `train_inputs`/`train_targets`/`noise`/`constant` reproduced `model.posterior().mean` to 1e-15: `[9.5904, 9.4685, 9.3950, 9.5154, 9.3435]` both ways. (ConstantMean means `m(X_train) == m(x*)`, so the "mean over TRAIN inputs" subtlety collapses to the same scalar — but the plan's formula is the general-correct one, not a lucky coincidence.) |
| The back-substitution loop `acc -= L[j][i]*alpha[j]` correctly solves `Lᵀα = z` | **TRUE** | `cholesky` at `docs/gp.mjs:243-256` returns a row-major 2D lower-triangular `L` with `L[i][j]` for `j ≤ i` and explicit zeros above. `(Lᵀ)[i][j] = L[j][i]`, so the indexing is right. It is also character-for-character the loop already in `initStrengthModel` (`gp.mjs:402-406`). |
| `solveTriangularLower(L, b)` (`gp.mjs:438-449`) is plain forward substitution and is used correctly | **TRUE** | It does not do anything the plan re-implements; the plan uses it for `z = L⁻¹(y−m)` and `v = L⁻¹k*`, both correct. It reads `n = b.length`, so passing an `Array(n)` `kStar` is fine. |
| `cholesky` and `solveTriangularLower` are already exported | **TRUE** | `docs/gp.mjs:613`. Import from Node succeeded with no side effects. |
| BoTorch `Standardize` untransforms variance as `var · stdvs²` | **TRUE** | Manual `var_s · y_std²` matched `posterior.variance` exactly for both `observation_noise` settings. |
| **`posterior(observation_noise=True)` adds noise in standardized space BEFORE untransform** (so it scales by `stdvs²`), matching the plan's ordering | **TRUE** | The variance delta is exactly `noise · y_std²`, not `noise`. If BoTorch added noise post-untransform the delta would have been `0.2444`. |
| **End-to-end**: plan's `slump.mjs` reproduces `model.posterior(X, observation_noise=True)` | **TRUE** | 85/85 golden vectors, worst rel err `6.7e-15` (mean) / `7.2e-15` (variance). Plus 43 out-of-hull slider-bound probes at `3.5e-14`. No NaN, no non-finite. |

### `model.train_inputs[0]` (Step 1)

**POST-input-transform — but only because `model.eval()` runs first.** Measured: shape
`[61, 10]`, per-dim min all `0.0`, per-dim max all `1.0` — normalized, not raw kg/m³.
Mechanism confirmed in source: `botorch/models/gp_regression.py:183-184` passes the **raw**
`train_X` to `ExactGP.__init__`; `botorch/models/model.py:246-249` (`Model.eval()`) calls
`_set_transformed_inputs()`, which does `set_train_data(input_transform.preprocess_transform(...))`.
`Model.train(mode=True)` reverts it. The plan's exporter calls `model.eval()` at the right point,
so the claim holds — see S2 for hardening.

### `Normalize` bounds (Step 1)

* `norm.bounds` gives `[lower; upper]` rows: **TRUE**. Measured
  `bounds[0] = [80, 0, 0, 120, 0, 423, 928, 1, 10, 0]`,
  `bounds[1] = [887, 533, 367, 333, 4.63, 992, 1356, 2, 22, 0.0069415]`.
* `bounds` is a **property**, not a buffer: `torch.cat([offset, offset + coefficient], dim=-2)`
  (`botorch/models/transforms/input.py:747-749`). The registered buffers are `_coefficient` and
  `_offset`. Because `bounds` is derived, it is **always in sync** — the sync concern raised in the
  brief does not apply. `learn_bounds` is `False` (explicit bounds were supplied), so it cannot
  drift during fitting either.
* Round-trip is exact for this artifact: JS recomputes `span = upper[d] − lower[d]` and gets
  `[807, 533, 367, 213, 4.63, 569, 428, 1, 12, 0.006941529235382308]`, matching
  `norm.coefficient` exactly. (In general `(offset + coefficient) − offset` can differ by an ulp;
  immaterial at the measured 1e-15 parity.)

### Test-vector generation (Step 1 `_write_test_vectors`)

* Passing **raw 9-dim** `X` to `model.posterior()` is correct: `GPyTorchModel.posterior` applies
  `self.transform_inputs(X)` in eval mode (`transform_on_eval = True` on both `AppendDerivedFeatures`
  and `Normalize`). Ran it; produced sane values.
* **No double-transform hazard.** `train_inputs` are transformed once at `eval()` and guarded by
  `_has_transformed_inputs`; test inputs are transformed once in `posterior()`. I deliberately fed
  an already-transformed 10-dim tensor back through `model.input_transform` and it raised
  `BotorchTensorDimensionError: Wrong input dimension. Received 11, expected 10` — i.e. a
  double-transform would fail loudly, not silently.
* Produced exactly **85** vectors, each with a 9-element `input`. Step 2b's assertions all pass.

### `load_concrete_strength(Y_columns=SLUMP_Y_COLUMNS)` and `.slump_data` (Step 1)

**Signature and tuple order correct.** `load_concrete_strength` takes `Y_columns` as a keyword
(`boxcrete/utils.py:372`); `SLUMP_Y_COLUMNS = ["GWP", "Strength (Mean)", "Slump (in)"]` at line 55.
`slump_data` returns `(X, Y, Yvar, X_bounds)` — measured shapes
`torch.Size([61, 9]) torch.Size([61, 1]) torch.Size([61, 1]) torch.Size([2, 9])`, dtype `float64`.
Matches the plan's unpacking `X, Y, Yvar, _bounds = slump`.

### The `observed != SUPPORTED_SOURCE_CLASSES` assertion (Step 1)

**Index 7 is Material Source. Confirmed twice**: `dataset.X_columns[7] == "Material Source"`, and
`docs/model/compositions.json::column_names[7] == "Material Source"` (the catalog the UI feeds
`predictSlump`). `sorted({int(v) for v in X[:, 7].tolist()}) == [1, 2]`, so the assertion passes.
Plan fact #5 also verified: the 149-row catalog is `Counter({0: 69, 2: 53, 1: 27})` — 69 mortar
mixes get `n/a`.

### Units (Step 6)

* `test/test_js_units.mjs` has `assertClose(actual, expected, name)` at line 34 and
  `assertEqual(actual, expected, name)` at line 48 — the plan's line references and call
  signatures are **correct**, and there is indeed no boolean `check()` helper.
* Appending at the end of the file does **not** work — see C1/C2.
* `"mm"` / `"in"` labels and the `25.4` factor match `boxcrete/units.py`:
  `slump_label()` returns `"in"` for imperial and `"mm"` for metric (lines 104-115);
  `convert_slump(v, METRIC) = v * INCHES_TO_MM` (lines 70-87);
  `SLUMP_DISPLAY_SCALE = INCHES_TO_MM if metric else 1.0` (line 47). The inverted factor direction
  relative to `massFactor` is correct.

### Other plan facts spot-checked

| Fact | Verdict |
|---|---|
| #1 kernel is ARD `RBFKernel`, `lengthscale.shape == (1, 10)`, no `outputscale` | **TRUE** (`hasattr(covar_module, "outputscale") == False`) |
| #2 `mean_module` is `ConstantMean`, constant ≈ `0.0261` standardized | **TRUE** (`0.026112`) |
| #3 `Standardize(1)`, `means ≈ 6.8098`, `stdvs ≈ 2.9598` | **TRUE** |
| #4 `AppendDerivedFeatures` uses `binder.clamp(min=1.0)`; `FEATURE_FNS.hrwr_binder` uses `+ 1.0` | **TRUE** (`boxcrete/features.py:276-283` vs `docs/feature_registry.mjs:41`) |
| Step 2a's expected console numbers (`X=(61,9)`, `d_aug=10`, `n=61`, `noise=0.244367`, `mean_constant=0.026112`, `y_mean=6.8098`, `y_std=2.9598`) | **TRUE**, reproduced exactly |
| Step 0b's `pytest test/test_models.py -k slump` selects real tests | **TRUE** (10 collected, incl. `test_fit_slump_gp_*` and `test_slump_loo_r2`) |
| Step 1a passes `flake8` with the repo's `FLAKE8_SELECT` | **TRUE** (0 errors) — but fails `black`, see C5 |
| Artifact size claim (~15 KB) | Measured 11,696 bytes as written (10,944 with compact separators) — under budget |
| No enumeration test would break on two new `docs/model/*.json` files | **TRUE** (`test_export_artifacts.py`, `test_data_freshness.mjs`, and `check_artifacts_drift.py` all name files explicitly) |
