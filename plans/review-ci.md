# CI / Build / Pipeline Review — Steps 12–16

VERDICT: NEEDS_REVISION

Scope: Steps 12–16 only (`Makefile`, `.github/workflows/*`,
`experiments/regenerate_all_artifacts.sh`, `experiments/check_artifacts_drift.py`,
`docs/model/README.md`, verification). Every claim below was checked by reading the
file at
`/Users/sebastianament/Code/SustainableConcrete-slump/<path>`; nothing was assumed
from the plan's own description.

---

## Summary Assessment

The plan's *architecture* for CI integration is sound — it correctly identifies that
`regenerate_all_artifacts.sh` is the gate that feeds `model-artifacts-coherence.yml`, that
the Makefile `JS_TESTS` list and `js-sync.yml` must move together, and that bit-equality
is the wrong bar for a non-convex fit. The 100%-coverage worry raised in the review brief
is a **non-issue** and the plan is right to ignore it.

But there are seven concrete defects that will either break CI or, worse, silently produce
zero coverage while appearing green:

* The plan's chosen drift helper (`assert_variance_close`) **physically cannot accept a
  tolerance**, and its hardcoded 2000 psi² floor makes the slump variance check vacuous by
  ~2 orders of magnitude.
* `test/e2e/visual-regression.spec.ts` holds committed **Linux full-page baselines** that
  the readout change invalidates. It `test.skip`s on macOS, so the author will never see it
  locally — it fails on the `e2e.yml` ubuntu matrix. The plan never mentions it.
* Step 14d leaves `slump_test_vectors.json` desynchronized from `slump.json`, which will
  hard-fail `make check-all` two steps later.
* Neither workflow's `paths:` filter is fully updated, so the two new CI gates won't fire on
  the changes they exist to catch.
* `check_artifacts_drift.py`'s dispatcher **skips missing files without failing**, so Step
  14d's "Expect: passes" is satisfied even if the new checkers were never wired in.
* The proposed exporter source will fail `black --check`.

Fix these seven and the CI portion is solid.

---

## Critical Issues (must fix)

### C1. `assert_variance_close` accepts no tolerance — Step 14b is unimplementable as written, and the obvious literal implementation is a no-op check

`experiments/check_artifacts_drift.py:143-160`:

```python
    def assert_variance_close(
        self,
        committed: float,
        fresh: float,
        *,
        label: str,
    ) -> None:
        self.checks += 1
        diff = abs(fresh - committed)
        delta = max(VARIANCE_PSI2_FLOOR, VARIANCE_RTOL * abs(committed))
```

`VARIANCE_PSI2_FLOOR = 2_000.0` (line 79) is hardcoded, **not** a parameter. Slump
posterior variances are in **inches²**: with `y_std ≈ 2.9598 in` and a standardized
variance in roughly `[noise, outputscale+noise] ≈ [0.24, 1.24]`, `variance` lands around
**2–11 in²**. `delta = max(2000, 0.05 · v)` is therefore always `2000`, and *every*
conceivable drift — including a totally stale export — passes. The plan's own instruction
("Pick the tolerance from the data, not by guessing") cannot be followed through this API.

By contrast `assert_prediction_close` (line 123) **does** take `psi_floor` and `rtol`
keyword-only args, so the mean check is fine — but its failure message hardcodes `" psi"`
(line 139), which will print inches as psi.

**Exact fix** — parameterize the helper, keeping the strength callers on defaults:

```python
    def assert_variance_close(
        self,
        committed: float,
        fresh: float,
        *,
        floor: float = VARIANCE_PSI2_FLOOR,
        rtol: float = VARIANCE_RTOL,
        unit: str = "psi^2",
        label: str,
    ) -> None:
        self.checks += 1
        diff = abs(fresh - committed)
        delta = max(floor, rtol * abs(committed))
        if diff > delta:
            rel = diff / max(abs(committed), 1.0)
            self.failures.append(
                f"  {label}: committed={committed:.4f}, fresh={fresh:.4f}, "
                f"abs={diff:.2f} {unit} ({rel * 100:.2f}% rel; tol="
                f"max({floor:.0f}, {rtol * 100:.1f}%)={delta:.2f})"
            )
```

Add the same `unit: str = "psi"` kwarg to `assert_prediction_close` and rename its
`psi_floor` to `floor` (update the three existing call sites at lines 267, 288, 321 and the
one in `check_compositions_json`). Then `check_slump_test_vectors_json` calls:

```python
        drift.assert_prediction_close(
            cv["expected_mean"], fv["expected_mean"],
            floor=0.1, rtol=0.05, unit="in",
            label=f"slump_test_vectors[{i}].expected_mean",
        )
        drift.assert_variance_close(
            cv["expected_variance"], fv["expected_variance"],
            floor=0.25, rtol=0.10, unit="in^2",
            label=f"slump_test_vectors[{i}].expected_variance",
        )
```

(`floor=0.25 in²` ≈ 0.5 in on the σ scale — same spirit as the 0.1 in mean floor the plan
already reasons its way to. Still confirm both against a deliberate re-fit, as Step 14b says.)

---

### C2. Committed Linux visual-regression baselines will fail `e2e.yml`; the plan never regenerates them

`test/e2e/visual-regression.spec.ts` takes `fullPage: true` screenshots of `/` and compares
against baselines committed in `test/e2e/visual-regression.spec.ts-snapshots/`:

```
about-modal-desktop-desktop-linux.png
about-modal-mobile-mobile-linux.png
composition-mobile-mobile-linux.png
home-desktop-desktop-linux.png      <-- contains the readouts strip
home-mobile-mobile-linux.png        <-- contains the readouts strip
```

The spec begins with

```ts
  test.skip(
    process.platform !== "linux",
    `visual baselines are Linux-rendered; skipping on ${process.platform}`,
  );
```

so it is **silently skipped on the author's macOS** (Step 16a's `make check-all` will not
catch it) and **active on `e2e.yml`'s `runs-on: ubuntu-latest` matrix for both `desktop` and
`mobile`**. `e2e.yml`'s `paths:` includes `'docs/**'`, so this PR triggers it.

Replacing `W/B: 0.42` with `Slump: 6.5 ± 3.1 (2σ) in` changes glyphs, string width, and
(per Step 9a) mobile visibility of a span. `playwright.config.ts` allows
`maxDiffPixelRatio: 0.02` — the readout strip *might* squeak under 2% of a full-page
screenshot, but that is a coin flip, and if the wider string reflows the strip the diff
region grows. Do not gamble on it.

**Exact fix** — add a new step between Step 11 and Step 16e (and a line in the Step 16
checklist):

```bash
cd /Users/sebastianament/Code/SustainableConcrete-slump
# Tag MUST match package-lock.json, per test/e2e/README.md:105-114
node -p "require('./package-lock.json').packages['node_modules/@playwright/test'].version"
docker run --rm --network host -v $(pwd):/work -w /work \
  mcr.microsoft.com/playwright:v1.59.1-noble \
  bash -lc "npx --yes playwright@1.59.1 test --grep @visual --update-snapshots"
# Verify by COMPARISON, not by writing (test/e2e/README.md:118-119):
docker run --rm --network host -v $(pwd):/work -w /work \
  mcr.microsoft.com/playwright:v1.59.1-noble \
  bash -lc "npx --yes playwright@1.59.1 test --grep @visual"
git add test/e2e/visual-regression.spec.ts-snapshots/
```

Fallback if Docker is unavailable: dispatch `e2e.yml` with `update_snapshots: true` and
commit the `snapshots-desktop` / `snapshots-mobile` artifacts (`e2e.yml:229-251`).

---

### C3. Step 14d desynchronizes `slump_test_vectors.json` from `slump.json`, guaranteeing a `make check-all` failure in Step 16a

Step 14d, as written, does:

1. `cp docs/model/slump*.json /tmp/committed_check/`
2. `python experiments/regenerate_slump_json.py` — **rewrites BOTH** `slump.json` *and*
   `slump_test_vectors.json` from a fresh, non-convex fit
3. corrupt `slump.json`
4. `git checkout docs/model/slump.json` ← **only one of the two files is reverted**

The working tree is left with the *committed* `slump.json` paired with a *freshly re-fit*
`slump_test_vectors.json`. `test/test_js_slump.mjs` compares them at `RTOL = 1e-6`
(plan Step 3a), so it fails — in Step 16a's `make check-all`, and again in the `js-sync` job
(`docs/model/**` is in `js_sync_paths`, line 32) if the stray file gets committed.

**Exact fix** — do the corruption in a throwaway fresh-dir and never dirty the tracked tree:

```bash
cd /Users/sebastianament/Code/SustainableConcrete-slump
rm -rf /tmp/committed_check /tmp/fresh_check && mkdir -p /tmp/committed_check /tmp/fresh_check
cp docs/model/slump*.json /tmp/committed_check/
python experiments/regenerate_slump_json.py
cp docs/model/slump*.json /tmp/fresh_check/
git checkout docs/model/slump.json docs/model/slump_test_vectors.json   # <-- BOTH

python experiments/check_artifacts_drift.py \
  --committed-dir /tmp/committed_check --fresh-dir /tmp/fresh_check   # expect pass

python - <<'PY'
import json
p = json.load(open("/tmp/fresh_check/slump.json"))
p["lengthscales"][0] *= 100
json.dump(p, open("/tmp/fresh_check/slump.json", "w"))
PY
python experiments/check_artifacts_drift.py \
  --committed-dir /tmp/committed_check --fresh-dir /tmp/fresh_check \
  && { echo "BUG: should have failed"; exit 1; }
git status --porcelain docs/model/   # must be empty
```

Apply the same discipline to Step 13b: it too runs the exporter against the tracked tree.
Add `git checkout docs/model/slump.json docs/model/slump_test_vectors.json` (or commit both)
so the pair never diverges.

---

### C4. `js-sync.yml`'s `paths:` filter is not updated — the new test won't run on the PRs it is meant to guard

`js-sync.yml:27-47` defines `&js_sync_paths`, aliased into *both* the `push` and
`pull_request` triggers (lines 27 and 50). It lists `docs/gp.mjs`, `docs/gp_v2_fast.mjs`,
`docs/feature_registry.mjs`, `docs/units.mjs`, `docs/model/**`, and ten `test/*.mjs` files.
It does **not** list `docs/slump.mjs` or `test/test_js_slump.mjs`.

Step 12b only says "Add `test_js_slump.mjs` in the same form the workflow uses" and prescribes
`grep -n "test_js_filters" .github/workflows/js-sync.yml`. That grep returns exactly one hit
(line 105, the `run:` step) **because `test_js_filters.mjs` is itself already missing from the
paths list** — so following the plan's instruction literally reproduces the omission and hides
it from the implementer.

Net effect: a later PR touching only `docs/slump.mjs` never runs the parity test.
(`docs/model/**` does cover artifact-only changes, so the first PR is fine — the hole opens
later.)

**Exact fix** — in Step 12b, in addition to the `run:` step, insert into the `paths:` block
immediately before `- '.github/workflows/js-sync.yml'` (line 47):

```yaml
      - 'docs/slump.mjs'
      - 'test/test_js_slump.mjs'
```

While there, close the three pre-existing gaps so the Makefile's lockstep comment
(`Makefile:107-109`) is actually true:

```yaml
      - 'test/test_js_preview_state.mjs'
      - 'test/test_js_categorical_source.mjs'
      - 'test/test_js_filters.mjs'
```

And add the new step in the existing per-test form (after line 105):

```yaml
      - name: Slump GP Python ↔ JS parity (slump_test_vectors.json)
        run: node test/test_js_slump.mjs
```

---

### C5. `model-artifacts-coherence.yml`'s `paths:` omits `boxcrete/slump_model.py` — the coherence gate never fires on the file that defines the slump fit

`model-artifacts-coherence.yml:22-39` (`&artifacts_paths`) enumerates every input to the
*strength* fit: `boxcrete/strength_model.py`, `kernels.py`, `likelihoods.py`, `priors.py`,
`features.py`, `utils.py`, `__init__.py`, `data/**`, and the regen scripts.
`boxcrete/slump_model.py` is absent, and Step 14c adds only
`experiments/regenerate_slump_json.py` and `docs/slump.mjs`.

So the single highest-value trigger — someone editing `fit_slump_gp` — will land a stale
`slump.json` with **no** coherence signal. That is precisely the failure mode Step 14 exists
to prevent, and the plan's Step 14 preamble ("without this step the new artifacts get zero
coherence coverage") is only half-solved.

**Exact fix** — Step 14c should add three lines, not two:

```yaml
      - 'boxcrete/slump_model.py'
      - 'experiments/regenerate_slump_json.py'
      - 'docs/slump.mjs'
```

Also add `- 'experiments/check_artifacts_drift.py'` — a pre-existing gap: today, editing the
drift checker itself does not re-run the workflow it gates.

---

### C6. `check_artifacts_drift.py` skips missing artifacts *without failing*, so Step 14d's "Expect: passes" is satisfied even if the new checkers were never reached

`main()`, lines 348-355:

```python
        if not c_path.exists() or not f_path.exists():
            print(
                f"::warning::skipping {fname}: missing on at least one side "
                f"(committed={c_path.exists()}, fresh={f_path.exists()})"
            )
            continue
```

`continue`, not a failure. In Step 14d, `/tmp/committed_check` contains only `slump*.json`, so
the run emits three `::warning::skipping` lines for the strength artifacts and exits 0 —
and would *also* exit 0 if the dispatch entry were misspelled, omitted, or if
`check_slump_json` raised no comparisons. The plan's positive assertion ("Expect: passes") is
therefore not evidence that the new code ran.

(The negative half of 14d — the ×100 lengthscale corruption against the 10× ratio band — is a
valid detector, so the wiring is not *entirely* unverified. But a plain exit-0 must not be
read as coverage.)

**Exact fix** — make Step 14d assert on the comparison count and the absence of a slump skip:

```bash
python experiments/check_artifacts_drift.py \
  --committed-dir /tmp/committed_check --fresh-dir /tmp/fresh_check | tee /tmp/drift.log
grep -q "::warning::skipping slump" /tmp/drift.log && { echo "BUG: slump artifacts skipped"; exit 1; }
grep -q "ran 0 numerical comparisons" /tmp/drift.log && { echo "BUG: no checks ran"; exit 1; }
```

Optional hardening (recommended, small): make the skip fatal when the file exists in
`--fresh-dir` but not `--committed-dir`, since that is exactly "the workflow forgot to snapshot
a new artifact" rather than a benign absence.

---

### C7. The proposed exporter fails `black --check`, which `tests.yml :lint` runs unfiltered over `experiments/`

Plan Step 1a, lines 287-288:

```python
        train_X_aug = model.train_inputs[0].detach()   # [n, 10], post-transform
        train_Y_std = model.train_targets.detach()     # [n], standardized
```

Three and five spaces before the inline `#`. Black normalizes trailing comments to exactly
two spaces. Verified empirically against this repo: `grep -rn "[A-Za-z0-9_)\]\"']   *# "
boxcrete/ experiments/ --include=*.py` returns **zero** hits across the whole
already-black-formatted tree.

`tests.yml` has **no `paths:` filter** (it runs on every push and PR) and its `lint` job runs
`black --check --diff .` and `flake8 .` from the repo root — both cover `experiments/`.
So this is a guaranteed red check on the PR.

**Exact fix** — two spaces:

```python
        train_X_aug = model.train_inputs[0].detach()  # [n, 10], post-transform
        train_Y_std = model.train_targets.detach()  # [n], standardized
```

(Black will also want the second one at two spaces even though the columns then no longer
align; that is expected.) Everything else in the proposed exporter is lint-clean — I extracted
the block and checked: no line exceeds 88 chars, no unused imports, and both
`load_concrete_strength` (`boxcrete/utils.py:365`) and `SLUMP_Y_COLUMNS`
(`boxcrete/utils.py:56`) exist. Keep Step 1b's `black --check` + `flake8` gate; it is the right
guard, it just needs the source to be correct up front.

---

## Suggestions

**S1. Step 16b's premise about `budgets.json` is wrong — the file is dead.**
`budgets.json` is referenced by **nothing** except a stale comment in `lighthouse.yml:3`.
Neither `lighthouserc.json` nor `lighthouserc.mobile.json` contains a `budgetsFile` or
`budgets` key, and no workflow passes it to `lhci`. The real gates are the
`resource-summary:*` assertions inline in `lighthouserc.json`. Reword Step 16b (and Step 8c's
"Confirm the transferred size stays within `budgets.json`") to reference
`lighthouserc.json`'s assertions instead. Optionally delete `budgets.json` in a separate change.

**S2. The size budgets will not trip; say so and stop worrying about it.**
Actual effective limits (`lighthouserc.json`): `document` 50 KB **warn**, `stylesheet` 50 KB
warn, `script` 300 KB warn, `image` 1.5 MB **error**, `total` 3 MB **error**. A ~15 KB JSON
fetched by `loadJSON` is classified `other`/`xhr`, not `script`, and current
`docs/model/` totals ≈ 340 KB uncompressed. There is no request-count budget that a new
`.mjs` module could trip (only `third-party:count` 5, `image:count` 10, `font:count` 4, all
warn). Mobile's only **error** performance gate is `total-blocking-time ≤ 300 ms`; a 61×61
Cholesky is microseconds. So Step 16b's fallback plan is prudent but almost certainly
unnecessary.

**S3. The real Lighthouse risk is CLS, not size — and it's an ERROR gate on both profiles.**
`cumulative-layout-shift ≤ 0.1` is `error` in *both* configs. Step 8d-i deliberately leaves the
slump readout at `–` until the blocking `Promise.all` resolves, then swaps in
`Slump: 6.5 ± 3.1 (2σ) in` — a large horizontal text-width jump inside a flex strip that also
holds GWP and Cost. Consider reserving width (`min-width` / `ch` on `#slump-value` +
`#slump-uncertainty`, or a `–` placeholder of similar length) in Step 9, and add a CLS check to
Step 16b's manual list.

**S4. `make check-all` does not run the mobile Lighthouse profile.**
`Makefile:186-187` is `test-lighthouse: npx lhci autorun` with no `--config`, so lhci's default
discovery picks up `lighthouserc.json` only. `lighthouse.yml:134-148` runs both profiles.
Step 16a therefore gives no mobile signal; Step 16b's explicit two-config invocation is the
only place mobile is exercised (and its first line duplicates 16a's desktop run). Either note
this in Step 16a, or fix `test-lighthouse` to run both:

```make
test-lighthouse:
	npx lhci autorun --config=lighthouserc.json
	npx lhci autorun --config=lighthouserc.mobile.json
```

**S5. Do NOT put `mean_constant` through `assert_within_ratio`.** The plan correctly limits
`check_slump_json`'s ratio checks to `lengthscales` and `noise`. Make that explicit as a
warning, because `mean_constant ≈ 0.0261` is near zero *and sign-flippable across basins*;
`ratio = (fresh + atol) / (committed + atol)` (line 116) would go negative and trip
`ratio < lo` on a harmless refit. Same reasoning for `y_mean`/`y_std`: those are data-derived
and belong in the exact-match group, not the ratio group.

**S6. `normalize_lower` / `normalize_upper` exact equality is safe, but note *why*.**
`boxcrete/slump_model.py:70-73` builds `Normalize(d=d_aug, bounds=derive_bounds_from_X(X_aug))`
— data-derived min/max over an IEEE-deterministic `hrwr / max(binder, 1.0)`, not fit-derived.
So the plan's "must match bit-for-bit" is correct. If you want belt-and-braces, compare with
`math.isclose(rel_tol=1e-12)` rather than `==` so a future float-formatting change in
`json.dumps` doesn't cause a spurious red.

**S7. `_zip_lists` raises an uncaught `ValueError`.** Lines 176-181 raise on length mismatch;
`main()` does not catch it, so a changed `n_train` or `d_aug` produces a raw traceback rather
than the curated `::error::` block. If `check_slump_json` routes `normalize_lower` /
`normalize_upper` / `lengthscales` through `_zip_lists`, it inherits this. Either wrap the
`checker(...)` call at line 358 in `try/except ValueError` and append to `drift.failures`, or
guard the lengths in `check_slump_json` first (as `check_test_vectors_json` does at 258-264).

**S8. Step 13a's renumbering instruction is workable but under-specified.** The script has
six numbered steps (`regenerate_all_artifacts.sh:37, 41, 45, 49, 53, 57`) where 5 is the
freshness check and 6 is the JS suite. Inserting slump "as step 5" means the final ordering
must be: 1 strength → 2 augment → 3 compositions → 4 JS predictions → **5 slump** →
6 freshness → 7 JS suite. Spell that out, and note the summary comment to update is
**lines 12-23** (not 12-24; line 24 is blank). Also add `test_js_slump.mjs` to the `for t in`
list at line 58.

**S9. Pipefail question — confirmed safe.** `set -euo pipefail` is at line 32, and
`python -u ... | tail -5` does propagate: empirically,
`bash -c 'set -euo pipefail; python3 -c "import sys; sys.exit(1)" | tail -5; echo NOPE'`
exits 1 without printing `NOPE`, whereas without `pipefail` it exits 0. Python's traceback goes
to stderr, which is *not* piped, so `| tail -5` does not swallow the error message. No change
needed — but the plan's Step 13b dry-run should still keep the pipe consistent with the script.

**S10. Add `regenerate_slump_json.py` to `experiments/__init__.py`'s docstring.** Lines 11-19
enumerate the pipeline scripts. Not CI-enforced, but the file's whole purpose is that
inventory and it is the natural place a reader looks. One line in Step 15.

**S11. Step 15 (`docs/model/README.md`) — while editing, note the pre-existing typo.** The
second row of the Files table is labelled `strength_model.py` but describes
`docs/model/strength_model.pt`. Out of this change's scope; fix it or leave it, but don't
propagate the pattern.

---

## Verified Claims

Everything below was confirmed by reading the file.

| # | Claim | Verdict |
|---|---|---|
| 1 | `experiments/regenerate_slump_json.py` does **not** affect the `--cov-fail-under=100` gate | **TRUE.** `Makefile:102` and `tests.yml:312` both use `--cov=boxcrete`; `pyproject.toml [tool.coverage.run] source = ["boxcrete"]`. `experiments/` is outside coverage entirely. |
| 2 | `docs/slump.mjs` needs no JS coverage / no JS lint | **TRUE.** No `eslint`/`prettier` anywhere (`package.json` devDeps are only `@lhci/cli`, `@playwright/test`, `@types/node`, `http-server`); `.pre-commit-config.yaml` is black + flake8 only; `make lint` (Makefile:77-80) is scoped to `TRACKED_PY`. There is no JS coverage tooling at all. |
| 3 | `make check-all` is `lint test-py test-js test-notebook-fmt test-notebooks test-e2e test-lighthouse` | **TRUE** (Makefile:192). |
| 4 | `JS_TESTS` currently has 13 entries; adding one gives "All 14 JS tests passed." | **TRUE** (Makefile:110-123, 13 entries; `$(words ...)` at line 130). |
| 5 | `js-sync.yml` structure = YAML anchor `&js_sync_paths` (27-47) + one `- name:`/`run: node ...` step per test (68-105) | **TRUE.** Step 12b's "explicit `node` step" reading is right; there is no matrix and **no count assertion** anywhere. |
| 6 | No automated parity check between Makefile `JS_TESTS` and `js-sync.yml` | **TRUE.** Only the prose comment at `Makefile:107-109`. Grep across `test/` and `experiments/` for `js-sync`/`JS_TESTS` returns nothing. |
| 7 | Coherence workflow snapshot step is at lines 73-77 | **Essentially true** — `run: |` at 73, `mkdir` 74, three `cp` at 75-77. Step 14a's insertion point is unambiguous. |
| 8 | Coherence workflow's explanatory docstring is at 82-95 | **Essentially true** — the comment block is 83-95 (`- name:` at 82). |
| 9 | Coherence `paths:` filter is "around line 31" | **TRUE-ish** — the block is 22-39; line 31 is `experiments/regenerate_strength_json.py`. See **C5** for what's missing. |
| 10 | `regenerate_all_artifacts.sh` failing on a missing slump dep breaks the coherence workflow | **TRUE**, and that is desirable. `set -euo pipefail` (line 32) + `run: bash experiments/...` (workflow line 80) → non-zero aborts the job before `check_artifacts_drift.py`. Note the workflow installs `pip install -e .` (**not** `[dev]`), which is sufficient: `fit_slump_gp` needs only torch/botorch/gpytorch, all base deps in `pyproject.toml`. |
| 11 | `regenerate_all_artifacts.sh` has 6 numbered steps, 5 = freshness, 6 = JS suite | **TRUE** (lines 37, 41, 45, 49, 53, 57). Plan's renumbering is coherent — see **S8**. |
| 12 | `set -euo pipefail` + `\| tail -5` propagates a Python failure | **TRUE**, empirically confirmed. stderr is not piped, so the traceback stays visible. |
| 13 | `check_strength_json` at 191, `check_test_vectors_json` at 252, dispatch list at ~344, `assert_within_ratio` 102, `assert_prediction_close` 123, `assert_variance_close` 143 | **ALL TRUE.** (File is 382 lines, not 381 — immaterial.) |
| 14 | Dispatcher contract: `checker(committed: dict, fresh: dict, drift: DriftCollector) -> None`, driven by a `(filename, callable)` list at 343-347 | **TRUE.** The proposed `check_slump_json` / `check_slump_test_vectors_json` fit it exactly. |
| 15 | The dispatcher tolerates a file being absent | **TRUE — and this is a hazard, not a feature.** Lines 350-355 print `::warning::skipping` and `continue`; exit status unaffected. See **C6**. |
| 16 | `assert_prediction_close` signature is caller-parameterizable | **TRUE** — `(committed, fresh, *, psi_floor, rtol, label)`. Slump can pass `psi_floor=0.1`. Message hardcodes `" psi"`. |
| 17 | `assert_variance_close` signature is caller-parameterizable | **FALSE** — `(committed, fresh, *, label)`, tolerances hardcoded. **See C1.** |
| 18 | `test/test_data_freshness.mjs` enumerates `docs/model/` and a new `slump.json` would break it or need registering | **FALSE.** It reads five explicit paths (lines 46, 49, 52, 55, 58) via `readFileSync`; no `readdirSync`/glob. A new `slump.json` neither breaks it nor needs registering. Corollary: it also grants slump **zero** freshness coverage, so `check_artifacts_drift.py` (Step 14) is the only guard — which is why **C1** and **C6** matter. |
| 19 | `budgets.json` caps the payload | **FALSE.** Orphaned; not wired into either `lighthouserc*.json`. See **S1**. |
| 20 | A blocking ~15 KB JSON fetch plausibly trips a budget | **NO.** All size gates are `warn` except `image` (1.5 MB) and `total` (3 MB); current `docs/model/` ≈ 340 KB. See **S2**. Real risk is CLS (**S3**). |
| 21 | A new `.mjs` module trips a script-count or request-count budget | **NO.** Only `third-party:count` (5), `image:count` (10), `font:count` (4) exist, all `warn`, and none counts first-party modules. |
| 22 | `make test-e2e` needs a server; does `make check-all` start one? | **Self-contained.** `playwright.config.ts:62-67` declares `webServer: { command: "npx http-server docs -p 4173 -s -c-1 -a 127.0.0.1", url: ".../index.html", reuseExistingServer: !process.env.CI }`. `make test-e2e` (Makefile:180-182) just runs `npx playwright test --project=...`; Playwright starts and stops the server. No manual server needed. It does require a prior `npm ci` + `npx playwright install --with-deps chromium` (documented at Makefile:27-28). |
| 23 | Is there a JS lint step `docs/slump.mjs` must satisfy? | **NO.** See row 2. |
| 24 | `test/e2e/README.md` W/B rows are at lines 44-45 | **TRUE.** Both `readouts-strip.spec.ts` rows. No count assertion on the table. |
| 25 | `#curve-summary-note` (Step 7b) is asserted by an e2e spec | **NO** — grep across `docs/` + `test/` finds it only at `docs/index.html:249`. Safe to reword. |
| 26 | Other e2e specs depend on the readouts strip | **Only two.** `readouts-strip.spec.ts` (rewritten in Step 10) and `drag-redraw.spec.ts:25-55`, which reads `#gwp-value` only — unaffected. `unit-toggle.spec.ts` and `accessibility.spec.ts` scope to `#sliders .slider-unit` and `#curve-summary`, not the readouts. |
| 27 | `readouts-strip.spec.ts` mobile invariant tightens from 2 to 3 visible items | **TRUE** and correctly anticipated by Step 10a. Current selector is `#readouts > div:not(.wb-readout)` (line 37) with `maxTop - minTop <= 2`; after the change all three divs are visible on a Pixel 7 viewport. Genuine wrap risk, correctly mitigated by Step 9a. |
| 28 | `e2e.yml` / `lighthouse.yml` will trigger on this change | **TRUE** — both filter on `'docs/**'` (e2e.yml:166, lighthouse.yml:94). `tests.yml` has no `paths:` filter at all and runs on every PR. |
| 29 | `SLUMP_Y_COLUMNS` and `load_concrete_strength` are importable from `boxcrete.utils` | **TRUE** (`boxcrete/utils.py:56` and `:365`; also re-exported from `boxcrete/__init__.py:69`). `dataset.slump_data` exists at `boxcrete/utils.py:233`. |
| 30 | `Normalize` bounds in the slump model are data-derived (so exact comparison is legitimate) | **TRUE** — `boxcrete/slump_model.py:70-73`: `aug_bounds = derive_bounds_from_X(X_aug)` then `Normalize(d=d_aug, bounds=aug_bounds)`. See **S6**. |

---

## Recommended order of fixes

1. **C7** (one-line black fix in the Step 1 source block).
2. **C1** (parameterize `assert_variance_close`; rewrite Step 14b's tolerance guidance).
3. **C6** (make Step 14d's verification assert on the comparison count).
4. **C3** (rewrite Step 14d to use a throwaway fresh-dir; revert both files in 13b).
5. **C4** + **C5** (extend both `paths:` filters).
6. **C2** (add a visual-baseline regeneration step before Step 16e).
