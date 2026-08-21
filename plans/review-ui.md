# Adversarial review — UI/frontend portion (Steps 7–11)

VERDICT: NEEDS_REVISION

> Scope: Steps 7–11 only (`docs/index.html`, `docs/ui.mjs`, `docs/style.css`,
> `test/e2e/readouts-strip.spec.ts`, `test/e2e/README.md`). Steps 0–6 and 12–16 were read
> only where they constrain the UI work. Every claim below was checked against the working
> tree at `/Users/sebastianament/Code/SustainableConcrete-slump`.

---

## Summary Assessment

The plan's *markup* step (Step 7) is exact and correct: line numbers, the strings to remove,
and the `#curve-summary-note` text all match the file byte-for-byte. The data-plumbing
assumptions behind Step 8 also hold up — `compositions.json`'s `column_names` is exactly the
9-dim, Time-free vector in the same order as the exporter's `RAW_FEATURE_NAMES`, so
`compForGWP.length === 9 === d_in` and `augmentSlumpInput`'s hard-coded indices
(`I_CEMENT=0, I_FLYASH=1, I_SLAG=2, I_HRWR=4`) land on the right columns.

The problems are concentrated in the **wiring and the e2e specs**, and three of them are
guaranteed failures rather than risks:

1. The mortar e2e test uses a selector that **cannot match any element** (`hasText: "0"`
   against buttons that read `"Source A"`/`"Source B"`/`"Source C"`).
2. That same test is unguarded by project and will run on `mobile`, where the composition
   panel is `hidden` behind `#mobile-show-sliders` — a second, independent hard failure.
3. Step 8e's insertion point (`docs/ui.mjs:185-186`) is **the wrong block** — it is the
   mobile *composition-button* label inside `if (mobileSlidersBtn) { … }`, not the
   gwp/cost unit labels (those are 187–188).

Beyond those, the plan's unit-toggle approach is *unit-only* and leaves the slump **value**
stale across a toggle — printing a 6.8 in reading as `6.8 mm`, a 25.4× error. And the mobile
layout budget almost certainly does not accommodate a third readout even with `±2σ` hidden,
which means Step 9a's fix is insufficient and Step 10a's no-wrap assertion will fail.

Good news that the plan worried about unnecessarily: **the default mix is Material Source 1**
(`compositions[74]`, the median index), so first paint shows a real number, not `n/a`. And
`predictSlump` is nowhere near a performance concern — see Verified Claims.

---

## Critical Issues (must fix)

### C1 — Step 10b: the mortar-toggle selector matches nothing

`docs/ui.mjs:411` sets the button text from `sourceLabel(cls)`, and `sourceLabel`
(`docs/ui.mjs:347-351`) maps `{0: "Source A", 1: "Source B", 2: "Source C"}`. No button
contains the character `0`. The plan's

```ts
await page.locator('.material-source-group button', { hasText: "0" }).click();
```

will exhaust `actionTimeout` (10 s) and fail on every run.

The structural half of the selector *is* right: `docs/ui.mjs:384` sets
`group.className = "slider-group material-source-group"`, and `docs/ui.mjs:406-410` puts real
`<button class="toggle-btn">` elements inside a `.toggle-row` child of that group.

**Exact fix** — use the rendered label and the button class:

```ts
await page.locator('.material-source-group .toggle-btn', { hasText: "Source A" }).click();
```

`_sourceClasses` is derived from the catalog (`docs/ui.mjs:401-403`) and is `[0, 1, 2]`
(verified: `compositions.json` Material Source counts are `{0: 69, 1: 27, 2: 53}`), so
exactly one button reads `"Source A"`.

---

### C2 — Step 10b: the test has no project guard and will fail on `mobile`

`playwright.config.ts:42-58` defines projects `desktop` and `mobile` (Pixel 7, 412 px), and
`testMatch` runs every test in **both** unless skipped. On mobile,
`docs/index.html:325-332` moves `#sliders` into `.mobile-sliders-view`, which
`docs/index.html:204` renders as `<div class="mobile-sliders-view hidden">`.
`test/e2e/mobile-panel-toggle.spec.ts:16` pins that: `.mobile-sliders-view` is
`toBeHidden()` at load. Playwright's actionability check on `.click()` requires visibility,
so the mortar test fails on `mobile` even after C1 is fixed.

Step 11a compounds this by documenting the row as project `both`.

**Exact fix** — reuse the pattern the repo already has in
`test/e2e/accessibility.spec.ts:19-28`:

```ts
test("mortar mixes show n/a instead of a slump number", async ({ page }, testInfo) => {
  await page.goto("/");
  if (testInfo.project.name === "mobile") {
    await page.locator("#mobile-show-sliders").click();
    await expect(page.locator(".mobile-sliders-view")).toBeVisible({ timeout: 5000 });
  }
  await expect(page.locator("#slump-value")).toBeVisible();
  await page.locator('.material-source-group .toggle-btn', { hasText: "Source A" }).click();
  await expect(page.locator("#slump-value")).toHaveText("n/a");
  await expect(page.locator("#slump-uncertainty")).toBeEmpty();
});
```

(`toBeEmpty()` rather than `toHaveText("")` — clearer intent, and it does not depend on how
Playwright normalizes an empty string.)

If you keep the row in `test/e2e/README.md`, the file's own convention for the project column
is `desktop+mobile`, not `both` (see e.g. the `font-uniformity.spec.ts` row at line 41).

---

### C3 — Step 8e: `docs/ui.mjs:185-186` is the wrong insertion point

The plan says "At `docs/ui.mjs:185-186`, next to the gwp/cost unit-label updates". Those two
statements name different places. Lines 183–186 are:

```js
  const mobileSlidersBtn = document.getElementById("mobile-show-sliders");
  if (mobileSlidersBtn) {
    mobileSlidersBtn.textContent = unitSystem === "metric" ? "Composition (kg/m³)" : "Composition (lb/yd³)";
  }
```

The gwp/cost unit labels are at **187–188**. Inserting the slump line at 185–186 puts it
inside the `if (mobileSlidersBtn)` body, so the slump unit silently stops updating on any
page where that button is absent, and the code reads as if it belongs to the mobile toggle.

**Exact fix**: insert after line 188 (`document.getElementById("cost-unit").textContent = U().cost;`),
not at 185–186. See C4 for what should actually go there.

---

### C4 — Step 8e updates the slump *unit* but never the slump *value* → `6.8 mm`

This is the substantive bug behind C3. Trace the `toggle-units` handler
(`docs/ui.mjs:158-197`) to completion: it relabels, calls `updateSliderLabels()`
(which I read at 924–966 — it touches sliders only), `scheduleCurveSummary()`, and
`startAnimLoop()`. `animLoop` (2425–2565) never calls `updateReadouts`. Grep confirms
`updateReadouts` has exactly three call sites: `requestRedraw` (846), the slider idle timer
(894), and `update()` (1329) — **none reachable from the toggle**.

So after Step 8e as written, the DOM reads `Slump: 6.8 mm` for a 6.8 in prediction. The
factors are real: `docs/units.mjs` gives `slumpFactor` 1 → 25.4.

This is a *pre-existing* defect for GWP and Cost too (`gwpFactor` 1 → 1.6856,
`costFactor` 1 → 1/1.30795 — both values go stale with a fresh label until the next slider
move). At 1.7× it is easy to miss; at 25.4× it is not. Shipping a third readout onto the
same broken path is not acceptable.

The plan asks whether the `if (textContent !== "")` guard "actually works". It does work as
written *for the mortar case* — nothing rewrites `#slump-unit` between the toggle and the
next `updateReadouts`, precisely because `updateReadouts` is never called. But it is the
wrong shape of fix.

**Exact fix** — replace Step 8e entirely. Keep lines 187–188 and add immediately after 188:

```js
  // Readout VALUES are unit-scaled, so relabelling alone leaves them numerically
  // stale (a 6.8 in slump would render as "6.8 mm"). Re-render the strip, which
  // also keeps #slump-unit and the mortar "n/a" state consistent — unlike a
  // blind textContent write, which would print a bare "mm" next to "n/a".
  // Guarded: both unit toggles are live in the DOM before init() resolves.
  if (compositionsData && currentComposition) updateReadouts();
```

This is the "simpler correct approach" the review question asks about: one call replaces the
per-element write *and* the `!== ""` guard, and it fixes the GWP/Cost staleness at the same
time. Verify with a new assertion in Step 10 (or `unit-toggle.spec.ts`) that
`#slump-value` scales by 25.4 across a toggle, and add a manual check to Step 16c item 3.

Note also that `getDisplayFactors()` (`docs/ui.mjs:137-155`) — the *animated* unit
interpolator — hand-enumerates `strength/mass/gwp/cost` and would return
`slumpFactor: undefined`. `updateReadouts` uses `U()` (line 1417), not `getDisplayFactors()`,
so this is currently harmless — but see S4.

---

### C5 — Step 9a is not enough: the mobile strip will very likely wrap, and Step 10a asserts it won't

The plan's own source is the strongest evidence against it. `docs/style.css:1693-1697`, the
comment being deleted, says:

> Hide the W/B (water-to-binder ratio) readout on mobile. With the bumped 0.88rem font,
> **GWP + Cost already fill the strip**; adding W/B forces a wrap that steals vertical space
> from the strength curve canvas above.

The string that was proven to force a wrap is `W/B: 0.408` — **10 characters**. The
replacement is `Slump: 6.8 in` — **13 characters**, i.e. strictly worse, and Step 9a only
removes the `± 6.6 (2σ)` suffix, not the base row.

Rough budget at the `mobile` project's 412 px (DM Sans, 0.88rem ≈ 14 px, ~7.3 px/glyph):
usable content width is ~370 px after `body` padding `0.5rem`×2 and `.panel` padding
`0.8rem`×2. `GWP: 337.5 kg CO₂e/m³` ≈ 153 px + `Cost: 121.9 ± 8.2 (2σ) $/m³` ≈ 197 px
already ≈ 350 px before the `gap: 0.4rem 0.8rem` column gap — exactly the "already fill the
strip" the comment describes. Adding ~95 px puts the row at ~470 px against ~370 px
available. `.readouts` is `flex-wrap: wrap` on mobile (`docs/style.css:1561-1566`), so it
wraps rather than overflows, and **Step 10a's no-wrap assertion fails**.

I could not measure this directly — `node_modules/` is not installed in this worktree, so
Playwright cannot run here. Treat the numbers as an estimate; the *direction* is
source-backed and does not depend on the estimate's precision.

**Exact fix** — make Step 9a empirical rather than assumed. Before writing the CSS, measure:

```bash
cd /Users/sebastianament/Code/SustainableConcrete-slump
npm ci && npx playwright install --with-deps chromium
npx playwright test readouts-strip --project=mobile
```

If it wraps (expected), escalate in this order and re-measure after each:

1. Hide the cost suffix too — it is the single widest non-essential token:
   ```css
   #slump-uncertainty, #cost-uncertainty { display: none; }
   ```
   (≈ −73 px. Note this changes an existing desktop-parity behavior; call it out.)
2. Drop the slump unit label on mobile and put the unit in the row label instead
   (`Slump (in): 6.8`), or hide `#slump-unit` and rely on the `#curve-summary-note`.
3. If it still wraps, **change the invariant instead of pretending**: assert at most 2 rows
   *and* that `#readouts` `offsetHeight` stays under a pinned pixel budget, so the curve
   canvas cannot silently lose height. Document the wrap as intentional in Step 11.

Do not land Step 9a + Step 10a as written on the assumption that hiding `±2σ` suffices.

---

### C6 — Step 10a's mobile assertion, as worded, permits the very wrap it claims to forbid

> "all three `#readouts > div` share **at most 2 distinct `top` values**, i.e. the strip
> still does not wrap"

Two distinct `top` values *is* a wrap — that is exactly what a second row looks like. The
existing test (`test/e2e/readouts-strip.spec.ts:43-48`) gets this right with
`maxTop - minTop <= 2`, where the `2` is **pixels of tolerance**, not a count of rows. The
plan appears to have misread the tolerance constant as a row count.

**Exact fix** — keep the existing assertion verbatim and only drop the
`:not(.wb-readout)` qualifier from the selector at line 37:

```ts
const tops = await page.locator("#readouts > div").evaluateAll((els) =>
  els
    .filter((el) => (el as HTMLElement).offsetParent !== null)
    .map((el) => el.getBoundingClientRect().top),
);
expect(tops.length, "expected 3 visible readouts on mobile").toBe(3);
const minTop = Math.min(...tops);
const maxTop = Math.max(...tops);
expect(maxTop - minTop, `readouts strip wrapped (min=${minTop}, max=${maxTop})`)
  .toBeLessThanOrEqual(2);
```

Also bump the count assertion from `toBeGreaterThanOrEqual(2)` to `toBe(3)` — with W/B gone,
all three rows are now visible on mobile and a silently-hidden slump row should fail.

---

## Suggestions

**S1 — Step 8d-i: the guard's stated justification is wrong; it is dead code as planned.**
The plan says "`updateReadouts` can run before `init()` resolves (the shell renders first —
`drawStrengthCurve` bails on `!strengthParams` at line 1627)". The `drawStrengthCurve` bail
is real (verified at 1627), but it is about `strengthParams`, which arrives *after* the
shell — a different lifecycle from the artifacts in the blocking `Promise.all`. In the
planned wiring, `slumpParams` is assigned around line 271, well before the first
`update()` at line 319, so `if (!slumpParams) return;` **can never fire**. The only path that
reaches `updateReadouts` before `init()` resolves is the theme `MutationObserver` at
`docs/ui.mjs:2985`, and on that path `currentComposition` is still `null`, so line 1421
(`currentComposition[msIdx]`) throws long before the slump block — a pre-existing latent bug
the guard cannot help with. `test/e2e/pre-model-shell.spec.ts` stalls the *worker*, not the
JSON fetches, so it does not exercise this either.

Keep the guard (it becomes load-bearing under Step 16b's "move it out of the blocking
`Promise.all`" contingency), but rewrite the comment to say so honestly:

```js
  // Only reachable if slump.json is moved off the blocking Promise.all (see the
  // Step 16b budget fallback); in the default wiring slumpParams is set before
  // the first update(). Leaves the "–" placeholder from index.html in place.
  if (!slumpParams) return;
```

Returning early skips nothing else — the slump block is the last thing in `updateReadouts`
(the function ends at 1454).

**S2 — `ms` when `COL_MS < 0` silently becomes a permanent `n/a`.** Line 1421 is
`const ms = msIdx >= 0 ? Math.round(currentComposition[msIdx]) : 0;`. If the
`Material Source` column ever disappears from `compositions.json`, `ms` is `0`,
`slumpSupportsSource(0, …)` is `false`, and the readout pins to `n/a` forever with a mortar
tooltip that is now a lie. Contrast `colIdx` (`docs/ui.mjs:29-40`), which the codebase
deliberately made *throw* on exactly this class of schema drift. Consider making the slump
branch distinguish "mortar" from "no source column" — or at minimum note the fallback in the
comment.

**S3 — Step 8d leaves `colIdx` dead.** `docs/ui.mjs:1442-1445` are the only callers of
`colIdx` (defined at line 29) anywhere in `ui.mjs`; `filters.mjs` has its own `colIdxStrict`.
After Step 8d, `colIdx` and its 12-line docblock are unreachable. Either delete them in the
same commit or note explicitly that they are retained.

**S4 — `getDisplayFactors()` is a landmine for slump.** It hand-enumerates
`strength/mass/gwp/cost` (`docs/ui.mjs:145-154`). Anyone who later routes the readouts
through the animated path — a natural follow-up, since the readouts are the one thing that
*doesn't* animate across a unit toggle — gets `slumpFactor: undefined` → `NaN`. Add
`slump: to.slump, slumpFactor: from.slumpFactor + (to.slumpFactor - from.slumpFactor) * ease`
in Step 6, or leave a one-line comment at 154 pointing at the omission.

**S5 — Step 8f gives ~zero coverage of Step 8.** `make test-js` runs 13 Node tests
(`Makefile:110-123`); I checked their imports — none of them import `docs/ui.mjs`, and
`test_js_ui_smoke.mjs` imports only `gp.mjs`. Step 8 is covered exclusively by Playwright.
Reword 8f so it does not read as verification, and move the real gate to Step 10c.
Related: `test/e2e/home-loads.spec.ts:9-28` fails the build on any `pageerror`, so a throw
inside the rAF callback *will* be caught — that is the real safety net for Step 8.

**S6 — `predictSlump` performance is a non-issue; state it and move on.** Per call:
61 ARD-RBF evals over 10 dims (~610 mul/add + 61 `Math.exp`) plus one 61-dim forward
substitution (~1.9k flops) — order 3k flops, single-digit microseconds. `initSlumpModel` is
one 61×61 Cholesky (~38k flops), also microseconds. More importantly, `updateReadouts` is
**not** on the preview path: `requestRedraw` (842–851) is called from exactly one place
(`onSliderChange`, line 898), and `animLoop` (2425–2565) and `acquireClassPreview`
(2359–2380) never touch it. So there is no per-frame `predictSlump` during hover previews or
curve transitions, and even during a slider drag it is one call per animation frame. The
plan's silence here is fine; the concern does not survive contact with the code.

**S7 — Cosmetic, Step 8d.** In the `n/a` branch, blanking both `#slump-uncertainty` and
`#slump-unit` leaves `Slump: n/a` followed by two collapsed inter-span spaces. Harmless, but
if you want it tidy, toggle a class on `.slump-readout` instead of blanking `textContent` —
which would also give the mobile CSS in Step 9a a cleaner hook than
`.slump-readout #slump-uncertainty`.

**S8 — Step 16d's screen-reader concern is real, not hypothetical.** `title` on a `<div>` is
mouse-only and is not reliably announced. Since Step 7b already rewrites
`#curve-summary-note` to include "Slump is not measured for mortar mixes.", the `n/a` state
is at least explained *somewhere* in the a11y tree. Consider that sufficient and drop the
`title`, or commit to the `.sr-only` span up front rather than deferring it to a manual pass.

---

## Verified Claims

Everything in this section was checked by reading the file, not inferred.

| Plan claim | Verdict | Evidence |
|---|---|---|
| Step 7a: `docs/index.html:253-254` are the `.wb-readout` div and the slump TODO comment | ✅ exact | `index.html:253-254` |
| Step 7b: `#curve-summary-note` reads "Strength figures are model predictions with a 95 percent interval." | ✅ exact | `index.html:249` |
| `ms` is in scope where the slump block goes, as `Math.round(currentComposition[COL_MS])` | ✅ | `ui.mjs:1419-1421` (with the `COL_MS < 0 → 0` fallback, see S2) |
| `compForGWP` is 9-dim and matches `compositionsData.column_names` | ✅ | `column_names` = `[Cement, Fly Ash, Slag, Water, HRWR, Fine Agg, Coarse Agg, Material Source, Temp (C)]`; **no Time**; `currentComposition = [...compositions[medianIdx]]` at `ui.mjs:371` |
| Column order matches the exporter's `RAW_FEATURE_NAMES` and `augmentSlumpInput`'s `I_CEMENT/I_FLYASH/I_SLAG/I_HRWR = 0/1/2/4`, and `source_dim_raw: 7` | ✅ | same; indices line up exactly, so `predictSlump`'s `d_in` check will pass |
| Step 8d replaces `ui.mjs:1440-1453` | ✅ exact | 1440 = `// W/B ratio`, 1453 = last TODO line; function ends 1454 |
| **Default composition is Material Source 1**, so first paint shows a number, not `n/a` | ✅ | `compositions[74]` (median of 149) has MS = 1; the desktop "visible number" assertion is safe |
| Step 9a: `@media (max-width: 1050px)` opens at line 1438 | ✅ | closes at 1703 |
| Step 9a: `.wb-readout { display: none; }` is at line 1698, inside that query, with the comment at 1693-1697 | ✅ exact | |
| Step 9a: the comment at 1689-1691 names "GWP / Cost / W/B" | ✅ | |
| `.slump-readout #slump-uncertainty` is a valid, sufficiently specific selector | ✅ | specificity (1,1,0); `#slump-uncertainty` is a child of `.slump-readout` in the Step 7 markup. The class qualifier is redundant given the ID, but harmless |
| No other `.wb-readout` / `#wb-value` references in `docs/style.css` | ✅ | only line 1698; the landscape query at 1706-1715 has none |
| `test/e2e/mobile-value-fit.spec.ts` does not count readout children | ✅ | it inspects `.slider-value` only; its `docOverflow` check won't trip on a readout wrap because `.readouts` is `flex-wrap: wrap` |
| `playwright.config.ts` defines projects named exactly `desktop` and `mobile` | ✅ | lines 42-58; mobile is Pixel 7 (412 px), **not** iPhone 14 despite the comment at line 11 |
| Step 10a's `evaluateAll` top-coordinate helper is at `readouts-strip.spec.ts:37-48` | ✅ exact | |
| Step 11a: the two W/B rows are `test/e2e/README.md:44-45` | ✅ exact | (project-column convention is `desktop+mobile`, not `both`) |
| Step 12c: `make test-js` will report "All 14 JS tests passed" | ✅ | `JS_TESTS` currently has 13 entries (`Makefile:110-123`) |
| Plan fact #7: `cholesky` and `solveTriangularLower` exported from `docs/gp.mjs:613` | ✅ exact | |
| Step 8d comment: "60 of 61 observations at 22 °C" | ✅ | 61 unique slump compositions; Temp = 22.0 for 60, 10.0 for 1 |
| Plan fact #5 / Step 14b: slump has no Material Source 0; range 0.8–11.1 in | ✅ | unique-composition MS counts are `{1: 27, 2: 34}`; min 0.8, max 11.1 |
| Step 16b: `slump.json` fits the perf budget | ✅ (not a risk) | `budgets.json` has no per-JSON bucket; ~14 KB against a 3000 KB `total` |
