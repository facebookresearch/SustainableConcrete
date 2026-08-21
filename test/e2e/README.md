# End-to-end tests for the BOxCrete website

These Playwright specs pin down the **design and behavioral invariants**
of the interactive site at `docs/`. Every spec corresponds to at least
one named invariant — when you find yourself manually checking a
property on a PR, add a spec for it before merging.

## Invariants currently covered

| Spec file                     | Invariant                                                                 | Project(s) |
|-------------------------------|---------------------------------------------------------------------------|------------|
| `home-loads.spec.ts`          | Home page loads with no console/page errors and core canvases visible     | desktop+mobile |
| `home-loads.spec.ts`          | Strength curve canvas renders within a few seconds of load                | desktop+mobile |
| `header-layout.spec.ts`       | Theme toggle is the rightmost element in the header                       | desktop+mobile |
| `header-layout.spec.ts`       | On desktop, cite group is to the left of the theme toggle and not overlapping | desktop |
| `header-layout.spec.ts`       | On mobile, cite group is hidden                                           | mobile |
| `header-layout.spec.ts`       | On mobile, the 5 visible header items are evenly spaced (gap deltas < 6px) | mobile |
| `header-layout.spec.ts`       | Header stays sticky at the top of the viewport when scrolling             | desktop+mobile |
| `header-layout.spec.ts`       | No horizontal scroll on mobile (drag pan is locked)                       | mobile |
| `scatter-toggle.spec.ts`      | X-axis label cycles when `#toggle-x` is clicked                           | desktop |
| `scatter-toggle.spec.ts`      | Y-axis label cycles when `#toggle-day` is clicked                         | desktop |
| `mobile-panel-toggle.spec.ts` | Tapping "Composition" hides scatter content and shows sliders             | mobile |
| `mobile-panel-toggle.spec.ts` | Tapping "Performance Tradeoffs" hides sliders and shows scatter           | mobile |
| `theme-toggle.spec.ts`        | Theme toggle flips `data-theme` attribute on `<html>`                     | desktop+mobile |
| `theme-toggle.spec.ts`        | Theme choice persists across reload via localStorage                      | desktop+mobile |
| `about-modal.spec.ts`         | About modal opens on link click and closes on `Escape`                    | desktop+mobile |
| `about-modal.spec.ts`         | About modal closes on overlay click and on the × button                   | desktop+mobile |
| `sliders.spec.ts`             | At least one slider is rendered with min/max labels                       | desktop |
| `sliders.spec.ts`             | Slider input redraws the strength curve canvas                            | desktop |
| `sliders.spec.ts`             | Click-to-edit: typing a value + Enter commits to slider and curve         | desktop |
| `sliders.spec.ts`             | Click-to-edit: Escape reverts an in-progress edit (slider unchanged)      | desktop |
| `sliders.spec.ts`             | Click-to-edit: out-of-range typed value is clamped to slider [min, max]   | desktop |
| `sliders.spec.ts`             | Click-to-edit: non-numeric input reverts on Enter                         | desktop |
| `sliders.spec.ts`             | Click-to-edit: blur commits the edit (same as Enter)                      | desktop |
| `sliders.spec.ts`             | Material Source value display is NOT an editable input                    | desktop |
| `sliders.spec.ts`             | Unit toggle while focused on a value input commits the edit               | desktop |
| `preview-curve.spec.ts`       | `displayPreviewComp` matches `currentComposition` after Material Source toggle | desktop |
| `preview-curve.spec.ts`       | Mix Insight description is refreshed (not stale) after Material Source toggle | desktop |
| `preview-curve.spec.ts`       | Strength curve transition state is active right after MS toggle and clears after ~350 ms | desktop |
| `lengthscale-identifiability.spec.ts` | Served `docs/model/strength.json` has every feature lengthscale below 100 (else sliders go unresponsive) | desktop |
| `font-uniformity.spec.ts`     | `.mix-insight-text`, `.ingredient-insight-text`, `.ref-desc` share computed font size | desktop+mobile |
| `og-meta.spec.ts`             | Required Open Graph + Twitter Card meta tags present with expected content | desktop |
| `og-meta.spec.ts`             | `og-image.jpg` is reachable, JPEG, and within 50–250 KB budget            | desktop |
| `readouts-strip.spec.ts`      | Desktop: GWP, Cost, and Slump readouts are all visible with ±2σ           | desktop |
| `readouts-strip.spec.ts`      | Mobile: ±2σ suffixes and the GWP unit are hidden so readouts stay on one line | mobile |
| `readouts-strip.spec.ts`      | Mortar (Source A) shows "n/a" — no slump data for that material source    | desktop+mobile |
| `readouts-strip.spec.ts`      | Unit toggle rescales the slump value 25.4x, not just its label            | desktop+mobile |
| `seo.spec.ts`                 | `<meta name="description">`, canonical link, and JSON-LD WebApplication present | desktop |
| `seo.spec.ts`                 | `/robots.txt` and `/sitemap.xml` reachable and well-formed                | desktop |
| `mobile-slider-layout.spec.ts`| Label, slider, and info-row stack vertically without overlap              | mobile |
| `mobile-slider-layout.spec.ts`| Ingredient names are left-aligned (consistent left edge across rows)      | mobile |
| `mobile-slider-layout.spec.ts`| Ingredient name and info-row min share the same left edge                 | mobile |
| `mobile-slider-layout.spec.ts`| Ingredient name and value input share a vertical centerline (±2 px)       | mobile |
| `mobile-slider-layout.spec.ts`| Ingredient name shows a dashed underline as click affordance (no border)  | mobile |
| `mobile-slider-layout.spec.ts`| Mobile font hierarchy: label fontSize ≥ info-row fontSize                 | mobile |
| `mobile-slider-layout.spec.ts`| All slider tracks have uniform width and aligned left/right edges         | mobile |
| `mobile-slider-layout.spec.ts`| Slider is centered in the panel and narrower than full panel width        | mobile |
| `mobile-slider-layout.spec.ts`| Value input offsetHeight ≥ 32 px (tap target)                             | mobile |
| `mobile-slider-layout.spec.ts`| Material Source: toggle row visible, redundant value-span hidden          | mobile |
| `mobile-slider-layout.spec.ts`| Value input glyph-end aligns with info-row max bound (right-edge)         | mobile |
| `mobile-slider-layout.spec.ts`| Slider preview marker lands on the visible track when scatter is hovered  | mobile |
| `scatter-filter.spec.ts`      | Filter min/max placeholders fit fully inside the input box (no spinner clip) | desktop |
| `mobile-value-fit.spec.ts`    | Metric values fit (no panel overflow) at first paint                      | mobile |
| `mobile-value-fit.spec.ts`    | Fractional metric values (.3) fit                                         | mobile |
| `mobile-value-fit.spec.ts`    | Imperial values fit (worst-case mass conversion)                          | mobile |
| `mobile-value-fit.spec.ts`    | Imperial max-bound values fit                                             | mobile |
| `mobile-value-fit.spec.ts`    | Narrow viewport (320 px): values fit at all unit/value combinations       | mobile |
| `visual-regression.spec.ts`   | Full-page screenshot matches committed Linux baseline (active on Linux)   | desktop+mobile |
| `touch-targets.spec.ts`       | Slider hit area meets WCAG 2.2 AA (24 px) / Apple HIG (44 px mobile)      | desktop+mobile |
| `mobile-behavior.spec.ts`     | Material Source tap commits an integral class; curve transition runs and clears; click-to-edit commits; curve non-blank | mobile |

## Running

```bash
# install deps + browsers (first time)
npm install
npx playwright install --with-deps chromium

# run all tests
npm run test:e2e

# run only mobile project
npm run test:e2e -- --project=mobile

# headed (watch the browser)
npm run test:e2e:headed

# interactive debugger
npm run test:e2e:ui

# show last HTML report
npm run test:e2e:report
```

## Updating visual snapshots

Visual regression snapshots are **OS-specific** — fonts and anti-aliasing
differ between macOS, Linux, and Windows. CI runs on Ubuntu, so the
snapshots committed must be Linux-rendered.

Baselines live in `visual-regression.spec.ts-snapshots/`. The specs run on
Linux (what CI uses) and skip automatically elsewhere, so a local macOS run
stays green instead of diffing against a renderer it can never match.

To regenerate after an intentional UI change:

1. **Locally (recommended)** — run in the official Playwright Docker image.
   The tag MUST match the Playwright version in `package-lock.json`, or the
   browser build differs and the baselines will not match CI:
   ```bash
   node -p "require('./package-lock.json').packages['node_modules/@playwright/test'].version"

   docker run --rm --network host -v $(pwd):/work -w /work \
     mcr.microsoft.com/playwright:v1.59.1-noble \
     bash -lc "npx --yes playwright@1.59.1 test --grep @visual --update-snapshots"
   ```
2. **Via GitHub Actions** — manually dispatch the `e2e` workflow with
   `update_snapshots: true` and commit the resulting artifact.

Visual specs are tagged `@visual`. Verify a regeneration by re-running
**without** `--update-snapshots`: it must pass by comparison, not by writing.

## Adding a new invariant

1. Decide which spec file it belongs in (or create a new one with a clear name).
2. Write the test as a single `test('<plain-English invariant>', ...)`.
3. Use `testInfo.project.name` to scope to desktop/mobile when needed.
4. Add a row to the table above.
5. Run locally to confirm it passes.
6. PR it.

## Anti-patterns to avoid

- **`page.waitForTimeout` longer than 500ms** — replace with `expect(...).toPass()`
  or `waitForFunction` to wait for the actual condition. Long fixed waits
  are slow on CI and still flaky.
- **Tests that retry to mask flakiness** — fix the race condition. CI retries
  exist for transient infrastructure issues, not for "sometimes the animation
  hasn't finished".
- **Tests with no plain-English description** — every test should pin down
  one named property. If you can't name it, you don't need it yet.
- **Tests that don't fail when the feature breaks** — write the assertion
  first, break the feature, confirm the test fails, then fix the feature.
