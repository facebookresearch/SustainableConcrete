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
| `header-layout.spec.ts`       | On mobile, citation controls remain reachable without overlapping header actions | mobile |
| `header-layout.spec.ts`       | On mobile, the 5 visible header items are evenly spaced (gap deltas < 6px) | mobile |
| `header-layout.spec.ts`       | Header stays sticky at the top of the viewport when scrolling             | desktop+mobile |
| `header-layout.spec.ts`       | No horizontal scroll on mobile (drag pan is locked)                       | mobile |
| `background-coverage.spec.ts` | Photograph is viewport-fixed with stable cover/focal framing through panel content growth, local/document scrolling, and mobile orientation changes | desktop+mobile+mobile-webkit |
| `panel-geometry.spec.ts`      | Desktop panels form top-aligned intrinsic stacks; Strength owns Strength Curve, Filters, References, and the single semantic credit subtree in that order | desktop+desktop-webkit |
| `panel-geometry.spec.ts`      | Desktop Composition is intrinsically sized with no local scroll or viewport-derived cap; References grows intrinsically to a stable viewport-minus-header cap and leaves overflow exclusively to `.ref-list` | desktop+desktop-webkit |
| `panel-geometry.spec.ts`      | Potential scroll regions receive `tabindex=0` only while visible, active, and actually overflowing | desktop+mobile+mobile-webkit |
| `panel-geometry.spec.ts`      | Mobile Scatter keeps its outer body inert while Composition owns the unified local scroller | mobile+mobile-webkit |
| `plot-geometry.spec.ts`       | Scatter and Strength share responsive desktop `80/20/15/54` and mobile `84/20/15/70` inset contracts, aligned drawables, exact drawable equality, HiDPI backing fidelity, landscape geometry, and layout stability; Tradeoffs omits redundant instructional copy and keeps the X selector centered without a reserved hint band | desktop+mobile+mobile-webkit+desktop-webkit |
| `ingredient-insight.spec.ts`  | Cement is initially selected; exactly one semantic ingredient button remains selected; content commits immediately while one interruptible intrinsic-size transaction cleans paint-only clones and preserves cross-engine focus | desktop+mobile+mobile-webkit+desktop-webkit |
| `ingredient-insight.spec.ts`  | Composition remains complete, compact, and fully utilized; Material Source labels remain one-line/equal-height and every primary panel preserves essential information | desktop+mobile+mobile-webkit+desktop-webkit |
| `scatter-toggle.spec.ts`      | Native radio faces exchange horizontally for X and vertically for Y; full labels retain stable identity, arrow-key focus, synchronized `350ms` position/paint easing, opaque overlap masking, frame-sampled bidirectional Y motion, painted-state rapid reversal, and current-point-inclusive axis endpoints across units and responsive Chromium/WebKit lanes | desktop+mobile+mobile-webkit+desktop-webkit |
| `mobile-panel-toggle.spec.ts` | Tapping "Composition" hides Scatter and Filters, gives the unified body sole local-scroll ownership, and transfers inert state immediately | mobile+mobile-webkit |
| `mobile-panel-toggle.spec.ts` | Returning to "Performance Tradeoffs" restores Scatter and its single live filter subtree; live reduced-motion changes settle the crossfade and state survives breakpoint reparenting | mobile+mobile-webkit |
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
| `preview-curve.spec.ts`       | Initial Mix population is immediate; later Mix Insight changes commit semantics immediately and use the shared interruptible intrinsic-size transaction without stale content | desktop |
| `preview-curve.spec.ts`       | Strength curve transition state is active right after MS toggle and clears after ~350 ms | desktop |
| `preview-curve.spec.ts`       | Edge Material Source preview pulses stay inside the nearest intentional Composition clip under the shared all-sided safe-area contract and stack above sibling controls | desktop+mobile |
| `lengthscale-identifiability.spec.ts` | Served `docs/model/strength.json` has every feature lengthscale below 100 (else sliders go unresponsive) | desktop |
| `font-uniformity.spec.ts`     | `.mix-insight-text`, `.ingredient-insight-text`, and an expanded `.ref-desc` share computed font size | desktop+mobile |
| `references.spec.ts`          | Stable wrapping metadata keeps citation actions at inline-end and summaries full-width; native disclosures retain focus/inert/interruption semantics, mobile summaries meet 44px, intrinsic growth precedes scrolling, and `.ref-list` focusability follows overflow | desktop+mobile+mobile-webkit+desktop-webkit |
| `reduced-motion.spec.ts`      | Startup X/Y axis changes bypass Scatter transition state under reduced motion; live preference changes settle Y geometry and all structural/canvas transitions, and normal `350ms` Y motion resumes when the preference is restored | desktop+desktop-webkit |
| `og-meta.spec.ts`             | Required Open Graph + Twitter Card meta tags present with expected content | desktop |
| `og-meta.spec.ts`             | `og-image.jpg` is reachable, JPEG, and within 50–250 KB budget            | desktop |
| `readouts-strip.spec.ts`      | Desktop: GWP, Cost, and W/B readouts are all visible                      | desktop |
| `readouts-strip.spec.ts`      | Mobile: GWP, Cost, and W/B remain present and fit the bounded curve panel  | mobile |
| `seo.spec.ts`                 | `<meta name="description">`, canonical link, and JSON-LD WebApplication present | desktop |
| `seo.spec.ts`                 | `/robots.txt` and `/sitemap.xml` reachable and well-formed                | desktop |
| `mobile-slider-layout.spec.ts`| Label, slider, and info-row stack vertically without overlap              | mobile |
| `mobile-slider-layout.spec.ts`| Ingredient names are left-aligned (consistent left edge across rows)      | mobile |
| `mobile-slider-layout.spec.ts`| Ingredient name and info-row min share the same left edge                 | mobile |
| `mobile-slider-layout.spec.ts`| Ingredient name and value input share a vertical centerline (±2 px)       | mobile |
| `mobile-slider-layout.spec.ts`| Ingredient controls stay compact and the selected control reads as a pill | mobile |
| `mobile-slider-layout.spec.ts`| Mobile font hierarchy: label fontSize ≥ info-row fontSize                 | mobile |
| `mobile-slider-layout.spec.ts`| All slider tracks have uniform width and aligned left/right edges         | mobile |
| `mobile-slider-layout.spec.ts`| Slider uses ≥94% of its owner while keeping a symmetric 5–8px effect gutter | mobile+mobile-webkit |
| `mobile-slider-layout.spec.ts`| Value input offsetHeight ≥ 32 px (tap target)                             | mobile |
| `mobile-slider-layout.spec.ts`| Material Source: all three labels stay one line in equal 44px mobile targets; redundant value-span is hidden | mobile+mobile-webkit |
| `mobile-slider-layout.spec.ts`| Value input glyph-end aligns with info-row max bound (right-edge)         | mobile |
| `mobile-slider-layout.spec.ts`| Slider preview marker lands on the visible track when scatter is hovered  | mobile |
| `scatter-filter.spec.ts`      | Filter min/max placeholders fit fully inside the input box (no spinner clip) | desktop |
| `scatter-filter.spec.ts`      | Filters show two rows by default/mobile, three at 1728×1000, and four at 1728×1117; later rows scroll only `#filter-rows` without moving either plot, References, or the credit | desktop+mobile |
| `scatter-filter.spec.ts`      | Add/remove/clear use interruptible intrinsic motion with immediate model/inert/focus semantics, continuous bottom anchoring with manual-scroll cancellation, clean supersession, categorical rows, native touch scrolling, and boundary chaining | desktop+mobile+desktop-webkit |
| `filter-motion.spec.ts`       | `0→1`, `1→0`, `2→1`, clear-all, and rapid supersession animate shell chrome and row contributions monotonically with elapsed-time velocity bounds and no stale state | desktop+desktop-webkit |
| `mobile-value-fit.spec.ts`    | Metric values fit (no panel overflow) at first paint                      | mobile |
| `mobile-value-fit.spec.ts`    | Fractional metric values (.3) fit                                         | mobile |
| `mobile-value-fit.spec.ts`    | Imperial values fit (worst-case mass conversion)                          | mobile |
| `mobile-value-fit.spec.ts`    | Imperial max-bound values fit                                             | mobile |
| `mobile-value-fit.spec.ts`    | At 320 px, composition values plus numeric/categorical filter controls fit without clipping or horizontal document overflow | mobile |
| `visual-regression.spec.ts`   | Real viewport captures cover the managed dashboard, compact desktop Composition and Tradeoffs selector lanes, open References metadata/disclosure layout on desktop and 320px mobile, local scroll boundaries, and representative effect envelopes | desktop+mobile |
| `touch-targets.spec.ts`       | Sliders, Material Source, axis pills, and filter controls meet WCAG/HIG sizes across Chromium/WebKit without fattening visual tracks | desktop+mobile+mobile-webkit |
| `mobile-behavior.spec.ts`     | Material Source tap commits an integral class; curve transition runs and clears; click-to-edit commits; curve non-blank | mobile |

## Running

```bash
# install deps + browsers (first time)
npm install
npx playwright install --with-deps chromium webkit

# fast local loop: desktop + mobile Chromium
make test-e2e

# comprehensive pre-merge loop: all four Chromium/WebKit projects
make test-e2e-all

# run every configured project in a single Playwright invocation
npm run test:e2e

# run only mobile Chromium project
npm run test:e2e -- --project=mobile

# run scoped iOS Safari engine-sensitive contracts
npm run test:e2e -- --project=mobile-webkit

# run the focused desktop Safari/WebKit compatibility smoke suite
npm run test:e2e -- --project=desktop-webkit

# run focused plot geometry in Chromium and WebKit
make test-plot-geometry

# headed (watch the browser)
npm run test:e2e:headed

# interactive debugger
npm run test:e2e:ui

# show last HTML report
npm run test:e2e:report
```

## Structural motion and effect safety

Structural changes to Filters, Mix Insight, Ingredient Insight, and reference descriptions use interruptible intrinsic-size transactions. Filters coordinate shell margin/padding with in-flow row size, opacity, and wrapper-owned trailing spacing under one composite settlement handle. Tests wait for observable settlement (no running effective animations, final DOM state, focus, or overflow) rather than sleeping for a guessed duration; frame-sampled motion uses rAF timestamps so dropped frames do not look like layout snaps. Initial layout, responsive reparenting, unit/theme changes, ordinary scrolling, and reduced-motion execution remain immediate.

Desktop axis selector paint is intentionally slimmer than mobile hit geometry. The responsive contracts are desktop `80/20/15/54` and mobile `84/20/15/70` for left/right/top/bottom insets. Desktop headings bottom-align above a `2px` canvas gap, leaving `17px` from title text to drawable while retaining a `15px` top effect envelope. Each lane reserves its selector, effect envelope, tick offset, worst-case painted label, and the shared `8px` visual gap. X placement follows its measured painted descent. Y ticks use deterministic compact `k` notation at and above `1,000`, and Y placement is anchored to the fixed `27px` worst-case tick-paint lane, so units and nice-tick thresholds cannot move the selector horizontally. The live Y gap is bounded rather than positioned from live text: it remains clipping-safe at or above `8px` while avoiding a visibly oversized gutter. Mobile faces remain at least `44×44px` with the established reserves.

Scatter transitions use one authoritative range formula for direct renders and both animation endpoints: catalog predictions plus the editable current point, with the existing `1.05` X and `1.1` Y headroom. The exact `t=1` endpoint is painted before transition state clears, and interruption retargets from the last painted arrays, maxima, Pareto state, and current point. During Y exchanges each full-label pill has an opaque theme-card base beneath the selected accent, preventing double-painted text when the rotating pills overlap without shortening either label.

The managed `1728×1000` and `1728×1117` viewports render equal Scatter/Strength drawables at `480×320` in Chromium and WebKit. That is `153,600 px²`, about `12.8%` above the former `448×304` (`136,192 px²`) ceiling. The redundant Scatter instruction is omitted at every responsive size, so no hint band consumes plot-panel space. Units and objectives never change drawable dimensions.

References metadata uses one stable wrapping row: authors flex, citation actions remain at inline-end, and the native disclosure stays full-width below it. `--references-usable-cap` is maintained from visual viewport height minus sticky-header height and a `16px` bottom gap; it does not depend on document position. The panel grows intrinsically until that cap, then only `.ref-list` scrolls and enters sequential focus. Desktop WebKit smoke coverage samples the stable layout, native disclosure, overflow-cap, and focusability contracts; Chromium and mobile WebKit retain the broader motion matrix.

Every outward focus, hover, preview, selected, shadow, or transform envelope must fit inside its nearest intentional clipping ancestor on all four sides. New effects or clipping/scroll owners require both an `effect-envelope.ts` containment assertion and, where paint is user-visible, a strict cropped Linux snapshot.

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

- **`page.waitForTimeout` for animation settlement** — replace with `expect.poll()`,
  `animation.finished`, or a DOM/geometry condition. Fixed waits
  are slow on CI and still flaky.
- **Tests that retry to mask flakiness** — fix the race condition. CI retries
  exist for transient infrastructure issues, not for "sometimes the animation
  hasn't finished".
- **Tests with no plain-English description** — every test should pin down
  one named property. If you can't name it, you don't need it yet.
- **Tests that don't fail when the feature breaks** — write the assertion
  first, break the feature, confirm the test fails, then fix the feature.
