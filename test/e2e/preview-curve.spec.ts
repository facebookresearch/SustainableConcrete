import { test, expect } from "@playwright/test";

/**
 * Regression pin for two related Material Source toggle bugs:
 *
 *   (1) `displayPreviewComp` used to lag behind `currentComposition` after
 *       a toggle, which made the dashed preview curve "ghost" the previous
 *       mix. The fix synchronizes both arrays inside the toggle handlers.
 *
 *   (2) The Mix Insight panel used to retain the previous mix's description
 *       after toggling, even when the new (median + other source)
 *       composition is not in the training set. The fix schedules an
 *       insight update on every toggle.
 *
 * These specs rely on the `?test=1` window hook (`window.__test`) which
 * exposes read-only views of `currentComposition` and `displayPreviewComp`.
 */
test.describe("preview curve composition sync", () => {
  test("displayPreviewComp matches currentComposition after Material Source toggle", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop only — toggle visible there");
    await page.goto("/?test=1");
    await expect(page.locator("#sliders .slider-group").first()).toBeVisible({ timeout: 5000 });
    await page.waitForFunction(() => typeof (window as any).__test !== "undefined");
    // The shell renders before the GP finishes building in the worker, so
    // wait for the model itself before asserting on predictions.
    await page.waitForFunction(() => (window as any).__test.modelReady === true, null, { timeout: 20000 });

    const toggleButtons = page.locator(".material-source-group .toggle-btn");
    expect(await toggleButtons.count(), "expected three toggle buttons").toBe(3);

    for (const idx of [1, 2, 0]) {
      await toggleButtons.nth(idx).click();
      // Curve transition is 350ms; wait it out before sampling state.
      await page.waitForTimeout(450);
      const result = await page.evaluate(() => {
        const t = (window as any).__test;
        return { current: t.currentComposition, preview: t.displayPreviewComp };
      });
      expect(result.preview).toEqual(result.current);
    }
  });
});

test.describe("mix insight refreshes on Material Source toggle", () => {
  test("does not retain previous mix's description after toggle", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mix insight only visible on desktop");
    await page.goto("/");
    // The insight panel is populated after the strength model resolves (now
    // off-thread in a worker), so wait for the app to finish wiring first.
    await expect(page.locator("#sliders .slider-group").first()).toBeVisible({ timeout: 15000 });
    const insightText = page.locator("#mix-insight-text");
    await expect(insightText).toBeVisible();

    // Wait for the initial mix insight to populate (median composition usually
    // matches a training mix, so we get a real description rather than the
    // placeholder). If that's not true on this dataset, we still proceed —
    // the test only asserts that the insight is REFRESHED, not its specific
    // content before/after.
    await page.waitForFunction(
      () => {
        const el = document.getElementById("mix-insight-text");
        return el !== null && el.textContent !== null;
      },
      { timeout: 5000 },
    );
    // Settle any in-flight content swap animation
    await page.waitForTimeout(700);
    const before = (await insightText.textContent())?.trim() ?? "";

    // Click whichever Material Source toggle is currently inactive
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)");
    await inactive.first().click();

    // Wait for: 350ms curve transition + 300ms scheduleInsightUpdate delay +
    // 300ms content-swap animation = ~950ms. Use 1300ms to be safe.
    await page.waitForTimeout(1300);

    const after = (await insightText.textContent())?.trim() ?? "";

    // The displayed insight must reflect the post-toggle composition, not the
    // previous one. Either it's a different real description (the new
    // composition matches a training mix), or it's the "not available"/
    // placeholder text. The one thing it must NOT be is the same text as
    // before (which would indicate the bug).
    expect(
      after === "" || after !== before,
      `mix-insight-text must update on Material Source toggle (still: "${after.slice(0, 80)}...")`,
    ).toBe(true);
  });
});

test.describe("strength curve transitions smoothly on Material Source toggle", () => {
  test("curve transition state is active immediately after toggle and clears after 350ms", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "strength curve canvas is most visible on desktop");
    // The pixel-diff alternative (sampling canvas at before/mid/after) was
    // racing with screenshot timing — the browser sometimes batched rAF
    // frames so the mid screenshot captured the post-transition state.
    // The deterministic substitute is a state hook on `_curveTransition`
    // exposed via `?test=1`. We assert two things:
    //   (1) Right after the click, `_curveTransition` is active.
    //   (2) After waiting longer than the 350 ms blend window plus a
    //       safety margin, the state has cleared back to `null`.
    // We additionally assert the canvas pixels change overall (toggle
    // produced a visible difference), which is robust because the wait
    // is ≥ 600 ms long.
    await page.goto("/?test=1");
    await expect(page.locator("canvas#curve-canvas")).toBeVisible();
    await expect(page.locator(".material-source-group .toggle-btn").first()).toBeVisible({ timeout: 5000 });
    await page.waitForFunction(() => typeof (window as any).__test !== "undefined");
    // The shell renders before the GP finishes building in the worker, so
    // wait for the model itself before asserting on predictions.
    await page.waitForFunction(() => (window as any).__test.modelReady === true, null, { timeout: 20000 });
    await page.waitForTimeout(800); // settle initial fade-ins / WASM init

    const curve = page.locator("canvas#curve-canvas");
    const before = await curve.screenshot();

    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await inactive.click();

    // Within a few milliseconds of the click handler firing, the transition
    // state should be active. Use waitForFunction with a tight timeout so
    // we don't accidentally observe the post-transition state.
    await page.waitForFunction(
      () => (window as any).__test.isCurveTransitionActive === true,
      null,
      { timeout: 100 },
    );

    // After the 350 ms duration plus generous safety margin, the state
    // should clear. The animation loop runs `drawStrengthCurve` which
    // sets `_curveTransition = null` once `t >= 1`.
    await page.waitForFunction(
      () => (window as any).__test.isCurveTransitionActive === false,
      null,
      { timeout: 1500 },
    );

    // Sanity: the toggle visibly changed the curve.
    const after = await curve.screenshot();
    expect(
      Buffer.compare(before, after),
      "post-toggle canvas must differ from pre-toggle canvas",
    ).not.toBe(0);
  });
});

test.describe("hover preview aligns with the committed prediction", () => {
  // Regression: the dashed hover-preview curve visibly disagreed with the
  // solid curve for the same mix. Two independent causes, both guarded here.

  test("preview curve is drawn on the same time grid as the main curve", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "scatter hover is a desktop interaction");
    // Cause 1: the preview used a fixed 48-point grid while the main curve
    // used 32 points during any interaction. Both curves were numerically
    // correct, but overlaying polylines sampled at different times left a
    // ~17% gap at t=0.07 d, in the steep gate-opening region. Sampling both
    // on one grid makes the gap identically zero at any resolution.
    await page.goto("/?test=1");
    await page.waitForFunction(() => typeof (window as any).__test !== "undefined");
    // The shell renders before the GP finishes building in the worker, so
    // wait for the model itself before asserting on predictions.
    await page.waitForFunction(() => (window as any).__test.modelReady === true, null, { timeout: 20000 });
    await page.waitForTimeout(800);

    const canvas = page.locator("canvas#scatter-canvas");
    await expect(canvas).toBeVisible();
    const box = await canvas.boundingBox();
    if (!box) throw new Error("scatter canvas has no bounding box");

    // Sweep the canvas so several points get hovered; record the verdict on
    // every frame where a preview was actually drawn.
    await page.evaluate(() => {
      (window as any).__gridSamples = [];
      const tick = () => {
        const v = (window as any).__test.previewSharesMainGrid;
        if (v !== null) (window as any).__gridSamples.push(v);
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });

    for (const fx of [0.3, 0.45, 0.6, 0.75]) {
      await canvas.hover({ position: { x: box.width * fx, y: box.height * 0.5 } });
      await page.waitForTimeout(180);
    }

    const samples: boolean[] = await page.evaluate(() => (window as any).__gridSamples);
    expect(samples.length, "expected the preview curve to be drawn at least once").toBeGreaterThan(0);
    expect(
      samples.every((v) => v === true),
      `preview used a different grid from the main curve on ${samples.filter((v) => !v).length}/${samples.length} frames`,
    ).toBe(true);
  });

  test("preview never feeds the GP a fractional Material Source", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "scatter hover is a desktop interaction");
    // Cause 2: displayPreviewComp lerped every dimension toward the hover
    // target, including the categorical Material Source. The preview curve is
    // predicted straight from displayPreviewComp, so for the ~1.4 s the lerp
    // took to converge the Hamming kernel saw a fractional class, which
    // matches no training row: the ghost rendered a collapsed "unseen class"
    // posterior (-3%) and passed through the neighbouring real class (+19%)
    // en route. Material Source must be snapped, never interpolated.
    const msIdx = await page.evaluate(async () => {
      const r = await fetch("model/compositions.json");
      const j = await r.json();
      return j.column_names.indexOf("Material Source");
    }).catch(() => -1);

    await page.goto("/?test=1");
    await page.waitForFunction(() => typeof (window as any).__test !== "undefined");
    // The shell renders before the GP finishes building in the worker, so
    // wait for the model itself before asserting on predictions.
    await page.waitForFunction(() => (window as any).__test.modelReady === true, null, { timeout: 20000 });
    await page.waitForTimeout(800);

    const idx = msIdx >= 0 ? msIdx : await page.evaluate(async () => {
      const r = await fetch("model/compositions.json");
      const j = await r.json();
      return j.column_names.indexOf("Material Source");
    });
    expect(idx, "Material Source column not found").toBeGreaterThanOrEqual(0);

    const canvas = page.locator("canvas#scatter-canvas");
    await expect(canvas).toBeVisible();
    const box = await canvas.boundingBox();
    if (!box) throw new Error("scatter canvas has no bounding box");

    await page.evaluate((msCol) => {
      (window as any).__msSamples = [];
      const tick = () => {
        const c = (window as any).__test.displayPreviewComp;
        if (c) (window as any).__msSamples.push(c[msCol]);
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }, idx);

    // Sweep across the scatter so the hover target crosses material classes.
    for (const fx of [0.2, 0.4, 0.55, 0.7, 0.85]) {
      await canvas.hover({ position: { x: box.width * fx, y: box.height * 0.5 } });
      await page.waitForTimeout(160);
    }

    const samples: number[] = await page.evaluate(() => (window as any).__msSamples);
    expect(samples.length, "expected displayPreviewComp samples").toBeGreaterThan(10);
    const fractional = samples.filter((v) => !Number.isInteger(v));
    expect(
      fractional.length,
      `Material Source was fractional on ${fractional.length}/${samples.length} frames, e.g. ${fractional.slice(0, 5).join(", ")}`,
    ).toBe(0);
    // Sanity: the sweep actually visited more than one class, otherwise the
    // assertion above is vacuous.
    expect(new Set(samples).size, "sweep never crossed a material class boundary").toBeGreaterThan(1);
  });
});
