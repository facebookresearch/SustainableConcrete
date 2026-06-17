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

    const toggleButtons = page.locator(".material-source-group .toggle-btn");
    const nButtons = await toggleButtons.count();
    expect(nButtons, "expected at least two toggle buttons").toBeGreaterThanOrEqual(2);

    // Click through the available source classes: 1 -> 0 -> last (back to non-default).
    const sequence = nButtons >= 3 ? [1, 0, 2] : [1, 0, 1];
    for (const idx of sequence) {
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
    // The deterministic substitute is a state hook on `_msCurveTransition`
    // exposed via `?test=1`. We assert two things:
    //   (1) Right after the click, `_msCurveTransition` is active.
    //   (2) After waiting longer than the 350 ms blend window plus a
    //       safety margin, the state has cleared back to `null`.
    // We additionally assert the canvas pixels change overall (toggle
    // produced a visible difference), which is robust because the wait
    // is ≥ 600 ms long.
    await page.goto("/?test=1");
    await expect(page.locator("canvas#curve-canvas")).toBeVisible();
    await expect(page.locator(".material-source-group .toggle-btn").first()).toBeVisible({ timeout: 5000 });
    await page.waitForFunction(() => typeof (window as any).__test !== "undefined");
    await page.waitForTimeout(800); // settle initial fade-ins / WASM init

    const curve = page.locator("canvas#curve-canvas");
    const before = await curve.screenshot();

    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await inactive.click();

    // Within a few milliseconds of the click handler firing, the transition
    // state should be active. Use waitForFunction with a tight timeout so
    // we don't accidentally observe the post-transition state.
    await page.waitForFunction(
      () => (window as any).__test.isMsCurveTransitionActive === true,
      null,
      { timeout: 100 },
    );

    // After the 350 ms duration plus generous safety margin, the state
    // should clear. The animation loop runs `drawStrengthCurve` which
    // sets `_msCurveTransition = null` once `t >= 1`.
    await page.waitForFunction(
      () => (window as any).__test.isMsCurveTransitionActive === false,
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
