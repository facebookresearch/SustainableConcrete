import { test, expect } from "@playwright/test";

/**
 * Mobile behavioural coverage.
 *
 * The mobile project previously ran ~24 layout assertions (bounding boxes,
 * overflow, font fit) and essentially zero behaviour: no mobile test exercised
 * the Material Source toggle, the curve transition, or click-to-edit. Those are
 * taps, not hovers, so they work fine on touch — the existing specs skip mobile
 * only because the controls sit behind the `#mobile-show-sliders` view toggle.
 *
 * This file opens that view and then drives the same interactions.
 */

async function openSliders(page: import("@playwright/test").Page) {
  await page.goto("/?test=1");
  await page.waitForFunction(() => typeof (window as any).__test !== "undefined");
  // The shell renders before the GP finishes building in the worker, so wait
  // for the model itself before asserting on predictions.
  await page.waitForFunction(
    () => (window as any).__test.modelReady === true,
    null,
    { timeout: 20000 },
  );
  await page.locator("#mobile-show-sliders").click();
  await expect(page.locator(".mobile-sliders-view")).toBeVisible({ timeout: 2000 });
}

test.describe("mobile behaviour", () => {
  test.beforeEach(async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "mobile", "mobile-only behaviour");
  });

  test("Material Source toggle commits an integral class on tap", async ({ page }) => {
    await openSliders(page);

    const msIdx = await page.evaluate(async () => {
      const j = await (await fetch("model/compositions.json")).json();
      return j.column_names.indexOf("Material Source");
    });
    expect(msIdx).toBeGreaterThanOrEqual(0);

    const before = await page.evaluate(
      (i) => (window as any).__test.currentComposition[i], msIdx);

    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await expect(inactive).toBeVisible();
    await inactive.click();

    // Sample across the whole transition: a categorical dim must never be
    // interpolated, on touch just as on desktop.
    const samples: number[] = [];
    for (let i = 0; i < 12; i++) {
      samples.push(await page.evaluate((k) => (window as any).__test.displayPreviewComp[k], msIdx));
      await page.waitForTimeout(40);
    }
    const fractional = samples.filter((v) => !Number.isInteger(v));
    expect(fractional.length, `fractional Material Source: ${JSON.stringify(fractional)}`).toBe(0);

    const after = await page.evaluate(
      (i) => (window as any).__test.currentComposition[i], msIdx);
    expect(after, "tapping an inactive class must change the committed class").not.toBe(before);
    expect(Number.isInteger(after)).toBe(true);
  });

  test("curve transition runs and clears on tap", async ({ page }) => {
    await openSliders(page);
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await inactive.click();
    await page.waitForFunction(
      () => (window as any).__test.isCurveTransitionActive === true, null, { timeout: 300 });
    await page.waitForFunction(
      () => (window as any).__test.isCurveTransitionActive === false, null, { timeout: 3000 });
  });

  test("click-to-edit commits a typed value on mobile", async ({ page }) => {
    await openSliders(page);
    // `.slider-value` is the click-to-edit text input built in buildSliders();
    // the Material Source row uses a plain span instead, so filter to inputs.
    const input = page.locator(".mobile-sliders-view input.slider-value").first();
    await expect(input).toBeVisible();
    const idx = await input.getAttribute("data-idx");
    expect(idx, "value input should carry data-idx").not.toBeNull();

    await input.click();
    await input.fill("300");
    await input.press("Enter");
    await page.waitForTimeout(700);
    const committed = await page.evaluate(
      (i) => (window as any).__test.currentComposition[Number(i)], idx);
    expect(committed, "typed value should commit to the composition").toBeCloseTo(300, 0);
  });

  test("strength curve renders non-blank after interaction", async ({ page }) => {
    await openSliders(page);
    await page.locator("#mobile-show-scatter").click();
    const nonBlank = await page.evaluate(() => {
      const c = document.querySelector("canvas#curve-canvas") as HTMLCanvasElement;
      if (!c) return false;
      const d = c.getContext("2d")!.getImageData(0, 0, c.width, c.height).data;
      for (let i = 3; i < d.length; i += 400) if (d[i] !== 0) return true;
      return false;
    });
    expect(nonBlank, "curve canvas is blank on mobile").toBe(true);
  });
});
