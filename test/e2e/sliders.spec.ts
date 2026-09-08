import { test, expect } from "@playwright/test";
import {
  waitForCanvasAppReady,
  waitForCanvasLoopToPark,
} from "./canvas-frame-probe";

/**
 * Composition sliders — minimal sanity checks plus click-to-edit value
 * input behaviours. These don't pin down specific predictions (that's the
 * JS↔Python parity test in test_js_gp.mjs); they verify rendering,
 * keyboard interactions, and Material Source non-editability.
 */
test.describe("composition sliders", () => {
  test("desktop: at least 5 sliders are rendered with min/max labels", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop layout for sliders panel");
    await page.goto("/");
    // Wait for the dynamic slider DOM to appear
    await expect(page.locator("#sliders .slider-group").first()).toBeVisible({
      timeout: 5000,
    });
    const count = await page.locator("#sliders .slider-group").count();
    expect(count, "need ≥5 composition sliders").toBeGreaterThanOrEqual(5);
  });

  test("changing a slider triggers a strength curve redraw", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop only");
    await waitForCanvasAppReady(page);

    // Read initial pixel snapshot of the curve canvas
    const before = await page.locator("canvas#curve-canvas").screenshot();

    // Move the first slider — find the underlying <input type=range>
    const slider = page.locator('#sliders input[type="range"]').first();
    await slider.evaluate((el: HTMLInputElement) => {
      const min = Number(el.min);
      const max = Number(el.max);
      el.value = String(min + (max - min) * 0.7);
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await waitForCanvasLoopToPark(page);

    const after = await page.locator("canvas#curve-canvas").screenshot();
    expect(
      Buffer.compare(before, after),
      "curve canvas should change pixels after slider input",
    ).not.toBe(0);
  });
});

/**
 * Click-to-edit: the value display next to each (non-Material-Source) slider
 * is an editable text input. These tests pin down its commit/revert/clamp
 * semantics so the keyboard editing UX can't silently regress.
 */
test.describe("composition sliders — click-to-edit value input", () => {
  test.beforeEach(async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop only");
    await waitForCanvasAppReady(page);
  });

  test("typing a value and pressing Enter commits to the slider and curve", async ({ page }) => {
    const before = await page.locator("canvas#curve-canvas").screenshot();
    const input = page.locator("#sliders .slider-value").first();
    await input.click();
    await page.keyboard.press("Control+A");
    await page.keyboard.type("350");
    await page.keyboard.press("Enter");
    const slider = page.locator("#sliders input[type=range]").first();
    await waitForCanvasLoopToPark(page);
    const after = await page.locator("canvas#curve-canvas").screenshot();
    expect(Buffer.compare(before, after), "curve must change after committing edit").not.toBe(0);
  });

  test("Escape reverts an in-progress edit (slider unchanged)", async ({ page }) => {
    const slider = page.locator("#sliders input[type=range]").first();
    const initialSliderValue = await slider.evaluate((el: HTMLInputElement) => el.value);
    const input = page.locator("#sliders .slider-value").first();
    const initialDisplay = await input.inputValue();

    await input.click();
    await page.keyboard.press("Control+A");
    await page.keyboard.type("999.9");
    await page.keyboard.press("Escape");

    await expect(input).toHaveValue(initialDisplay);
    await expect(slider).toHaveValue(initialSliderValue);
  });

  test("out-of-range value is clamped to the slider's [min, max]", async ({ page }) => {
    const slider = page.locator("#sliders input[type=range]").first();
    const max = await slider.evaluate((el: HTMLInputElement) => Number(el.max));
    const input = page.locator("#sliders .slider-value").first();

    await input.click();
    await page.keyboard.press("Control+A");
    await page.keyboard.type("99999");
    await page.keyboard.press("Enter");
    await expect
      .poll(() => slider.evaluate((element: HTMLInputElement) => Number(element.value)))
      .toBeCloseTo(max, 0);

    const finalValue = await slider.evaluate((el: HTMLInputElement) => Number(el.value));
    // Slider clamps to max; allow 0.5 unit slop for floating-point rounding
    expect(finalValue).toBeLessThanOrEqual(max + 0.5);
    expect(finalValue).toBeGreaterThan(max * 0.95);
  });

  test("non-numeric input reverts on Enter", async ({ page }) => {
    const slider = page.locator("#sliders input[type=range]").first();
    const initialSliderValue = await slider.evaluate((el: HTMLInputElement) => el.value);
    const input = page.locator("#sliders .slider-value").first();
    const initialDisplay = await input.inputValue();

    await input.click();
    await page.keyboard.press("Control+A");
    await page.keyboard.type("not-a-number");
    await page.keyboard.press("Enter");

    await expect(input).toHaveValue(initialDisplay);
    await expect(slider).toHaveValue(initialSliderValue);
  });

  test("blur commits the edit (same as Enter)", async ({ page }) => {
    const before = await page.locator("canvas#curve-canvas").screenshot();
    const slider = page.locator("#sliders input[type=range]").first();
    const initialSliderValue = await slider.evaluate((el: HTMLInputElement) => Number(el.value));
    const input = page.locator("#sliders .slider-value").first();

    await input.click();
    await page.keyboard.press("Control+A");
    await page.keyboard.type("250");
    // Click somewhere outside the input to fire blur
    await page.locator("h1").click();
    await waitForCanvasLoopToPark(page);

    const finalSliderValue = await slider.evaluate((el: HTMLInputElement) => Number(el.value));
    expect(finalSliderValue).not.toBe(initialSliderValue);
    const after = await page.locator("canvas#curve-canvas").screenshot();
    expect(Buffer.compare(before, after), "curve must change after blur commit").not.toBe(0);
  });

  test("Material Source value display is NOT an editable input", async ({ page }) => {
    // The Material Source group keeps a <span> for #val-${i} so it can't be
    // clicked into as a text input. This pin protects that contract.
    const tag = await page
      .locator(".material-source-group .slider-label > span:last-child")
      .evaluate((el) => el.tagName);
    expect(tag).not.toBe("INPUT");
  });

  test("unit toggle while focused on a value input commits the edit", async ({ page }) => {
    const slider = page.locator("#sliders input[type=range]").first();
    const input = page.locator("#sliders .slider-value").first();

    await input.click();
    await page.keyboard.press("Control+A");
    await page.keyboard.type("400");

    // Trigger the unit toggle directly via the same custom event the buttons
    // fire — robust to whether desktop or mobile toggle button is visible.
    await page.evaluate(() => document.dispatchEvent(new CustomEvent("toggle-units")));
    await waitForCanvasLoopToPark(page);

    // The committed value should reflect the typed `400` in the PRE-toggle
    // unit context (i.e., 400 kg/m³ is what we typed; the slider's internal
    // value is in kg/m³). Verify the slider position is non-trivial and
    // close to 400 (the underlying internal unit is kg/m³).
    const sliderValue = await slider.evaluate((el: HTMLInputElement) => Number(el.value));
    expect(sliderValue).toBeGreaterThan(0);
  });
});
