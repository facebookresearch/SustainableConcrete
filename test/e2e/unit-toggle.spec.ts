import { test, expect } from "@playwright/test";
import { waitForCanvasLoopToPark } from "./canvas-frame-probe";

/**
 * Unit-toggle behaviour for the composition setter panel:
 *   - per-row unit suffixes flip between metric (kg/m³ + °C) and
 *     imperial (lb/yd³ + °F)
 *   - the temperature display uses the offset conversion F = C × 9/5 + 32,
 *     not just a multiplicative factor (so 22°C → 71.6°F, NOT 22°F)
 *   - the round-trip (metric → imperial → metric) returns the same display
 *
 * Toggling is dispatched as the same `toggle-units` custom event that
 * both the desktop and mobile unit buttons fire, so this test is layout-
 * agnostic.
 */
test.describe("composition panel — unit toggle", () => {
  test.beforeEach(async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop only");
    await page.goto("/?test=1");
    await expect(page.locator("#sliders .slider-group").first()).toBeVisible({
      timeout: 5000,
    });
  });

  test("metric default shows kg/m³ on mass rows and °C on temperature row", async ({ page }) => {
    const units = await page.locator("#sliders .slider-unit").allInnerTexts();
    expect(units.length).toBeGreaterThan(0);
    // Expect at least one mass-row label and exactly one temp-row label.
    expect(units).toContain("kg/m³");
    expect(units).toContain("°C");
    // No imperial labels in the default view.
    expect(units).not.toContain("lb/yd³");
    expect(units).not.toContain("°F");
  });

  test("toggle to imperial flips mass to lb/yd³ and temperature to °F", async ({ page }) => {
    await page.evaluate(() =>
      document.dispatchEvent(new CustomEvent("toggle-units")),
    );
    await waitForCanvasLoopToPark(page);
    const units = await page.locator("#sliders .slider-unit").allInnerTexts();
    expect(units).toContain("lb/yd³");
    expect(units).toContain("°F");
    expect(units).not.toContain("kg/m³");
    expect(units).not.toContain("°C");
  });

  test("temperature value uses the °F offset conversion (22°C → 71.6°F)", async ({ page }) => {
    // Locate the temperature row by its `Temperature` ingredient name.
    // The friendly label is rendered by `buildSliders` (`Temp` → "Temperature").
    const tempRow = page.locator("#sliders .slider-group").filter({
      has: page.locator("button.ingredient-name", { hasText: "Temperature" }),
    });
    await expect(tempRow).toHaveCount(1);

    const valueBefore = await tempRow.locator(".slider-value").inputValue();
    const celsius = parseFloat(valueBefore);
    expect(Number.isFinite(celsius)).toBe(true);

    // Flip to imperial
    await page.evaluate(() =>
      document.dispatchEvent(new CustomEvent("toggle-units")),
    );
    await waitForCanvasLoopToPark(page);

    const valueAfter = await tempRow.locator(".slider-value").inputValue();
    const fahrenheit = parseFloat(valueAfter);
    const expectedF = (celsius * 9) / 5 + 32;
    // `.toFixed(1)` rendering, so up to 0.05 absolute tolerance.
    expect(Math.abs(fahrenheit - expectedF)).toBeLessThan(0.1);
    // Sanity: 22°C must NOT round-trip to 22°F via a naive identity.
    if (Math.abs(celsius - 22) < 0.5) {
      expect(Math.abs(fahrenheit - 71.6)).toBeLessThan(0.5);
    }
  });

  test("round-trip metric → imperial → metric preserves the temperature display", async ({ page }) => {
    const tempRow = page.locator("#sliders .slider-group").filter({
      has: page.locator("button.ingredient-name", { hasText: "Temperature" }),
    });
    const before = await tempRow.locator(".slider-value").inputValue();

    await page.evaluate(() =>
      document.dispatchEvent(new CustomEvent("toggle-units")),
    );
    await waitForCanvasLoopToPark(page);
    await page.evaluate(() =>
      document.dispatchEvent(new CustomEvent("toggle-units")),
    );
    await waitForCanvasLoopToPark(page);

    const after = await tempRow.locator(".slider-value").inputValue();
    expect(parseFloat(after)).toBeCloseTo(parseFloat(before), 1);
  });

  test("min/max info-row for temperature reflects the active unit system", async ({ page }) => {
    const tempRow = page.locator("#sliders .slider-group").filter({
      has: page.locator("button.ingredient-name", { hasText: "Temperature" }),
    });
    const metricBounds = (await tempRow.locator(".info-row span").allInnerTexts()).map(
      (s) => parseInt(s, 10),
    );
    expect(metricBounds.length).toBe(2);
    const [metricMin, metricMax] = metricBounds;

    await page.evaluate(() =>
      document.dispatchEvent(new CustomEvent("toggle-units")),
    );
    await waitForCanvasLoopToPark(page);
    const imperialBounds = (await tempRow.locator(".info-row span").allInnerTexts()).map(
      (s) => parseInt(s, 10),
    );
    const [imperialMin, imperialMax] = imperialBounds;

    // F = C × 9/5 + 32, then `.toFixed(0)` → off-by-one rounding ok
    expect(Math.abs(imperialMin - ((metricMin * 9) / 5 + 32))).toBeLessThan(1.5);
    expect(Math.abs(imperialMax - ((metricMax * 9) / 5 + 32))).toBeLessThan(1.5);
  });
});
