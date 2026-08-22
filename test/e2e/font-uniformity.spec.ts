import { test, expect } from "@playwright/test";

/**
 * The mix-insight, ingredient-insight, and reference-description text blocks
 * all share a unified font-size (~0.82rem) for visual consistency. This pin
 * regresses against accidental drift in any of the three rules.
 */
test.describe("font uniformity across insight panels", () => {
  test("mix-insight, ingredient-insight, ref-desc share computed fontSize", async ({ page }) => {
    await page.goto("/");
    // Reference descriptions are collapsed by default; open one before measuring it.
    await expect(page.locator(".mix-insight-text").first()).toBeVisible();
    await expect(page.locator(".ingredient-insight-text").first()).toBeVisible();
    await page.locator(".ref-details > summary").first().click();
    await expect(page.locator(".ref-desc").first()).toBeVisible();

    const sizes = await page.evaluate(() => {
      const get = (sel: string) => {
        const el = document.querySelector(sel);
        return el ? parseFloat(getComputedStyle(el).fontSize) : null;
      };
      return {
        mix: get(".mix-insight-text"),
        ing: get(".ingredient-insight-text"),
        ref: get(".ref-desc"),
      };
    });

    expect(sizes.mix, "all three font sizes must be measured").not.toBeNull();
    expect(sizes.mix).toBe(sizes.ing);
    expect(sizes.ing).toBe(sizes.ref);
  });
});
