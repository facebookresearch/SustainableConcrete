import { test, expect } from "@playwright/test";
import {
  openMobileComposition,
  waitForDashboardLayoutReady,
} from "./dashboard-helpers";

/**
 * Touch-target sizing.
 *
 * The `input[type=range]` element used to BE the visible 6 px bar, so the whole
 * hit area was 6 px tall (measured 220x6 on the mobile project) — far under the
 * WCAG 2.2 AA "Target Size (Minimum)" floor of 24x24 CSS px, and further still
 * under Apple's 44x44 HIG recommendation. All 8 sliders failed.
 *
 * The visible bar now lives on the track pseudo-element, freeing the input box
 * to be a proper target. These tests pin that so the two can't be re-merged.
 */

const WCAG_MIN = 24;
const HIG_MIN = 44;

test.describe("slider touch targets", () => {
  test("mobile sliders meet the Apple HIG 44px target height", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile touch sizing");
    await page.goto("/");
    await waitForDashboardLayoutReady(page);
    await openMobileComposition(page);

    const heights = await page.evaluate(() =>
      Array.from(document.querySelectorAll(".mobile-sliders-view input[type=range]"))
        .filter((el) => (el as HTMLElement).offsetParent !== null)
        .map((el) => +el.getBoundingClientRect().height.toFixed(1)),
    );
    expect(heights.length, "expected visible mobile sliders").toBeGreaterThan(0);
    const tooSmall = heights.filter((h) => h < HIG_MIN);
    expect(
      tooSmall.length,
      `${tooSmall.length}/${heights.length} sliders under ${HIG_MIN}px: ${JSON.stringify(heights)}`,
    ).toBe(0);
  });

  test("mobile slider thumbs are at least 22px", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile thumb sizing");
    await page.goto("/");
    const thumbSize = await page.evaluate(() => {
      for (const sheet of Array.from(document.styleSheets)) {
        let rules: CSSRuleList;
        try { rules = (sheet as CSSStyleSheet).cssRules; } catch { continue; }
        for (const rule of Array.from(rules)) {
          if (!(rule instanceof CSSMediaRule) ||
              !rule.conditionText.includes("max-width: 1050px")) continue;
          for (const nested of Array.from(rule.cssRules)) {
            const styleRule = nested as CSSStyleRule;
            if (styleRule.selectorText?.includes("mobile-sliders-view") &&
                styleRule.selectorText.includes("::-webkit-slider-thumb")) {
              return parseFloat(styleRule.style.width);
            }
          }
        }
      }
      return null;
    });
    expect(thumbSize, "mobile WebKit thumb rule not found").not.toBeNull();
    expect(thumbSize!).toBeGreaterThanOrEqual(22);
  });

  test("desktop sliders meet the WCAG 2.2 AA 24px minimum", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop pointer sizing");
    await page.goto("/");
    await waitForDashboardLayoutReady(page);
    const heights = await page.evaluate(() =>
      Array.from(document.querySelectorAll("input[type=range]"))
        .filter((el) => (el as HTMLElement).offsetParent !== null)
        .map((el) => +el.getBoundingClientRect().height.toFixed(1)),
    );
    expect(heights.length, "expected visible sliders").toBeGreaterThan(0);
    const tooSmall = heights.filter((h) => h < WCAG_MIN);
    expect(
      tooSmall.length,
      `${tooSmall.length}/${heights.length} sliders under ${WCAG_MIN}px: ${JSON.stringify(heights)}`,
    ).toBe(0);
  });

  test("the visible track stays thin while the hit area is tall", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop pointer sizing");
    await page.goto("/");
    // The bar the user sees must remain the slim 6px line; growing the hit
    // area must not fatten the visual.
    const trackHeight = await page.evaluate(() => {
      for (const sheet of Array.from(document.styleSheets)) {
        let rules: CSSRuleList;
        try { rules = (sheet as CSSStyleSheet).cssRules; } catch { continue; }
        for (const r of Array.from(rules)) {
          const sel = (r as CSSStyleRule).selectorText || "";
          if (sel.includes("slider-runnable-track")) {
            return (r as CSSStyleRule).style.height;
          }
        }
      }
      return null;
    });
    expect(trackHeight, "runnable-track rule not found — visual moved back onto the input?").toBe("6px");
  });
});

test.describe("mobile Scatter and filter touch targets", () => {
  test("axis and filter controls meet the 44px mobile target", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile touch sizing");
    await page.goto("/?test=1");
    await page.locator("#filter-add").click();
    await expect(page.locator(".filter-row")).toBeVisible();

    const selectors = [
      "#axis-selector-x .axis-option-face",
      "#axis-selector-y .axis-option-face",
      "#filter-add",
      ".filter-remove-btn",
    ];
    for (const selector of selectors) {
      const box = await page.locator(selector).first().boundingBox();
      expect(box, `${selector} must have a box`).not.toBeNull();
      expect(box!.width, `${selector} width`).toBeGreaterThanOrEqual(HIG_MIN);
      expect(box!.height, `${selector} height`).toBeGreaterThanOrEqual(HIG_MIN);
    }

    await page.locator("#mobile-show-sliders").click();
    await expect(page.locator(".mobile-sliders-view")).toBeVisible();
    const materialSource = await page
      .locator(".mobile-sliders-view .material-source-group .toggle-btn")
      .first()
      .boundingBox();
    expect(materialSource, "Material Source button must have a box").not.toBeNull();
    expect(materialSource!.width).toBeGreaterThanOrEqual(HIG_MIN);
    expect(materialSource!.height).toBeGreaterThanOrEqual(HIG_MIN);

    await page.locator("#mobile-show-scatter").click();
    await expect(page.locator(".scatter-content")).toBeVisible();
    await expect(page.locator(".filter-row")).toBeVisible();
    for (const selector of [".filter-col", ".filter-min", ".filter-max"]) {
      const box = await page.locator(selector).first().boundingBox();
      expect(box, `${selector} must have a box`).not.toBeNull();
      expect(box!.width, `${selector} width`).toBeGreaterThanOrEqual(WCAG_MIN);
      expect(box!.height, `${selector} height`).toBeGreaterThanOrEqual(WCAG_MIN);
    }
  });
});
