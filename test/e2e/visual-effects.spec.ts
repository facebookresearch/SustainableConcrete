import { expect, test } from "@playwright/test";
import { expectEffectInsideClippingAncestors } from "./effect-envelope";

async function waitForDashboard(page: import("@playwright/test").Page) {
  await page.goto("/?test=1");
  await expect(page.locator("#sliders .slider-group").last()).toBeAttached({ timeout: 15_000 });
  await page.evaluate(async () => {
    await document.fonts.ready;
    await Promise.all(
      [...document.querySelectorAll(".fade-in-up")].flatMap((element) =>
        element.getAnimations().map((animation) => animation.finished),
      ),
    );
  });
}

test.describe("all-sided visual-effect safe areas", () => {
  test.beforeEach(async ({ page }) => waitForDashboard(page));

  test("Material Source preview pulse and hover lift fit every clipping ancestor", async ({ page }) => {
    if (await page.locator("#sliders").isHidden()) {
      await page.locator("#mobile-show-sliders").click();
    }
    for (const button of await page.locator(".material-source-group .toggle-btn").all()) {
      await button.scrollIntoViewIfNeeded();
      await expectEffectInsideClippingAncestors(button, { top: 7, right: 7, bottom: 7, left: 7 });
    }
  });

  test("Composition edge controls reserve their outward focus envelopes", async ({ page }) => {
    if (await page.locator("#sliders").isHidden()) {
      await page.locator("#mobile-show-sliders").click();
    }
    for (const control of [
      page.locator("#sliders input[type=range]").first(),
      page.locator("#sliders .ingredient-name").first(),
      page.locator("#sliders .slider-value").last(),
    ]) {
      await control.scrollIntoViewIfNeeded();
      await expectEffectInsideClippingAncestors(control, { top: 5, right: 5, bottom: 5, left: 5 });
    }
  });

  test("filter and reference edge controls reserve outward focus and hover paint", async ({ page }) => {
    await page.locator("#filter-add").click();
    await expect.poll(() => page.locator(".filter-row-wrapper").first().evaluate((element) =>
      element.getAnimations().length,
    )).toBe(0);
    for (const control of [
      page.locator(".filter-col").first(),
      page.locator(".filter-min").first(),
      page.locator(".filter-remove-btn").first(),
      page.locator(".ref-details > summary").first(),
      page.locator(".ref-link").first(),
    ]) {
      await control.scrollIntoViewIfNeeded();
      await expectEffectInsideClippingAncestors(control, { top: 4, right: 4, bottom: 4, left: 4 });
    }
  });

  test("mobile body keeps a base bottom gutter in addition to any safe-area inset", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile safe-area contract");
    const state = await page.locator("body").evaluate((element) => ({
      paddingBottom: parseFloat(getComputedStyle(element).paddingBottom),
      rootFontSize: parseFloat(getComputedStyle(document.documentElement).fontSize),
    }));
    expect(state.paddingBottom).toBeGreaterThanOrEqual(state.rootFontSize);
  });

  test("rotated Y pills and mobile panel toggles fit their nearest panel clips", async ({ page }, testInfo) => {
    for (const pill of await page.locator("#axis-selector-y .axis-option-pill").all()) {
      await expectEffectInsideClippingAncestors(pill, { top: 4, right: 4, bottom: 4, left: 4 });
    }
    if (testInfo.project.name.startsWith("mobile")) {
      for (const toggle of await page.locator(".mobile-panel-header button").all()) {
        await expectEffectInsideClippingAncestors(toggle, { top: 4, right: 4, bottom: 4, left: 4 });
      }
    }
  });
});
