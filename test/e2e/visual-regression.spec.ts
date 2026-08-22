import { expect, test, type Page } from "@playwright/test";

async function freezeVisualMotion(page: Page) {
  await page.addStyleTag({
    content: `*, *::before, *::after {
      animation-duration: 0s !important;
      animation-delay: 0s !important;
      transition-duration: 0s !important;
      transition-delay: 0s !important;
    }`,
  });
}

async function settleDashboard(page: Page) {
  await page.goto("/");
  await expect(page.locator("#sliders .slider-group").last()).toBeAttached({ timeout: 15_000 });
  await page.evaluate(() => document.fonts.ready);
  await expect
    .poll(() =>
      page.locator("#curve-canvas").evaluate((canvas: HTMLCanvasElement) => {
        const data = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height).data;
        return data.some((value, index) => index % 4 === 3 && value !== 0);
      }),
    )
    .toBe(true);
  await freezeVisualMotion(page);
  await page.evaluate(() => window.scrollTo(0, 0));
}

/**
 * Visual regression uses real viewport states rather than stitched full-page
 * captures. Baselines are Linux-only because font rendering differs by OS.
 */
test.describe("@visual fixed-dashboard snapshots", () => {
  test.skip(
    process.platform !== "linux",
    `visual baselines are Linux-rendered; skipping on ${process.platform}`,
  );

  test("managed-laptop dashboard viewport", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "managed-laptop dashboard baseline");
    await page.setViewportSize({ width: 1728, height: 1000 });
    await settleDashboard(page);
    await expect(page).toHaveScreenshot("dashboard-managed-laptop.png");
  });

  test("mobile document top and lower panels", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "mobile", "mobile document baselines");
    await settleDashboard(page);
    await expect(page).toHaveScreenshot("dashboard-mobile-top.png");

    await page.locator("#ingredient-insight").scrollIntoViewIfNeeded();
    await expect(page).toHaveScreenshot("dashboard-mobile-lower-panels.png");
  });

  test("mobile Composition at local scroll boundaries", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "mobile", "mobile Composition baselines");
    await settleDashboard(page);
    await page.locator("#mobile-show-sliders").click();
    await expect(page.locator(".mobile-sliders-view")).toBeVisible();
    const panel = page.locator("#tradeoffs-panel");
    const body = page.locator(".mobile-scroll-content");
    await panel.evaluate((element) => element.scrollIntoView({ block: "center" }));

    await body.evaluate((element) => {
      element.scrollTop = 0;
    });
    await expect(page.locator("#sliders .slider-group").first()).toBeInViewport();
    await expect(page).toHaveScreenshot("composition-mobile-start.png");

    await body.evaluate((element) => {
      element.scrollTop = element.scrollHeight;
    });
    await expect(page.locator("#sliders .slider-group").last()).toBeInViewport();
    await expect(page).toHaveScreenshot("composition-mobile-end.png");
  });

  test("compact desktop Composition", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "strict layout crops use desktop rendering");
    await page.setViewportSize({ width: 1728, height: 1000 });
    await settleDashboard(page);
    await expect(page.locator("#sliders-panel")).toHaveScreenshot("composition-desktop-compact.png");
  });

  test("compact Tradeoffs selector lanes", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "strict layout crops use desktop rendering");
    await page.setViewportSize({ width: 1728, height: 1000 });
    await settleDashboard(page);
    await expect(page.locator("#tradeoffs-panel")).toHaveScreenshot("tradeoffs-selector-lanes.png");
  });

  test("narrow mobile Composition uses panel width", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "mobile", "narrow mobile layout crop");
    await page.setViewportSize({ width: 320, height: 720 });
    await settleDashboard(page);
    await page.locator("#mobile-show-sliders").click();
    await expect(page.locator(".mobile-sliders-view")).toBeVisible();
    await expect(page.locator("#tradeoffs-panel")).toHaveScreenshot("composition-mobile-320.png");
  });

  test("open References metadata and disclosure layout", async ({ page }, testInfo) => {
    if (testInfo.project.name === "desktop") {
      await page.setViewportSize({ width: 1728, height: 1000 });
    } else if (testInfo.project.name === "mobile") {
      await page.setViewportSize({ width: 320, height: 720 });
    } else {
      test.skip(true, "strict References crops use Chromium rendering");
    }
    await settleDashboard(page);
    const panel = page.locator("#references-panel");
    const wrapper = panel.locator(".ref-description-motion").first();
    await panel.locator(".ref-details > summary").first().click();
    await expect.poll(() => wrapper.evaluate((element) => element.getAnimations().length)).toBe(0);
    await panel.scrollIntoViewIfNeeded();
    await expect(panel).toHaveScreenshot(`references-open-${testInfo.project.name}.png`);
  });

  test("representative visual-effect envelopes", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "strict effect crops use desktop rendering");
    await settleDashboard(page);

    const materialRow = page.locator(".material-source-group .toggle-row");
    await materialRow.scrollIntoViewIfNeeded();
    const edgeButton = materialRow.locator(".toggle-btn:not(.active)").first();
    await edgeButton.focus();
    await expect(edgeButton).toHaveAttribute("data-previewing", "");
    await expect(materialRow).toHaveScreenshot("material-source-effect-envelope.png", {
      animations: "disabled",
    });

    const summary = page.locator(".ref-details > summary").first();
    await summary.scrollIntoViewIfNeeded();
    await summary.focus();
    await expect(summary).toBeFocused();
    await expect(summary).toHaveScreenshot("reference-summary-focus-envelope.png", {
      animations: "disabled",
    });
  });

  test("about modal open", async ({ page }, testInfo) => {
    await page.goto("/");
    await page.locator("#about-link").click();
    await freezeVisualMotion(page);
    await expect(page.locator("#about-overlay")).toHaveScreenshot(
      `about-modal-${testInfo.project.name}.png`,
    );
  });
});
