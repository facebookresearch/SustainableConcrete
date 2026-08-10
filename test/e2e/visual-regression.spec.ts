import { test, expect } from "@playwright/test";

/**
 * Visual regression — full-page screenshots compared against committed
 * Linux baselines in `visual-regression.spec.ts-snapshots/`.
 *
 * These are ACTIVE on Linux (which is what CI runs) and skipped elsewhere.
 * Font rendering, anti-aliasing and emoji glyphs differ across platforms, so a
 * macOS-rendered screenshot always diffs against a Linux baseline. Rather than
 * disable the suite outright — it sat unconditionally skipped and therefore
 * never ran at all — we gate on platform so it protects CI while staying green
 * for local development on a Mac.
 *
 * To regenerate after an intentional UI change, run the docker command in
 * test/e2e/README.md (use the image tag matching the Playwright version in
 * package-lock.json) and commit the updated PNGs.
 */
test.describe("@visual full-page snapshots", () => {
  // Baselines are Linux-rendered; comparing them on another OS is guaranteed
  // to fail for reasons unrelated to the change under test.
  test.skip(
    process.platform !== "linux",
    `visual baselines are Linux-rendered; skipping on ${process.platform}`,
  );

  test("home page", async ({ page }, testInfo) => {
    await page.goto("/");
    // Wait for charts to fully render and any animations to settle
    await page.waitForTimeout(1500);
    // Stop animations so the screenshot is deterministic
    await page.addStyleTag({
      content: `*, *::before, *::after {
        animation-duration: 0s !important;
        animation-delay: 0s !important;
        transition-duration: 0s !important;
        transition-delay: 0s !important;
      }`,
    });
    await page.waitForTimeout(300);
    await expect(page).toHaveScreenshot(`home-${testInfo.project.name}.png`, {
      fullPage: true,
    });
  });

  test("about modal open", async ({ page }, testInfo) => {
    await page.goto("/");
    await page.locator("#about-link").click();
    await page.waitForTimeout(500);
    await expect(page.locator("#about-overlay")).toHaveScreenshot(
      `about-modal-${testInfo.project.name}.png`,
    );
  });
});
