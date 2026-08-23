import { expect, test } from "@playwright/test";

/** Strength-curve readouts remain present and contained at every breakpoint. */
test.describe("readouts strip", () => {
  test("desktop: GWP, Cost, and W/B are all visible", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop-only layout");
    await page.goto("/");
    await expect(page.locator("#readouts")).toBeVisible();
    await expect(page.locator("#gwp-value")).toBeVisible();
    await expect(page.locator("#cost-value")).toBeVisible();
    await expect(page.locator("#wb-value")).toBeVisible();
  });

  test("mobile: GWP, Cost, and W/B remain visible inside the curve body", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "mobile", "mobile-only invariant");
    await page.goto("/");
    const readouts = page.locator("#readouts");
    const body = page.locator(".curve-body");
    await expect(readouts).toBeVisible();
    await expect(page.locator("#gwp-value")).toBeVisible();
    await expect(page.locator("#cost-value")).toBeVisible();
    await expect(page.locator("#wb-value")).toBeVisible();

    const [readoutsBox, bodyBox] = await Promise.all([readouts.boundingBox(), body.boundingBox()]);
    expect(readoutsBox).not.toBeNull();
    expect(bodyBox).not.toBeNull();
    expect(readoutsBox!.x).toBeGreaterThanOrEqual(bodyBox!.x - 1);
    expect(readoutsBox!.x + readoutsBox!.width).toBeLessThanOrEqual(bodyBox!.x + bodyBox!.width + 1);
    expect(readoutsBox!.y + readoutsBox!.height).toBeLessThanOrEqual(bodyBox!.y + bodyBox!.height + 1);
  });
});
