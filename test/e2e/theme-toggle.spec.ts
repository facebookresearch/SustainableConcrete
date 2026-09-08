import { test, expect } from "@playwright/test";

/**
 * Theme toggle invariants:
 *   1. Clicking flips data-theme on <html> between "light" and "dark".
 *   2. Choice persists across reloads via localStorage.
 */
test.describe("theme toggle", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      if (localStorage.getItem("boxcrete-theme") === null) {
        localStorage.setItem("boxcrete-theme", "dark");
      }
    });
  });

  test("clicking flips html[data-theme] between light and dark", async ({ page }) => {
    await page.goto("/");

    const before = await page.evaluate(() =>
      document.documentElement.getAttribute("data-theme"),
    );
    expect(before).toBe("dark");

    await page.locator("#theme-toggle").click();

    await expect
      .poll(
        async () =>
          page.evaluate(() => document.documentElement.getAttribute("data-theme")),
        { timeout: 2000 },
      )
      .toBe("light");
  });

  test("theme choice persists across page reload", async ({ page }) => {
    await page.goto("/");

    // Toggle to light, then reload and confirm it stuck
    await page.locator("#theme-toggle").click();
    await expect
      .poll(
        async () =>
          page.evaluate(() => document.documentElement.getAttribute("data-theme")),
        { timeout: 2000 },
      )
      .toBe("light");

    await page.reload();
    const after = await page.evaluate(() =>
      document.documentElement.getAttribute("data-theme"),
    );
    expect(after, "theme should persist across reload").toBe("light");
  });
});
