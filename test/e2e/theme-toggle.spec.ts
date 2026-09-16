// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import { test, expect } from "@playwright/test";

/**
 * Theme toggle invariants:
 *   1. Clicking flips data-theme on <html> between "light" and "dark".
 *   2. Choice persists across reloads via localStorage.
 */
test.describe("theme toggle", () => {
  test("clicking flips html[data-theme] between light and dark", async ({ page }) => {
    await page.goto("/");
    // Force a known starting state
    await page.evaluate(() => {
      localStorage.setItem("boxcrete-theme", "dark");
    });
    await page.reload();

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
    await page.evaluate(() => localStorage.setItem("boxcrete-theme", "dark"));
    await page.reload();

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
