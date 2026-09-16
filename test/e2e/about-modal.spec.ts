// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import { test, expect } from "@playwright/test";

/**
 * About modal invariants:
 *   - Opens on #about-link click.
 *   - Closes via the × button.
 *   - Closes via Escape key.
 *   - Closes when clicking the dim overlay outside the modal body.
 */
test.describe("about modal", () => {
  test("opens via #about-link, closes via × button", async ({ page }) => {
    await page.goto("/");
    const overlay = page.locator("#about-overlay");
    await expect(overlay).not.toHaveClass(/visible/);

    await page.locator("#about-link").click();
    await expect(overlay).toHaveClass(/visible/);
    await expect(page.locator(".about-modal h2")).toBeVisible();

    await page.locator(".about-overlay [data-modal-close]").click();
    await expect(overlay).not.toHaveClass(/visible/);
  });

  test("closes via Escape key", async ({ page }) => {
    await page.goto("/");
    await page.locator("#about-link").click();
    await expect(page.locator("#about-overlay")).toHaveClass(/visible/);

    await page.keyboard.press("Escape");
    await expect(page.locator("#about-overlay")).not.toHaveClass(/visible/);
  });

  test("closes when clicking the overlay outside the modal body", async ({ page }) => {
    await page.goto("/");
    await page.locator("#about-link").click();
    const overlay = page.locator("#about-overlay");
    await expect(overlay).toHaveClass(/visible/);

    // Click near the very top-left of the overlay (well outside the centered modal box)
    const box = await overlay.boundingBox();
    expect(box).not.toBeNull();
    if (!box) return;
    await page.mouse.click(box.x + 5, box.y + 5);

    await expect(overlay).not.toHaveClass(/visible/);
  });
});
