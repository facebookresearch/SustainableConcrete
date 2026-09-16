// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import { test, expect } from "@playwright/test";

/**
 * Smoke tests — page loads cleanly with no errors and the core
 * UI elements are visible. If these fail, everything else fails too,
 * so they run first.
 */
test.describe("home page smoke", () => {
  test("loads with no console errors and core canvases visible", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    page.on("pageerror", (err) => errors.push(err.message));

    await page.goto("/");

    await expect(page.locator("h1", { hasText: "BOxCrete" })).toBeVisible();
    await expect(page.locator("canvas#scatter-canvas")).toBeVisible();
    await expect(page.locator("canvas#curve-canvas")).toBeVisible();

    // Wait for any deferred WASM/init work to settle, then assert no errors
    await page.waitForTimeout(1500);
    expect(
      errors,
      `Page produced console errors:\n${errors.join("\n")}`,
    ).toEqual([]);
  });

  test("strength curve canvas renders content (non-blank)", async ({ page }) => {
    await page.goto("/");
    // Wait for the prediction pipeline to draw something on the canvas
    await expect
      .poll(
        async () => {
          return page.locator("canvas#curve-canvas").evaluate((c: HTMLCanvasElement) => {
            const ctx = c.getContext("2d");
            if (!ctx) return false;
            const data = ctx.getImageData(0, 0, c.width, c.height).data;
            // Non-zero alpha somewhere → something was drawn
            for (let i = 3; i < data.length; i += 4) {
              if (data[i] !== 0) return true;
            }
            return false;
          });
        },
        { timeout: 10_000, message: "curve canvas remained blank" },
      )
      .toBe(true);
  });
});
