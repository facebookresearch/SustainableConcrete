import { test, expect } from "@playwright/test";

/**
 * iPad-landscape layout invariants.
 *
 * The site has two layouts gated by a viewport-width breakpoint in
 * `docs/style.css` (`@media (max-width: 900px)`). Below 900 px → mobile
 * stacked layout. ≥ 901 px → desktop 3-column grid.
 *
 * iPad in landscape (≥ 944 px on the narrowest modern iPad, 1080 px on gen 7,
 * 1194 px on iPad Pro 11) should land in the desktop layout. These tests pin
 * that down on a touch-enabled, iPad-sized viewport and confirm the basic
 * touch interactions we rely on still work.
 *
 * Out of scope (deliberately deferred):
 *   - scatter canvas hover-preview parity on touch (requires `pointer*`
 *     events; today's `mousemove` handler is mouse-only).
 *   - curve-canvas tap-to-show-tooltip persistence (separate change).
 */
test.describe("iPad landscape layout", () => {
  test.beforeEach(({}, testInfo) => {
    test.skip(
      testInfo.project.name !== "tablet-landscape",
      "tablet-landscape only",
    );
  });

  test("desktop layout is active: .layout is a grid with 3 columns", async ({
    page,
  }) => {
    await page.goto("/");
    const info = await page.locator(".layout").evaluate((el) => {
      const cs = getComputedStyle(el);
      return {
        display: cs.display,
        columnCount: cs.gridTemplateColumns.split(/\s+/).filter(Boolean).length,
      };
    });
    expect(info.display).toBe("grid");
    expect(info.columnCount).toBe(3);
  });

  test("the desktop sliders panel is visible (mobile would hide it)", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(page.locator(".panel.sliders")).toBeVisible();
    // The unified mobile panel is hidden on desktop layout.
    await expect(page.locator(".mobile-sliders-view")).toBeHidden();
  });

  test("no horizontal scroll", async ({ page }) => {
    await page.goto("/");
    const scrollWidth = await page.evaluate(
      () => document.documentElement.scrollWidth,
    );
    const clientWidth = await page.evaluate(
      () => document.documentElement.clientWidth,
    );
    // Allow 1px sub-pixel rounding tolerance — same threshold as
    // header-layout.spec.ts.
    expect(scrollWidth - clientWidth).toBeLessThan(2);
  });

  test("both chart canvases are visible", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#scatter-canvas")).toBeVisible();
    await expect(page.locator("#curve-canvas")).toBeVisible();
  });

  test("tapping the inactive Material Source toggle activates it", async ({
    page,
  }) => {
    await page.goto("/");
    const buttons = page.locator(".material-source-group .toggle-btn");
    await expect(buttons).toHaveCount(2);

    // Find which button is currently inactive and tap it.
    const activeIdxBefore = await buttons.evaluateAll((els) =>
      els.findIndex((el) => el.classList.contains("active")),
    );
    expect(activeIdxBefore).toBeGreaterThanOrEqual(0);
    const inactiveIdx = activeIdxBefore === 0 ? 1 : 0;

    const inactiveBox = await buttons.nth(inactiveIdx).boundingBox();
    expect(inactiveBox).not.toBeNull();
    if (!inactiveBox) return;

    // Use touchscreen.tap to mirror real iPad input rather than synthetic
    // click; verifies the toggle works under touch event ordering too.
    await page.touchscreen.tap(
      inactiveBox.x + inactiveBox.width / 2,
      inactiveBox.y + inactiveBox.height / 2,
    );

    await expect(buttons.nth(inactiveIdx)).toHaveClass(/active/);
    await expect(buttons.nth(activeIdxBefore)).not.toHaveClass(/active/);
  });
});
