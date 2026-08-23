import { test, expect } from "@playwright/test";

/**
 * Header layout invariants.
 *
 * These tests pin down the design rules we keep regressing on:
 *  1. Theme toggle is always the rightmost element.
 *  2. On desktop: cite group sits adjacent and to the LEFT of the theme toggle.
 *  3. On mobile: citation controls remain reachable and header groups do not overlap.
 *  4. Header is sticky to the top while scrolling.
 *  5. No horizontal pan on mobile (would expose header endpoints).
 */
test.describe("header layout", () => {
  test("theme toggle is the rightmost element in the header", async ({ page }) => {
    await page.goto("/");
    const header = await page.locator(".site-header").boundingBox();
    const toggle = await page.locator(".theme-toggle").boundingBox();
    expect(header, "header should be visible").not.toBeNull();
    expect(toggle, "theme toggle should be visible").not.toBeNull();
    if (!header || !toggle) return;
    // Tolerance: header padding (~16-24px). Theme toggle right edge should
    // be within this padding from the header right edge.
    const distFromRight = header.x + header.width - (toggle.x + toggle.width);
    expect(distFromRight).toBeLessThan(28);
    expect(distFromRight).toBeGreaterThanOrEqual(0);
  });

  test("desktop: cite group is adjacent to and left of the theme toggle, no overlap", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop-only layout");
    await page.goto("/");
    const cite = await page.locator(".site-header .cite-group").boundingBox();
    const toggle = await page.locator(".theme-toggle").boundingBox();
    expect(cite, "cite group should be visible on desktop").not.toBeNull();
    expect(toggle).not.toBeNull();
    if (!cite || !toggle) return;
    // Cite group's right edge must not extend past the theme toggle's left edge
    expect(cite.x + cite.width).toBeLessThanOrEqual(toggle.x);
    // And they must be reasonably close (within ~32px gap)
    expect(toggle.x - (cite.x + cite.width)).toBeLessThan(32);
  });

  test("mobile: citation controls remain visible", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "mobile", "mobile-only layout");
    await page.goto("/");
    await expect(page.locator(".site-header .cite-group")).toBeVisible();
    await expect(page.locator("#cite-bibtex")).toBeVisible();
    await expect(page.locator("#cite-apa")).toBeVisible();
  });

  test("mobile: visible header groups stay ordered and non-overlapping", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "mobile", "mobile-only layout");
    await page.goto("/");
    const selectors = [
      ".site-header h1",
      "#about-link",
      'a[href*="youtube.com"], #video-link',
      'a[href*="github.com/facebookresearch"]',
      ".site-header .cite-group",
      ".theme-toggle",
    ];
    const boxes = await Promise.all(
      selectors.map((s) => page.locator(s).first().boundingBox()),
    );
    boxes.forEach((b, i) => {
      expect(b, `selector "${selectors[i]}" must have a bounding box`).not.toBeNull();
    });
    if (boxes.some((b) => !b)) return;

    const xs = boxes.map((b) => ({ left: b!.x, right: b!.x + b!.width }));
    // Verify left-to-right ordering
    for (let i = 1; i < xs.length; i++) {
      expect(xs[i].left, `item ${i} should be right of item ${i - 1}`).toBeGreaterThan(
        xs[i - 1].right - 1, // -1 to allow exact-touch with no overlap
      );
    }
    const header = await page.locator(".site-header").boundingBox();
    expect(header).not.toBeNull();
    expect(xs[0].left).toBeGreaterThanOrEqual(header!.x);
    expect(xs.at(-1)!.right).toBeLessThanOrEqual(header!.x + header!.width + 1);
  });

  test("mobile: every header control stays inside a 320px viewport", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "mobile", "mobile-only layout");
    await page.setViewportSize({ width: 320, height: 700 });
    await page.goto("/");

    const selectors = [
      ".site-header h1",
      "#about-link",
      "#video-link",
      'a[href*="github.com/facebookresearch"]',
      "#cite-bibtex",
      "#cite-apa",
      ".theme-toggle",
    ];
    const boxes = [];
    for (const selector of selectors) {
      const box = await page.locator(selector).boundingBox();
      expect(box, `${selector} must remain rendered`).not.toBeNull();
      expect(box!.x, `${selector} crosses the left viewport edge`).toBeGreaterThanOrEqual(0);
      expect(box!.x + box!.width, `${selector} crosses the right viewport edge`).toBeLessThanOrEqual(320);
      if (selector !== ".site-header h1") {
        expect(box!.width, `${selector} must keep a 44px mobile target width`).toBeGreaterThanOrEqual(44);
        expect(box!.height, `${selector} must keep a 44px mobile target height`).toBeGreaterThanOrEqual(44);
      }
      boxes.push(box!);
    }
    const actionBoxes = boxes.slice(1);
    expect(boxes[0].y + boxes[0].height, "brand must occupy the first header row").toBeLessThanOrEqual(
      Math.min(...actionBoxes.map((box) => box.y)) + 1,
    );
    expect(Math.max(...actionBoxes.map((box) => box.y)) - Math.min(...actionBoxes.map((box) => box.y)))
      .toBeLessThanOrEqual(1);
    await page.locator("#about-link").focus();
    for (const selector of selectors.slice(2)) {
      await page.keyboard.press("Tab");
      await expect(page.locator(selector)).toBeFocused();
    }
  });

  test("header stays sticky at top of viewport when scrolling to references panel", async ({
    page,
  }) => {
    await page.goto("/");
    // Initial: header visible at top
    const before = await page.locator(".site-header").boundingBox();
    expect(before?.y).toBeLessThan(10);

    // Scroll the references panel into view (it's near the bottom of the page)
    await page.locator(".references-panel").scrollIntoViewIfNeeded();
    // Allow any scroll-snap / momentum to settle
    await page.waitForTimeout(400);

    // Header is still pinned at top (within a few px for sticky offset)
    const after = await page.locator(".site-header").boundingBox();
    expect(after?.y, "header should still be sticky at top after scrolling").toBeLessThan(
      10,
    );
  });

  test("mobile: page does not scroll horizontally when dragged left", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "mobile", "mobile-only behavior");
    await page.goto("/");
    // Body should never have a horizontal scrollable extent
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
    // Allow 1px sub-pixel rounding tolerance
    expect(scrollWidth - clientWidth).toBeLessThan(2);
  });
});
