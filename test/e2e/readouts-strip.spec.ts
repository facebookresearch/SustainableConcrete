import { test, expect } from "@playwright/test";

/**
 * Strength-curve readouts strip (`.readouts`): GWP, Cost, and Slump — three
 * predicted outcome properties. On mobile we hide the slump ±2σ suffix so the
 * three items sit on a single line; wrapping onto two rows would steal
 * vertical space from the strength curve canvas above. Pinned here so the
 * layout can't silently regress when font sizes or content change.
 */
test.describe("readouts strip", () => {
  test("desktop: GWP, Cost, and Slump are all visible", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop layout");
    await page.goto("/");
    await expect(page.locator("#readouts")).toBeVisible();
    await expect(page.locator("#gwp-value")).toBeVisible();
    await expect(page.locator("#cost-value")).toBeVisible();
    await expect(page.locator("#slump-value")).toBeVisible();
    await expect(page.locator("#slump-unit")).toBeVisible();
    await expect(page.locator("#cost-uncertainty")).toContainText("2σ");
    // The W/B readout was replaced by Slump; pin the removal so a stale
    // element can't linger.
    await expect(page.locator("#wb-value")).toHaveCount(0);
  });

  test("the slump ±2σ suffix appears only above the 1440px breakpoint", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "viewport-driven invariant");
    await page.goto("/");
    // The desktop project runs at 1280, where the strip would otherwise wrap
    // mid-value; the suffix is dropped to keep it on one line.
    await expect(page.locator("#slump-uncertainty")).toBeHidden();

    await page.setViewportSize({ width: 1600, height: 900 });
    await expect(page.locator("#slump-uncertainty")).toBeVisible();
    await expect(page.locator("#slump-uncertainty")).toContainText("2σ");

    // …and the strip still occupies a single row at that width.
    const tops = await page.locator("#readouts > div").evaluateAll((els) =>
      els.map((el) => el.getBoundingClientRect().top),
    );
    expect(Math.max(...tops) - Math.min(...tops)).toBeLessThanOrEqual(2);
  });

  test("mobile: suffixes and the GWP unit are hidden so the readouts stay on one line", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "mobile", "mobile-only invariant");
    await page.goto("/");
    // The strength-curve panel is visible by default on mobile (it's the
    // top half of the dvh-split). The readouts live underneath.
    await expect(page.locator("#readouts")).toBeVisible();
    await expect(page.locator(".slump-readout")).toBeVisible();

    // The uncertainty spans and the GWP unit are hidden via CSS to fit three
    // readouts on one line; the slump and cost units stay because slump's
    // flips between mm and in with the unit toggle.
    const hidden = await page.evaluate(() =>
      ["cost-uncertainty", "slump-uncertainty", "gwp-unit"].map(
        (id) => getComputedStyle(document.getElementById(id)!).display,
      ),
    );
    expect(hidden, "±2σ suffixes and the GWP unit should be hidden on mobile").toEqual([
      "none",
      "none",
      "none",
    ]);
    await expect(page.locator("#slump-unit")).toBeVisible();
    await expect(page.locator("#cost-unit")).toBeVisible();

    // The visible readout `<div>`s should all sit on the same row, i.e.
    // their `top` y-coordinates are within a few pixels of each other.
    // The `2` below is a PIXEL tolerance, not a row count: if the strip ever
    // wraps, the second row's `top` jumps by the line height (~14px+).
    const tops = await page.locator("#readouts > div").evaluateAll((els) =>
      els
        .filter((el) => (el as HTMLElement).offsetParent !== null)
        .map((el) => el.getBoundingClientRect().top),
    );
    expect(tops.length, "expected 3 visible readouts on mobile").toBe(3);
    const minTop = Math.min(...tops);
    const maxTop = Math.max(...tops);
    expect(
      maxTop - minTop,
      `readouts strip wrapped (top deltas: min=${minTop}, max=${maxTop})`,
    ).toBeLessThanOrEqual(2);
  });

  test("mortar mixes show n/a instead of a slump number", async ({ page }, testInfo) => {
    await page.goto("/");
    if (testInfo.project.name === "mobile") {
      await page.locator("#mobile-show-sliders").click();
      await expect(page.locator(".mobile-sliders-view")).toBeVisible({ timeout: 5000 });
    }
    await expect(page.locator("#slump-value")).toBeVisible();
    // Mortar (Material Source 0, rendered "Source A") has no slump data:
    // workability there is measured with a flow table, not a slump cone.
    await page
      .locator(".material-source-group .toggle-btn", { hasText: "Source A" })
      .click();
    await expect(page.locator("#slump-value")).toHaveText("n/a");
    await expect(page.locator("#slump-uncertainty")).toBeEmpty();
  });

  test("toggling units rescales the slump value, not just its label", async ({
    page,
  }, testInfo) => {
    await page.goto("/");
    await expect(page.locator("#slump-value")).not.toHaveText("–");

    // The app defaults to metric, so slump starts in mm and the toggle takes
    // it to inches. Assert the ratio rather than a direction, so this stays
    // correct if the default unit system ever flips.
    const before = parseFloat(await page.locator("#slump-value").innerText());
    const unitBefore = await page.locator("#slump-unit").innerText();
    expect(Number.isFinite(before)).toBe(true);

    const toggleId =
      testInfo.project.name === "mobile" ? "#mobile-unit-toggle" : "#unit-toggle";
    await page.locator(toggleId).click();

    await expect(page.locator("#slump-unit")).toHaveText(
      unitBefore === "mm" ? "in" : "mm",
    );
    const after = parseFloat(await page.locator("#slump-value").innerText());
    const ratio = unitBefore === "mm" ? before / after : after / before;
    // 25.4x; loose bounds so rounding to 1dp can't make this flaky. Before
    // this was wired, the label changed but the number did not (ratio 1).
    expect(ratio).toBeGreaterThan(20);
    expect(ratio).toBeLessThan(30);
  });
});
