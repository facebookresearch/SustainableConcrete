import { test, expect } from "@playwright/test";

/**
 * prefers-reduced-motion (WCAG 2.3.3; 2.2.2 for the infinite animations).
 *
 * The first pass gated only four decorative canvas easings, which missed every
 * interaction-triggered transition — the composition slide, the Material
 * Source crossfade, the unit transition — and all CSS animation, including a
 * 30 s infinite title gradient with no pause control.
 *
 * Uses page.emulateMedia() rather than `test.use({ reducedMotion })`: the
 * projects spread a `devices[...]` descriptor, which was taking precedence and
 * silently leaving the preference unset — the media query never matched and
 * these tests would have passed against ungated code. It must also run BEFORE
 * goto, because ui.mjs samples the preference once at module load.
 */
async function openReducedMotion(
  page: import("@playwright/test").Page,
  opts: { waitForModel?: boolean } = {},
) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/?test=1");
  // Guard the guard: if emulation ever stops working these tests must fail
  // loudly rather than quietly assert nothing.
  expect(
    await page.evaluate(() => matchMedia("(prefers-reduced-motion: reduce)").matches),
    "reduced-motion emulation did not reach the page",
  ).toBe(true);
  await page.locator("input[type=range]").first().waitFor({ timeout: 15000 });
  if (opts.waitForModel) {
    await page.waitForFunction(() => (window as any).__test?.modelReady === true, null, {
      timeout: 20000,
    });
  }
}

test.describe("reduced motion", () => {
  test("interaction transitions complete immediately", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "toggle is desktop-visible");
    await openReducedMotion(page, { waitForModel: true });

    // The Material Source crossfade is 350 ms normally; under reduced motion
    // it must be finished well inside that window.
    await page.locator(".material-source-group .toggle-btn:not(.active)").first().click();
    await page.waitForTimeout(120);
    expect(
      await page.evaluate(() => (window as any).__test.isCurveTransitionActive),
      "curve transition still running 120ms in under reduced motion",
    ).toBe(false);
  });

  test("no infinite CSS animation is left running", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "one project is enough");
    await openReducedMotion(page);
    await page.waitForTimeout(500);

    const running = await page.evaluate(() =>
      document
        .getAnimations()
        .filter((a) => a.playState === "running" && a.effect?.getTiming().iterations === Infinity)
        .map((a) => (a as any).animationName || "unnamed"),
    );
    expect(
      running,
      `infinite animations still running under reduced motion: ${JSON.stringify(running)}`,
    ).toEqual([]);
  });

  test("the Material Source preview curve completes on the next frame", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "Material Source controls are desktop-visible");
    await openReducedMotion(page, { waitForModel: true });
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();

    await inactive.hover();
    await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => resolve())));

    expect(
      await page.evaluate(() => (window as any).__test.isPreviewCurveTransitionActive),
    ).toBe(false);
    expect(Number.isInteger(
      await page.evaluate(() => (window as any).__test.displayPreviewComp[7]),
    )).toBe(true);
  });

  test("the Material Source preview ring does not animate", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "Material Source controls are desktop-visible");
    await openReducedMotion(page, { waitForModel: true });
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await inactive.hover();
    await expect(inactive).toHaveAttribute("data-previewing", "");
    expect(
      await inactive.evaluate((el) => getComputedStyle(el, "::after").animationName),
    ).toBe("none");
  });

  test("the title gradient is pinned, not frozen mid-sweep", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "one project is enough");
    await openReducedMotion(page);
    const style = await page.evaluate(() => {
      const h1 = document.querySelector("h1")!;
      const cs = getComputedStyle(h1);
      return { animationName: cs.animationName, backgroundPosition: cs.backgroundPosition };
    });
    expect(style.animationName, "title animation should be off").toBe("none");
    expect(style.backgroundPosition, "gradient should be pinned to its start").toMatch(/^0%|^0px/);
  });

  test("panels do not stage a delayed pop-in", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "one project is enough");
    await openReducedMotion(page);
    // .fade-in-up uses animation-fill-mode: both with per-panel delays up to
    // 350ms. Neutralising duration alone leaves the backwards fill holding
    // opacity 0 for the delay, so the panels still appear one at a time.
    const delays = await page.evaluate(() =>
      Array.from(document.querySelectorAll(".fade-in-up")).map((el) => ({
        delay: getComputedStyle(el).animationDelay,
        opacity: Number(getComputedStyle(el).opacity),
      })),
    );
    expect(delays.length, "expected staggered panels").toBeGreaterThan(0);
    for (const d of delays) {
      expect(
        parseFloat(d.delay),
        `animation-delay ${d.delay} still positive under reduced motion`,
      ).toBeLessThanOrEqual(0);
      expect(d.opacity, "panel should be visible immediately").toBe(1);
    }
  });

  test("script-created animations are collapsed too", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "filter UI is desktop-only");
    await openReducedMotion(page, { waitForModel: true });

    // element.animate() is not covered by the CSS reset -- CSS
    // animation-duration does not apply to script-created animations -- so
    // these have to consult the preference themselves.
    await page.locator("#filter-add").click();
    const durations = await page.evaluate(() =>
      document.getAnimations().map((a) => Number(a.effect?.getTiming().duration ?? 0)),
    );
    const slow = durations.filter((d) => d > 50);
    expect(
      slow.length,
      `script animations still running long under reduced motion: ${JSON.stringify(slow)}`,
    ).toBe(0);
  });
});
