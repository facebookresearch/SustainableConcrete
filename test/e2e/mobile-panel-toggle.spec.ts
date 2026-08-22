import { test, expect } from "@playwright/test";

/**
 * Mobile-only: the unified panel switches between scatter and sliders
 * via two tab buttons (#mobile-show-scatter / #mobile-show-sliders).
 * After a 300ms crossfade, the inactive view should be hidden.
 */
test.describe("mobile panel toggle", () => {
  test("initial state: scatter visible, sliders hidden, scatter button active", async ({
    page,
  }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile-only");
    await page.goto("/");
    await expect(page.locator("#mobile-show-scatter")).toHaveClass(/active/);
    await expect(page.locator(".scatter-content")).toBeVisible();
    await expect(page.locator(".mobile-sliders-view")).toBeHidden();
    await expect(page.locator("#tradeoffs-panel #filter-add")).toBeVisible();
    await expect(page.locator("#filters-panel")).toBeHidden();
  });

  test("tapping Composition shows one unified scroller with no nested sliders scrollport", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile-only");
    await page.goto("/");
    await page.locator("#mobile-show-sliders").click();
    await expect(page.locator("#mobile-show-sliders")).toHaveClass(/active/);
    await expect(page.locator(".mobile-sliders-view")).toBeVisible({ timeout: 2000 });
    await expect(page.locator(".scatter-content")).toBeHidden();
    await expect(page.locator("#tradeoffs-panel #filter-add")).toBeHidden();

    const state = await page.evaluate(() => {
      const unified = document.querySelector<HTMLElement>(".mobile-scroll-content")!;
      const sliders = document.querySelector<HTMLElement>("#sliders")!;
      sliders.scrollTop = 100;
      return {
        unifiedOverflowY: getComputedStyle(unified).overflowY,
        unifiedRange: unified.scrollHeight - unified.clientHeight,
        slidersOverflowY: getComputedStyle(sliders).overflowY,
        slidersRange: sliders.scrollHeight - sliders.clientHeight,
        slidersScrollTop: sliders.scrollTop,
        nestedScrollers: [...unified.querySelectorAll<HTMLElement>("*")].filter((element) => {
          const overflow = getComputedStyle(element).overflowY;
          return /auto|scroll/.test(overflow) && element.scrollHeight > element.clientHeight + 1;
        }).map((element) => element.id || element.className),
      };
    });
    expect(state.unifiedOverflowY).toMatch(/auto|scroll/);
    expect(state.unifiedRange).toBeGreaterThan(0);
    expect(state.slidersOverflowY).not.toMatch(/auto|scroll/);
    expect(state.slidersRange).toBeLessThanOrEqual(1);
    expect(state.slidersScrollTop).toBe(0);
    expect(state.nestedScrollers).toEqual([]);
  });

  test("activation updates outgoing and incoming accessibility state immediately", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile-only");
    await page.goto("/");
    const scatter = page.locator(".scatter-content");
    const composition = page.locator(".mobile-sliders-view");

    await page.locator("#mobile-show-sliders").click();

    await expect(scatter).toHaveAttribute("inert", "");
    await expect(composition).not.toHaveAttribute("inert", "");
    await expect(page.locator("#mobile-show-sliders")).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator("#mobile-show-scatter")).toHaveAttribute("aria-pressed", "false");
    await expect(composition).toBeVisible();

    await page.locator("#mobile-show-scatter").click();

    await expect(composition).toHaveAttribute("inert", "");
    await expect(scatter).not.toHaveAttribute("inert", "");
  });

  test("a live reduced-motion change never leaves the requested view blank", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile-only");
    await page.emulateMedia({ reducedMotion: "no-preference" });
    await page.goto("/");
    await page.emulateMedia({ reducedMotion: "reduce" });

    await page.locator("#mobile-show-sliders").click();
    const immediate = await page.evaluate(() => ({
      scatterHidden: document.querySelector(".scatter-content")!.classList.contains("hidden"),
      compositionHidden: document.querySelector(".mobile-sliders-view")!.classList.contains("hidden"),
      compositionFading: document.querySelector(".mobile-sliders-view")!.classList.contains("fading"),
    }));
    expect(immediate).toEqual({
      scatterHidden: true,
      compositionHidden: false,
      compositionFading: false,
    });
    await expect(page.locator(".mobile-sliders-view")).toBeVisible();
  });

  test("enabling reduced motion promptly settles an in-flight crossfade", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile-only");
    await page.emulateMedia({ reducedMotion: "no-preference" });
    await page.goto("/");
    await page.evaluate(() => {
      (window as any).__crossfadeLayoutChanges = 0;
      document.addEventListener("dashboard-layout-change", () => {
        (window as any).__crossfadeLayoutChanges += 1;
      });
      document.querySelector<HTMLElement>("#mobile-show-sliders")!.click();
      if (!document.querySelector(".scatter-content")!.classList.contains("fading")) {
        throw new Error("crossfade did not start synchronously");
      }
    });

    await page.emulateMedia({ reducedMotion: "reduce" });
    await expect.poll(() => page.evaluate(() =>
      matchMedia("(prefers-reduced-motion: reduce)").matches,
    )).toBe(true);
    await expect.poll(() => page.evaluate(() => ({
      scatterHidden: document.querySelector(".scatter-content")!.classList.contains("hidden"),
      scatterFading: document.querySelector(".scatter-content")!.classList.contains("fading"),
      compositionHidden: document.querySelector(".mobile-sliders-view")!.classList.contains("hidden"),
      compositionFading: document.querySelector(".mobile-sliders-view")!.classList.contains("fading"),
    }))).toEqual({
      scatterHidden: true,
      scatterFading: false,
      compositionHidden: false,
      compositionFading: false,
    });
    await page.waitForTimeout(180); // Past the original 150ms timer deadline.
    expect(await page.evaluate(() => ({
      layoutChanges: (window as any).__crossfadeLayoutChanges,
      scatterHidden: document.querySelector(".scatter-content")!.classList.contains("hidden"),
      compositionHidden: document.querySelector(".mobile-sliders-view")!.classList.contains("hidden"),
      fadingCount: document.querySelectorAll(".scatter-content.fading, .mobile-sliders-view.fading").length,
    }))).toEqual({
      layoutChanges: 2,
      scatterHidden: true,
      compositionHidden: false,
      fadingCount: 0,
    });
  });

  test("tapping Performance Tradeoffs returns to scatter", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile-only");
    await page.goto("/");
    // Switch to sliders first
    await page.locator("#mobile-show-sliders").click();
    await expect(page.locator(".mobile-sliders-view")).toBeVisible();
    // Switch back
    await page.locator("#mobile-show-scatter").click();
    await expect(page.locator(".scatter-content")).toBeVisible({ timeout: 2000 });
    await expect(page.locator(".mobile-sliders-view")).toBeHidden();
    await expect(page.locator("#tradeoffs-panel #filter-add")).toBeVisible();
  });

  test("filter state follows its single control subtree across the breakpoint", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile breakpoint ownership");
    await page.goto("/?test=1");
    await page.locator("#filter-add").click();
    const minimum = page.locator(".filter-min").first();
    await minimum.fill("123");

    await page.setViewportSize({ width: 1280, height: 800 });
    await expect(page.locator("#filters-panel #filter-controls")).toBeVisible();
    await expect(page.locator("#filters-panel .filter-min").first()).toHaveValue("123");
    await expect(page.locator("#filter-controls")).toHaveCount(1);

    await page.setViewportSize({ width: 412, height: 915 });
    await expect(page.locator("#tradeoffs-panel #filter-controls")).toBeVisible();
    await expect(page.locator("#tradeoffs-panel .filter-min").first()).toHaveValue("123");
    await expect(page.locator("#filter-controls")).toHaveCount(1);
  });

  test("a pending Composition transition cannot overwrite breakpoint restoration", async ({
    page,
  }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile-only");
    await page.goto("/");

    await page.locator("#mobile-show-sliders").click();
    await page.setViewportSize({ width: 1280, height: 800 });
    await expect(page.locator("#filters-panel #filter-add")).toBeVisible();
    await page.setViewportSize({ width: 412, height: 915 });

    await expect(page.locator(".scatter-content")).toBeVisible();
    await expect(page.locator(".mobile-sliders-view")).toBeHidden();
    await expect(page.locator("#mobile-show-scatter")).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator("#mobile-show-sliders")).toHaveAttribute("aria-pressed", "false");
    await expect(page.locator(".mobile-scroll-content")).toHaveAttribute(
      "aria-label",
      "Performance Tradeoffs",
    );
    await expect(page.locator("#tradeoffs-panel #filter-add")).toBeVisible();
  });
});
