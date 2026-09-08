import { test, expect } from "@playwright/test";
import { hoverRenderedScatterPoint } from "./canvas-frame-probe";

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
 * these tests would have passed against ungated code. Most cases set it before
 * goto to cover startup, while a dedicated case verifies live preference changes.
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
    test.skip(!testInfo.project.name.startsWith("desktop"), "toggle is desktop-visible");
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

  for (const axis of ["x", "y"] as const) {
    test(`${axis.toUpperCase()} axis point and label transitions collapse together`, async ({ page }, testInfo) => {
      test.skip(!testInfo.project.name.startsWith("desktop"), "one project is enough");
      await openReducedMotion(page, { waitForModel: true });
      const group = page.locator(`#axis-selector-${axis}`);
      const alternate = group.locator('.axis-option-face:has(input[type="radio"]:not(:checked))');
      const initialWrapper = axis === "y"
        ? await page.locator(".axis-selector-y-wrap").boundingBox()
        : null;
      const initialFaces = await group.locator(".axis-option-face").evaluateAll((faces) =>
        faces.map((face) => face.getBoundingClientRect().top),
      );

      await alternate.click();
      expect(await page.evaluate(() => (window as any).__test.isScatterTransitionActive)).toBe(false);
      await expect(group).not.toHaveAttribute("data-transitioning", "true");
      await expect(group.locator('input[type="radio"]:checked')).toBeFocused();
      await expect.poll(() => group.locator(".axis-option-face, .axis-option-pill").evaluateAll((elements) =>
        elements.reduce((count, element) => count + element.getAnimations()
          .filter((animation) => animation.playState === "running").length, 0),
      ), { timeout: 150, intervals: [10, 20, 40, 80] }).toBe(0);
      if (axis === "y") {
        const finalWrapper = await page.locator(".axis-selector-y-wrap").boundingBox();
        expect(Math.abs(finalWrapper!.x - initialWrapper!.x)).toBeLessThanOrEqual(0.5);
        const finalFaces = await group.locator(".axis-option-face").evaluateAll((faces) =>
          faces.map((face) => face.getBoundingClientRect().top),
        );
        expect(Math.abs(finalFaces[0] - initialFaces[1])).toBeLessThanOrEqual(1);
        expect(Math.abs(finalFaces[1] - initialFaces[0])).toBeLessThanOrEqual(1);
      }
    });
  }

  test("no infinite CSS animation is left running", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "one project is enough");
    await openReducedMotion(page);

    await expect
      .poll(() => page.evaluate(() =>
        document
          .getAnimations()
          .filter((animation) =>
            animation.playState === "running" &&
            animation.effect?.getTiming().iterations === Infinity
          )
          .map((animation) => (animation as any).animationName || "unnamed"),
      ))
      .toEqual([]);
  });

  test("the Material Source preview curve completes on the next frame", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "Material Source controls are desktop-visible");
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
    test.skip(!testInfo.project.name.startsWith("desktop"), "Material Source controls are desktop-visible");
    await openReducedMotion(page, { waitForModel: true });
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await inactive.hover();
    await expect(inactive).toHaveAttribute("data-previewing", "");
    expect(
      await inactive.evaluate((el) => getComputedStyle(el, "::after").animationName),
    ).toBe("none");
  });

  test("the title gradient is pinned, not frozen mid-sweep", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "one project is enough");
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
    test.skip(!testInfo.project.name.startsWith("desktop"), "one project is enough");
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

  test("insight and disclosure structural changes settle immediately", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop controls are visible together");
    await openReducedMotion(page, { waitForModel: true });

    await page.getByRole("button", { name: "Water ingredient insight" }).click();
    const ingredient = await page.locator(".ingredient-insight-body").evaluate((element) => ({
      ghosts: element.querySelectorAll(".content-swap-ghost").length,
      animations: element.getAnimations({ subtree: true }).length,
      text: element.textContent,
    }));
    expect(ingredient.ghosts).toBe(0);
    expect(ingredient.animations).toBe(0);
    expect(ingredient.text).toContain("water-to-binder");

    const details = page.locator(".ref-details").first();
    const wrapper = details.locator(".ref-description-motion");
    const panel = page.locator("#references-panel");
    const list = page.locator(".ref-list");
    const closedHeight = await panel.evaluate((element) => element.getBoundingClientRect().height);
    await details.locator("summary").click();
    expect(await wrapper.evaluate((element) => element.getAnimations().length)).toBe(0);
    const openState = await list.evaluate((element) => ({
      panelHeight: element.closest("#references-panel")!.getBoundingClientRect().height,
      overflow: element.scrollHeight > element.clientHeight + 1,
      tabIndex: (element as HTMLElement).tabIndex,
    }));
    expect(openState.panelHeight).toBeGreaterThan(closedHeight);
    expect(openState.tabIndex).toBe(openState.overflow ? 0 : -1);

    await details.locator("summary").click();
    expect(await wrapper.evaluate((element) => element.getAnimations().length)).toBe(0);
    await expect(details).not.toHaveAttribute("open", "");
    await expect(wrapper).toHaveAttribute("inert", "");
    const restoredState = await list.evaluate((element) => ({
      panelHeight: element.closest("#references-panel")!.getBoundingClientRect().height,
      overflow: element.scrollHeight > element.clientHeight + 1,
      tabIndex: (element as HTMLElement).tabIndex,
    }));
    expect(Math.abs(restoredState.panelHeight - closedHeight)).toBeLessThanOrEqual(1);
    expect(restoredState.tabIndex).toBe(restoredState.overflow ? 0 : -1);
  });

  test("a live preference change settles an active Y exchange and normal motion resumes", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "axis controls are desktop-visible");
    await page.emulateMedia({ reducedMotion: "no-preference" });
    await page.goto("/?test=1");
    await expect(page.locator("#scatter-canvas")).toBeVisible({ timeout: 15_000 });
    const group = page.locator("#axis-selector-y");
    const wrapper = page.locator(".axis-selector-y-wrap");
    const initialLeft = (await wrapper.boundingBox())!.x;

    await group.locator('.axis-option-face:has(input[type="radio"]:not(:checked))').click();
    await expect(group).toHaveAttribute("data-transitioning", "true");
    await page.emulateMedia({ reducedMotion: "reduce" });
    await expect.poll(() => page.evaluate(() => ({
      preference: matchMedia("(prefers-reduced-motion: reduce)").matches,
      transitioning: (window as any).__test.isScatterTransitionActive,
    }))).toEqual({ preference: true, transitioning: false });
    await expect(group).not.toHaveAttribute("data-transitioning", "true");
    expect(Math.abs((await wrapper.boundingBox())!.x - initialLeft)).toBeLessThanOrEqual(0.5);
    await expect.poll(() => group.locator(".axis-option-face, .axis-option-pill").evaluateAll((elements) =>
      elements.reduce((count, element) => count + element.getAnimations()
        .filter((animation) => animation.playState === "running").length, 0),
    ), { timeout: 150, intervals: [10, 20, 40, 80] }).toBe(0);

    await page.emulateMedia({ reducedMotion: "no-preference" });
    await expect.poll(() => page.evaluate(() =>
      !matchMedia("(prefers-reduced-motion: reduce)").matches,
    )).toBe(true);
    await group.locator('.axis-option-face:has(input[type="radio"]:not(:checked))').click();
    await expect(group).toHaveAttribute("data-transitioning", "true");
    const timings = await group.locator(".axis-option-face").evaluateAll((faces) =>
      faces.flatMap((face) => face.getAnimations())
        .filter((animation) => animation instanceof CSSTransition)
        .filter((animation) => (animation as CSSTransition).transitionProperty === "transform")
        .map((animation) => animation.effect?.getTiming().duration),
    );
    expect(timings.length).toBeGreaterThan(0);
    expect(timings.every((duration) => duration === 350)).toBe(true);
  });

  test("a live preference change settles every active canvas transition", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop controls are visible together");
    await page.emulateMedia({ reducedMotion: "no-preference" });
    await page.goto("/?test=1");
    await page.waitForFunction(() => (window as any).__test?.modelReady === true, null, {
      timeout: 20_000,
    });

    await hoverRenderedScatterPoint(page);
    await page.locator("#scatter-canvas").click();
    await expect.poll(() => page.evaluate(() => ({
      composition: (window as any).__test.isCompositionTransitionActive,
      curve: (window as any).__test.isCurveTransitionActive,
    }))).toEqual({ composition: true, curve: true });
    await page.evaluate(() => {
      const query = matchMedia("(prefers-reduced-motion: reduce)");
      (window as any).__canvasReducedState = new Promise((resolve) => {
        query.addEventListener("change", () => requestAnimationFrame(() => resolve({
          composition: (window as any).__test.isCompositionTransitionActive,
          curve: (window as any).__test.isCurveTransitionActive,
          loop: (window as any).__test.isAnimLoopActive,
        })), { once: true });
      });
    });
    await page.emulateMedia({ reducedMotion: "reduce" });
    expect(await page.evaluate(() => (window as any).__canvasReducedState))
      .toEqual({ composition: false, curve: false, loop: false });

    await page.emulateMedia({ reducedMotion: "no-preference" });
    const source = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await source.hover();
    await expect.poll(() => page.evaluate(() =>
      (window as any).__test.isPreviewCurveTransitionActive,
    )).toBe(true);
    await page.evaluate(() => {
      const query = matchMedia("(prefers-reduced-motion: reduce)");
      (window as any).__previewReducedState = new Promise((resolve) => {
        query.addEventListener("change", () => requestAnimationFrame(() => resolve({
          preview: (window as any).__test.isPreviewCurveTransitionActive,
          loop: (window as any).__test.isAnimLoopActive,
        })), { once: true });
      });
    });
    await page.emulateMedia({ reducedMotion: "reduce" });
    expect(await page.evaluate(() => (window as any).__previewReducedState))
      .toEqual({ preview: false, loop: false });

    await page.mouse.move(0, 0);
    await expect.poll(() => page.evaluate(() => !(window as any).__test.isAnimLoopActive)).toBe(true);
    await page.emulateMedia({ reducedMotion: "no-preference" });
    await page.locator("#unit-toggle").click();
    await expect.poll(() => page.evaluate(() => (window as any).__test.isAnimLoopActive)).toBe(true);
    await page.evaluate(() => {
      const query = matchMedia("(prefers-reduced-motion: reduce)");
      (window as any).__unitReducedState = new Promise((resolve) => {
        query.addEventListener("change", () => requestAnimationFrame(() => resolve({
          loop: (window as any).__test.isAnimLoopActive,
        })), { once: true });
      });
    });
    await page.emulateMedia({ reducedMotion: "reduce" });
    expect(await page.evaluate(() => (window as any).__unitReducedState))
      .toEqual({ loop: false });
  });

  test("a live preference change suppresses the next structural transaction", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "filter UI is desktop-only");
    await page.emulateMedia({ reducedMotion: "no-preference" });
    await page.goto("/?test=1");
    await expect(page.locator("#filter-add")).toBeVisible({ timeout: 15_000 });

    await page.emulateMedia({ reducedMotion: "reduce" });
    await expect.poll(() => page.evaluate(() =>
      matchMedia("(prefers-reduced-motion: reduce)").matches,
    )).toBe(true);
    await page.locator("#filter-add").click();

    expect(await page.locator("#filter-rows").evaluate((shell) => ({
      animations: shell.getAnimations({ subtree: true })
        .filter((animation) => !(animation instanceof CSSTransition))
        .filter((animation) => animation.playState === "running" && animation.effect !== null)
        .length,
      blockSize: (shell as HTMLElement).style.blockSize,
      rowCount: shell.querySelectorAll(".filter-row-wrapper:not([data-exiting])").length,
    }))).toEqual({ animations: 0, blockSize: "", rowCount: 1 });
  });

  test("script-created structural motion settles immediately without stale state", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "filter UI is desktop-only");
    await openReducedMotion(page, { waitForModel: true });

    await page.locator("#filter-add").click();
    const state = await page.locator(".filter-row-wrapper").first().evaluate((element) => ({
      animations: element.getAnimations().length,
      blockSize: (element as HTMLElement).style.blockSize,
      height: (element as HTMLElement).style.height,
      overflow: (element as HTMLElement).style.overflow,
      opacity: (element as HTMLElement).style.opacity,
      transform: (element as HTMLElement).style.transform,
      wrapperHeight: element.getBoundingClientRect().height,
      rowHeight: element.firstElementChild!.getBoundingClientRect().height,
    }));
    expect(state).toEqual({
      animations: 0,
      blockSize: "",
      height: "",
      overflow: "",
      opacity: "",
      transform: "",
      wrapperHeight: state.rowHeight,
      rowHeight: state.rowHeight,
    });
    await page.locator(".filter-remove-btn").click();
    await expect(page.locator(".filter-row-wrapper")).toHaveCount(0);
    expect(await page.locator("#filter-rows").evaluate((shell) => ({
      animations: shell.getAnimations({ subtree: true })
        .filter((animation) => !(animation instanceof CSSTransition))
        .filter((animation) => animation.playState === "running" && animation.effect !== null)
        .map((animation) => ({
          type: animation.constructor.name,
          target: (animation.effect as KeyframeEffect).target?.className,
        })),
      blockSize: (shell as HTMLElement).style.blockSize,
      paddingBlock: (shell as HTMLElement).style.paddingBlock,
      marginBlockStart: (shell as HTMLElement).style.marginBlockStart,
    }))).toEqual({ animations: [], blockSize: "", paddingBlock: "", marginBlockStart: "" });
  });
});
