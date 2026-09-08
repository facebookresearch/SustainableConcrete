import { test, expect } from "@playwright/test";
import { waitForDashboardLayoutReady } from "./dashboard-helpers";

/**
 * Scatter filter UI invariants. Filters are a multi-row "+/−" interface
 * inside the scatter panel; each row has a column-select dropdown and
 * either numeric min/max controls or categorical class toggles.
 */
test.describe("scatter filter rows", () => {
  test("filter min/max placeholders fit fully inside the input box", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop placeholder geometry");
    await page.goto("/");
    await expect(page.locator("#sliders .slider-group").first()).toBeVisible({ timeout: 15000 });
    await page.locator("#filter-add").click();
    const minInput = page.locator(".filter-min").first();
    const maxInput = page.locator(".filter-max").first();
    await expect(minInput).toBeVisible();
    await expect(maxInput).toBeVisible();

    async function check(locator: import("@playwright/test").Locator) {
      const verdict = await locator.evaluate((el) => {
        const input = el as HTMLInputElement;
        const cs = getComputedStyle(input);
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        if (!ctx) return null;
        ctx.font = `${cs.fontWeight || "normal"} ${cs.fontSize || "12px"} ${cs.fontFamily || "sans-serif"}`;
        const textWidth = ctx.measureText(input.placeholder).width;
        const available = input.clientWidth - (parseFloat(cs.paddingLeft) || 0) - (parseFloat(cs.paddingRight) || 0);
        return { placeholder: input.placeholder, textWidth, available };
      });
      expect(verdict, "input must be measurable").not.toBeNull();
      expect(verdict!.textWidth, `placeholder ${verdict!.placeholder} must fit`).toBeLessThanOrEqual(verdict!.available + 1);
    }
    await check(minInput);
    await check(maxInput);
  });
});

test.describe("two-row filter viewport and focus containment", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/?test=1");
    await waitForDashboardLayoutReady(page);
  });

  async function waitForRows(page: import("@playwright/test").Page, count: number) {
    await expect(page.locator(".filter-row-wrapper:not(.collapsed)")).toHaveCount(count);
    await expect
      .poll(() => page.locator(".filter-row-wrapper:not(.collapsed)").evaluateAll((rows) => rows.reduce((total, row) => total + row.getAnimations().length, 0)))
      .toBe(0);
  }

  async function geometry(page: import("@playwright/test").Page) {
    return page.evaluate(() => {
      const read = (selector: string) => {
        const element = document.querySelector<HTMLElement>(selector)!;
        const rect = element.getBoundingClientRect();
        return {
          left: rect.left,
          right: rect.right,
          top: rect.top,
          bottom: rect.bottom,
          width: rect.width,
          height: rect.height,
        };
      };
      const rows = document.querySelector<HTMLElement>("#filter-rows")!;
      const owner = rows.closest<HTMLElement>(".panel")!;
      const documentY = window.scrollY;
      const ownerRect = owner.getBoundingClientRect();
      return {
        documentY,
        plot: read("#scatter-canvas"),
        strengthPlot: read("#curve-canvas"),
        controls: read(".filter-header"),
        list: read("#filter-rows"),
        shell: { top: ownerRect.top, bottom: ownerRect.bottom, height: ownerRect.height },
        follower: read(window.innerWidth > 1050 ? "#references-panel" : "#mix-insight"),
        footer: read(".site-footer"),
        listClientHeight: rows.clientHeight,
        listScrollHeight: rows.scrollHeight,
        listScrollTop: rows.scrollTop,
        listTabIndex: rows.tabIndex,
        outerScrollTop: document.querySelector<HTMLElement>(".mobile-scroll-content")!.scrollTop,
      };
    });
  }

  test("managed laptop row capacity grows from three to four without moving fixed content", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "managed filter capacity contract");
    const add = page.locator("#filter-add");
    for (const scenario of [
      { viewport: { width: 1728, height: 1000 }, expanded: 3, overflowAt: 4 },
      { viewport: { width: 1728, height: 1117 }, expanded: 4, overflowAt: 5 },
    ]) {
      await page.setViewportSize(scenario.viewport);
      await page.reload();
      await waitForDashboardLayoutReady(page);
      const zero = await geometry(page);
      const snapshots = [zero];
      for (let count = 1; count <= scenario.expanded; count += 1) {
        await add.click();
        await waitForRows(page, count);
        snapshots.push(await geometry(page));
      }
      const expanded = snapshots.at(-1)!;
      expect(expanded.listScrollHeight).toBeLessThanOrEqual(expanded.listClientHeight + 1);
      expect(expanded.list.height).toBeGreaterThan(zero.list.height + 20);
      await add.click();
      await waitForRows(page, scenario.overflowAt);
      const overflow = await geometry(page);
      snapshots.push(overflow);
      expect(overflow.listScrollHeight).toBeGreaterThan(overflow.listClientHeight + 1);
      expect(Math.abs(overflow.list.height - expanded.list.height)).toBeLessThanOrEqual(1);
      expect(Math.abs(overflow.shell.height - expanded.shell.height)).toBeLessThanOrEqual(1);
      expect(
        Math.abs(overflow.follower.top + overflow.documentY - expanded.follower.top - expanded.documentY),
        "References moved after managed filter capacity was reached",
      ).toBeLessThanOrEqual(1);
      for (const [count, snapshot] of snapshots.entries()) {
        for (const selector of ["plot", "strengthPlot"] as const) {
          for (const key of ["left", "right", "top", "bottom", "width", "height"] as const) {
            const documentOffset = key === "top" || key === "bottom"
              ? snapshot.documentY - zero.documentY
              : 0;
            expect(
              Math.abs(snapshot[selector][key] + documentOffset - zero[selector][key]),
              `${selector} ${key} changed in document space at ${count} rows`,
            ).toBeLessThanOrEqual(1);
          }
        }
      }
      for (const snapshot of [expanded, overflow]) {
        for (const selector of ["follower", "footer"] as const) {
          for (const key of ["left", "right", "top", "bottom", "width", "height"] as const) {
            const documentOffset = key === "top" || key === "bottom"
              ? snapshot.documentY - expanded.documentY
              : 0;
            expect(
              Math.abs(snapshot[selector][key] + documentOffset - expanded[selector][key]),
              `${selector} ${key} changed after managed capacity was reached`,
            ).toBeLessThanOrEqual(1);
          }
        }
      }
    }
  });

  test("zero through three rows reserve only two slots and keep the plot stationary", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name === "desktop" && (await page.viewportSize())!.width >= 1200 && (await page.viewportSize())!.height >= 900, "managed laptop has a taller filter cap");
    const add = page.locator("#filter-add");
    const zero = await geometry(page);
    expect(zero.list.height).toBe(0);
    expect(zero.listScrollHeight).toBe(0);
    expect(zero.listTabIndex).toBe(-1);

    await add.click();
    await waitForRows(page, 1);
    const one = await geometry(page);
    expect(one.list.height).toBeGreaterThan(20);
    expect(one.listScrollHeight).toBeLessThanOrEqual(one.listClientHeight + 1);
    expect(one.shell.height).toBeGreaterThan(zero.shell.height + 20);

    await add.click();
    await waitForRows(page, 2);
    const two = await geometry(page);
    expect(two.list.height).toBeGreaterThan(one.list.height + 20);
    expect(two.listScrollHeight).toBeLessThanOrEqual(two.listClientHeight + 1);
    expect(two.shell.height).toBeGreaterThan(one.shell.height + 20);
    expect(two.listTabIndex).toBe(-1);

    await add.click();
    await waitForRows(page, 3);
    const three = await geometry(page);
    expect(Math.abs(three.list.height - two.list.height)).toBeLessThanOrEqual(1);
    expect(three.listScrollHeight).toBeGreaterThan(three.listClientHeight + 1);
    expect(Math.abs(three.shell.height - two.shell.height)).toBeLessThanOrEqual(1);
    expect(
      Math.abs(three.follower.top + three.documentY - two.follower.top - two.documentY),
      "Filters follower moved in document space",
    ).toBeLessThanOrEqual(1);
    expect(three.listScrollTop).toBeGreaterThan(0);
    expect(three.outerScrollTop).toBe(0);
    expect(three.listTabIndex).toBe(0);

    for (const key of ["top", "bottom", "height"] as const) {
      const documentOffset = key === "height" ? 0 : three.documentY - zero.documentY;
      expect(
        Math.abs(three.plot[key] + documentOffset - zero.plot[key]),
        `plot ${key} changed in document space`,
      ).toBeLessThanOrEqual(1);
      expect(
        Math.abs(three.controls[key] + documentOffset - zero.controls[key]),
        `filter controls ${key} changed in document space`,
      ).toBeLessThanOrEqual(1);
    }
  });

  test("filter rows own native scrolling and permit boundary chaining", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "mobile", "mobile scroll ownership contract");
    const add = page.locator("#filter-add");
    for (let index = 0; index < 6; index += 1) await add.click();
    await waitForRows(page, 6);

    const rows = page.locator("#filter-rows");
    await rows.scrollIntoViewIfNeeded();
    expect(await rows.evaluate((element) => {
      const style = getComputedStyle(element);
      return { touchAction: style.touchAction, overscrollBehaviorY: style.overscrollBehaviorY };
    })).toEqual({ touchAction: "pan-y", overscrollBehaviorY: "auto" });
    const wheelOverRows = async () => {
      const box = await rows.boundingBox();
      if (!box) throw new Error("filter rows have no bounding box");
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.wheel(0, box.height);
    };

    await rows.evaluate((element) => {
      element.scrollTop = 0;
    });
    const before = await geometry(page);
    await wheelOverRows();
    await expect.poll(() => rows.evaluate((element) => element.scrollTop)).toBeGreaterThan(0);
    const local = await geometry(page);
    expect(local.documentY).toBe(before.documentY);
    expect(local.outerScrollTop).toBe(0);
    expect(local.plot.top).toBeCloseTo(before.plot.top, 0);
    expect(local.controls.top).toBeCloseTo(before.controls.top, 0);

    await rows.evaluate((element) => {
      element.scrollTop = element.scrollHeight;
    });
    await expect.poll(() => rows.evaluate((element) =>
      Math.abs(element.scrollHeight - element.clientHeight - element.scrollTop),
    )).toBeLessThanOrEqual(1);
    expect(await page.locator(".mobile-scroll-content").evaluate((element) => element.scrollTop)).toBe(0);
  });

  test("third and later additions reveal only the filter list and preserve Add focus", async ({ page }) => {
    const add = page.locator("#filter-add");
    await add.focus();
    for (let index = 0; index < 8; index += 1) await add.click();
    await waitForRows(page, 8);
    await expect(add).toBeFocused();
    const state = await geometry(page);
    expect(state.listScrollTop).toBeGreaterThan(0);
    expect(state.outerScrollTop).toBe(0);
    const reveal = await page.locator(".filter-row-wrapper:not(.collapsed)").last().evaluate((element) => {
      const row = element.getBoundingClientRect();
      const owner = document.querySelector("#filter-rows")!.getBoundingClientRect();
      return { rowTop: row.top, rowBottom: row.bottom, ownerTop: owner.top, ownerBottom: owner.bottom };
    });
    expect(reveal.rowTop).toBeGreaterThanOrEqual(reveal.ownerTop - 1);
    expect(reveal.rowBottom).toBeLessThanOrEqual(reveal.ownerBottom + 1);
  });

  test("capacity-crossing entry stays bottom anchored throughout structural motion", async ({ page }) => {
    const add = page.locator("#filter-add");
    await add.click();
    await add.click();
    await waitForRows(page, 2);

    const samples = await page.evaluate(async () => {
      const owner = document.querySelector<HTMLElement>("#filter-rows")!;
      document.querySelector<HTMLButtonElement>("#filter-add")!.click();
      const newest = owner.querySelector<HTMLElement>(".filter-row-wrapper:last-child")!;
      const frames = [];
      for (let frame = 0; frame < 40; frame += 1) {
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        const ownerRect = owner.getBoundingClientRect();
        const rowRect = newest.getBoundingClientRect();
        frames.push({
          rowBottom: rowRect.bottom,
          ownerBottom: ownerRect.bottom,
          scrollTop: owner.scrollTop,
          running: newest.getAnimations().some((animation) => animation.playState === "running"),
        });
        if (!frames.at(-1)!.running) break;
      }
      return frames;
    });

    const animated = samples.filter((sample) => sample.running && sample.rowBottom > 0);
    expect(animated.length).toBeGreaterThan(2);
    for (const sample of animated) {
      expect(sample.rowBottom).toBeLessThanOrEqual(sample.ownerBottom + 1);
    }
    expect(samples.some((sample) => sample.scrollTop > 0)).toBe(true);
  });

  test("rapid capacity-crossing additions keep the newest transaction bottom anchored", async ({ page }) => {
    const add = page.locator("#filter-add");
    await add.click();
    await add.click();
    await waitForRows(page, 2);

    const samples = await page.evaluate(async () => {
      const owner = document.querySelector<HTMLElement>("#filter-rows")!;
      const button = document.querySelector<HTMLButtonElement>("#filter-add")!;
      button.click();
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      button.click();
      const frames = [];
      for (let frame = 0; frame < 45; frame += 1) {
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        const maxScroll = Math.max(0, owner.scrollHeight - owner.clientHeight);
        frames.push({ maxScroll, scrollTop: owner.scrollTop });
        if (owner.getAnimations({ subtree: true }).every((animation) => animation.playState !== "running")) {
          break;
        }
      }
      return frames;
    });

    const overflowing = samples.filter((sample) => sample.maxScroll > 1);
    expect(overflowing.length).toBeGreaterThan(2);
    for (const sample of overflowing) {
      expect(
        Math.abs(sample.maxScroll - sample.scrollTop),
        "the newest insertion must own bottom anchoring on every overflow frame",
      ).toBeLessThanOrEqual(1);
    }
  });

  test("manual filter scrolling cancels automatic entry anchoring", async ({ page }) => {
    const add = page.locator("#filter-add");
    await add.click();
    await add.click();
    await waitForRows(page, 2);

    const state = await page.evaluate(async () => {
      const owner = document.querySelector<HTMLElement>("#filter-rows")!;
      document.querySelector<HTMLButtonElement>("#filter-add")!.click();
      const newest = owner.querySelector<HTMLElement>(".filter-row-wrapper:last-child")!;
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      owner.dispatchEvent(new WheelEvent("wheel", { deltaY: -20, bubbles: true }));
      owner.scrollTop = 0;
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      return {
        scrollTop: owner.scrollTop,
        running: newest.getAnimations().some((animation) => animation.playState === "running"),
      };
    });

    expect(state.running).toBe(true);
    expect(state.scrollTop).toBe(0);
    await waitForRows(page, 3);
  });

  test("resize interruption settles to intrinsic geometry without stale state", async ({ page }) => {
    await page.locator("#filter-add").click();
    const wrapper = page.locator(".filter-row-wrapper").first();
    await expect.poll(() => wrapper.evaluate((element) => element.getAnimations().length)).toBeGreaterThan(0);

    const runningAfterResize = await page.evaluate(async () => {
      window.dispatchEvent(new Event("resize"));
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      return document.querySelector(".filter-row-wrapper")!.getAnimations()
        .filter((animation) => animation.playState === "running" && animation.effect !== null)
        .map((animation) => ({
          type: animation.constructor.name,
          target: (animation.effect as KeyframeEffect).target?.className,
          property: animation instanceof CSSTransition ? animation.transitionProperty : null,
        }));
    });

    expect(runningAfterResize).toEqual([]);
    const state = await wrapper.evaluate((element) => ({
      blockSize: (element as HTMLElement).style.blockSize,
      height: (element as HTMLElement).style.height,
      overflow: (element as HTMLElement).style.overflow,
      opacity: (element as HTMLElement).style.opacity,
      transform: (element as HTMLElement).style.transform,
      wrapperHeight: element.getBoundingClientRect().height,
      rowHeight: element.firstElementChild!.getBoundingClientRect().height,
    }));
    expect(state).toEqual({
      blockSize: "",
      height: "",
      overflow: "",
      opacity: "",
      transform: "",
      wrapperHeight: state.rowHeight,
      rowHeight: state.rowHeight,
    });
  });

  test("removing rows compacts below two slots and moves focus deterministically", async ({ page }) => {
    const add = page.locator("#filter-add");
    for (let index = 0; index < 3; index += 1) await add.click();
    await waitForRows(page, 3);
    const activeRows = page.locator(".filter-row-wrapper:not(.collapsed)");

    await activeRows.nth(1).locator(".filter-remove-btn").focus();
    await activeRows.nth(1).locator(".filter-remove-btn").click();
    await waitForRows(page, 2);
    await expect(activeRows.nth(1).locator(".filter-col")).toBeFocused();
    const two = await geometry(page);
    expect(two.listScrollHeight).toBeLessThanOrEqual(two.listClientHeight + 1);
    expect(two.listTabIndex).toBe(-1);

    await activeRows.nth(1).locator(".filter-remove-btn").click();
    await waitForRows(page, 1);
    await expect(activeRows.nth(0).locator(".filter-col")).toBeFocused();
    const one = await geometry(page);
    expect(one.shell.height).toBeLessThan(two.shell.height - 20);

    await activeRows.nth(0).locator(".filter-remove-btn").click();
    await waitForRows(page, 0);
    await expect(add).toBeFocused();
    const zero = await geometry(page);
    expect(zero.list.height).toBe(0);
    expect(zero.shell.height).toBeLessThan(one.shell.height - 20);
  });

  test("removal updates semantics and focus before structural motion settles", async ({ page }) => {
    const add = page.locator("#filter-add");
    await add.click();
    await add.click();
    await waitForRows(page, 2);
    const rows = page.locator(".filter-row-wrapper:not(.collapsed)");
    const exiting = rows.first();
    await exiting.locator(".filter-min").fill("10000");
    await exiting.locator(".filter-min").dispatchEvent("change");
    await expect.poll(() => page.evaluate(() => (window as any).__test.scatterFilterCount)).toBe(2);

    await exiting.locator(".filter-remove-btn").click();

    await expect(exiting).toHaveAttribute("inert", "");
    await expect.poll(() => page.evaluate(() => (window as any).__test.scatterFilterCount)).toBe(1);
    await expect(rows.nth(1).locator(".filter-col")).toBeFocused();
    await expect.poll(() => exiting.evaluate((element) => element.getAnimations().length)).toBeGreaterThan(0);
    await waitForRows(page, 1);
  });

  test("filter wrappers animate through real intermediate intrinsic sizes and clean up", async ({ page }) => {
    const samples = await page.evaluate(async () => {
      document.querySelector<HTMLButtonElement>("#filter-add")!.click();
      const wrapper = document.querySelector<HTMLElement>(".filter-row-wrapper")!;
      const row = wrapper.querySelector<HTMLElement>(".filter-row")!;
      const heights: number[] = [];
      const shell = document.querySelector<HTMLElement>("#filter-rows")!;
      for (let frame = 0; frame < 120; frame += 1) {
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        heights.push(wrapper.getBoundingClientRect().height);
        const running = shell.getAnimations({ subtree: true })
          .some((animation) => animation.playState === "running" && animation.effect !== null);
        if (!running) break;
      }
      return {
        heights,
        rowHeight: row.getBoundingClientRect().height,
        inlineBlockSize: wrapper.style.blockSize,
        inlineHeight: wrapper.style.height,
        inlineOverflow: wrapper.style.overflow,
        running: wrapper.getAnimations().filter((animation) => animation.playState === "running").length,
      };
    });

    const distinct = samples.heights.filter(
      (height, index, values) => index === 0 || Math.abs(height - values[index - 1]) > 0.5,
    );
    expect(distinct.length, JSON.stringify(samples.heights)).toBeGreaterThan(2);
    expect(distinct[0]).toBeLessThan(samples.rowHeight - 1);
    for (let index = 1; index < distinct.length; index += 1) {
      expect(distinct[index]).toBeGreaterThanOrEqual(distinct[index - 1] - 0.5);
    }
    expect(samples.heights.at(-1)).toBeCloseTo(samples.rowHeight, 0);
    expect(samples.inlineBlockSize).toBe("");
    expect(samples.inlineHeight).toBe("");
    expect(samples.inlineOverflow).toBe("");
    expect(samples.running).toBe(0);
  });

  test("a new filter survives completion of an earlier Clear all without stale transition state", async ({ page }) => {
    const add = page.locator("#filter-add");
    await add.click();
    await waitForRows(page, 1);
    await page.locator("#filter-clear").click();
    await add.click();
    const newest = page.locator(".filter-row-wrapper:not([data-exiting])").last();
    await newest.locator(".filter-min").fill("10000");
    await newest.locator(".filter-min").dispatchEvent("change");
    await expect.poll(() => page.evaluate(() => (window as any).__test.scatterFilterCount)).toBe(1);
    await expect(newest).toBeVisible();
    await expect.poll(() => page.locator(".filter-row-wrapper").evaluateAll((rows) =>
      rows.reduce((total, row) => total + row.getAnimations().length, 0),
    )).toBe(0);
    expect(await page.locator(".filter-row-wrapper").count()).toBe(1);
    const cleanup = await newest.evaluate((element) => ({
      inert: element.hasAttribute("inert"),
      blockSize: (element as HTMLElement).style.blockSize,
      height: (element as HTMLElement).style.height,
      overflow: (element as HTMLElement).style.overflow,
      opacity: (element as HTMLElement).style.opacity,
      transform: (element as HTMLElement).style.transform,
    }));
    expect(cleanup).toEqual({
      inert: false,
      blockSize: "",
      height: "",
      overflow: "",
      opacity: "",
      transform: "",
    });
  });

  test("expanded row releases animated height before categorical controls grow", async ({ page }) => {
    const msIdx = await page.evaluate(async () => {
      const payload = await (await fetch("model/compositions.json")).json();
      return payload.column_names.indexOf("Material Source");
    });
    await page.locator("#filter-add").click();
    const wrapper = page.locator(".filter-row-wrapper:not(.collapsed)").first();
    await expect(wrapper.locator(".filter-min")).toBeVisible();
    await expect.poll(() => wrapper.evaluate((element) => getComputedStyle(element).overflow)).not.toBe("hidden");
    await wrapper.locator(".filter-col").selectOption(String(msIdx));
    await expect(wrapper.locator(".filter-cat-btn").first()).toBeVisible();
    const geometry = await wrapper.evaluate((element) => ({
      inlineHeight: (element as HTMLElement).style.height,
      overflow: getComputedStyle(element).overflow,
      rowHeight: element.firstElementChild!.getBoundingClientRect().height,
      wrapperHeight: element.getBoundingClientRect().height,
    }));
    expect(geometry.inlineHeight).toBe("");
    expect(geometry.overflow).not.toBe("hidden");
    expect(geometry.rowHeight).toBeLessThanOrEqual(geometry.wrapperHeight + 1);
  });
});

test.describe("categorical filter rows", () => {
  test("Material Source filters by class, not by min/max", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#sliders .slider-group").first()).toBeAttached({ timeout: 15000 });
    const msIdx = await page.evaluate(async () => {
      const payload = await (await fetch("model/compositions.json")).json();
      return payload.column_names.indexOf("Material Source");
    });
    expect(msIdx, "Material Source column not found").toBeGreaterThanOrEqual(0);

    await page.locator("#filter-add").click();
    const row = page.locator(".filter-row").first();
    await expect(row.locator(".filter-min")).toBeVisible();
    expect(await row.locator(".filter-cat-btn").count()).toBe(0);

    await row.locator(".filter-col").selectOption(String(msIdx));
    const catBtns = row.locator(".filter-cat-btn");
    await expect(catBtns.first()).toBeVisible();
    const count = await catBtns.count();
    expect(count).toBeGreaterThan(1);
    expect(await row.locator(".filter-min").count()).toBe(0);
    expect(await row.locator(".filter-cat-btn.active").count()).toBe(count);

    await catBtns.first().click();
    expect(await row.locator(".filter-cat-btn.active").count()).toBe(count - 1);
    await expect(catBtns.first()).toHaveAttribute("aria-pressed", "false");

    await row.locator(".filter-col").selectOption("0");
    await expect(row.locator(".filter-min")).toBeVisible();
    expect(await row.locator(".filter-cat-btn").count()).toBe(0);
  });
});
