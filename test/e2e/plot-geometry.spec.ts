import { expect, test, type Page } from "@playwright/test";

type Geometry = {
  scatter: { canvas: DOMRectJSON; plot: DOMRectJSON };
  strength: { canvas: DOMRectJSON; plot: DOMRectJSON };
};

type DOMRectJSON = {
  x: number;
  y: number;
  width: number;
  height: number;
  top: number;
  right: number;
  bottom: number;
  left: number;
};

async function waitForDashboard(page: Page) {
  await page.goto("/?test=1");
  await expect(page.locator("#sliders .slider-group").last()).toBeAttached({ timeout: 15_000 });
  await page.evaluate(async () => {
    await document.fonts.ready;
    await Promise.all(
      [...document.querySelectorAll(".fade-in-up")].flatMap((element) =>
        element.getAnimations().map((animation) => animation.finished),
      ),
    );
  });
}

async function readGeometry(page: Page): Promise<Geometry> {
  return page.evaluate(async () => {
    const { computePlotRect, resolvePlotInsets } = await import("/plot-geometry.mjs");
    const insets = resolvePlotInsets(innerWidth);
    const measure = (selector: string) => {
      const canvas = document.querySelector<HTMLCanvasElement>(selector)!;
      const box = canvas.getBoundingClientRect();
      const plot = computePlotRect(box.width, box.height, insets);
      const json = (value: DOMRect | Record<string, number>) => ({
        x: "x" in value ? value.x : value.left,
        y: "y" in value ? value.y : value.top,
        width: value.width,
        height: value.height,
        top: value.top,
        right: value.right,
        bottom: value.bottom,
        left: value.left,
      });
      return {
        canvas: json(box),
        plot: json({
          ...plot,
          x: box.left + plot.left,
          y: box.top + plot.top,
          top: box.top + plot.top,
          right: box.left + plot.right,
          bottom: box.top + plot.bottom,
          left: box.left + plot.left,
        }),
      };
    };
    return { scatter: measure("#scatter-canvas"), strength: measure("#curve-canvas") };
  });
}

async function stableGeometry(page: Page): Promise<Geometry> {
  let previous = JSON.stringify(await readGeometry(page));
  await expect
    .poll(async () => {
      await page.evaluate(() =>
        new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))),
      );
      const current = JSON.stringify(await readGeometry(page));
      const stable = current === previous;
      previous = current;
      return stable;
    })
    .toBe(true);
  return JSON.parse(previous);
}

function expectLandscape(geometry: Geometry) {
  for (const [name, value] of Object.entries(geometry)) {
    expect(value.plot.width, `${name} drawable plot must never be portrait`).toBeGreaterThanOrEqual(
      value.plot.height - 1,
    );
  }
}

function expectDesktopPeers(geometry: Geometry) {
  expect(
    Math.abs(geometry.scatter.plot.width - geometry.strength.plot.width),
    "desktop drawable widths must match",
  ).toBeLessThanOrEqual(1);
  expect(
    Math.abs(geometry.scatter.plot.height - geometry.strength.plot.height),
    "desktop drawable heights must match",
  ).toBeLessThanOrEqual(1);
  expect(
    Math.abs(geometry.scatter.plot.top - geometry.strength.plot.top),
    "desktop drawable tops must align",
  ).toBeLessThanOrEqual(1);
}

test.describe("shared plot geometry", () => {
  test("Scatter omits instructional copy and keeps selectors centered without a reserved hint band", async ({ page }, testInfo) => {
    for (const viewport of testInfo.project.name.startsWith("desktop")
      ? [{ width: 1900, height: 1000 }, { width: 1051, height: 900 }]
      : [{ width: 844, height: 390 }, { width: 412, height: 915 }]) {
      await page.setViewportSize(viewport);
      await waitForDashboard(page);
      await expect.poll(() => page.evaluate(() => {
        const selector = document.querySelector("#axis-selector-x")!.getBoundingClientRect();
        const canvas = document.querySelector<HTMLCanvasElement>("#scatter-canvas")! as HTMLCanvasElement & {
          _plotRect?: { left: number; right: number };
        };
        const strengthCanvas = document.querySelector<HTMLCanvasElement>("#curve-canvas")! as HTMLCanvasElement & {
          _plotRect?: { top: number };
        };
        if (!canvas._plotRect || !strengthCanvas._plotRect) return Number.POSITIVE_INFINITY;
        const canvasBox = canvas.getBoundingClientRect();
        const selectorCenter = (selector.left + selector.right) / 2;
        const plotCenter = canvasBox.left + (canvas._plotRect.left + canvas._plotRect.right) / 2;
        return Math.abs(selectorCenter - plotCenter);
      })).toBeLessThanOrEqual(1);
      const values = await page.evaluate(() => {
        const panel = document.querySelector("#tradeoffs-panel")!;
        const heading = document.querySelector(".tradeoffs-heading")!;
        const title = document.querySelector("#tradeoffs-title")!;
        const selector = document.querySelector("#axis-selector-x")!;
        const canvas = document.querySelector<HTMLCanvasElement>("#scatter-canvas")! as HTMLCanvasElement & {
          _plotRect: { left: number; right: number; top: number };
        };
        const canvasBox = canvas.getBoundingClientRect();
        const selectorBox = selector.getBoundingClientRect();
        const headingBox = heading.getBoundingClientRect();
        const titleBox = title.getBoundingClientRect();
        const titleText = document.createRange();
        titleText.selectNodeContents(title);
        const strengthTitle = document.querySelector("#strength-title")!;
        const strengthTitleText = document.createRange();
        strengthTitleText.selectNodeContents(strengthTitle);
        const strengthCanvas = document.querySelector<HTMLCanvasElement>("#curve-canvas")! as HTMLCanvasElement & {
          _plotRect: { top: number };
        };
        const strengthCanvasBox = strengthCanvas.getBoundingClientRect();
        return {
          instructionPresent: panel.textContent?.includes("Choose either objective on each axis."),
          compactHintDisplay: getComputedStyle(document.querySelector("#mobile-hint")!).display,
          headingBottom: headingBox.bottom,
          titleBottom: titleBox.bottom,
          canvasTop: canvasBox.top,
          plotTop: canvasBox.top + canvas._plotRect.top,
          titleTextBottom: titleText.getBoundingClientRect().bottom,
          strengthTitleTextBottom: strengthTitleText.getBoundingClientRect().bottom,
          strengthPlotTop: strengthCanvasBox.top + strengthCanvas._plotRect.top,
          selectorCenter: (selectorBox.left + selectorBox.right) / 2,
          plotCenter: canvasBox.left + (canvas._plotRect.left + canvas._plotRect.right) / 2,
        };
      });
      expect(values.instructionPresent, `${viewport.width}px instruction removed`).toBe(false);
      expect(values.compactHintDisplay, `${viewport.width}px no scatter hint row`).toBe("none");
      if (testInfo.project.name.startsWith("desktop")) {
        expect(values.canvasTop - values.headingBottom, `${viewport.width}px heading-to-canvas gap`).toBeLessThanOrEqual(2.5);
        expect(values.plotTop - values.titleTextBottom, `${viewport.width}px Tradeoffs text-to-drawable gap`).toBeLessThanOrEqual(18);
        expect(values.strengthPlotTop - values.strengthTitleTextBottom, `${viewport.width}px Strength text-to-drawable gap`).toBeLessThanOrEqual(18);
      }
      expect(Math.abs(values.selectorCenter - values.plotCenter)).toBeLessThanOrEqual(1);
    }
  });

  test("desktop plots stay landscape, equal, and aligned across laptop and fallback sizes", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop geometry contract");
    for (const viewport of [
      { width: 1728, height: 1117 },
      { width: 1728, height: 1000 },
      { width: 1280, height: 800 },
      { width: 1280, height: 400 },
      { width: 1051, height: 900 },
    ]) {
      await page.setViewportSize(viewport);
      await waitForDashboard(page);
      const geometry = await stableGeometry(page);
      expectLandscape(geometry);
      expectDesktopPeers(geometry);
      if (viewport.width === 1728 && [1000, 1117].includes(viewport.height)) {
        for (const [name, value] of Object.entries(geometry)) {
          expect(Math.abs(value.plot.width - 480), `${name} managed desktop drawable width`).toBeLessThanOrEqual(1);
          expect(Math.abs(value.plot.height - 320), `${name} managed desktop drawable height`).toBeLessThanOrEqual(1);
        }
      }
    }
  });

  test("crossing responsive and managed-laptop thresholds restores valid geometry", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop resize contract");
    await page.setViewportSize({ width: 1051, height: 900 });
    await waitForDashboard(page);
    let beforeManaged: Geometry | null = null;
    for (const viewport of [
      { width: 1050, height: 900 },
      { width: 1051, height: 900 },
      { width: 1199, height: 1000 },
      { width: 1200, height: 1000 },
      { width: 1200, height: 899 },
      { width: 1200, height: 900 },
    ]) {
      await page.setViewportSize(viewport);
      const geometry = await stableGeometry(page);
      expectLandscape(geometry);
      for (const value of Object.values(geometry)) {
        expect(value.canvas.width).toBeGreaterThan(100);
        expect(value.canvas.height).toBeGreaterThan(80);
      }
      if (viewport.width >= 1051) expectDesktopPeers(geometry);
      if (viewport.width === 1200 && viewport.height === 899) beforeManaged = geometry;
      if (viewport.width === 1200 && viewport.height === 900) {
        expect(beforeManaged).not.toBeNull();
        expect(
          geometry.scatter.plot.width,
          "entering managed mode must not shrink drawable width",
        ).toBeGreaterThanOrEqual(beforeManaged!.scatter.plot.width);
        expect(
          geometry.scatter.plot.height,
          "entering managed mode must not shrink drawable height",
        ).toBeGreaterThanOrEqual(beforeManaged!.scatter.plot.height);
        expect(
          geometry.scatter.plot.width * geometry.scatter.plot.height,
          "entering managed mode must not shrink drawable area",
        ).toBeGreaterThanOrEqual(
          beforeManaged!.scatter.plot.width * beforeManaged!.scatter.plot.height,
        );
      }
    }
  });

  test("mobile drawable plots stay equal and never become portrait", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile geometry contract");
    for (const viewport of [
      { width: 412, height: 915 },
      { width: 320, height: 720 },
      { width: 844, height: 390 },
    ]) {
      await page.setViewportSize(viewport);
      await waitForDashboard(page);
      const geometry = await stableGeometry(page);
      expectLandscape(geometry);
      expect(Math.abs(geometry.scatter.plot.width - geometry.strength.plot.width)).toBeLessThanOrEqual(1);
      expect(Math.abs(geometry.scatter.plot.height - geometry.strength.plot.height)).toBeLessThanOrEqual(1);
    }
  });

  test("visible Strength resizes while mobile Scatter is hidden", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile hidden-view geometry contract");
    await page.setViewportSize({ width: 412, height: 915 });
    await waitForDashboard(page);
    await page.locator("#mobile-show-sliders").click();
    await expect(page.locator(".scatter-content")).toBeHidden();

    await page.setViewportSize({ width: 320, height: 720 });
    await expect.poll(() => page.locator("#curve-canvas").evaluate((canvas) => {
      const element = canvas as HTMLCanvasElement;
      const rect = element.getBoundingClientRect();
      const owner = element.parentElement!.getBoundingClientRect();
      return {
        fitsOwner: rect.width <= owner.width + 1,
        dprWidthError: Math.abs(element.width - Math.round(rect.width * devicePixelRatio)),
        dprHeightError: Math.abs(element.height - Math.round(rect.height * devicePixelRatio)),
      };
    })).toEqual({ fitsOwner: true, dprWidthError: 0, dprHeightError: 0 });
  });

  test("canvas backing stores track CSS geometry and DPR after resize", async ({ page }) => {
    await waitForDashboard(page);
    for (const viewport of [
      { width: 1280, height: 800 },
      { width: 1050, height: 900 },
      { width: 412, height: 915 },
    ]) {
      await page.setViewportSize(viewport);
      await stableGeometry(page);
      const values = await page.locator("#scatter-canvas, #curve-canvas").evaluateAll((canvases) =>
        canvases.map((canvas) => {
          const element = canvas as HTMLCanvasElement;
          const rect = element.getBoundingClientRect();
          return {
            cssWidth: rect.width,
            cssHeight: rect.height,
            backingWidth: element.width,
            backingHeight: element.height,
            dpr: devicePixelRatio,
          };
        }),
      );
      for (const value of values) {
        expect(Math.abs(value.backingWidth - Math.round(value.cssWidth * value.dpr))).toBeLessThanOrEqual(1);
        expect(Math.abs(value.backingHeight - Math.round(value.cssHeight * value.dpr))).toBeLessThanOrEqual(1);
      }
    }
  });

  test("opening References does not change scatter or strength geometry", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop cross-column geometry contract");
    await page.setViewportSize({ width: 1728, height: 1000 });
    await waitForDashboard(page);
    const before = await stableGeometry(page);

    await page.locator(".ref-details > summary").first().click();
    await expect.poll(() => page.locator(".ref-description-motion").first().evaluate((element) =>
      element.getAnimations().length,
    )).toBe(0);
    const after = await stableGeometry(page);

    for (const name of ["scatter", "strength"] as const) {
      for (const key of ["top", "width", "height"] as const) {
        expect(Math.abs(after[name].plot[key] - before[name].plot[key]), `${name} plot ${key} changed`)
          .toBeLessThanOrEqual(1);
      }
    }
  });

  test("filter count never changes scatter or strength geometry", async ({ page }) => {
    await waitForDashboard(page);
    const before = await stableGeometry(page);
    const beforeScrollY = await page.evaluate(() => scrollY);
    const add = page.locator("#filter-add");
    for (let index = 0; index < 5; index += 1) await add.click();
    await expect(page.locator(".filter-row-wrapper:not(.collapsed)")).toHaveCount(5);
    await expect
      .poll(() =>
        page
          .locator(".filter-row-wrapper:not(.collapsed)")
          .evaluateAll((rows) => rows.reduce((count, row) => count + row.getAnimations().length, 0)),
      )
      .toBe(0);
    const insertionScrollY = await page.evaluate(() => scrollY);
    const after = await stableGeometry(page);
    for (const name of ["scatter", "strength"] as const) {
      for (const key of ["width", "height"] as const) {
        expect(
          Math.abs(after[name].canvas[key] - before[name].canvas[key]),
          `${name} canvas ${key} changed`,
        ).toBeLessThanOrEqual(1);
        expect(
          Math.abs(after[name].plot[key] - before[name].plot[key]),
          `${name} plot ${key} changed`,
        ).toBeLessThanOrEqual(1);
      }
      expect(
        Math.abs(after[name].canvas.y + insertionScrollY - before[name].canvas.y - beforeScrollY),
        `${name} document-space y changed`,
      ).toBeLessThanOrEqual(1);
      expect(
        Math.abs(after[name].plot.y + insertionScrollY - before[name].plot.y - beforeScrollY),
        `${name} plot document-space y changed`,
      ).toBeLessThanOrEqual(1);
    }
  });
});
