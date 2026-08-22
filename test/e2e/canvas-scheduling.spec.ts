import { expect, test } from "@playwright/test";
import {
  drawCount,
  expectAtMostOneDrawPerCanvasPerFrame,
  hoverRenderedCurveObservation,
  hoverRenderedScatterPoint,
  installCanvasFrameProbe,
  resetCanvasFrameProbe,
  snapshotCanvasFrames,
  waitForCanvasAppReady,
  waitForCanvasLoopToPark,
} from "./canvas-frame-probe";

test.beforeEach(async ({ page }) => {
  await installCanvasFrameProbe(page);
});

async function expectCurveOnly(page: import("@playwright/test").Page) {
  const snapshot = await snapshotCanvasFrames(page);
  expect(drawCount(snapshot, "curve"), "expected curve redraws").toBeGreaterThan(0);
  expect(drawCount(snapshot, "scatter"), JSON.stringify(snapshot.frames)).toBe(0);
  expectAtMostOneDrawPerCanvasPerFrame(snapshot);
}

async function expectScatterOnly(page: import("@playwright/test").Page) {
  const snapshot = await snapshotCanvasFrames(page);
  expect(drawCount(snapshot, "scatter"), "expected scatter redraws").toBeGreaterThan(0);
  expect(drawCount(snapshot, "curve"), JSON.stringify(snapshot.frames)).toBe(0);
  expectAtMostOneDrawPerCanvasPerFrame(snapshot);
}

test.describe("canvas scheduling", () => {
  test("scatter composition animation draws each canvas at most once per frame", async ({
    page,
  }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop interaction");
    await waitForCanvasAppReady(page);
    await hoverRenderedScatterPoint(page);
    await resetCanvasFrameProbe(page);

    await page.locator("#scatter-canvas").click();
    await expect
      .poll(() => page.evaluate(() => window.__test?.isCompositionTransitionActive))
      .toBe(true);
    await waitForCanvasLoopToPark(page);

    const snapshot = await snapshotCanvasFrames(page);
    expect(drawCount(snapshot, "curve")).toBeGreaterThan(0);
    expect(drawCount(snapshot, "scatter")).toBeGreaterThan(0);
    expectAtMostOneDrawPerCanvasPerFrame(snapshot);
  });

  test("slider preview and return redraw only the curve", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop hover interaction");
    await waitForCanvasAppReady(page);
    const slider = page.locator("#sliders input[type=range]").nth(1);
    const box = await slider.boundingBox();
    if (!box) throw new Error("slider has no bounding box");
    await resetCanvasFrameProbe(page);

    await page.mouse.move(box.x + box.width * 0.75, box.y + box.height / 2);
    await expect.poll(() => page.evaluate(() => window.__test?.previewSource)).toBe("slider");
    await expect.poll(() => page.evaluate(() => {
      const t = window.__test;
      return t.displayPreviewComp.some(
        (value: number, idx: number) => Math.abs(value - t.currentComposition[idx]) > 1e-6,
      );
    })).toBe(true);
    await waitForCanvasLoopToPark(page);
    await expectCurveOnly(page);

    await resetCanvasFrameProbe(page);
    await page.mouse.move(0, 0);
    await waitForCanvasLoopToPark(page);
    await expectCurveOnly(page);
  });

  test("Material Source hover and focus preview redraw only the curve", async ({
    page,
  }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop hover interaction");
    await waitForCanvasAppReady(page);
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await resetCanvasFrameProbe(page);

    await inactive.hover();
    await inactive.focus();
    await expect.poll(() => page.evaluate(() => window.__test?.previewSource)).toBe("class");
    await waitForCanvasLoopToPark(page);
    await expectCurveOnly(page);

    await resetCanvasFrameProbe(page);
    await inactive.blur();
    await page.mouse.move(0, 0);
    await waitForCanvasLoopToPark(page);
    await expectCurveOnly(page);
  });

  test("direct slider commit redraws scatter only at the boundary", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop interaction");
    await waitForCanvasAppReady(page);
    const slider = page.locator("#sliders input[type=range]").first();
    await resetCanvasFrameProbe(page);

    await slider.focus();
    await page.keyboard.press("ArrowRight");
    await expect
      .poll(() => page.evaluate(() => window.__test?.isCurveTransitionActive))
      .toBe(true);
    await waitForCanvasLoopToPark(page);

    const snapshot = await snapshotCanvasFrames(page);
    expect(drawCount(snapshot, "scatter"), JSON.stringify(snapshot.frames)).toBe(1);
    expect(drawCount(snapshot, "curve")).toBeGreaterThan(1);
    expectAtMostOneDrawPerCanvasPerFrame(snapshot);
  });

  test("direct Material Source commit redraws scatter only at the boundary", async ({
    page,
  }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop interaction");
    await waitForCanvasAppReady(page);
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await resetCanvasFrameProbe(page);

    await inactive.click();
    await expect
      .poll(() => page.evaluate(() => window.__test?.isCurveTransitionActive))
      .toBe(true);
    await waitForCanvasLoopToPark(page);

    const snapshot = await snapshotCanvasFrames(page);
    expect(drawCount(snapshot, "scatter"), JSON.stringify(snapshot.frames)).toBe(1);
    expect(drawCount(snapshot, "curve")).toBeGreaterThan(1);
    expectAtMostOneDrawPerCanvasPerFrame(snapshot);
  });

  for (const selector of ["#axis-selector-x", "#axis-selector-y"]) {
    test(`${selector} transition redraws only scatter`, async ({ page }, testInfo) => {
      test.skip(!testInfo.project.name.startsWith("desktop"), "desktop scatter controls");
      await waitForCanvasAppReady(page);
      await resetCanvasFrameProbe(page);

      await page.locator(`${selector} .axis-option-face:has(input:not(:checked))`).click();
      await expect.poll(() => page.evaluate(() => window.__test?.isAnimLoopActive)).toBe(true);
      await waitForCanvasLoopToPark(page);

      await expectScatterOnly(page);
    });
  }

  test("curve observation hover redraws only the curve", async ({ page }, testInfo) => {
    test.setTimeout(60_000);
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop hover interaction");
    await waitForCanvasAppReady(page);
    await resetCanvasFrameProbe(page);

    await hoverRenderedCurveObservation(page);
    await page.mouse.move(0, 0);
    await waitForCanvasLoopToPark(page);

    await expectCurveOnly(page);
  });

  test("scatter hover redraws both canvases at most once per frame", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop hover interaction");
    await waitForCanvasAppReady(page);
    await resetCanvasFrameProbe(page);

    await hoverRenderedScatterPoint(page);
    await page.mouse.move(0, 0);
    await waitForCanvasLoopToPark(page);

    const snapshot = await snapshotCanvasFrames(page);
    expect(drawCount(snapshot, "curve")).toBeGreaterThan(0);
    expect(drawCount(snapshot, "scatter")).toBeGreaterThan(0);
    expectAtMostOneDrawPerCanvasPerFrame(snapshot);
  });

  test("theme and unit changes redraw both canvases", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop controls");
    await waitForCanvasAppReady(page);

    for (const selector of ["#theme-toggle", "#unit-toggle"]) {
      await resetCanvasFrameProbe(page);
      await page.locator(selector).click();
      await expect
        .poll(async () => {
          const snapshot = await snapshotCanvasFrames(page);
          return drawCount(snapshot, "curve") > 0 && drawCount(snapshot, "scatter") > 0;
        })
        .toBe(true);
      await waitForCanvasLoopToPark(page);
      expectAtMostOneDrawPerCanvasPerFrame(await snapshotCanvasFrames(page));
    }
  });

  test("resizing a chart container coordinates both desktop canvases", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop layout");
    await waitForCanvasAppReady(page);
    await resetCanvasFrameProbe(page);

    await page.locator(".scatter-content").evaluate((element) => {
      (element as HTMLElement).style.inlineSize = "calc(100% - 24px)";
    });
    await expect
      .poll(async () => {
        const snapshot = await snapshotCanvasFrames(page);
        return drawCount(snapshot, "curve") > 0 && drawCount(snapshot, "scatter") > 0;
      })
      .toBe(true);
    await waitForCanvasLoopToPark(page);

    const sizes = await page.evaluate(() => {
      const read = (selector: string) => {
        const rect = document.querySelector(selector)!.getBoundingClientRect();
        return { width: rect.width, height: rect.height };
      };
      return { curve: read("#curve-canvas"), scatter: read("#scatter-canvas") };
    });
    expect(sizes.scatter).toEqual(sizes.curve);
    expectAtMostOneDrawPerCanvasPerFrame(await snapshotCanvasFrames(page));
  });

  test("filter structural motion invalidates scatter only at transaction boundaries", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop filter controls");
    await waitForCanvasAppReady(page);
    await resetCanvasFrameProbe(page);
    await page.evaluate(() => {
      document.body.dataset.layoutChangeCount = "0";
      document.addEventListener("dashboard-layout-change", () => {
        document.body.dataset.layoutChangeCount = String(
          Number(document.body.dataset.layoutChangeCount ?? 0) + 1,
        );
      });
    });

    await page.locator("#filter-add").click();
    await expect.poll(() => page.locator(".filter-row-wrapper").first().evaluate(
      (element) => element.getAnimations().length,
    )).toBe(0);
    await waitForCanvasLoopToPark(page);

    const snapshot = await snapshotCanvasFrames(page);
    expect(await page.locator("body").getAttribute("data-layout-change-count")).toBe("2");
    expect(drawCount(snapshot, "curve"), JSON.stringify(snapshot.frames)).toBe(0);
    expect(drawCount(snapshot, "scatter")).toBeLessThanOrEqual(2);
    expect(snapshot.events.filter((event) => event.phase === "sync")).toEqual([]);
    if (snapshot.frames.length > 0) expectAtMostOneDrawPerCanvasPerFrame(snapshot);
  });

  test("filters redraw only scatter", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop filter controls");
    await waitForCanvasAppReady(page);
    await page.locator("#filter-add").click();
    await resetCanvasFrameProbe(page);

    const minimum = page.locator(".filter-min").first();
    await minimum.fill("1");
    await minimum.dispatchEvent("change");
    await expect
      .poll(async () => drawCount(await snapshotCanvasFrames(page), "scatter"))
      .toBeGreaterThan(0);

    await expectScatterOnly(page);
  });

  test("mobile scatter reveal invalidates only scatter", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile-only");
    await waitForCanvasAppReady(page);
    await page.locator("#mobile-show-sliders").click();
    await expect(page.locator(".mobile-sliders-view")).toBeVisible();
    await resetCanvasFrameProbe(page);

    await page.locator("#mobile-show-scatter").click();
    await expect(page.locator(".scatter-content")).toBeVisible();
    await expect
      .poll(async () => drawCount(await snapshotCanvasFrames(page), "scatter"))
      .toBeGreaterThan(0);

    await expectScatterOnly(page);
  });
});
