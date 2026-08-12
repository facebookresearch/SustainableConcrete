import { test, expect } from "@playwright/test";

/**
 * The UI shell renders before the strength GP finishes building in a worker,
 * so there is a window where `strengthParams` is null. Everything that does
 * NOT depend on the model must still work in that window.
 *
 * Regression pin: the model-dependent guard in drawScatter was originally an
 * early `return`, which also skipped the axes and the canvas scale stash
 * (`_pad`/`_xMin`/...). The mousemove handler bails on `!pad`, so hover was
 * dead and clicking a point was a silent no-op until the model loaded.
 */
test.describe("pre-model shell", () => {
  test("scatter keeps its axes and scale state before the model resolves", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "scatter interaction is desktop");

    // Stall the worker so the pre-model window is wide enough to observe.
    await page.route("**/model_init_worker.mjs", async (route) => {
      await new Promise((r) => setTimeout(r, 4000));
      await route.continue();
    });

    await page.goto("/?test=1");
    await page.waitForFunction(() => typeof (window as any).__test !== "undefined");
    await page.locator("#sliders input[type=range]").first().waitFor({ timeout: 15000 });

    const pre = await page.evaluate(() => {
      const c = document.querySelector("canvas#scatter-canvas") as any;
      return {
        modelReady: (window as any).__test.modelReady,
        hasPad: c._pad !== undefined,
        hasScales: c._xMin !== undefined && c._yMax !== undefined,
      };
    });

    expect(pre.modelReady, "test needs the pre-model window; worker resolved too fast").toBe(false);
    expect(pre.hasPad, "canvas._pad must be stashed pre-model or hover breaks").toBe(true);
    expect(pre.hasScales, "canvas scale state must be stashed pre-model").toBe(true);
  });

  test("clicking a scatter point works before the model resolves", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "scatter interaction is desktop");

    // Must outlast the hover sweep below, otherwise the model can resolve
    // mid-sweep and the test silently stops exercising the pre-model window.
    await page.route("**/model_init_worker.mjs", async (route) => {
      await new Promise((r) => setTimeout(r, 30000));
      await route.continue();
    });

    await page.goto("/?test=1");
    await page.waitForFunction(() => typeof (window as any).__test !== "undefined");
    await page.locator("#sliders input[type=range]").first().waitFor({ timeout: 15000 });
    expect(await page.evaluate(() => (window as any).__test.modelReady)).toBe(false);

    const before = await page.evaluate(() =>
      JSON.stringify((window as any).__test.currentComposition));

    // Sweep the canvas to find a point, then click it.
    const box = (await page.locator("canvas#scatter-canvas").boundingBox())!;
    let clicked = false;
    for (let fx = 0.2; fx <= 0.8 && !clicked; fx += 0.05) {
      for (let fy = 0.2; fy <= 0.8 && !clicked; fy += 0.05) {
        await page.mouse.move(box.x + box.width * fx, box.y + box.height * fy);
        if (await page.evaluate(() => (window as any).__test.hoveredPointIdx !== null)) {
          await page.mouse.down();
          await page.mouse.up();
          clicked = true;
        }
      }
    }
    expect(
      await page.evaluate(() => (window as any).__test.modelReady),
      "model resolved during the sweep — this test no longer covers the pre-model window",
    ).toBe(false);
    expect(clicked, "no scatter point was hoverable pre-model — hover is broken").toBe(true);

    await page.waitForTimeout(800);
    const after = await page.evaluate(() =>
      JSON.stringify((window as any).__test.currentComposition));
    expect(after, "clicking a point pre-model must still select it").not.toBe(before);
  });
});
