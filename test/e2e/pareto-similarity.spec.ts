import { test, expect, Page } from "@playwright/test";
import {
  installCanvasFrameProbe,
  resetCanvasFrameProbe,
  snapshotCanvasFrames,
  drawCount,
  hoverRenderedScatterPoint,
  waitForCanvasLoopToPark,
} from "./canvas-frame-probe";

/**
 * Recipe-similarity encoding on the Pareto scatter plot.
 *
 * Each catalog point's FILL is faded by how unlike the currently selected
 * composition its recipe is; its OUTLINE is drawn in the point's own hue at full
 * opacity so position and contrast never degrade. Hue and radius remain the
 * Pareto encoding.
 *
 * Assertions are made against `__test.pointRenderStyles` wherever possible.
 * Pixel comparison is used only for the reversibility check, and only under
 * reduced motion — the selected-composition ring pulses off `Date.now()`
 * (ui.mjs), so the scatter canvas is otherwise time-dependent and no two draws
 * are ever byte-identical.
 */

// The selected-composition ring pulses off Date.now() (ui.mjs), so the scatter
// canvas is time-dependent unless reduced motion pins the pulse at mid-phase.
// `_reduceMotion` is captured when ui.mjs is evaluated, so the emulation has to
// be in place before goto(). `test.use({ reducedMotion })` does not reach the
// page here, so each navigation calls emulateMedia explicitly.
async function gotoWithReducedMotion(page: Page) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/?test=1");
}

/** Canvas normalizes the legacy outline colour to this exact spacing. */
const WHITE_OUTLINE = "rgba(255, 255, 255, 0.7)";

async function ready(page: Page) {
  await gotoWithReducedMotion(page);
  await page.waitForFunction(
    () => (window as any).__test?.modelReady === true,
    null,
    { timeout: 30000 },
  );
}

const scatter = (page: Page) => page.locator("canvas#scatter-canvas");
const toggle = (page: Page) => page.locator("#toggle-similarity");

/** Styles of the points that are not filtered out. */
function visibleStyles(page: Page) {
  return page.evaluate(() =>
    ((window as any).__test.pointRenderStyles || []).filter(
      (s: any) => s && s.kind !== "filtered",
    ),
  );
}

test.describe("Pareto plot recipe-similarity encoding", () => {
  test.beforeEach(async ({}, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "scatter interaction is desktop");
  });

  // --- the control -----------------------------------------------------

  test("toggle defaults to on and reports state via aria-pressed", async ({ page }) => {
    await ready(page);
    await expect(toggle(page)).toHaveAttribute("aria-pressed", "true");
    await expect(toggle(page)).toHaveText("on");
    await toggle(page).click();
    await expect(toggle(page)).toHaveAttribute("aria-pressed", "false");
    await expect(toggle(page)).toHaveText("off");
  });

  test("toggle is operable from the keyboard", async ({ page }) => {
    await ready(page);
    await toggle(page).focus();
    await expect(toggle(page)).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(toggle(page)).toHaveAttribute("aria-pressed", "false");
    await page.keyboard.press("Space");
    await expect(toggle(page)).toHaveAttribute("aria-pressed", "true");
  });

  // --- the rendering contract ------------------------------------------

  test("on state fades distant mixes and gives the outline the point hue", async ({ page }) => {
    await ready(page);
    const styles = await visibleStyles(page);
    expect(styles.length).toBeGreaterThan(0);
    expect(
      styles.some((s: any) => s.fillAlpha < 0.5),
      "some mixes must render faded",
    ).toBe(true);
    expect(
      styles.some((s: any) => s.fillAlpha > 0.95),
      "the selected mix's neighbourhood must render solid",
    ).toBe(true);
    expect(
      styles.every((s: any) => s.stroke !== WHITE_OUTLINE),
      "the outline must carry the point hue, or hollow points vanish on light backgrounds",
    ).toBe(true);
    expect(await page.evaluate(() => (window as any).__test.scatterLegend)).toBe(
      "solid = similar recipe",
    );
  });

  test("off state restores the previous rendering contract exactly", async ({ page }) => {
    await ready(page);
    await toggle(page).click();
    await expect
      .poll(async () => {
        const styles = await visibleStyles(page);
        return (
          styles.length > 0 &&
          styles.every((s: any) => s.fillAlpha === 1 && s.stroke === WHITE_OUTLINE)
        );
      })
      .toBe(true);
    expect(await page.evaluate(() => (window as any).__test.scatterLegend)).toBeNull();
    expect(await page.evaluate(() => (window as any).__test.similarities)).toBeNull();
  });

  test("toggling is pixel-reversible and on differs from off", async ({ page }) => {
    await ready(page);

    // The redraw is scheduled through requestAnimationFrame, so a screenshot
    // taken straight after the click can still capture the previous mode.
    async function flip() {
      await toggle(page).click();
      await waitForCanvasLoopToPark(page);
      return scatter(page).screenshot();
    }

    const off1 = await flip();
    const on1 = await flip();
    const off2 = await flip();
    const on2 = await flip();

    expect(Buffer.compare(off1, off2), "off rendering must be reproducible").toBe(0);
    expect(Buffer.compare(on1, on2), "on rendering must be reproducible").toBe(0);
    expect(Buffer.compare(off1, on1), "on and off must look different").not.toBe(0);
  });

  // --- coupling to the composition --------------------------------------

  test("similarity tracks the selected composition and peaks at 1", async ({ page }) => {
    await ready(page);
    const before = await page.evaluate(() => (window as any).__test.similarities);
    expect(before.length).toBeGreaterThan(0);
    expect(Math.max(...before)).toBeGreaterThan(0.99);
    expect(Math.min(...before)).toBeGreaterThanOrEqual(0);

    await page
      .locator('#sliders input[type="range"]')
      .first()
      .evaluate((el: HTMLInputElement) => {
        el.value = String(Number(el.max) * 0.85);
        el.dispatchEvent(new Event("input", { bubbles: true }));
      });

    await expect
      .poll(async () =>
        JSON.stringify(await page.evaluate(() => (window as any).__test.similarities)),
      )
      .not.toBe(JSON.stringify(before));
  });

  test("clicking a point makes that exact mix maximally similar", async ({ page }) => {
    await ready(page);
    await hoverRenderedScatterPoint(page);
    const idx = await page.evaluate(() => (window as any).__test.hoveredPointIdx);
    expect(idx, "hover helper must land on a point").not.toBeNull();
    await page.mouse.down();
    await page.mouse.up();
    await expect
      .poll(async () =>
        page.evaluate((i) => (window as any).__test.similarities[i], idx),
      )
      .toBeGreaterThan(0.999);
  });

  test("hovering a faded point restores it to full emphasis", async ({ page }) => {
    await ready(page);
    await hoverRenderedScatterPoint(page);
    await expect
      .poll(async () =>
        page.evaluate(() => {
          const t = (window as any).__test;
          if (t.hoveredPointIdx === null) return null;
          return t.pointRenderStyles[t.hoveredPointIdx].fillAlpha;
        }),
      )
      .toBe(1);
  });

  // --- interaction with the other encodings ------------------------------

  test("points are drawn least-similar first", async ({ page }) => {
    await ready(page);
    const { order, sims } = await page.evaluate(() => ({
      order: (window as any).__test.similarityDrawOrder,
      sims: (window as any).__test.similarities,
    }));
    expect(order.length).toBeGreaterThan(1);
    for (let k = 1; k < order.length; k++) {
      expect(
        sims[order[k]],
        "draw order must be ascending in similarity so neighbours land on top",
      ).toBeGreaterThanOrEqual(sims[order[k - 1]]);
    }
  });

  test("filtered-out points keep grey styling and stay out of the reorder", async ({ page }) => {
    await ready(page);
    await page.locator("#filter-add").click();
    // Selecting a column alone filters nothing; a range is what excludes points.
    const row = page.locator("#filter-rows .filter-row").first();
    await row.locator(".filter-col").selectOption("0"); // Cement
    const maxInput = row.locator(".filter-max");
    await expect(maxInput).toBeVisible();
    // The filter inputs listen on "change", not "input".
    await maxInput.fill("200");
    await maxInput.dispatchEvent("change");

    await expect
      .poll(async () =>
        page.evaluate(() => {
          const t = (window as any).__test;
          const styles = t.pointRenderStyles || [];
          const filtered = styles
            .map((s: any, i: number) => (s && s.kind === "filtered" ? i : -1))
            .filter((i: number) => i >= 0);
          // Sentinel rather than `true`, so a filter that excludes nothing fails
          // loudly instead of passing vacuously.
          if (filtered.length === 0) return "no-filtered-points";
          const allGrey = filtered.every((i: number) => styles[i].fillAlpha === 0.3);
          const excluded = filtered.every(
            (i: number) => !t.similarityDrawOrder.includes(i),
          );
          return allGrey && excluded;
        }),
      )
      .toBe(true);
  });

  test("similarity survives axis and curing-day changes", async ({ page }) => {
    await ready(page);
    for (const id of ["#toggle-x", "#toggle-day"]) {
      await page.locator(id).click();
      await expect
        .poll(async () =>
          page.evaluate(() => {
            const s = (window as any).__test.similarities;
            return (
              s !== null &&
              s.length > 0 &&
              s.every((v: number) => Number.isFinite(v) && v >= 0 && v <= 1)
            );
          }),
        )
        .toBe(true);
    }
  });

  test("the encoding works in dark and light themes", async ({ page }) => {
    await ready(page);
    await page.locator("#theme-toggle").click();
    await expect
      .poll(async () => (await visibleStyles(page)).some((s: any) => s.fillAlpha < 0.5))
      .toBe(true);
    const on = await scatter(page).screenshot();
    await toggle(page).click();
    expect(Buffer.compare(on, await scatter(page).screenshot())).not.toBe(0);
  });

  // --- robustness --------------------------------------------------------

  test("the scatter renders un-encoded before the model resolves", async ({ page }) => {
    // Stall the worker so the pre-model window is wide enough to observe.
    await page.route("**/model_init_worker.mjs", async (route) => {
      await new Promise((r) => setTimeout(r, 4000));
      await route.continue();
    });
    await gotoWithReducedMotion(page);
    await page.waitForFunction(() => typeof (window as any).__test !== "undefined");
    await page.locator("#sliders input[type=range]").first().waitFor({ timeout: 15000 });

    const pre = await page.evaluate(() => ({
      modelReady: (window as any).__test.modelReady,
      sims: (window as any).__test.similarities,
    }));
    expect(
      pre.modelReady,
      "test needs the pre-model window; worker resolved too fast",
    ).toBe(false);
    expect(pre.sims, "similarities must be null before the model lands").toBeNull();

    const painted = await scatter(page).evaluate((c: HTMLCanvasElement) => {
      const d = c.getContext("2d")!.getImageData(0, 0, c.width, c.height).data;
      for (let i = 3; i < d.length; i += 400) if (d[i] !== 0) return true;
      return false;
    });
    expect(painted, "scatter must still paint before the model resolves").toBe(true);
  });

  // --- scheduling and cost ------------------------------------------------

  test("the toggle does not draw outside requestAnimationFrame", async ({ page }) => {
    await installCanvasFrameProbe(page);
    await ready(page);
    await resetCanvasFrameProbe(page);
    await toggle(page).click();
    await expect
      .poll(async () => drawCount(await snapshotCanvasFrames(page), "scatter"))
      .toBeGreaterThan(0);
    const snap = await snapshotCanvasFrames(page);
    expect(
      snap.events.filter((event) => event.phase === "sync"),
      "canvas renderers must be owned by requestAnimationFrame",
    ).toHaveLength(0);
  });

  test("enabling similarity does not add scatter draws", async ({ page }) => {
    await installCanvasFrameProbe(page);
    await ready(page);

    async function drawsForTenSliderSteps() {
      await resetCanvasFrameProbe(page);
      const slider = page.locator('#sliders input[type="range"]').first();
      for (let k = 0; k < 10; k++) {
        await slider.evaluate((el: HTMLInputElement, step: number) => {
          el.value = String(Number(el.max) * (0.3 + step * 0.05));
          el.dispatchEvent(new Event("input", { bubbles: true }));
        }, k);
      }
      await expect
        .poll(async () => drawCount(await snapshotCanvasFrames(page), "scatter"))
        .toBeGreaterThan(0);
      return drawCount(await snapshotCanvasFrames(page), "scatter");
    }

    const withOn = await drawsForTenSliderSteps();
    await toggle(page).click(); // off
    const withOff = await drawsForTenSliderSteps();

    expect(withOn, "the probe must observe draws in both modes").toBeGreaterThan(0);
    expect(withOff, "the probe must observe draws in both modes").toBeGreaterThan(0);
    expect(
      withOn,
      "similarity must not increase the scatter draw count",
    ).toBeLessThanOrEqual(withOff + 2);
  });
});
