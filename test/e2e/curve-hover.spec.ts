import { test, expect } from "@playwright/test";

/**
 * Strength-curve continuous hover tooltip + indicator.
 *
 * The curve canvas now responds to pointer-move (and pointer-down on touch)
 * everywhere inside the plot area — not only on the discrete training-data
 * observation dots. The tooltip shows the predicted strength + uncertainty
 * at the cursor's time projection. A vertical dashed guide and a dot at
 * the predicted mean are rendered on the canvas as a visual affordance.
 *
 * Two formats coexist on the same `.tooltip` DOM element:
 *   (1) "Day N: <value> <unit>"           ← cursor on an observation dot
 *   (2) "Day X.X · <mean> <unit> (±<std>)" ← cursor anywhere else
 * (1) takes priority via observation hit-test inside the pointer handler.
 */

const CONTINUOUS_PATTERN = /^Day \d+(?:\.\d)? · [\d,]+(?:\.\d)? (?:psi|MPa) \(±[\d,]+(?:\.\d)?\)$/;
const OBSERVATION_PATTERN = /^Day \d+(?:\.\d)?: [\d,]+(?:\.\d)? (?:psi|MPa)$/;

async function getCanvasBox(page: import("@playwright/test").Page) {
  const canvas = page.locator("canvas#curve-canvas");
  await expect(canvas).toBeVisible();
  // Wait for the GP to draw so observations + curve are rendered before
  // any hover tests fire. The home-loads spec already pins this; reproducing
  // the wait here keeps the test self-contained.
  await expect
    .poll(
      async () =>
        canvas.evaluate((c: HTMLCanvasElement) => {
          const ctx = c.getContext("2d");
          if (!ctx) return false;
          const data = ctx.getImageData(0, 0, c.width, c.height).data;
          for (let i = 3; i < data.length; i += 4) if (data[i] !== 0) return true;
          return false;
        }),
      { timeout: 10_000, message: "curve canvas remained blank" },
    )
    .toBe(true);
  const box = await canvas.boundingBox();
  if (!box) throw new Error("canvas has no bounding box");
  return { canvas, box };
}

test.describe("strength curve hover tooltip", () => {
  test("desktop: hovering inside the plot shows continuous tooltip with day + strength + uncertainty", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop hover-driven UX");
    await page.goto("/");
    const { canvas } = await getCanvasBox(page);
    const tooltip = page.locator(".tooltip");

    // Hover at horizontal midpoint, vertical midpoint
    await canvas.hover({ position: { x: 200, y: 120 } });
    await expect(tooltip).toBeVisible();
    const text = (await tooltip.textContent())?.trim() ?? "";
    expect(text, `tooltip text = "${text}"`).toMatch(CONTINUOUS_PATTERN);
  });

  test("desktop: tooltip's day value increases as cursor moves right", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop hover-driven UX");
    await page.goto("/");
    const { canvas, box } = await getCanvasBox(page);
    const tooltip = page.locator(".tooltip");

    async function dayAt(xPos: number): Promise<number> {
      await canvas.hover({ position: { x: xPos, y: box.height * 0.5 } });
      await expect(tooltip).toBeVisible();
      const text = (await tooltip.textContent()) ?? "";
      const m = text.match(/Day\s+(\d+(?:\.\d+)?)/);
      if (!m) throw new Error(`no day in tooltip: "${text}"`);
      return parseFloat(m[1]);
    }

    const dayLeft = await dayAt(box.width * 0.25);
    const dayRight = await dayAt(box.width * 0.75);
    expect(
      dayRight,
      `day at right (${dayRight}) should be > day at left (${dayLeft})`,
    ).toBeGreaterThan(dayLeft);
  });

  test("desktop: pointer-leave hides the tooltip", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop hover-driven UX");
    await page.goto("/");
    const { canvas, box } = await getCanvasBox(page);
    const tooltip = page.locator(".tooltip");

    await canvas.hover({ position: { x: box.width * 0.5, y: box.height * 0.5 } });
    await expect(tooltip).toBeVisible();
    // Move the cursor far off-canvas to fire pointerleave.
    await page.mouse.move(0, 0);
    await expect(tooltip).toBeHidden();
  });

  test("desktop: unit toggle flips tooltip strength format (MPa ↔ psi)", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop hover-driven UX");
    await page.goto("/");
    const { canvas, box } = await getCanvasBox(page);
    const tooltip = page.locator(".tooltip");

    await canvas.hover({ position: { x: box.width * 0.5, y: box.height * 0.5 } });
    await expect(tooltip).toBeVisible();
    const before = (await tooltip.textContent()) ?? "";
    const beforeUnit = /psi/.test(before) ? "psi" : /MPa/.test(before) ? "MPa" : null;
    expect(beforeUnit, `tooltip "${before}" should mention psi or MPa`).not.toBeNull();

    // Toggle units via the same custom event the toolbar button dispatches.
    await page.evaluate(() => document.dispatchEvent(new CustomEvent("toggle-units")));
    // Re-hover to refresh the tooltip in the new unit context (otherwise it
    // shows pre-toggle text until the next pointermove).
    await page.mouse.move(0, 0);
    await canvas.hover({ position: { x: box.width * 0.5, y: box.height * 0.5 } });
    await expect(tooltip).toBeVisible();
    const after = (await tooltip.textContent()) ?? "";
    const afterUnit = beforeUnit === "psi" ? "MPa" : "psi";
    expect(after, `tooltip "${after}" should now mention ${afterUnit}`).toContain(afterUnit);
  });

  test("desktop: hovering an observation dot shows observation format (not continuous)", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop hover-driven UX");
    await page.goto("/");
    const { canvas } = await getCanvasBox(page);
    const tooltip = page.locator(".tooltip");

    // Click a Pareto-optimal scatter point to load a training mix, ensuring
    // observation dots are present on the strength curve canvas.
    const scatter = page.locator("canvas#scatter-canvas");
    const scatterBox = await scatter.boundingBox();
    if (!scatterBox) throw new Error("scatter canvas has no bounding box");
    await scatter.click({ position: { x: scatterBox.width * 0.3, y: scatterBox.height * 0.3 } });
    await page.waitForTimeout(800); // animateToComposition + draw

    // Find a real observation dot via the curveObsPositions cache exposed
    // through window.__test? No — we don't expose it. Instead, walk pixel
    // candidates and check tooltip text. Try a coarse grid until we hit
    // observation format.
    let hitText = "";
    outer: for (let xPct = 0.1; xPct <= 0.95; xPct += 0.05) {
      const cb = await canvas.boundingBox();
      if (!cb) break;
      for (let yPct = 0.2; yPct <= 0.9; yPct += 0.05) {
        await canvas.hover({ position: { x: cb.width * xPct, y: cb.height * yPct } });
        const text = ((await tooltip.textContent()) ?? "").trim();
        if (OBSERVATION_PATTERN.test(text)) {
          hitText = text;
          break outer;
        }
      }
    }
    expect(
      hitText,
      "expected to find an observation hover (format 'Day N: ...') somewhere on the curve",
    ).toMatch(OBSERVATION_PATTERN);
    // Observation tooltips do not contain '·' (continuous format does)
    expect(hitText).not.toContain("·");
  });

  test("mobile: tap on the curve canvas shows the tooltip", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "mobile", "touch-only behavior");
    await page.goto("/");
    const { canvas, box } = await getCanvasBox(page);
    const tooltip = page.locator(".tooltip");

    // Use the touchscreen API so a real pointerdown event fires (not a
    // synthesized mouse event). Tap roughly mid-canvas.
    const targetX = box.x + box.width * 0.5;
    const targetY = box.y + box.height * 0.5;
    await page.touchscreen.tap(targetX, targetY);

    // The tooltip must become visible after the tap. Allow a moment for
    // the pointer event handler to update the DOM.
    await expect(tooltip).toBeVisible({ timeout: 1000 });
    const text = (await tooltip.textContent())?.trim() ?? "";
    expect(text, `mobile tooltip text = "${text}"`).toMatch(/Day \d+(?:\.\d)?/);
  });
});
