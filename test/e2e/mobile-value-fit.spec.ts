import { test, expect } from "@playwright/test";
import { waitForCanvasLoopToPark } from "./canvas-frame-probe";

/**
 * Mobile-only overflow regression pin. A previous iteration of the slider
 * compaction work cut off the value display at the right panel border,
 * especially with fractional values like `100.3` and worst-case imperial
 * values like `1180.6`. These tests ensure that can never silently regress.
 *
 * Strategy: assert two invariants for every `.slider-value` in the mobile
 * sliders view, across multiple value/unit combinations:
 *   1. value.right <= panel.right - 1  (no panel overflow)
 *   2. value.scrollWidth <= value.clientWidth + 1  (no input-internal clipping)
 */
const ALLOWANCE_PX = 1;

async function gotoSlidersView(page: import("@playwright/test").Page) {
  await page.goto("/?test=1");
  await page.locator("#mobile-show-sliders").click();
  await expect(page.locator(".mobile-sliders-view")).toBeVisible({ timeout: 2000 });
  await expect(page.locator(".mobile-sliders-view .slider-group").first()).toBeVisible();
}

async function assertNoOverflow(page: import("@playwright/test").Page, label: string) {
  const verdict = await page.evaluate((allowance) => {
    const view = document.querySelector(".mobile-sliders-view") as HTMLElement | null;
    if (!view) return { error: "no .mobile-sliders-view" };
    // Use the chart-panel containing the scatter canvas as the "panel".
    // That's the unified mobile panel that hosts the sliders.
    let panel: HTMLElement | null = null;
    for (const cp of Array.from(document.querySelectorAll(".chart-panel"))) {
      if (cp.querySelector("#scatter-canvas")) { panel = cp as HTMLElement; break; }
    }
    if (!panel) return { error: "no scatter chart-panel" };
    const panelRight = panel.getBoundingClientRect().right;
    const offenders: string[] = [];
    const inputs = Array.from(view.querySelectorAll<HTMLElement>(".slider-value"));
    for (const el of inputs) {
      // Skip elements that are not visible (e.g. hidden Material Source value)
      if (el.offsetParent === null) continue;
      const r = el.getBoundingClientRect();
      if (r.right > panelRight - allowance) {
        offenders.push(`right=${r.right} > panel.right-${allowance}=${panelRight - allowance}`);
      }
      if (el.scrollWidth > el.clientWidth + allowance) {
        offenders.push(`scrollWidth=${el.scrollWidth} > clientWidth+${allowance}=${el.clientWidth + allowance}`);
      }
    }
    const docOverflow = document.documentElement.scrollWidth > document.documentElement.clientWidth + allowance;
    return { offenders, docOverflow, count: inputs.length };
  }, ALLOWANCE_PX);
  expect((verdict as any).error, "panel/view should be present").toBeFalsy();
  expect(
    (verdict as any).offenders,
    `[${label}] expected no value overflow; got ${(verdict as any).offenders?.length ?? 0} offenders: ${(verdict as any).offenders?.join(" | ")}`,
  ).toEqual([]);
  expect(
    (verdict as any).docOverflow,
    `[${label}] document should not introduce horizontal scroll`,
  ).toBe(false);
}

test.describe("mobile slider value never overflows the panel", () => {
  test.beforeEach(async ({}, testInfo) => {
    test.skip(testInfo.project.name !== "mobile", "mobile-only");
  });

  test("metric (default) values fit at first paint", async ({ page }) => {
    await gotoSlidersView(page);
    await assertNoOverflow(page, "metric default");
  });

  test("fractional metric values (.3) fit", async ({ page }) => {
    await gotoSlidersView(page);
    // Programmatically set every regular slider to its midpoint, then nudge
    // until the displayed value ends in `.3` (forces 4-character fractional).
    await page.evaluate(() => {
      const sliders = Array.from(
        document.querySelectorAll<HTMLInputElement>(".mobile-sliders-view input[type=range]"),
      );
      for (const s of sliders) {
        const min = parseFloat(s.min);
        const max = parseFloat(s.max);
        // Walk in fine steps until the displayed value rounds to X.3.
        const step = (max - min) / 200;
        const want = (v: number) => {
          // Match the formatter: (v * factor).toFixed(1) ends with ".3"
          const idx = Number(s.dataset.idx);
          const valEl = document.getElementById(`val-${idx}`) as HTMLInputElement | null;
          return valEl ? valEl.value.endsWith(".3") : false;
        };
        let v = min + (max - min) * 0.5;
        for (let k = 0; k < 200; k++) {
          s.value = String(v);
          s.dispatchEvent(new Event("input", { bubbles: true }));
          if (want(v)) break;
          v += step;
          if (v > max) v = min;
        }
      }
    });
    await waitForCanvasLoopToPark(page);
    await assertNoOverflow(page, "fractional metric");
  });

  test("imperial values fit (worst case mass conversion)", async ({ page }) => {
    await gotoSlidersView(page);
    // Toggle to imperial via the same custom event the buttons dispatch
    await page.evaluate(() => {
      document.dispatchEvent(new CustomEvent("toggle-units"));
    });
    await waitForCanvasLoopToPark(page);
    await assertNoOverflow(page, "imperial default");
  });

  test("imperial max-bound values fit", async ({ page }) => {
    await gotoSlidersView(page);
    await page.evaluate(() => {
      document.dispatchEvent(new CustomEvent("toggle-units"));
      const sliders = Array.from(
        document.querySelectorAll<HTMLInputElement>(".mobile-sliders-view input[type=range]"),
      );
      for (const s of sliders) {
        s.value = s.max;
        s.dispatchEvent(new Event("input", { bubbles: true }));
      }
    });
    await waitForCanvasLoopToPark(page);
    await assertNoOverflow(page, "imperial max-bound");
  });

  test("narrow viewport (320px): values still fit at all breakpoints", async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 720 });
    await gotoSlidersView(page);
    await assertNoOverflow(page, "320px metric");

    // Imperial after toggle on the narrow viewport
    await page.evaluate(() => document.dispatchEvent(new CustomEvent("toggle-units")));
    await waitForCanvasLoopToPark(page);
    await assertNoOverflow(page, "320px imperial");

    // Max-bound imperial on narrow viewport
    await page.evaluate(() => {
      const sliders = Array.from(
        document.querySelectorAll<HTMLInputElement>(".mobile-sliders-view input[type=range]"),
      );
      for (const s of sliders) {
        s.value = s.max;
        s.dispatchEvent(new Event("input", { bubbles: true }));
      }
    });
    await waitForCanvasLoopToPark(page);
    await assertNoOverflow(page, "320px imperial max-bound");
  });

  test("narrow viewport (320px): every filter control fits the unified panel", async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 720 });
    await page.goto("/?test=1");
    await page.locator("#filter-add").click();
    await expect(page.locator(".filter-row")).toBeVisible();

    const assertControlsFit = async (selectors: string[]) => {
      const result = await page.evaluate((controlSelectors) => {
        const panel = document.querySelector<HTMLElement>("#tradeoffs-panel")!;
        const panelRect = panel.getBoundingClientRect();
        return {
          horizontalOverflow:
            document.documentElement.scrollWidth - document.documentElement.clientWidth,
          controls: controlSelectors.map((selector) => {
            const element = document.querySelector<HTMLElement>(selector)!;
            const rect = element.getBoundingClientRect();
            return {
              selector,
              left: rect.left,
              right: rect.right,
              width: rect.width,
              height: rect.height,
              panelLeft: panelRect.left,
              panelRight: panelRect.right,
            };
          }),
        };
      }, selectors);

      expect(result.horizontalOverflow).toBeLessThanOrEqual(ALLOWANCE_PX);
      for (const control of result.controls) {
        expect(control.left, `${control.selector} left edge`).toBeGreaterThanOrEqual(
          control.panelLeft - ALLOWANCE_PX,
        );
        expect(control.right, `${control.selector} right edge`).toBeLessThanOrEqual(
          control.panelRight + ALLOWANCE_PX,
        );
        expect(control.width, `${control.selector} width`).toBeGreaterThanOrEqual(24);
        expect(control.height, `${control.selector} height`).toBeGreaterThanOrEqual(24);
      }
    };

    await assertControlsFit([
      "#filter-add",
      "#filter-clear",
      ".filter-col",
      ".filter-min",
      ".filter-max",
      ".filter-remove-btn",
    ]);

    const materialSourceIndex = await page.evaluate(async () => {
      const payload = await (await fetch("model/compositions.json")).json();
      return payload.column_names.indexOf("Material Source");
    });
    await page.locator(".filter-col").selectOption(String(materialSourceIndex));
    await expect(page.locator(".filter-cat-btn").first()).toBeVisible();
    await assertControlsFit([".filter-col", ".filter-cat-btn", ".filter-remove-btn"]);
  });
});
