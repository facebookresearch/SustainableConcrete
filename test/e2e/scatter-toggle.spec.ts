import { expect, test, type Locator, type Page } from "@playwright/test";

import { waitForCanvasLoopToPark } from "./canvas-frame-probe";

async function waitForDashboard(page: Page) {
  await page.goto("/?test=1");
  await expect(page.locator("#scatter-canvas")).toBeVisible({ timeout: 15_000 });
  await page.evaluate(async () => {
    await document.fonts.ready;
    await Promise.all(
      [...document.querySelectorAll(".fade-in-up")].flatMap((element) =>
        element.getAnimations().map((animation) => animation.finished),
      ),
    );
  });
}

async function checkedValue(group: Locator) {
  return group.locator('input[type="radio"]:checked').inputValue();
}

async function optionPositions(group: Locator) {
  return group.locator(".axis-option-face").evaluateAll((faces) =>
    faces.map((face) => {
      const rect = face.getBoundingClientRect();
      const input = face.querySelector<HTMLInputElement>('input[type="radio"]')!;
      return {
        value: input.value,
        checked: input.checked,
        left: rect.left,
        right: rect.right,
        top: rect.top,
        bottom: rect.bottom,
        width: rect.width,
        height: rect.height,
        centerX: rect.left + rect.width / 2,
        centerY: rect.top + rect.height / 2,
      };
    }),
  );
}

type AxisExchangeFrame = {
  time: number;
  active: boolean;
  transitioning: boolean;
  wrapper: { left: number; right: number; centerX: number; top: number; bottom: number };
  faces: Array<{
    top: number;
    bottom: number;
    left: number;
    right: number;
    centerX: number;
    centerY: number;
    transform: string;
    zIndex: string;
  }>;
  pills: Array<{
    top: number;
    bottom: number;
    left: number;
    right: number;
    centerX: number;
    centerY: number;
    transform: string;
    backgroundColor: string;
    color: string;
    opacity: string;
  }>;
  canvas: {
    xMin: number;
    xMax: number;
    yMin: number;
    yMax: number;
  };
};

async function driveEditableCompositionTo(page: Page, edge: "min" | "max") {
  await page.locator('#sliders input[type="range"]').evaluateAll((nodes, requestedEdge) => {
    for (const node of nodes) {
      const slider = node as HTMLInputElement;
      slider.value = requestedEdge === "min" ? slider.min : slider.max;
      slider.dispatchEvent(new Event("input", { bubbles: true }));
      slider.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }, edge);
  await waitForCanvasLoopToPark(page);
}

async function sampleYExchange(page: Page, targetValue: "1" | "28") {
  await waitForCanvasLoopToPark(page);
  return page.evaluate(async (value): Promise<{ initial: AxisExchangeFrame; frames: AxisExchangeFrame[] }> => {
    const group = document.querySelector<HTMLElement>("#axis-selector-y")!;
    const wrapper = document.querySelector<HTMLElement>(".axis-selector-y-wrap")!;
    const faces = [...group.querySelectorAll<HTMLElement>(".axis-option-face")];
    const pills = [...group.querySelectorAll<HTMLElement>(".axis-option-pill")];
    const canvas = document.querySelector<HTMLCanvasElement>("#scatter-canvas")! as HTMLCanvasElement & {
      _xMin: number;
      _xMax: number;
      _yMin: number;
      _yMax: number;
    };
    const rect = (element: Element) => {
      const bounds = element.getBoundingClientRect();
      return {
        top: bounds.top,
        bottom: bounds.bottom,
        left: bounds.left,
        right: bounds.right,
        centerX: bounds.left + bounds.width / 2,
        centerY: bounds.top + bounds.height / 2,
      };
    };
    const snapshot = (time: number): AxisExchangeFrame => {
      const wrapperRect = rect(wrapper);
      return {
        time,
        active: Boolean(window.__test?.isScatterTransitionActive),
        transitionFrom: window.__test?.scatterTransitionFromSnapshot
          ? {
              xMax: window.__test.scatterTransitionFromSnapshot.xMax,
              yMax: window.__test.scatterTransitionFromSnapshot.yMax,
            }
          : null,
        transitioning: group.dataset.transitioning === "true",
        wrapper: wrapperRect,
        faces: faces.map((face) => {
          const style = getComputedStyle(face);
          return { ...rect(face), transform: style.transform, zIndex: style.zIndex };
        }),
        pills: pills.map((pill) => {
          const style = getComputedStyle(pill);
          return {
            ...rect(pill),
            transform: style.transform,
            backgroundColor: style.backgroundColor,
            color: style.color,
            opacity: style.opacity,
          };
        }),
        canvas: {
          xMin: canvas._xMin,
          xMax: canvas._xMax,
          yMin: canvas._yMin,
          yMax: canvas._yMax,
        },
      };
    };

    const initial = snapshot(performance.now());
    group.querySelector<HTMLInputElement>(`input[value="${value}"]`)!.click();
    const frames: AxisExchangeFrame[] = [];
    const deadline = performance.now() + 2_000;
    let observedActive = Boolean(window.__test?.isScatterTransitionActive);
    await new Promise<void>((resolve, reject) => {
      const sample = (time: number) => {
        const frame = snapshot(time);
        frames.push(frame);
        observedActive ||= frame.active;
        const animationsRunning = [...faces, ...pills].some((element) =>
          element.getAnimations().some((animation) => animation.playState === "running"),
        );
        if (observedActive && !frame.active && !frame.transitioning && !animationsRunning) {
          resolve();
        } else if (performance.now() >= deadline) {
          reject(new Error(`Y exchange to ${value} did not settle`));
        } else {
          requestAnimationFrame(sample);
        }
      };
      requestAnimationFrame(sample);
    });
    return { initial, frames };
  }, targetValue);
}

type ScatterRangeFrame = {
  time: number;
  active: boolean;
  transitionFrom: { xMax: number; yMax: number } | null;
  range: { xMin: number; xMax: number; yMin: number; yMax: number };
  plot: { left: number; top: number; right: number; bottom: number; width: number; height: number };
  current: { x: number; y: number | null };
  point: { x: number; y: number };
  catalogMax: { x: number; y: number };
};

async function sampleScatterRangeExchange(
  page: Page,
  selector: string,
): Promise<{ initial: ScatterRangeFrame; frames: ScatterRangeFrame[] }> {
  return page.evaluate(async (targetSelector) => {
    const canvas = document.querySelector<HTMLCanvasElement>("#scatter-canvas")! as HTMLCanvasElement & {
      _xMin: number;
      _xMax: number;
      _yMin: number;
      _yMax: number;
      _plotRect: ScatterRangeFrame["plot"];
    };
    const snapshot = (time: number): ScatterRangeFrame => {
      const displayed = window.__test!.scatterDisplayedSnapshot;
      const plot = { ...canvas._plotRect };
      const current = { ...displayed.current };
      const drawnY = Math.max(0, current.y ?? 0);
      const transitionFrom = window.__test?.scatterTransitionFromSnapshot;
      return {
        time,
        active: Boolean(window.__test?.isScatterTransitionActive),
        transitionFrom: transitionFrom
          ? { xMax: transitionFrom.xMax, yMax: transitionFrom.yMax }
          : null,
        range: {
          xMin: canvas._xMin,
          xMax: canvas._xMax,
          yMin: canvas._yMin,
          yMax: canvas._yMax,
        },
        plot,
        current,
        point: {
          x: plot.left + ((current.x - canvas._xMin) / (canvas._xMax - canvas._xMin)) * plot.width,
          y: plot.bottom - ((drawnY - canvas._yMin) / (canvas._yMax - canvas._yMin)) * plot.height,
        },
        catalogMax: {
          x: Math.max(...displayed.x),
          y: Math.max(...displayed.y.map((value: number) => Math.max(0, value))),
        },
      };
    };

    const initial = snapshot(performance.now());
    document.querySelector<HTMLInputElement>(targetSelector)!.click();
    const frames: ScatterRangeFrame[] = [];
    const deadline = performance.now() + 2_000;
    let observedActive = false;
    await new Promise<void>((resolve, reject) => {
      const sample = (time: number) => {
        const frame = snapshot(time);
        frames.push(frame);
        observedActive ||= frame.active;
        if (observedActive && !frame.active) resolve();
        else if (performance.now() >= deadline) reject(new Error(`Scatter exchange ${targetSelector} did not settle`));
        else requestAnimationFrame(sample);
      };
      requestAnimationFrame(sample);
    });
    return { initial, frames };
  }, selector);
}

function rangeDeltaInPixels(
  before: ScatterRangeFrame,
  after: ScatterRangeFrame,
  axis: "x" | "y",
) {
  const maxKey = axis === "x" ? "xMax" : "yMax";
  const span = before.range[maxKey] - before.range[axis === "x" ? "xMin" : "yMin"];
  const pixels = axis === "x" ? before.plot.width : before.plot.height;
  return span > 0 ? Math.abs(after.range[maxKey] - before.range[maxKey]) / span * pixels : 0;
}

function assertRangeExchangeContinuous(
  exchange: Awaited<ReturnType<typeof sampleScatterRangeExchange>>,
  label: string,
) {
  const firstActive = exchange.frames.find((frame) => frame.active);
  expect(firstActive, `${label} must capture an active frame`).toBeDefined();
  const settledIndex = exchange.frames.findIndex((frame, index) =>
    index > 0 && exchange.frames[index - 1].active && !frame.active,
  );
  expect(settledIndex, `${label} must capture settlement`).toBeGreaterThan(0);
  const lastActive = exchange.frames[settledIndex - 1];
  const firstSettled = exchange.frames[settledIndex];

  for (const frame of [exchange.initial, ...exchange.frames]) {
    expect(frame.current.x, `${label} current X must stay inside the domain`).toBeLessThanOrEqual(frame.range.xMax + 1e-6);
    expect(Math.max(0, frame.current.y ?? 0), `${label} current Y must stay inside the domain`).toBeLessThanOrEqual(frame.range.yMax + 1e-6);
    expect(frame.point.x, `${label} current point must stay inside the plot horizontally`).toBeGreaterThanOrEqual(frame.plot.left - 1);
    expect(frame.point.x, `${label} current point must stay inside the plot horizontally`).toBeLessThanOrEqual(frame.plot.right + 1);
    expect(frame.point.y, `${label} current point must stay inside the plot vertically`).toBeGreaterThanOrEqual(frame.plot.top - 1);
    expect(frame.point.y, `${label} current point must stay inside the plot vertically`).toBeLessThanOrEqual(frame.plot.bottom + 1);
  }

  expect(firstActive!.transitionFrom, `${label} must expose its captured start range`).toBeDefined();
  expect(firstActive!.transitionFrom, `${label} must expose its captured start range`).not.toBeNull();
  const transitionSource = {
    ...exchange.initial,
    range: {
      ...exchange.initial.range,
      xMax: firstActive!.transitionFrom!.xMax,
      yMax: firstActive!.transitionFrom!.yMax,
    },
  };
  expect(
    rangeDeltaInPixels(exchange.initial, transitionSource, "x"),
    `${label} X transition source must match the trigger-frame range on screen`,
  ).toBeLessThanOrEqual(1);
  expect(
    rangeDeltaInPixels(exchange.initial, transitionSource, "y"),
    `${label} Y transition source must match the trigger-frame range on screen`,
  ).toBeLessThanOrEqual(1);
  expect(rangeDeltaInPixels(lastActive, firstSettled, "x"), `${label} X range must not snap at settlement`).toBeLessThanOrEqual(1);
  expect(rangeDeltaInPixels(lastActive, firstSettled, "y"), `${label} Y range must not snap at settlement`).toBeLessThanOrEqual(1);
  expect(Math.hypot(firstSettled.point.x - lastActive.point.x, firstSettled.point.y - lastActive.point.y), `${label} current point must not jump at settlement`).toBeLessThanOrEqual(1);
}

function assertStrictlyVerticalYExchange(
  exchange: Awaited<ReturnType<typeof sampleYExchange>>,
  label: string,
) {
  const { initial, frames } = exchange;
  expect(frames.some((frame) => frame.active), `${label} must observe active canvas frames`).toBe(true);
  const settledIndex = frames.findIndex((frame, index) =>
    index > 0 && frames[index - 1].active && !frame.active,
  );
  expect(settledIndex, `${label} must capture the first settled frame`).toBeGreaterThan(0);
  const lastActive = frames[settledIndex - 1];
  const firstSettled = frames[settledIndex];

  const all = [initial, ...frames];
  for (const [name, getRect] of [
    ["wrapper", (frame: AxisExchangeFrame) => frame.wrapper],
    ["face 0", (frame: AxisExchangeFrame) => frame.faces[0]],
    ["face 1", (frame: AxisExchangeFrame) => frame.faces[1]],
    ["pill 0", (frame: AxisExchangeFrame) => frame.pills[0]],
    ["pill 1", (frame: AxisExchangeFrame) => frame.pills[1]],
  ] as const) {
    for (const edge of ["left", "right", "centerX"] as const) {
      const values = all.map((frame) => getRect(frame)[edge]);
      expect(
        Math.max(...values) - Math.min(...values),
        `${label} ${name} ${edge} must not drift horizontally`,
      ).toBeLessThanOrEqual(0.5);
    }
  }

  for (const faceIndex of [0, 1]) {
    const positions = [initial, ...frames].map((frame) => frame.faces[faceIndex].top);
    const direction = Math.sign(positions.at(-1)! - positions[0]);
    expect(direction, `${label} face ${faceIndex} must change vertical slots`).not.toBe(0);
    for (let index = 1; index < positions.length; index += 1) {
      expect(
        direction * (positions[index] - positions[index - 1]),
        `${label} face ${faceIndex} must remain vertically monotonic`,
      ).toBeGreaterThanOrEqual(-0.25);
    }
    expect(
      Math.abs(positions.at(-1)! - initial.faces[1 - faceIndex].top),
      `${label} face ${faceIndex} must finish in the opposite slot`,
    ).toBeLessThanOrEqual(1);
  }

  for (const edge of ["left", "right", "centerX", "top", "bottom"] as const) {
    expect(
      Math.abs(lastActive.wrapper[edge] - firstSettled.wrapper[edge]),
      `${label} wrapper ${edge} must remain continuous at settlement`,
    ).toBeLessThanOrEqual(0.5);
  }
}

async function assertGroupSemantics(group: Locator, legend: RegExp) {
  await expect(group).toHaveRole("radiogroup");
  await expect(group.locator("legend")).toHaveText(legend);
  const radios = group.locator('input[type="radio"]');
  await expect(radios).toHaveCount(2);
  await expect(group.locator('input[type="radio"]:checked')).toHaveCount(1);
  expect(await radios.evaluateAll((items) => items.map((item) => item.id))).toEqual(
    expect.arrayContaining([expect.any(String), expect.any(String)]),
  );
}

test.describe("scatter plot two-option axis selectors", () => {
  test.beforeEach(async ({ page }) => waitForDashboard(page));

  test("both axis objectives remain visible with native radio semantics", async ({ page }) => {
    const x = page.locator("#axis-selector-x");
    const y = page.locator("#axis-selector-y");
    await assertGroupSemantics(x, /X-axis objective/i);
    await assertGroupSemantics(y, /Y-axis objective/i);
    await expect(x.getByText("GWP", { exact: true })).toBeVisible();
    await expect(x.getByText("Cost", { exact: true })).toBeVisible();
    await expect(y.getByText("28-day Strength", { exact: true })).toBeVisible();
    await expect(y.getByText("1-day Strength", { exact: true })).toBeVisible();
    await expect(y.getByRole("radio", { name: "28-day strength", exact: true })).toBeVisible();
    await expect(y.getByRole("radio", { name: "1-day strength", exact: true })).toBeVisible();
    await expect(page.locator("#toggle-x, #toggle-day")).toHaveCount(0);
  });

  test("stable X faces exchange horizontally while native arrows retain focus", async ({ page }) => {
    const x = page.locator("#axis-selector-x");
    const radios = x.locator('input[type="radio"]');
    const idsBefore = await radios.evaluateAll((items) => items.map((item) => item.id));
    const before = await optionPositions(x);
    const checkedBefore = await checkedValue(x);
    const selectedBefore = before.find((position) => position.checked)!;
    const alternateBefore = before.find((position) => !position.checked)!;
    expect(Math.abs(selectedBefore.centerY - alternateBefore.centerY)).toBeLessThanOrEqual(1);
    expect(selectedBefore.centerX, "selected X option must occupy the leading slot").toBeLessThan(
      alternateBefore.centerX,
    );

    await x.locator(`input[value="${checkedBefore}"]`).focus();
    await page.keyboard.press("ArrowRight");
    await expect(x.locator('input[type="radio"]:checked')).not.toHaveValue(checkedBefore);
    await expect(x.locator('input[type="radio"]:checked')).toBeFocused();
    await expect.poll(() => x.locator(".axis-option-face").evaluateAll((faces) =>
      faces.reduce((count, face) => count + face.getAnimations().length, 0),
    )).toBe(0);

    const after = await optionPositions(x);
    const selectedAfter = after.find((position) => position.checked)!;
    const alternateAfter = after.find((position) => !position.checked)!;
    expect(Math.abs(selectedAfter.centerY - alternateAfter.centerY)).toBeLessThanOrEqual(1);
    expect(selectedAfter.centerX, "newly selected X option must settle in the leading slot").toBeLessThan(
      alternateAfter.centerX,
    );
    expect(await radios.evaluateAll((items) => items.map((item) => item.id))).toEqual(idsBefore);
  });

  test("Y labels use thin dedicated bottom-to-top visual pills", async ({ page }, testInfo) => {
    const pills = page.locator("#axis-selector-y .axis-option-pill");
    await expect(pills).toHaveCount(2);
    const geometry = await pills.evaluateAll((items) => items.map((item) => {
      const style = getComputedStyle(item);
      const matrix = new DOMMatrixReadOnly(style.transform);
      return {
        angle: Math.round(Math.atan2(matrix.b, matrix.a) * 180 / Math.PI),
        width: item.getBoundingClientRect().width,
        height: item.getBoundingClientRect().height,
      };
    }));
    for (const pill of geometry) {
      expect(pill.angle).toBe(-90);
      expect(pill.height).toBeGreaterThan(pill.width);
      if (testInfo.project.name.startsWith("desktop")) {
        expect(pill.width, "desktop Y pill cross-axis paint should be thin").toBeLessThanOrEqual(26);
      }
    }
  });

  test("desktop X faces keep thin paint while mobile retains touch geometry", async ({ page }, testInfo) => {
    const faces = await page.locator("#axis-selector-x .axis-option-face").evaluateAll((items) =>
      items.map((item) => item.getBoundingClientRect().height),
    );
    for (const height of faces) {
      if (testInfo.project.name.startsWith("mobile")) expect(height).toBeGreaterThanOrEqual(44);
      else expect(height, "desktop X pill paint should be thin").toBeLessThanOrEqual(22);
    }
  });

  for (const { initial, target, label } of [
    { initial: "28", target: "1", label: "28-day to 1-day" },
    { initial: "1", target: "28", label: "1-day to 28-day" },
  ] as const) {
    test(`Y faces exchange strictly vertically from ${label}`, async ({ page }) => {
      const initialInput = page.locator(`#axis-y-${initial}`);
      if (!(await initialInput.isChecked())) {
        await page.locator(`.axis-selector-y .axis-option-face:has(#axis-y-${initial})`).click();
        await waitForCanvasLoopToPark(page);
      }
      const exchange = await sampleYExchange(page, target);
      assertStrictlyVerticalYExchange(exchange, label);
    });
  }

  test("stable Y faces exchange vertically while native arrows retain focus", async ({ page }) => {
    const y = page.locator("#axis-selector-y");
    const radios = y.locator('input[type="radio"]');
    const idsBefore = await radios.evaluateAll((items) => items.map((item) => item.id));
    const before = await optionPositions(y);
    const checkedBefore = await checkedValue(y);
    const selectedBefore = before.find((position) => position.checked)!;
    const alternateBefore = before.find((position) => !position.checked)!;
    expect(Math.abs(selectedBefore.centerX - alternateBefore.centerX)).toBeLessThanOrEqual(1);
    expect(selectedBefore.centerY, "selected Y option must occupy the leading slot").toBeLessThan(
      alternateBefore.centerY,
    );

    await y.locator(`input[value="${checkedBefore}"]`).focus();
    await page.keyboard.press("ArrowDown");
    await expect(y.locator('input[type="radio"]:checked')).not.toHaveValue(checkedBefore);
    await expect(y.locator('input[type="radio"]:checked')).toBeFocused();
    await expect.poll(() => y.locator(".axis-option-face").evaluateAll((faces) =>
      faces.reduce((count, face) => count + face.getAnimations().length, 0),
    )).toBe(0);

    const after = await optionPositions(y);
    const selectedAfter = after.find((position) => position.checked)!;
    const alternateAfter = after.find((position) => !position.checked)!;
    expect(Math.abs(selectedAfter.centerX - alternateAfter.centerX)).toBeLessThanOrEqual(1);
    expect(selectedAfter.centerY, "newly selected Y option must settle in the leading slot").toBeLessThan(
      alternateAfter.centerY,
    );
    expect(await radios.evaluateAll((items) => items.map((item) => item.id))).toEqual(idsBefore);
  });

  test("capsules stay inside dedicated lanes with slack from drawable ticks", async ({ page }) => {
    const geometry = await page.evaluate(async () => {
      const { computePlotRect, resolvePlotInsets } = await import("/plot-geometry.mjs");
      const canvas = document.querySelector<HTMLCanvasElement>("#scatter-canvas")!;
      const canvasRect = canvas.getBoundingClientRect();
      const plot = computePlotRect(
        canvasRect.width,
        canvasRect.height,
        resolvePlotInsets(innerWidth),
      );
      const x = document.querySelector("#axis-selector-x")!.getBoundingClientRect();
      const y = document.querySelector("#axis-selector-y")!.getBoundingClientRect();
      return {
        canvas: { left: canvasRect.left, right: canvasRect.right, top: canvasRect.top, bottom: canvasRect.bottom },
        plot: {
          left: canvasRect.left + plot.left,
          right: canvasRect.left + plot.right,
          top: canvasRect.top + plot.top,
          bottom: canvasRect.top + plot.bottom,
        },
        x: { left: x.left, right: x.right, top: x.top, bottom: x.bottom },
        y: { left: y.left, right: y.right, top: y.top, bottom: y.bottom },
      };
    });

    expect(geometry.x.left).toBeGreaterThanOrEqual(geometry.plot.left);
    expect(geometry.x.right).toBeLessThanOrEqual(geometry.plot.right);
    expect(geometry.x.top).toBeGreaterThanOrEqual(geometry.plot.bottom + 8);
    expect(geometry.x.bottom).toBeLessThanOrEqual(geometry.canvas.bottom + 1);
    expect(geometry.y.left).toBeGreaterThanOrEqual(geometry.canvas.left - 1);
    expect(geometry.y.right).toBeLessThanOrEqual(geometry.plot.left - 8);
    expect(geometry.y.top).toBeGreaterThanOrEqual(geometry.plot.top);
    expect(geometry.y.bottom).toBeLessThanOrEqual(geometry.plot.bottom);
    expect(
      Math.abs((geometry.y.top + geometry.y.bottom) / 2 - (geometry.plot.top + geometry.plot.bottom) / 2),
      "Y selector must be centered on the drawable plot",
    ).toBeLessThanOrEqual(1);
  });

  test("selector gaps stay compact while Y keeps a fixed drift-free lane", async ({ page }, testInfo) => {
    const measure = () => page.evaluate(async () => {
      const {
        PLOT_LANE_COMPONENTS,
        SELECTOR_TICK_GAP,
        formatYAxisTick,
        resolvePlotLaneComponents,
      } = await import("/plot-geometry.mjs");
      const niceTickValues = (min: number, max: number, approxCount: number) => {
        const rawStep = (max - min) / approxCount;
        const magnitude = 10 ** Math.floor(Math.log10(rawStep));
        const ratio = rawStep / magnitude;
        const step = ratio < 1.5 ? magnitude : ratio < 3.5 ? 2 * magnitude : ratio < 7.5 ? 5 * magnitude : 10 * magnitude;
        const ticks: number[] = [];
        for (let value = Math.ceil(min / step) * step; value <= max; value += step) {
          ticks.push(Math.round(value * 1e6) / 1e6);
        }
        return ticks;
      };
      const canvas = document.querySelector<HTMLCanvasElement>("#scatter-canvas")! as HTMLCanvasElement & {
        _plotRect: { left: number; bottom: number };
        _xMin: number;
        _xMax: number;
        _yMin: number;
        _yMax: number;
      };
      const canvasRect = canvas.getBoundingClientRect();
      const ySelector = document.querySelector("#axis-selector-y")!.getBoundingClientRect();
      const xSelector = document.querySelector("#axis-selector-x")!.getBoundingClientRect();
      const context = canvas.getContext("2d")!;
      context.font = "bold 12px -apple-system, BlinkMacSystemFont, sans-serif";
      const xMetrics = niceTickValues(canvas._xMin, canvas._xMax, 5)
        .map((value) => context.measureText(String(Math.round(value))));
      const yLabels = niceTickValues(canvas._yMin, canvas._yMax, 5).map(formatYAxisTick);
      const yMetrics = yLabels.map((label) => context.measureText(label));
      const descent = (metric: TextMetrics) => Number.isFinite(metric.actualBoundingBoxDescent)
        ? metric.actualBoundingBoxDescent
        : 3;
      const yTickLeft = canvasRect.left + canvas._plotRect.left - 8 - Math.max(...yMetrics.map((metric) => metric.width));
      const xTickBottom = canvasRect.top + canvas._plotRect.bottom + 14 + Math.max(...xMetrics.map(descent));
      return {
        expected: SELECTOR_TICK_GAP,
        worstTickPaint: resolvePlotLaneComponents(innerWidth).left.worstTickPaint,
        configuredWorstTickPaint: PLOT_LANE_COMPONENTS.desktop.left.worstTickPaint,
        yLabels,
        yTickPaint: Math.max(...yMetrics.map((metric) => metric.width)),
        yGap: yTickLeft - ySelector.right,
        yLeft: ySelector.left,
        xGap: xSelector.top - xTickBottom,
        canvas: { width: canvasRect.width, height: canvasRect.height },
        plot: { ...canvas._plotRect },
      };
    });
    const tolerance = testInfo.project.name.includes("webkit") ? 2 : 1;
    const unitToggle = testInfo.project.name.startsWith("mobile")
      ? "#mobile-unit-toggle"
      : "#unit-toggle";

    let fixedYLeft: number | null = null;
    let previousGeometry: { canvas: { width: number; height: number }; plot: object } | null = null;
    for (const unit of ["SI", "US"] as const) {
      for (const xAxis of ["gwp", "cost"] as const) {
        for (const yDay of [28, 1] as const) {
          const xInput = page.locator(`#axis-x-${xAxis}`);
          if (!(await xInput.isChecked())) {
            await page.locator(`.axis-selector-x .axis-option-face:has(#axis-x-${xAxis})`).click();
          }
          const yInput = page.locator(`#axis-y-${yDay}`);
          if (!(await yInput.isChecked())) {
            await page.locator(`.axis-selector-y .axis-option-face:has(#axis-y-${yDay})`).click();
          }
          await expect(xInput).toBeChecked();
          await expect(yInput).toBeChecked();
          await expect.poll(() => page.evaluate(() => window.__test?.isAnimLoopActive)).toBe(false);
          const current = await measure();
          expect(Math.abs(current.xGap - current.expected), `${unit}/${xAxis}/${yDay} X painted gap`).toBeLessThanOrEqual(tolerance);
          expect(current.yGap, `${unit}/${xAxis}/${yDay} Y painted gap minimum`).toBeGreaterThanOrEqual(current.expected - tolerance);
          expect(current.yGap, `${unit}/${xAxis}/${yDay} Y painted gap optical bound`).toBeLessThanOrEqual(22 + tolerance);
          expect(Math.abs(current.yGap - current.xGap), `${unit}/${xAxis}/${yDay} X/Y gap parity`).toBeLessThanOrEqual(14 + tolerance);
          expect(current.yTickPaint, `${unit}/${xAxis}/${yDay} Y tick paint budget`).toBeLessThanOrEqual(27);
          expect(new Set(current.yLabels).size, `${unit}/${xAxis}/${yDay} Y tick labels stay distinct`).toBe(current.yLabels.length);
          expect(current.worstTickPaint).toBe(27);
          expect(current.worstTickPaint).toBe(current.configuredWorstTickPaint);
          fixedYLeft ??= current.yLeft;
          expect(Math.abs(current.yLeft - fixedYLeft), `${unit}/${xAxis}/${yDay} fixed Y wrapper X`).toBeLessThanOrEqual(0.5);
          if (previousGeometry) {
            expect(current.canvas, `${unit}/${xAxis}/${yDay} canvas stable through state change`).toEqual(previousGeometry.canvas);
            expect(current.plot, `${unit}/${xAxis}/${yDay} plot stable through state change`).toEqual(previousGeometry.plot);
          }
          previousGeometry = { canvas: current.canvas, plot: current.plot };
        }
      }
      if (unit === "SI") {
        await page.locator(unitToggle).click();
        await expect.poll(() => page.evaluate(() => window.__test?.isAnimLoopActive)).toBe(true);
        await expect.poll(() => page.evaluate(() => window.__test?.isAnimLoopActive)).toBe(false);
      }
    }
  });

  test("overlapping Y pills keep the top label fully masked across responsive lanes", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "one viewport sweep per browser engine");
    for (const width of [320, 1050, 1051, 1280]) {
      await page.setViewportSize({ width, height: width === 320 ? 720 : 900 });
      await waitForCanvasLoopToPark(page);
      for (const target of ["1", "28"] as const) {
        const samples = await page.evaluate(async (targetValue) => {
          const group = document.querySelector<HTMLElement>("#axis-selector-y")!;
          const faces = [...group.querySelectorAll<HTMLElement>(".axis-option-face")];
          const pills = faces.map((face) => face.querySelector<HTMLElement>(".axis-option-pill")!);
          group.querySelector<HTMLInputElement>(`input[value="${targetValue}"]`)!.click();
          group.getBoundingClientRect();
          const animations = group.getAnimations({ subtree: true })
            .filter((animation) => animation instanceof CSSTransition);
          for (const animation of animations) animation.pause();

          const alpha = (color: string) => {
            if (color === "transparent") return 0;
            const rgba = color.match(/^rgba\([^,]+,[^,]+,[^,]+,\s*([\d.]+)\)$/);
            if (rgba) return Number(rgba[1]);
            const modern = color.match(/\/\s*([\d.]+)\s*\)$/);
            return modern ? Number(modern[1]) : 1;
          };
          const samples = [];
          for (const progress of [0, 0.25, 0.5, 0.75, 1]) {
            for (const animation of animations) {
              const duration = Number(animation.effect?.getTiming().duration ?? 0);
              animation.currentTime = duration * progress;
            }
            await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
            const rects = pills.map((pill) => pill.getBoundingClientRect());
            const overlapHeight = Math.max(0, Math.min(rects[0].bottom, rects[1].bottom) - Math.max(rects[0].top, rects[1].top));
            const overlapWidth = Math.max(0, Math.min(rects[0].right, rects[1].right) - Math.max(rects[0].left, rects[1].left));
            const overlapArea = overlapHeight * overlapWidth;
            const pillArea = Math.min(...rects.map((rect) => rect.width * rect.height));
            const topIndex = Number(getComputedStyle(faces[1]).zIndex) > Number(getComputedStyle(faces[0]).zIndex) ? 1 : 0;
            samples.push({
              progress,
              overlapRatio: pillArea > 0 ? overlapArea / pillArea : 0,
              topAlpha: alpha(getComputedStyle(pills[topIndex]).backgroundColor),
            });
          }
          for (const animation of animations) animation.finish();
          return samples;
        }, target);
        const midpoint = samples.find((sample) => sample.progress === 0.5)!;
        expect(midpoint.overlapRatio, `${width}px ${target}-day pills must overlap at midpoint`)
          .toBeGreaterThanOrEqual(0.8);
        expect(
          Math.min(...samples.map((sample) => sample.topAlpha)),
          `${width}px ${target}-day exchange must mask the lower label at every progress point`,
        ).toBeGreaterThanOrEqual(0.98);
        await waitForCanvasLoopToPark(page);
      }
    }
  });

  test("extreme current compositions keep animated axis endpoints continuous", async ({ page }) => {
    await driveEditableCompositionTo(page, "min");
    const low = await page.evaluate(() => window.__test!.scatterDisplayedSnapshot.current);
    expect(Number.isFinite(low.x)).toBe(true);
    expect(Number.isFinite(low.y)).toBe(true);

    await driveEditableCompositionTo(page, "max");
    const high = await page.evaluate(() => {
      const displayed = window.__test!.scatterDisplayedSnapshot;
      return {
        current: displayed.current,
        catalogMaxX: Math.max(...displayed.x),
      };
    });
    expect(high.current.x, "maximum editable composition must exercise the outlying-current path")
      .toBeGreaterThan(high.catalogMaxX);

    for (const scenario of [
      { selector: "#axis-y-1", label: "SI 28-day to 1-day" },
      { selector: "#axis-y-28", label: "SI 1-day to 28-day" },
      { selector: "#axis-x-cost", label: "SI GWP to Cost" },
    ]) {
      const exchange = await sampleScatterRangeExchange(page, scenario.selector);
      assertRangeExchangeContinuous(exchange, scenario.label);
    }

    await page.evaluate(() => document.dispatchEvent(new CustomEvent("toggle-units")));
    await waitForCanvasLoopToPark(page);
    const usExchange = await sampleScatterRangeExchange(page, "#axis-y-1");
    assertRangeExchangeContinuous(usExchange, "US 28-day to 1-day");
  });

  test("CSS and JavaScript share the same responsive plot inset contract", async ({ page }) => {
    const values = await page.evaluate(async () => {
      const { resolvePlotInsets } = await import("/plot-geometry.mjs");
      const style = getComputedStyle(document.documentElement);
      const css = {
        left: parseFloat(style.getPropertyValue("--plot-inset-left")),
        right: parseFloat(style.getPropertyValue("--plot-inset-right")),
        top: parseFloat(style.getPropertyValue("--plot-inset-top")),
        bottom: parseFloat(style.getPropertyValue("--plot-inset-bottom")),
      };
      return { css, js: resolvePlotInsets(innerWidth) };
    });
    expect(values.css).toEqual(values.js);
  });

  test("compact Y paint and aligned drawables survive responsive and theme boundaries", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "one responsive sweep per browser engine");
    for (const width of [320, 412, 844, 1050, 1051, 1280, 1728]) {
      await page.setViewportSize({ width, height: width === 320 ? 720 : 1000 });
      for (const theme of ["light", "dark"] as const) {
        await page.evaluate((requestedTheme) => {
          document.documentElement.dataset.theme = requestedTheme;
          window.dispatchEvent(new Event("resize"));
        }, theme);
        await waitForCanvasLoopToPark(page);
        const geometry = await page.evaluate(async () => {
          const {
            computePlotRect,
            formatYAxisTick,
            resolvePlotInsets,
            resolvePlotLaneComponents,
          } = await import("/plot-geometry.mjs");
          const niceTickValues = (min: number, max: number, approxCount: number) => {
            const rawStep = (max - min) / approxCount;
            const magnitude = 10 ** Math.floor(Math.log10(rawStep));
            const ratio = rawStep / magnitude;
            const step = ratio < 1.5 ? magnitude : ratio < 3.5 ? 2 * magnitude : ratio < 7.5 ? 5 * magnitude : 10 * magnitude;
            const ticks: number[] = [];
            for (let value = Math.ceil(min / step) * step; value <= max; value += step) {
              ticks.push(Math.round(value * 1e6) / 1e6);
            }
            return ticks;
          };
          const scatter = document.querySelector<HTMLCanvasElement>("#scatter-canvas")! as HTMLCanvasElement & {
            _plotRect: { left: number; right: number; top: number; bottom: number; width: number; height: number };
            _yMin: number;
            _yMax: number;
          };
          const strength = document.querySelector<HTMLCanvasElement>("#curve-canvas")! as HTMLCanvasElement & {
            _plotRect: { width: number; height: number };
            _yMin: number;
            _yMax: number;
          };
          const scatterBox = scatter.getBoundingClientRect();
          const scatterContext = scatter.getContext("2d")!;
          const strengthContext = strength.getContext("2d")!;
          for (const context of [scatterContext, strengthContext]) {
            context.font = "bold 12px -apple-system, BlinkMacSystemFont, sans-serif";
          }
          const representativeTickValues = [2_500, 3_500, 8_500, 9_500, 20_000];
          const measureTickPaint = (
            context: CanvasRenderingContext2D,
            min: number,
            max: number,
          ) => [
            ...niceTickValues(min, max, 5),
            ...representativeTickValues,
          ]
            .map(formatYAxisTick)
            .reduce((width, label) => Math.max(width, context.measureText(label).width), 0);
          const selector = document.querySelector(".axis-selector-y-wrap")!.getBoundingClientRect();
          const lane = resolvePlotLaneComponents(innerWidth).left;
          return {
            theme: document.documentElement.dataset.theme,
            scatterTickPaint: measureTickPaint(scatterContext, scatter._yMin, scatter._yMax),
            strengthTickPaint: measureTickPaint(strengthContext, strength._yMin, strength._yMax),
            tickBudget: lane.worstTickPaint,
            scatterPlot: { width: scatter._plotRect.width, height: scatter._plotRect.height },
            strengthPlot: { width: strength._plotRect.width, height: strength._plotRect.height },
            effectLeft: selector.left - lane.selectorEffectReserve,
            canvasLeft: scatterBox.left,
            selectorRight: selector.right,
            plotLeft: scatterBox.left + scatter._plotRect.left,
          };
        });
        const label = `${testInfo.project.name}/${width}px/${theme}`;
        expect(geometry.theme, `${label} theme applied`).toBe(theme);
        expect(geometry.scatterTickPaint, `${label} Scatter compact Y tick paint`)
          .toBeLessThanOrEqual(geometry.tickBudget);
        expect(geometry.strengthTickPaint, `${label} Strength compact Y tick paint`)
          .toBeLessThanOrEqual(geometry.tickBudget);
        expect(Math.abs(geometry.scatterPlot.width - geometry.strengthPlot.width), `${label} plot width parity`)
          .toBeLessThanOrEqual(1);
        expect(Math.abs(geometry.scatterPlot.height - geometry.strengthPlot.height), `${label} plot height parity`)
          .toBeLessThanOrEqual(1);
        expect(geometry.effectLeft, `${label} Y selector effect stays inside canvas lane`)
          .toBeGreaterThanOrEqual(geometry.canvasLeft - 1);
        expect(geometry.selectorRight, `${label} Y selector stays left of drawable`)
          .toBeLessThanOrEqual(geometry.plotLeft - 8);
      }
    }
  });

  test("axis changes update the canvas name and share one transition window", async ({ page }) => {
    const x = page.locator("#axis-selector-x");
    const alternate = x.locator('.axis-option-face:has(input[type="radio"]:not(:checked))');
    const target = await alternate.locator('input[type="radio"]').inputValue();
    await alternate.click();

    await expect(page.locator("#scatter-canvas")).toHaveAttribute(
      "aria-label",
      new RegExp(target === "cost" ? "Cost" : "GWP", "i"),
    );
    await expect.poll(() => page.evaluate(() => (window as any).__test?.isScatterTransitionActive)).toBe(true);
    const labelTiming = await x.locator(".axis-option-face").evaluateAll((faces) =>
      faces.flatMap((face) => face.getAnimations())
        .filter((animation) => animation instanceof CSSTransition)
        .filter((animation) => (animation as CSSTransition).transitionProperty === "transform")
        .map((animation) => ({
          duration: animation.effect?.getTiming().duration,
          easing: animation.effect?.getTiming().easing,
        })),
    );
    expect(labelTiming.length, "axis change must create real label transitions").toBeGreaterThan(0);
    expect(labelTiming.every(({ duration }) => duration === 350)).toBe(true);
    expect(labelTiming.every(({ easing }) => easing === "cubic-bezier(0.645, 0.045, 0.355, 1)"))
      .toBe(true);
    expect(await page.evaluate(() => (window as any).__test.scatterTransitionEasing))
      .toBe("cubic-bezier(0.645, 0.045, 0.355, 1)");
    await expect.poll(() => page.evaluate(() => (window as any).__test?.isScatterTransitionActive), {
      timeout: 5000,
    }).toBe(false);
    await expect(x).not.toHaveAttribute("data-transitioning", "true");
  });

  test("rapid X-axis reversals retarget continuously from the displayed interpolation", async ({ page }) => {
    const x = page.locator("#axis-selector-x");
    const first = x.locator('input[type="radio"]:checked');
    const second = x.locator('.axis-option-face:has(input[type="radio"]:not(:checked))');
    const initial = await first.inputValue();
    const target = await second.locator('input[type="radio"]').inputValue();
    const { before, after } = await page.evaluate(async ({ initialValue, targetValue }) => {
      const group = document.querySelector<HTMLElement>("#axis-selector-x")!;
      group.querySelector<HTMLInputElement>(`input[value="${targetValue}"]`)!.click();
      const started = performance.now();
      await new Promise<void>((resolve) => {
        const sample = () => {
          if (performance.now() - started >= 120 && window.__test?.isScatterTransitionActive) resolve();
          else requestAnimationFrame(sample);
        };
        requestAnimationFrame(sample);
      });
      const before = window.__test!.scatterDisplayedSnapshot;
      group.querySelector<HTMLInputElement>(`input[value="${initialValue}"]`)!.click();
      const after = window.__test!.scatterTransitionFromSnapshot;
      return { before, after };
    }, { initialValue: initial, targetValue: target });
    expect(after.x.length).toBe(before.x.length);
    expect(Math.max(...after.x.map((value: number, index: number) => Math.abs(value - before.x[index]))))
      .toBeLessThan(1e-6);
    expect(Math.max(...after.y.map((value: number, index: number) => Math.abs(value - before.y[index]))))
      .toBeLessThan(1e-6);
    expect(Math.abs(after.current.x - before.current.x)).toBeLessThan(1e-6);
    expect(Math.abs(after.current.y - before.current.y)).toBeLessThan(1e-6);
    expect(Math.abs(after.xMax - before.xMax)).toBeLessThan(1e-6);
    expect(Math.abs(after.yMax - before.yMax)).toBeLessThan(1e-6);
  });

  for (const { initial, target, label } of [
    { initial: "28", target: "1", label: "28-day toward 1-day and back" },
    { initial: "1", target: "28", label: "1-day toward 28-day and back" },
  ] as const) {
    test(`rapid Y reversal stays continuous: ${label}`, async ({ page }) => {
      const initialInput = page.locator(`#axis-y-${initial}`);
      if (!(await initialInput.isChecked())) {
        await page.locator(`.axis-selector-y .axis-option-face:has(#axis-y-${initial})`).click();
        await waitForCanvasLoopToPark(page);
      }
      const result = await page.evaluate(async ({ initial, target }) => {
        const group = document.querySelector<HTMLElement>("#axis-selector-y")!;
        const wrapper = document.querySelector<HTMLElement>(".axis-selector-y-wrap")!;
        const faces = [...group.querySelectorAll<HTMLElement>(".axis-option-face")];
        const snapshotFaces = () => faces.map((face) => {
          const rect = face.getBoundingClientRect();
          return { top: rect.top, left: rect.left, centerX: rect.left + rect.width / 2 };
        });
        type ReversalSample = {
          active: boolean;
          wrapperLeft: number;
          faces: ReturnType<typeof snapshotFaces>;
        };
        const forwardSamples: ReversalSample[] = [];
        const reverseSamples: ReversalSample[] = [];
        group.querySelector<HTMLInputElement>(`input[value="${target}"]`)!.click();
        await new Promise<void>((resolve) => {
          const started = performance.now();
          const sample = () => {
            forwardSamples.push({
              active: Boolean(window.__test?.isScatterTransitionActive),
              wrapperLeft: wrapper.getBoundingClientRect().left,
              faces: snapshotFaces(),
            });
            if (performance.now() - started >= 120) resolve();
            else requestAnimationFrame(sample);
          };
          requestAnimationFrame(sample);
        });
        const before = window.__test!.scatterDisplayedSnapshot;
        const beforeFaces = snapshotFaces();
        group.querySelector<HTMLInputElement>(`input[value="${initial}"]`)!.click();
        const after = window.__test!.scatterTransitionFromSnapshot;
        const reversalFaces = snapshotFaces();
        const deadline = performance.now() + 2_000;
        await new Promise<void>((resolve, reject) => {
          const sample = () => {
            reverseSamples.push({
              active: Boolean(window.__test?.isScatterTransitionActive),
              wrapperLeft: wrapper.getBoundingClientRect().left,
              faces: snapshotFaces(),
            });
            if (!window.__test?.isScatterTransitionActive) resolve();
            else if (performance.now() >= deadline) reject(new Error("Y reversal did not settle"));
            else requestAnimationFrame(sample);
          };
          requestAnimationFrame(sample);
        });
        return { before, after, beforeFaces, reversalFaces, forwardSamples, reverseSamples };
      }, { initial, target });

      expect(result.after.x.length).toBe(result.before.x.length);
      expect(Math.max(...result.after.x.map((value: number, index: number) => Math.abs(value - result.before.x[index]))))
        .toBeLessThan(1e-6);
      expect(Math.max(...result.after.y.map((value: number, index: number) => Math.abs(value - result.before.y[index]))))
        .toBeLessThan(1e-6);
      expect(Math.abs(result.after.current.x - result.before.current.x)).toBeLessThan(1e-6);
      expect(Math.abs(result.after.current.y - result.before.current.y)).toBeLessThan(1e-6);
      expect(Math.abs(result.after.xMax - result.before.xMax)).toBeLessThan(1e-6);
      expect(Math.abs(result.after.yMax - result.before.yMax)).toBeLessThan(1e-6);
      const wrapperLefts = [...result.forwardSamples, ...result.reverseSamples]
        .map((sample) => sample.wrapperLeft);
      expect(Math.max(...wrapperLefts) - Math.min(...wrapperLefts), "Y reversal wrapper X must remain fixed")
        .toBeLessThanOrEqual(0.5);
      const settled = result.reverseSamples.at(-1)!;
      expect(settled.active).toBe(false);
      for (const faceIndex of [0, 1]) {
        const reversalTop = result.reversalFaces[faceIndex].top;
        const afterReversal = [
          reversalTop,
          ...result.reverseSamples.map((sample) => sample.faces[faceIndex].top),
        ];
        const direction = Math.sign(afterReversal.at(-1)! - reversalTop);
        for (let index = 1; index < afterReversal.length; index += 1) {
          expect(
            direction * (afterReversal[index] - afterReversal[index - 1]),
            `${label} face ${faceIndex} must not reverse after retargeting`,
          ).toBeGreaterThanOrEqual(-0.25);
        }
      }
    });
  }
});
