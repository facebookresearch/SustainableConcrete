import { expect, type Page } from "@playwright/test";

export type CanvasName = "curve" | "scatter";

export type CanvasDrawEvent = {
  canvas: CanvasName;
  frameTime: number | null;
  phase: "raf" | "sync";
};

export type CanvasFrame = {
  frameTime: number | null;
  curve: number;
  scatter: number;
};

export type CanvasFrameSnapshot = {
  events: CanvasDrawEvent[];
  frames: CanvasFrame[];
};

declare global {
  interface Window {
    __test?: any;
    __canvasFrameProbe?: {
      reset(): void;
      snapshot(): CanvasDrawEvent[];
    };
  }
}

export async function installCanvasFrameProbe(page: Page): Promise<void> {
  await page.addInitScript(() => {
    if (window.__canvasFrameProbe) return;

    const nativeRequestAnimationFrame = window.requestAnimationFrame.bind(window);
    const widthDescriptor = Object.getOwnPropertyDescriptor(
      HTMLCanvasElement.prototype,
      "width",
    );
    if (!widthDescriptor?.get || !widthDescriptor.set) {
      throw new Error("HTMLCanvasElement.width accessors are unavailable");
    }

    let events: CanvasDrawEvent[] = [];
    let activeFrameTime: number | null = null;
    let rafDepth = 0;

    window.requestAnimationFrame = (callback: FrameRequestCallback): number =>
      nativeRequestAnimationFrame((frameTime) => {
        const previousFrameTime = activeFrameTime;
        activeFrameTime = frameTime;
        rafDepth += 1;
        try {
          callback(frameTime);
        } finally {
          rafDepth -= 1;
          activeFrameTime = previousFrameTime;
        }
      });

    Object.defineProperty(HTMLCanvasElement.prototype, "width", {
      ...widthDescriptor,
      set(this: HTMLCanvasElement, value: number) {
        widthDescriptor.set!.call(this, value);
        const canvas = this.id === "curve-canvas"
          ? "curve"
          : this.id === "scatter-canvas"
            ? "scatter"
            : null;
        if (canvas === null) return;
        events.push({
          canvas,
          frameTime: rafDepth > 0 ? activeFrameTime : null,
          phase: rafDepth > 0 ? "raf" : "sync",
        });
      },
    });

    window.__canvasFrameProbe = {
      reset() {
        events = [];
      },
      snapshot() {
        return events.map((event) => ({ ...event }));
      },
    };
  });
}

export async function waitForCanvasAppReady(page: Page): Promise<void> {
  await page.goto("/?test=1");
  await page.waitForFunction(
    () => window.__test?.modelReady === true,
    null,
    { timeout: 30_000 },
  );
  await page.locator("#sliders input[type=range]").first().waitFor({ state: "attached" });
  await waitForCanvasLoopToPark(page);
}

export async function waitForCanvasLoopToPark(page: Page): Promise<void> {
  await expect
    .poll(
      () => page.evaluate(() => {
        const testState = window.__test;
        return Boolean(
          testState &&
            !testState.isAnimLoopActive &&
            !testState.isCurveTransitionActive &&
            !testState.isPreviewCurveTransitionActive &&
            !testState.isCompositionTransitionActive,
        );
      }),
      { timeout: 10_000 },
    )
    .toBe(true);

  await page.evaluate(() =>
    new Promise<void>((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
    ),
  );

  await expect
    .poll(() => page.evaluate(() => !window.__test?.isAnimLoopActive))
    .toBe(true);
}

export async function resetCanvasFrameProbe(page: Page): Promise<void> {
  await page.evaluate(() => window.__canvasFrameProbe?.reset());
}

export async function snapshotCanvasFrames(page: Page): Promise<CanvasFrameSnapshot> {
  const events = await page.evaluate(() => window.__canvasFrameProbe?.snapshot() ?? []);
  const byFrame = new Map<number, CanvasFrame>();
  for (const event of events) {
    if (event.phase !== "raf" || event.frameTime === null) continue;
    let frame = byFrame.get(event.frameTime);
    if (!frame) {
      frame = { frameTime: event.frameTime, curve: 0, scatter: 0 };
      byFrame.set(event.frameTime, frame);
    }
    frame[event.canvas] += 1;
  }
  return { events, frames: [...byFrame.values()] };
}

export function drawCount(snapshot: CanvasFrameSnapshot, canvas: CanvasName): number {
  return snapshot.events.filter((event) => event.canvas === canvas).length;
}

export function maxDrawsPerFrame(
  snapshot: CanvasFrameSnapshot,
  canvas: CanvasName,
): number {
  return Math.max(0, ...snapshot.frames.map((frame) => frame[canvas]));
}

export function expectAtMostOneDrawPerCanvasPerFrame(
  snapshot: CanvasFrameSnapshot,
): void {
  expect(snapshot.frames, "expected at least one canvas frame").not.toHaveLength(0);
  expect(
    snapshot.events.filter((event) => event.phase === "sync"),
    "canvas renderers must be owned by requestAnimationFrame",
  ).toHaveLength(0);
  expect(
    maxDrawsPerFrame(snapshot, "curve"),
    `curve draw counts by frame: ${JSON.stringify(snapshot.frames)}`,
  ).toBeLessThanOrEqual(1);
  expect(
    maxDrawsPerFrame(snapshot, "scatter"),
    `scatter draw counts by frame: ${JSON.stringify(snapshot.frames)}`,
  ).toBeLessThanOrEqual(1);
}

export async function hoverRenderedScatterPoint(page: Page): Promise<void> {
  const canvas = page.locator("#scatter-canvas");
  const box = await canvas.boundingBox();
  if (!box) throw new Error("scatter canvas has no bounding box");

  for (let y = 30; y < box.height - 30; y += 8) {
    for (let x = 75; x < box.width - 20; x += 8) {
      await page.mouse.move(box.x + x, box.y + y);
      if (await page.evaluate(() => window.__test?.hoveredPointIdx !== null)) return;
    }
  }
  throw new Error("could not locate a rendered scatter point");
}

export async function hoverRenderedCurveObservation(page: Page): Promise<void> {
  const canvas = page.locator("#curve-canvas");
  const box = await canvas.boundingBox();
  if (!box) throw new Error("curve canvas has no bounding box");

  for (let y = 20; y < box.height - 20; y += 5) {
    for (let x = 50; x < box.width - 20; x += 5) {
      await page.mouse.move(box.x + x, box.y + y);
      if (await page.locator(".tooltip").isVisible()) return;
    }
  }
  throw new Error("could not locate a rendered curve observation");
}
