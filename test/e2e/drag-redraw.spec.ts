import { expect, test } from "@playwright/test";
import {
  drawCount,
  expectAtMostOneDrawPerCanvasPerFrame,
  installCanvasFrameProbe,
  resetCanvasFrameProbe,
  snapshotCanvasFrames,
  waitForCanvasAppReady,
  waitForCanvasLoopToPark,
} from "./canvas-frame-probe";

/**
 * Drag redraw efficiency.
 *
 * Slider input can arrive faster than display refresh. The application must
 * coalesce that work so each canvas renders at most once per browser frame,
 * while keeping synchronous readouts current throughout the gesture.
 */
test("drag redraws each canvas at most once per frame", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "one project is enough");
  await installCanvasFrameProbe(page);
  await waitForCanvasAppReady(page);
  await resetCanvasFrameProbe(page);

  const readout = await page.evaluate(async () => {
    const slider = document.querySelector("#sliders input[type=range]") as HTMLInputElement;
    const initialReadout = document.getElementById("gwp-value")?.textContent;
    const seenReadouts = new Set<string | undefined>([initialReadout]);
    const min = Number(slider.min);
    const max = Number(slider.max);
    const deadline = performance.now() + 750;
    let events = 0;

    while (performance.now() < deadline) {
      slider.value = String(min + (max - min) * ((events % 20) / 20));
      slider.dispatchEvent(new Event("input", { bubbles: true }));
      seenReadouts.add(document.getElementById("gwp-value")?.textContent);
      events += 1;
      await new Promise<void>((resolve) => setTimeout(resolve, 8));
    }

    return {
      final: document.getElementById("gwp-value")?.textContent,
      distinctValues: seenReadouts.size,
    };
  });

  await waitForCanvasLoopToPark(page);
  const snapshot = await snapshotCanvasFrames(page);

  expect(drawCount(snapshot, "curve"), "expected curve redraws during drag").toBeGreaterThan(10);
  expectAtMostOneDrawPerCanvasPerFrame(snapshot);
  expect(readout.final, "readouts stopped updating during the drag").toBeTruthy();
  expect(readout.final).not.toBe("–");
  expect(readout.distinctValues, "readout stayed stale throughout the drag").toBeGreaterThan(1);
});
