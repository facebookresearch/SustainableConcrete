import { test, expect } from "@playwright/test";

/**
 * Drag redraw efficiency.
 *
 * `onSliderChange` schedules a coalesced redraw AND keeps the animation loop
 * alive. The loop already redraws both canvases every frame, so update() from
 * the coalesced callback drew them a second time — measured at 1.68 canvas
 * redraws per animation frame during a sustained drag, i.e. ~40% wasted work
 * on the one interaction path that the earlier transition/worker optimisations
 * never touched.
 *
 * The loop owns the canvases while it runs; the coalesced callback still has
 * to do the non-canvas work, so this also pins that the readouts keep updating.
 */
test("drag does not redraw the canvases twice per frame", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "one project is enough");

  await page.goto("/?test=1");
  await page.waitForFunction(() => (window as any).__test?.modelReady === true, null, {
    timeout: 30000,
  });
  await page.waitForTimeout(800);

  const r = await page.evaluate(async () => {
    // setupHiDPICanvas assigns canvas.width on every draw, so counting
    // assignments counts redraws directly.
    const counts: Record<string, number> = { curve: 0, scatter: 0 };
    for (const [key, id] of [
      ["curve", "curve-canvas"],
      ["scatter", "scatter-canvas"],
    ] as const) {
      const c = document.getElementById(id) as HTMLCanvasElement;
      let real = c.width;
      Object.defineProperty(c, "width", {
        get: () => real,
        set: (v) => { counts[key]++; real = v; },
        configurable: true,
      });
    }

    let frames = 0;
    let running = true;
    const tick = () => { if (running) { frames++; requestAnimationFrame(tick); } };
    requestAnimationFrame(tick);

    // Drive input faster than 60 Hz, as a touch drag or high-polling mouse does.
    const s = document.querySelector("#sliders input[type=range]") as HTMLInputElement;
    const min = parseFloat(s.min);
    const max = parseFloat(s.max);
    const t0 = performance.now();
    let events = 0;
    while (performance.now() - t0 < 1500) {
      s.value = String(min + (max - min) * ((events % 20) / 20));
      s.dispatchEvent(new Event("input", { bubbles: true }));
      events++;
      await new Promise((res) => setTimeout(res, 8));
    }
    running = false;
    await new Promise((res) => setTimeout(res, 300));

    return { ...counts, frames, readout: document.getElementById("gwp-value")?.textContent };
  });

  expect(r.frames, "no animation frames observed").toBeGreaterThan(10);

  const perFrame = r.curve / r.frames;
  expect(
    perFrame,
    `${r.curve} curve redraws over ${r.frames} frames = ${perFrame.toFixed(2)}/frame ` +
      "(was 1.68 when the loop and the coalesced callback both drew)",
  ).toBeLessThan(1.35);

  // The coalesced callback must still do the non-canvas work.
  expect(r.readout, "readouts stopped updating during the drag").toBeTruthy();
  expect(r.readout).not.toBe("–");
});
