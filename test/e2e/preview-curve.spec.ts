import { test, expect } from "@playwright/test";

/**
 * Regression pin for two related Material Source toggle bugs:
 *
 *   (1) `displayPreviewComp` used to lag behind `currentComposition` after
 *       a toggle, which made the dashed preview curve "ghost" the previous
 *       mix. The fix synchronizes both arrays inside the toggle handlers.
 *
 *   (2) The Mix Insight panel used to retain the previous mix's description
 *       after toggling, even when the new (median + other source)
 *       composition is not in the training set. The fix schedules an
 *       insight update on every toggle.
 *
 * These specs rely on the `?test=1` window hook (`window.__test`) which
 * exposes read-only views of `currentComposition` and `displayPreviewComp`.
 */
async function openPreviewTestPage(page: import("@playwright/test").Page) {
  await page.goto("/?test=1");
  await page.waitForFunction(() => (window as any).__test?.modelReady === true, null, {
    timeout: 20000,
  });
  await expect(page.locator("#sliders input[type=range]").first()).toBeVisible();
}

async function hoverRenderedScatterPoint(page: import("@playwright/test").Page) {
  const canvas = page.locator("#scatter-canvas");
  const box = await canvas.boundingBox();
  if (!box) throw new Error("scatter canvas has no bounding box");
  for (let y = 30; y < box.height - 30; y += 8) {
    for (let x = 75; x < box.width - 20; x += 8) {
      await page.mouse.move(box.x + x, box.y + y);
      if (await page.evaluate(() => (window as any).__test.hoveredPointIdx !== null)) {
        return canvas;
      }
    }
  }
  throw new Error("could not locate a rendered scatter point");
}

async function waitForPreviewToSettle(page: import("@playwright/test").Page) {
  await expect
    .poll(
      () =>
        page.evaluate(() => {
          const t = (window as any).__test;
          return t.previewSource === null &&
            t.displayPreviewComp.every(
              (value: number, idx: number) =>
                Math.abs(value - t.currentComposition[idx]) <= 1e-6,
            );
        }),
      { timeout: 5000 },
    )
    .toBe(true);
}

test.describe("preview curve composition sync", () => {
  test("displayPreviewComp matches currentComposition after Material Source toggle", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop only — toggle visible there");
    await page.goto("/?test=1");
    await expect(page.locator("#sliders .slider-group").first()).toBeVisible({ timeout: 5000 });
    await page.waitForFunction(() => typeof (window as any).__test !== "undefined");
    // The shell renders before the GP finishes building in the worker, so
    // wait for the model itself before asserting on predictions.
    await page.waitForFunction(() => (window as any).__test.modelReady === true, null, { timeout: 20000 });

    const toggleButtons = page.locator(".material-source-group .toggle-btn");
    expect(await toggleButtons.count(), "expected three toggle buttons").toBe(3);

    for (const idx of [1, 2, 0]) {
      await toggleButtons.nth(idx).click();
      // Curve transition is 350ms; wait it out before sampling state.
      await page.waitForTimeout(450);
      const result = await page.evaluate(() => {
        const t = (window as any).__test;
        return { current: t.currentComposition, preview: t.displayPreviewComp };
      });
      expect(result.preview).toEqual(result.current);
    }
  });
});

test.describe("mix insight refreshes on Material Source toggle", () => {
  test("does not retain previous mix's description after toggle", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mix insight only visible on desktop");
    await page.goto("/");
    // The insight panel is populated after the strength model resolves (now
    // off-thread in a worker), so wait for the app to finish wiring first.
    await expect(page.locator("#sliders .slider-group").first()).toBeVisible({ timeout: 15000 });
    const insightText = page.locator("#mix-insight-text");
    await expect(insightText).toBeVisible();

    // Wait for the initial mix insight to populate (median composition usually
    // matches a training mix, so we get a real description rather than the
    // placeholder). If that's not true on this dataset, we still proceed —
    // the test only asserts that the insight is REFRESHED, not its specific
    // content before/after.
    await page.waitForFunction(
      () => {
        const el = document.getElementById("mix-insight-text");
        return el !== null && el.textContent !== null;
      },
      { timeout: 5000 },
    );
    // Settle any in-flight content swap animation
    await page.waitForTimeout(700);
    const before = (await insightText.textContent())?.trim() ?? "";

    // Click whichever Material Source toggle is currently inactive
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)");
    await inactive.first().click();

    // Wait for: 350ms curve transition + 300ms scheduleInsightUpdate delay +
    // 300ms content-swap animation = ~950ms. Use 1300ms to be safe.
    await page.waitForTimeout(1300);

    const after = (await insightText.textContent())?.trim() ?? "";

    // The displayed insight must reflect the post-toggle composition, not the
    // previous one. Either it's a different real description (the new
    // composition matches a training mix), or it's the "not available"/
    // placeholder text. The one thing it must NOT be is the same text as
    // before (which would indicate the bug).
    expect(
      after === "" || after !== before,
      `mix-insight-text must update on Material Source toggle (still: "${after.slice(0, 80)}...")`,
    ).toBe(true);
  });
});

test.describe("strength curve transitions smoothly on Material Source toggle", () => {
  test("curve transition state is active immediately after toggle and clears after 350ms", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "strength curve canvas is most visible on desktop");
    // The pixel-diff alternative (sampling canvas at before/mid/after) was
    // racing with screenshot timing — the browser sometimes batched rAF
    // frames so the mid screenshot captured the post-transition state.
    // The deterministic substitute is a state hook on `_curveTransition`
    // exposed via `?test=1`. We assert two things:
    //   (1) Right after the click, `_curveTransition` is active.
    //   (2) After waiting longer than the 350 ms blend window plus a
    //       safety margin, the state has cleared back to `null`.
    // We additionally assert the canvas pixels change overall (toggle
    // produced a visible difference), which is robust because the wait
    // is ≥ 600 ms long.
    await page.goto("/?test=1");
    await expect(page.locator("canvas#curve-canvas")).toBeVisible();
    await expect(page.locator(".material-source-group .toggle-btn").first()).toBeVisible({ timeout: 5000 });
    await page.waitForFunction(() => typeof (window as any).__test !== "undefined");
    // The shell renders before the GP finishes building in the worker, so
    // wait for the model itself before asserting on predictions.
    await page.waitForFunction(() => (window as any).__test.modelReady === true, null, { timeout: 20000 });
    await page.waitForTimeout(800); // settle initial fade-ins / WASM init

    const curve = page.locator("canvas#curve-canvas");
    const before = await curve.screenshot();

    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await inactive.click();

    // Within a few milliseconds of the click handler firing, the transition
    // state should be active. Use waitForFunction with a tight timeout so
    // we don't accidentally observe the post-transition state.
    await page.waitForFunction(
      () => (window as any).__test.isCurveTransitionActive === true,
      null,
      { timeout: 100 },
    );

    // After the 350 ms duration plus generous safety margin, the state
    // should clear. The animation loop runs `drawStrengthCurve` which
    // sets `_curveTransition = null` once `t >= 1`.
    await page.waitForFunction(
      () => (window as any).__test.isCurveTransitionActive === false,
      null,
      { timeout: 1500 },
    );

    // Sanity: the toggle visibly changed the curve.
    const after = await curve.screenshot();
    expect(
      Buffer.compare(before, after),
      "post-toggle canvas must differ from pre-toggle canvas",
    ).not.toBe(0);
  });
});

test.describe("control-scoped predictive preview", () => {
  test("slider hover previews one dimension with one marker pinned to the pointer", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse hover is desktop-only");
    await openPreviewTestPage(page);

    const sliders = page.locator("#sliders input[type=range]");
    const slider = sliders.nth(1);
    const box = await slider.boundingBox();
    if (!box) throw new Error("slider has no bounding box");
    const pointerX = box.x + box.width * 0.78;
    const panel = page.locator("#sliders-panel");
    const baselineShadow = await panel.evaluate((el) => getComputedStyle(el).boxShadow);

    await page.mouse.move(pointerX, box.y + box.height / 2);
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("slider");
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewScopeIdx)).toBe(
      Number(await slider.getAttribute("data-idx")),
    );

    const state = await page.evaluate(() => {
      const t = (window as any).__test;
      const changed = t.displayPreviewComp
        .map((value: number, idx: number) => Math.abs(value - t.currentComposition[idx]) > 1e-6)
        .reduce((indices: number[], differs: boolean, idx: number) => {
          if (differs) indices.push(idx);
          return indices;
        }, []);
      const visible = Array.from(document.querySelectorAll(".slider-preview-marker"))
        .filter((marker) => getComputedStyle(marker).display !== "none");
      const marker = visible[0]?.getBoundingClientRect();
      return {
        changed,
        visibleMarkers: visible.length,
        markerCenterX: marker ? marker.left + marker.width / 2 : null,
        sharesGrid: t.previewSharesMainGrid,
      };
    });
    expect(state.changed).toEqual([Number(await slider.getAttribute("data-idx"))]);
    expect(state.visibleMarkers).toBe(1);
    expect(state.markerCenterX).not.toBeNull();
    expect(Math.abs(state.markerCenterX! - pointerX)).toBeLessThanOrEqual(2);
    expect(state.sharesGrid).toBe(true);
    await expect.poll(() => panel.evaluate((el) => getComputedStyle(el).boxShadow)).not.toBe(
      baselineShadow,
    );

    await page.evaluate(() => {
      (window as any).__visibleMarkerCounts = [];
      const sample = () => {
        (window as any).__visibleMarkerCounts.push(
          Array.from(document.querySelectorAll(".slider-preview-marker"))
            .filter((marker) => getComputedStyle(marker).display !== "none").length,
        );
        if ((window as any).__test.previewSource !== null ||
            (window as any).__test.displayPreviewComp.some(
              (value: number, idx: number) =>
                Math.abs(value - (window as any).__test.currentComposition[idx]) > 1e-6,
            )) {
          requestAnimationFrame(sample);
        }
      };
      requestAnimationFrame(sample);
    });
    await page.mouse.move(box.x - 20, box.y + box.height / 2);
    await waitForPreviewToSettle(page);
    const exitCounts: number[] = await page.evaluate(() => (window as any).__visibleMarkerCounts);
    expect(exitCounts.length, "expected marker samples during preview revert").toBeGreaterThan(0);
    expect(Math.max(...exitCounts), "slider exit flashed markers on other controls").toBeLessThanOrEqual(1);
  });

  test("starting a drag cancels the hover preview before values commit", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse drag is desktop-only");
    await openPreviewTestPage(page);
    const slider = page.locator("#sliders input[type=range]").first();
    const idx = Number(await slider.getAttribute("data-idx"));
    const box = await slider.boundingBox();
    if (!box) throw new Error("slider has no bounding box");

    await page.mouse.move(box.x + box.width * 0.8, box.y + box.height / 2);
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("slider");
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.3, box.y + box.height / 2, { steps: 3 });

    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBeNull();
    await expect.poll(() => page.evaluate((scopeIdx) => {
      const t = (window as any).__test;
      return Math.abs(t.displayPreviewComp[scopeIdx] - t.currentComposition[scopeIdx]);
    }, idx)).toBeLessThanOrEqual(1e-6);
    await page.mouse.up();
  });

  test("slider input commits before its rAF-bounded curve retarget", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "one project is enough");
    await openPreviewTestPage(page);

    const observed = await page.evaluate(() => {
      const slider = document.querySelector("#sliders input[type=range]") as HTMLInputElement;
      const idx = Number(slider.dataset.idx);
      const next = Math.min(Number(slider.max), Number(slider.value) + Number(slider.step || 1));
      slider.value = String(next);
      slider.dispatchEvent(new Event("input", { bubbles: true }));
      return {
        composition: (window as any).__test.currentComposition[idx],
        transitionActive: (window as any).__test.isCurveTransitionActive,
        value: next,
      };
    });

    expect(observed.composition).toBe(observed.value);
    expect(observed.transitionActive).toBe(false);
    await expect.poll(
      () => page.evaluate(() => (window as any).__test.isCurveTransitionActive),
      { timeout: 500 },
    ).toBe(true);
  });

  test("slider commit transitions the posterior mean and uncertainty", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse drag is desktop-only");
    await openPreviewTestPage(page);
    const slider = page.locator("#sliders input[type=range]").first();
    const box = await slider.boundingBox();
    if (!box) throw new Error("slider has no bounding box");

    await page.mouse.move(box.x + box.width * 0.8, box.y + box.height / 2);
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("slider");
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.25, box.y + box.height / 2, { steps: 4 });

    await expect.poll(
      () => page.evaluate(() => (window as any).__test.isCurveTransitionActive),
      { timeout: 500 },
    ).toBe(true);
    const delta = await page.evaluate(() => (window as any).__test.curveTransitionEndpointDelta);
    expect(delta.mean, "slider transition mean endpoints should differ").toBeGreaterThan(1e-3);
    expect(delta.std, "slider transition uncertainty endpoints should differ").toBeGreaterThan(1e-3);
    await page.mouse.up();
    await expect.poll(
      () => page.evaluate(() => (window as any).__test.isCurveTransitionActive),
      { timeout: 1500 },
    ).toBe(false);
  });

  test("slider commit never draws the new posterior before its transition begins", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse drag is desktop-only");
    await openPreviewTestPage(page);
    const slider = page.locator("#sliders input[type=range]").first();
    const box = await slider.boundingBox();
    if (!box) throw new Error("slider has no bounding box");

    await page.mouse.move(box.x + box.width * 0.85, box.y + box.height / 2);
    await page.mouse.down();
    const modes = await page.evaluate(async () => {
      const t = (window as any).__test;
      const range = document.querySelector("#sliders input[type=range]") as HTMLInputElement;
      const startSequence = t.curveDrawSequence;
      range.value = String(Number(range.min) + (Number(range.max) - Number(range.min)) * 0.1);
      range.dispatchEvent(new Event("input", { bubbles: true }));
      const seen: string[] = [];
      let lastSequence = startSequence;
      for (let frame = 0; frame < 4; frame++) {
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        if (t.curveDrawSequence !== lastSequence) {
          seen.push(t.lastCurveRenderMode);
          lastSequence = t.curveDrawSequence;
        }
        if (t.isCurveTransitionActive) break;
      }
      return seen;
    });
    await page.mouse.up();

    expect(modes.length, "no curve draw observed after slider input").toBeGreaterThan(0);
    expect(modes).not.toContain("direct");
  });

  test("lost pointer capture disarms slider preview commits", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse gesture is desktop-only");
    await openPreviewTestPage(page);
    const slider = page.locator("#sliders input[type=range]").first();
    const box = await slider.boundingBox();
    if (!box) throw new Error("slider has no bounding box");

    await page.mouse.move(box.x + box.width * 0.8, box.y + box.height / 2);
    await slider.evaluate((range: HTMLInputElement) => {
      range.dispatchEvent(new PointerEvent("pointerdown", {
        bubbles: true,
        pointerType: "mouse",
      }));
      range.value = String(Math.max(Number(range.min), Number(range.value) - Number(range.step || 1)));
      range.dispatchEvent(new Event("input", { bubbles: true }));
      range.dispatchEvent(new PointerEvent("lostpointercapture", {
        bubbles: true,
        pointerType: "mouse",
      }));
    });
    await expect.poll(
      () => page.evaluate(() => (window as any).__test.isCurveTransitionActive),
      { timeout: 500 },
    ).toBe(true);
    expect(await page.evaluate(() => (window as any).__test.isPreviewVisualHeld)).toBe(false);
  });

  test("slider commit holds its approved preview through the solid transition", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse drag is desktop-only");
    await openPreviewTestPage(page);
    const slider = page.locator("#sliders input[type=range]").first();
    const box = await slider.boundingBox();
    if (!box) throw new Error("slider has no bounding box");

    await page.mouse.move(box.x + box.width * 0.8, box.y + box.height / 2);
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("slider");
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.25, box.y + box.height / 2, { steps: 4 });
    await page.mouse.up();

    await expect.poll(
      () => page.evaluate(() => (window as any).__test.isCurveTransitionActive),
      { timeout: 500 },
    ).toBe(true);
    expect(await page.evaluate(() => (window as any).__test.isPreviewVisualHeld)).toBe(true);
    await expect.poll(
      () => page.evaluate(() => (window as any).__test.isCurveTransitionActive),
      { timeout: 1500 },
    ).toBe(false);
    expect(await page.evaluate(() => (window as any).__test.isPreviewVisualHeld)).toBe(false);
  });

  test("reduced-motion slider commit clears held preview before parking", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse gesture is desktop-only");
    await page.emulateMedia({ reducedMotion: "reduce" });
    await openPreviewTestPage(page);
    const slider = page.locator("#sliders input[type=range]").first();
    const box = await slider.boundingBox();
    if (!box) throw new Error("slider has no bounding box");

    await page.mouse.move(box.x + box.width * 0.8, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.2, box.y + box.height / 2);
    await page.mouse.up();
    await expect.poll(() => page.evaluate(() => {
      const t = (window as any).__test;
      return !t.isCurveTransitionActive && !t.isAnimLoopActive;
    })).toBe(true);
    expect(await page.evaluate(() => (window as any).__test.isPreviewVisualHeld)).toBe(false);
    expect(await page.evaluate(() => (window as any).__test.lastDrawHadPreview)).toBe(false);
  });

  test("material class preview morphs between valid endpoint means", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "Material Source controls are desktop-visible");
    await openPreviewTestPage(page);
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();

    await inactive.hover();

    await expect.poll(
      () => page.evaluate(() => (window as any).__test.isPreviewCurveTransitionActive),
      { timeout: 200 },
    ).toBe(true);
    expect(
      await page.evaluate(() => (window as any).__test.previewCurveTransitionEndpointDelta),
      "class preview endpoint means should differ",
    ).toBeGreaterThan(1e-3);
    const state = await page.evaluate(() => {
      const t = (window as any).__test;
      return { source: t.previewSource, materialSource: t.displayPreviewComp[7] };
    });
    expect(state.source).toBe("class");
    expect(Number.isInteger(state.materialSource)).toBe(true);
  });

  test("committing a previewed class holds its approved target through the solid transition", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "Material Source controls are desktop-visible");
    await openPreviewTestPage(page);
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await inactive.hover();
    await expect.poll(
      () => page.evaluate(() => !(window as any).__test.isPreviewCurveTransitionActive),
      { timeout: 1500 },
    ).toBe(true);

    await inactive.click();

    await expect.poll(() => page.evaluate(() => (window as any).__test.isCurveTransitionActive)).toBe(true);
    expect(await page.evaluate(() => (window as any).__test.isPreviewVisualHeld)).toBe(true);
    await expect.poll(
      () => page.evaluate(() => (window as any).__test.isCurveTransitionActive),
      { timeout: 1500 },
    ).toBe(false);
    expect(await page.evaluate(() => (window as any).__test.isPreviewVisualHeld)).toBe(false);
  });

  test("a new class preview supersedes a held commit target", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "Material Source controls are desktop-visible");
    await openPreviewTestPage(page);
    const firstCandidate = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    const firstLabel = (await firstCandidate.textContent())?.trim() ?? "";
    await firstCandidate.hover();
    await page.getByRole("button", { name: firstLabel, exact: true }).click();
    await expect.poll(() => page.evaluate(() => (window as any).__test.isPreviewVisualHeld)).toBe(true);

    const second = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await second.hover();

    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("class");
    expect(await page.evaluate(() => (window as any).__test.isPreviewVisualHeld)).toBe(false);
  });

  test("class preview rebases at most once per frame during slider input", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "one project is enough");
    await openPreviewTestPage(page);
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await inactive.focus();

    const counts = await page.evaluate(async () => {
      const t = (window as any).__test;
      const slider = document.querySelector("#sliders input[type=range]") as HTMLInputElement;
      const before = t.previewCurveTransitionCount;
      for (let i = 0; i < 5; i++) {
        slider.value = String(Math.min(
          Number(slider.max),
          Number(slider.value) + Number(slider.step || 1),
        ));
        slider.dispatchEvent(new Event("input", { bubbles: true }));
      }
      const immediate = t.previewCurveTransitionCount;
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      return { before, immediate, afterFrame: t.previewCurveTransitionCount };
    });

    expect(counts.immediate).toBe(counts.before);
    expect(counts.afterFrame - counts.before).toBeLessThanOrEqual(1);
  });

  test("queued class rebase cannot overwrite a new class owner", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "one project is enough");
    await openPreviewTestPage(page);
    const classes = page.locator(".material-source-group .toggle-btn:not(.active)");
    const first = classes.nth(0);
    const second = classes.nth(1);
    await first.focus();
    const secondLabel = (await second.textContent())?.trim() ?? "";

    const result = await page.evaluate(async (label) => {
      const slider = document.querySelector("#sliders input[type=range]") as HTMLInputElement;
      slider.value = String(Math.min(
        Number(slider.max),
        Number(slider.value) + Number(slider.step || 1),
      ));
      slider.dispatchEvent(new Event("input", { bubbles: true }));
      const nextOwner = Array.from(document.querySelectorAll<HTMLButtonElement>(
        ".material-source-group .toggle-btn",
      )).find((button) => button.textContent?.trim() === label)!;
      nextOwner.focus();
      const expectedSequence = (window as any).__test.previewCurveTransitionSequence;
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const t = (window as any).__test;
      return {
        expectedSequence,
        finalSequence: t.previewCurveTransitionSequence,
        owner: t.previewScopeBtnLabel?.trim(),
      };
    }, secondLabel);

    expect(result.owner).toBe(secondLabel);
    expect(result.finalSequence).toBe(result.expectedSequence);
  });

  test("slider-to-slider handoff never previews two dimensions", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse hover is desktop-only");
    await openPreviewTestPage(page);
    const sliders = page.locator("#sliders input[type=range]");
    const first = sliders.nth(0);
    const second = sliders.nth(1);
    const secondIdx = Number(await second.getAttribute("data-idx"));
    const firstBox = await first.boundingBox();
    const secondBox = await second.boundingBox();
    if (!firstBox || !secondBox) throw new Error("slider has no bounding box");

    await page.mouse.move(firstBox.x + firstBox.width * 0.8, firstBox.y + firstBox.height / 2);
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("slider");
    await expect.poll(() => page.evaluate(() => {
      const t = (window as any).__test;
      return t.displayPreviewComp.some(
        (value: number, idx: number) => Math.abs(value - t.currentComposition[idx]) > 1e-6,
      );
    })).toBe(true);

    await page.mouse.move(secondBox.x + secondBox.width * 0.2, secondBox.y + secondBox.height / 2);
    const changed = await page.evaluate(() => {
      const t = (window as any).__test;
      return t.displayPreviewComp.flatMap((value: number, idx: number) =>
        Math.abs(value - t.currentComposition[idx]) > 1e-6 ? [idx] : []
      );
    });
    expect(changed).toEqual([secondIdx]);
  });

  test("class preview keeps keyboard focus ownership after pointer leave", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse and focus interaction is desktop-only");
    await openPreviewTestPage(page);
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await inactive.focus();
    await inactive.hover();
    await expect(inactive).toHaveAttribute("data-previewing", "");

    await page.mouse.move(0, 0);

    await expect(inactive).toBeFocused();
    expect(await page.evaluate(() => (window as any).__test.previewSource)).toBe("class");
    await expect(inactive).toHaveAttribute("data-previewing", "");
    await inactive.blur();
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBeNull();
  });

  test("class preview rebases when a slider commits", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "keyboard interaction is desktop-only");
    await openPreviewTestPage(page);
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await inactive.hover();
    const initialPreviewTransition = await page.evaluate(() =>
      (window as any).__test.previewCurveTransitionSequence);
    const slider = page.locator("#sliders input[type=range]").first();
    await slider.focus();
    await page.keyboard.press("ArrowRight");
    await expect.poll(() => page.evaluate(() =>
      (window as any).__test.previewCurveTransitionSequence)).not.toBe(initialPreviewTransition);
    await page.evaluate(async () => {
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    });

    await expect.poll(() => page.evaluate(() => {
      const t = (window as any).__test;
      return t.displayPreviewComp.flatMap((value: number, idx: number) =>
        Math.abs(value - t.currentComposition[idx]) > 1e-6 ? [idx] : []
      );
    })).toEqual([7]);
  });

  test("focused class preview returns after temporary pointer ownership", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse and focus interaction is desktop-only");
    await openPreviewTestPage(page);
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)");
    const focused = inactive.nth(0);
    const hovered = inactive.nth(1);
    const focusedLabel = (await focused.textContent())?.trim() ?? "";
    await focused.focus();
    await hovered.hover();
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewScopeBtnLabel)).not.toBe(
      focusedLabel,
    );

    await page.mouse.move(0, 0);

    await expect(focused).toBeFocused();
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("class");
    expect(await page.evaluate(() => (window as any).__test.previewScopeBtnLabel)).toBe(focusedLabel);
    await expect(focused).toHaveAttribute("data-previewing", "");
  });

  test("stationary class preview parks canvas redraws and restarts for a new owner", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "one project is enough");
    await openPreviewTestPage(page);
    await page.evaluate(() => {
      (window as any).__curveDrawCount = 0;
      const canvas = document.getElementById("curve-canvas") as HTMLCanvasElement;
      let width = canvas.width;
      Object.defineProperty(canvas, "width", {
        get: () => width,
        set: (value) => { (window as any).__curveDrawCount++; width = value; },
        configurable: true,
      });
    });

    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)");
    await inactive.nth(0).hover();
    const parkedCount = await page.evaluate(async () => {
      let last = (window as any).__curveDrawCount;
      let stableFrames = 0;
      for (let frame = 0; frame < 240; frame++) {
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        const count = (window as any).__curveDrawCount;
        stableFrames = count === last ? stableFrames + 1 : 0;
        last = count;
        if (stableFrames >= 5) return count;
      }
      return null;
    });
    expect(parkedCount, "preview loop never parked after convergence").not.toBeNull();

    await inactive.nth(1).hover();
    await expect.poll(() => page.evaluate(() => (window as any).__curveDrawCount)).toBeGreaterThan(
      parkedCount!,
    );
  });

  test("moving from a focused class button to a slider transfers the affordance", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse hover is desktop-only");
    await openPreviewTestPage(page);
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await inactive.focus();
    await expect(inactive).toHaveAttribute("data-previewing", "");

    const slider = page.locator("#sliders input[type=range]").first();
    const box = await slider.boundingBox();
    if (!box) throw new Error("slider has no bounding box");
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);

    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("slider");
    await expect(inactive).not.toHaveAttribute("data-previewing", "");
    expect(await page.evaluate(() => (window as any).__test.previewScopeBtnLabel)).toBeNull();
    expect(
      await page.locator(".slider-preview-marker").evaluateAll((markers) =>
        markers.filter((marker) => getComputedStyle(marker).display !== "none").length
      ),
    ).toBe(1);
  });

  test("focused class restoration waits for scatter composition completion", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse and focus interaction is desktop-only");
    await openPreviewTestPage(page);
    const focused = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await focused.focus();
    const scatter = await hoverRenderedScatterPoint(page);
    await expect(focused).toBeFocused();
    await scatter.evaluate((canvas: HTMLCanvasElement) => canvas.click());
    const restoredTooEarly = await page.evaluate(async () => {
      let violation = false;
      await new Promise<void>((resolve) => {
        const sample = () => {
          const t = (window as any).__test;
          if (t.isCompositionTransitionActive && t.previewSource === "class") {
            violation = true;
          }
          if (t.isCompositionTransitionActive) {
            requestAnimationFrame(sample);
          } else {
            resolve();
          }
        };
        requestAnimationFrame(sample);
      });
      return violation;
    });
    expect(restoredTooEarly).toBe(false);
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("class");
  });

  test("scatter commit clears preview ownership and markers without pointer movement", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "scatter hover is desktop-only");
    await openPreviewTestPage(page);
    const scatter = await hoverRenderedScatterPoint(page);
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("scatter");

    await scatter.click();

    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBeNull();
    expect(
      await page.locator(".slider-preview-marker").evaluateAll((markers) =>
        markers.filter((marker) => getComputedStyle(marker).display !== "none").length
      ),
    ).toBe(0);
    await expect(page.locator("#sliders-panel")).not.toHaveClass(/\bpreviewing\b/);
  });

  test("empty scatter space restores a still-focused class preview", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse and focus interaction is desktop-only");
    await openPreviewTestPage(page);
    const focused = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await focused.focus();
    const scatter = await hoverRenderedScatterPoint(page);
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("scatter");
    const box = await scatter.boundingBox();
    if (!box) throw new Error("scatter canvas has no bounding box");

    await page.mouse.move(box.x + 4, box.y + 4);

    await expect(focused).toBeFocused();
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("class");
    await expect(focused).toHaveAttribute("data-previewing", "");
  });

  test("scatter leave restores a still-focused class preview", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse and focus interaction is desktop-only");
    await openPreviewTestPage(page);
    const focused = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    const label = (await focused.textContent())?.trim() ?? "";
    await focused.focus();
    await hoverRenderedScatterPoint(page);
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("scatter");

    await page.mouse.move(0, 0);

    await expect(focused).toBeFocused();
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("class");
    expect(await page.evaluate(() => (window as any).__test.previewScopeBtnLabel)).toBe(label);
    await expect(focused).toHaveAttribute("data-previewing", "");
  });

  test("slider commit interrupts a stale scatter composition animation", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse interaction is desktop-only");
    await openPreviewTestPage(page);
    const scatter = await hoverRenderedScatterPoint(page);
    await scatter.click();
    await expect.poll(
      () => page.evaluate(() => (window as any).__test.isCompositionTransitionActive),
      { timeout: 300 },
    ).toBe(true);

    const slider = page.locator("#sliders input[type=range]").first();
    const idx = Number(await slider.getAttribute("data-idx"));
    await slider.focus();
    await page.keyboard.press("ArrowRight");
    const committed = await page.evaluate((scopeIdx) =>
      (window as any).__test.currentComposition[scopeIdx], idx);

    await expect.poll(
      () => page.evaluate(() => (window as any).__test.isCompositionTransitionActive),
      { timeout: 1000 },
    ).toBe(false);
    await page.evaluate(async () => {
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    });
    expect(await page.evaluate((scopeIdx) =>
      (window as any).__test.currentComposition[scopeIdx], idx)).toBeCloseTo(committed, 8);
  });

  test("scatter commit cancels a queued slider curve retarget", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse interaction is desktop-only");
    await openPreviewTestPage(page);
    const scatter = await hoverRenderedScatterPoint(page);
    const slider = page.locator("#sliders input[type=range]").first();

    const result = await page.evaluate(() => {
      const range = document.querySelector("#sliders input[type=range]") as HTMLInputElement;
      const max = Number(range.max);
      const step = Number(range.step || 1);
      range.value = String(Math.min(max, Number(range.value) + step));
      range.dispatchEvent(new Event("input", { bubbles: true }));
      range.value = String(Math.min(max, Number(range.value) + step));
      range.dispatchEvent(new Event("input", { bubbles: true }));
      (document.getElementById("scatter-canvas") as HTMLCanvasElement).click();
      const target = (window as any).__test.compositionTransitionTarget;
      return { target };
    });

    await page.evaluate(async () => {
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    });
    expect(await page.evaluate(() => {
      const t = (window as any).__test;
      return t.curveTransitionTarget?.every(
        (value: number, idx: number) => Math.abs(value - t.compositionTransitionTarget[idx]) < 1e-6,
      ) ?? false;
    })).toBe(true);
    expect(result.target).not.toBeNull();
  });

  test("clicking the pinned scatter class still cancels the composition animation", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse interaction is desktop-only");
    await openPreviewTestPage(page);
    await hoverRenderedScatterPoint(page);
    const afterClick = await page.evaluate(async () => {
      const canvas = document.getElementById("scatter-canvas") as HTMLCanvasElement;
      canvas.click();
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const t = (window as any).__test;
      const beforeTransition = t.curveTransitionSequence;
      const currentClass = Math.round(t.currentComposition[7]);
      const button = document.querySelectorAll<HTMLButtonElement>(
        ".material-source-group .toggle-btn",
      )[currentClass];
      button.click();
      return {
        beforeTransition,
        composition: t.currentComposition,
        animationActive: t.isCompositionTransitionActive,
        transitionSequence: t.curveTransitionSequence,
        transitionTarget: t.curveTransitionTarget,
      };
    });

    const beforeTransition = afterClick.beforeTransition;

    expect(afterClick.animationActive).toBe(false);
    expect(afterClick.transitionSequence).not.toBe(beforeTransition);
    expect(afterClick.transitionTarget).toEqual(afterClick.composition);
  });

  test("class commit interrupts a stale scatter composition animation", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse interaction is desktop-only");
    await openPreviewTestPage(page);
    const scatter = await hoverRenderedScatterPoint(page);
    await scatter.click();
    await expect.poll(
      () => page.evaluate(() => (window as any).__test.isCompositionTransitionActive),
      { timeout: 300 },
    ).toBe(true);

    const buttons = page.locator(".material-source-group .toggle-btn");
    const currentClass = await page.evaluate(() =>
      Math.round((window as any).__test.currentComposition[7]));
    const inactive = buttons.nth((currentClass + 1) % await buttons.count());
    await inactive.click();
    const committedClass = await page.evaluate(() =>
      (window as any).__test.currentComposition[7]);

    await expect.poll(
      () => page.evaluate(() => (window as any).__test.isCompositionTransitionActive),
      { timeout: 500 },
    ).toBe(false);
    await page.evaluate(async () => {
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    });
    expect(await page.evaluate(() => (window as any).__test.currentComposition[7])).toBe(committedClass);
    await expect(inactive).toHaveClass(/\bactive\b/);
  });

  test("touch pointer movement over a slider is preview-inert", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop DOM keeps the slider visible");
    await openPreviewTestPage(page);
    const slider = page.locator("#sliders input[type=range]").first();
    await slider.dispatchEvent("pointermove", {
      pointerType: "touch",
      clientX: 100,
      clientY: 100,
      buttons: 0,
    });
    expect(await page.evaluate(() => (window as any).__test.previewSource)).toBeNull();
  });

  test("clicking a previewed material class commits and ends the preview", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse hover is desktop-only");
    await openPreviewTestPage(page);
    const inactiveCandidate = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    const label = (await inactiveCandidate.textContent())?.trim() ?? "";
    const inactive = page.getByRole("button", { name: label, exact: true });
    await inactive.hover();
    await expect(inactive).toHaveAttribute("data-previewing", "");

    await inactive.click();

    await expect(inactive).toHaveClass(/\bactive\b/);
    expect(await page.evaluate(() => (window as any).__test.previewSource)).toBeNull();
    await expect(inactive).not.toHaveAttribute("data-previewing", "");
  });

  test("class preview return restores the idle-quality curve grid", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "Material Source controls are desktop-visible");
    await openPreviewTestPage(page);
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    await inactive.hover();
    await page.mouse.move(0, 0);

    await expect.poll(() => page.evaluate(() => {
      const t = (window as any).__test;
      return !t.isPreviewCurveTransitionActive && !t.isAnimLoopActive;
    }), { timeout: 1500 }).toBe(true);
    expect(await page.evaluate(() => (window as any).__test.mainCurvePointCount)).toBe(64);
  });

  test("inactive material class hover previews only that integer class", async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "mouse hover is desktop-only");
    await openPreviewTestPage(page);
    const active = page.locator(".material-source-group .toggle-btn.active");
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();

    await active.hover();
    expect(await page.evaluate(() => (window as any).__test.previewSource)).toBeNull();

    const label = (await inactive.textContent())?.trim() ?? "";
    const buttonIndex = await page.locator(".material-source-group .toggle-btn").allTextContents()
      .then((labels) => labels.findIndex((text) => text.trim() === label));
    await inactive.hover();
    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("class");
    expect(await page.evaluate(() => (window as any).__test.previewScopeBtnLabel)).toBe(label);
    await expect(inactive).toHaveAttribute("data-previewing", "");

    const state = await page.evaluate(async () => {
      const columns = await (await fetch("model/compositions.json")).json();
      const msIdx = columns.column_names.indexOf("Material Source");
      const t = (window as any).__test;
      return {
        msIdx,
        current: t.currentComposition,
        preview: t.displayPreviewComp,
        visibleMarkers: Array.from(document.querySelectorAll(".slider-preview-marker"))
          .filter((marker) => getComputedStyle(marker).display !== "none").length,
      };
    });
    expect(state.msIdx).toBeGreaterThanOrEqual(0);
    expect(Number.isInteger(state.preview[state.msIdx])).toBe(true);
    expect(state.preview[state.msIdx]).toBe(buttonIndex);
    expect(state.preview[state.msIdx]).not.toBe(state.current[state.msIdx]);
    for (let idx = 0; idx < state.preview.length; idx++) {
      if (idx !== state.msIdx) expect(state.preview[idx]).toBe(state.current[idx]);
    }
    expect(state.visibleMarkers).toBe(0);
  });
});

test.describe("hover preview aligns with the committed prediction", () => {
  // Regression: the dashed hover-preview curve visibly disagreed with the
  // solid curve for the same mix. Two independent causes, both guarded here.

  test("preview curve is drawn on the same time grid as the main curve", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "scatter hover is a desktop interaction");
    // Cause 1: the preview used a fixed 48-point grid while the main curve
    // used 32 points during any interaction. Both curves were numerically
    // correct, but overlaying polylines sampled at different times left a
    // ~17% gap at t=0.07 d, in the steep gate-opening region. Sampling both
    // on one grid makes the gap identically zero at any resolution.
    await page.goto("/?test=1");
    await page.waitForFunction(() => typeof (window as any).__test !== "undefined");
    // The shell renders before the GP finishes building in the worker, so
    // wait for the model itself before asserting on predictions.
    await page.waitForFunction(() => (window as any).__test.modelReady === true, null, { timeout: 20000 });
    await page.waitForTimeout(800);

    const canvas = page.locator("canvas#scatter-canvas");
    await expect(canvas).toBeVisible();
    const box = await canvas.boundingBox();
    if (!box) throw new Error("scatter canvas has no bounding box");

    // Sweep the canvas so several points get hovered; record the verdict on
    // every frame where a preview was actually drawn.
    await page.evaluate(() => {
      (window as any).__gridSamples = [];
      const tick = () => {
        const v = (window as any).__test.previewSharesMainGrid;
        if (v !== null) (window as any).__gridSamples.push(v);
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });

    for (const fx of [0.3, 0.45, 0.6, 0.75]) {
      await canvas.hover({ position: { x: box.width * fx, y: box.height * 0.5 } });
      await page.waitForTimeout(180);
    }

    const samples: boolean[] = await page.evaluate(() => (window as any).__gridSamples);
    expect(samples.length, "expected the preview curve to be drawn at least once").toBeGreaterThan(0);
    expect(
      samples.every((v) => v === true),
      `preview used a different grid from the main curve on ${samples.filter((v) => !v).length}/${samples.length} frames`,
    ).toBe(true);
  });

  test("preview never feeds the GP a fractional Material Source", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "scatter hover is a desktop interaction");
    // Cause 2: displayPreviewComp lerped every dimension toward the hover
    // target, including the categorical Material Source. The preview curve is
    // predicted straight from displayPreviewComp, so for the ~1.4 s the lerp
    // took to converge the Hamming kernel saw a fractional class, which
    // matches no training row: the ghost rendered a collapsed "unseen class"
    // posterior (-3%) and passed through the neighbouring real class (+19%)
    // en route. Material Source must be snapped, never interpolated.
    const msIdx = await page.evaluate(async () => {
      const r = await fetch("model/compositions.json");
      const j = await r.json();
      return j.column_names.indexOf("Material Source");
    }).catch(() => -1);

    await page.goto("/?test=1");
    await page.waitForFunction(() => typeof (window as any).__test !== "undefined");
    // The shell renders before the GP finishes building in the worker, so
    // wait for the model itself before asserting on predictions.
    await page.waitForFunction(() => (window as any).__test.modelReady === true, null, { timeout: 20000 });
    await page.waitForTimeout(800);

    const idx = msIdx >= 0 ? msIdx : await page.evaluate(async () => {
      const r = await fetch("model/compositions.json");
      const j = await r.json();
      return j.column_names.indexOf("Material Source");
    });
    expect(idx, "Material Source column not found").toBeGreaterThanOrEqual(0);

    const canvas = page.locator("canvas#scatter-canvas");
    await expect(canvas).toBeVisible();
    const box = await canvas.boundingBox();
    if (!box) throw new Error("scatter canvas has no bounding box");

    await page.evaluate((msCol) => {
      (window as any).__msSamples = [];
      const tick = () => {
        const c = (window as any).__test.displayPreviewComp;
        if (c) (window as any).__msSamples.push(c[msCol]);
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }, idx);

    // Sweep across the scatter so the hover target crosses material classes.
    for (const fx of [0.2, 0.4, 0.55, 0.7, 0.85]) {
      await canvas.hover({ position: { x: box.width * fx, y: box.height * 0.5 } });
      await page.waitForTimeout(160);
    }

    const samples: number[] = await page.evaluate(() => (window as any).__msSamples);
    expect(samples.length, "expected displayPreviewComp samples").toBeGreaterThan(10);
    const fractional = samples.filter((v) => !Number.isInteger(v));
    expect(
      fractional.length,
      `Material Source was fractional on ${fractional.length}/${samples.length} frames, e.g. ${fractional.slice(0, 5).join(", ")}`,
    ).toBe(0);
    // Sanity: the sweep actually visited more than one class, otherwise the
    // assertion above is vacuous.
    expect(new Set(samples).size, "sweep never crossed a material class boundary").toBeGreaterThan(1);
  });
});
