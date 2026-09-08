import { expect, test, type Page } from "@playwright/test";
import { waitForDashboardLayoutReady } from "./dashboard-helpers";

async function openDashboard(page: Page) {
  await page.goto("/?test=1");
  await waitForDashboardLayoutReady(page);
  await expect(page.locator("#filter-add")).toBeVisible();
}

async function waitForFilterMotion(page: Page) {
  await expect.poll(() => page.locator("#filter-rows").evaluate((shell) =>
    shell.getAnimations({ subtree: true }).filter((animation) => animation.playState === "running").length,
  )).toBe(0);
}

async function sampleTransaction(page: Page, selector: string) {
  return page.evaluate(async (triggerSelector) => {
    const shell = document.querySelector<HTMLElement>("#filter-rows")!;
    const panel = document.querySelector<HTMLElement>("#filters-panel")!;
    const follower = document.querySelector<HTMLElement>("#references-panel")!;
    const trigger = document.querySelector<HTMLElement>(triggerSelector)!;
    const samples: Array<{
      time: number;
      shellHeight: number;
      panelHeight: number;
      followerTop: number;
      rowTops: number[];
    }> = [];
    const read = (time: number) => {
      const shellRect = shell.getBoundingClientRect();
      const panelRect = panel.getBoundingClientRect();
      const followerRect = follower.getBoundingClientRect();
      samples.push({
        time,
        shellHeight: shellRect.height,
        panelHeight: panelRect.height,
        followerTop: followerRect.top + scrollY,
        rowTops: [...shell.querySelectorAll<HTMLElement>(".filter-row-wrapper")]
          .map((row) => row.getBoundingClientRect().top + scrollY),
      });
    };

    read(performance.now());
    trigger.click();
    for (let frame = 0, settledFrames = 0; frame < 120 && settledFrames < 3; frame += 1) {
      const frameTime = await new Promise<number>((resolve) => requestAnimationFrame(resolve));
      read(frameTime);
      const running = shell.getAnimations({ subtree: true })
        .some((animation) => animation.playState === "running");
      settledFrames = running ? 0 : settledFrames + 1;
    }
    return samples;
  }, selector);
}

function expectContinuous(
  samples: Array<{ time: number; value: number }>,
  direction: "growing" | "shrinking",
  label: string,
) {
  expect(samples.length).toBeGreaterThan(3);
  for (let index = 1; index < samples.length; index += 1) {
    const delta = samples[index].value - samples[index - 1].value;
    const elapsed = Math.max(1, samples[index].time - samples[index - 1].time);
    if (direction === "growing") expect(delta, `${label} reversed at frame ${index}`).toBeGreaterThanOrEqual(-1);
    else expect(delta, `${label} reversed at frame ${index}`).toBeLessThanOrEqual(1);
    expect(Math.abs(delta) / elapsed, `${label} jumped at frame ${index}`).toBeLessThanOrEqual(0.55);
  }
}

function values(samples: Array<{ time: number }>, read: (sample: any) => number) {
  return samples.map((sample) => ({ time: sample.time, value: read(sample) }));
}

test.describe("filter structural motion", () => {
  test.beforeEach(async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop structural motion contract");
    await openDashboard(page);
  });

  test("zero-to-one and one-to-zero animate the complete shell continuously", async ({ page }) => {
    const insertion = await sampleTransaction(page, "#filter-add");
    expectContinuous(values(insertion, ({ panelHeight }) => panelHeight), "growing", "Filters panel insertion");
    expectContinuous(values(insertion, ({ followerTop }) => followerTop), "growing", "References insertion");
    await waitForFilterMotion(page);

    const removal = await sampleTransaction(page, ".filter-remove-btn");
    expectContinuous(values(removal, ({ panelHeight }) => panelHeight), "shrinking", "Filters panel removal");
    expectContinuous(values(removal, ({ followerTop }) => followerTop), "shrinking", "References removal");
    await expect(page.locator(".filter-row-wrapper")).toHaveCount(0);
    await expect(page.locator("#filter-add")).toBeFocused();
  });

  test("two-to-one keeps the surviving row continuous", async ({ page }) => {
    await page.locator("#filter-add").click();
    await waitForFilterMotion(page);
    await page.locator("#filter-add").click();
    await waitForFilterMotion(page);

    const samples = await sampleTransaction(page, ".filter-remove-btn");
    const survivorTops = values(samples, ({ rowTops }) => rowTops.at(-1));
    for (let index = 1; index < survivorTops.length; index += 1) {
      const elapsed = Math.max(1, survivorTops[index].time - survivorTops[index - 1].time);
      expect(Math.abs(survivorTops[index].value - survivorTops[index - 1].value) / elapsed,
        `surviving row jumped at frame ${index}`).toBeLessThanOrEqual(0.55);
    }
    expectContinuous(values(samples, ({ panelHeight }) => panelHeight), "shrinking", "two-to-one panel");
    await expect(page.locator(".filter-row-wrapper:not([data-exiting])")).toHaveCount(1);
  });

  test("rapid remove and add retarget from the painted frame", async ({ page }) => {
    await page.locator("#filter-add").click();
    await waitForFilterMotion(page);
    await page.locator("#filter-add").click();
    await waitForFilterMotion(page);

    await page.locator(".filter-remove-btn").first().click();
    const reversal = await page.locator("#filter-rows").evaluate(async (shell) => {
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const survivor = shell.querySelector<HTMLElement>(
        ".filter-row-wrapper:not([data-exiting])",
      )!;
      const paintedHeight = shell.getBoundingClientRect().height;
      const paintedSurvivorTop = survivor.getBoundingClientRect().top;
      document.querySelector<HTMLElement>("#filter-add")!.click();
      const retargetedHeight = shell.getBoundingClientRect().height;
      const retargetedSurvivorTop = survivor.getBoundingClientRect().top;
      return { paintedHeight, retargetedHeight, paintedSurvivorTop, retargetedSurvivorTop };
    });
    expect(
      Math.abs(reversal.retargetedHeight - reversal.paintedHeight),
      "superseding motion must preserve the currently painted shell geometry",
    ).toBeLessThanOrEqual(1);
    expect(
      Math.abs(reversal.retargetedSurvivorTop - reversal.paintedSurvivorTop),
      "superseding motion must preserve every surviving row position",
    ).toBeLessThanOrEqual(1);
    await waitForFilterMotion(page);
    await expect(page.locator(".filter-row-wrapper:not([data-exiting])")).toHaveCount(2);
    await expect(page.locator(".filter-row-wrapper[data-exiting]")).toHaveCount(0);
    expect(await page.locator("#filter-rows").evaluate((shell) => ({
      animations: shell.getAnimations({ subtree: true })
        .filter((animation) => animation.playState === "running" && animation.effect !== null).length,
      blockSize: (shell as HTMLElement).style.blockSize,
      paddingBlock: (shell as HTMLElement).style.paddingBlock,
    }))).toEqual({ animations: 0, blockSize: "", paddingBlock: "" });
  });

  test("rapid Add then Clear cancels insertion anchoring before removal", async ({ page }) => {
    const state = await page.locator("#filter-rows").evaluate(async (shell) => {
      const prototypeDescriptor = Object.getOwnPropertyDescriptor(Element.prototype, "scrollTop")!;
      Object.defineProperty(shell, "scrollTop", {
        configurable: true,
        get() {
          return prototypeDescriptor.get!.call(this);
        },
        set(value) {
          if ((this as HTMLElement).dataset.phase === "clear") {
            (this as HTMLElement).dataset.writesAfterClear = String(
              Number((this as HTMLElement).dataset.writesAfterClear ?? "0") + 1,
            );
          }
          prototypeDescriptor.set!.call(this, value);
        },
      });
      const clear = document.querySelector<HTMLElement>("#filter-clear")!;
      clear.addEventListener("click", () => {
        (shell as HTMLElement).dataset.phase = "clear";
      }, { capture: true, once: true });
      document.querySelector<HTMLElement>("#filter-add")!.click();
      const anchoringArmed = prototypeDescriptor.get!.call(shell) === 0;
      clear.click();
      for (let frame = 0; frame < 90; frame += 1) {
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        if (shell.getAnimations({ subtree: true }).every((animation) => animation.playState !== "running")) {
          break;
        }
      }
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      return {
        anchoringArmed,
        writesAfterClear: (shell as HTMLElement).dataset.writesAfterClear ?? null,
        activeRows: shell.querySelectorAll(".filter-row-wrapper:not([data-exiting])").length,
      };
    });
    expect(state).toEqual({ anchoringArmed: true, writesAfterClear: null, activeRows: 0 });
  });

  test("clear-all settles every row through one composite operation without writing scrollTop", async ({ page }) => {
    for (let count = 0; count < 3; count += 1) {
      await page.locator("#filter-add").click();
      await waitForFilterMotion(page);
    }
    await page.locator("#filter-rows").evaluate((shell) => {
      const ownDescriptor = Object.getOwnPropertyDescriptor(shell, "scrollTop");
      const prototype = Object.getPrototypeOf(shell);
      const prototypeDescriptor = Object.getOwnPropertyDescriptor(prototype, "scrollTop")
        ?? Object.getOwnPropertyDescriptor(Element.prototype, "scrollTop")!;
      Object.defineProperty(shell, "scrollTop", {
        configurable: true,
        get() {
          return ownDescriptor?.get?.call(this) ?? prototypeDescriptor.get!.call(this);
        },
        set(value) {
          (this as HTMLElement).dataset.applicationScrollWrites = String(
            Number((this as HTMLElement).dataset.applicationScrollWrites ?? "0") + 1,
          );
          if (ownDescriptor?.set) ownDescriptor.set.call(this, value);
          else prototypeDescriptor.set!.call(this, value);
        },
      });
    });
    await page.locator("#filter-clear").focus();
    const samples = await sampleTransaction(page, "#filter-clear");
    expectContinuous(values(samples, ({ panelHeight }) => panelHeight), "shrinking", "clear-all panel");
    await expect(page.locator(".filter-row-wrapper")).toHaveCount(0);
    await expect(page.locator("#filter-clear")).toBeFocused();
    expect(await page.locator("#filter-rows").getAttribute("data-application-scroll-writes")).toBeNull();
    expect(await page.locator("#filter-rows").evaluate((shell) => ({
      animations: shell.getAnimations({ subtree: true })
        .filter((animation) => animation.playState === "running" && animation.effect !== null).length,
      blockSize: (shell as HTMLElement).style.blockSize,
      paddingBlock: (shell as HTMLElement).style.paddingBlock,
      marginBlockStart: (shell as HTMLElement).style.marginBlockStart,
    }))).toEqual({ animations: 0, blockSize: "", paddingBlock: "", marginBlockStart: "" });
  });
});
