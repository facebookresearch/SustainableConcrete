import { expect, test, type Page } from "@playwright/test";
import {
  openMobileComposition,
  waitForDashboardLayoutReady,
} from "./dashboard-helpers";

async function openComposition(page: Page, projectName: string) {
  await page.goto("/");
  await waitForDashboardLayoutReady(page);
  if (projectName.startsWith("mobile")) {
    await openMobileComposition(page);
  }
  await expect(page.getByRole("button", { name: "Cement ingredient insight" })).toBeVisible();
}

test.describe("ingredient insight discovery", () => {
  test.beforeEach(async ({ page }, testInfo) => {
    await openComposition(page, testInfo.project.name);
  });

  test("starts with Cement selected and useful content visible", async ({ page }) => {
    const selected = page.locator('.ingredient-name[aria-pressed="true"]');

    await expect(selected).toHaveCount(1);
    await expect(selected).toHaveText("Cement");
    await expect(page.locator("#ingredient-insight-text")).toContainText(
      "Portland cement",
    );
    await expect(page.locator("#ingredient-insight-text")).not.toContainText(
      "Click an ingredient",
    );
  });

  test("desktop keeps the complete Composition panel visible in the first column", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop grid invariant");

    const panel = page.locator("#sliders-panel");
    await expect(panel).toBeVisible();
    await expect(panel.locator(".slider-group")).toHaveCount(9);
    await expect(panel.getByRole("button", { name: "Material Source ingredient insight" })).toBeVisible();
    await expect(panel.getByRole("button", { name: "Temperature ingredient insight" })).toBeVisible();

    const state = await page.evaluate(() => {
      const layout = document.querySelector(".layout");
      const compositionColumn = document.querySelector("#composition-column");
      const composition = document.querySelector("#sliders-panel");
      const scatter = document.querySelector("#scatter-canvas")?.closest(".chart-panel");
      const curve = document.querySelector("#curve-canvas")?.closest(".chart-panel");
      if (!layout || !compositionColumn || !composition || !scatter || !curve) return null;

      const rect = (element: Element) => {
        const value = element.getBoundingClientRect();
        return {
          left: value.left,
          top: value.top,
          right: value.right,
          bottom: value.bottom,
          width: value.width,
          height: value.height,
        };
      };
      const directChildren = Array.from(layout.children);
      const style = getComputedStyle(composition);
      const compositionRect = rect(composition);
      const foreground = document.elementFromPoint(
        compositionRect.left + compositionRect.width / 2,
        compositionRect.top + 24,
      );

      return {
        opacity: Number(style.opacity),
        animationName: style.animationName,
        composition: compositionRect,
        scatter: rect(scatter),
        curve: rect(curve),
        compositionChildIndex: directChildren.indexOf(compositionColumn),
        compositionScrollRange:
          composition.querySelector("#sliders")!.scrollHeight -
          composition.querySelector("#sliders")!.clientHeight,
        documentScrollRange:
          document.documentElement.scrollHeight -
          document.documentElement.clientHeight,
        scatterChildIndex: directChildren.findIndex((child) => child.contains(scatter)),
        curveChildIndex: directChildren.findIndex((child) => child.contains(curve)),
        compositionOwnsForegroundPoint: foreground !== null && composition.contains(foreground),
      };
    });

    expect(state, "all three desktop columns must exist").not.toBeNull();
    expect(state!.compositionChildIndex, "Composition must be the first desktop column").toBe(0);
    expect(state!.scatterChildIndex, "scatter must be the second desktop column").toBe(1);
    expect(state!.curveChildIndex, "strength curve must be the third desktop column").toBe(2);
    expect(state!.composition.right, "Composition must not overlap scatter").toBeLessThanOrEqual(state!.scatter.left);
    expect(state!.scatter.right, "scatter must not overlap strength curve").toBeLessThanOrEqual(state!.curve.left);
    expect(state!.composition.left).toBeLessThan(state!.scatter.left);
    expect(state!.scatter.left).toBeLessThan(state!.curve.left);
    expect(Math.abs(state!.composition.top - state!.scatter.top), "desktop columns must share a top edge").toBeLessThanOrEqual(1);
    expect(Math.abs(state!.scatter.top - state!.curve.top), "desktop columns must share a top edge").toBeLessThanOrEqual(1);
    expect(state!.compositionOwnsForegroundPoint, "Composition must not be visually covered").toBe(true);
    expect(state!.opacity, "Composition must be fully opaque after load").toBe(1);
    expect(state!.animationName, "primary controls must not depend on an entrance animation").toBe("none");
    expect(state!.composition.width, "Composition must not reserve unused desktop width").toBeLessThanOrEqual(282);
    expect(state!.composition.height).toBeGreaterThan(500);
    expect(state!.compositionScrollRange).toBeLessThanOrEqual(1);
    expect(state!.documentScrollRange).toBeGreaterThan(0);
  });

  test("Material Source buttons are compact, equal, and single-line", async ({ page }, testInfo) => {
    const buttons = page.locator(".material-source-group .toggle-btn");
    await expect(buttons).toHaveCount(3);
    const geometry = await buttons.evaluateAll((items) => items.map((item) => {
      const rect = item.getBoundingClientRect();
      const range = document.createRange();
      range.selectNodeContents(item);
      return {
        text: item.textContent,
        height: rect.height,
        textLines: range.getClientRects().length,
        overflow: item.scrollWidth - item.clientWidth,
      };
    }));
    expect(geometry.map(({ text }) => text)).toEqual(["Source A", "Source B", "Source C"]);
    expect(Math.max(...geometry.map(({ height }) => height)) - Math.min(...geometry.map(({ height }) => height)))
      .toBeLessThanOrEqual(1);
    for (const item of geometry) {
      expect(item.textLines, `${item.text} must stay on one line`).toBe(1);
      expect(item.overflow, `${item.text} must fit its button`).toBeLessThanOrEqual(1);
      if (testInfo.project.name.startsWith("mobile")) {
        expect(item.height, `${item.text} must retain its mobile touch target`).toBeGreaterThanOrEqual(44);
      } else {
        expect(item.height, `${item.text} must remain visually compact`).toBeLessThanOrEqual(26);
      }
    }
  });

  test("desktop Composition controls use the complete content width", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop width contract");
    const widths = await page.locator("#sliders-panel").evaluate((panel) => {
      const content = panel.querySelector<HTMLElement>("#sliders")!;
      const toggle = panel.querySelector<HTMLElement>(".material-source-group .toggle-row")!;
      const ranges = [...panel.querySelectorAll<HTMLElement>('input[type="range"]')];
      return {
        content: content.getBoundingClientRect().width,
        toggle: toggle.getBoundingClientRect().width,
        ranges: ranges.map((range) => range.getBoundingClientRect().width),
        overflow: panel.scrollWidth - panel.clientWidth,
      };
    });
    expect(Math.abs(widths.toggle - widths.content)).toBeLessThanOrEqual(1);
    for (const width of widths.ranges) expect(Math.abs(width - widths.content)).toBeLessThanOrEqual(1);
    expect(widths.overflow).toBeLessThanOrEqual(1);
  });

  test("desktop preserves every primary panel and its information", async ({
    page,
  }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop content invariant");

    const requiredPanels = [
      {
        name: "Composition",
        locator: page.locator("#sliders-panel"),
        content: ["Composition", "Adjust sliders", "Material Source", "Temperature"],
      },
      {
        name: "Filters",
        locator: page.locator("#filters-panel"),
        content: ["Filters", "Clear all"],
      },
      {
        name: "Performance Tradeoffs",
        locator: page.locator("#tradeoffs-panel"),
        content: ["Performance Tradeoffs", "GWP", "28-day Strength"],
      },
      {
        name: "Mix Insight",
        locator: page.locator("#mix-insight"),
        content: ["Mix Insight"],
      },
      {
        name: "Ingredient Insight",
        locator: page.locator("#ingredient-insight"),
        content: ["Ingredient Insight", "Cement", "Portland cement"],
      },
      {
        name: "Predicted Strength Curve",
        locator: page.locator("#strength-panel"),
        content: ["Predicted Strength Curve", "GWP", "Cost", "W/B"],
      },
      {
        name: "References",
        locator: page.locator(".references-panel"),
        content: [
          "References",
          "BOxCrete: A Bayesian Optimization",
          "Sustainable Concrete via Bayesian Optimization",
        ],
      },
    ];

    for (const panel of requiredPanels) {
      await expect(panel.locator, `${panel.name} panel must remain visible`).toBeVisible();
      for (const text of panel.content) {
        await expect(
          panel.locator,
          `${panel.name} must preserve “${text}”`,
        ).toContainText(text);
      }
    }

    await expect(page.getByRole("radio", { name: "28-day strength", exact: true })).toBeVisible();
    await expect(page.locator("#sliders-panel .slider-group")).toHaveCount(9);
    await expect(page.locator(".references-panel .ref-item")).toHaveCount(2);
    await expect(page.locator("#readouts > div")).toHaveCount(3);
    await expect(page.locator("#scatter-canvas")).toBeVisible();
    await expect(page.locator("#curve-canvas")).toBeVisible();
  });

  test("mobile preserves every primary panel and its information", async ({
    page,
  }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile content invariant");

    const mobileComposition = page.locator(".mobile-sliders-view");
    await expect(mobileComposition).toBeVisible();
    await expect(mobileComposition.locator(".slider-group")).toHaveCount(9);
    await expect(mobileComposition).toContainText("Material Source");
    await expect(mobileComposition).toContainText("Temperature");

    await expect(page.getByRole("heading", { name: "Predicted Strength Curve" })).toBeVisible();
    await expect(page.locator("#readouts")).toContainText("GWP");
    await expect(page.locator("#readouts")).toContainText("Cost");
    await expect(page.locator("#mix-insight")).toContainText("Mix Insight");
    await expect(page.locator("#ingredient-insight")).toContainText("Portland cement");
    await expect(page.locator(".references-panel .ref-item")).toHaveCount(2);

    await page.locator("#mobile-show-scatter").click();
    await expect(page.locator("#scatter-canvas")).toBeVisible();
    await expect(page.locator("#axis-selector-x")).toContainText("GWP");
    await expect(page.locator("#axis-selector-x")).toContainText("Cost");
    await expect(page.locator("#axis-selector-y")).toContainText("28-day Strength");
    await expect(page.locator("#axis-selector-y")).toContainText("1-day Strength");
    await expect(page.getByRole("radio", { name: "28-day strength", exact: true })).toBeVisible();
    await expect(page.getByRole("radio", { name: "1-day strength", exact: true })).toBeVisible();
    await expect(page.locator("#filter-add")).toBeVisible();
    await expect(page.locator("#filter-clear")).toHaveCount(1);
    await expect(page.locator("#filter-clear")).toHaveText("Clear all");
  });

  test("preserves Composition on both sides of the responsive breakpoint", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "custom viewport invariant");

    await page.setViewportSize({ width: 1051, height: 900 });
    await page.goto("/");
    const desktopPanel = page.locator("#sliders-panel");
    await expect(desktopPanel).toBeVisible();
    await expect(desktopPanel.locator(".slider-group")).toHaveCount(9);
    const desktopGeometry = await page.evaluate(() => {
      const composition = document.querySelector("#sliders-panel")!.getBoundingClientRect();
      const scatter = document.querySelector("#scatter-canvas")!.closest(".chart-panel")!.getBoundingClientRect();
      const curve = document.querySelector("#curve-canvas")!.closest(".chart-panel")!.getBoundingClientRect();
      return {
        compositionRight: composition.right,
        scatterLeft: scatter.left,
        scatterRight: scatter.right,
        curveLeft: curve.left,
      };
    });
    expect(desktopGeometry.compositionRight).toBeLessThanOrEqual(desktopGeometry.scatterLeft);
    expect(desktopGeometry.scatterRight).toBeLessThanOrEqual(desktopGeometry.curveLeft);

    await page.setViewportSize({ width: 1050, height: 900 });
    await page.reload();
    await expect(page.locator("#sliders-panel")).toBeHidden();
    await page.locator("#mobile-show-sliders").click();
    const mobileComposition = page.locator(".mobile-sliders-view");
    await expect(mobileComposition).toBeVisible();
    await expect(mobileComposition.locator(".slider-group")).toHaveCount(9);
    await expect(mobileComposition.getByRole("button", { name: "Cement ingredient insight" })).toBeVisible();
    await expect(mobileComposition.getByRole("button", { name: "Temperature ingredient insight" })).toBeVisible();
  });

  test("ingredient swaps reset local scroll and let intrinsic shell height follow content", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop intrinsic-shell invariant");
    const body = page.locator(".ingredient-insight-body");
    const shell = page.locator("#ingredient-insight");
    const beforeHeight = await shell.evaluate((element) => element.getBoundingClientRect().height);

    await body.evaluate((element) => {
      const filler = document.createElement("div");
      filler.dataset.testFiller = "true";
      filler.style.cssText = "height:800px;min-height:800px;flex:0 0 auto";
      element.appendChild(filler);
      element.scrollTop = element.scrollHeight;
      document.dispatchEvent(new CustomEvent("dashboard-layout-change"));
    });
    expect(await body.evaluate((element) => element.scrollTop)).toBeGreaterThan(0);

    await page.getByRole("button", { name: "Fly Ash ingredient insight" }).click();
    await expect(page.locator("#ingredient-insight-text")).toContainText("pozzolanic byproduct");
    expect(await body.evaluate((element) => element.scrollTop), "swap must reset local scroll").toBe(0);
    expect(await body.evaluate((element: HTMLElement) => element.style.height), "body must not receive inline height").toBe("");
    expect(await shell.evaluate((element: HTMLElement) => element.style.height), "shell must not receive inline height").toBe("");

    await body.locator('[data-test-filler="true"]').evaluate((element) => element.remove());
    await page.evaluate(() => document.dispatchEvent(new CustomEvent("dashboard-layout-change")));
    await expect.poll(() => shell.evaluate((element) => element.getBoundingClientRect().height)).toBeLessThan(beforeHeight + 40);
  });

  test("rapid ingredient updates commit semantics immediately and clean visual clones", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop timing invariant");
    const flyAsh = page.getByRole("button", { name: "Fly Ash ingredient insight" });
    const slag = page.getByRole("button", { name: "Slag ingredient insight" });
    const text = page.locator("#ingredient-insight-text");
    const body = page.locator(".ingredient-insight-body");

    await flyAsh.click();
    await expect(text).toContainText("pozzolanic byproduct");
    const firstClone = body.locator(".content-swap-ghost");
    await expect(firstClone).toHaveCount(1);
    await expect(firstClone).toHaveAttribute("aria-hidden", "true");
    await expect(firstClone).toHaveAttribute("inert", "");
    expect(await firstClone.locator("[id]").count()).toBe(0);
    await expect(body.locator('[aria-live="polite"]')).toHaveCount(1);

    await slag.click();
    await expect(text).not.toContainText("pozzolanic byproduct");
    await expect(text).toContainText("Ground granulated blast furnace slag");
    await expect(slag).toBeFocused();
    await expect(body.locator(".content-swap-ghost")).toHaveCount(1);

    await expect
      .poll(() =>
        body.evaluate((element) => ({
          clones: element.querySelectorAll(".content-swap-ghost").length,
          animations: element.getAnimations({ subtree: true }).length,
          blockSize: (element as HTMLElement).style.blockSize,
          overflow: (element as HTMLElement).style.overflow,
        })),
      )
      .toEqual({ clones: 0, animations: 0, blockSize: "", overflow: "" });
  });

  test("ingredient shell retargets through intermediate intrinsic sizes", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop structural motion invariant");
    const samples = await page.evaluate(async () => {
      const shell = document.querySelector<HTMLElement>("#ingredient-insight")!;
      const control = [...document.querySelectorAll<HTMLButtonElement>(".ingredient-name")]
        .find((button) => button.textContent === "Water")!;
      const before = shell.getBoundingClientRect().height;
      control.click();
      const animation = shell.getAnimations()[0];
      if (!animation) throw new Error("ingredient shell animation did not start");
      animation.pause();
      await animation.ready;
      const duration = Number(animation.effect?.getTiming().duration);
      if (!Number.isFinite(duration) || duration <= 0) throw new Error("invalid ingredient shell duration");
      const heights = [];
      for (const progress of [0.2, 0.5, 0.8]) {
        animation.currentTime = duration * progress;
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        heights.push(shell.getBoundingClientRect().height);
      }
      animation.finish();
      await animation.finished;
      return { before, heights, after: shell.getBoundingClientRect().height };
    });
    expect(samples.heights).toHaveLength(3);
    expect(Math.abs(samples.after - samples.before)).toBeGreaterThan(1);
    for (let index = 1; index < samples.heights.length; index += 1) {
      const previousDelta = samples.after - samples.heights[index - 1];
      const currentDelta = samples.after - samples.heights[index];
      expect(Math.abs(currentDelta)).toBeLessThan(Math.abs(previousDelta) - 0.5);
    }
  });

  test("three same-frame updates do not strand inline transition overrides", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop timing invariant");
    const text = page.locator("#ingredient-insight-text");

    await page.evaluate(() => {
      for (const name of ["Fly Ash", "Slag", "Water"]) {
        const control = [...document.querySelectorAll<HTMLButtonElement>(".ingredient-name")]
          .find((button) => button.textContent === name);
        control?.click();
      }
    });
    await expect(text).toContainText("Controls the water-to-binder");
    await page.evaluate(async () => {
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    });

    const inlineStyles = await text.evaluate((element: HTMLElement) => ({
      opacity: element.style.opacity,
      transform: element.style.transform,
      transition: element.style.transition,
    }));
    expect(inlineStyles).toEqual({ opacity: "", transform: "", transition: "" });
  });

  test("always keeps exactly one ingredient selected", async ({ page }) => {
    const cement = page.getByRole("button", { name: "Cement ingredient insight" });
    const flyAsh = page.getByRole("button", { name: "Fly Ash ingredient insight" });

    await cement.click();
    await expect(cement).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator('.ingredient-name[aria-pressed="true"]')).toHaveCount(1);

    await flyAsh.click();
    await expect(flyAsh).toHaveAttribute("aria-pressed", "true");
    await expect(cement).toHaveAttribute("aria-pressed", "false");
    await expect(page.locator('.ingredient-name[aria-pressed="true"]')).toHaveCount(1);
    await expect(page.locator("#ingredient-insight-text")).toContainText(
      "pozzolanic byproduct",
    );

    await flyAsh.click();
    await expect(flyAsh).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator('.ingredient-name[aria-pressed="true"]')).toHaveCount(1);
  });

  test("ingredient controls expose button semantics and panel relationship", async ({ page }) => {
    const controls = page.locator("button.ingredient-name");
    expect(await controls.count()).toBeGreaterThanOrEqual(5);

    for (const control of await controls.all()) {
      await expect(control).toHaveAttribute("type", "button");
      await expect(control).toHaveAttribute("aria-controls", "ingredient-insight-text");
      await expect(control).toHaveAttribute("aria-pressed", /true|false/);
    }

    await expect(page.locator("#ingredient-insight-text")).toHaveAttribute(
      "aria-live",
      "polite",
    );
  });

  test("keyboard activation transfers selection and focus remains visible", async ({ page }) => {
    const slag = page.getByRole("button", { name: "Slag ingredient insight" });
    await slag.focus();
    await page.keyboard.press("Enter");

    await expect(slag).toHaveAttribute("aria-pressed", "true");
    const focusStyle = await slag.evaluate((element) => {
      const style = getComputedStyle(element);
      return {
        outlineStyle: style.outlineStyle,
        outlineWidth: parseFloat(style.outlineWidth),
      };
    });
    expect(focusStyle.outlineStyle).not.toBe("none");
    expect(focusStyle.outlineWidth).toBeGreaterThan(0);
  });

  test("selected control reads as an accessible pill and dark panels stay legible", async ({ page }) => {
    const styles = await page.evaluate(() => {
      const parseRgb = (value: string) =>
        value.match(/[\d.]+/g)!.slice(0, 3).map(Number);
      const luminance = (rgb: number[]) => {
        const channels = rgb.map((value) => {
          const channel = value / 255;
          return channel <= 0.04045
            ? channel / 12.92
            : ((channel + 0.055) / 1.055) ** 2.4;
        });
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
      };
      const contrast = (foreground: string, background: string) => {
        const first = luminance(parseRgb(foreground));
        const second = luminance(parseRgb(background));
        return (Math.max(first, second) + 0.05) / (Math.min(first, second) + 0.05);
      };
      const readTheme = (theme: "light" | "dark") => {
        document.documentElement.dataset.theme = theme;
        const pill = document.querySelector('.ingredient-name[aria-pressed="true"]')!;
        const pillStyle = getComputedStyle(pill);
        const panelStyle = getComputedStyle(document.querySelector(".panel")!);
        return {
          radius: parseFloat(pillStyle.borderRadius),
          borderWidth: parseFloat(pillStyle.borderWidth),
          background: pillStyle.backgroundColor,
          contrast: contrast(pillStyle.color, pillStyle.backgroundColor),
          panelBackground: panelStyle.backgroundColor,
        };
      };
      return { light: readTheme("light"), dark: readTheme("dark") };
    });

    expect(styles.light.radius).toBeGreaterThanOrEqual(8);
    expect(styles.light.borderWidth).toBeGreaterThan(0);
    expect(styles.light.background).not.toBe("rgba(0, 0, 0, 0)");
    expect(styles.light.contrast).toBeGreaterThanOrEqual(4.5);
    expect(styles.dark.contrast).toBeGreaterThanOrEqual(4.5);
    const darkAlpha = styles.dark.panelBackground.match(/rgba?\([^,]+,[^,]+,[^,]+(?:,\s*([\d.]+))?\)/)?.[1];
    expect(darkAlpha === undefined ? 1 : Number(darkAlpha)).toBeGreaterThanOrEqual(0.82);

    const transitionContrasts = await page.evaluate(async () => {
      const parseRgb = (value: string) =>
        value.match(/[\d.]+/g)!.slice(0, 3).map(Number);
      const luminance = (rgb: number[]) => {
        const channels = rgb.map((value) => {
          const channel = value / 255;
          return channel <= 0.04045
            ? channel / 12.92
            : ((channel + 0.055) / 1.055) ** 2.4;
        });
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
      };
      const pill = document.querySelector('.ingredient-name[aria-pressed="true"]')!;
      const readPillColors = () => {
        const style = getComputedStyle(pill);
        return {
          color: style.color,
          backgroundColor: style.backgroundColor,
        };
      };
      document.documentElement.dataset.theme = "light";

      const probe = pill.cloneNode(true) as HTMLElement;
      probe.style.transition = "none";
      probe.style.position = "fixed";
      probe.style.visibility = "hidden";
      document.body.append(probe);
      const expectedLightColors = (() => {
        const style = getComputedStyle(probe);
        return {
          color: style.color,
          backgroundColor: style.backgroundColor,
        };
      })();
      probe.remove();

      await new Promise<void>((resolve, reject) => {
        const started = performance.now();
        const waitForSettledLightTheme = () => {
          const current = readPillColors();
          if (
            current.color === expectedLightColors.color &&
            current.backgroundColor === expectedLightColors.backgroundColor
          ) {
            resolve();
          } else if (performance.now() - started > 1_500) {
            reject(
              new Error(
                `light theme did not settle: ${JSON.stringify({ current, expectedLightColors })}`,
              ),
            );
          } else {
            requestAnimationFrame(waitForSettledLightTheme);
          }
        };
        requestAnimationFrame(waitForSettledLightTheme);
      });

      const lightStart = readPillColors();
      if (
        lightStart.color !== expectedLightColors.color ||
        lightStart.backgroundColor !== expectedLightColors.backgroundColor
      ) {
        throw new Error("contrast sampling did not start from rendered light colors");
      }
      document.documentElement.dataset.theme = "dark";

      const samples: number[] = [];
      await new Promise<void>((resolve) => {
        const started = performance.now();
        const sample = (now: number) => {
          const style = getComputedStyle(pill);
          const foreground = luminance(parseRgb(style.color));
          const background = luminance(parseRgb(style.backgroundColor));
          samples.push(
            (Math.max(foreground, background) + 0.05) /
              (Math.min(foreground, background) + 0.05),
          );
          if (now - started < 550) requestAnimationFrame(sample);
          else resolve();
        };
        requestAnimationFrame(sample);
      });
      return samples;
    });
    expect(Math.min(...transitionContrasts)).toBeGreaterThanOrEqual(4.5);
  });
});
