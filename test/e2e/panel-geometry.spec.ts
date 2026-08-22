import { expect, test, type Locator, type Page } from "@playwright/test";

const GAP_TOLERANCE = 1;

type Metrics = {
  top: number;
  bottom: number;
  height: number;
  clientHeight: number;
  scrollHeight: number;
  maxBlockSize: number;
  overflowY: string;
  inlineBlockSize: string;
  inlineMaxBlockSize: string;
  paddingBottom: number;
  tabIndex: number;
};

async function waitForDashboard(page: Page) {
  await page.goto("/?test=1");
  await expect(page.locator("#sliders .slider-group").last()).toBeAttached({ timeout: 15_000 });
  await page.evaluate(async () => {
    await document.fonts.ready;
    await Promise.all([...document.querySelectorAll(".fade-in-up")].flatMap((element) => element.getAnimations().map((animation) => animation.finished)));
  });
}

async function metrics(locator: Locator): Promise<Metrics> {
  return locator.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    return {
      top: rect.top,
      bottom: rect.bottom,
      height: rect.height,
      clientHeight: element.clientHeight,
      scrollHeight: element.scrollHeight,
      maxBlockSize: style.maxBlockSize === "none"
        ? Number.POSITIVE_INFINITY
        : parseFloat(style.maxBlockSize),
      overflowY: style.overflowY,
      inlineBlockSize: (element as HTMLElement).style.blockSize,
      inlineMaxBlockSize: (element as HTMLElement).style.maxBlockSize,
      paddingBottom: parseFloat(style.paddingBottom),
      tabIndex: (element as HTMLElement).tabIndex,
    };
  });
}

async function addFiller(locator: Locator, height: number) {
  await locator.evaluate((element, fillerHeight) => {
    const filler = document.createElement("div");
    filler.dataset.testFiller = "true";
    filler.style.cssText = `height:${fillerHeight}px;min-height:${fillerHeight}px;flex:0 0 auto`;
    filler.textContent = "Oversized content";
    element.appendChild(filler);
    document.dispatchEvent(new CustomEvent("dashboard-layout-change"));
  }, height);
}

async function removeFiller(locator: Locator) {
  await locator.locator('[data-test-filler="true"]').evaluate((element) => element.remove());
  await locator.page().evaluate(() => document.dispatchEvent(new CustomEvent("dashboard-layout-change")));
}

async function configuredGap(page: Page) {
  return page.locator(".column-flow").first().evaluate((element) => parseFloat(getComputedStyle(element).gap));
}

test.describe("intrinsic capped dashboard panels", () => {
  test.beforeEach(async ({ page }) => waitForDashboard(page));

  test("desktop columns are top-aligned intrinsic stacks separated only by the configured gap", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop column contract");
    const columnTops = await Promise.all([
      page.locator("#composition-column"),
      page.locator("#tradeoffs-column"),
      page.locator("#strength-column"),
    ].map(async (locator) => (await metrics(locator)).top));
    expect(Math.max(...columnTops) - Math.min(...columnTops)).toBeLessThanOrEqual(1);

    const gap = await configuredGap(page);
    for (const [first, second] of [
      ["#tradeoffs-panel", "#mix-insight"],
      ["#mix-insight", "#ingredient-insight"],
      ["#strength-panel", "#filters-panel"],
      ["#filters-panel", "#references-panel"],
      ["#references-panel", ".site-footer"],
    ] as const) {
      const [a, b] = await Promise.all([metrics(page.locator(first)), metrics(page.locator(second))]);
      expect(Math.abs(b.top - a.bottom - gap), `${second} must follow ${first} by one gap`).toBeLessThanOrEqual(GAP_TOLERANCE);
    }
  });

  for (const viewport of [
    { width: 1051, height: 900 },
    { width: 1280, height: 700 },
    { width: 1280, height: 800 },
    { width: 1728, height: 900 },
    { width: 1728, height: 1000 },
    { width: 1728, height: 1117 },
  ]) {
    test(`desktop Composition is intrinsic at ${viewport.width}x${viewport.height}`, async ({ page }, testInfo) => {
      test.skip(!testInfo.project.name.startsWith("desktop"), "desktop Composition contract");
      await page.setViewportSize(viewport);
      await waitForDashboard(page);

      const panel = page.locator("#sliders-panel");
      const body = page.locator("#sliders");
      const lastGroup = body.locator(".slider-group").last();
      await expect(body.locator(".slider-group")).toHaveCount(9);
      await expect(lastGroup.getByRole("button", { name: "Temperature ingredient insight" })).toBeVisible();

      const [shell, bodyMetrics, finalExtent] = await Promise.all([
        metrics(panel),
        metrics(body),
        lastGroup.evaluate((element) => {
          const rect = element.getBoundingClientRect();
          return rect.bottom + parseFloat(getComputedStyle(element).marginBottom);
        }),
      ]);
      expect(bodyMetrics.scrollHeight).toBeLessThanOrEqual(bodyMetrics.clientHeight + 1);
      expect(bodyMetrics.overflowY).not.toMatch(/auto|scroll/);
      expect(shell.maxBlockSize).toBe(Number.POSITIVE_INFINITY);
      expect(shell.inlineBlockSize).toBe("");
      expect(shell.inlineMaxBlockSize).toBe("");
      expect(bodyMetrics.inlineBlockSize).toBe("");
      expect(bodyMetrics.inlineMaxBlockSize).toBe("");
      expect(Math.abs(shell.bottom - finalExtent - shell.paddingBottom)).toBeLessThanOrEqual(2);

      await body.evaluate((element: HTMLElement) => { element.scrollTop = 100; });
      expect(await body.evaluate((element) => element.scrollTop)).toBe(0);

      const documentRange = await page.evaluate(() =>
        document.documentElement.scrollHeight - document.documentElement.clientHeight,
      );
      if (documentRange > 1) {
        await lastGroup.scrollIntoViewIfNeeded();
        await expect(lastGroup).toBeVisible();
        const viewportFit = await lastGroup.evaluate((element) => {
          const rect = element.getBoundingClientRect();
          return rect.top >= -1 && rect.bottom <= innerHeight + 1;
        });
        expect(viewportFit, "the final Composition control must be reachable in the document viewport").toBe(true);
      }
    });
  }

  test("desktop filters and credit are single live subtrees owned by the Strength column", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop filter ownership");
    const filterPanel = page.locator("#filters-panel");
    const credit = page.locator(".site-footer");
    await expect(filterPanel).toHaveCount(1);
    await expect(credit).toHaveCount(1);
    await expect(filterPanel.locator("#filter-controls")).toHaveCount(1);
    await expect(page.locator("#composition-column #filters-panel")).toHaveCount(0);
    await expect(page.locator("#strength-column > #filters-panel")).toHaveCount(1);
    await expect(page.locator("#strength-column > .site-footer")).toHaveCount(1);
    const [strength, filters, references, footer] = await Promise.all([
      metrics(page.locator("#strength-panel")),
      metrics(filterPanel),
      metrics(page.locator("#references-panel")),
      metrics(credit),
    ]);
    expect(filters.top).toBeGreaterThanOrEqual(strength.bottom);
    expect(references.top).toBeGreaterThanOrEqual(filters.bottom);
    expect(footer.top).toBeGreaterThanOrEqual(references.bottom);
  });

  test("each managed panel cap leaves oversized content scrolling only in its established body", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "managed cap ownership contract");
    await page.setViewportSize({ width: 1728, height: 1000 });
    await waitForDashboard(page);

    for (const contract of [
      { shell: "#mix-insight", body: ".mix-insight-body", cap: 14 * 16 },
      { shell: "#ingredient-insight", body: ".ingredient-insight-body", cap: 18 * 16 },
    ]) {
      const shell = page.locator(contract.shell);
      const body = page.locator(contract.body);
      await addFiller(body, 1200);
      await expect.poll(async () => (await metrics(body)).tabIndex).toBe(0);

      const cappedShell = await metrics(shell);
      const overflowingBody = await metrics(body);
      expect(cappedShell.maxBlockSize, `${contract.shell} computed cap`).toBeCloseTo(contract.cap, 0);
      expect(cappedShell.height, `${contract.shell} rendered cap`).toBeLessThanOrEqual(contract.cap + 1);
      expect(overflowingBody.scrollHeight, `${contract.body} owns overflow`)
        .toBeGreaterThan(overflowingBody.clientHeight + 500);
      expect(overflowingBody.tabIndex).toBe(0);

      await removeFiller(body);
      await expect
        .poll(async () => {
          const restoredBody = await metrics(body);
          return restoredBody.tabIndex ===
            (restoredBody.scrollHeight > restoredBody.clientHeight + 1 ? 0 : -1);
        })
        .toBe(true);
    }
  });

  test("References uses a stable viewport/header cap and keeps overflow in its list", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop References cap contract");
    await page.setViewportSize({ width: 1280, height: 700 });
    await waitForDashboard(page);

    const readCap = () => page.evaluate(() => ({
      css: parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--references-usable-cap")),
      viewport: window.visualViewport?.height ?? window.innerHeight,
      header: document.querySelector(".site-header")!.getBoundingClientRect().height,
    }));
    const initialCap = await readCap();
    expect(initialCap.css).toBeCloseTo(initialCap.viewport - initialCap.header - 16, 0);

    const shell = page.locator("#references-panel");
    const body = page.locator(".ref-list");
    const intrinsic = await metrics(shell);
    expect(intrinsic.height).toBeLessThan(initialCap.css - 1);
    expect((await metrics(body)).scrollHeight).toBeLessThanOrEqual((await metrics(body)).clientHeight + 1);

    await addFiller(body, 1200);
    await expect.poll(async () => (await metrics(body)).tabIndex).toBe(0);
    const [capped, overflowing] = await Promise.all([metrics(shell), metrics(body)]);
    expect(capped.height).toBeLessThanOrEqual(initialCap.css + 1);
    expect(capped.height).toBeGreaterThanOrEqual(initialCap.css - 1);
    expect(overflowing.scrollHeight).toBeGreaterThan(overflowing.clientHeight + 500);
    expect(capped.overflowY).not.toMatch(/auto|scroll/);
    expect(overflowing.overflowY).toBe("auto");

    await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
    expect((await readCap()).css).toBe(initialCap.css);
    await page.locator("#mix-insight-text").evaluate((element) => {
      element.textContent = `${element.textContent} ${"Earlier panel growth. ".repeat(40)}`;
      document.dispatchEvent(new CustomEvent("dashboard-layout-change"));
    });
    await expect.poll(async () => (await readCap()).css).toBe(initialCap.css);
  });

  test("shrinking an earlier panel compacts the next same-column panel upward", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop compaction contract");
    const body = page.locator(".mix-insight-body");
    const ingredient = page.locator("#ingredient-insight");
    await page.locator("#mix-insight-text").evaluate((element) => {
      element.textContent = "Short mix insight.";
      document.dispatchEvent(new CustomEvent("dashboard-layout-change"));
    });
    const shortTop = (await metrics(ingredient)).top;
    await addFiller(body, 900);
    const longTop = (await metrics(ingredient)).top;
    expect(longTop).toBeGreaterThan(shortTop + 100);
    await removeFiller(body);
    await expect.poll(async () => (await metrics(ingredient)).top).toBeLessThan(longTop - 100);
  });

  test("empty Mix remains a compact heading plus one-line placeholder", async ({ page }) => {
    await page.locator("#sliders input[type=range]").first().evaluate((input: HTMLInputElement) => {
      input.value = String((Number(input.min) + Number(input.max)) / 2 + 0.37);
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
    });
    const placeholder = page.locator("#mix-insight .mix-insight-placeholder");
    await expect(placeholder).toContainText(/not available|Click a data point/);
    await expect(page.locator("#mix-insight-text")).not.toHaveClass(/fade-out|fade-in/);
    await expect
      .poll(() =>
        page.locator(".mix-insight-body").evaluate((body) =>
          body.scrollHeight - body.clientHeight,
        ),
      )
      .toBeLessThanOrEqual(1);
    const state = await page.locator("#mix-insight").evaluate((shell) => {
      const body = shell.querySelector<HTMLElement>(".mix-insight-body")!;
      const text = shell.querySelector<HTMLElement>(".mix-insight-placeholder")!;
      const shellRect = shell.getBoundingClientRect();
      const textRect = text.getBoundingClientRect();
      return {
        shellHeight: shellRect.height,
        textHeight: textRect.height,
        lineHeight: parseFloat(getComputedStyle(text).lineHeight),
        bodyOverflow: body.scrollHeight - body.clientHeight,
        slack: shellRect.bottom - textRect.bottom,
      };
    });
    expect(state.textHeight).toBeLessThanOrEqual(state.lineHeight + 1);
    expect(state.bodyOverflow).toBeLessThanOrEqual(1);
    expect(state.slack).toBeLessThan(40);
    expect(state.shellHeight).toBeLessThan(130);
  });

  test("Composition height is invariant between taller managed viewports", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop intrinsic-height contract");
    const measurements: number[] = [];
    for (const height of [1000, 1117]) {
      await page.setViewportSize({ width: 1728, height });
      await waitForDashboard(page);
      measurements.push((await metrics(page.locator("#sliders-panel"))).height);
    }
    expect(Math.abs(measurements[1] - measurements[0])).toBeLessThanOrEqual(1);
  });

  test("desktop document scrolling reaches the final Composition control", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop document-scroll contract");
    await page.setViewportSize({ width: 1280, height: 700 });
    await waitForDashboard(page);
    const finalControl = page.getByRole("button", { name: "Temperature ingredient insight" });
    const documentRange = await page.evaluate(() =>
      document.documentElement.scrollHeight - document.documentElement.clientHeight,
    );
    expect(documentRange).toBeGreaterThan(1);
    await finalControl.scrollIntoViewIfNeeded();
    await expect(finalControl).toBeVisible();
    expect(await page.evaluate(() => scrollY)).toBeGreaterThan(0);
  });

  test("short desktop preserves useful plots and lets the document own overflow", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop document-scroll fallback");
    await page.setViewportSize({ width: 1280, height: 400 });
    const state = await page.evaluate(() => ({
      documentRange: document.documentElement.scrollHeight - document.documentElement.clientHeight,
      canvasHeights: ["#scatter-canvas", "#curve-canvas"].map((selector) => document.querySelector(selector)!.getBoundingClientRect().height),
    }));
    expect(state.documentRange).toBeGreaterThan(100);
    expect(Math.min(...state.canvasHeights)).toBeGreaterThan(250);
    await page.locator(".site-footer").scrollIntoViewIfNeeded();
    await expect(page.locator(".site-footer")).toBeVisible();
    expect(await page.evaluate(() => scrollY)).toBeGreaterThan(0);
  });

  test("scroll regions enter sequential focus only while visibly overflowing", async ({ page }) => {
    const regions = [
      "#sliders",
      ".mobile-scroll-content",
      ".mix-insight-body",
      ".ingredient-insight-body",
      ".curve-body",
      ".ref-list",
      "#filter-rows",
    ];
    for (const selector of regions) {
      const locator = page.locator(selector);
      const value = await locator.evaluate((element) => ({
        visible: (element as HTMLElement).offsetParent !== null,
        overflow: element.scrollHeight > element.clientHeight + 1,
        tabIndex: (element as HTMLElement).tabIndex,
      }));
      expect(value.tabIndex, selector).toBe(value.visible && value.overflow ? 0 : -1);
    }

    const body = page.locator(".ingredient-insight-body");
    await addFiller(body, 1000);
    await expect(body).toHaveAttribute("tabindex", "0");
    await removeFiller(body);
    await expect(body).not.toHaveAttribute("tabindex");
  });
});
