import { expect, test, type Page } from "@playwright/test";
import { waitForDashboardLayoutReady } from "./dashboard-helpers";

async function overflowState(page: Page) {
  return page.locator(".ref-list").evaluate((element) => ({
    overflow: element.scrollHeight > element.clientHeight + 1,
    tabIndex: (element as HTMLElement).tabIndex,
  }));
}

test.describe("compact reference disclosures", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/?test=1");
    await waitForDashboardLayoutReady(page);
  });

  test("keeps bibliography and citation actions visible while descriptions are collapsed", async ({ page }) => {
    const disclosures = page.locator(".ref-details");
    await expect(disclosures).toHaveCount(2);
    expect(await disclosures.evaluateAll((items) => items.map((item) => (item as HTMLDetailsElement).open)))
      .toEqual([false, false]);

    await expect(page.locator(".ref-title")).toHaveCount(2);
    await expect(page.locator(".ref-authors")).toHaveCount(2);
    await expect(page.locator(".ref-link")).toHaveCount(2);
    await expect(page.locator(".ref-cite[data-format=bibtex]")).toHaveCount(2);
    await expect(page.locator(".ref-cite[data-format=apa]")).toHaveCount(2);
    for (const selector of [".ref-title", ".ref-authors", ".ref-link", ".ref-cite"]) {
      await expect(page.locator(selector).first()).toBeVisible();
    }
    await expect(page.locator(".ref-desc").first()).toBeHidden();
  });

  test("citation actions stay at inline-end while metadata wraps without squeezing summaries", async ({ page }, testInfo) => {
    const wide = testInfo.project.name.startsWith("desktop")
      ? { width: 1728, height: 1000 }
      : { width: 844, height: 390 };
    await page.setViewportSize(wide);
    await page.goto("/?test=1");
    await waitForDashboardLayoutReady(page);

    const read = () => page.locator(".ref-item").first().evaluate((item) => {
      const content = item.querySelector(".ref-content")!.getBoundingClientRect();
      const authors = item.querySelector(".ref-authors")!.getBoundingClientRect();
      const actions = item.querySelector(".ref-actions-row")!.getBoundingClientRect();
      const summary = item.querySelector(".ref-details > summary")!.getBoundingClientRect();
      const meta = item.querySelector(".ref-meta-row")!.getBoundingClientRect();
      return {
        content: { left: content.left, right: content.right, width: content.width },
        authors: { top: authors.top, bottom: authors.bottom, right: authors.right },
        actions: { left: actions.left, right: actions.right, top: actions.top, bottom: actions.bottom },
        summary: { left: summary.left, right: summary.right, width: summary.width },
        meta: { top: meta.top, bottom: meta.bottom },
      };
    });

    const collapsed = await read();
    expect(Math.abs(collapsed.actions.right - collapsed.content.right)).toBeLessThanOrEqual(1);
    expect(collapsed.authors.bottom).toBeGreaterThan(collapsed.actions.top);
    expect(collapsed.actions.bottom).toBeGreaterThan(collapsed.authors.top);
    expect(Math.abs(collapsed.summary.left - collapsed.content.left)).toBeLessThanOrEqual(1);
    expect(Math.abs(collapsed.summary.right - collapsed.content.right)).toBeLessThanOrEqual(1);

    await page.locator(".ref-details > summary").first().click();
    await expect.poll(() => page.locator(".ref-description-motion").first().evaluate((element) =>
      element.getAnimations().length,
    )).toBe(0);
    const expanded = await read();
    expect(Math.abs(expanded.actions.right - expanded.content.right)).toBeLessThanOrEqual(1);
    expect(Math.abs(expanded.actions.left - collapsed.actions.left)).toBeLessThanOrEqual(1);
    expect(Math.abs(expanded.actions.right - collapsed.actions.right)).toBeLessThanOrEqual(1);

    await page.setViewportSize({ width: 320, height: 720 });
    await expect.poll(() => page.locator(".ref-meta-row").first().evaluate((element) =>
      element.getBoundingClientRect().width > 0,
    )).toBe(true);
    const narrow = await read();
    expect(narrow.actions.top).toBeGreaterThanOrEqual(narrow.authors.bottom - 1);
    expect(narrow.actions.left).toBeGreaterThanOrEqual(narrow.content.left - 1);
    expect(Math.abs(narrow.actions.right - narrow.content.right)).toBeLessThanOrEqual(1);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth))
      .toBeLessThanOrEqual(1);
  });

  test("keyboard order follows metadata actions then the native disclosure summary", async ({ page }, testInfo) => {
    const first = page.locator(".ref-item").first();
    const selectors = [
      ".ref-title",
      ".ref-link",
      '.ref-cite[data-format="bibtex"]',
      '.ref-cite[data-format="apa"]',
      ".ref-details > summary",
    ];
    expect(await first.evaluate((item, orderedSelectors) => {
      const nodes = orderedSelectors.map((selector) => item.querySelector(selector)!);
      return nodes.slice(1).every((node, index) =>
        Boolean(nodes[index].compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING),
      );
    }, selectors)).toBe(true);

    for (const selector of selectors) {
      await first.locator(selector).focus();
      await expect(first.locator(selector)).toBeFocused();
    }

    // WebKit follows the host Safari preference that can omit links from Tab
    // traversal. Chromium exposes the complete sequential keyboard order.
    if (!testInfo.project.name.includes("webkit")) {
      await first.locator(selectors[0]).focus();
      for (const selector of selectors.slice(1)) {
        await page.keyboard.press("Tab");
        await expect(first.locator(selector)).toBeFocused();
      }
    }
  });

  test("native summaries toggle with Enter and Space while retaining focus", async ({ page }) => {
    const summaries = page.locator(".ref-details > summary");
    await expect(summaries).toHaveText(["About BOxCrete", "About the 2023 study"]);

    const first = summaries.first();
    await first.focus();
    await page.keyboard.press("Enter");
    await expect(page.locator(".ref-details").first()).toHaveAttribute("open", "");
    await expect(first).toBeFocused();
    await expect(page.locator(".ref-desc").first()).toBeVisible();
    const focusStyle = await first.evaluate((element) => {
      const style = getComputedStyle(element);
      return { style: style.outlineStyle, width: parseFloat(style.outlineWidth) };
    });
    expect(focusStyle.style).not.toBe("none");
    expect(focusStyle.width).toBeGreaterThan(0);

    await page.keyboard.press("Space");
    await expect(page.locator(".ref-details").first()).not.toHaveAttribute("open", "");
    await expect(first).toBeFocused();
    await expect(page.locator(".ref-desc").first()).toBeHidden();
  });

  test("the Meta data-center link is reachable only while BOxCrete details are open", async ({ page }) => {
    const disclosure = page.locator(".ref-details").first();
    const link = page.getByRole("link", { name: "a Meta data center", exact: true });
    await expect(link).toBeHidden();
    await disclosure.locator("summary").click();
    await expect(link).toBeVisible();
    await disclosure.locator("summary").click();
    await expect(link).toBeHidden();
  });

  test("description wrapper animates real geometry and cleans intrinsic state", async ({ page }) => {
    const details = page.locator(".ref-details").first();
    const summary = details.locator("summary");
    const wrapper = details.locator(".ref-description-motion");
    await expect(wrapper).toHaveCount(1);

    const samples = await page.evaluate(async () => {
      const summary = document.querySelector<HTMLElement>(".ref-details > summary")!;
      const wrapper = document.querySelector<HTMLElement>(".ref-description-motion")!;
      summary.focus();
      summary.click();
      const panel = document.querySelector<HTMLElement>("#references-panel")!;
      const actions = document.querySelector<HTMLElement>(".ref-actions-row")!;
      const animation = wrapper.getAnimations()[0];
      if (!animation) throw new Error("reference disclosure animation did not start");
      animation.pause();
      await animation.ready;
      const duration = Number(animation.effect?.getTiming().duration);
      if (!Number.isFinite(duration) || duration <= 0) throw new Error("invalid reference disclosure duration");
      const samples = [];
      for (const progress of [0.2, 0.5, 0.8]) {
        animation.currentTime = duration * progress;
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        samples.push({
          wrapperHeight: wrapper.getBoundingClientRect().height,
          panelHeight: panel.getBoundingClientRect().height,
          actionsLeft: actions.getBoundingClientRect().left,
          actionsRight: actions.getBoundingClientRect().right,
        });
      }
      animation.play();
      return samples;
    });
    expect(samples).toHaveLength(3);
    expect(samples[0].wrapperHeight).toBeGreaterThan(0);
    for (let index = 1; index < samples.length; index += 1) {
      expect(samples[index].wrapperHeight).toBeGreaterThan(samples[index - 1].wrapperHeight + 0.5);
      expect(samples[index].panelHeight).toBeGreaterThanOrEqual(samples[index - 1].panelHeight - 0.5);
    }
    expect(Math.max(...samples.map((sample) => sample.actionsLeft)) - Math.min(...samples.map((sample) => sample.actionsLeft)))
      .toBeLessThanOrEqual(1);
    expect(Math.max(...samples.map((sample) => sample.actionsRight)) - Math.min(...samples.map((sample) => sample.actionsRight)))
      .toBeLessThanOrEqual(1);
    await expect(details).toHaveAttribute("open", "");
    await expect.poll(() => wrapper.evaluate((element) => element.getAnimations().length)).toBe(0);
    expect(await wrapper.evaluate((element: HTMLElement) => ({
      blockSize: element.style.blockSize,
      height: element.style.height,
      overflow: element.style.overflow,
      inert: element.hasAttribute("inert"),
    }))).toEqual({ blockSize: "", height: "", overflow: "", inert: false });
    await expect(summary).toBeFocused();
  });

  test("closing removes descendant tab stops immediately and open only at settlement", async ({ page }) => {
    const details = page.locator(".ref-details").first();
    const summary = details.locator("summary");
    const wrapper = details.locator(".ref-description-motion");
    const link = page.getByRole("link", { name: "a Meta data center", exact: true });
    await summary.click();
    await expect.poll(() => wrapper.evaluate((element) => element.getAnimations().length)).toBe(0);

    await summary.click();

    await expect(wrapper).toHaveAttribute("inert", "");
    await expect(details).toHaveAttribute("open", "");
    expect(await link.evaluate((element) => (element as HTMLElement).tabIndex)).toBe(-1);
    await expect.poll(() => wrapper.evaluate((element) => element.getAnimations().length)).toBe(0);
    await expect(details).not.toHaveAttribute("open", "");
    await expect(summary).toBeFocused();
  });

  test("rapid open close open settles to latest native disclosure state", async ({ page }) => {
    const details = page.locator(".ref-details").first();
    const summary = details.locator("summary");
    const wrapper = details.locator(".ref-description-motion");

    await summary.click();
    await summary.click();
    await summary.click();

    await expect(details).toHaveAttribute("open", "");
    await expect(wrapper).not.toHaveAttribute("inert", "");
    await expect.poll(() => wrapper.evaluate((element) => element.getAnimations().length)).toBe(0);
    expect(await wrapper.evaluate((element: HTMLElement) => ({
      blockSize: element.style.blockSize,
      height: element.style.height,
      overflow: element.style.overflow,
      opacity: element.style.opacity,
      transform: element.style.transform,
    }))).toEqual({ blockSize: "", height: "", overflow: "", opacity: "", transform: "" });
  });

  test("description copy is concise without changing canonical titles or citation payloads", async ({ page }) => {
    const boxcreteBibtex = `@misc{baten2026boxcrete,
  title={BOxCrete: A Bayesian Optimization Open-Source AI Model for Concrete Strength Forecasting and Mix Optimization},
  author={Bayezid Baten and M. Ayyan Iqbal and Sebastian Ament and Julius Kusuma and Nishant Garg},
  year={2026},
  eprint={2603.21525},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2603.21525},
}`;
    const boxcreteApa = "Baten, B., Iqbal, M. A., Ament, S., Kusuma, J., & Garg, N. (2026). BOxCrete: A Bayesian Optimization Open-Source AI Model for Concrete Strength Forecasting and Mix Optimization. arXiv preprint arXiv:2603.21525.";
    const sustainableBibtex = `@misc{ament2023sustainable,
  title={Sustainable Concrete via Bayesian Optimization},
  author={Sebastian Ament and Andrew Witte and Nishant Garg and Julius Kusuma},
  year={2023},
  eprint={2310.18288},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2310.18288},
}`;
    const sustainableApa = "Ament, S., Witte, A., Garg, N., & Kusuma, J. (2023). Sustainable Concrete via Bayesian Optimization. arXiv preprint arXiv:2310.18288.";

    await page.evaluate(() => {
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: {
          writeText(text: string) {
            (window as Window & { __copiedCitation?: string }).__copiedCitation = text;
            return Promise.resolve();
          },
        },
      });
    });
    const expectCopy = async (selector: string, expected: string) => {
      await page.locator(selector).click();
      await expect.poll(() => page.evaluate(() => (window as Window & { __copiedCitation?: string }).__copiedCitation)).toBe(expected);
    };

    await expect(page.locator(".ref-title").nth(1)).toHaveText("Sustainable Concrete via Bayesian Optimization");
    await page.locator(".ref-details").nth(1).locator("summary").click();
    await expect(page.locator(".ref-desc").nth(1)).toHaveText(
      "Introduces the custom Gaussian process model and multi-objective Bayesian optimization for concrete mix design, demonstrating efficient discovery of low-carbon, high-strength mortar.",
    );
    await expectCopy('.ref-cite[data-paper="boxcrete"][data-format="bibtex"]', boxcreteBibtex);
    await expectCopy('.ref-cite[data-paper="boxcrete"][data-format="apa"]', boxcreteApa);
    await expectCopy('.ref-cite[data-paper="sustainable"][data-format="bibtex"]', sustainableBibtex);
    await expectCopy('.ref-cite[data-paper="sustainable"][data-format="apa"]', sustainableApa);
    await expectCopy("#cite-bibtex", `${boxcreteBibtex}\n${sustainableBibtex}`);
    await expectCopy("#cite-apa", `${boxcreteApa}\n\n${sustainableApa}`);
  });

  test("one open description grows naturally without local overflow when it fits", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop reference capacity contract");
    await page.setViewportSize({ width: 1728, height: 1000 });
    await page.goto("/?test=1");
    await waitForDashboardLayoutReady(page);
    await page.locator(".ref-details > summary").first().click();
    await expect.poll(() => page.locator(".ref-description-motion").first().evaluate((element) =>
      element.getAnimations().length,
    )).toBe(0);

    expect(await overflowState(page)).toEqual({ overflow: false, tabIndex: -1 });
    await expect(page.locator(".ref-desc").first()).toBeVisible();
    const visibility = await page.locator(".ref-desc").first().evaluate((element) => {
      const description = element.getBoundingClientRect();
      const list = element.closest(".ref-list")!.getBoundingClientRect();
      return description.bottom <= list.bottom + 1;
    });
    expect(visibility).toBe(true);
  });

  test("oversized References stop at the usable cap and make only the list focusable", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("desktop"), "desktop reference overflow contract");
    await page.setViewportSize({ width: 1280, height: 700 });
    await page.goto("/?test=1");
    await waitForDashboardLayoutReady(page);
    await page.locator(".ref-list").evaluate((element) => {
      const filler = document.createElement("div");
      filler.dataset.testFiller = "true";
      filler.style.cssText = "height:1200px;min-height:1200px;flex:0 0 auto";
      element.appendChild(filler);
      document.dispatchEvent(new CustomEvent("dashboard-layout-change"));
    });
    await expect.poll(() => overflowState(page)).toEqual({ overflow: true, tabIndex: 0 });
    const state = await page.evaluate(() => {
      const rootStyle = getComputedStyle(document.documentElement);
      const panel = document.querySelector<HTMLElement>("#references-panel")!;
      const list = document.querySelector<HTMLElement>(".ref-list")!;
      const panelRect = panel.getBoundingClientRect();
      return {
        cap: parseFloat(rootStyle.getPropertyValue("--references-usable-cap")),
        panelHeight: panelRect.height,
        panelOverflow: getComputedStyle(panel).overflowY,
        listOverflow: getComputedStyle(list).overflowY,
      };
    });
    expect(state.cap).toBeGreaterThan(0);
    expect(state.panelHeight).toBeLessThanOrEqual(state.cap + 1);
    expect(state.panelOverflow).not.toMatch(/auto|scroll/);
    expect(state.listOverflow).toBe("auto");
  });

  test("mobile summaries are 44px targets without horizontal overflow", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile disclosure geometry");
    for (const summary of await page.locator(".ref-details > summary").all()) {
      const box = await summary.boundingBox();
      expect(box, "summary must have a rendered box").not.toBeNull();
      expect(box!.height).toBeGreaterThanOrEqual(44);
    }
    const horizontalRange = await page.evaluate(() =>
      document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(horizontalRange).toBeLessThanOrEqual(1);
  });
});
