import { test, expect } from "@playwright/test";

/**
 * Scatter filter UI invariants. Filters are a multi-row "+/−" interface
 * inside the scatter panel; each row has a column-select dropdown and
 * two narrow `<input type="number">` boxes for min/max bounds.
 *
 * Regression pin: `<input type="number">` natively renders spinner
 * buttons that consume horizontal space; Chromium shows them by default,
 * which used to clip the "min" and "max" placeholders to "mi" and "ma".
 * Mobile browsers hide the spinners by default, so the bug only ever
 * appeared on desktop.
 *
 * The CSS fix sets `appearance: textfield` and zeros the
 * `::-webkit-inner-spin-button` / `::-webkit-outer-spin-button`
 * pseudo-elements. This spec asserts the placeholder still fits in the
 * input's visible content area.
 */
test.describe("scatter filter rows", () => {
  test("filter min/max placeholders fit fully inside the input box", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "filter UI is desktop-only — hidden on mobile");
    await page.goto("/");
    // The filter controls are wired by setupEventListeners(), which runs after
    // the strength model resolves (now off-thread in a worker). Wait on a
    // rendered slider rather than clicking straight into an unwired button.
    await expect(page.locator("#sliders .slider-group").first()).toBeVisible({ timeout: 15000 });
    // Add a filter row by clicking the "+" button
    await page.locator("#filter-add").click();
    const minInput = page.locator(".filter-min").first();
    const maxInput = page.locator(".filter-max").first();
    await expect(minInput).toBeVisible();
    await expect(maxInput).toBeVisible();

    // Measure the placeholder text width vs the input's available content
    // width (clientWidth - horizontal padding). Use an offscreen canvas
    // with the input's computed font so the measurement matches what's
    // rendered inside the input.
    async function check(locator: import("@playwright/test").Locator) {
      const verdict = await locator.evaluate((el) => {
        const input = el as HTMLInputElement;
        const cs = getComputedStyle(input);
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        if (!ctx) return null;
        const fontWeight = cs.fontWeight || "normal";
        const fontSize = cs.fontSize || "12px";
        const fontFamily = cs.fontFamily || "sans-serif";
        ctx.font = `${fontWeight} ${fontSize} ${fontFamily}`;
        const textWidth = ctx.measureText(input.placeholder).width;
        const padL = parseFloat(cs.paddingLeft) || 0;
        const padR = parseFloat(cs.paddingRight) || 0;
        const available = input.clientWidth - padL - padR;
        return { placeholder: input.placeholder, textWidth, available };
      });
      expect(verdict, "input must be measurable").not.toBeNull();
      // Allow 1px for sub-pixel rounding. If the spinner is showing, the
      // available width is reduced by ~14 px and this fails clearly.
      expect(
        verdict!.textWidth,
        `placeholder "${verdict!.placeholder}" textWidth=${verdict!.textWidth}px > available=${verdict!.available}px`,
      ).toBeLessThanOrEqual(verdict!.available + 1);
    }
    await check(minInput);
    await check(maxInput);
  });
});

test.describe("categorical filter rows", () => {
  // Material Source is an unordered class axis, so a numeric min/max range is
  // meaningless for it ("between Source A and Source B" says nothing). Its
  // filter row must render one toggle per class instead.
  test("Material Source filters by class, not by min/max", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "filter UI is desktop-only — hidden on mobile");
    await page.goto("/");
    // Same readiness wait as above: the filter UI is wired after the model
    // resolves in the worker.
    await expect(page.locator("#sliders .slider-group").first()).toBeVisible({ timeout: 15000 });

    const msIdx = await page.evaluate(async () => {
      const j = await (await fetch("model/compositions.json")).json();
      return j.column_names.indexOf("Material Source");
    });
    expect(msIdx, "Material Source column not found").toBeGreaterThanOrEqual(0);

    await page.locator("#filter-add").click();
    const row = page.locator(".filter-row").first();

    // A numeric column keeps the min/max inputs.
    await expect(row.locator(".filter-min")).toBeVisible();
    expect(await row.locator(".filter-cat-btn").count()).toBe(0);

    // Selecting Material Source swaps them for per-class toggles.
    await row.locator(".filter-col").selectOption(String(msIdx));
    const catBtns = row.locator(".filter-cat-btn");
    await expect(catBtns.first()).toBeVisible();
    const nCat = await catBtns.count();
    expect(nCat, "expected one toggle per material class").toBeGreaterThan(1);
    expect(
      await row.locator(".filter-min").count(),
      "min/max inputs must not be shown for a categorical column",
    ).toBe(0);

    // All classes start active, so adding the row is a no-op — the same
    // default as empty min/max meaning "unbounded".
    expect(await row.locator(".filter-cat-btn.active").count()).toBe(nCat);

    // Toggling a class off narrows the filter and greys out its points.
    await catBtns.first().click();
    expect(await row.locator(".filter-cat-btn.active").count()).toBe(nCat - 1);
    await expect(catBtns.first()).toHaveAttribute("aria-pressed", "false");

    // Switching back to a numeric column restores the min/max inputs.
    await row.locator(".filter-col").selectOption("0");
    await expect(row.locator(".filter-min")).toBeVisible();
    expect(await row.locator(".filter-cat-btn").count()).toBe(0);
  });
});
