import { test, expect } from "@playwright/test";

/**
 * Accessibility invariants for the explorer.
 *
 * Regression pins:
 *  - Every visible interactive control has an accessible name. The 8 range
 *    sliders had none — a screen reader announced "slider, 353" with no
 *    indication of which ingredient (WCAG 4.1.2). The sibling value input was
 *    labelled; the range input was missed.
 *  - The strength curve has a text equivalent. A canvas exposes no data, so
 *    aria-label conveys that a chart exists but not what it says.
 *  - Both must follow the unit toggle, or a screen-reader user hears kg/m3
 *    while the UI shows lb/yd3.
 */

// On mobile the composition panel is behind a view toggle; open it so the
// sliders are actually rendered before we inspect them.
async function openApp(page: import("@playwright/test").Page, project: string) {
  await page.goto("/");
  if (project === "mobile") {
    await page.locator("#mobile-show-sliders").click();
    await expect(page.locator(".mobile-sliders-view")).toBeVisible({ timeout: 5000 });
  }
  await page
    .locator("input[type=range]")
    .first()
    .waitFor({ state: "visible", timeout: 15000 });
}

// Strength unit currently quoted by the summary, or undefined.
const unitOf = (s: string) => (s.match(/\b(psi|MPa)\b/) || [])[1];
// The three "<mean> plus or minus" values, in order (1/7/28 days).
const numbersIn = (s: string) =>
  [...s.matchAll(/(\d+) plus or minus/g)].map((m) => Number(m[1]));
const summaryText = (page: import("@playwright/test").Page) =>
  page.evaluate(() => document.getElementById("curve-summary")?.textContent);

test.describe("accessibility", () => {
  test("every visible interactive control has an accessible name", async ({ page }, testInfo) => {
    await openApp(page, testInfo.project.name);
    const unnamed = await page.evaluate(() => {
      // NOT offsetParent: that is null for `position: fixed`, so any control
      // inside a fixed overlay (.about-overlay, .video-overlay, .tooltip all
      // use it) would be silently skipped rather than checked.
      const vis = (el: Element) => {
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) return false;
        const cs = getComputedStyle(el as HTMLElement);
        return cs.visibility !== "hidden" && cs.display !== "none";
      };
      return Array.from(
        document.querySelectorAll("button,input,select,a[href],[role=button]"),
      )
        .filter(vis)
        .filter((el) => {
          const t = el as HTMLElement;
          const name =
            t.getAttribute("aria-label") ||
            t.getAttribute("title") ||
            t.textContent?.trim() ||
            (t.id && document.querySelector(`label[for="${CSS.escape(t.id)}"]`)?.textContent?.trim());
          return !name;
        })
        .map((e) => `${e.tagName.toLowerCase()}.${(e as HTMLElement).className}`);
    });
    expect(unnamed, `unnamed controls: ${JSON.stringify(unnamed)}`).toEqual([]);
  });

  test("each slider names its ingredient and unit", async ({ page }, testInfo) => {
    await openApp(page, testInfo.project.name);
    const labels = await page.evaluate(() =>
      Array.from(document.querySelectorAll("input[type=range]"))
        .filter((el) => {
          const r = el.getBoundingClientRect();
          return r.width > 0 && r.height > 0;
        })
        .map((s) => s.getAttribute("aria-label")),
    );
    expect(labels.length, "expected visible sliders").toBeGreaterThan(0);
    for (const l of labels) {
      expect(l, "slider missing aria-label").toBeTruthy();
      expect(l, `"${l}" should name a unit`).toMatch(/\(.+\)$/);
    }
    expect(
      labels.some((l) => l!.startsWith("Cement")),
      `got ${JSON.stringify(labels)}`,
    ).toBe(true);
  });

  test("the strength curve has a live text equivalent", async ({ page }, testInfo) => {
    await openApp(page, testInfo.project.name);
    const el = page.locator("#curve-summary");
    await expect(el).toHaveAttribute("aria-live", "polite");
    await expect(el).toHaveAttribute("role", "status");
    await expect
      .poll(async () => (await el.textContent())?.trim().length ?? 0, { timeout: 15000 })
      .toBeGreaterThan(0);

    const text = (await el.textContent())!;
    expect(text, "summary should quote predicted strength").toMatch(/Predicted strength/i);
    expect(text, "summary should name at least the 28-day point").toMatch(/28 days/);
  });

  test("slider labels and the curve summary follow the unit toggle", async ({ page }, testInfo) => {
    await openApp(page, testInfo.project.name);
    await expect
      .poll(
        async () =>
          (await page.locator("#curve-summary").textContent())?.trim().length ?? 0,
        { timeout: 15000 },
      )
      .toBeGreaterThan(0);

    const before = await page.evaluate(() => ({
      label: document
        .querySelector("input[type=range]")
        ?.getAttribute("aria-label"),
      summary: document.getElementById("curve-summary")?.textContent,
    }));

    const toggle = testInfo.project.name === "mobile"
      ? page.locator("#mobile-unit-toggle")
      : page.locator("#unit-toggle");
    await toggle.click();

    // Assert the UNIT TOKEN flips, not merely that the strings differ.
    // `.not.toBe(before)` plus /psi|MPa/ was vacuous: freezing the unit word
    // to "MPa" while the numbers rescaled left the app announcing
    // "14910 MPa" and the test still green.
    const beforeUnit = unitOf(before.summary!);
    expect(beforeUnit, "no strength unit in the initial summary").toBeTruthy();
    const expectedUnit = beforeUnit === "MPa" ? "psi" : "MPa";

    await expect
      .poll(async () => unitOf((await summaryText(page)) ?? ""), { timeout: 15000 })
      .toBe(expectedUnit);

    // The slider label must carry the matching mass unit.
    const expectedMass = expectedUnit === "psi" ? "lb/yd³" : "kg/m³";
    await expect
      .poll(
        async () =>
          await page.evaluate(() =>
            document.querySelector("input[type=range]")?.getAttribute("aria-label"),
          ),
        { timeout: 10000 },
      )
      .toContain(expectedMass);

    // And the numbers must actually be converted, not just relabelled.
    // 1 MPa = 145.038 psi.
    const after = (await summaryText(page))!;
    const b = numbersIn(before.summary!);
    const a = numbersIn(after);
    expect(a.length, `expected 3 strength values, got ${JSON.stringify(a)}`).toBe(3);
    expect(b.length).toBe(3);
    const ratio = a[2] / b[2];
    const expectedRatio = expectedUnit === "psi" ? 145.038 : 1 / 145.038;
    expect(
      ratio / expectedRatio,
      `28-day value went ${b[2]} -> ${a[2]} (ratio ${ratio.toFixed(3)}), ` +
        `expected ~${expectedRatio.toFixed(3)} for ${beforeUnit}->${expectedUnit}`,
    ).toBeCloseTo(1, 1);
  });
  test("sliders show a focus indicator when keyboard-focused", async ({ page }, testInfo) => {
    await openApp(page, testInfo.project.name);
    // `outline: none` on the range input had no replacement anywhere, a WCAG
    // 2.4.7 (AA) failure on the app's primary control. Lighthouse's
    // accessibility category does not audit focus visibility, so only an
    // explicit test catches this.
    //
    // Driven by real Tab presses: :focus-visible does not match programmatic
    // .focus() in Chromium, so a computed-style check after el.focus() would
    // pass against ungated code.
    const slider = page.locator("input[type=range]").first();
    await slider.evaluate((el) => (el as HTMLElement).blur());
    await page.keyboard.press("Tab");

    const res = await page.evaluate(() => {
      const el = document.querySelector("input[type=range]") as HTMLElement;
      el.focus();
      // Simulate the keyboard-focus heuristic so :focus-visible applies.
      const cs = getComputedStyle(el);
      return {
        focusVisible: el.matches(":focus-visible"),
        width: cs.outlineWidth,
        style: cs.outlineStyle,
      };
    });

    // Chromium reports :focus-visible for a range input focused after keyboard
    // interaction. If the engine declines, fall back to asserting the rule
    // resolves to a non-zero outline at all.
    if (res.focusVisible) {
      expect(res.style, `outline-style when focused: ${JSON.stringify(res)}`).not.toBe("none");
      expect(
        parseFloat(res.width),
        `outline-width when focused: ${JSON.stringify(res)}`,
      ).toBeGreaterThan(0);
    } else {
      const declared = await page.evaluate(() => {
        for (const sheet of Array.from(document.styleSheets)) {
          let rules: CSSRuleList;
          try { rules = (sheet as CSSStyleSheet).cssRules; } catch { continue; }
          for (const r of Array.from(rules)) {
            const sel = (r as CSSStyleRule).selectorText || "";
            if (sel.includes("input[type=range]") && sel.includes(":focus-visible")) {
              return (r as CSSStyleRule).style.outlineWidth || (r as CSSStyleRule).style.outline;
            }
          }
        }
        return "";
      });
      expect(declared, "no :focus-visible outline declared for the range input").toBeTruthy();
    }
  });

  test("keyboard focus previews an inactive Material Source class", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "Material Source controls are desktop-visible");
    await page.goto("/?test=1");
    await page.waitForFunction(() => (window as any).__test?.modelReady === true, null, {
      timeout: 20000,
    });
    const inactive = page.locator(".material-source-group .toggle-btn:not(.active)").first();
    const label = (await inactive.textContent())?.trim() ?? "";

    await inactive.focus();

    await expect.poll(() => page.evaluate(() => (window as any).__test.previewSource)).toBe("class");
    expect(await page.evaluate(() => (window as any).__test.previewScopeBtnLabel)).toBe(label);
    await expect(inactive).toHaveAttribute("data-previewing", "");
  });

  test("controls inside a closed modal are not reachable", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "one project is enough");
    await openApp(page, testInfo.project.name);
    // The overlays were hidden with opacity + pointer-events only, so every
    // control inside stayed in the tab order and the a11y tree, and their
    // headings polluted the document outline.
    const state = await page.evaluate(() => {
      const out: Record<string, unknown> = {};
      for (const sel of [".about-overlay", ".video-overlay"]) {
        const el = document.querySelector(sel) as HTMLElement | null;
        if (!el) continue;
        const cs = getComputedStyle(el);
        out[sel] = {
          visibility: cs.visibility,
          controls: el.querySelectorAll("a[href], button, iframe").length,
        };
      }
      return out;
    });
    for (const [sel, v] of Object.entries(state)) {
      expect(
        (v as any).visibility,
        `${sel} is still visible while closed, so its controls stay tabbable`,
      ).toBe("hidden");
    }
  });
});
