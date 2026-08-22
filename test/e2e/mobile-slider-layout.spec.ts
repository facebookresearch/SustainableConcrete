import { test, expect } from "@playwright/test";

/**
 * Mobile-only: pin down the multi-row slider layout. Each `.slider-group`
 * stacks vertically as:
 *   row 1: <label> with the ingredient name on the left and the editable
 *          value-input on the right (flex space-between).
 *   row 2: <input type="range"> at full panel content width.
 *   row 3: <div class="info-row"> with min on the left and max on the right.
 *
 * Invariants pinned here:
 *   - Rows do not overlap vertically (no horizontal-collision concerns).
 *   - The slider track has the same width on every row (uniform across
 *     ingredients) and aligned left/right edges.
 *   - The ingredient name's left edge aligns with the info-row min's left
 *     edge (consistent left margin from the panel content edge).
 *   - The value input's rendered glyph aligns with the info-row max
 *     (consistent right margin).
 *   - The value input is at least 32 px tall (touch target).
 *   - The Material Source row's redundant value-span is hidden.
 *   - The slider preview marker (shown when hovering a scatter point) lands
 *     on the visible track, not at x=0 of the panel.
 */
test.describe("mobile slider multi-row layout", () => {
  test.beforeEach(async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile-only layout");
    await page.goto("/");
    // Switch to the Composition view (sliders are hidden by default on mobile)
    await page.locator("#mobile-show-sliders").click();
    await expect(page.locator(".mobile-sliders-view")).toBeVisible({ timeout: 2000 });
    await expect(page.locator(".mobile-sliders-view .slider-group").first()).toBeVisible();
  });

  test("label, slider, and info-row stack vertically without overlap", async ({ page }) => {
    const verdict = await page
      .locator(".mobile-sliders-view .slider-group")
      .first()
      .evaluate((group) => {
        const label = group.querySelector(".slider-label") as HTMLElement | null;
        const slider = group.querySelector("input[type=range]") as HTMLElement | null;
        const info = group.querySelector(".info-row") as HTMLElement | null;
        if (!label || !slider || !info) return null;
        const lr = label.getBoundingClientRect();
        const sr = slider.getBoundingClientRect();
        const ir = info.getBoundingClientRect();
        return {
          labelBottom: lr.bottom,
          sliderTop: sr.top,
          sliderBottom: sr.bottom,
          infoTop: ir.top,
        };
      });
    expect(verdict, "label + slider + info-row should be measurable").not.toBeNull();
    // Each row's top must be at-or-below the previous row's bottom.
    // Allow 1 px sub-pixel rounding.
    expect(
      verdict!.sliderTop,
      `slider top (${verdict!.sliderTop}) must be ≥ label bottom (${verdict!.labelBottom}) — overlap detected`,
    ).toBeGreaterThanOrEqual(verdict!.labelBottom - 1);
    expect(
      verdict!.infoTop,
      `info-row top (${verdict!.infoTop}) must be ≥ slider bottom (${verdict!.sliderBottom}) — overlap detected`,
    ).toBeGreaterThanOrEqual(verdict!.sliderBottom - 1);
  });

  test("ingredient names are left-aligned (consistent left edge across rows)", async ({ page }) => {
    const verdict = await page.evaluate(() => {
      const groups = Array.from(
        document.querySelectorAll<HTMLElement>(".mobile-sliders-view .slider-group"),
      );
      const lefts = groups
        .map((g) => g.querySelector(".ingredient-name") as HTMLElement | null)
        .filter((n): n is HTMLElement => n !== null)
        .map((n) => n.getBoundingClientRect().left);
      return {
        count: lefts.length,
        minL: Math.min(...lefts),
        maxL: Math.max(...lefts),
      };
    });
    expect(verdict.count).toBeGreaterThanOrEqual(5);
    expect(
      verdict.maxL - verdict.minL,
      `ingredient name left edges drift: min=${verdict.minL}, max=${verdict.maxL}`,
    ).toBeLessThanOrEqual(1);
  });

  test("ingredient name and info-row min share the same left edge", async ({ page }) => {
    const verdict = await page
      .locator(".mobile-sliders-view .slider-group")
      .first()
      .evaluate((group) => {
        const name = group.querySelector(".ingredient-name") as HTMLElement | null;
        const min = group.querySelector(".info-row > span:first-child") as HTMLElement | null;
        if (!name || !min) return null;
        return {
          nameLeft: name.getBoundingClientRect().left,
          minLeft: min.getBoundingClientRect().left,
        };
      });
    expect(verdict).not.toBeNull();
    expect(
      Math.abs(verdict!.nameLeft - verdict!.minLeft),
      `name left (${verdict!.nameLeft}) should align with min left (${verdict!.minLeft})`,
    ).toBeLessThanOrEqual(1);
  });

  test("ingredient name and value input share a vertical centerline", async ({ page }) => {
    // Pinned by `align-items: center` on `.slider-label`. Without it
    // the default flex `stretch` makes the value-input fill the row's
    // height while the name's text sits at the line-box top — visually
    // offset. Asserting the children share a center y-coordinate within
    // 2 px catches a regression to the old behaviour.
    const verdict = await page
      .locator(".mobile-sliders-view .slider-group")
      .first()
      .evaluate((group) => {
        const name = group.querySelector(".ingredient-name") as HTMLElement | null;
        const val = group.querySelector(".slider-value") as HTMLElement | null;
        if (!name || !val) return null;
        const cy = (el: HTMLElement) => {
          const r = el.getBoundingClientRect();
          return r.top + r.height / 2;
        };
        return { nameCY: cy(name), valCY: cy(val) };
      });
    expect(verdict).not.toBeNull();
    expect(
      Math.abs(verdict!.nameCY - verdict!.valCY),
      `name center y (${verdict!.nameCY}) should match value-input center y (${verdict!.valCY})`,
    ).toBeLessThanOrEqual(2);
  });

  test("ingredient controls stay compact while the selected item reads as a pill", async ({ page }) => {
    const selected = page.locator('.mobile-sliders-view .ingredient-name[aria-pressed="true"]');
    const inactive = page.locator('.mobile-sliders-view .ingredient-name[aria-pressed="false"]').first();

    await expect(selected).toHaveCount(1);
    const styles = await Promise.all([
      selected.evaluate((el) => {
        const cs = getComputedStyle(el);
        return {
          borderWidth: parseFloat(cs.borderWidth),
          background: cs.backgroundColor,
          height: el.getBoundingClientRect().height,
        };
      }),
      inactive.evaluate((el) => {
        const cs = getComputedStyle(el);
        return {
          background: cs.backgroundColor,
          height: el.getBoundingClientRect().height,
        };
      }),
    ]);

    expect(styles[0].borderWidth).toBeGreaterThan(0);
    expect(styles[0].background).not.toBe("rgba(0, 0, 0, 0)");
    expect(styles[1].background).toBe("rgba(0, 0, 0, 0)");
    expect(styles[0].height).toBeLessThanOrEqual(32);
    expect(styles[1].height).toBeLessThanOrEqual(32);
  });

  test("font-size hierarchy on mobile: label ≥ info-row (legibility)", async ({ page }) => {
    // Regression pin for the mobile bump (label = 0.85rem, info-row =
    // 0.75rem). The label font drives both the name and the value-input;
    // it must be at least as large as the bound labels below to maintain
    // the visual hierarchy `composition value > range bound`.
    const verdict = await page
      .locator(".mobile-sliders-view .slider-group")
      .first()
      .evaluate((group) => {
        const label = group.querySelector(".slider-label") as HTMLElement | null;
        const info = group.querySelector(".info-row") as HTMLElement | null;
        if (!label || !info) return null;
        return {
          labelFs: parseFloat(getComputedStyle(label).fontSize),
          infoFs: parseFloat(getComputedStyle(info).fontSize),
        };
      });
    expect(verdict).not.toBeNull();
    expect(
      verdict!.labelFs,
      `label fontSize (${verdict!.labelFs}px) must be ≥ info-row fontSize (${verdict!.infoFs}px)`,
    ).toBeGreaterThanOrEqual(verdict!.infoFs);
  });

  test("all slider tracks are uniform width and horizontally aligned across rows", async ({ page }) => {
    const verdict = await page.evaluate(() => {
      const sliders = Array.from(
        document.querySelectorAll<HTMLElement>(".mobile-sliders-view input[type=range]"),
      );
      const rects = sliders.map((s) => s.getBoundingClientRect());
      return {
        count: rects.length,
        widths: rects.map((r) => r.width),
        lefts: rects.map((r) => r.left),
        rights: rects.map((r) => r.right),
      };
    });
    expect(verdict.count, "expected at least 5 mobile sliders").toBeGreaterThanOrEqual(5);
    const minW = Math.min(...verdict.widths);
    const maxW = Math.max(...verdict.widths);
    expect(
      maxW - minW,
      `slider widths vary across rows: min=${minW}, max=${maxW}`,
    ).toBeLessThanOrEqual(1);
    const minL = Math.min(...verdict.lefts);
    const maxL = Math.max(...verdict.lefts);
    expect(maxL - minL).toBeLessThanOrEqual(1);
    const minR = Math.min(...verdict.rights);
    const maxR = Math.max(...verdict.rights);
    expect(maxR - minR).toBeLessThanOrEqual(1);
  });

  test("slider uses the panel width while preserving a fixed effect gutter", async ({ page }) => {
    // The slider should use its owner rather than a viewport-relative cap.
    // A small symmetric gutter protects the 5px focus envelope and preview
    // marker without wasting tablet-width panel space.
    const verdict = await page
      .locator(".mobile-sliders-view .slider-group")
      .first()
      .evaluate((group) => {
        const slider = group.querySelector("input[type=range]") as HTMLElement | null;
        if (!slider) return null;
        // The unified mobile panel that hosts the sliders:
        let panel: HTMLElement | null = null;
        for (const cp of Array.from(document.querySelectorAll(".chart-panel"))) {
          if (cp.querySelector("#scatter-canvas")) { panel = cp as HTMLElement; break; }
        }
        if (!panel) return null;
        const cs = getComputedStyle(panel);
        const padL = parseFloat(cs.paddingLeft) || 0;
        const padR = parseFloat(cs.paddingRight) || 0;
        const panelRect = panel.getBoundingClientRect();
        const innerLeft = panelRect.left + padL;
        const innerRight = panelRect.right - padR;
        const innerWidth = innerRight - innerLeft;
        const sr = slider.getBoundingClientRect();
        return {
          marginLeft: sr.left - innerLeft,
          marginRight: innerRight - sr.right,
          sliderWidth: sr.width,
          innerWidth,
        };
      });
    expect(verdict, "slider + panel should be measurable").not.toBeNull();
    // Centered: equal left/right margins from the panel content area.
    expect(
      Math.abs(verdict!.marginLeft - verdict!.marginRight),
      `slider not centered: marginLeft=${verdict!.marginLeft}, marginRight=${verdict!.marginRight}`,
    ).toBeLessThanOrEqual(2);
    expect(verdict!.marginLeft, "focus and preview paint need a local gutter").toBeGreaterThanOrEqual(5);
    expect(verdict!.marginLeft, "viewport-based caps must not waste panel width").toBeLessThanOrEqual(8);
    expect(verdict!.sliderWidth / verdict!.innerWidth).toBeGreaterThanOrEqual(0.94);
  });

  test("value input offsetHeight is ≥ 32px (tap target)", async ({ page }) => {
    // `offsetHeight` is the border-box height — the actual visible/tappable
    // area. `clientHeight` excludes the border, so with `box-sizing:
    // border-box` and a 1 px border, clientHeight = min-height − 2 px.
    // Tap-target accessibility cares about the rendered bounds (offsetHeight).
    const heights = await page
      .locator(".mobile-sliders-view .slider-value")
      .evaluateAll((els) => els.map((el) => (el as HTMLElement).offsetHeight));
    for (const h of heights) {
      expect(h, `slider-value offsetHeight = ${h}px (need ≥ 32)`).toBeGreaterThanOrEqual(32);
    }
  });

  test("Material Source: toggle row visible, redundant value-span hidden", async ({ page }) => {
    const ms = page.locator(".mobile-sliders-view .material-source-group").first();
    await expect(ms).toBeVisible();
    const buttons = ms.locator(".toggle-btn");
    await expect(buttons).toHaveCount(3);
    await expect(buttons.nth(0)).toBeVisible();
    await expect(buttons.nth(1)).toBeVisible();
    await expect(buttons.nth(2)).toHaveText("Source C");
    const buttonGeometry = await buttons.evaluateAll((items) => items.map((item) => {
      const rect = item.getBoundingClientRect();
      const range = document.createRange();
      range.selectNodeContents(item);
      return {
        width: rect.width,
        height: rect.height,
        textLines: range.getClientRects().length,
        overflow: item.scrollWidth - item.clientWidth,
      };
    }));
    expect(new Set(buttonGeometry.map(({ height }) => Math.round(height))).size).toBe(1);
    for (const geometry of buttonGeometry) {
      expect(geometry.textLines).toBe(1);
      expect(geometry.overflow).toBeLessThanOrEqual(1);
      expect(geometry.height).toBeGreaterThanOrEqual(44);
    }
    // The duplicate label-span is the second child of the inner <label>
    const valSpanIsHidden = await ms.evaluate((group) => {
      const label = group.querySelector(".slider-label");
      if (!label) return true;
      const span = label.querySelector("span:last-child");
      return !span || (span as HTMLElement).offsetParent === null;
    });
    expect(valSpanIsHidden).toBe(true);
  });

  test("unit suffix right edge aligns with info-row max bound (right-edge)", async ({ page }) => {
    // The composition setter renders [value-input][unit-suffix] on each row,
    // with the wrap right-padded to match the `.info-row` right inset
    // (mobile: 7 px). The unit `<span>` has `text-align: right`, so its
    // bounding-box right edge IS the rendered glyph end.
    //
    // (Pre-units, this test pinned the value-input's text right edge against
    // the max bound; the unit suffix moved that contract to the unit span,
    // which now anchors the row's right baseline.)
    const verdict = await page
      .locator(".mobile-sliders-view .slider-group")
      .first()
      .evaluate((group) => {
        const unit = group.querySelector(".slider-unit") as HTMLElement | null;
        const info = group.querySelector(".info-row") as HTMLElement | null;
        const maxSpan = info?.querySelector("span:last-child") as HTMLElement | null;
        if (!unit || !info || !maxSpan) return null;
        const unitTextRight = unit.getBoundingClientRect().right;
        const maxRight = maxSpan.getBoundingClientRect().right;
        return { unitTextRight, maxRight };
      });
    expect(verdict, "unit suffix + info-row max should be measurable").not.toBeNull();
    expect(
      Math.abs(verdict!.unitTextRight - verdict!.maxRight),
      `unit text end (${verdict!.unitTextRight}) should align with max bound (${verdict!.maxRight})`,
    ).toBeLessThanOrEqual(1);
  });

  test("slider preview marker lands on the visible track when scatter point is hovered", async ({ page }) => {
    // Click a scatter point to drive the slider preview compositions through
    // animateToComposition. The marker positions are computed from
    // `slider.offsetLeft + thumbHalf + fraction * trackWidth`. After the
    // change to slider preview marker math, the marker should land within
    // the slider's own bounding box on mobile too.
    await page.locator("#mobile-show-scatter").click();
    await expect(page.locator(".scatter-content")).toBeVisible();

    const canvas = page.locator("canvas#scatter-canvas");
    const box = await canvas.boundingBox();
    if (!box) throw new Error("scatter canvas has no bounding box");
    await canvas.hover({ position: { x: box.width * 0.5, y: box.height * 0.5 } });

    // Switch back to sliders view to inspect the markers
    await page.locator("#mobile-show-sliders").click();
    await expect(page.locator(".mobile-sliders-view")).toBeVisible();
    await page.waitForTimeout(300);

    const verdict = await page.evaluate(() => {
      const groups = Array.from(document.querySelectorAll(".mobile-sliders-view .slider-group"));
      let onTrack = 0;
      let total = 0;
      for (const g of groups) {
        const slider = g.querySelector("input[type=range]") as HTMLElement | null;
        const marker = g.querySelector(".slider-preview-marker") as HTMLElement | null;
        if (!slider || !marker || marker.style.display === "none") continue;
        total++;
        const sr = slider.getBoundingClientRect();
        const mr = marker.getBoundingClientRect();
        const mx = mr.left + mr.width / 2;
        if (mx >= sr.left - 1 && mx <= sr.right + 1) onTrack++;
      }
      const eligible = groups.filter((group) => group.querySelector("input[type=range]")).length;
      return { onTrack, total, eligible };
    });
    expect(verdict.total, "scatter preview should mark every slider").toBe(verdict.eligible);
    expect(
      verdict.onTrack,
      `markers off-track: ${verdict.total - verdict.onTrack}/${verdict.total}`,
    ).toBe(verdict.total);
  });
});
