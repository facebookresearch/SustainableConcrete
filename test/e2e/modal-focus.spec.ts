import { test, expect } from "@playwright/test";

/**
 * Modal focus management (WCAG 2.4.3 Focus Order, 4.1.2 Name/Role/Value).
 *
 * The overlays previously had no dialog semantics and no focus handling at
 * all: opening one left focus behind on the page, closing it dropped the user
 * at the top of the document, and Tab walked straight out into the page
 * behind the dialog.
 */
test.describe("modal focus management", () => {
  test.beforeEach(async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "About link is desktop-visible");
    await page.goto("/");
    await page.locator("#sliders input[type=range]").first().waitFor({ timeout: 15000 });
  });

  test("the dialog is announced as a dialog and is labelled", async ({ page }) => {
    const dialog = page.locator("#about-overlay [role=dialog]");
    await expect(dialog).toHaveAttribute("aria-modal", "true");
    const labelledby = await dialog.getAttribute("aria-labelledby");
    expect(labelledby, "dialog needs an accessible name").toBeTruthy();
    await expect(page.locator(`#${labelledby}`)).toHaveText(/About/i);
  });

  test("opening moves focus into the dialog", async ({ page }) => {
    await page.locator("#about-link").click();
    await expect(page.locator("#about-overlay")).toHaveClass(/visible/);
    const inside = await page.evaluate(() => {
      const d = document.querySelector("#about-overlay [role=dialog]")!;
      return d.contains(document.activeElement);
    });
    expect(inside, "focus should land inside the dialog, not stay behind it").toBe(true);
  });

  test("closing restores focus to the trigger", async ({ page }) => {
    await page.locator("#about-link").click();
    await expect(page.locator("#about-overlay")).toHaveClass(/visible/);
    await page.keyboard.press("Escape");
    await expect(page.locator("#about-overlay")).not.toHaveClass(/visible/);
    const id = await page.evaluate(() => document.activeElement?.id);
    expect(id, "focus should return to the element that opened the dialog").toBe("about-link");
  });

  test("opening a second modal transfers exclusive ownership", async ({ page }) => {
    await page.locator("#about-link").click();
    await expect(page.locator("#about-overlay")).toHaveClass(/visible/);

    await page.evaluate(() => {
      (document.getElementById("video-link") as HTMLElement).click();
    });

    await expect(page.locator("#about-overlay")).not.toHaveClass(/visible/);
    await expect(page.locator("#video-overlay")).toHaveClass(/visible/);
    await expect(page.locator(".about-overlay.visible, .video-overlay.visible")).toHaveCount(1);
  });

  test("reopening video before fade cleanup keeps the player source", async ({ page }) => {
    await page.locator("#video-link").click();
    const overlay = page.locator("#video-overlay");
    const iframe = page.locator("#video-iframe");
    await expect(overlay).toHaveClass(/visible/);
    await expect(iframe).toHaveAttribute("src", /youtube-nocookie/);

    await overlay.locator("[data-modal-close]").click();
    await page.locator("#video-link").click();
    await page.waitForTimeout(300);

    await expect(overlay).toHaveClass(/visible/);
    await expect(iframe).toHaveAttribute("src", /youtube-nocookie/);
  });

  test("Tab is trapped inside the open dialog", async ({ page }) => {
    await page.locator("#about-link").click();
    await expect(page.locator("#about-overlay")).toHaveClass(/visible/);

    // Walk well past the number of focusables; focus must never escape.
    for (let i = 0; i < 12; i++) {
      await page.keyboard.press("Tab");
      const inside = await page.evaluate(() => {
        const d = document.querySelector("#about-overlay [role=dialog]")!;
        return d.contains(document.activeElement);
      });
      expect(inside, `focus escaped the dialog after ${i + 1} Tab presses`).toBe(true);
    }
    // And backwards.
    for (let i = 0; i < 4; i++) {
      await page.keyboard.press("Shift+Tab");
      const inside = await page.evaluate(() => {
        const d = document.querySelector("#about-overlay [role=dialog]")!;
        return d.contains(document.activeElement);
      });
      expect(inside, `focus escaped backwards after ${i + 1} Shift+Tab presses`).toBe(true);
    }
  });
});
