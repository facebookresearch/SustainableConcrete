import { expect, type Page } from "@playwright/test";

export async function waitForDashboardLayoutReady(page: Page): Promise<void> {
  await expect(page.locator("#sliders .slider-group").last()).toBeAttached({
    timeout: 15_000,
  });
  await page.evaluate(async () => {
    await document.fonts.ready;
    await Promise.all(
      [...document.querySelectorAll(".fade-in-up")].flatMap((element) =>
        element.getAnimations().map((animation) => animation.finished),
      ),
    );
  });
}

export async function openMobileComposition(page: Page): Promise<void> {
  const toggle = page.locator("#mobile-show-sliders");
  if (await toggle.getAttribute("aria-pressed") === "false") {
    await toggle.click();
  }

  await expect(toggle).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".mobile-sliders-view")).toBeVisible();
  await expect(page.locator(".mobile-sliders-view")).not.toHaveAttribute("inert", /.*/);
  await expect(page.locator(".scatter-content")).toBeHidden();
  await expect(page.locator("#sliders input[type=range]").first()).toBeVisible();
}
