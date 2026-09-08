import { test, expect } from "@playwright/test";

/**
 * Open Graph + Twitter Card meta tag pin. Social previews (Facebook, LinkedIn,
 * Slack, iMessage, Discord, X) read these tags to render link cards. If the
 * tags drift, the cards either fall back to a generic preview or break.
 *
 * Also checks the og-image asset itself: must be a reachable JPEG within a
 * sensible size range so the page doesn't ship a multi-MB image.
 */
test.describe("Open Graph + Twitter Card metadata", () => {
  test("required OG/Twitter tags are present with expected values", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "metadata is identical across viewports");
    await page.goto("/");

    const get = (sel: string) =>
      page.locator(sel).first().getAttribute("content");

    expect(await get('meta[property="og:title"]')).toMatch(/BOxCrete/i);
    expect((await get('meta[property="og:description"]'))?.length ?? 0).toBeGreaterThan(20);
    expect(await get('meta[property="og:image"]')).toContain("og-image.jpg");
    expect(await get('meta[property="og:image:width"]')).toBe("1200");
    expect(await get('meta[property="og:image:height"]')).toBe("630");
    expect(await get('meta[property="og:url"]')).toContain("facebookresearch.github.io/SustainableConcrete");

    expect(await get('meta[name="twitter:card"]')).toBe("summary_large_image");
    expect(await get('meta[name="twitter:image"]')).toContain("og-image.jpg");
  });

  test("og-image.jpg loads as a JPEG within a reasonable size budget", async ({ request }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "asset fetch is identical across viewports");
    const resp = await request.get("/og-image.jpg");
    expect(resp.status(), "og-image.jpg must be reachable").toBe(200);
    const ct = resp.headers()["content-type"] ?? "";
    expect(ct, `unexpected content-type: ${ct}`).toMatch(/image\/jpeg/i);
    const buf = await resp.body();
    expect(
      buf.byteLength,
      `og-image.jpg size out of budget (${buf.byteLength} bytes)`,
    ).toBeGreaterThan(50 * 1024);
    expect(buf.byteLength).toBeLessThan(250 * 1024);
  });
});
