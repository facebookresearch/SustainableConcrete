// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import { test, expect } from "@playwright/test";

/**
 * SEO basics pin: meta description, canonical URL, JSON-LD WebApplication
 * structured data, robots.txt, and sitemap.xml. These collectively make the
 * site discoverable and well-cited by search engines and AI assistants.
 */
test.describe("SEO metadata and crawler files", () => {
  test("meta description, canonical, and JSON-LD WebApplication are present", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "metadata is identical across viewports");
    await page.goto("/");

    const description = await page.locator('meta[name="description"]').first().getAttribute("content");
    expect(description?.length ?? 0).toBeGreaterThanOrEqual(50);
    expect(description?.length ?? 0).toBeLessThanOrEqual(300);

    const canonical = await page.locator('link[rel="canonical"]').first().getAttribute("href");
    expect(canonical).toContain("facebookresearch.github.io/SustainableConcrete");

    // JSON-LD parses to a WebApplication
    const jsonLd = await page.locator('script[type="application/ld+json"]').first().textContent();
    expect(jsonLd).toBeTruthy();
    const parsed = JSON.parse(jsonLd!);
    expect(parsed["@type"]).toBe("WebApplication");
    expect(parsed.name).toBeTruthy();
    expect(parsed.description).toBeTruthy();
  });

  test("robots.txt and sitemap.xml are reachable and well-formed", async ({ request }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "asset fetch is identical across viewports");

    const robots = await request.get("/robots.txt");
    expect(robots.status()).toBe(200);
    const robotsText = await robots.text();
    expect(robotsText).toMatch(/User-agent:\s*\*/);
    expect(robotsText).toMatch(/Allow:\s*\//);

    const sitemap = await request.get("/sitemap.xml");
    expect(sitemap.status()).toBe(200);
    const sitemapText = await sitemap.text();
    expect(sitemapText).toContain("facebookresearch.github.io/SustainableConcrete");
    expect(sitemapText).toMatch(/<urlset/);
  });
});
