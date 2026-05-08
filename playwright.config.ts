import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright configuration for the BOxCrete interactive web explorer.
 *
 * Tests live in test/e2e/*.spec.ts and run against a local http-server
 * serving the docs/ folder.
 *
 * Three projects:
 *   - desktop:          1280×800 Chromium
 *   - mobile:           Pixel 7 emulation (touch + 412×915 viewport)
 *   - tablet-landscape: iPad (gen 7) landscape (1080×810, touch) — forced
 *                       onto Chromium so CI doesn't have to install WebKit.
 *
 * Each test runs once per project unless skipped via testInfo.project.name.
 *
 * See test/e2e/README.md for the catalogue of invariants.
 */
export default defineConfig({
  testDir: "./test/e2e",
  testMatch: /.*\.spec\.ts/,
  fullyParallel: true,
  // Fail fast on accidentally-committed `.only`
  forbidOnly: !!process.env.CI,
  // Retry transient failures (animations, network) twice on CI; never locally
  retries: process.env.CI ? 2 : 0,
  // CI: serial workers for stable timing; local: full parallelism
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI
    ? [["html", { open: "never" }], ["github"], ["list"]]
    : [["html", { open: "never" }], ["list"]],

  use: {
    baseURL: "http://127.0.0.1:4173",
    // Capture diagnostics only on failure to keep artifacts small
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    // Wait for actions to complete before timing out (animations, etc.)
    actionTimeout: 10_000,
    navigationTimeout: 30_000,
  },

  projects: [
    {
      name: "desktop",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1280, height: 800 },
      },
    },
    {
      name: "mobile",
      // Use Chrome with iPhone-like viewport instead of devices['iPhone 14']
      // (which defaults to WebKit). This keeps CI fast (Chromium only) and
      // matches the engine real Android Chrome users will hit.
      use: {
        ...devices["Pixel 7"],
      },
    },
    {
      name: "tablet-landscape",
      // iPad gen 7 (1080 × 810) is representative of the iPad-landscape band:
      // covers the cramped end (1024 × 768 on gen 5/6/Mini/9) reasonably and
      // the comfortable end (1194 × 834 on iPad Pro 11) without two projects.
      // Force Chromium so CI doesn't need a WebKit install — same trade-off as
      // the mobile project above.
      use: {
        ...devices["iPad (gen 7) landscape"],
        browserName: "chromium",
      },
    },
  ],

  webServer: {
    command: "npx http-server docs -p 4173 -s -c-1 -a 127.0.0.1",
    url: "http://127.0.0.1:4173/index.html",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },

  expect: {
    // Visual snapshot tolerance — small differences across runners are normal
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.02,
      // Fonts and gradients sometimes shift sub-pixels; ignore tiny diffs
      threshold: 0.2,
    },
  },
});
