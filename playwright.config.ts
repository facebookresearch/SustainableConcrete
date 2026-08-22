import { defineConfig, devices } from "@playwright/test";

const desktopWebKitSmoke = new RegExp([
  "starts with Cement selected",
  "ingredient controls expose button semantics",
  "desktop columns are top-aligned",
  "each managed panel cap",
  "References uses a stable viewport",
  "short desktop preserves useful plots",
  "scroll regions enter sequential focus",
  "Scatter omits instructional copy",
  "desktop plots stay landscape",
  "canvas backing stores track CSS geometry",
  "keeps bibliography and citation actions visible",
  "native summaries toggle with Enter and Space",
  "oversized References stop at the usable cap",
  "both axis objectives remain visible",
  "CSS and JavaScript share the same responsive plot inset contract",
  "filter and reference edge controls reserve outward focus",
].join("|"));

/**
 * Playwright configuration for the BOxCrete interactive web explorer.
 *
 * Tests live in test/e2e/*.spec.ts and run against a local http-server
 * serving the docs/ folder.
 *
 * Projects:
 *   - desktop: 1280×800 Chromium
 *   - mobile: Pixel 7 emulation in Chromium
 *   - mobile-webkit: iPhone 14 emulation for engine-sensitive geometry and page chrome
 *   - desktop-webkit: desktop Safari/WebKit for engine-sensitive layout contracts
 *
 * Tests run in desktop and mobile unless scoped by project name. The WebKit
 * projects intentionally match only engine-sensitive integration specs.
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
    ? [["html", { open: "never" }], ["github"]]
    : [["html", { open: "never" }], ["list"]],

  use: {
    baseURL: "http://127.0.0.1:4173",
    // Record expensive diagnostics only when a failure triggers a retry.
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "on-first-retry",
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
      // Keep the broad mobile suite on Chromium to match Android Chrome.
      use: {
        ...devices["Pixel 7"],
      },
    },
    {
      name: "mobile-webkit",
      // Sticky positioning, overflow clipping, and responsive canvas sizing
      // have engine-specific iOS behavior. Exercise only those integration
      // contracts in WebKit so Safari coverage does not duplicate the full suite.
      testMatch: /(background-coverage|canvas-scheduling|filter-motion|ingredient-insight|mobile-panel-toggle|mobile-slider-layout|panel-geometry|plot-geometry|reduced-motion|references|scatter-toggle|touch-targets|visual-effects)\.spec\.ts/,
      use: {
        ...devices["iPhone 14"],
      },
    },
    {
      name: "desktop-webkit",
      // Keep desktop Safari as a focused compatibility smoke lane. The broader
      // interaction and timing matrices run in Chromium and mobile WebKit.
      testMatch: /(ingredient-insight|panel-geometry|plot-geometry|references|scatter-toggle|visual-effects)\.spec\.ts/,
      grep: desktopWebKitSmoke,
      use: {
        ...devices["Desktop Safari"],
        viewport: { width: 1280, height: 800 },
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
