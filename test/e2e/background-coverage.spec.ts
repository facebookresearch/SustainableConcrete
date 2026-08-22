import { expect, test, type Page } from "@playwright/test";

type BackgroundState = NonNullable<Awaited<ReturnType<typeof pageChromeState>>>;

async function pageChromeState(page: Page) {
  return page.evaluate(() => {
    const background = document.querySelector<HTMLElement>(".bg-layer");
    const headers = Array.from(document.querySelectorAll<HTMLElement>(".site-header"));
    const header = headers[0];
    if (!background || !header) return null;

    const backgroundRect = background.getBoundingClientRect();
    const backgroundStyle = getComputedStyle(background);
    const headerRect = header.getBoundingClientRect();
    const headerStyle = getComputedStyle(header);
    const foreground = document.elementFromPoint(
      headerRect.left + headerRect.width / 2,
      Math.max(1, headerRect.top + Math.min(8, headerRect.height / 2)),
    );

    return {
      background: {
        cssPosition: backgroundStyle.position,
        pointerEvents: backgroundStyle.pointerEvents,
        size: backgroundStyle.backgroundSize,
        focalPosition: backgroundStyle.backgroundPosition,
        repeat: backgroundStyle.backgroundRepeat,
        image: backgroundStyle.backgroundImage,
        rect: {
          x: backgroundRect.x,
          y: backgroundRect.y,
          width: backgroundRect.width,
          height: backgroundRect.height,
        },
      },
      viewport: { width: window.innerWidth, height: window.innerHeight },
      header: {
        count: headers.length,
        position: headerStyle.position,
        top: headerRect.top,
        bottom: headerRect.bottom,
        ownsForegroundPoint: foreground !== null && header.contains(foreground),
      },
      scrollY: window.scrollY,
      horizontalOverflow:
        document.documentElement.scrollWidth - document.documentElement.clientWidth,
    };
  });
}

function expectFixedPageChrome(state: BackgroundState, stateName: string) {
  expect(state.background.image, `${stateName}: photograph must render`).not.toBe("none");
  expect(state.background.cssPosition, `${stateName}: background must be viewport-fixed`).toBe(
    "fixed",
  );
  expect(state.background.pointerEvents, `${stateName}: background must not intercept input`).toBe(
    "none",
  );
  expect(state.background.size, `${stateName}: photograph must cover the viewport`).toBe("cover");
  expect(state.background.focalPosition, `${stateName}: photograph must retain an explicit focal position`).not.toBe("");
  expect(state.background.repeat, `${stateName}: photograph must not tile`).toBe("no-repeat");
  expect(Math.abs(state.background.rect.x), `${stateName}: background left edge`).toBeLessThanOrEqual(1);
  expect(Math.abs(state.background.rect.y), `${stateName}: background top edge`).toBeLessThanOrEqual(1);
  expect(
    Math.abs(state.background.rect.width - state.viewport.width),
    `${stateName}: background width must match viewport`,
  ).toBeLessThanOrEqual(1);
  expect(
    Math.abs(state.background.rect.height - state.viewport.height),
    `${stateName}: background height must match viewport`,
  ).toBeLessThanOrEqual(1);

  expect(state.header.count, `${stateName}: header must render exactly once`).toBe(1);
  expect(state.header.position, `${stateName}: header must remain sticky`).toBe("sticky");
  expect(Math.abs(state.header.top), `${stateName}: header must stay at viewport top`).toBeLessThanOrEqual(
    1,
  );
  expect(state.header.bottom, `${stateName}: header must remain visible`).toBeGreaterThan(1);
  expect(state.header.ownsForegroundPoint, `${stateName}: header must own its foreground`).toBe(true);
  expect(state.horizontalOverflow, `${stateName}: page must not overflow horizontally`).toBeLessThan(2);
}

function expectSameBackgroundFrame(
  before: BackgroundState,
  after: BackgroundState,
  stateName: string,
) {
  expect(after.background.rect, `${stateName}: background viewport rectangle changed`).toEqual(
    before.background.rect,
  );
  expect(after.background.size, `${stateName}: background sizing changed`).toBe(before.background.size);
  expect(after.background.focalPosition, `${stateName}: background focal position changed`).toBe(before.background.focalPosition);
  expect(after.background.image, `${stateName}: background image changed`).toBe(before.background.image);
}

async function injectOversizedContent(page: Page) {
  await page.evaluate(() => {
    for (const selector of [".mix-insight-text", ".ingredient-insight-text", ".ref-list"]) {
      const target = document.querySelector<HTMLElement>(selector);
      if (!target) throw new Error(`Missing injection target: ${selector}`);
      const filler = document.createElement("p");
      filler.dataset.testFiller = "true";
      filler.textContent = "Stable viewport background test content. ".repeat(120);
      target.appendChild(filler);
    }
  });
}

test.describe("viewport-fixed background and sticky header", () => {
  test("desktop background frame survives panel scrolling and content growth", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop dashboard invariant");
    await page.goto("/");
    await expect(page.locator("#sliders .slider-group").last()).toBeVisible({ timeout: 15_000 });

    const initial = await pageChromeState(page);
    expect(initial, "initial page chrome must exist").not.toBeNull();
    expectFixedPageChrome(initial!, "initial load");

    await page.locator("#sliders").evaluate((element) => {
      element.scrollTop = element.scrollHeight;
    });
    const afterPanelScroll = await pageChromeState(page);
    expectFixedPageChrome(afterPanelScroll!, "desktop panel scroll");
    expectSameBackgroundFrame(initial!, afterPanelScroll!, "desktop panel scroll");

    await injectOversizedContent(page);
    const afterContentGrowth = await pageChromeState(page);
    expectFixedPageChrome(afterContentGrowth!, "desktop content growth");
    expectSameBackgroundFrame(initial!, afterContentGrowth!, "desktop content growth");
  });

  test("desktop local scrolling chains to the document at the panel boundary", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "desktop wheel chaining contract");
    await page.goto("/?test=1");
    await expect(page.locator("#sliders .slider-group").last()).toBeVisible({ timeout: 15_000 });

    const body = page.locator(".ingredient-insight-body");
    await body.evaluate((element) => {
      const filler = document.createElement("p");
      filler.dataset.testFiller = "true";
      filler.textContent = "Local scroll chaining content. ".repeat(250);
      element.appendChild(filler);

      // Keep downstream document range available after centering the local owner,
      // otherwise the page can already be at its maximum and chaining is unobservable.
      const documentTail = document.createElement("div");
      documentTail.dataset.testDocumentTail = "true";
      documentTail.style.height = "600px";
      documentTail.style.flex = "0 0 600px";
      document.body.appendChild(documentTail);
    });
    await expect.poll(() => body.evaluate((element) => element.scrollHeight - element.clientHeight)).toBeGreaterThan(100);
    await body.scrollIntoViewIfNeeded();
    await body.evaluate((element) => {
      element.scrollTop = 0;
    });

    const box = await body.boundingBox();
    if (!box) throw new Error("Ingredient body has no bounding box");
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    const initialDocumentY = await page.evaluate(() => window.scrollY);
    await page.mouse.wheel(0, 180);
    await expect.poll(() => body.evaluate((element) => element.scrollTop)).toBeGreaterThan(0);
    expect(await page.evaluate(() => window.scrollY)).toBe(initialDocumentY);

    await body.evaluate((element) => {
      element.scrollTop = element.scrollHeight;
    });
    const boundaryDocumentY = await page.evaluate(() => window.scrollY);
    for (let attempt = 0; attempt < 3; attempt++) {
      await page.mouse.wheel(0, 180);
      await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => resolve())));
      if (await page.evaluate((startY) => window.scrollY > startY, boundaryDocumentY)) break;
    }
    expect(await page.evaluate(() => window.scrollY)).toBeGreaterThan(boundaryDocumentY);
  });

  test("mobile background frame survives document scroll and content growth", async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith("mobile"), "mobile document invariant");
    await page.goto("/");
    await expect(page.locator("#sliders .slider-group").last()).toBeAttached({ timeout: 15_000 });

    const initial = await pageChromeState(page);
    expect(initial, "initial page chrome must exist").not.toBeNull();
    expectFixedPageChrome(initial!, "initial load");

    await injectOversizedContent(page);
    await page.locator(".references-panel").scrollIntoViewIfNeeded();
    const afterDocumentScroll = await pageChromeState(page);
    expect(afterDocumentScroll!.scrollY, "mobile contract must exercise document scrolling").toBeGreaterThan(0);
    expectFixedPageChrome(afterDocumentScroll!, "mobile document scroll");
    expectSameBackgroundFrame(initial!, afterDocumentScroll!, "mobile document scroll");

    const viewport = page.viewportSize()!;
    await page.setViewportSize({ width: viewport.height, height: viewport.width });
    const afterOrientationChange = await pageChromeState(page);
    expectFixedPageChrome(afterOrientationChange!, "mobile orientation change");
    expect(afterOrientationChange!.background.focalPosition).toBe(initial!.background.focalPosition);
    expect(afterOrientationChange!.background.size).toBe(initial!.background.size);
  });
});
