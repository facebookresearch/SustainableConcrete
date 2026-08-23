import { expect, type Locator } from "@playwright/test";

export type EffectEnvelope = {
  top: number;
  right: number;
  bottom: number;
  left: number;
};

export async function expectEffectInsideClippingAncestors(
  locator: Locator,
  envelope: EffectEnvelope,
) {
  const result = await locator.evaluate((element, effect) => {
    const rect = element.getBoundingClientRect();
    const painted = {
      top: rect.top - effect.top,
      right: rect.right + effect.right,
      bottom: rect.bottom + effect.bottom,
      left: rect.left - effect.left,
    };
    const failures: Array<{ selector: string; clip: DOMRect; painted: typeof painted }> = [];
    let ancestor = element.parentElement;
    while (ancestor && ancestor !== document.body && ancestor !== document.documentElement) {
      const style = getComputedStyle(ancestor);
      const clipsX = /(hidden|clip|auto|scroll)/.test(style.overflowX);
      const clipsY = /(hidden|clip|auto|scroll)/.test(style.overflowY);
      if (clipsX || clipsY) {
        const clip = ancestor.getBoundingClientRect();
        const outsideX = clipsX && (painted.left < clip.left - 0.5 || painted.right > clip.right + 0.5);
        const outsideY = clipsY && (painted.top < clip.top - 0.5 || painted.bottom > clip.bottom + 0.5);
        if (outsideX || outsideY) {
          failures.push({
            selector: ancestor.id ? `#${ancestor.id}` : `.${ancestor.className}`,
            clip,
            painted,
          });
        }
      }
      ancestor = ancestor.parentElement;
    }
    return failures.map(({ selector, clip, painted: union }) => ({
      selector,
      clip: { top: clip.top, right: clip.right, bottom: clip.bottom, left: clip.left },
      painted: union,
    }));
  }, envelope);
  expect(result, `effect envelope for ${locator}`).toEqual([]);
}
