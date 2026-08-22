import assert from "node:assert/strict";

import {
  DESKTOP_PLOT_INSETS,
  MOBILE_PLOT_INSETS,
  PLOT_LANE_COMPONENTS,
  SELECTOR_TICK_GAP,
  computeAvailablePlotSize,
  computeCanvasSizeForPlotRect,
  computeFixedYSelectorPlacement,
  computePlotRect,
  computeSelectorPlacement,
  derivePlotInsets,
  formatYAxisTick,
  resolvePlotInsets,
} from "../docs/plot-geometry.mjs";

function insetTotals(insets) {
  return {
    horizontal: insets.left + insets.right,
    vertical: insets.top + insets.bottom,
  };
}

assert.equal(SELECTOR_TICK_GAP, 8, "painted ticks and selectors share one canonical gap");
for (const [value, expected] of [
  [0, "0"],
  [12.4, "12"],
  [-12.6, "-13"],
  [999.4, "999"],
  [999.5, "1000"],
  [-999.6, "-1000"],
  [1_000, "1k"],
  [-1_000, "-1k"],
  [1_500, "1.5k"],
  [5_000, "5k"],
  [20_000, "20k"],
  [-20_000, "-20k"],
]) {
  assert.equal(formatYAxisTick(value), expected, `${value} has deterministic compact Y tick text`);
}
assert.equal(PLOT_LANE_COMPONENTS.mobile.left.worstTickPaint, 27);
assert.equal(PLOT_LANE_COMPONENTS.desktop.left.worstTickPaint, 27);
assert.deepEqual(derivePlotInsets(PLOT_LANE_COMPONENTS.mobile), {
  left: 84,
  right: 20,
  top: 15,
  bottom: 70,
});
assert.deepEqual(derivePlotInsets(PLOT_LANE_COMPONENTS.desktop), {
  left: 80,
  right: 20,
  top: 15,
  bottom: 54,
});
assert.deepEqual(MOBILE_PLOT_INSETS, derivePlotInsets(PLOT_LANE_COMPONENTS.mobile));
assert.deepEqual(DESKTOP_PLOT_INSETS, derivePlotInsets(PLOT_LANE_COMPONENTS.desktop));
assert.equal(resolvePlotInsets(1050), MOBILE_PLOT_INSETS);
assert.equal(resolvePlotInsets(1051), DESKTOP_PLOT_INSETS);
assert.equal(resolvePlotInsets(Number.NaN), MOBILE_PLOT_INSETS);

assert.deepEqual(
  computeSelectorPlacement({
    axis: "x",
    plotStart: 94.25,
    plotEnd: 412.75,
    paintedTickEdge: 352.375,
    selectorSize: { width: 157.5, height: 27.75 },
    effectReserve: 1.25,
  }),
  {
    left: 174.75,
    top: 360.375,
    right: 332.25,
    bottom: 388.125,
    effectBounds: {
      left: 173.5,
      top: 359.125,
      right: 333.5,
      bottom: 389.375,
    },
  },
  "X placement stays centered on the fractional drawable and clears painted ticks by 8px",
);
assert.deepEqual(
  computeSelectorPlacement({
    axis: "y",
    plotStart: 20.5,
    plotEnd: 338.25,
    paintedTickEdge: 43.5625,
    selectorSize: { width: 36, height: 221.5 },
    effectReserve: 0.5,
  }),
  {
    left: -0.4375,
    top: 68.625,
    right: 35.5625,
    bottom: 290.125,
    effectBounds: {
      left: -0.9375,
      top: 68.125,
      right: 36.0625,
      bottom: 290.625,
    },
  },
  "the generic Y primitive follows a fractional painted-label edge without changing the plot",
);

for (const [mode, components, insets] of [
  ["mobile", PLOT_LANE_COMPONENTS.mobile.left, MOBILE_PLOT_INSETS],
  ["desktop", PLOT_LANE_COMPONENTS.desktop.left, DESKTOP_PLOT_INSETS],
]) {
  const selectorSize = {
    width: components.selectorSize,
    height: mode === "mobile" ? 236.5 : 221.5,
  };
  const plotTop = 15;
  const plotBottom = 338.25;
  const expectedLeft = insets.left
    - components.tickOffset
    - components.worstTickPaint
    - SELECTOR_TICK_GAP
    - selectorSize.width;
  const placement = computeFixedYSelectorPlacement({
    plotLeft: insets.left,
    plotTop,
    plotBottom,
    selectorSize,
    lane: components,
  });

  assert.equal(placement.left, expectedLeft, `${mode} Y selector uses the reserved left lane`);
  assert.equal(placement.right, expectedLeft + selectorSize.width);
  assert.equal(placement.top, plotTop + (plotBottom - plotTop - selectorSize.height) / 2);
  assert.equal(placement.effectBounds.left, expectedLeft - components.selectorEffectReserve);
  assert.ok(placement.effectBounds.left >= 0, `${mode} Y effect stays inside the left inset`);
  assert.ok(placement.effectBounds.right <= insets.left, `${mode} Y effect stays left of the plot`);

  for (const liveTickPaint of [8, 19.25, components.worstTickPaint]) {
    const liveTickLeft = insets.left - components.tickOffset - liveTickPaint;
    assert.equal(
      computeFixedYSelectorPlacement({
        plotLeft: insets.left,
        plotTop,
        plotBottom,
        selectorSize,
        lane: components,
      }).left,
      placement.left,
      `${mode} Y selector X is invariant for a ${liveTickPaint}px live tick (edge ${liveTickLeft}px)`,
    );
  }
}

assert.deepEqual(MOBILE_PLOT_INSETS, {
  left: 84,
  right: 20,
  top: 15,
  bottom: 70,
}, "fixed Y placement does not change the mobile drawable contract");
assert.deepEqual(DESKTOP_PLOT_INSETS, {
  left: 80,
  right: 20,
  top: 15,
  bottom: 54,
}, "fixed Y placement does not change the desktop drawable contract");

assert.deepEqual(
  computeSelectorPlacement({
    axis: "x",
    plotStart: Number.NaN,
    plotEnd: Infinity,
    paintedTickEdge: -4,
    selectorSize: { width: -2, height: Number.NaN },
    effectReserve: -1,
  }),
  {
    left: 0,
    top: SELECTOR_TICK_GAP,
    right: 0,
    bottom: SELECTOR_TICK_GAP,
    effectBounds: {
      left: 0,
      top: SELECTOR_TICK_GAP,
      right: 0,
      bottom: SELECTOR_TICK_GAP,
    },
  },
  "invalid placement metrics collapse to a deterministic zero-size fallback",
);

for (const contract of [MOBILE_PLOT_INSETS, DESKTOP_PLOT_INSETS]) {
  const { horizontal, vertical } = insetTotals(contract);
  for (const dimensions of [
    { width: 0, height: 0 },
    { width: 1, height: 1 },
    { width: 188, height: 188 },
    { width: 480, height: 320 },
  ]) {
    const canvas = computeCanvasSizeForPlotRect(dimensions.width, dimensions.height, contract);
    assert.deepEqual(canvas, {
      width: dimensions.width + horizontal,
      height: dimensions.height + vertical,
    });
    const plot = computePlotRect(canvas.width, canvas.height, contract);
    assert.deepEqual(plot, {
      left: contract.left,
      top: contract.top,
      right: contract.left + dimensions.width,
      bottom: contract.top + dimensions.height,
      width: dimensions.width,
      height: dimensions.height,
    });
  }
}

assert.deepEqual(
  computeCanvasSizeForPlotRect(480, 320, DESKTOP_PLOT_INSETS),
  { width: 580, height: 389 },
  "the desktop maximum must reclaim selector-lane space without increasing canvas height",
);
assert.deepEqual(
  computeCanvasSizeForPlotRect(448, 304, MOBILE_PLOT_INSETS),
  { width: 552, height: 389 },
  "mobile keeps the established touch-safe geometry",
);

assert.deepEqual(computePlotRect(40, 20, DESKTOP_PLOT_INSETS), {
  left: 40,
  top: 15,
  right: 40,
  bottom: 15,
  width: 0,
  height: 0,
});
assert.deepEqual(computePlotRect(0, 0, MOBILE_PLOT_INSETS), {
  left: 0,
  top: 0,
  right: 0,
  bottom: 0,
  width: 0,
  height: 0,
});

const desktopTotals = insetTotals(DESKTOP_PLOT_INSETS);
assert.deepEqual(
  computeAvailablePlotSize([620, 660], 394, { width: 480, height: 320 }, DESKTOP_PLOT_INSETS),
  { width: 480, height: 320 },
);
assert.deepEqual(
  computeAvailablePlotSize([400, 620], 310, { width: 480, height: 320 }, DESKTOP_PLOT_INSETS),
  {
    width: Math.max(0, 400 - desktopTotals.horizontal),
    height: Math.max(0, 310 - desktopTotals.vertical),
  },
);
const oldDesktopInsets = derivePlotInsets({
  ...PLOT_LANE_COMPONENTS.desktop,
  left: {
    ...PLOT_LANE_COMPONENTS.desktop.left,
    worstTickPaint: 42.5,
  },
});
assert.deepEqual(
  computeAvailablePlotSize([580], 389, { width: 480, height: 320 }, DESKTOP_PLOT_INSETS),
  { width: 480, height: 320 },
  "the reduced left lane gives constrained desktop width back to the drawable",
);
assert.deepEqual(
  computeAvailablePlotSize([580], 389, { width: 480, height: 320 }, oldDesktopInsets),
  { width: 465, height: 320 },
  "only the old wider left lane constrains drawable width; height and other reserves are unchanged",
);
assert.deepEqual(
  computeAvailablePlotSize([260], 600, { width: 480, height: 320 }, DESKTOP_PLOT_INSETS),
  {
    width: Math.max(0, 260 - desktopTotals.horizontal),
    height: Math.max(0, 260 - desktopTotals.horizontal),
  },
  "a narrow container clamps height to width instead of producing a portrait plot",
);
assert.deepEqual(
  computeAvailablePlotSize([0, 500], 394, { width: 480, height: 320 }, DESKTOP_PLOT_INSETS),
  { width: 0, height: 0 },
);
assert.deepEqual(
  computeAvailablePlotSize([], 394, { width: 480, height: 320 }, DESKTOP_PLOT_INSETS),
  { width: 0, height: 0 },
);
assert.deepEqual(
  computeAvailablePlotSize([Infinity, 620], 394, { width: 480, height: 320 }, DESKTOP_PLOT_INSETS),
  { width: 0, height: 0 },
);
assert.deepEqual(
  computeAvailablePlotSize([620], Number.NaN, { width: 480, height: 320 }, DESKTOP_PLOT_INSETS),
  { width: 0, height: 0 },
);
assert.deepEqual(
  computeAvailablePlotSize([620], 394, { width: -1, height: 320 }, DESKTOP_PLOT_INSETS),
  { width: 0, height: 0 },
);

console.log("plot geometry contracts passed");
