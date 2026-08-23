export const SELECTOR_TICK_GAP = 8;

export function formatYAxisTick(value) {
  if (Math.abs(value) < 1_000) return String(Math.round(value));

  const thousands = value / 1_000;
  if (Number.isInteger(thousands)) return `${thousands}k`;
  if (Math.abs(thousands) < 10) return `${Number(thousands.toFixed(1))}k`;
  return `${Math.round(thousands)}k`;
}

// Fixed lanes

// Fixed lanes are sized from the widest supported painted tick text and the
// controls' real border boxes. Fractions preserve the measured browser paint;
// derivePlotInsets() rounds only the final lane so CSS can use whole pixels.
export const PLOT_LANE_COMPONENTS = Object.freeze({
  mobile: Object.freeze({
    left: Object.freeze({
      selectorSize: 40,
      selectorEffectReserve: 0.5,
      tickOffset: 8,
      worstTickPaint: 27,
    }),
    rightEffectReserve: 20,
    topEffectReserve: 15,
    bottom: Object.freeze({
      selectorSize: 46,
      selectorEffectReserve: 1.75,
      tickOffset: 14,
      worstTickPaint: 0.25,
    }),
  }),
  desktop: Object.freeze({
    left: Object.freeze({
      selectorSize: 36,
      selectorEffectReserve: 0.5,
      tickOffset: 8,
      worstTickPaint: 27,
    }),
    rightEffectReserve: 20,
    topEffectReserve: 15,
    bottom: Object.freeze({
      selectorSize: 28,
      selectorEffectReserve: 3.75,
      tickOffset: 14,
      worstTickPaint: 0.25,
    }),
  }),
});

export function derivePlotInsets(components) {
  return {
    left: Math.ceil(
      components.left.selectorSize +
      components.left.selectorEffectReserve +
      components.left.tickOffset +
      components.left.worstTickPaint +
      SELECTOR_TICK_GAP,
    ),
    right: Math.ceil(components.rightEffectReserve),
    top: Math.ceil(components.topEffectReserve),
    bottom: Math.ceil(
      components.bottom.tickOffset +
      components.bottom.worstTickPaint +
      SELECTOR_TICK_GAP +
      components.bottom.selectorSize +
      components.bottom.selectorEffectReserve,
    ),
  };
}

// Mobile keeps 44px touch targets. Desktop uses slimmer controls while both
// modes preserve a 15px top and 20px right visual-effect reserve.
export const MOBILE_PLOT_INSETS = Object.freeze(
  derivePlotInsets(PLOT_LANE_COMPONENTS.mobile),
);
export const DESKTOP_PLOT_INSETS = Object.freeze(
  derivePlotInsets(PLOT_LANE_COMPONENTS.desktop),
);
export const PLOT_INSETS = MOBILE_PLOT_INSETS;

export function resolvePlotLaneComponents(viewportWidth) {
  return Number.isFinite(viewportWidth) && viewportWidth >= 1051
    ? PLOT_LANE_COMPONENTS.desktop
    : PLOT_LANE_COMPONENTS.mobile;
}

export function resolvePlotInsets(viewportWidth) {
  return Number.isFinite(viewportWidth) && viewportWidth >= 1051
    ? DESKTOP_PLOT_INSETS
    : MOBILE_PLOT_INSETS;
}

function finiteNonnegative(value) {
  return Number.isFinite(value) ? Math.max(0, value) : 0;
}

function validPositive(value) {
  return Number.isFinite(value) && value > 0;
}

function validInsets(insets) {
  return insets && [insets.left, insets.right, insets.top, insets.bottom]
    .every((value) => Number.isFinite(value) && value >= 0);
}

function insetTotals(insets) {
  return {
    horizontal: insets.left + insets.right,
    vertical: insets.top + insets.bottom,
  };
}

export function computeSelectorPlacement({
  axis,
  plotStart,
  plotEnd,
  paintedTickEdge,
  selectorSize,
  effectReserve = 0,
}) {
  const start = finiteNonnegative(plotStart);
  const end = finiteNonnegative(plotEnd);
  const paintEdge = finiteNonnegative(paintedTickEdge);
  const width = finiteNonnegative(selectorSize?.width);
  const height = finiteNonnegative(selectorSize?.height);
  const reserve = finiteNonnegative(effectReserve);
  const center = start + Math.max(0, end - start) / 2;
  const left = axis === "y" ? paintEdge - SELECTOR_TICK_GAP - width : center - width / 2;
  const top = axis === "y" ? center - height / 2 : paintEdge + SELECTOR_TICK_GAP;
  const right = left + width;
  const bottom = top + height;
  return {
    left,
    top,
    right,
    bottom,
    effectBounds: {
      left: left - reserve,
      top: top - reserve,
      right: right + reserve,
      bottom: bottom + reserve,
    },
  };
}

export function computeFixedYSelectorPlacement({
  plotLeft,
  plotTop,
  plotBottom,
  selectorSize,
  lane,
}) {
  const leftLane = lane ?? PLOT_LANE_COMPONENTS.mobile.left;
  const reservedTickEdge = finiteNonnegative(plotLeft)
    - finiteNonnegative(leftLane.tickOffset)
    - finiteNonnegative(leftLane.worstTickPaint);
  return computeSelectorPlacement({
    axis: "y",
    plotStart: plotTop,
    plotEnd: plotBottom,
    paintedTickEdge: reservedTickEdge,
    selectorSize,
    effectReserve: leftLane.selectorEffectReserve,
  });
}

export function computePlotRect(canvasWidth, canvasHeight, insets = PLOT_INSETS) {
  const width = finiteNonnegative(canvasWidth);
  const height = finiteNonnegative(canvasHeight);
  const contract = validInsets(insets) ? insets : PLOT_INSETS;
  const totals = insetTotals(contract);
  const left = Math.min(contract.left, width);
  const top = Math.min(contract.top, height);
  const plotWidth = Math.max(0, width - totals.horizontal);
  const plotHeight = Math.max(0, height - totals.vertical);
  return {
    left,
    top,
    right: left + plotWidth,
    bottom: top + plotHeight,
    width: plotWidth,
    height: plotHeight,
  };
}

export function computeCanvasSizeForPlotRect(plotWidth, plotHeight, insets = PLOT_INSETS) {
  const contract = validInsets(insets) ? insets : PLOT_INSETS;
  const totals = insetTotals(contract);
  return {
    width: finiteNonnegative(plotWidth) + totals.horizontal,
    height: finiteNonnegative(plotHeight) + totals.vertical,
  };
}

export function computeAvailablePlotSize(
  containerWidths,
  availableCanvasHeight,
  preferredMaximum,
  insets = PLOT_INSETS,
) {
  const contract = validInsets(insets) ? insets : PLOT_INSETS;
  const totals = insetTotals(contract);
  if (
    containerWidths.length === 0 ||
    containerWidths.some((width) => !validPositive(width)) ||
    !validPositive(availableCanvasHeight) ||
    !validPositive(preferredMaximum?.width) ||
    !validPositive(preferredMaximum?.height)
  ) {
    return { width: 0, height: 0 };
  }

  const width = Math.min(
    Math.max(0, Math.min(...containerWidths) - totals.horizontal),
    preferredMaximum.width,
  );
  const height = Math.min(
    Math.max(0, availableCanvasHeight - totals.vertical),
    preferredMaximum.height,
    width,
  );
  return { width, height };
}
