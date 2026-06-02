// Pure unit-system definitions and converters for the web UI.
//
// All native (stored) values are in metric — see `boxcrete/units.py` for the
// canonical authority over data units in this repo:
//   compositions: kg/m³,  strength: psi,  temperature: °C,
//   GWP: kg CO₂e/m³,  cost: $/m³,  slump: inches.
//
// Imperial display layers in the UI apply the conversions below at render
// time. Composition/GWP/cost are simple linear factors; temperature has an
// offset (F = C × 9/5 + 32) and needs explicit converter functions.
//
// This module is pure (no DOM, no global state) so it can be imported by
// `ui.mjs` *and* by Node-only tests (`test/test_js_units.mjs`).

export const UNITS = {
  metric: {
    strength: "MPa",
    strengthFactor: 1 / 145.04, // psi → MPa
    mass: "kg/m³",
    massFactor: 1,
    gwp: "kg CO₂e/m³",
    gwpFactor: 1,
    cost: "$/m³",
    costFactor: 1,
    temp: "°C",
  },
  imperial: {
    strength: "psi",
    strengthFactor: 1, // already psi
    mass: "lb/yd³",
    massFactor: 1.6856, // kg/m³ → lb/yd³
    gwp: "lb CO₂/yd³",
    gwpFactor: 1.6856,
    cost: "$/yd³",
    costFactor: 1 / 1.30795, // $/m³ → $/yd³  (1 yd³ = 0.7646 m³)
    temp: "°F",
  },
};

// Whether a column name represents a temperature column (matches the
// dataset's "Temp (C)" naming convention).
export function isTempColumn(colName) {
  return colName.includes("Temp");
}

// Convert a celsius value to the active unit system's display value.
// `F = C × 9/5 + 32` for imperial; identity for metric.
// NOTE: temperature conversion is *not* a single multiplicative factor
// because of the `+32` offset — animation/interpolation code that mixes
// two unit systems linearly cannot use this in the same way as mass or
// strength. Sliders snap on unit toggle, so this is fine for them.
export function celsiusToDisplay(celsius, unitSystem) {
  return unitSystem === "imperial" ? celsius * 9 / 5 + 32 : celsius;
}

// Inverse of `celsiusToDisplay`: parse a displayed value back to celsius
// for storage in the model-native composition vector.
export function displayToCelsius(display, unitSystem) {
  return unitSystem === "imperial" ? (display - 32) * 5 / 9 : display;
}

// Convert a stored composition value (the model-native units: kg/m³ for
// masses, °C for temperature) to the value that should be shown to the
// user given the active unit system.
//
// Caller is responsible for excluding non-numeric columns like
// "Material Source" before invoking this.
export function compToDisplay(colName, internal, unitSystem) {
  if (isTempColumn(colName)) return celsiusToDisplay(internal, unitSystem);
  return internal * UNITS[unitSystem].massFactor;
}

// Inverse of `compToDisplay`. Round-trip identity:
//   compFromDisplay(col, compToDisplay(col, x, u), u) === x  (mod fp).
export function compFromDisplay(colName, display, unitSystem) {
  if (isTempColumn(colName)) return displayToCelsius(display, unitSystem);
  return display / UNITS[unitSystem].massFactor;
}

// Per-row unit suffix label: "kg/m³"/"lb/yd³" for mass, "°C"/"°F" for
// temperature. Used by the composition setter panel and updated whenever
// the unit system toggles.
export function sliderUnitLabel(colName, unitSystem) {
  if (isTempColumn(colName)) return UNITS[unitSystem].temp;
  return UNITS[unitSystem].mass;
}
