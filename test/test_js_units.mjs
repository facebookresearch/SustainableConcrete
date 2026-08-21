/**
 * Pure-logic tests for the web UI's unit-conversion module
 * (`docs/units.mjs`).
 *
 * Run: node test/test_js_units.mjs
 *
 * These do not exercise the DOM — they verify the math (factors,
 * temperature offset conversion, round-trip identity) that the slider
 * rendering, click-to-edit commit, and unit-toggle logic depend on.
 */

import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, "..");

const {
  UNITS,
  isTempColumn,
  celsiusToDisplay,
  displayToCelsius,
  compToDisplay,
  compFromDisplay,
  sliderUnitLabel,
} = await import(join(REPO_ROOT, "docs", "units.mjs"));

const RTOL = 1e-6;
const ATOL = 1e-9;

let passed = 0;
let failed = 0;

function assertClose(actual, expected, name) {
  // A missing field yields NaN here, and `NaN > RTOL` is false — without this
  // guard an absent UNITS key would be scored as a PASS.
  if (!Number.isFinite(actual)) {
    failed++;
    console.error(`✗ ${name}: expected ${expected}, got non-finite ${actual}`);
    return;
  }
  const absErr = Math.abs(actual - expected);
  const denom = Math.abs(expected) > ATOL ? Math.abs(expected) : 1;
  const relErr = absErr / denom;
  if (relErr > RTOL && absErr > ATOL) {
    failed++;
    console.error(
      `✗ ${name}: expected ${expected}, got ${actual} (relErr=${relErr.toExponential(2)})`,
    );
  } else {
    passed++;
  }
}

function assertEqual(actual, expected, name) {
  if (actual !== expected) {
    failed++;
    console.error(`✗ ${name}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  } else {
    passed++;
  }
}

// --- 1. Constant factor sanity ---
// Strength: 1 MPa ≈ 145.04 psi, so psi → MPa factor ≈ 0.006895.
assertClose(UNITS.metric.strengthFactor, 1 / 145.04, "metric strengthFactor");
assertEqual(UNITS.imperial.strengthFactor, 1, "imperial strengthFactor (no-op)");
// Mass: 1 kg/m³ = 2.2046 lb / 1.30795 yd³ = 1.6856 lb/yd³.
assertEqual(UNITS.metric.massFactor, 1, "metric massFactor (no-op)");
assertClose(UNITS.imperial.massFactor, 1.6856, "imperial massFactor");
// GWP shares the density-style conversion with mass.
assertClose(UNITS.imperial.gwpFactor, UNITS.imperial.massFactor, "gwpFactor === massFactor (imperial)");
// Cost: $/m³ → $/yd³ = × 0.7646 (1 yd³ = 0.7646 m³).
assertClose(UNITS.imperial.costFactor, 1 / 1.30795, "imperial costFactor");

// --- 2. Unit labels ---
assertEqual(UNITS.metric.strength, "MPa", "metric strength label");
assertEqual(UNITS.imperial.strength, "psi", "imperial strength label");
assertEqual(UNITS.metric.mass, "kg/m³", "metric mass label");
assertEqual(UNITS.imperial.mass, "lb/yd³", "imperial mass label");
assertEqual(UNITS.metric.temp, "°C", "metric temp label");
assertEqual(UNITS.imperial.temp, "°F", "imperial temp label");

// --- 3. Temperature conversion (offset, not just factor) ---
assertClose(celsiusToDisplay(0, "imperial"), 32, "0°C → 32°F (water freeze)");
assertClose(celsiusToDisplay(100, "imperial"), 212, "100°C → 212°F (water boil)");
assertClose(celsiusToDisplay(22, "imperial"), 71.6, "22°C → 71.6°F (room temp)");
assertClose(celsiusToDisplay(-40, "imperial"), -40, "-40°C ≡ -40°F (scales cross)");
assertClose(celsiusToDisplay(-20, "imperial"), -4, "-20°C → -4°F (cold curing)");
// Metric is identity
assertClose(celsiusToDisplay(22, "metric"), 22, "metric celsiusToDisplay identity");
assertClose(celsiusToDisplay(-20, "metric"), -20, "metric celsiusToDisplay identity (negative)");

// Inverse
assertClose(displayToCelsius(32, "imperial"), 0, "32°F → 0°C");
assertClose(displayToCelsius(71.6, "imperial"), 22, "71.6°F → 22°C");
assertClose(displayToCelsius(22, "metric"), 22, "metric displayToCelsius identity");

// Round-trip identity for a few canonical values
for (const c of [-40, -20, 0, 10, 22, 100]) {
  assertClose(
    displayToCelsius(celsiusToDisplay(c, "imperial"), "imperial"),
    c,
    `temperature round-trip at ${c}°C`,
  );
}

// --- 4. compToDisplay / compFromDisplay (column-aware) ---
// Mass columns
assertClose(compToDisplay("Cement (kg/m3)", 100, "metric"), 100, "cement metric identity");
assertClose(compToDisplay("Cement (kg/m3)", 100, "imperial"), 168.56, "100 kg/m³ → 168.56 lb/yd³");
assertClose(compToDisplay("Slag (kg/m3)", 267, "imperial"), 267 * 1.6856, "slag 267 kg/m³ → lb/yd³");
// Temperature column
assertClose(compToDisplay("Temp (C)", 22, "imperial"), 71.6, "comp temp 22°C → 71.6°F");
assertClose(compToDisplay("Temp (C)", -20, "imperial"), -4, "comp temp -20°C → -4°F");
assertClose(compToDisplay("Temp (C)", 22, "metric"), 22, "comp temp metric identity");

// Inverse
assertClose(compFromDisplay("Cement (kg/m3)", 168.56, "imperial"), 100, "inverse cement");
assertClose(compFromDisplay("Temp (C)", 71.6, "imperial"), 22, "inverse temp");

// Round-trip for both column kinds and both unit systems
for (const colName of ["Cement (kg/m3)", "Temp (C)", "HRWR (kg/m3)"]) {
  for (const u of ["metric", "imperial"]) {
    for (const v of [0, 1, 22, 250.5]) {
      assertClose(
        compFromDisplay(colName, compToDisplay(colName, v, u), u),
        v,
        `round-trip ${colName} ${v} (${u})`,
      );
    }
  }
}

// --- 5. Column-name dispatch ---
assertEqual(isTempColumn("Temp (C)"), true, "isTempColumn detects Temp (C)");
assertEqual(isTempColumn("Cement (kg/m3)"), false, "isTempColumn rejects Cement");
assertEqual(isTempColumn("HRWR (kg/m3)"), false, "isTempColumn rejects HRWR");

// --- 6. sliderUnitLabel ---
assertEqual(sliderUnitLabel("Cement (kg/m3)", "metric"), "kg/m³", "label cement metric");
assertEqual(sliderUnitLabel("Cement (kg/m3)", "imperial"), "lb/yd³", "label cement imperial");
assertEqual(sliderUnitLabel("Temp (C)", "metric"), "°C", "label temp metric");
assertEqual(sliderUnitLabel("Temp (C)", "imperial"), "°F", "label temp imperial");

// --- Slump units ---
// Slump is stored in INCHES (unlike everything else, which is metric-native),
// so the factors run the other way: metric display multiplies, imperial is 1.
assertEqual(UNITS.metric.slump, "mm", "metric slump label");
assertEqual(UNITS.imperial.slump, "in", "imperial slump label");
assertEqual(UNITS.imperial.slumpFactor, 1, "imperial slumpFactor (no-op)");
assertEqual(UNITS.metric.slumpFactor, 25.4, "metric slumpFactor (in -> mm)");
// 6 * 25.4 === 152.39999999999998, so this one genuinely needs a tolerance.
assertClose(6 * UNITS.metric.slumpFactor, 152.4, "6 in reads as 152.4 mm");

// --- Summary ---
console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) {
  process.exit(1);
}
