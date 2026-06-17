# Mobile performance investigation — V2 / v5 strength GP

This is investigation-only — no code changes. Captures ideas for a
follow-up PR after the v5 + `joint_hamming_matern` merge.

## Current cost profile

| dimension | value |
|---|---|
| Training rows `n` | 647 |
| Augmented feature dim `d_aug` | 17 (raw 10 + F5_alllog 7) |
| Kernel matrix `K` (one-time init) | 647 × 647 × 8 B = **3.3 MB float64** |
| Cholesky `L` (one-time init) | same 3.3 MB |
| Per single-point prediction | O(n · d_aug) kernel evals + O(n²) triangular solve = ~430 K ops |
| Per 64-point curve | ~27 M ops (~30-40 MFLOPS incl. Matern transcendentals) |
| `initStrengthModel` Cholesky | O(n³) = 270 M FLOPs |

On desktop Chrome with WASM dtrsm: curve eval ≈ **10 ms**, init ≈
**70 ms**. On mobile (Apple A14-class CPU, single-threaded JS): curve
eval **30-50 ms** per slider movement at 60Hz drag = ~80 % CPU time on
GP compute alone. Init: **150-300 ms**. The lag is primarily the
30 M-op kernel build that doesn't use WASM (only the dtrsm does).

## Optimization opportunities (ranked by ROI for mobile)

### Tier 0 — biggest UX win, trivial effort

#### 0. Coarse-grained time grid during transition animations  (effort: trivial, speedup: ~4× during animations)

Currently each animation frame (60 Hz) recomputes a 64-point strength
curve. A 350-ms Material Source transition (or scatter-snap animation)
fires ~21 frames × 64 = **1344 GP evaluations**, swamping mobile.

The strength curve is smooth and log-time-monotonic with ~3-5
effective DOFs from the Matern fit — a 16-point log-spaced grid
captures the shape visually. Chart.js interpolates between them.

**Plan**: drop animation-time-grid resolution to 16 points; settle
at 64 points only at animation END.

| state | times/curve | curves/animation | GP evals |
|---|---|---|---|
| current | 64 | ~21 | 1344 |
| 16-pt during anim + settle at 64 | 16 → 64 | 21 + 1 | **400** (3.4× fewer) |

Implementation (sketch):

```js
// docs/ui.mjs::triggerMaterialSourceTransition
const ANIM_TIMES = logSpacedTimes(16);    // coarse during anim
// ...existing _msCurveTransition setup, using ANIM_TIMES instead of 64
// At anim end:
const FULL_TIMES = logSpacedTimes(64);
finalCurve = predictStrengthCurve(currentComposition, FULL_TIMES, params);
```

Same pattern applies to `animateToComposition` (slider drag, scatter
click-to-snap) — the most frequent user interaction.

Expected: animation frame budget drops from ~50 ms to ~12 ms per
frame on mobile → 60 Hz holds instead of dropping to 20-30 Hz. Final
settled curve unchanged.

### Tier 1 — high impact, low/medium effort

#### 1. Per-curve composition-distance reuse  (effort: low, speedup: ~1.5× wall-time)

For a 64-point curve at a fixed composition, the kernel matrix builder
recomputes the feature-distance Σ_f Δx_f²/ℓ_f² **64 times** even though
only the time dim changes. Cache the (n × 1) composition-distance
vector once, then add the per-time Time-dim contribution in the inner
loop.

```js
// Before (current):
for j in times: for i in n: compute Σ_f Δx_f²/ℓ_f² over ALL dims  // 64 × n × d_aug
// After:
for i in n: compute d2_comp[i] = Σ_{f≠t} Δx_f²/ℓ_f²              // n × (d_aug-1)
for j in times: for i in n: d2_total[i,j] = d2_comp[i] + dt²/ℓ_t² // 64 × n × 1
```

Per kernel evaluation: with the cache, the inner-loop scalar-op
count drops ~6× (75 → 13 ops) but the per-eval `sqrt` + `exp`
transcendentals (~30 cycles) are unchanged and dominate. Net:
**kernel-build portion ~1.7-2× faster, curve-prediction wall time
~1.4-1.5× faster** (kernel build is ~50% of curve cost).

#### 2. WebWorker offload  (effort: low-medium, speedup: 0× but UX wins)

GP eval blocks the main thread → input lag, dropped frames, jank.
Moving `predictStrengthCurve` to a dedicated worker keeps the UI
responsive (slider scrolling, scatter pan/zoom, ChartJS redraw) while
GP compute runs in parallel. Doesn't reduce wall time but eliminates
visible UI freeze.

Pattern: `postMessage(composition, transferList)` → worker computes
→ `onmessage` updates chart. Wire the existing in-flight-cancel logic
(`triggerMaterialSourceTransition`) to use worker job IDs.

Expected: drag stays at 60Hz (slider tracks the finger smoothly)
even when GP compute lags by 30-50 ms.

#### 3. Inducing-point sparse GP (FITC / Titsias VFE) (effort: medium, speedup: 5-10×)

The full 647-point exact GP has redundant rows (compositional
fingerprint near-duplicates). An M-inducing-point sparse approximation
(M = 64-128) reduces cost dramatically:

| metric | full (n=647) | sparse (M=128) | speedup |
|---|---|---|---|
| Cholesky init | 270 M FLOPs | 2 M FLOPs | **135×** |
| Per-prediction kernel | 647 evals | 128 evals | **5×** |
| Memory (K + L) | 6.6 MB | 260 KB | **25×** |

Implementation: precompute inducing points + inducing-set kernel
parameters offline (BoTorch's `SingleTaskVariationalGP` or
`InducingPointKernel`). Serialise into a "compressed" `strength.json`
variant; load that on mobile (UA-sniffed) or as the default once
quality is verified within ~10 psi of the full model.

Expected mobile curve from 30-50 ms → 5-10 ms; init from 200 ms → 30 ms.

### Tier 2 — meaningful, medium effort

#### 4. Memo'd kernel-vector for slider deltas (effort: medium, speedup: 5-10× during drag)

When the user drags a single slider, only one feature dim changes.
The kernel vector `k(x*, X_train)` mostly stays the same; the only
dim contributing to `Δx_f²/ℓ_f²` changes is the dragged one.

Cache the per-(test-point, train-row) pair's d²_feat from the previous
prediction. On the next prediction:
- For each train row, subtract the old contribution of the dragged
  dim and add the new contribution. O(n) per dim, vs O(n · d_aug)
  for a fresh build.

Trade-off: cache state must be invalidated on Material Source toggle
(class change → discrete) and on the "scatter snap" animation
target jumps. The current `_msCurveTransition` and
`animateToComposition` paths are the integration points.

#### 5. Cataloged-prediction interpolation for small drag deltas (effort: medium)

The explorer ships with 144 precomputed compositions in
`compositions.json::strength_predictions`. For slider movements that
land near a cataloged composition, interpolate between the cataloged
prediction and the current live prediction with weight by composition
distance. Refresh the live prediction asynchronously.

Net: drag feels instant; the curve "settles" to the exact GP value
~50 ms after the user releases the slider.

### Tier 3 — incremental, lower priority

#### 6. SIMD-friendly inner loops (effort: medium, speedup: 1.5-3×)

The kernel-build hot loop is JIT-friendly but not WASM-SIMD. Move the
inner radial-basis loop to a WASM function with explicit `v128` SIMD
(4-wide fp32 or 2-wide fp64). Already have a WASM dtrsm pipeline for
the variance solve — extending to kernel build is incremental.

#### 7. fp32 throughout (effort: low, speedup: 1.5-2×)

Replace `Float64Array` with `Float32Array` for kernel evaluation
(keep fp64 for the Cholesky solve, where ill-conditioned cases
benefit from extra precision). Memory bandwidth halves; mobile JS
engines optimize fp32 better in some cases.

Risk: kernel evaluation precision loss on the order of 1e-6 — within
the tolerance of strength predictions (psi-level).

#### 8. Training-set pruning (effort: low, speedup: 2-4×)

Many training points contribute < 1% of the posterior mass at typical
test points. Greedy pruning to keep the M=200 most-informative rows
(by max kernel weight across the data distribution) reduces n by 3×
with minimal accuracy loss.

This is a special case of the inducing-point approximation but
simpler — keeps the exact GP topology, just fewer points.

#### 9. Slider drag throttling (effort: trivial)

Throttle update at 30Hz instead of 60Hz during active drag. Doesn't
help compute but halves the GP-eval rate.

```js
let lastUpdate = 0;
slider.addEventListener("input", () => {
  if (performance.now() - lastUpdate < 33) return;
  lastUpdate = performance.now();
  update();
});
```

### Tier 4 — high effort, situational

#### 10. WebGPU compute shaders (effort: very high, speedup: 5-20×)

Move kernel-matrix build + Cholesky to WebGPU. Massive parallelism
on mobile GPUs. Browser support: iOS 18+ Safari, modern Chrome/Edge.
Requires WGSL shader code + JS marshalling. Worth it only if Tiers 1-3
hit a wall.

#### 11. Quantized inference (int8/fp16) (effort: high, speedup: 2-4×)

Cast kernel intermediates to fp16. ARM CPUs have dedicated fp16 ops.
Numerical-precision audit needed; risk of degraded predictive intervals.

## Recommended sequence for the follow-up PR

1. **Land Tier 1 #2 (WebWorker)** first — biggest UX win, low risk, no
   numerical changes. Eliminates visible lag even before any speedup.
2. **Land Tier 1 #1 (per-curve reuse)** — ~2× speedup, low risk.
3. **Profile real mobile devices** after #1+#2; identify whether
   remaining lag is compute-bound or other (chart redraw, layout
   thrash).
4. **If still compute-bound, land Tier 1 #3 (sparse GP)** — biggest
   compute speedup, medium risk (requires accuracy verification vs
   the full GP).
5. **Tier 2 (memoization, catalog interpolation) and Tier 3** as
   incremental wins as needed.

## What NOT to do

* Avoid Tier 4 unless Tiers 1-3 fall short. The complexity-to-benefit
  ratio is poor for our problem size (n=647 is small enough that
  CPU-side optimizations can succeed).
* Avoid removing engineered features (F5_alllog) for speedup — they
  encode domain knowledge that materially helps prediction quality.
* Avoid switching the source kernel to a simpler one (e.g. back to
  `hamming`) for speedup — the joint_hamming_matern's algebra is
  cheap; the bottleneck is the n=647 row count, not per-kernel-eval
  cost.

## Profiling recommendations before optimizing

* Add `performance.mark()` / `performance.measure()` around:
  - `initStrengthModel` (one-time)
  - `kernel matrix build` inside `predictStrengthCurveV2`
  - WASM `_dtrsm_lower` solve
  - ChartJS draw
* Use Chrome DevTools mobile-emulation Performance tab (CPU 4×
  throttling matches an iPhone 12-class device).
* Test on a real low-end Android (e.g., Moto G or Samsung A-series)
  before committing to any approach — JS engine quirks vary.
