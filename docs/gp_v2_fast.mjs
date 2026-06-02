/**
 * Schema-v2 batched curve prediction with WASM-accelerated variance solve.
 *
 * Strategy:
 *   1. For each test time, build the post-transform 17-dim test point
 *      (engineered features via feature_registry → log10 on time → normalize).
 *   2. Build the kernel matrix K_test_train of shape [n, nTimes] in JS
 *      with the multi-Matern + RBF + gating math inlined for the JIT.
 *   3. Mean: K_test_train.T @ alpha (one nested loop, fused with kernel build).
 *   4. Variance: WASM dtrsm L V = K_test_train (one BLAS call), then
 *      variance[i] = k(x*, x*) - ||V[:, i]||² via WASM col_norms_sq.
 *
 * Cost (n=647, nTimes=64, d_aug=17):
 *   - Kernel matrix build: ~4 M FLOPS, ~5 ms in JS.
 *   - WASM dtrsm + col_norms_sq: ~30 M FLOPS, ~5 ms with SIMD.
 *   - Total: ~10 ms for the full curve.
 */

import { appendFeatures } from "./feature_registry.mjs";

const SQRT5 = 2.23606797749979;

export function predictStrengthCurveV2(
  composition,
  times,
  params,
  // Optional WASM handle from gp.mjs's initWASM. Pass nulls to use JS fallback.
  wasmCtx,
) {
  const n = params.n_train;
  const dAug = params.d_aug;
  const nTimes = times.length;

  const blindActiveDims = params.matern_blind.active_dims;
  const blindLS = params.matern_blind.lengthscales;
  const blindOS = params.matern_blind.outputscale;
  const specificActiveDims = params.matern_specific.active_dims;
  const specificLS = params.matern_specific.lengthscales;
  const specificOS = params.matern_specific.outputscale;
  const rbfTimeIdx = params.rbf_time.active_dims[0];
  const rbfLS = params.rbf_time.lengthscale;
  const rbfOS = params.rbf_time.outputscale;
  const gateTau = params.gate_tau;
  const timeDimAug = params.time_dim_aug;
  const yMax = params.y_max;
  const normLo = params.normalize_lower;
  const normHi = params.normalize_upper;
  const logTimeOffset = params.log_time_offset || 1.0;
  const X_flat = params.X_train_flat;
  const alpha = params.alpha_f64;
  const L_flat = params.L_flat;

  const nBlindDims = blindActiveDims.length;
  const nSpecificDims = specificActiveDims.length;

  // Precompute h(t_train) for all training rows once, cached on params.
  if (!params._hTrain) {
    const hTrain = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const t = X_flat[i * dAug + timeDimAug];
      hTrain[i] = 1.0 - Math.exp(-Math.max(0.0, t) / gateTau);
    }
    params._hTrain = hTrain;
  }
  const hTrain = params._hTrain;

  // Build post-transform test inputs: [nTimes × dAug] row-major.
  const testX = new Float64Array(nTimes * dAug);
  const hTest = new Float64Array(nTimes);
  // Separate gate evaluation for the *noise* term, in case a future
  // variant detunes ``noise_gate_tau`` from the kernel ``gate_tau``.
  // Today both are 0.05 so ``hTestNoise`` is numerically identical to
  // ``hTest`` — keeping them separate avoids a silent disagreement with
  // ``gp.mjs::predictStrength`` (which already routes the noise term
  // through ``noise_gate_tau`` independently).
  const noiseGateTau = params.noise_gate_tau ?? gateTau;
  const hTestNoise = noiseGateTau === gateTau
    ? hTest // alias when the two gates agree (today's case)
    : new Float64Array(nTimes);
  // Build testX rows (each is the 17-dim post-transform vector for one timestep).
  const featNames = params.engineered_feature_names;
  const dRaw = params.d_in || 10;
  const timeDimRaw = params.time_dim_raw ?? 9;
  const xRawScratch = new Float64Array(dRaw);
  // Pre-fill the constant raw dims (composition values don't change across times).
  for (let k = 0; k < dRaw; k++) xRawScratch[k] = composition[k] ?? 0;
  for (let ti = 0; ti < nTimes; ti++) {
    // Step 1: append engineered features using RAW input (before log on time).
    const tRaw = times[ti];
    xRawScratch[timeDimRaw] = tRaw;
    const augPostFeatures = appendFeatures(xRawScratch, featNames);

    // Step 2+3: log_offset + log10 on time (in the augmented input).
    const tLog = Math.log10(tRaw + logTimeOffset);
    augPostFeatures[timeDimRaw] = tLog;

    // Step 4: normalize all dims to [0, 1]. (For the time slot, JSON has
    // bounds [0, 1] so this is a no-op — matches Python's
    // skip_time_in_normalize=True.)
    const off = ti * dAug;
    for (let k = 0; k < dAug; k++) {
      testX[off + k] = (augPostFeatures[k] - normLo[k]) / (normHi[k] - normLo[k]);
    }

    const tPost = testX[off + timeDimAug];
    hTest[ti] = 1.0 - Math.exp(-Math.max(0.0, tPost) / gateTau);
    if (hTestNoise !== hTest) {
      hTestNoise[ti] = 1.0 - Math.exp(-Math.max(0.0, tPost) / noiseGateTau);
    }
  }

  // Determine WASM availability. Caller passes wasmCtx if available.
  const useWasm = wasmCtx && wasmCtx.wasm && wasmCtx.wasmN === n;
  let K_buf;          // Float64Array view into K matrix (column-major [n, nTimes])
  let K_wasm_ptr = 0;
  if (useWasm) {
    K_wasm_ptr = wasmCtx.wasm._malloc(n * nTimes * 8);
    K_buf = new Float64Array(wasmCtx.wasm.HEAPF64.buffer, K_wasm_ptr, n * nTimes);
  } else {
    K_buf = new Float64Array(n * nTimes);
  }

  const means = new Float64Array(nTimes);
  const kSelfPerTime = new Float64Array(nTimes);

  // Build K (column-major: K[i, j] is at K_buf[j*n + i]) and accumulate mean.
  for (let j = 0; j < nTimes; j++) {
    const xOff = j * dAug;
    const colOff = j * n;
    const hT = hTest[j];
    let mean = 0;

    for (let i = 0; i < n; i++) {
      const rowOff = i * dAug;

      // M_blind
      let r2 = 0;
      for (let k = 0; k < nBlindDims; k++) {
        const dim = blindActiveDims[k];
        const dd = (testX[xOff + dim] - X_flat[rowOff + dim]) / blindLS[k];
        r2 += dd * dd;
      }
      let r = Math.sqrt(r2);
      let s5r = SQRT5 * r;
      const kBlind = blindOS * (1 + s5r + (5 * r2) / 3) * Math.exp(-s5r);

      // M_specific
      r2 = 0;
      for (let k = 0; k < nSpecificDims; k++) {
        const dim = specificActiveDims[k];
        const dd = (testX[xOff + dim] - X_flat[rowOff + dim]) / specificLS[k];
        r2 += dd * dd;
      }
      r = Math.sqrt(r2);
      s5r = SQRT5 * r;
      const kSpecific = specificOS * (1 + s5r + (5 * r2) / 3) * Math.exp(-s5r);

      // RBF on time
      const dt = (testX[xOff + rbfTimeIdx] - X_flat[rowOff + rbfTimeIdx]) / rbfLS;
      const kRbf = rbfOS * Math.exp(-0.5 * dt * dt);

      const kVal = (kBlind + kSpecific + kRbf) * hT * hTrain[i];
      K_buf[colOff + i] = kVal;
      mean += kVal * alpha[i];
    }

    means[j] = mean * yMax;

    // k(x*, x*) for the variance computation: at self, all squared distances
    // are 0 → matern52 = outputscale. Time RBF self = rbfOS. Total pre-gate
    // self-kernel = blindOS + specificOS + rbfOS. Multiply by h(t_test)² for
    // the gated self-kernel.
    kSelfPerTime[j] = (blindOS + specificOS + rbfOS) * hT * hT;
  }

  // Variance: solve L V = K, then variance[j] = k_self[j] - ||V[:, j]||².
  const variances = new Float64Array(nTimes);
  const noiseKind = params.noise_kind || "global";
  const noiseScalar = params.noise || 0.0;
  if (useWasm && wasmCtx.ptrL) {
    // V is in-place: WASM _dtrsm_lower transforms K_buf into V.
    wasmCtx.wasm._dtrsm_lower(n, nTimes, wasmCtx.ptrL, K_wasm_ptr);
    // Compute column norms via WASM.
    wasmCtx.wasm._col_norms_sq_f64(n, nTimes, K_wasm_ptr, wasmCtx.ptrNorms);
    const normsView = new Float64Array(
      wasmCtx.wasm.HEAPF64.buffer,
      wasmCtx.ptrNorms,
      nTimes,
    );
    for (let j = 0; j < nTimes; j++) {
      let sigVar = Math.max(0, kSelfPerTime[j] - normsView[j]);
      if (noiseKind === "gated") {
        // h(t_test_j) is hTestNoise[j]; noise scales as h(t)² * σ²_n.
        sigVar += hTestNoise[j] * hTestNoise[j] * noiseScalar;
      }
      variances[j] = sigVar * yMax * yMax;
    }
    wasmCtx.wasm._free(K_wasm_ptr);
  } else {
    // JS fallback: per-column triangular solve (forward substitution).
    const v = new Float64Array(n);
    for (let j = 0; j < nTimes; j++) {
      const colOff = j * n;
      // Forward solve L v = K[:, j]
      for (let i = 0; i < n; i++) {
        let s = K_buf[colOff + i];
        const rowOff = i * n;
        for (let kk = 0; kk < i; kk++) s -= L_flat[rowOff + kk] * v[kk];
        v[i] = s / L_flat[rowOff + i];
      }
      let vNormSq = 0;
      for (let i = 0; i < n; i++) vNormSq += v[i] * v[i];
      let sigVar = Math.max(0, kSelfPerTime[j] - vNormSq);
      if (noiseKind === "gated") {
        sigVar += hTestNoise[j] * hTestNoise[j] * noiseScalar;
      }
      variances[j] = sigVar * yMax * yMax;
    }
  }

  return { means, variances };
}
