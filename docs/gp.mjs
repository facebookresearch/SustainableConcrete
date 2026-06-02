// GP inference library for BOxCrete interactive demo.
// Implements Matérn 5/2 + RBF additive kernel, input transforms,
// and exact GP posterior (mean + variance).
// Optionally uses WASM SIMD for accelerated variance computation.

import { predictStrengthCurveV2 } from "./gp_v2_fast.mjs";
import { appendFeatures } from "./feature_registry.mjs";

// --- WASM BLAS state (initialized asynchronously) ---
let _wasm = null;
let _wasmN = 0;     // matrix size stored in WASM
let _wasmD = 0;     // input dimensionality
// Persistent WASM memory pointers (allocated once, reused every frame)
let _ptrL = 0;      // L matrix (n×n, F64)
let _ptrX = 0;      // X_train (n×d, F64) — stored for reference
let _ptrNorms = 0;  // norms output buffer (max_nrhs, F64)
const _MAX_NRHS = 256;

/**
 * Initialize WASM BLAS for accelerated GP inference.
 * Allocates persistent WASM-resident buffers and copies the rebuilt
 * Cholesky factor `L` (from `initStrengthModel`) into them. `alpha`,
 * `X_train`, and lengthscales remain JS-side typed arrays and are
 * passed by pointer to BLAS calls per request. Call after
 * initStrengthModel(). Falls back gracefully if unavailable.
 * @param {object} params - Model params
 * @returns {Promise<boolean>} true if WASM loaded
 */
export async function initWASM(params) {
  try {
    const module = await import("./blas_f64.js");
    _wasm = await module.default();
    const n = params.n_train;
    const d = params.d_in;
    _wasmN = n;
    _wasmD = d;

    // Allocate persistent F64 buffers
    _ptrL = _wasm._malloc(n * n * 8);   // L matrix (F64)
    _ptrNorms = _wasm._malloc(_MAX_NRHS * 8); // norms output (F64)

    // Copy L into WASM memory (F64)
    const h64 = new Float64Array(_wasm.HEAPF64.buffer);
    const L = params.L_flat;
    for (let i = 0; i < n * n; i++) h64[_ptrL / 8 + i] = L[i];

    return true;
  } catch (e) {
    console.warn("WASM BLAS not available:", e.message);
    return false;
  }
}

/** @returns {boolean} Whether WASM acceleration is active */
export function isWASMReady() { return _wasm !== null; }

/**
 * Matérn 5/2 kernel with ARD lengthscales.
 * k(x1, x2) = outputscale * (1 + √5r + 5r²/3) * exp(-√5r)
 * where r = sqrt(Σᵢ ((x1ᵢ - x2ᵢ)/lᵢ)²)
 */
function matern52(x1, x2, lengthscales, outputscale) {
  let r2 = 0;
  for (let i = 0; i < x1.length; i++) {
    const d = (x1[i] - x2[i]) / lengthscales[i];
    r2 += d * d;
  }
  const r = Math.sqrt(r2);
  const sqrt5r = Math.sqrt(5) * r;
  return outputscale * (1 + sqrt5r + (5 * r2) / 3) * Math.exp(-sqrt5r);
}

/**
 * RBF (squared exponential) kernel on a single dimension.
 * k(t1, t2) = outputscale * exp(-0.5 * ((t1-t2)/l)²)
 */
function rbf(t1, t2, lengthscale, outputscale) {
  const d = (t1 - t2) / lengthscale;
  return outputscale * Math.exp(-0.5 * d * d);
}

/**
 * Matern-5/2 kernel restricted to a subset of input dims.
 * Used for the "blind" component of the multi-Matern kernel — it
 * operates on all dims EXCEPT the source dim, so the kernel is
 * source-agnostic.
 *
 * @param x1, x2 - full input vectors (length d_aug)
 * @param activeDims - array of dim indices to include (e.g. all except source)
 * @param lengthscales - per-active-dim lengthscales (length matches activeDims)
 * @param outputscale - kernel outputscale
 */
function matern52ActiveDims(x1, x2, activeDims, lengthscales, outputscale) {
  let r2 = 0;
  for (let k = 0; k < activeDims.length; k++) {
    const i = activeDims[k];
    const d = (x1[i] - x2[i]) / lengthscales[k];
    r2 += d * d;
  }
  const r = Math.sqrt(r2);
  const sqrt5r = Math.sqrt(5) * r;
  return outputscale * (1 + sqrt5r + (5 * r2) / 3) * Math.exp(-sqrt5r);
}

/**
 * Time gate: h(t) = 1 - exp(-t / tau).
 * Applied multiplicatively to the kernel: K_gated(x1, x2) = h(t1) * K(x1, x2) * h(t2).
 * This makes the prior covariance vanish at t = 0, structurally enforcing the
 * physics constraint f(x, 0) = 0 without anchor pseudo-observations.
 *
 * @param t - time value (post-input-transform; in [0, 1] roughly)
 * @param tau - gate timescale (default 0.05)
 */
function gateFunction(t, tau) {
  if (t < 0) t = 0;
  return 1.0 - Math.exp(-t / tau);
}

/**
 * Combined kernel.
 *
 * Schema v1 (legacy): ScaleKernel(Matérn5/2) + ScaleKernel(RBF on time).
 * Schema v2 (V2 strength GP): time-gated multi-Matern.
 *   K_gated(x1, x2) = h(t1) * (M_blind + M_specific + RBF_time) * h(t2)
 * where:
 *   - M_blind:    matern52 over all dims EXCEPT the source dim
 *   - M_specific: matern52 over ALL dims (including source)
 *   - RBF_time:   rbf on the time dim
 *   - h(t) = 1 - exp(-t / tau) with tau = 0.05
 */
function kernel(x1, x2, params) {
  const timeDim = params.time_dim_aug;
  const blind = params.matern_blind;
  const specific = params.matern_specific;
  const rbfT = params.rbf_time;
  // Multi-Matern base
  let kBase = matern52ActiveDims(
    x1, x2, blind.active_dims, blind.lengthscales, blind.outputscale,
  );
  kBase += matern52ActiveDims(
    x1, x2, specific.active_dims, specific.lengthscales, specific.outputscale,
  );
  // RBF on time dim only
  const tIdx = rbfT.active_dims[0];
  kBase += rbf(x1[tIdx], x2[tIdx], rbfT.lengthscale, rbfT.outputscale);
  // Multiplicative time gate: kernel * h(t1) * h(t2). At t=0 the kernel
  // is exactly zero, enforcing f(x, 0) = 0 in the prior (and posterior).
  const h1 = gateFunction(x1[timeDim], params.gate_tau);
  const h2 = gateFunction(x2[timeDim], params.gate_tau);
  return kBase * h1 * h2;
}

/**
 * F5_alllog engineered features — see `feature_registry.mjs` for the
 * actual implementations. Order MUST match Python's
 * `_FEATURE_CONFIGS["F5_alllog"]` and `engineered_feature_names`
 * in the JSON.
 */

/**
 * Apply input transforms (matches Python `ChainedInputTransform`):
 *   1. derive (append 7 engineered features) — uses RAW values for ALL
 *      dims, including raw time. `log_maturity_robust = log((T+10)*t + 1)`.
 *   2. log_offset on time: `time += 1`.
 *   3. log10 on time: `time = log10(time)`.
 *   4. normalize all dims (with the time slot using identity bounds [0, 1]
 *      because Python's Normalize was built with `skip_time_in_normalize=True`).
 *
 * KEY: features see RAW time, NOT post-log time. The Python order is
 * derive → log_offset → log → normalize (kwargs to ChainedInputTransform).
 */
function transformInput(x, params) {
  const timeDim = params.time_dim_raw;  // 9
  // 1. Append features using RAW input (before any time transformation).
  //    Use the registry-driven appendFeatures which dispatches by name
  //    so the JS stays in sync with whatever feature set the deployed
  //    model was trained on (advertised via engineered_feature_names).
  const out = appendFeatures(x, params.engineered_feature_names);
  // 2. log_offset on time (now in the augmented input).
  out[timeDim] = out[timeDim] + (params.log_time_offset || 1.0);
  // 3. log10 on time.
  out[timeDim] = Math.log10(out[timeDim]);
  // 4. Normalize all dims to [0, 1] using stored bounds.
  for (let i = 0; i < out.length; i++) {
    const lo = params.normalize_lower[i];
    const hi = params.normalize_upper[i];
    out[i] = (out[i] - lo) / (hi - lo);
  }
  return out;
}

/**
 * Compute the 7 engineered F5_alllog features from the RAW 10-dim
 * composition vector (NOT post-log on time). Order MUST match Python's
 * `_FEATURE_CONFIGS["F5_alllog"]`:
 *   wb_ratio, scm_frac, log_hrwr_binder, log_wc_ratio,
 *   log_coarse_fine, log_agg_paste, log_maturity_robust
 *
 * The +1.0 / +1e-3 / +1e-4 offsets match Python's `_FEATURE_BUILDERS`.
 *
 * Engineered features are computed via the registry in
 * `feature_registry.mjs`; the deployed model declares its feature set
 * via `engineered_feature_names` in strength.json.
 */

/**
 * Compute kernel vector k(x*, X_train) for a single test point.
 */
function kernelVector(xStar, XTrain, params) {
  const n = XTrain.length;
  const kVec = new Array(n);
  for (let i = 0; i < n; i++) {
    kVec[i] = kernel(xStar, XTrain[i], params);
  }
  return kVec;
}

/**
 * Cholesky decomposition of a symmetric positive-definite matrix.
 * Returns lower triangular L such that A = L L^T.
 * @param {number[][]} A - Symmetric positive-definite matrix (n×n).
 * @returns {number[][]} Lower triangular factor L.
 */
function cholesky(A) {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) {
        s -= L[i][k] * L[j][k];
      }
      L[i][j] = i === j ? Math.sqrt(s) : s / L[j][j];
    }
  }
  return L;
}

/**
 * Initialize a strength model: rebuild the training kernel matrix
 * `K_lik = K + σ²·I` from the kernel ingredients in `params`,
 * factor it via Cholesky, and solve `α = K_lik⁻¹ Y_train`. Call once on
 * page load.
 *
 * Note on the noise diagonal: we add the **bare scalar σ²**, not
 * `σ²·h(t)²`, exactly mirroring Python's posterior path (the V2
 * `GatedGaussianLikelihood` caches *raw* days in `_train_times`, and
 * `h(raw_t≥1 / 0.05)` saturates to ≈1.0 for the strength dataset, so
 * the gate is empirically a no-op on the training-data diagonal). The
 * kernel-side gate `h(t1)·k·h(t2)` IS still applied via `kernel(...)`
 * below — it's only the additional diagonal-noise gate that's skipped.
 *
 * Why we (re)compute these client-side instead of consuming a serialized
 * Cholesky factor: the Python and JS Cholesky implementations differ
 * subtly (jitter strategy, summation order), and pre-shipping `L` couples
 * the JS posterior to whichever Python backend happened to factor it.
 * That coupling caused a ~1e-3 relative variance drift between JS
 * inference and `test_vectors.json`. Rebuilding `K` here from the same
 * `kernel(x, x', params)` function the predictor uses, then running our
 * own Cholesky, makes both sides reach the same K from the same kernel
 * evaluations — drift is now bounded by FP-summation order (~ε·n ≈ 1e-13).
 *
 * Mutates params by adding `L_factor` (2D), `L_flat`, `alpha`, `alpha_f64`,
 * and `X_train_flat` fields used by the inference paths.
 * @param {object} params - Raw parameters from strength.json.
 */
export function initStrengthModel(params) {
  if (params.schema_version !== 2) {
    throw new Error(
      "initStrengthModel: only schema_version=2 is supported. " +
      "The legacy v1 schema was retired in 2026-05-17 — see " +
      "experiments/STRENGTH_GP_BENCHMARK.md §0a."
    );
  }
  if (!Array.isArray(params.Y_train)) {
    throw new Error(
      "initStrengthModel: `Y_train` is missing from strength.json. " +
      "Re-run experiments/regenerate_strength_json.py and commit the " +
      "regenerated artifact."
    );
  }
  if (!Array.isArray(params.X_train) || params.X_train.length === 0) {
    throw new Error(
      "initStrengthModel: `X_train` is missing or empty in strength.json. " +
      "Re-run experiments/regenerate_strength_json.py and commit the " +
      "regenerated artifact."
    );
  }
  if (typeof params.n_train !== "number" || typeof params.d_aug !== "number") {
    throw new Error(
      "initStrengthModel: `n_train` / `d_aug` schema fields missing. " +
      "Re-run experiments/regenerate_strength_json.py and commit the " +
      "regenerated artifact."
    );
  }
  const n = params.n_train;
  const X = params.X_train;
  const d = params.d_aug;
  params.n = n;

  // Convert X_train to flat Float64Array for cache-friendly access.
  const X_flat = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) X_flat[i * d + j] = X[i][j];
  }
  params.X_train_flat = X_flat;

  // Build the noisy training kernel matrix K_lik = K + σ²·I.
  // K[i, j] uses the same time-gated multi-Matern + RBF-time kernel that
  // the predictor calls. The diagonal noise is the bare scalar σ², NOT
  // σ²·h(t_i)² — this mirrors Python's actual posterior path: the V2
  // model's GatedGaussianLikelihood stores raw-day train times in its
  // `_train_times` cache, and h(raw_t ≥ 1 day / 0.05) saturates to ≈1.0,
  // so the gate is a no-op on the training diagonal. Mirroring this
  // empirical behaviour is what makes the JS posterior match
  // `test_vectors.json` (which was generated by the same Python posterior
  // path). The kernel gate h(t_i)·k·h(t_j) IS still applied — that comes
  // from the `kernel(...)` call below — only the additional diagonal-noise
  // gate is skipped.
  const noise = params.noise;
  const K_lik = Array.from({ length: n }, () => new Array(n));
  for (let i = 0; i < n; i++) {
    K_lik[i][i] = kernel(X[i], X[i], params) + noise;
    for (let j = 0; j < i; j++) {
      const k_ij = kernel(X[i], X[j], params);
      K_lik[i][j] = k_ij;
      K_lik[j][i] = k_ij;
    }
  }

  // Cholesky factor: K_lik = L Lᵀ. The kernel is PSD by construction, so
  // a successful chol with no jitter is the expected path; we add a tiny
  // adaptive jitter only on numerical failure (matches BoTorch's
  // psd_safe_cholesky escalation strategy).
  let L_2d;
  try {
    L_2d = cholesky(K_lik);
    // Detect NaN/Inf in the diagonal — silent failure mode of naive chol.
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(L_2d[i][i]) || L_2d[i][i] <= 0) throw new Error("non-PSD");
    }
  } catch (_e) {
    // Escalating jitter, mirroring linear_operator's psd_safe_cholesky.
    // Snapshot the original diagonal so each attempt sets jitter from the
    // base matrix rather than accumulating it across attempts.
    const diag0 = new Array(n);
    for (let i = 0; i < n; i++) diag0[i] = K_lik[i][i];
    let jitter = 1e-8;
    for (let attempt = 0; attempt < 6; attempt++) {
      for (let i = 0; i < n; i++) K_lik[i][i] = diag0[i] + jitter;
      try {
        L_2d = cholesky(K_lik);
        let ok = true;
        for (let i = 0; i < n; i++) {
          if (!Number.isFinite(L_2d[i][i]) || L_2d[i][i] <= 0) { ok = false; break; }
        }
        if (ok) break;
      } catch (_e2) { /* fall through */ }
      jitter *= 10;
    }
    if (!L_2d) {
      throw new Error("initStrengthModel: Cholesky factorization failed even after jitter escalation.");
    }
  }

  // Solve L Lᵀ α = Y_train (forward then back substitution).
  const Y = params.Y_train;
  const z = solveTriangularLower(L_2d, Y);
  const alpha = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = z[i];
    for (let j = i + 1; j < n; j++) s -= L_2d[j][i] * alpha[j];
    alpha[i] = s / L_2d[i][i];
  }

  // Materialize on params in both nested-2D (legacy predict path) and
  // flat Float64Array (gp_v2_fast.mjs / WASM BLAS) forms.
  params.L_factor = L_2d;
  const L_flat = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) L_flat[i * n + j] = L_2d[i][j];
  }
  params.L_flat = L_flat;
  params.alpha = alpha;
  const alpha_f64 = new Float64Array(n);
  for (let i = 0; i < n; i++) alpha_f64[i] = alpha[i];
  params.alpha_f64 = alpha_f64;

  // Map v2 fields onto the legacy y_std / y_mean / prior_mean fields so
  // the existing predict* functions un-scale correctly:
  // - max-scale: Y_scaled = Y / y_max, so un-scale: mean * y_max + 0.
  // - ZeroMean: prior_mean = 0.
  params.y_std = params.y_max;
  params.y_mean = params.y_mean ?? 0.0;
  params.prior_mean = 0.0;
  // UI compatibility: ui.mjs computes display std as sqrt(var + noiseVar)
  // where noiseVar = params.noise_variance * y_std². The schema-v2
  // export stores the noise as `params.noise` (in scaled-output space);
  // alias it to `noise_variance` for the legacy UI code path.
  params.noise_variance = params.noise;
}

/**
 * Solve L z = b for z, where L is lower triangular (forward substitution).
 */
function solveTriangularLower(L, b) {
  const n = b.length;
  const z = new Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let j = 0; j < i; j++) {
      s -= L[i][j] * z[j];
    }
    z[i] = s / L[i][i];
  }
  return z;
}

/**
 * Predict strength (mean and variance) for a raw composition + time.
 *
 * The returned variance is the **total** predictive variance: the latent
 * f-variance ``kSelf − ‖L⁻¹ k_*‖²`` PLUS the gated aleatoric term
 * ``h(t)² · σ²``, in scaled-output (Y / y_max) space, then un-standardised
 * by ``y_max²`` to psi². Both this single-point predictor and
 * :func:`predictStrengthCurve` honour the same contract, advertised by
 * ``strength.json::variance_includes_aleatoric: true`` — UI consumers
 * MUST NOT add noise on top.
 *
 * @param {number[]} composition - Raw composition values (without time).
 * @param {number} time - Curing time in days.
 * @param {object} params - Model parameters from strength.json.
 * @returns {{mean: number, variance: number}} variance includes aleatoric.
 */
export function predictStrength(composition, time, params) {
  // Build full input vector (composition + time)
  const x = [...composition, time];
  // Apply input transforms
  const xT = transformInput(x, params);

  // Fused mean + variance: compute kVec once
  const kVec = kernelVector(xT, params.X_train, params);

  let mean = params.prior_mean;
  for (let i = 0; i < kVec.length; i++) {
    mean += kVec[i] * params.alpha[i];
  }

  const kSelf = kernel(xT, xT, params);
  const v = solveTriangularLower(params.L_factor, kVec);
  let vNormSq = 0;
  for (let i = 0; i < v.length; i++) vNormSq += v[i] * v[i];
  let varStd = Math.max(0, kSelf - vNormSq);

  // Gated aleatoric: ``h(t_post)² · σ²`` in scaled-output space. Mirrors
  // ``predictStrengthCurveV2`` so single-point and batch paths return the
  // same total variance — see docstring above. Only applied when the
  // schema declares the gated-noise model (``noise_kind === "gated"``);
  // legacy / non-gated schemas read ``params.noise`` as a plain scalar
  // already added by the kernel-side noise term in those variants.
  if (params.noise_kind === "gated") {
    const tIdx = params.time_dim_aug;
    const h = gateFunction(xT[tIdx], params.noise_gate_tau ?? params.gate_tau);
    varStd += h * h * params.noise;
  }

  return {
    mean: mean * params.y_std + params.y_mean,
    variance: varStd * params.y_std * params.y_std,
  };
}

/**
 * Predict strength curve over a time range (optimized batch version).
 * Uses pre-computed L⁻¹ (flat Float64Array) for parallelizable variance,
 * flat X_train for cache locality, and typed alpha for fast dot products.
 * @param {number[]} composition - Raw composition values (without time).
 * @param {number[]} times - Array of time points.
 * @param {object} params - Model parameters from strength.json.
 * @returns {{means: number[], variances: number[]}}
 */
export function predictStrengthCurve(composition, times, params) {
  // Delegates to the WASM-accelerated batched V2 path. This inlines the
  // kernel computation (no per-pair function-call overhead so the JIT
  // can vectorise), batches all `times` test points, and routes the
  // variance solve through the WASM dtrsm + col_norms_sq if available.
  return predictStrengthCurveV2(composition, times, params, {
    wasm: _wasm, wasmN: _wasmN, ptrL: _ptrL, ptrNorms: _ptrNorms,
  });
}

/**
 * Predict strength curve MEAN ONLY (variance discarded). Currently a thin
 * wrapper around the full predictStrengthCurveV2 path that throws away
 * the variance — TODO: route to a dedicated mean-only fast path that
 * skips the Cholesky solve entirely (k_star^T @ alpha is much cheaper).
 */
export function predictStrengthMeanOnly(composition, times, params) {
  // Reuse the full V2 path and discard the variance.
  return predictStrengthCurveV2(composition, times, params, {
    wasm: _wasm, wasmN: _wasmN, ptrL: _ptrL, ptrNorms: _ptrNorms,
  }).means;
}

/**
 * Predict GWP using the linear model.
 * mean = Σᵢ xᵢ * cᵢ (negated, as model stores -GWP)
 * variance = Σᵢ xᵢ² * σᵢ²
 * @param {number[]} composition - Raw composition values (without time).
 * @param {object} gwpParams - Model parameters from gwp.json.
 * @param {number} [materialSource=0] - Material source class index.
 * @returns {{mean: number, variance: number}}
 */
export function predictGWP(composition, gwpParams, materialSource = 0) {
  const cls = String(materialSource);
  const coeffs = gwpParams.coefficients[cls];
  const means = coeffs.means;
  const variances = coeffs.variances;

  let mean = 0;
  let variance = 0;

  // If class-indexed, skip the class_dim column in dot product
  const classDim = gwpParams.class_dim;
  let ci = 0;
  for (let i = 0; i < composition.length; i++) {
    if (i === classDim) continue;
    mean += composition[i] * means[ci];
    variance += composition[i] * composition[i] * variances[ci];
    ci++;
  }

  return { mean, variance };
}

/**
 * Predict Cost using the linear model.
 * mean = Σᵢ xᵢ * cᵢ (negated, as model stores -Cost)
 * variance = Σᵢ xᵢ² * σᵢ²
 *
 * **Asymmetry vs predictGWP**: cost coefficients today are class-agnostic
 * (a single global ``{means, variances}`` pair, no ``class_dim`` field
 * in cost.json), so this function deliberately ignores the material
 * source class. ``predictGWP``, by contrast, indexes coefficients by
 * material-source class. If cost is ever retrained with class structure
 * — and cost.json grows a ``class_dim`` field — this function must be
 * updated symmetrically with predictGWP; the asymmetry is otherwise a
 * silent-fallback hazard.
 *
 * @param {number[]} composition - Raw composition values (without time).
 * @param {object} costParams - Model parameters (coefficients.means/variances).
 * @returns {{mean: number, variance: number}}
 */
export function predictCost(composition, costParams) {
  if (costParams.class_dim !== undefined) {
    throw new Error(
      "predictCost: cost.json has a ``class_dim`` field but this implementation " +
      "is class-agnostic. Update predictCost to mirror predictGWP's class-indexed " +
      "lookup before shipping the new cost.json."
    );
  }
  const means = costParams.coefficients.means;
  const variances = costParams.coefficients.variances;

  let mean = 0;
  let variance = 0;

  for (let i = 0; i < means.length; i++) {
    mean += composition[i] * means[i];
    variance += composition[i] * composition[i] * variances[i];
  }

  return { mean, variance };
}

// Export kernel functions for testing
export { matern52, rbf, kernel, transformInput, solveTriangularLower, cholesky };
