/**
 * Slump GP posterior for the web explorer.
 *
 * Mirrors the BoTorch model fitted by boxcrete.slump_model.fit_slump_gp and
 * exported by experiments/regenerate_slump_json.py. Pinned against Python
 * reference posteriors by test/test_js_slump.mjs.
 *
 * Why this is separate from docs/gp.mjs rather than a generalization of it:
 * gp.mjs models strength, which is time-dependent. Its transformInput()
 * unconditionally log-scales a time dimension and its kernel() unconditionally
 * applies a time gate plus an RBF-time term. Slump is measured pre-cure and
 * has no time dimension at all. Threading schema flags through the strength
 * model's 60 fps hot path — which four existing tests pin — to share ~50 lines
 * is a bad trade. The linear algebra IS shared: cholesky and
 * solveTriangularLower are imported below rather than duplicated.
 *
 * Model: ConstantMean + ARD RBF (BoTorch SingleTaskGP default, so no
 * ScaleKernel and outputscale is exactly 1) over min-max normalized inputs
 * with the HRWR/binder ratio appended, with a Standardize outcome transform.
 */

import { cholesky, solveTriangularLower } from "./gp.mjs";

// Raw composition column indices. slump.json's exporter hard-fails if
// DEFAULT_X_COLUMNS ever reorders, so these cannot drift silently.
const I_CEMENT = 0;
const I_FLYASH = 1;
const I_SLAG = 2;
const I_HRWR = 4;

let _L = null; // Cholesky factor of K + noise·I, lower-triangular
let _alpha = null; // (K + noise·I)^-1 (y_std - mean_constant)
let _params = null;

/**
 * Append the HRWR/binder ratio, matching
 * boxcrete.features.AppendDerivedFeatures.transform.
 *
 * The clamp is max(binder, 1.0). This differs from
 * FEATURE_FNS.hrwr_binder in docs/feature_registry.mjs, which the STRENGTH
 * model uses and which computes hrwr / (binder + 1.0). The two disagree by
 * ~0.3% on real mixes; test/test_js_slump.mjs pins the difference in both
 * directions so neither can drift into the other.
 */
export function augmentSlumpInput(composition) {
  const binder = Math.max(
    composition[I_CEMENT] + composition[I_FLYASH] + composition[I_SLAG],
    1.0
  );
  return [...composition, composition[I_HRWR] / binder];
}

/** Min-max to the unit cube. BoTorch's Normalize does not clamp. */
function normalize(xAug, params) {
  const lo = params.normalize_lower;
  const hi = params.normalize_upper;
  const out = new Array(xAug.length);
  for (let d = 0; d < xAug.length; d++) {
    const range = hi[d] - lo[d];
    out[d] = range > 0 ? (xAug[d] - lo[d]) / range : 0;
  }
  return out;
}

/** k(a,b) = exp(-0.5 · Σ_d ((a_d − b_d)/ℓ_d)²). Outputscale is 1. */
function rbfArd(a, b, lengthscales) {
  let acc = 0;
  for (let d = 0; d < a.length; d++) {
    const r = (a[d] - b[d]) / lengthscales[d];
    acc += r * r;
  }
  return Math.exp(-0.5 * acc);
}

/**
 * Factor the training covariance once. Called from init() before first paint;
 * ~61×61, so a few milliseconds.
 */
export function initSlumpModel(params) {
  const n = params.n_train;
  const X = params.X_train; // already normalized + augmented
  const ls = params.lengthscales;

  const K = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      const v = rbfArd(X[i], X[j], ls);
      K[i][j] = v;
      K[j][i] = v;
    }
    K[i][i] += params.noise;
  }

  _L = cholesky(K);
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(_L[i][i]) || _L[i][i] <= 0) {
      throw new Error(
        `slump.json produced a non-PSD covariance at row ${i} ` +
          `(L[i][i]=${_L[i][i]}); the artifact is corrupt.`
      );
    }
  }

  // α = (K + σ²I)^-1 (y − m), via L Lᵀ α = (y − m).
  const centered = params.Y_train.map((y) => y - params.mean_constant);
  const z = solveTriangularLower(_L, centered);
  const alpha = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let acc = z[i];
    for (let j = i + 1; j < n; j++) {
      acc -= _L[j][i] * alpha[j]; // Lᵀ[i][j] === L[j][i]
    }
    alpha[i] = acc / _L[i][i];
  }
  _alpha = alpha;
  _params = params;
}

/** True when the Material Source class has slump training data. */
export function slumpSupportsSource(sourceClass, params) {
  return (params || _params).supported_source_classes.includes(sourceClass);
}

/**
 * Posterior slump in INCHES, including observation noise (matching
 * observation_noise=True in the exporter, so the ±2σ band the UI renders is
 * predictive rather than epistemic-only).
 *
 * @param {number[]} composition 9 raw dims, no Time.
 * @returns {{mean: number, variance: number}}
 */
export function predictSlump(composition, params) {
  const p = params || _params;
  if (!_L) throw new Error("initSlumpModel() must run before predictSlump()");
  if (composition.length !== p.d_in) {
    throw new Error(`Expected ${p.d_in} composition dims, got ${composition.length}`);
  }

  const x = normalize(augmentSlumpInput(composition), p);
  const n = p.n_train;

  const kStar = new Array(n);
  for (let i = 0; i < n; i++) {
    kStar[i] = rbfArd(x, p.X_train[i], p.lengthscales);
  }

  let meanStd = p.mean_constant;
  for (let i = 0; i < n; i++) meanStd += kStar[i] * _alpha[i];

  // var = k(x,x) − ‖L⁻¹k*‖² + noise. k(x,x) === 1 exactly for an RBF with
  // no ScaleKernel.
  const v = solveTriangularLower(_L, kStar);
  let quad = 0;
  for (let i = 0; i < n; i++) quad += v[i] * v[i];
  let varStd = 1.0 - quad + p.noise;
  if (varStd < 0) varStd = 0; // floating-point floor near training points

  // Untransform Standardize: y = μ·σ_y + m_y, var scales by σ_y².
  return {
    mean: meanStd * p.y_std + p.y_mean,
    variance: varStd * p.y_std * p.y_std,
  };
}
