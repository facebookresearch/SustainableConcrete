/**
 * Off-main-thread strength-model initialization.
 *
 * `initStrengthModel` rebuilds the 670x670 training kernel and Cholesky-factors
 * it -- about 100 MFLOP, measured at ~70 ms on desktop arm64 and therefore
 * roughly 230-460 ms on a mid-range phone. Run inline it blocks first paint and
 * every tap for that whole window.
 *
 * The factorization itself is not meaningfully optimizable in JS: array-of-
 * arrays measured 38-40 ms and a flat Float64Array rewrite came out *slower*
 * (46-50 ms), because V8 hoists the row pointer for `L[i][k]` while flat
 * indexing needs two adds per inner-loop access. So the win has to come from
 * moving the work, not shrinking it.
 *
 * This worker runs the identical `initStrengthModel` and posts the fully
 * derived params back. The big buffers are handed over as transferables, so
 * the ~3.6 MB factor costs nothing to return.
 */

import { initStrengthModel } from "./gp.mjs";

// Refuse to run outside a dedicated worker.
//
// This is what makes the handler below safe without an origin check, and it is
// enforced rather than assumed. A dedicated worker is addressable only by the
// document that constructed it -- no cross-origin window can postMessage into
// it -- and its message events carry no origin to verify. Measured in
// Chromium: inside a dedicated worker `event.origin` is "" and `event.source`
// is null, where a window handler sees the real origin.
//
// The hazard the check removes is real. This file is a static asset, so if it
// were ever imported into the page instead of constructed as a Worker, `self`
// would be the Window and `self.onmessage` would install an unguarded
// window-level message handler that any cross-origin opener could drive --
// precisely the CodeQL js/missing-origin-check pattern. Failing loudly means
// that misuse can never silently become a vulnerability.
if (
  typeof DedicatedWorkerGlobalScope === "undefined" ||
  !(self instanceof DedicatedWorkerGlobalScope)
) {
  throw new Error(
    "model_init_worker.mjs must be constructed as a dedicated Worker, not " +
    "imported into a document. Loading it in a window scope would install " +
    "an unguarded message handler.",
  );
}

self.onmessage = (e) => {
  // Provenance gate. In a dedicated worker the only possible sender is the
  // parent document, and such events are dispatched with an empty origin.
  // Anything else means this is not the context asserted above, so drop the
  // message rather than act on untrusted input.
  if (e.origin !== "") return;

  const params = e.data;
  try {
    initStrengthModel(params);
  } catch (err) {
    // Surface the real reason; ui.mjs falls back to synchronous init.
    self.postMessage({ __error: String((err && err.message) || err) });
    return;
  }

  // Hand over the large typed arrays instead of copying them. Anything not
  // listed here is structure-cloned, which is fine for the small fields.
  const transfer = [];
  for (const key of ["L_flat", "X_train_flat", "alpha_f64", "_hTrain"]) {
    const buf = params[key] && params[key].buffer;
    // Guard against two views sharing one buffer -- transferring twice throws.
    if (buf && !transfer.includes(buf)) transfer.push(buf);
  }
  self.postMessage(params, transfer);
};
