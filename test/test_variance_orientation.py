# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Variance-orientation test: at every training composition, the
posterior variance should be lowest at the OBSERVED Material Source
class. If any other class shows lower σ², either:

  (a) the catalog has the wrong Material Source label for that
      composition (the v5 migration bug we just fixed); or
  (b) the kernel is mis-specified so that switching class makes the
      model artificially confident (an actual GP bug).

This test would have caught the catalog mislabelling that slipped
through every other test in the v5 migration: the GP predicts at
``(composition, catalog_class)`` and reports a high variance because
the model has no training data at that (composition, catalog_class)
pair — even though it *does* have training data at
``(composition, real_class)``. Toggling class in the explorer
contracts the variance, which is mathematically correct but only
because the catalog was wrong upstream.

Pre-registered tolerance: σ²(observed) <= σ²(any other class) for
EVERY training row. Even 1 swap is a structural failure; the failure
text identifies the row and which non-observed class undercut it.

The test uses ``fit_strength_gp`` to refit (~60 s); for CI it can be
gated behind a "slow" marker or only run on full-suite milestones.
"""

from __future__ import annotations

import torch

from boxcrete import fit_strength_gp, load_concrete_strength
from boxcrete.features import IDX


def test_variance_at_observed_class_is_lowest():
    """For every training row, σ²(observed class) ≤ σ²(other classes)."""
    torch.manual_seed(0)
    data = load_concrete_strength()
    X, Y, Yvar, bounds = data.strength_data
    source_dim = IDX["source"]
    model = fit_strength_gp(X=X, Y=Y, Yvar=Yvar, X_bounds=bounds, seed=0)
    model.eval()

    src = X[:, source_dim].round().long()
    unique_classes = sorted(src.unique().tolist())

    swaps: list[dict] = []
    with torch.no_grad():
        # Batch the rows for speed: ~647 rows × 3 classes = 1941 test
        # points; one posterior call handles them all.
        n = len(X)
        test_rows = []
        for i in range(n):
            x_i = X[i].clone()
            for c in unique_classes:
                x_c = x_i.clone()
                x_c[source_dim] = float(c)
                test_rows.append(x_c)
        x_test = torch.stack(test_rows)  # shape (n * C, d)
        post = model.posterior(x_test.unsqueeze(0))
        sigma2_all = post.variance.detach().squeeze().reshape(n, len(unique_classes))

    for i in range(n):
        obs_class = int(X[i, source_dim].round().item())
        obs_var = float(sigma2_all[i, obs_class])
        for c_idx, c in enumerate(unique_classes):
            if c == obs_class:
                continue
            other_var = float(sigma2_all[i, c_idx])
            # Strict inequality: σ²(other) < σ²(observed) means the
            # model is MORE confident at the unobserved class — a
            # symptom of the catalog/data mislabelling bug.
            if other_var < obs_var:
                swaps.append(
                    {
                        "row": i,
                        "observed_class": obs_class,
                        "other_class": c,
                        "obs_var": obs_var,
                        "other_var": other_var,
                    }
                )

    if swaps:
        first = swaps[0]
        msg = (
            f"\nVariance-orientation failure: {len(swaps)} of "
            f"{n * (len(unique_classes) - 1)} "
            f"(row, other-class) pairs have σ²(other) < σ²(observed). "
            f"This indicates the GP is more confident at an unobserved class "
            f"than at the observed one — either a data-labelling bug "
            f"(catalog/compositions.json drift from canonical) or a kernel "
            f"misspecification.\n\n"
            f"First failure: row {first['row']}, "
            f"observed class {first['observed_class']}, "
            f"σ²={first['obs_var']:.2f}; "
            f"class {first['other_class']} has "
            f"σ²={first['other_var']:.2f}."
        )
        raise AssertionError(msg)


if __name__ == "__main__":
    test_variance_at_observed_class_is_lowest()
    print("[ok] variance-orientation test passed on all training rows.")
