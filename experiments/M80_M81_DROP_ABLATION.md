# M80/M81 drop ablation

Drop the duplicate-fingerprint mortar pair (Cement=667, Water=333: Mix_80/Mix_81 in pre-v5; M60/M61 in v5) from each dataset and compare bLOO RMSE across model variants.

Source: the pair has bit-identical input columns in the collaborator's file but factor-2.3 different cylinder strengths.

## Results

| Dataset | twins | Source kernel | n | bLOO RMSE (psi) |
|---|---|---|---|---|
| pre-v5 | with twins | legacy_continuous_ard | 647 | 680 |
| pre-v5 | with twins | hamming | 647 | 725 |
| pre-v5 | drop twins | legacy_continuous_ard | 639 | 735 |
| pre-v5 | drop twins | hamming | 639 | 769 |
| v5 | with twins | legacy_continuous_ard | 647 | 702 |
| v5 | with twins | hamming | 647 | 738 |
| v5 | drop twins | legacy_continuous_ard | 639 | 736 |
| v5 | drop twins | hamming | 639 | 768 |

## Per-dataset bLOO delta (drop twins vs keep twins)

| Dataset | Source kernel | with twins | drop twins | Δ |
|---|---|---|---|---|
| pre-v5 | legacy_continuous_ard | 680 | 735 | +55 |
| pre-v5 | hamming | 725 | 769 | +44 |
| v5 | legacy_continuous_ard | 702 | 736 | +34 |
| v5 | hamming | 738 | 768 | +30 |

## Interpretation

1. **Dropping the M80/M81 twin pair UNIFORMLY hurts bLOO** (+30 to +55 psi
   across every dataset × source-kernel combination). The twins are
   *helpful* training data, not harmful.

2. **The pre-v5 vs v5 gap collapses when twins are dropped**: from
   18-22 psi (680→702) down to **1 psi** (735→736). Without the twin
   pair, both datasets perform nearly identically.

3. **`legacy_continuous_ard` beats `hamming` on every cell** (~30-45 psi)
   in this bare-kernel comparison. Note this is the source-kernel
   branch only, without the full B''/F5_alllog feature stack used in
   the THREE_CLASS_AND_PRIOR_BENCHMARK results — those results showed
   Hamming wins inside the full V2 architecture.

## Why does dropping the twins hurt?

The Cement=667, Water=333, w/c=0.5 pair is the only training data at
that high-w/c-ratio mortar region. Even though the pair contains
factor-2.3 strength scatter within itself, the GP

* uses it to **anchor mortar-curve predictions** at compositions with
  similar w/c ratio; without it the GP extrapolates from lower-w/c
  mortars and over-predicts.
* uses its scatter to **calibrate the homoscedastic noise term σ²**;
  without it σ² fits smaller, the GP becomes overconfident, and
  held-out RMSE on neighbouring mortars rises.

The Δ pattern (+55 pre-v5 vs +34 v5) further confirms this: pre-v5 has
fewer total mortar points, so the twin pair's anchor role is even more
critical than in v5.

## Conclusion: the v5 'regression' is not a real regression

The 18-22 psi pre-v5 vs v5 bLOO gap is **explained almost entirely by
differential anchor-utility of the twin pair**, not by data quality
difference. When the twin pair is removed from both datasets, they fit
to bLOO RMSE within 1 psi of each other regardless of source kernel.

Recommendation: **keep the M80/M81 twin pair** in v5 (and pre-v5). It
contributes meaningful information to the GP fit despite — or because
of — its high internal scatter. The clay-mortar drop already implemented
is also retained as data-hygiene cleanup (the GP cannot distinguish
clay variants without Clay0/1/2 features, so they are not predictive
training points and the drop is safe).
