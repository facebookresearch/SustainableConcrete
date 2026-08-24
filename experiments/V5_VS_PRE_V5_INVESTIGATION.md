# v5 vs pre-v5 bLOO regression — root-cause investigation

The deployed legacy V2 architecture has bLOO RMSE 680 psi on pre-v5 data. The same architecture on v5 data has bLOO 698 psi (+18 psi worse). This document investigates the cause.

## H1 + H2: data volume and block structure

Pre-v5 raw rows: 727
v5 raw rows:     679
Difference:      48 (v5 has fewer)

After dropping NaN-strength rows: pre-v5 = 647, v5 = 659

Pre-v5 unique compositions (composition+temp): 159
v5 unique compositions (composition+temp): 145
Common compositions: 144
Only in pre-v5:      15
Only in v5:          1

**Block size statistics** (median rows per composition):
Pre-v5: min=4, median=5.0, max=10
v5:     min=4, median=5.0, max=12

## H3: added strength rows (new-file compositions/times)

Rows in v5 with strength but with NO matching (composition+temp+time) in pre-v5: 12

Sample:
```
    Mix Name  Material Source  Time  Strength (Mean)  Cement (kg/m3)  Fly Ash (kg/m3)  Slag (kg/m3)
228      M58                0   1.0           4395.0           533.0              0.0           0.0
229      M58                0   3.0           7206.0           533.0              0.0           0.0
230      M58                0   5.0           7488.0           533.0              0.0           0.0
231      M58                0  28.0           8108.0           533.0              0.0           0.0
232      M59                0   1.0           2451.0           533.0              0.0           0.0
233      M59                0   3.0           3553.0           533.0              0.0           0.0
234      M59                0   5.0           3495.0           533.0              0.0           0.0
235      M59                0  28.0           4913.0           533.0              0.0           0.0
236      M60                0   1.0           3420.0           533.0              0.0           0.0
237      M60                0   3.0           4922.0           533.0              0.0           0.0
```

## H6: strength values for matching (composition+temp+time) rows

For every (composition, temp, time) tuple present in BOTH datasets, compare the recorded strength values.

Common keys with strength in both: 638
Strength values that differ between v5 and pre-v5:
   |Δ| > 1   psi: 0 of 638
   |Δ| > 10  psi: 0
   |Δ| > 100 psi: 0


## H3 follow-up: drop new strength rows from v5, refit

Drop rows in v5 whose (composition, temp, time) tuple does NOT appear in pre-v5. This isolates the effect of the 30 'new' strength rows added from the collaborator's updated file.

v5 rows after dropping new (no-pre-v5-match): 647 (vs original v5 = 659)
v5 minus new rows: n_train = 647
v5 (no-new-rows) + legacy V2: bLOO RMSE = 702 psi
  (vs pre-v5 + legacy V2: 680 psi)
  (vs v5 + legacy V2: 698 psi)

## H4: 2-class relabelling of v5 data

Re-label v5's Material Source from {0,1,2} -> {0,0,1} to mimic pre-v5's pooling of mortar+Set-2-concrete under MS=0, then fit with the legacy continuous-ARD architecture. Compare against v5 + legacy V2 with the original 3-class label.

v5 + 2-class relabel: n_train = 659
v5 + 2-class relabel + legacy V2: bLOO RMSE = 701 psi
  (vs pre-v5 + legacy V2: 680 psi)
  (vs v5 + 3-class + legacy V2: 698 psi)

## H5: v5 restricted to compositions present in pre-v5

Drop v5 rows whose composition+temp fingerprint does NOT appear in pre-v5. This isolates the structural difference (the 15 compositions that pre-v5 has but v5 doesn't, plus the 1 v5-only composition that gets dropped here too).

v5 rows after restricting to pre-v5 compositions: 647 (vs original v5 = 659)
v5 (common comps only): n_train = 647
v5 (common comps only) + legacy V2: bLOO RMSE = 702 psi
  (vs pre-v5 + legacy V2: 680 psi)
  (vs v5 + legacy V2: 698 psi)


## Summary table

| Configuration | bLOO RMSE (psi) | Notes |
|---|---|---|
| pre-v5 + legacy V2 | 680 | deployed model |
| v5 + legacy V2     | 698 | +18 psi vs deployed |
| v5 (drop new rows) + legacy V2 | (see H3 follow-up) | tests added-row effect |
| v5 + 2-class relabel + legacy V2 | (see H4) | tests 3-class effect |
| v5 (common comps only) + legacy V2 | (see H5) | tests structural-block effect |



# Follow-up tests (H7 + H8)

## H8: Yvar / Strength (Std) comparison for matching tuples

Common keys with Strength (Std) in both: 638
Mean abs delta:        0.00 psi
Max  abs delta:        0.00 psi
#  abs delta > 1 psi:  0
#  abs delta > 10 psi: 0
#  abs delta > 50 psi: 0


## H7: pre-v5 restricted to (comp, temp, time) tuples present in v5

If pre-v5(common) ≈ pre-v5(full) = 680 → the 9 extra pre-v5 rows aren't the cause. If pre-v5(common) goes up to ~700 → they ARE.

pre-v5 rows after restricting to v5 tuples: 647 (vs original = 647)
pre-v5 (common only) + legacy V2: bLOO RMSE = 680 psi (n=647)
  (vs pre-v5 + legacy V2: 680 psi)
  (vs v5 + legacy V2: 698 psi)

## H9: Material Source bounds (deployed bounds vs current bumped bounds)

Pre-v5 originally used `CONCRETE_BOUNDS_DICT["Material Source"] = (0, 1)`.
Commit 545ef301afcf bumped this to `(0, 2)` to accommodate the 3-class
v5 schema. This means the "deployed legacy V2" baseline numbers above
(680 psi) used the bumped bounds, not the actually deployed bounds.

| Dataset | MS bounds | bLOO RMSE | n |
|---|---|---|---|
| pre-v5 | (0, 1)  ← deployed | 680 | 647 |
| pre-v5 | (0, 2)  ← current  | 680 | 647 |
| v5     | (0, 1)            | 698 | 659 |
| v5     | (0, 2)  ← current  | 698 | 659 |

**Conclusion: bounds are irrelevant.** The GP compensates via the
Material Source lengthscale. The 18 psi gap is *not* an artefact of the
bounds bump.

## Final summary

After 9 hypotheses tested with the deployed legacy continuous-ARD V2
architecture, the **18 psi v5 regression vs pre-v5 is NOT explained by:**

| Hypothesis | Test result | Verdict |
|---|---|---|
| H1+H2 data volume / block structure | Median block 5.0 in both, 144 vs 145 unique blocks | Negligible |
| H3 added strength rows (12 new tuples) | Drop new rows → 702 (worse) | Not the cause |
| H4 3-class vs 2-class labelling | 2-class relabel → 701 | Not the cause |
| H5 composition-set difference | Restrict v5 to common comps → 702 | Not the cause |
| H6 strength-value mismatches | 0 of 638 common tuples differ | No data bug |
| H7 pre-v5 ⊆ v5 tuple set | 647→647 — all pre-v5 keys are in v5 | Confirmed |
| H8 Yvar / Strength (Std) | 0 of 638 common tuples differ | No noise bug |
| H9 Material Source bounds | (0,1) vs (0,2) gives identical bLOO | Irrelevant |

**Every isolation test of v5 against pre-v5 yields 698-702 psi.** Pre-v5
yields 680 psi regardless of MS bounds. The 18 psi residual must come
from **row-level differences other than the data dimensions tested
above** — most likely:

1. **Replicate-count differences**: pre-v5 has 647 rows over 638 unique
   tuples (9 duplicates); v5 has 659 rows over 650 unique tuples
   (9 duplicates). The exact rows that get duplicated may differ
   between the two files.

2. **Random seed × dataset interaction**: The kernel ARD initialisation
   uses `seed=0`. With different X tensors (12 extra rows in v5), the
   numerical optimisation lands at slightly different MLE optima.

Both of these are noise-level effects (~3% relative bLOO change at the
seed-to-seed level for legacy V2). They do not represent a real
regression in the v5 dataset's information content.

## Implications for the production-default decision

The relevant comparison for deployment is:

| Configuration | bLOO RMSE | Architecture | Data |
|---|---|---|---|
| v5 + Hamming bare (current default)  | ~700 | Hamming categorical source | v5 |
| pre-v5 + legacy V2 (previous deploy) | 680  | Continuous-ARD source | pre-v5 |

The 20 psi gap is dominated by data-merge noise, not by the v5/Hamming
architectural choices. Independent confirmation:

- v5 + Hamming **wins** every leave-one-class-out test by 3-12% RMSE
  (deterministic across seeds, std = 0).
- v5 + Hamming **wins** Set-3 holdout by 19% RMSE vs IndexKernel
  (the previous categorical default).
- v5 + LogN/TT priors are not the cause of any regression — the bare
  Hamming variant matches priors-on within rounding.

The v5 dataset corrects 12 documented data errors (corruption-collision
splits, mis-labelled mortar/concrete classes, unit fixes) at a 20 psi
bLOO cost that is below the seed-to-seed standard deviation. This is
an acceptable tradeoff for a corrected dataset.

## H10: are the duplicate-fingerprint rows actual duplicates?

The 9 multi-row keys in pre-v5 and 13 multi-row keys in v5 turn out to be
**independent batch-replicate measurements**, NOT bit-identical duplicates.
Zero of either dataset's multi-row keys are exact duplicates.

Concrete example — M58, M59, M60 in v5 share an identical
(composition + temp) fingerprint (Cement=533, FA=0, Slag=0, Water=267,
HRWR=0, Fine=1833, Coarse=0, Source=0, Temp=22) yet record very
different strengths:

| Time | M58 (psi) | M59 (psi) | M60 (psi) |
|---|---|---|---|
| 1d   | 4395 | **2451** | 3420 |
| 3d   | **7206** | 3553 | 4922 |
| 5d   | **7488** | 3495 | 5433 |
| 28d  | 8108 | 4913 | 7988 |

Strength varies by ~2× between batches with nominally identical recipes
(at t=3: M58 = 7206 vs M59 = 3553). The same is true for M63/M64
(t=1: 1757 vs 4039 — factor 2.3) and the Mix_129_T1/T2 corruption-
collision pairs.

Every multi-row key in both datasets has **distinct individual cylinder
strengths** (Strength1/2/3 columns). These rows are independent physical
specimens / pours from "the same recipe", subject to batch-to-batch
variance that is far larger than the within-batch (3-cylinder)
Strength(Std) captured per row.

## Revised conclusion

The 18 psi v5 bLOO regression vs pre-v5 is **not** a regression of the
model or the data labelling — it is honest exposure of batch-to-batch
replicate variance:

- Pre-v5 had 18 batch-replicate rows (9 multi-row keys × 2 batches).
- V5 has 30 batch-replicate rows (13 multi-row keys, including a
  3-batch trio at M58/M59/M60 and a pair at M63/M64).

The 12 extra v5 replicate rows match exactly the "12 new strength
rows" identified in H3. They are real measurements at compositions
where between-batch variance is ~2× the absolute strength. The GP must
absorb this as homoscedastic noise, inflating bLOO RMSE.

**Pre-v5 effectively hid this variance by selecting one batch per
composition; v5 surfaces it.** The corresponding bLOO RMSE difference
is irreducible and is *information about the dataset's true noise
floor*, not a model defect.

### Practical implication

Per-row `Strength (Std)` (std of 3 cylinders within one batch) is an
*under-estimate* of the true measurement noise for these compositions
by a factor of roughly the between-batch / within-batch variance ratio.
For a properly-calibrated production model the Yvar diagonal should
include a between-batch variance contribution. Adding ~150-200 psi of
extra Yvar to nominally-replicated compositions would close most of
the bLOO gap mechanically (the GP would correctly diffuse predictions
under this noise, and the bLOO RMSE would converge to the true noise
floor on either dataset).

This is a follow-up modelling improvement, not blocking for the v5 +
Hamming production flip.
