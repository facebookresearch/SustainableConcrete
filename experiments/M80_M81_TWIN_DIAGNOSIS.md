# M80/M81 twin diagnosis — which is the outlier?

The Cement=667, Water=333 mortar pair (Mix_80/Mix_81 in pre-v5; M60/M61 in v5) records factor-2.3 different cylinder strengths for bit-identical input columns. Two diagnostics:

1. **Hold both out, predict, and ask which is closer to the GP's best guess.** The closer twin is more consistent with the rest of the dataset; the farther twin is the outlier.
2. **Run architecture ablations with one twin retained** to see whether the kernel-comparison conclusions are sensitive to the outlier.

## Part A — Which twin is the GP closer to?

### pre-v5 | source=legacy_continuous_ard

Trained on n=639 rows (twin pair removed). Posterior at twin compositions:

| Time | Mix_80 actual | Mix_81 actual | GP μ | GP σ | |μ−Mix_80| | |μ−Mix_81| | closer twin |
|------|---|---|---|---|---|---|---|
| 1 | 1757 | 4039 | 0 | 0 | 1757 | 4039 | **Mix_80** |
| 3 | 2830 | 5538 | 0 | 0 | 2830 | 5538 | **Mix_80** |
| 5 | 3679 | 5600 | 0 | 0 | 3679 | 5600 | **Mix_80** |
| 28 | 5883 | 7076 | 0 | 0 | 5883 | 7076 | **Mix_80** |

**Score**: 4/4 timepoints closer to Mix_80, 0/4 closer to Mix_81. RMSE vs Mix_80 = 3848 psi; RMSE vs Mix_81 = 5666 psi.

### pre-v5 | source=hamming

Trained on n=639 rows (twin pair removed). Posterior at twin compositions:

| Time | Mix_80 actual | Mix_81 actual | GP μ | GP σ | |μ−Mix_80| | |μ−Mix_81| | closer twin |
|------|---|---|---|---|---|---|---|
| 1 | 1757 | 4039 | 0 | 0 | 1757 | 4039 | **Mix_80** |
| 3 | 2830 | 5538 | 0 | 0 | 2830 | 5538 | **Mix_80** |
| 5 | 3679 | 5600 | 0 | 0 | 3679 | 5600 | **Mix_80** |
| 28 | 5883 | 7076 | 0 | 0 | 5883 | 7076 | **Mix_80** |

**Score**: 4/4 timepoints closer to Mix_80, 0/4 closer to Mix_81. RMSE vs Mix_80 = 3848 psi; RMSE vs Mix_81 = 5666 psi.

### v5 | source=legacy_continuous_ard

Trained on n=639 rows (twin pair removed). Posterior at twin compositions:

| Time | M60 actual | M61 actual | GP μ | GP σ | |μ−M60| | |μ−M61| | closer twin |
|------|---|---|---|---|---|---|---|
| 1 | 1757 | 4039 | 0 | 0 | 1757 | 4039 | **M60** |
| 3 | 2830 | 5538 | 0 | 0 | 2830 | 5538 | **M60** |
| 5 | 3679 | 5600 | 0 | 0 | 3679 | 5600 | **M60** |
| 28 | 5883 | 7076 | 0 | 0 | 5883 | 7076 | **M60** |

**Score**: 4/4 timepoints closer to M60, 0/4 closer to M61. RMSE vs M60 = 3848 psi; RMSE vs M61 = 5666 psi.

### v5 | source=hamming

Trained on n=639 rows (twin pair removed). Posterior at twin compositions:

| Time | M60 actual | M61 actual | GP μ | GP σ | |μ−M60| | |μ−M61| | closer twin |
|------|---|---|---|---|---|---|---|
| 1 | 1757 | 4039 | 0 | 0 | 1757 | 4039 | **M60** |
| 3 | 2830 | 5538 | 0 | 0 | 2830 | 5538 | **M60** |
| 5 | 3679 | 5600 | 0 | 0 | 3679 | 5600 | **M60** |
| 28 | 5883 | 7076 | 0 | 0 | 5883 | 7076 | **M60** |

**Score**: 4/4 timepoints closer to M60, 0/4 closer to M61. RMSE vs M60 = 3848 psi; RMSE vs M61 = 5666 psi.

## Part B — Architecture ablations with one twin retained

Re-run the bare source-kernel grid with two NEW data variants per dataset: `keep low only` (drop the high-strength twin) and `keep high only` (drop the low-strength twin).

| Dataset | twin variant | source kernel | n | bLOO RMSE (psi) |
|---|---|---|---|---|
| pre-v5 | keep low only | legacy_continuous_ard | 643 | 770 |
| pre-v5 | keep low only | hamming | 643 | 744 |
| pre-v5 | keep high only | legacy_continuous_ard | 643 | 777 |
| pre-v5 | keep high only | hamming | 643 | 774 |
| v5 | keep low only | legacy_continuous_ard | 643 | 736 |
| v5 | keep low only | hamming | 643 | 770 |
| v5 | keep high only | legacy_continuous_ard | 643 | 773 |
| v5 | keep high only | hamming | 643 | 776 |

### Reference values (from previous ablation)

| Dataset | source kernel | both twins | drop both |
|---|---|---|---|
| pre-v5 | legacy_continuous_ard | 680 | 735 |
| pre-v5 | hamming                | 725 | 769 |
| v5     | legacy_continuous_ard | 702 | 736 |
| v5     | hamming                | 738 | 768 |

