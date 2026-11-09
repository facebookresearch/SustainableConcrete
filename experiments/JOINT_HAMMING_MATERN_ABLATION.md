# `joint_hamming_matern` kernel: ablation grid

Tests whether the joint feature + Hamming-categorical Matern kernel (a single Matern over a joint distance) closes the +33 psi bLOO architecture gap to ``legacy_continuous_ard`` while keeping proper categorical handling.

## In-distribution v5 (mean ± std across 3 seeds)

| variant | LOO RMSE | bLOO RMSE | bLOO PIT-KS | bLOO cov95 | S12-bLOO RMSE |
|---|---|---|---|---|---|
| `joint_hamming_matern` | 495 ± 0.0 | 717 ± 0.0 | 0.030 ± 0.0000 | 0.937 ± 0.0000 | 835 ± 0.0 |
| `hamming` | 510 ± 0.0 | 738 ± 0.0 | 0.039 ± 0.0000 | 0.927 ± 0.0000 | 871 ± 0.0 |
| `legacy_continuous_ard` | 539 ± 0.0 | 702 ± 0.0 | 0.040 ± 0.0000 | 0.921 ± 0.0000 | 836 ± 0.0 |
| `rbf_embedding_d2` | 497 ± 0.0 | 757 ± 0.0 | 0.036 ± 0.0000 | 0.926 ± 0.0000 | 878 ± 0.0 |

## LOCO class 0 held out (mean ± std across 3 seeds)

$n_{test} = 246$

| variant | RMSE | MAE | PIT-KS | cov95 | CRPS |
|---|---|---|---|---|---|
| `joint_hamming_matern` | 2853 ± 0.0 | 2385 ± 0.0 | 0.225 ± 0.0000 | 1.000 ± 0.0000 | 2022 ± 0.0 |
| `hamming` | 2730 ± 0.0 | 2265 ± 0.0 | 0.186 ± 0.0000 | 1.000 ± 0.0000 | 1720 ± 0.0 |
| `legacy_continuous_ard` | 2882 ± 0.0 | 2385 ± 0.0 | 0.214 ± 0.0000 | 1.000 ± 0.0000 | 1851 ± 0.0 |
| `rbf_embedding_d2` | 2824 ± 0.0 | 1980 ± 0.0 | 0.227 ± 0.0000 | 0.894 ± 0.0000 | 1477 ± 0.0 |

## LOCO class 1 held out (mean ± std across 3 seeds)

$n_{test} = 149$

| variant | RMSE | MAE | PIT-KS | cov95 | CRPS |
|---|---|---|---|---|---|
| `joint_hamming_matern` | 2736 ± 0.0 | 2136 ± 0.0 | 0.391 ± 0.0000 | 0.765 ± 0.0000 | 1610 ± 0.0 |
| `hamming` | 2649 ± 0.0 | 2016 ± 0.0 | 0.291 ± 0.0000 | 0.772 ± 0.0000 | 1517 ± 0.0 |
| `legacy_continuous_ard` | 2912 ± 0.0 | 2227 ± 0.0 | 0.397 ± 0.0000 | 0.691 ± 0.0000 | 1751 ± 0.0 |
| `rbf_embedding_d2` | 2705 ± 0.0 | 2061 ± 0.0 | 0.351 ± 0.0000 | 0.718 ± 0.0000 | 1591 ± 0.0 |

## LOCO class 2 held out (mean ± std across 3 seeds)

$n_{test} = 252$

| variant | RMSE | MAE | PIT-KS | cov95 | CRPS |
|---|---|---|---|---|---|
| `joint_hamming_matern` | 1113 ± 0.0 | 914 ± 0.0 | 0.261 ± 0.0000 | 1.000 ± 0.0000 | 687 ± 0.0 |
| `hamming` | 1319 ± 0.0 | 1051 ± 0.0 | 0.321 ± 0.0000 | 0.996 ± 0.0000 | 775 ± 0.0 |
| `legacy_continuous_ard` | 1139 ± 0.0 | 908 ± 0.0 | 0.340 ± 0.0000 | 1.000 ± 0.0000 | 718 ± 0.0 |
| `rbf_embedding_d2` | 849 ± 0.0 | 675 ± 0.0 | 0.302 ± 0.0000 | 1.000 ± 0.0000 | 710 ± 0.0 |

