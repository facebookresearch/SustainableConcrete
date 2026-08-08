# Joint-distance kernel family — comprehensive ablation

Identifies the best variant in the joint-distance kernel family (kernels of the form $K = M_\nu(\sqrt{d^2_\text{feat} + d^2_\text{cat}})$) and compares against the established baselines (`hamming`, `legacy_continuous_ard`, `rbf_embedding_d2`).

## In-distribution v5 (mean ± std across 3 seeds)

| variant | LOO RMSE | bLOO RMSE | bLOO PIT-KS | bLOO cov95 | S12-bLOO RMSE |
|---|---|---|---|---|---|
| `joint_chain_matern` | 495 ± 0.0 | 720 ± 0.0 | 0.033 ± 0.0000 | 0.937 ± 0.0000 | 839 ± 0.0 |
| `joint_hamming_matern_nu05` | 507 ± 0.0 | 720 ± 0.0 | 0.060 ± 0.0000 | 0.940 ± 0.0000 | 850 ± 0.0 |
| `joint_hamming_matern_nu25` | 523 ± 0.0 | 701 ± 0.0 | 0.030 ± 0.0000 | 0.937 ± 0.0000 | 817 ± 0.0 |
| `joint_embedding_matern_d1` | 495 ± 0.0 | 724 ± 0.0 | 0.034 ± 0.0000 | 0.937 ± 0.0000 | 844 ± 0.0 |
| `joint_embedding_matern_d2` | 496 ± 0.0 | 716 ± 0.0 | 0.031 ± 0.0000 | 0.937 ± 0.0000 | 833 ± 0.0 |
| `joint_embedding_matern_d3` | 495 ± 0.0 | 716 ± 0.0 | 0.030 ± 0.0000 | 0.938 ± 0.0000 | 832 ± 0.0 |
| `joint_embedding_matern_d1_linear_init` | 495 ± 0.0 | 716 ± 0.0 | 0.032 ± 0.0000 | 0.938 ± 0.0000 | 833 ± 0.0 |
| `joint_hamming_matern` | 495 ± 0.0 | 717 ± 0.0 | 0.030 ± 0.0000 | 0.937 ± 0.0000 | 835 ± 0.0 |
| `legacy_continuous_ard` | 539 ± 0.0 | 702 ± 0.0 | 0.040 ± 0.0000 | 0.921 ± 0.0000 | 836 ± 0.0 |
| `hamming` | 510 ± 0.0 | 738 ± 0.0 | 0.039 ± 0.0000 | 0.927 ± 0.0000 | 871 ± 0.0 |
| `rbf_embedding_d2` | 497 ± 0.0 | 757 ± 0.0 | 0.036 ± 0.0000 | 0.926 ± 0.0000 | 878 ± 0.0 |

## LOCO class 0 held out (mean ± std across 3 seeds)

$n_{test} = 246$

| variant | RMSE | MAE | PIT-KS | cov95 | CRPS |
|---|---|---|---|---|---|
| `joint_chain_matern` | 2853 ± 0.0 | 2385 ± 0.0 | 0.225 ± 0.0000 | 1.000 ± 0.0000 | 2022 ± 0.0 |
| `joint_hamming_matern_nu05` | 2379 ± 0.0 | 1916 ± 0.0 | 0.274 ± 0.0000 | 1.000 ± 0.0000 | 1734 ± 0.0 |
| `joint_hamming_matern_nu25` | 3719 ± 0.0 | 2340 ± 0.0 | 0.147 ± 0.0000 | 0.862 ± 0.0000 | 1792 ± 0.0 |
| `joint_embedding_matern_d1` | 3352 ± 0.0 | 2732 ± 0.0 | 0.300 ± 0.0000 | 1.000 ± 0.0000 | 2101 ± 0.0 |
| `joint_embedding_matern_d2` | 4148 ± 0.0 | 3431 ± 0.0 | 0.384 ± 0.0000 | 1.000 ± 0.0000 | 2497 ± 0.0 |
| `joint_embedding_matern_d3` | 2810 ± 0.0 | 2347 ± 0.0 | 0.238 ± 0.0000 | 1.000 ± 0.0000 | 2059 ± 0.0 |
| `joint_embedding_matern_d1_linear_init` | 3080 ± 0.0 | 2546 ± 0.0 | 0.263 ± 0.0000 | 1.000 ± 0.0000 | 2071 ± 0.0 |
| `joint_hamming_matern` | 2853 ± 0.0 | 2385 ± 0.0 | 0.225 ± 0.0000 | 1.000 ± 0.0000 | 2022 ± 0.0 |
| `legacy_continuous_ard` | 2882 ± 0.0 | 2385 ± 0.0 | 0.214 ± 0.0000 | 1.000 ± 0.0000 | 1851 ± 0.0 |
| `hamming` | 2730 ± 0.0 | 2265 ± 0.0 | 0.186 ± 0.0000 | 1.000 ± 0.0000 | 1720 ± 0.0 |
| `rbf_embedding_d2` | 2824 ± 0.0 | 1980 ± 0.0 | 0.227 ± 0.0000 | 0.894 ± 0.0000 | 1477 ± 0.0 |

## LOCO class 1 held out (mean ± std across 3 seeds)

$n_{test} = 149$

| variant | RMSE | MAE | PIT-KS | cov95 | CRPS |
|---|---|---|---|---|---|
| `joint_chain_matern` | 2807 ± 0.0 | 2165 ± 0.0 | 0.355 ± 0.0000 | 0.718 ± 0.0000 | 1671 ± 0.0 |
| `joint_hamming_matern_nu05` | 2228 ± 0.0 | 1734 ± 0.0 | 0.220 ± 0.0000 | 0.879 ± 0.0000 | 1256 ± 0.0 |
| `joint_hamming_matern_nu25` | 2731 ± 0.0 | 2119 ± 0.0 | 0.372 ± 0.0000 | 0.765 ± 0.0000 | 1583 ± 0.0 |
| `joint_embedding_matern_d1` | 2736 ± 0.0 | 2128 ± 0.0 | 0.374 ± 0.0000 | 0.765 ± 0.0000 | 1583 ± 0.0 |
| `joint_embedding_matern_d2` | 2795 ± 0.0 | 2174 ± 0.0 | 0.386 ± 0.0000 | 0.765 ± 0.0000 | 1621 ± 0.0 |
| `joint_embedding_matern_d3` | 2805 ± 0.0 | 2147 ± 0.0 | 0.320 ± 0.0000 | 0.745 ± 0.0000 | 1633 ± 0.0 |
| `joint_embedding_matern_d1_linear_init` | 2802 ± 0.0 | 2153 ± 0.0 | 0.337 ± 0.0000 | 0.745 ± 0.0000 | 1634 ± 0.0 |
| `joint_hamming_matern` | 2736 ± 0.0 | 2136 ± 0.0 | 0.391 ± 0.0000 | 0.765 ± 0.0000 | 1610 ± 0.0 |
| `legacy_continuous_ard` | 2912 ± 0.0 | 2227 ± 0.0 | 0.397 ± 0.0000 | 0.691 ± 0.0000 | 1751 ± 0.0 |
| `hamming` | 2649 ± 0.0 | 2016 ± 0.0 | 0.291 ± 0.0000 | 0.772 ± 0.0000 | 1517 ± 0.0 |
| `rbf_embedding_d2` | 2705 ± 0.0 | 2061 ± 0.0 | 0.351 ± 0.0000 | 0.718 ± 0.0000 | 1591 ± 0.0 |

## LOCO class 2 held out (mean ± std across 3 seeds)

$n_{test} = 252$

| variant | RMSE | MAE | PIT-KS | cov95 | CRPS |
|---|---|---|---|---|---|
| `joint_chain_matern` | 1113 ± 0.0 | 914 ± 0.0 | 0.261 ± 0.0000 | 1.000 ± 0.0000 | 687 ± 0.0 |
| `joint_hamming_matern_nu05` | 1577 ± 0.0 | 1289 ± 0.0 | 0.336 ± 0.0000 | 0.980 ± 0.0000 | 909 ± 0.0 |
| `joint_hamming_matern_nu25` | 871 ± 0.0 | 690 ± 0.0 | 0.296 ± 0.0000 | 1.000 ± 0.0000 | 617 ± 0.0 |
| `joint_embedding_matern_d1` | 1320 ± 0.0 | 1067 ± 0.0 | 0.315 ± 0.0000 | 1.000 ± 0.0000 | 784 ± 0.0 |
| `joint_embedding_matern_d2` | 1123 ± 0.0 | 918 ± 0.0 | 0.267 ± 0.0000 | 1.000 ± 0.0000 | 689 ± 0.0 |
| `joint_embedding_matern_d3` | 1304 ± 0.0 | 1053 ± 0.0 | 0.312 ± 0.0000 | 1.000 ± 0.0000 | 772 ± 0.0 |
| `joint_embedding_matern_d1_linear_init` | 1325 ± 0.0 | 1071 ± 0.0 | 0.314 ± 0.0000 | 1.000 ± 0.0000 | 785 ± 0.0 |
| `joint_hamming_matern` | 1113 ± 0.0 | 914 ± 0.0 | 0.261 ± 0.0000 | 1.000 ± 0.0000 | 687 ± 0.0 |
| `legacy_continuous_ard` | 1139 ± 0.0 | 908 ± 0.0 | 0.340 ± 0.0000 | 1.000 ± 0.0000 | 718 ± 0.0 |
| `hamming` | 1319 ± 0.0 | 1051 ± 0.0 | 0.321 ± 0.0000 | 0.996 ± 0.0000 | 775 ± 0.0 |
| `rbf_embedding_d2` | 849 ± 0.0 | 675 ± 0.0 | 0.302 ± 0.0000 | 1.000 ± 0.0000 | 710 ± 0.0 |

