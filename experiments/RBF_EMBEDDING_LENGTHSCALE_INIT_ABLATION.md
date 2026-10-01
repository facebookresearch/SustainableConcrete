# RBF-embedding kernel: lengthscale & init ablation

Tests two follow-up proposals: (1) fix the kernel lengthscale at 1 (since the embedding scale is a redundant DOF); (2) initialise the d=1 embedding at integer class labels (0, 1, 2) instead of the equilateral-simplex projection (-1, 0, +1).

## In-distribution v5 metrics

| variant | LOO RMSE | bLOO RMSE | bLOO PIT-KS | bLOO cov95 | S12-bLOO RMSE |
|---|---|---|---|---|---|
| `rbf_embedding_d1` | 497 | 761 | 0.037 | 0.927 | 885 |
| `rbf_embedding_d1_fixed_ell` | 506 | 729 | 0.032 | 0.926 | 854 |
| `rbf_embedding_d1_linear_init` | 498 | 761 | 0.035 | 0.923 | 884 |
| `rbf_embedding_d1_linear_init_fixed_ell` | 510 | 722 | 0.038 | 0.930 | 843 |
| `rbf_embedding_d2` | 497 | 757 | 0.036 | 0.926 | 878 |
| `rbf_embedding_d2_fixed_ell` | 512 | 723 | 0.040 | 0.932 | 848 |
| `rbf_embedding_d3` | 511 | 735 | 0.040 | 0.927 | 868 |
| `rbf_embedding_d3_fixed_ell` | 510 | 722 | 0.038 | 0.927 | 843 |
| `legacy_continuous_ard` | 539 | 702 | 0.040 | 0.921 | 836 |
| `hamming` | 510 | 738 | 0.039 | 0.927 | 871 |

## LOCO held-out class 0

$n_{test} = 246$

| variant | RMSE (psi) | MAE (psi) | PIT-KS | cov95 | CRPS (psi) |
|---|---|---|---|---|---|
| `rbf_embedding_d1` | 2891 | 2391 | 0.224 | 1.000 | 1881 |
| `rbf_embedding_d1_fixed_ell` | 3427 | 2150 | 0.156 | 0.878 | 1637 |
| `rbf_embedding_d1_linear_init` | 2503 | 2052 | 0.208 | 1.000 | 1537 |
| `rbf_embedding_d1_linear_init_fixed_ell` | 2624 | 2165 | 0.212 | 1.000 | 1561 |
| `rbf_embedding_d2` | 2824 | 1980 | 0.227 | 0.894 | 1477 |
| `rbf_embedding_d2_fixed_ell` | 2730 | 1964 | 0.295 | 0.943 | 1464 |
| `rbf_embedding_d3` | 3121 | 2220 | 0.299 | 0.874 | 1651 |
| `rbf_embedding_d3_fixed_ell` | 2795 | 2005 | 0.290 | 0.939 | 1489 |
| `legacy_continuous_ard` | 2882 | 2385 | 0.214 | 1.000 | 1851 |
| `hamming` | 2730 | 2265 | 0.186 | 1.000 | 1720 |

## LOCO held-out class 1

$n_{test} = 149$

| variant | RMSE (psi) | MAE (psi) | PIT-KS | cov95 | CRPS (psi) |
|---|---|---|---|---|---|
| `rbf_embedding_d1` | 2855 | 2204 | 0.393 | 0.691 | 1711 |
| `rbf_embedding_d1_fixed_ell` | 2764 | 2096 | 0.319 | 0.738 | 1599 |
| `rbf_embedding_d1_linear_init` | 2873 | 2220 | 0.406 | 0.691 | 1731 |
| `rbf_embedding_d1_linear_init_fixed_ell` | 2881 | 2221 | 0.364 | 0.718 | 1691 |
| `rbf_embedding_d2` | 2705 | 2061 | 0.351 | 0.718 | 1591 |
| `rbf_embedding_d2_fixed_ell` | 2917 | 2246 | 0.371 | 0.718 | 1714 |
| `rbf_embedding_d3` | 3166 | 2451 | 0.415 | 0.678 | 1932 |
| `rbf_embedding_d3_fixed_ell` | 2889 | 2222 | 0.366 | 0.718 | 1691 |
| `legacy_continuous_ard` | 2912 | 2227 | 0.397 | 0.691 | 1751 |
| `hamming` | 2649 | 2016 | 0.291 | 0.772 | 1517 |

## LOCO held-out class 2

$n_{test} = 252$

| variant | RMSE (psi) | MAE (psi) | PIT-KS | cov95 | CRPS (psi) |
|---|---|---|---|---|---|
| `rbf_embedding_d1` | 874 | 693 | 0.295 | 1.000 | 678 |
| `rbf_embedding_d1_fixed_ell` | 926 | 693 | 0.332 | 1.000 | 1046 |
| `rbf_embedding_d1_linear_init` | 770 | 603 | 0.305 | 1.000 | 772 |
| `rbf_embedding_d1_linear_init_fixed_ell` | 2425 | 1892 | 0.448 | 1.000 | 1644 |
| `rbf_embedding_d2` | 849 | 675 | 0.302 | 1.000 | 710 |
| `rbf_embedding_d2_fixed_ell` | 1141 | 851 | 0.425 | 1.000 | 1135 |
| `rbf_embedding_d3` | 1170 | 927 | 0.341 | 1.000 | 762 |
| `rbf_embedding_d3_fixed_ell` | 1219 | 940 | 0.436 | 1.000 | 1128 |
| `legacy_continuous_ard` | 1139 | 908 | 0.340 | 1.000 | 718 |
| `hamming` | 1319 | 1051 | 0.321 | 0.996 | 775 |

