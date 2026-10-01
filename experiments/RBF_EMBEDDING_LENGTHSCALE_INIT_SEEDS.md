# RBF-embedding kernel: lengthscale & init — multi-seed verification

Reruns the new variants at seeds 1 and 2 (seed 0 from the original ablation is folded in) to verify determinism (or detect non-determinism) for each (variant, phase) combination.

## In-distribution v5 (mean ± std across seeds 0, 1, 2)

| variant | n_seeds | LOO RMSE | bLOO RMSE | bLOO PIT-KS | bLOO cov95 |
|---|---|---|---|---|---|
| `rbf_embedding_d1_fixed_ell` | 3 | 506 ± 0.0 | 729 ± 0.0 | 0.032 ± 0.0000 | 0.926 ± 0.0000 |
| `rbf_embedding_d1_linear_init` | 3 | 498 ± 0.0 | 761 ± 0.0 | 0.035 ± 0.0000 | 0.923 ± 0.0000 |
| `rbf_embedding_d1_linear_init_fixed_ell` | 3 | 510 ± 0.0 | 722 ± 0.0 | 0.038 ± 0.0000 | 0.930 ± 0.0000 |
| `rbf_embedding_d2_fixed_ell` | 3 | 512 ± 0.0 | 723 ± 0.0 | 0.040 ± 0.0000 | 0.932 ± 0.0000 |
| `rbf_embedding_d3_fixed_ell` | 3 | 510 ± 0.0 | 722 ± 0.0 | 0.038 ± 0.0000 | 0.927 ± 0.0000 |

## LOCO class 0 held out (mean ± std across seeds 0, 1, 2)

$n_{test} = 246$

| variant | n_seeds | RMSE | MAE | PIT-KS | cov95 |
|---|---|---|---|---|---|
| `rbf_embedding_d1_fixed_ell` | 3 | 3427 ± 0.0 | 2150 ± 0.0 | 0.156 ± 0.0000 | 0.878 ± 0.0000 |
| `rbf_embedding_d1_linear_init` | 3 | 2503 ± 0.0 | 2052 ± 0.0 | 0.208 ± 0.0000 | 1.000 ± 0.0000 |
| `rbf_embedding_d1_linear_init_fixed_ell` | 3 | 2624 ± 0.0 | 2165 ± 0.0 | 0.212 ± 0.0000 | 1.000 ± 0.0000 |
| `rbf_embedding_d2_fixed_ell` | 3 | 2730 ± 0.0 | 1964 ± 0.0 | 0.295 ± 0.0000 | 0.943 ± 0.0000 |
| `rbf_embedding_d3_fixed_ell` | 3 | 2795 ± 0.0 | 2005 ± 0.0 | 0.290 ± 0.0000 | 0.939 ± 0.0000 |

## LOCO class 1 held out (mean ± std across seeds 0, 1, 2)

$n_{test} = 149$

| variant | n_seeds | RMSE | MAE | PIT-KS | cov95 |
|---|---|---|---|---|---|
| `rbf_embedding_d1_fixed_ell` | 3 | 2764 ± 0.0 | 2096 ± 0.0 | 0.319 ± 0.0000 | 0.738 ± 0.0000 |
| `rbf_embedding_d1_linear_init` | 3 | 2873 ± 0.0 | 2220 ± 0.0 | 0.406 ± 0.0000 | 0.691 ± 0.0000 |
| `rbf_embedding_d1_linear_init_fixed_ell` | 3 | 2881 ± 0.0 | 2221 ± 0.0 | 0.364 ± 0.0000 | 0.718 ± 0.0000 |
| `rbf_embedding_d2_fixed_ell` | 3 | 2917 ± 0.0 | 2246 ± 0.0 | 0.371 ± 0.0000 | 0.718 ± 0.0000 |
| `rbf_embedding_d3_fixed_ell` | 3 | 2889 ± 0.0 | 2222 ± 0.0 | 0.366 ± 0.0000 | 0.718 ± 0.0000 |

## LOCO class 2 held out (mean ± std across seeds 0, 1, 2)

$n_{test} = 252$

| variant | n_seeds | RMSE | MAE | PIT-KS | cov95 |
|---|---|---|---|---|---|
| `rbf_embedding_d1_fixed_ell` | 3 | 926 ± 0.0 | 693 ± 0.0 | 0.332 ± 0.0000 | 1.000 ± 0.0000 |
| `rbf_embedding_d1_linear_init` | 3 | 770 ± 0.0 | 603 ± 0.0 | 0.305 ± 0.0000 | 1.000 ± 0.0000 |
| `rbf_embedding_d1_linear_init_fixed_ell` | 3 | 2425 ± 0.0 | 1892 ± 0.0 | 0.448 ± 0.0000 | 1.000 ± 0.0000 |
| `rbf_embedding_d2_fixed_ell` | 3 | 1141 ± 0.0 | 851 ± 0.0 | 0.425 ± 0.0000 | 1.000 ± 0.0000 |
| `rbf_embedding_d3_fixed_ell` | 3 | 1219 ± 0.0 | 940 ± 0.0 | 0.436 ± 0.0000 | 1.000 ± 0.0000 |

