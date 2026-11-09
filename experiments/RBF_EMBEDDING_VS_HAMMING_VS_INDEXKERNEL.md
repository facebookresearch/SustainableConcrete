# RBF-embedding source kernel: ablation grid

Compares the new ``rbf_embedding_d{1,2,3}`` source kernel (learned per-class embedding + RBF distance in embedding space) against the existing ``hamming`` and ``indexkernel_r2`` baselines.

**Setup**: bare prior (no LogNormal, no time-tying), full V2 B''+F5_alllog architecture, 3 seeds per cell, ``experiments.three_class_ablation._eval_metrics`` for scoring (LOO + bLOO + Sets-1+2-bLOO each with RMSE, MAE, coverage_95, PIT-KS, MLPD, CRPS).

## Per-cell mean ± std across 3 seeds

| data | kernel | LOO RMSE | bLOO RMSE | bLOO cov95 | bLOO PIT-KS | S12-bLOO RMSE | S12-bLOO PIT-KS |
|---|---|---|---|---|---|---|---|
| pre-v5 | hamming | 508 ± 0 | 725 ± 0 | 0.924 ± 0.000 | 0.042 ± 0.000 | 848 ± 0 | 0.048 ± 0.000 |
| pre-v5 | indexkernel_r2 | 514 ± 19 | 730 ± 10 | 0.941 ± 0.005 | 0.044 ± 0.015 | 862 ± 11 | 0.057 ± 0.024 |
| pre-v5 | rbf_embedding_d1 | 507 ± 0 | 725 ± 0 | 0.926 ± 0.000 | 0.036 ± 0.000 | 848 ± 0 | 0.041 ± 0.000 |
| pre-v5 | rbf_embedding_d2 | 511 ± 0 | 722 ± 0 | 0.929 ± 0.000 | 0.038 ± 0.000 | 844 ± 0 | 0.042 ± 0.000 |
| pre-v5 | rbf_embedding_d3 | 507 ± 0 | 725 ± 0 | 0.926 ± 0.000 | 0.032 ± 0.000 | 848 ± 0 | 0.041 ± 0.000 |
| v5 | hamming | 510 ± 0 | 738 ± 0 | 0.927 ± 0.000 | 0.039 ± 0.000 | 871 ± 0 | 0.053 ± 0.000 |
| v5 | indexkernel_r2 | 509 ± 13 | 738 ± 13 | 0.948 ± 0.010 | 0.047 ± 0.008 | 876 ± 17 | 0.062 ± 0.025 |
| v5 | rbf_embedding_d1 | 497 ± 0 | 761 ± 0 | 0.927 ± 0.000 | 0.037 ± 0.000 | 885 ± 0 | 0.056 ± 0.000 |
| v5 | rbf_embedding_d2 | 497 ± 0 | 757 ± 0 | 0.926 ± 0.000 | 0.036 ± 0.000 | 878 ± 0 | 0.053 ± 0.000 |
| v5 | rbf_embedding_d3 | 511 ± 0 | 735 ± 0 | 0.927 ± 0.000 | 0.040 ± 0.000 | 868 ± 0 | 0.055 ± 0.000 |

## Learned RBF embeddings (best seed per cell)

Per-class embedding vectors fit by MLE. Class 0 is pinned at the origin and class 1 lies on the first axis (gauge fix); class 2's position is free.

### pre-v5 / rbf_embedding_d1 (best seed = 0)

Lengthscale: ``0.368``

| class | x_0 |
|---|---|
| 0 | +0.000 |
| 1 | -1.986 |
| 2 | +1.000 |

### pre-v5 / rbf_embedding_d2 (best seed = 0)

Lengthscale: ``0.357``

| class | x_0 | x_1 |
|---|---|---|
| 0 | +0.000 | +0.000 |
| 1 | +2.037 | +0.000 |
| 2 | +0.500 | -0.866 |

### pre-v5 / rbf_embedding_d3 (best seed = 0)

Lengthscale: ``0.476``

| class | x_0 | x_1 | x_2 |
|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 |
| 1 | +1.789 | +0.000 | +0.000 |
| 2 | +0.500 | -0.866 | +0.000 |

### v5 / rbf_embedding_d1 (best seed = 0)

Lengthscale: ``0.648``

| class | x_0 |
|---|---|
| 0 | +0.000 |
| 1 | -1.257 |
| 2 | +1.672 |

### v5 / rbf_embedding_d2 (best seed = 0)

Lengthscale: ``0.360``

| class | x_0 | x_1 |
|---|---|---|
| 0 | +0.000 | +0.000 |
| 1 | +1.058 | +0.000 |
| 2 | +0.023 | -1.953 |

### v5 / rbf_embedding_d3 (best seed = 0)

Lengthscale: ``0.280``

| class | x_0 | x_1 | x_2 |
|---|---|---|---|
| 0 | +0.000 | +0.000 | +0.000 |
| 1 | +1.352 | +0.000 | +0.000 |
| 2 | +0.007 | -2.062 | +0.000 |

