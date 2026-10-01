# `include_blind=False` ablation

Tests whether the materials-class-INDEPENDENT 'blind Matern' branch is load-bearing in the V2 strength kernel. The full kernel is:

```
K = blind_matern(no_source + extras)  +  categorical_source_branch(source, no_source + extras)  +  additive_rbf_time(time)
```

This ablation runs the top source-kernel candidates with the blind branch removed, leaving only `source-aware + time`. If the source-aware kernel alone can absorb the blind branch's load, dropping it is a clean simplification (~10 fewer hyperparameters).

## In-distribution v5 (mean across 3 seeds)

| base kernel | include_blind | LOO RMSE | bLOO RMSE | bLOO PIT-KS | S12-bLOO RMSE |
|---|---|---|---|---|---|
| `joint_hamming_matern` | yes | 495 | 717 | 0.030 | 835 |
| `joint_hamming_matern` | **NO** | 532 | 760 | 0.037 | 896 |
| `rbf_embedding_d2` | yes | 497 | 757 | 0.036 | 878 |
| `rbf_embedding_d2` | **NO** | 570 | 722 | 0.042 | 849 |
| `hamming` | yes | 510 | 738 | 0.039 | 871 |
| `hamming` | **NO** | 560 | 706 | 0.038 | 836 |

## LOCO class 0 (mean across 3 seeds)

$n_{test} = 246$

| base kernel | include_blind | RMSE | PIT-KS | cov95 |
|---|---|---|---|---|
| `joint_hamming_matern` | yes | 2853 | 0.225 | 1.000 |
| `joint_hamming_matern` | **NO** | 2851 | 0.275 | 1.000 |
| `rbf_embedding_d2` | yes | 2824 | 0.227 | 0.894 |
| `rbf_embedding_d2` | **NO** | 2811 | 0.138 | 1.000 |
| `hamming` | yes | 2730 | 0.186 | 1.000 |
| `hamming` | **NO** | 2813 | 0.139 | 1.000 |

## LOCO class 1 (mean across 3 seeds)

$n_{test} = 149$

| base kernel | include_blind | RMSE | PIT-KS | cov95 |
|---|---|---|---|---|
| `joint_hamming_matern` | yes | 2736 | 0.391 | 0.765 |
| `joint_hamming_matern` | **NO** | 2219 | 0.376 | 0.584 |
| `rbf_embedding_d2` | yes | 2705 | 0.351 | 0.718 |
| `rbf_embedding_d2` | **NO** | 2581 | 0.254 | 0.886 |
| `hamming` | yes | 2649 | 0.291 | 0.772 |
| `hamming` | **NO** | 2666 | 0.247 | 0.919 |

## LOCO class 2 (mean across 3 seeds)

$n_{test} = 252$

| base kernel | include_blind | RMSE | PIT-KS | cov95 |
|---|---|---|---|---|
| `joint_hamming_matern` | yes | 1113 | 0.261 | 1.000 |
| `joint_hamming_matern` | **NO** | 1548 | 0.393 | 1.000 |
| `rbf_embedding_d2` | yes | 849 | 0.302 | 1.000 |
| `rbf_embedding_d2` | **NO** | 1604 | 0.439 | 0.770 |
| `hamming` | yes | 1319 | 0.321 | 0.996 |
| `hamming` | **NO** | 1227 | 0.350 | 0.992 |

