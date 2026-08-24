# Pareto-frontier analysis across all v5 ablation variants

Aggregates per-cell metrics from all ablation grids we've run and identifies the Pareto-optimal frontier on the production-relevant trade-off:

* **in-distribution bLOO RMSE** (the in-distribution accuracy metric for BO over existing 3-class compositions)
* **held-out Class-2 (Set-3) LOCO RMSE** (the blind-class extrapolation metric)

## Per-variant summary (3-seed means)

| variant | LOO | bLOO | bLOO PIT-KS | S12-bLOO | LOCO-0 | LOCO-1 | LOCO-2 | weighted LOCO |
|---|---|---|---|---|---|---|---|---|
| `hamming` | 510 | 738 | 0.039 | 871 | 2730 | 2649 | 1319 | 2162 |
| `joint_chain_matern` | 495 | 720 | 0.033 | 839 | 2853 | 2807 | 1113 | 2165 |
| `joint_embedding_matern_d1` | 495 | 724 | 0.034 | 844 | 3352 | 2736 | 1320 | 2419 |
| `joint_embedding_matern_d1_linear_init` | 495 | 716 | 0.032 | 833 | 3080 | 2802 | 1325 | 2333 |
| `joint_embedding_matern_d2` | 496 | 716 | 0.031 | 833 | 4148 | 2795 | 1123 | 2658 |
| `joint_embedding_matern_d3` | 495 | 716 | 0.030 | 832 | 2810 | 2805 | 1304 | 2222 |
| `joint_hamming_matern` | 495 | 717 | 0.030 | 835 | 2853 | 2736 | 1113 | 2149 |
| `joint_hamming_matern_nu05` | 507 | 720 | 0.060 | 850 | 2379 | 2228 | 1577 | 2032 |
| `joint_hamming_matern_nu25` | 523 | 701 | 0.030 | 817 | 3719 | 2731 | 871 | 2382 |
| `legacy_continuous_ard` | 539 | 702 | 0.040 | 836 | 2882 | 2912 | 1139 | 2210 |
| `rbf_embedding_d2` | 497 | 757 | 0.036 | 878 | 2824 | 2705 | 849 | 2027 |

## Pareto frontier on (bLOO RMSE, Class-2 LOCO RMSE)

Variants on the frontier; no other variant beats them on both the in-distribution bLOO and the blind Class-2 LOCO metrics.

| rank | variant | bLOO RMSE | LOCO Class-2 RMSE |
|---|---|---|---|
| 1 | `joint_hamming_matern_nu25` | 701 | 871 |
| 2 | `rbf_embedding_d2` | 757 | 849 |

## Pareto frontier on (LOO RMSE, weighted LOCO RMSE)

Lower bound on the achievable LOO vs LOCO trade-off across all explored kernel architectures.

| rank | variant | LOO RMSE | weighted LOCO RMSE |
|---|---|---|---|
| 1 | `joint_hamming_matern` | 495 | 2149 |
| 2 | `rbf_embedding_d2` | 497 | 2027 |

## Best variant by criterion

| criterion | best variant | value |
|---|---|---|
| LOO RMSE | `joint_hamming_matern` | 495 |
| bLOO RMSE | `joint_hamming_matern_nu25` | 701 |
| bLOO PIT-KS | `joint_hamming_matern` | 0.030 |
| S12-bLOO RMSE | `joint_hamming_matern_nu25` | 817 |
| LOCO Class-0 RMSE | `joint_hamming_matern_nu05` | 2379 |
| LOCO Class-1 RMSE | `joint_hamming_matern_nu05` | 2228 |
| LOCO Class-2 RMSE | `rbf_embedding_d2` | 849 |
| weighted LOCO RMSE | `rbf_embedding_d2` | 2027 |

