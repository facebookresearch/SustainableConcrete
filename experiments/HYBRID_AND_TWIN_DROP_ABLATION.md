# Pareto-hybrid + twin-drop ablation

## In-distribution v5 (mean ± std across 3 seeds)

| variant | LOO RMSE | bLOO RMSE | bLOO PIT-KS | bLOO cov95 |
|---|---|---|---|---|
| `additive_joint_hamming_nu25_rbf_d2` | 488 ± 0.0 | 720 ± 0.0 | 0.036 ± 0.0000 | 0.935 ± 0.0000 |
| `additive_joint_hamming_rbf_d2` | 480 ± 0.0 | 746 ± 0.0 | 0.031 ± 0.0000 | 0.943 ± 0.0000 |
| `joint_hamming_matern_drop_m61` | 462 ± 0.0 | 719 ± 0.0 | 0.029 ± 0.0000 | 0.928 ± 0.0000 |

## LOCO class 0 (mean ± std across 3 seeds)

$n_{test} = 246$

| variant | RMSE | PIT-KS | cov95 |
|---|---|---|---|
| `additive_joint_hamming_nu25_rbf_d2` | 2958 ± 0.0 | 0.187 ± 0.0000 | 0.911 ± 0.0000 |
| `additive_joint_hamming_rbf_d2` | 2809 ± 0.0 | 0.181 ± 0.0000 | 0.931 ± 0.0000 |
| `joint_hamming_matern_drop_m61` | 2873 ± 0.0 | 0.223 ± 0.0000 | 1.000 ± 0.0000 |

## LOCO class 1 (mean ± std across 3 seeds)

$n_{test} = 149$

| variant | RMSE | PIT-KS | cov95 |
|---|---|---|---|
| `additive_joint_hamming_nu25_rbf_d2` | 2830 ± 0.0 | 0.408 ± 0.0000 | 0.732 ± 0.0000 |
| `additive_joint_hamming_rbf_d2` | 3387 ± 0.0 | 0.401 ± 0.0000 | 0.940 ± 0.0000 |
| `joint_hamming_matern_drop_m61` | 2805 ± 0.0 | 0.404 ± 0.0000 | 0.718 ± 0.0000 |

## LOCO class 2 (mean ± std across 3 seeds)

$n_{test} = 252$

| variant | RMSE | PIT-KS | cov95 |
|---|---|---|---|
| `additive_joint_hamming_nu25_rbf_d2` | 888 ± 0.0 | 0.237 ± 0.0000 | 1.000 ± 0.0000 |
| `additive_joint_hamming_rbf_d2` | 1235 ± 0.0 | 0.305 ± 0.0000 | 1.000 ± 0.0000 |
| `joint_hamming_matern_drop_m61` | 1251 ± 0.0 | 0.303 ± 0.0000 | 1.000 ± 0.0000 |

