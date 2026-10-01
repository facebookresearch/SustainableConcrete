# Per-class heteroscedastic noise ablation

Source kernel: `joint_hamming_matern` (current Occam-optimal production winner). Compares homoscedastic Gaussian likelihood (single global sigma_gp) against per-class heteroscedastic likelihood (3 separate sigma_c, one per material class).

## In-distribution v5 (mean across 3 seeds)

| noise mode | LOO RMSE | bLOO RMSE | bLOO PIT-KS | bLOO cov95 |
|---|---|---|---|---|
| homoscedastic | 495 | 717 | 0.030 | 0.937 |
| per_class | 500 | 743 | 0.030 | 0.937 |

### Learned per-class noise (best seed)

| class | sigma_c (psi, learned) | sigma_meas (psi, data) |
|---|---|---|
| 0 | 509 | 185 |
| 1 | 406 | 146 |
| 2 | 181 | 76 |

## LOCO class 0 (mean across 3 seeds)

$n_{test} = 246$

| noise mode | RMSE | PIT-KS | cov95 | CRPS |
|---|---|---|---|---|
| homoscedastic | 2853 | 0.225 | 1.000 | 2022 |
| per_class | 3596 | 0.369 | 1.000 | 2508 |

## LOCO class 1 (mean across 3 seeds)

$n_{test} = 149$

| noise mode | RMSE | PIT-KS | cov95 | CRPS |
|---|---|---|---|---|
| homoscedastic | 2736 | 0.391 | 0.765 | 1610 |
| per_class | 2865 | 0.369 | 0.993 | 1672 |

## LOCO class 2 (mean across 3 seeds)

$n_{test} = 252$

| noise mode | RMSE | PIT-KS | cov95 | CRPS |
|---|---|---|---|---|
| homoscedastic | 1113 | 0.261 | 1.000 | 687 |
| per_class | 1157 | 0.382 | 1.000 | 1357 |

