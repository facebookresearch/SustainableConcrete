# Architecture ablation: drop-high-twin (M81/Mix_81)

Re-run the source-kernel × dataset grid after dropping the high-strength twin (Mix_81 in pre-v5, M61 in v5). Metrics are computed by ``experiments.three_class_ablation._eval_metrics``: LOO + bLOO + Sets-1+2-only bLOO, each with RMSE, MAE, coverage_95, PIT-KS, MLPD, CRPS.

## Full metrics table

| data | twin | kernel | n | LOO RMSE | LOO MAE | LOO cov95 | LOO PIT-KS | LOO MLPD | LOO CRPS | bLOO RMSE | bLOO MAE | bLOO cov95 | bLOO PIT-KS | bLOO MLPD | bLOO CRPS | S12-bLOO RMSE | S12-bLOO cov95 | S12-bLOO PIT-KS | S12-bLOO MLPD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre-v5 | with twins | legacy_continuous_ard | 647 | 533 | 382 | 0.951 | 0.071 | -- | 285 | 680 | 517 | 0.930 | 0.037 | -- | 385 | 794 | 0.904 | 0.041 | -- |
| pre-v5 | with twins | hamming | 647 | 508 | 362 | 0.941 | 0.066 | -- | 271 | 725 | 550 | 0.924 | 0.042 | -- | 405 | 848 | 0.904 | 0.048 | -- |
| pre-v5 | with twins | indexkernel_r2 | 647 | 535 | 366 | 0.935 | 0.082 | -- | 279 | 734 | 559 | 0.935 | 0.052 | -- | 415 | 861 | 0.901 | 0.071 | -- |
| pre-v5 | drop high | legacy_continuous_ard | 643 | 459 | 321 | 0.932 | 0.079 | -- | 241 | 770 | 577 | 0.922 | 0.039 | -- | 438 | 899 | 0.918 | 0.058 | -- |
| pre-v5 | drop high | hamming | 643 | 466 | 333 | 0.935 | 0.068 | -- | 249 | 744 | 573 | 0.916 | 0.032 | -- | 423 | 855 | 0.900 | 0.044 | -- |
| pre-v5 | drop high | indexkernel_r2 | 643 | 465 | 322 | 0.936 | 0.065 | -- | 240 | 788 | 590 | 0.924 | 0.051 | -- | 445 | 928 | 0.926 | 0.074 | -- |
| v5 | with twins | legacy_continuous_ard | 647 | 539 | 385 | 0.943 | 0.071 | -- | 288 | 702 | 514 | 0.921 | 0.040 | -- | 385 | 836 | 0.884 | 0.047 | -- |
| v5 | with twins | hamming | 647 | 510 | 360 | 0.935 | 0.079 | -- | 271 | 738 | 556 | 0.927 | 0.039 | -- | 418 | 871 | 0.901 | 0.053 | -- |
| v5 | with twins | indexkernel_r2 | 647 | 498 | 343 | 0.937 | 0.081 | -- | 258 | 747 | 540 | 0.949 | 0.056 | -- | 395 | 889 | 0.957 | 0.080 | -- |
| v5 | drop high | legacy_continuous_ard | 643 | 461 | 323 | 0.941 | 0.083 | -- | 242 | 736 | 545 | 0.927 | 0.072 | -- | 412 | 870 | 0.898 | 0.075 | -- |
| v5 | drop high | hamming | 643 | 463 | 327 | 0.932 | 0.075 | -- | 244 | 770 | 579 | 0.922 | 0.037 | -- | 435 | 897 | 0.916 | 0.058 | -- |
| v5 | drop high | indexkernel_r2 | 643 | 474 | 329 | 0.924 | 0.061 | -- | 243 | 745 | 536 | 0.935 | 0.046 | -- | 394 | 889 | 0.951 | 0.072 | -- |

## Δ from dropping the high twin (drop − with)

Negative Δ on RMSE/MAE/PIT-KS = improvement; positive Δ on coverage_95 = improvement (toward 0.95).

| data | kernel | Δ LOO RMSE | Δ bLOO RMSE | Δ S12-bLOO RMSE | Δ LOO cov95 | Δ bLOO cov95 | Δ LOO PIT-KS | Δ bLOO PIT-KS | Δ LOO MLPD | Δ bLOO MLPD |
|---|---|---|---|---|---|---|---|---|---|---|
| pre-v5 | legacy_continuous_ard | -73 | +90 | +106 | -0.019 | -0.008 | +0.008 | +0.002 | -- | -- |
| pre-v5 | hamming | -42.501 | +18.945 | +6.583 | -0.007 | -0.008 | +0.002 | -0.009 | -- | -- |
| pre-v5 | indexkernel_r2 | -70 | +54 | +67 | +0.001 | -0.011 | -0.017 | -0.001 | -- | -- |
| v5 | legacy_continuous_ard | -78 | +34.844 | +34.605 | -0.002 | +0.006 | +0.012 | +0.032 | -- | -- |
| v5 | hamming | -46.966 | +32.642 | +25.857 | -0.004 | -0.005 | -0.005 | -0.003 | -- | -- |
| v5 | indexkernel_r2 | -24.669 | -1.236 | +0.563 | -0.013 | -0.014 | -0.020 | -0.010 | -- | -- |
