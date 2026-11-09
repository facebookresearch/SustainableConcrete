# Noise audit: GP fitted likelihood noise vs measurement variance

Inspects each fitted model's likelihood noise σ_gp against the data's measurement noise (median sqrt(Yvar)) from Strength(Std). If σ_gp >> sqrt(Yvar), the GP is absorbing data structure as homoscedastic noise; if σ_gp ≈ sqrt(Yvar), the noise model matches the data.

## Measurement noise by class (data)

| class | n | median sqrt(Yvar) (psi) |
|---|---|---|
| 0 | 246 | 185 |
| 1 | 149 | 146 |
| 2 | 252 | 76 |

## GP fitted noise vs median measurement noise

| kernel | σ_gp (psi) | median sqrt(Yvar) (psi) | ratio σ_gp / sqrt(Yvar) |
|---|---|---|---|
| `joint_hamming_matern` | 370 | 109 | 3.39 |
| `rbf_embedding_d2` | 376 | 109 | 3.45 |
| `hamming` | 396 | 109 | 3.63 |

## Interpretation

* σ_gp ≈ 0: GP noise is negligible vs measurement noise. Yvar is doing all the work. No headroom from a heteroscedastic likelihood — the data already encodes per-row noise.
* σ_gp ≫ sqrt(Yvar): GP is absorbing extra noise on top of the measurement noise. Likely batch-level variance, compositional fingerprint collisions, or model misspecification. A heteroscedastic per-class noise could help if the extra variance is class-dependent.
* σ_gp ≈ sqrt(Yvar): GP's noise model matches the data's measurement noise. Noise is well-calibrated; further calibration improvements need to come from the kernel.

