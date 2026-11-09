# M80/M81 twin diagnosis — Part A (corrected)

Hold both twins out, fit, and ask which is closer to the GP's best guess. Posterior is unscaled from `[0, 1]` (the V2 `maxscale_zeromean` outcome transform) back to psi.

### pre-v5 | source=legacy_continuous_ard

Trained on n=639 rows (twin pair removed). Posterior at twin compositions:

| Time | Mix_80 actual | Mix_81 actual | GP μ | GP σ | |μ−Mix_80| | |μ−Mix_81| | closer twin |
|------|---|---|---|---|---|---|---|
| 1 | 1757 | 4039 | 3089 | 1849 | 1332 | 950 | **Mix_81** |
| 3 | 2830 | 5538 | 4185 | 1847 | 1355 | 1353 | **Mix_81** |
| 5 | 3679 | 5600 | 4628 | 1847 | 949 | 972 | **Mix_80** |
| 28 | 5883 | 7076 | 5723 | 1855 | 160 | 1353 | **Mix_80** |

**Score**: 2/4 timepoints closer to Mix_80, 2/4 closer to Mix_81. RMSE vs Mix_80 = 1065 psi; RMSE vs Mix_81 = 1173 psi.

### pre-v5 | source=hamming

Trained on n=639 rows (twin pair removed). Posterior at twin compositions:

| Time | Mix_80 actual | Mix_81 actual | GP μ | GP σ | |μ−Mix_80| | |μ−Mix_81| | closer twin |
|------|---|---|---|---|---|---|---|
| 1 | 1757 | 4039 | 2972 | 1315 | 1215 | 1067 | **Mix_81** |
| 3 | 2830 | 5538 | 3915 | 1301 | 1085 | 1623 | **Mix_80** |
| 5 | 3679 | 5600 | 4319 | 1301 | 640 | 1281 | **Mix_80** |
| 28 | 5883 | 7076 | 5526 | 1322 | 357 | 1550 | **Mix_80** |

**Score**: 3/4 timepoints closer to Mix_80, 1/4 closer to Mix_81. RMSE vs Mix_80 = 893 psi; RMSE vs Mix_81 = 1398 psi.

### v5 | source=legacy_continuous_ard

Trained on n=639 rows (twin pair removed). Posterior at twin compositions:

| Time | M60 actual | M61 actual | GP μ | GP σ | |μ−M60| | |μ−M61| | closer twin |
|------|---|---|---|---|---|---|---|
| 1 | 1757 | 4039 | 3182 | 1854 | 1425 | 857 | **M61** |
| 3 | 2830 | 5538 | 4265 | 1852 | 1435 | 1273 | **M61** |
| 5 | 3679 | 5600 | 4701 | 1852 | 1022 | 899 | **M61** |
| 28 | 5883 | 7076 | 5809 | 1860 | 74 | 1267 | **M60** |

**Score**: 1/4 timepoints closer to M60, 3/4 closer to M61. RMSE vs M60 = 1134 psi; RMSE vs M61 = 1092 psi.

### v5 | source=hamming

Trained on n=639 rows (twin pair removed). Posterior at twin compositions:

| Time | M60 actual | M61 actual | GP μ | GP σ | |μ−M60| | |μ−M61| | closer twin |
|------|---|---|---|---|---|---|---|
| 1 | 1757 | 4039 | 2976 | 1320 | 1219 | 1063 | **M61** |
| 3 | 2830 | 5538 | 3908 | 1306 | 1078 | 1630 | **M60** |
| 5 | 3679 | 5600 | 4309 | 1306 | 630 | 1291 | **M60** |
| 28 | 5883 | 7076 | 5514 | 1327 | 369 | 1562 | **M60** |

**Score**: 3/4 timepoints closer to M60, 1/4 closer to M61. RMSE vs M60 = 892 psi; RMSE vs M61 = 1404 psi.

