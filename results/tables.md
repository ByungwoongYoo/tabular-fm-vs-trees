
### concrete  (1030 rows, 8 features, reg, test 300 rows, 5 repeats; metric RMSE mean ± std)

| Model | n=100 | n=250 | n=500 | n=730 |
|---|---|---|---|---|
| Kumo | **6.277 ± 0.693** | **4.842 ± 0.240** | **3.965 ± 0.225** | **3.550 ± 0.272** |
| Causilo | 6.487 ± 0.747 | 5.003 ± 0.315 | 3.998 ± 0.158 | 3.589 ± 0.322 |
| XGB-tuned (tree) | 7.428 ± 0.342 | 5.591 ± 0.201 | 4.525 ± 0.166 | 4.100 ± 0.228 |
| HGB (tree) | 8.381 ± 0.223 | 5.924 ± 0.363 | 4.950 ± 0.103 | 4.421 ± 0.202 |
| RF (tree) | 8.475 ± 0.459 | 6.648 ± 0.324 | 5.546 ± 0.236 | 5.038 ± 0.239 |

| n | best FM | best tree | FM vs tree |
|---|---|---|---|
| 100 | Kumo 6.277 | XGB-tuned 7.428 | -15.5 % RMSE |
| 250 | Kumo 4.842 | XGB-tuned 5.591 | -13.4 % RMSE |
| 500 | Kumo 3.965 | XGB-tuned 4.525 | -12.4 % RMSE |
| 730 | Kumo 3.55 | XGB-tuned 4.1 | -13.4 % RMSE |

Data efficiency (rows the FM needs to match the best tree at the largest size; log-interpolated):

- Causilo: tree XGB-tuned at n=730 scores 4.1; FM matches at n≈466.1 → ratio 1.57  
- Kumo: tree XGB-tuned at n=730 scores 4.1; FM matches at n≈449.4 → ratio 1.62  

Model status: RF: finished; HGB: finished; XGB-tuned: finished; Causilo: finished; LightPFN: skipped [classification-only model]; FTFM-1: unavailable (NotImplementedError: FTFMRegressor is not yet implemented: regression weights have not bee); Kumo: finished; TabPFN: unavailable (TabPFNLicenseError: TabPFN requires a one-time license acceptance to download)

### energy_heating  (768 rows, 8 features, reg, test 256 rows, 5 repeats; metric RMSE mean ± std)

| Model | n=100 | n=250 | n=500 |
|---|---|---|---|
| Kumo | 0.593 ± 0.080 | **0.481 ± 0.023** | **0.410 ± 0.021** |
| Causilo | **0.592 ± 0.063** | 0.511 ± 0.050 | 0.427 ± 0.018 |
| XGB-tuned (tree) | 0.813 ± 0.130 | 0.519 ± 0.033 | 0.444 ± 0.024 |
| HGB (tree) | 2.502 ± 0.184 | 1.303 ± 0.372 | 0.541 ± 0.032 |
| RF (tree) | 1.593 ± 0.174 | 0.701 ± 0.120 | 0.513 ± 0.029 |

| n | best FM | best tree | FM vs tree |
|---|---|---|---|
| 100 | Causilo 0.5916 | XGB-tuned 0.8131 | -27.2 % RMSE |
| 250 | Kumo 0.4807 | XGB-tuned 0.5186 | -7.3 % RMSE |
| 500 | Kumo 0.41 | XGB-tuned 0.4442 | -7.7 % RMSE |

Data efficiency (rows the FM needs to match the best tree at the largest size; log-interpolated):

- Causilo: tree XGB-tuned at n=500 scores 0.4442; FM matches at n≈434.1 → ratio 1.15  
- Kumo: tree XGB-tuned at n=500 scores 0.4442; FM matches at n≈357.5 → ratio 1.4  

Model status: RF: finished; HGB: finished; XGB-tuned: finished; Causilo: finished; LightPFN: skipped [classification-only model]; FTFM-1: unavailable (NotImplementedError: FTFMRegressor is not yet implemented: regression weights have not bee); Kumo: finished; TabPFN: unavailable (TabPFNLicenseError: TabPFN requires a one-time license acceptance to download)

### steel_faults  (1941 rows, 27 features, cls, test 441 rows, 3 repeats; metric AUC mean ± std)

| Model | n=100 | n=250 | n=500 | n=1000 | n=1500 |
|---|---|---|---|---|---|
| Kumo | **0.9189 ± 0.0065** | **0.9472 ± 0.0084** | **0.9649 ± 0.0024** | **0.9742 ± 0.0010** | **0.9793 ± 0.0010** |
| Causilo | 0.9094 ± 0.0059 | 0.9423 ± 0.0107 | 0.9600 ± 0.0003 | 0.9711 ± 0.0015 | 0.9765 ± 0.0015 |
| FTFM-1 | 0.9152 ± 0.0041 | 0.9445 ± 0.0106 | 0.9640 ± 0.0011 | 0.9727 ± 0.0015 | 0.9777 ± 0.0012 |
| LightPFN | 0.9043 ± 0.0057 | 0.9330 ± 0.0092 | 0.9541 ± 0.0015 | 0.9663 ± 0.0016 | 0.9709 ± 0.0013 |
| XGB-tuned (tree) | 0.8794 ± 0.0119 | 0.9127 ± 0.0069 | 0.9402 ± 0.0049 | 0.9531 ± 0.0079 | 0.9635 ± 0.0057 |
| HGB (tree) | 0.8668 ± 0.0107 | 0.9077 ± 0.0079 | 0.9377 ± 0.0058 | 0.9559 ± 0.0087 | 0.9657 ± 0.0093 |
| RF (tree) | 0.8896 ± 0.0126 | 0.9128 ± 0.0094 | 0.9359 ± 0.0056 | 0.9508 ± 0.0082 | 0.9600 ± 0.0077 |

| n | best FM | best tree | FM vs tree |
|---|---|---|---|
| 100 | Kumo 0.9189 | RF 0.8896 | -26.5 % (1−AUC) |
| 250 | Kumo 0.9472 | RF 0.9128 | -39.4 % (1−AUC) |
| 500 | Kumo 0.9649 | XGB-tuned 0.9402 | -41.4 % (1−AUC) |
| 1000 | Kumo 0.9742 | HGB 0.9559 | -41.5 % (1−AUC) |
| 1500 | Kumo 0.9793 | HGB 0.9657 | -39.6 % (1−AUC) |

Data efficiency (rows the FM needs to match the best tree at the largest size; log-interpolated):

- Causilo: tree HGB at n=1500 scores 0.9657; FM matches at n≈711.3 → ratio 2.11  
- FTFM-1: tree HGB at n=1500 scores 0.9657; FM matches at n≈570.4 → ratio 2.63  
- Kumo: tree HGB at n=1500 scores 0.9657; FM matches at n≈529.3 → ratio 2.83  
- LightPFN: tree HGB at n=1500 scores 0.9657; FM matches at n≈963.3 → ratio 1.56  

Model status: RF: finished; HGB: finished; XGB-tuned: finished; Causilo: finished; LightPFN: finished; FTFM-1: finished; Kumo: finished; TabPFN: unavailable (TabPFNLicenseError: TabPFN requires a one-time license acceptance to download)

### wine_quality  (6497 rows, 12 features, reg, test 1000 rows, 3 repeats; metric RMSE mean ± std)

| Model | n=100 | n=250 | n=500 | n=1000 | n=2000 |
|---|---|---|---|---|---|
| Kumo | **0.753 ± 0.012** | 0.724 ± 0.023 | **0.700 ± 0.007** | **0.657 ± 0.003** | **0.628 ± 0.002** |
| Causilo | 0.768 ± 0.008 | **0.724 ± 0.018** | 0.706 ± 0.010 | 0.660 ± 0.003 | 0.630 ± 0.002 |
| XGB-tuned (tree) | 0.774 ± 0.016 | 0.764 ± 0.017 | 0.726 ± 0.015 | 0.692 ± 0.005 | 0.666 ± 0.022 |
| HGB (tree) | 0.806 ± 0.010 | 0.787 ± 0.017 | 0.759 ± 0.014 | 0.715 ± 0.006 | 0.682 ± 0.008 |
| RF (tree) | 0.784 ± 0.011 | 0.749 ± 0.016 | 0.729 ± 0.011 | 0.692 ± 0.004 | 0.660 ± 0.004 |

| n | best FM | best tree | FM vs tree |
|---|---|---|---|
| 100 | Kumo 0.7526 | XGB-tuned 0.7743 | -2.8 % RMSE |
| 250 | Causilo 0.7238 | RF 0.749 | -3.4 % RMSE |
| 500 | Kumo 0.7003 | XGB-tuned 0.7262 | -3.6 % RMSE |
| 1000 | Kumo 0.6572 | RF 0.6918 | -5.0 % RMSE |
| 2000 | Kumo 0.628 | RF 0.6599 | -4.8 % RMSE |

Data efficiency (rows the FM needs to match the best tree at the largest size; log-interpolated):

- Causilo: tree RF at n=2000 scores 0.6599; FM matches at n≈1011.0 → ratio 1.98  
- Kumo: tree RF at n=2000 scores 0.6599; FM matches at n≈958.0 → ratio 2.09  

Model status: RF: finished; HGB: finished; XGB-tuned: finished; Causilo: finished; LightPFN: skipped [classification-only model]; FTFM-1: unavailable (NotImplementedError: FTFMRegressor is not yet implemented: regression weights have not bee); Kumo: finished; TabPFN: unavailable (TabPFNLicenseError: TabPFN requires a one-time license acceptance to download)

### MagicTelescope  (19020 rows, 10 features, cls, test 1000 rows, 3 repeats; metric AUC mean ± std)

| Model | n=100 | n=250 | n=500 | n=1000 | n=2000 |
|---|---|---|---|---|---|
| Kumo | 0.8758 ± 0.0241 | 0.9226 ± 0.0178 | 0.9334 ± 0.0145 | 0.9448 ± 0.0139 | **0.9477 ± 0.0139** |
| Causilo | 0.8615 ± 0.0181 | 0.9165 ± 0.0217 | 0.9285 ± 0.0125 | 0.9401 ± 0.0117 | 0.9436 ± 0.0117 |
| FTFM-1 | **0.8791 ± 0.0177** | **0.9250 ± 0.0184** | **0.9352 ± 0.0143** | **0.9454 ± 0.0119** | 0.9466 ± 0.0135 |
| LightPFN | 0.8665 ± 0.0192 | 0.9106 ± 0.0124 | 0.9220 ± 0.0158 | 0.9360 ± 0.0134 | 0.9381 ± 0.0162 |
| XGB-tuned (tree) | 0.8169 ± 0.0143 | 0.8759 ± 0.0205 | 0.8991 ± 0.0220 | 0.9167 ± 0.0171 | 0.9205 ± 0.0186 |
| HGB (tree) | 0.8059 ± 0.0298 | 0.8735 ± 0.0183 | 0.8934 ± 0.0179 | 0.9136 ± 0.0145 | 0.9153 ± 0.0179 |
| RF (tree) | 0.8372 ± 0.0109 | 0.8902 ± 0.0170 | 0.9016 ± 0.0159 | 0.9190 ± 0.0153 | 0.9200 ± 0.0196 |

| n | best FM | best tree | FM vs tree |
|---|---|---|---|
| 100 | FTFM-1 0.8791 | RF 0.8372 | -25.7 % (1−AUC) |
| 250 | FTFM-1 0.925 | RF 0.8902 | -31.7 % (1−AUC) |
| 500 | FTFM-1 0.9352 | RF 0.9016 | -34.1 % (1−AUC) |
| 1000 | FTFM-1 0.9454 | RF 0.919 | -32.6 % (1−AUC) |
| 2000 | Kumo 0.9477 | XGB-tuned 0.9205 | -34.2 % (1−AUC) |

Data efficiency (rows the FM needs to match the best tree at the largest size; log-interpolated):

- Causilo: tree XGB-tuned at n=2000 scores 0.9205; FM matches at n≈315.1 → ratio 6.35  
- FTFM-1: tree XGB-tuned at n=2000 scores 0.9205; FM matches at n≈228.4 → ratio 8.76  
- Kumo: tree XGB-tuned at n=2000 scores 0.9205; FM matches at n≈239.6 → ratio 8.35  
- LightPFN: tree XGB-tuned at n=2000 scores 0.9205; FM matches at n≈455.5 → ratio 4.39  

Model status: RF: finished; HGB: finished; XGB-tuned: finished; Causilo: finished; LightPFN: finished; FTFM-1: finished; Kumo: finished; TabPFN: unavailable (TabPFNLicenseError: TabPFN requires a one-time license acceptance to download)

### superconduct  (21263 rows, 81 features, reg, test 1000 rows, 3 repeats; metric RMSE mean ± std)

| Model | n=100 | n=250 | n=500 | n=1000 | n=2000 |
|---|---|---|---|---|---|
| Kumo | **18.152 ± 1.445** | – | – | – | – |
| Causilo | 18.445 ± 1.054 | **15.708 ± 0.509** | **13.657 ± 0.763** | **12.208 ± 0.314** | **11.294 ± 0.697** |
| XGB-tuned (tree) | 20.845 ± 2.069 | 17.830 ± 1.401 | 14.867 ± 0.605 | 13.345 ± 0.250 | 12.167 ± 0.849 |
| HGB (tree) | 19.617 ± 1.235 | 17.492 ± 1.077 | 15.289 ± 0.808 | 13.776 ± 0.171 | 12.453 ± 0.774 |
| RF (tree) | 19.938 ± 1.997 | 17.698 ± 1.597 | 15.279 ± 0.420 | 13.670 ± 0.145 | 12.534 ± 0.532 |

| n | best FM | best tree | FM vs tree |
|---|---|---|---|
| 100 | Kumo 18.15 | HGB 19.62 | -7.5 % RMSE |
| 250 | Causilo 15.71 | HGB 17.49 | -10.2 % RMSE |
| 500 | Causilo 13.66 | XGB-tuned 14.87 | -8.1 % RMSE |
| 1000 | Causilo 12.21 | XGB-tuned 13.34 | -8.5 % RMSE |
| 2000 | Causilo 11.29 | XGB-tuned 12.17 | -7.2 % RMSE |

Data efficiency (rows the FM needs to match the best tree at the largest size; log-interpolated):

- Causilo: tree XGB-tuned at n=2000 scores 12.17; FM matches at n≈1032.1 → ratio 1.94  
- Kumo: tree HGB at n=100 scores 19.62; FM matches at n≈100.0 → ratio 1.0  FM already matches at the smallest size measured (100); ratio is a lower bound

Model status: RF: finished; HGB: finished; XGB-tuned: finished; Causilo: finished; LightPFN: skipped [classification-only model]; FTFM-1: unavailable (NotImplementedError: FTFMRegressor is not yet implemented: regression weights have not bee); Kumo: failed [killed: RSS 3025 MB > limit 3000 MB]; TabPFN: unavailable (TabPFNLicenseError: TabPFN requires a one-time license acceptance to download)
