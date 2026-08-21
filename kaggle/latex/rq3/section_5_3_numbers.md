# Section 5.3 - computed values

All values are test accuracy in percentage points, over the four models trained in this project.

## Mean accuracy by bridge length

| Model | Length 1 | Length 2 | Length 3 | Within-language |
|---|---|---|---|---|
| SPECTRA-Siam | 59.62 | 63.28 | 61.13 | 68.54 |
| ASTNN | 57.93 | 58.40 | 58.86 | 66.67 |
| RtvNN | 65.59 | 64.63 | 64.01 | 74.85 |
| DeepSim | 62.96 | 63.27 | 62.35 | 75.06 |

## Best length-3 path per target, and its gap to within-language

| Target | Best length-3 | Within-language | Gap |
|---|---|---|---|
| C# (S) | 62.40 | 65.07 | +2.67 |
| Java (J) | 67.58 | 70.18 | +2.60 |
| Python (P) | 70.49 | 72.77 | +2.28 |
| C++ (C) | 67.40 | 66.13 | -1.27 |

## Effect of intermediate reinforcement (length 3)

| Model | +X2--X2 | +X3--X3 | +both |
|---|---|---|---|
| SPECTRA-Siam | +5.11 | +0.16 | +3.98 |
| ASTNN | +1.37 | +1.84 | +2.54 |
| RtvNN | +0.01 | +0.87 | +2.67 |
| DeepSim | +0.18 | +0.47 | +1.17 |

## Path spread at a fixed length (SPECTRA-Siam)

| Length | Target | Best | Worst | Spread |
|---|---|---|---|---|
| 1 | C# | 57.53 | 50.07 | 7.47 |
| 1 | Java | 68.78 | 59.17 | 9.61 |
| 1 | Python | 61.44 | 54.46 | 6.98 |
| 1 | C++ | 65.73 | 56.27 | 9.47 |
| 2 | C# | 63.07 | 54.67 | 8.40 |
| 2 | Java | 70.31 | 61.64 | 8.67 |
| 2 | Python | 69.08 | 61.64 | 7.44 |
| 2 | C++ | 69.60 | 55.73 | 13.87 |
| 3 | C# | 62.40 | 51.27 | 11.13 |
| 3 | Java | 67.58 | 55.50 | 12.07 |
| 3 | Python | 70.49 | 52.11 | 18.38 |
| 3 | C++ | 67.40 | 54.33 | 13.07 |

## Final bridge language before the target (SPECTRA-Siam, lengths 2 and 3)

| Final bridge | Mean accuracy | Configurations |
|---|---|---|
| C# | 63.22 | 36 |
| Python | 61.53 | 36 |
| C++ | 61.44 | 36 |
| Java | 61.19 | 36 |

## Ranking across all bridge configurations

- SPECTRA-Siam ranks first in **21 of 156** configurations

| Compared with | Mean margin (pp) | SPECTRA-Siam ahead in |
|---|---|---|
| ASTNN | +3.03 | 121 of 156 |
| RtvNN | -2.65 | 42 of 156 |
| DeepSim | -1.01 | 67 of 156 |
