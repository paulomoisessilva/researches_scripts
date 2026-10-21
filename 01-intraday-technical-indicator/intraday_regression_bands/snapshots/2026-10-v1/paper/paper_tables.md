# Tabelas do paper (walk-forward OOS)

## Sharpe OOS por ano — validação 7d, custo 2+2 bps/lado

| symbol | strategy | 2021 | 2022 | 2023 | 2024 | 2025 | ALL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | bollinger | -0.44 | -1.85 | -2.15 | -2.89 | -1.06 | -1.61 |
| BTCUSDT | regression_bands | -1.69 | -0.61 | -3.25 | -2.48 | -2.39 | -1.94 |
| BTCUSDT | return_residuals | -7.16 | -11.18 | -13.30 | -13.53 | -14.95 | -11.28 |
| ETHUSDT | bollinger | -0.64 | -1.80 | -1.16 | -0.67 | -1.63 | -1.16 |
| ETHUSDT | regression_bands | -0.34 | -0.24 | -2.82 | -3.48 | -2.28 | -1.60 |
| ETHUSDT | return_residuals | -7.10 | -8.25 | -11.90 | -10.84 | -9.73 | -8.97 |
| SOLUSDT | bollinger | -1.76 | -0.07 | -0.54 | -0.51 | -1.48 | -0.87 |
| SOLUSDT | regression_bands | 0.80 | -0.36 | -1.60 | -1.87 | -0.67 | -0.60 |
| SOLUSDT | return_residuals | -1.56 | -5.59 | -5.84 | -7.49 | -7.86 | -5.11 |

### Placebo `random` (entre seeds; CBE em bps por lado)

| symbol | period | n_seeds | sharpe_p05 | sharpe_p50 | sharpe_p95 | break_even_bps_p05 | break_even_bps_p50 | break_even_bps_p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 2021 | 100 | -9.70 | -7.25 | -5.59 | 0.00 | 0.06 | 0.96 |
| BTCUSDT | 2022 | 100 | -11.11 | -9.41 | -7.11 | 0.00 | 0.00 | 0.84 |
| BTCUSDT | 2023 | 100 | -15.58 | -13.80 | -10.57 | 0.00 | 0.00 | 0.43 |
| BTCUSDT | 2024 | 100 | -13.38 | -10.97 | -9.45 | 0.00 | 0.04 | 0.56 |
| BTCUSDT | 2025 | 100 | -14.72 | -13.15 | -11.32 | 0.00 | 0.01 | 0.48 |
| BTCUSDT | ALL | 100 | -11.00 | -10.20 | -9.25 | 0.00 | 0.01 | 0.32 |
| ETHUSDT | 2021 | 100 | -7.79 | -5.77 | -3.80 | 0.00 | 0.04 | 1.23 |
| ETHUSDT | 2022 | 100 | -8.55 | -7.08 | -5.06 | 0.00 | 0.00 | 0.91 |
| ETHUSDT | 2023 | 100 | -14.48 | -12.69 | -9.52 | 0.00 | 0.00 | 0.45 |
| ETHUSDT | 2024 | 100 | -11.28 | -9.37 | -7.47 | 0.00 | 0.00 | 0.65 |
| ETHUSDT | 2025 | 100 | -9.92 | -8.16 | -6.54 | 0.00 | 0.05 | 0.73 |
| ETHUSDT | ALL | 100 | -8.60 | -7.80 | -7.09 | 0.00 | 0.01 | 0.31 |
| SOLUSDT | 2021 | 100 | -5.49 | -3.53 | -1.70 | 0.00 | 0.09 | 2.18 |
| SOLUSDT | 2022 | 100 | -6.51 | -4.46 | -2.52 | 0.00 | 0.00 | 1.58 |
| SOLUSDT | 2023 | 100 | -8.09 | -6.14 | -3.84 | 0.00 | 0.03 | 1.19 |
| SOLUSDT | 2024 | 100 | -7.95 | -6.29 | -4.65 | 0.00 | 0.00 | 0.95 |
| SOLUSDT | 2025 | 100 | -8.53 | -6.83 | -5.30 | 0.00 | 0.07 | 0.89 |
| SOLUSDT | ALL | 100 | -5.90 | -5.14 | -3.97 | 0.00 | 0.00 | 0.71 |

## Custo de break-even por ano (bps por lado que zeram o retorno)

| symbol | strategy | 2021 | 2022 | 2023 | 2024 | 2025 | ALL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | bollinger | 2.25 | 0.00 | 0.00 | 0.00 | 1.38 | 0.00 |
| BTCUSDT | regression_bands | 0.00 | 2.36 | 0.00 | 0.00 | 0.00 | 0.00 |
| BTCUSDT | return_residuals | 0.80 | 0.00 | 0.22 | 0.00 | 0.07 | 0.10 |
| ETHUSDT | bollinger | 0.73 | 0.00 | 1.03 | 1.88 | 0.00 | 0.00 |
| ETHUSDT | regression_bands | 2.17 | 3.18 | 0.00 | 0.00 | 0.00 | 0.00 |
| ETHUSDT | return_residuals | 0.00 | 0.00 | 0.09 | 0.00 | 0.00 | 0.00 |
| SOLUSDT | bollinger | 0.00 | 3.58 | 1.56 | 1.46 | 0.00 | 0.00 |
| SOLUSDT | regression_bands | 9.24 | 1.49 | 0.00 | 0.00 | 1.30 | 0.76 |
| SOLUSDT | return_residuals | 2.63 | 0.00 | 0.75 | 0.00 | 0.30 | 0.67 |

## Estratégias frente ao placebo (fração de seeds abaixo da estratégia)

| symbol | strategy | period | sharpe | random_sharpe_p50 | pct_random_below_sharpe | break_even_bps | random_cbe_p50 | random_cbe_p95 | pct_random_below_cbe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | bollinger | 2021 | -0.440 | -7.253 | 1.000 | 2.254 | 0.057 | 0.965 | 1.000 |
| BTCUSDT | bollinger | 2022 | -1.845 | -9.413 | 1.000 | 0.000 | 0.000 | 0.835 | 0.290 |
| BTCUSDT | bollinger | 2023 | -2.151 | -13.797 | 1.000 | 0.000 | 0.000 | 0.434 | 0.270 |
| BTCUSDT | bollinger | 2024 | -2.888 | -10.970 | 1.000 | 0.000 | 0.035 | 0.559 | 0.225 |
| BTCUSDT | bollinger | 2025 | -1.065 | -13.149 | 1.000 | 1.380 | 0.005 | 0.479 | 1.000 |
| BTCUSDT | bollinger | ALL | -1.608 | -10.198 | 1.000 | 0.000 | 0.012 | 0.319 | 0.245 |
| BTCUSDT | regression_bands | 2021 | -1.690 | -7.253 | 1.000 | 0.000 | 0.057 | 0.965 | 0.225 |
| BTCUSDT | regression_bands | 2022 | -0.614 | -9.413 | 1.000 | 2.364 | 0.000 | 0.835 | 1.000 |
| BTCUSDT | regression_bands | 2023 | -3.251 | -13.797 | 1.000 | 0.000 | 0.000 | 0.434 | 0.270 |
| BTCUSDT | regression_bands | 2024 | -2.480 | -10.970 | 1.000 | 0.000 | 0.035 | 0.559 | 0.225 |
| BTCUSDT | regression_bands | 2025 | -2.388 | -13.149 | 1.000 | 0.000 | 0.005 | 0.479 | 0.245 |
| BTCUSDT | regression_bands | ALL | -1.941 | -10.198 | 1.000 | 0.000 | 0.012 | 0.319 | 0.245 |
| BTCUSDT | return_residuals | 2021 | -7.159 | -7.253 | 0.540 | 0.804 | 0.057 | 0.965 | 0.930 |
| BTCUSDT | return_residuals | 2022 | -11.178 | -9.413 | 0.050 | 0.000 | 0.000 | 0.835 | 0.290 |
| BTCUSDT | return_residuals | 2023 | -13.299 | -13.797 | 0.590 | 0.215 | 0.000 | 0.434 | 0.800 |
| BTCUSDT | return_residuals | 2024 | -13.534 | -10.970 | 0.040 | 0.000 | 0.035 | 0.559 | 0.225 |
| BTCUSDT | return_residuals | 2025 | -14.950 | -13.149 | 0.030 | 0.071 | 0.005 | 0.479 | 0.550 |
| BTCUSDT | return_residuals | ALL | -11.283 | -10.198 | 0.020 | 0.100 | 0.012 | 0.319 | 0.660 |
| ETHUSDT | bollinger | 2021 | -0.640 | -5.771 | 1.000 | 0.734 | 0.044 | 1.227 | 0.810 |
| ETHUSDT | bollinger | 2022 | -1.804 | -7.078 | 1.000 | 0.000 | 0.000 | 0.912 | 0.265 |
| ETHUSDT | bollinger | 2023 | -1.164 | -12.688 | 1.000 | 1.027 | 0.000 | 0.449 | 1.000 |
| ETHUSDT | bollinger | 2024 | -0.671 | -9.371 | 1.000 | 1.885 | 0.001 | 0.648 | 1.000 |
| ETHUSDT | bollinger | 2025 | -1.630 | -8.165 | 1.000 | 0.000 | 0.051 | 0.728 | 0.230 |
| ETHUSDT | bollinger | ALL | -1.160 | -7.801 | 1.000 | 0.000 | 0.014 | 0.313 | 0.245 |
| ETHUSDT | regression_bands | 2021 | -0.339 | -5.771 | 1.000 | 2.173 | 0.044 | 1.227 | 0.990 |
| ETHUSDT | regression_bands | 2022 | -0.236 | -7.078 | 1.000 | 3.176 | 0.000 | 0.912 | 1.000 |
| ETHUSDT | regression_bands | 2023 | -2.816 | -12.688 | 1.000 | 0.000 | 0.000 | 0.449 | 0.265 |
| ETHUSDT | regression_bands | 2024 | -3.483 | -9.371 | 1.000 | 0.000 | 0.001 | 0.648 | 0.250 |
| ETHUSDT | regression_bands | 2025 | -2.284 | -8.165 | 1.000 | 0.000 | 0.051 | 0.728 | 0.230 |
| ETHUSDT | regression_bands | ALL | -1.600 | -7.801 | 1.000 | 0.000 | 0.014 | 0.313 | 0.245 |
| ETHUSDT | return_residuals | 2021 | -7.100 | -5.771 | 0.150 | 0.000 | 0.044 | 1.227 | 0.240 |
| ETHUSDT | return_residuals | 2022 | -8.247 | -7.078 | 0.130 | 0.000 | 0.000 | 0.912 | 0.265 |
| ETHUSDT | return_residuals | 2023 | -11.898 | -12.688 | 0.740 | 0.092 | 0.000 | 0.449 | 0.630 |
| ETHUSDT | return_residuals | 2024 | -10.841 | -9.371 | 0.110 | 0.000 | 0.001 | 0.648 | 0.250 |
| ETHUSDT | return_residuals | 2025 | -9.727 | -8.165 | 0.080 | 0.000 | 0.051 | 0.728 | 0.230 |
| ETHUSDT | return_residuals | ALL | -8.974 | -7.801 | 0.020 | 0.000 | 0.014 | 0.313 | 0.245 |
| SOLUSDT | bollinger | 2021 | -1.759 | -3.530 | 0.930 | 0.000 | 0.088 | 2.178 | 0.210 |
| SOLUSDT | bollinger | 2022 | -0.071 | -4.460 | 1.000 | 3.576 | 0.000 | 1.582 | 1.000 |
| SOLUSDT | bollinger | 2023 | -0.544 | -6.142 | 1.000 | 1.556 | 0.027 | 1.194 | 0.990 |
| SOLUSDT | bollinger | 2024 | -0.515 | -6.287 | 1.000 | 1.465 | 0.003 | 0.946 | 0.990 |
| SOLUSDT | bollinger | 2025 | -1.481 | -6.831 | 1.000 | 0.000 | 0.071 | 0.894 | 0.220 |
| SOLUSDT | bollinger | ALL | -0.870 | -5.142 | 1.000 | 0.000 | 0.000 | 0.708 | 0.260 |
| SOLUSDT | regression_bands | 2021 | 0.799 | -3.530 | 1.000 | 9.244 | 0.088 | 2.178 | 1.000 |
| SOLUSDT | regression_bands | 2022 | -0.355 | -4.460 | 1.000 | 1.489 | 0.000 | 1.582 | 0.930 |
| SOLUSDT | regression_bands | 2023 | -1.601 | -6.142 | 1.000 | 0.000 | 0.027 | 1.194 | 0.240 |
| SOLUSDT | regression_bands | 2024 | -1.871 | -6.287 | 1.000 | 0.000 | 0.003 | 0.946 | 0.250 |
| SOLUSDT | regression_bands | 2025 | -0.672 | -6.831 | 1.000 | 1.301 | 0.071 | 0.894 | 0.990 |
| SOLUSDT | regression_bands | ALL | -0.602 | -5.142 | 1.000 | 0.763 | 0.000 | 0.708 | 0.960 |
| SOLUSDT | return_residuals | 2021 | -1.561 | -3.530 | 0.970 | 2.632 | 0.088 | 2.178 | 0.990 |
| SOLUSDT | return_residuals | 2022 | -5.587 | -4.460 | 0.210 | 0.000 | 0.000 | 1.582 | 0.260 |
| SOLUSDT | return_residuals | 2023 | -5.841 | -6.142 | 0.590 | 0.745 | 0.027 | 1.194 | 0.850 |
| SOLUSDT | return_residuals | 2024 | -7.492 | -6.287 | 0.140 | 0.000 | 0.003 | 0.946 | 0.250 |
| SOLUSDT | return_residuals | 2025 | -7.862 | -6.831 | 0.140 | 0.298 | 0.071 | 0.894 | 0.650 |
| SOLUSDT | return_residuals | ALL | -5.114 | -5.142 | 0.540 | 0.673 | 0.000 | 0.708 | 0.930 |

## Sharpe OOS por janela de validação = lookback de seleção (dias) — custo 2+2 bps/lado

| symbol | strategy | 7 | 14 | 28 |
| --- | --- | --- | --- | --- |
| BTCUSDT | bollinger | -1.61 | -1.08 | -1.09 |
| BTCUSDT | regression_bands | -1.94 | -2.33 | -1.78 |
| BTCUSDT | return_residuals | -11.28 | -10.76 | -10.78 |
| ETHUSDT | bollinger | -1.16 | -1.17 | -1.36 |
| ETHUSDT | regression_bands | -1.60 | -1.56 | -1.91 |
| ETHUSDT | return_residuals | -8.97 | -8.76 | -8.60 |
| SOLUSDT | bollinger | -0.87 | -0.65 | -0.40 |
| SOLUSDT | regression_bands | -0.60 | -0.52 | -0.19 |
| SOLUSDT | return_residuals | -5.11 | -5.25 | -4.90 |

## CBE OOS por janela de validação (dias, bps por lado)

| symbol | strategy | 7 | 14 | 28 |
| --- | --- | --- | --- | --- |
| BTCUSDT | bollinger | 0.00 | 0.59 | 0.46 |
| BTCUSDT | regression_bands | 0.00 | 0.00 | 0.00 |
| BTCUSDT | return_residuals | 0.10 | 0.14 | 0.09 |
| ETHUSDT | bollinger | 0.00 | 0.00 | 0.00 |
| ETHUSDT | regression_bands | 0.00 | 0.00 | 0.00 |
| ETHUSDT | return_residuals | 0.00 | 0.01 | 0.10 |
| SOLUSDT | bollinger | 0.00 | 0.35 | 1.50 |
| SOLUSDT | regression_bands | 0.76 | 1.28 | 3.02 |
| SOLUSDT | return_residuals | 0.67 | 0.60 | 0.77 |
