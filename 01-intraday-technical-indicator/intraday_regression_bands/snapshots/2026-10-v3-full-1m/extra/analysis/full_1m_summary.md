# Estudo de 1 minuto completo (BTC, ETH, SOL)

Dias de teste 2021-01-06 a 2025-12-30; 18 candidatas; custo 2+2 bps por lado. P1 = 2021-01-06 a 2024-12-31 (nunca vista); P2 = período completo.

## Candidatas em P1

| symbol | strategy | validation_days | gross_pct | net_pct | cbe_bps | trades_per_day | sharpe_net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | bollinger | 1 | 116.46 | -98.50 | 0.62 | 4.27 | -3.23 |
| BTCUSDT | bollinger | 2 | 67.92 | -94.62 | 0.60 | 2.95 | -2.50 |
| BTCUSDT | bollinger | 3 | 107.20 | -87.41 | 1.04 | 2.40 | -1.88 |
| BTCUSDT | regression_bands | 1 | 67.67 | -99.23 | 0.38 | 4.63 | -4.84 |
| BTCUSDT | regression_bands | 2 | 59.38 | -94.35 | 0.56 | 2.87 | -3.08 |
| BTCUSDT | regression_bands | 3 | 76.37 | -87.02 | 0.87 | 2.24 | -2.31 |
| ETHUSDT | bollinger | 1 | -50.71 | -99.85 | 0.00 | 5.00 | -3.60 |
| ETHUSDT | bollinger | 2 | 41.14 | -96.95 | 0.36 | 3.29 | -2.20 |
| ETHUSDT | bollinger | 3 | 45.06 | -91.67 | 0.52 | 2.45 | -1.71 |
| ETHUSDT | regression_bands | 1 | 62.29 | -99.46 | 0.34 | 4.90 | -3.62 |
| ETHUSDT | regression_bands | 2 | 8.82 | -97.37 | 0.09 | 3.20 | -2.65 |
| ETHUSDT | regression_bands | 3 | 28.51 | -92.03 | 0.36 | 2.39 | -2.46 |
| SOLUSDT | bollinger | 1 | 29.92 | -99.90 | 0.15 | 6.12 | -2.71 |
| SOLUSDT | bollinger | 2 | 139.53 | -97.90 | 0.74 | 4.06 | -1.68 |
| SOLUSDT | bollinger | 3 | 336.56 | -88.85 | 1.61 | 3.15 | -1.06 |
| SOLUSDT | regression_bands | 1 | 2649.21 | -99.51 | 1.54 | 7.41 | -2.52 |
| SOLUSDT | regression_bands | 2 | 2652.58 | -88.31 | 2.43 | 4.69 | -1.11 |
| SOLUSDT | regression_bands | 3 | 1000.62 | -83.36 | 2.29 | 3.60 | -0.95 |

## Candidatas em P2

| symbol | strategy | validation_days | gross_pct | net_pct | cbe_bps | trades_per_day | sharpe_net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | bollinger | 1 | 128.41 | -99.37 | 0.56 | 4.05 | -3.34 |
| BTCUSDT | bollinger | 2 | 55.48 | -97.37 | 0.43 | 2.80 | -2.66 |
| BTCUSDT | bollinger | 3 | 92.38 | -92.83 | 0.80 | 2.26 | -2.06 |
| BTCUSDT | regression_bands | 1 | 61.78 | -99.69 | 0.31 | 4.30 | -4.90 |
| BTCUSDT | regression_bands | 2 | 38.74 | -97.07 | 0.34 | 2.65 | -3.24 |
| BTCUSDT | regression_bands | 3 | 70.34 | -92.01 | 0.70 | 2.10 | -2.46 |
| ETHUSDT | bollinger | 1 | -56.89 | -99.96 | 0.00 | 4.89 | -3.61 |
| ETHUSDT | bollinger | 2 | 17.89 | -98.94 | 0.14 | 3.23 | -2.33 |
| ETHUSDT | bollinger | 3 | 46.88 | -95.62 | 0.44 | 2.41 | -1.77 |
| ETHUSDT | regression_bands | 1 | 52.45 | -99.89 | 0.23 | 4.96 | -3.94 |
| ETHUSDT | regression_bands | 2 | 12.42 | -98.85 | 0.10 | 3.15 | -2.70 |
| ETHUSDT | regression_bands | 3 | 25.25 | -95.94 | 0.26 | 2.35 | -2.56 |
| SOLUSDT | bollinger | 1 | -12.69 | -99.99 | 0.00 | 5.97 | -2.95 |
| SOLUSDT | bollinger | 2 | 133.42 | -99.21 | 0.60 | 3.91 | -1.78 |
| SOLUSDT | bollinger | 3 | 426.88 | -93.13 | 1.53 | 2.98 | -1.09 |
| SOLUSDT | regression_bands | 1 | 2599.67 | -99.93 | 1.25 | 7.23 | -2.90 |
| SOLUSDT | regression_bands | 2 | 2264.81 | -96.87 | 1.91 | 4.55 | -1.52 |
| SOLUSDT | regression_bands | 3 | 747.64 | -94.40 | 1.70 | 3.45 | -1.30 |

## Carteira igual-pesos por ativo e ano

| symbol | period | n_candidates | n_days | gross_pct | net_pct | cbe_bps | trades_per_day | turnover_units_per_day | sharpe_net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 2021 | 6 | 360 | 88.84 | -60.26 | 1.63 | 5.41 | 10.82 |  |
| BTCUSDT | 2022 | 6 | 365 | -4.10 | -60.15 | 0.00 | 3.01 | 6.01 |  |
| BTCUSDT | 2023 | 6 | 365 | 0.42 | -43.51 | 0.03 | 1.97 | 3.94 |  |
| BTCUSDT | 2024 | 6 | 366 | -0.31 | -52.71 | 0.00 | 2.55 | 5.09 |  |
| BTCUSDT | 2025 | 6 | 364 | -4.98 | -50.37 | 0.00 | 2.23 | 4.46 |  |
| ETHUSDT | 2021 | 6 | 360 | 22.98 | -72.81 | 0.55 | 5.24 | 10.48 |  |
| ETHUSDT | 2022 | 6 | 365 | -10.32 | -70.38 | 0.00 | 3.79 | 7.59 |  |
| ETHUSDT | 2023 | 6 | 365 | 12.96 | -40.96 | 0.75 | 2.22 | 4.44 |  |
| ETHUSDT | 2024 | 6 | 366 | -7.85 | -60.83 | 0.00 | 2.92 | 5.84 |  |
| ETHUSDT | 2025 | 6 | 364 | -5.78 | -64.41 | 0.00 | 3.34 | 6.69 |  |
| SOLUSDT | 2021 | 6 | 360 | 183.75 | -51.44 | 2.36 | 6.13 | 12.26 |  |
| SOLUSDT | 2022 | 6 | 365 | 49.41 | -66.28 | 1.08 | 5.10 | 10.20 |  |
| SOLUSDT | 2023 | 6 | 365 | 40.66 | -54.17 | 1.22 | 3.84 | 7.68 |  |
| SOLUSDT | 2024 | 6 | 366 | 16.62 | -66.93 | 0.49 | 4.30 | 8.61 |  |
| SOLUSDT | 2025 | 6 | 364 | -10.51 | -72.50 | 0.00 | 4.05 | 8.10 |  |

## Fração de candidatas com retorno bruto > 0

| period | n_candidates | share_gross_positive | median_cbe_bps | max_cbe_bps |
| --- | --- | --- | --- | --- |
| 2021 | 18 | 0.89 | 1.52 | 4.15 |
| 2022 | 18 | 0.39 | 0.00 | 2.92 |
| 2023 | 18 | 0.72 | 0.42 | 2.94 |
| 2024 | 18 | 0.39 | 0.00 | 1.28 |
| 2025 | 18 | 0.22 | 0.00 | 1.12 |
| ALL | 18 | 0.89 | 0.44 | 1.91 |

## RC/SPA, família completa, P1

| cost_bps_per_side | n_candidates | n_days | best_candidate | best_mean_bps_per_day | rc_pvalue | spa_pvalue_lower | spa_pvalue_consistent | spa_pvalue_upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 18 | 1456 | SOLUSDT|regression_bands|2 | 22.769 | 0.003 | 0.002 | 0.002 | 0.002 |
| 1 | 18 | 1456 | SOLUSDT|regression_bands|2 | 13.391 | 0.083 | 0.058 | 0.115 | 0.124 |
| 2 | 18 | 1456 | SOLUSDT|regression_bands|2 | 4.013 | 0.814 | 0.363 | 0.610 | 0.910 |
| 3 | 18 | 1456 | SOLUSDT|regression_bands|3 | -5.120 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 18 | 1456 | SOLUSDT|regression_bands|3 | -12.318 | 1.000 | 1.000 | 1.000 | 1.000 |
| 6 | 18 | 1456 | BTCUSDT|regression_bands|3 | -22.979 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 18 | 1456 | BTCUSDT|regression_bands|3 | -31.938 | 1.000 | 1.000 | 1.000 | 1.000 |

## SPA consistente por família, P1

| family | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 0.167 | 0.841 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ETHUSDT | 0.574 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SOLUSDT | 0.000 | 0.044 | 0.503 | 1.000 | 1.000 | 1.000 | 1.000 |
| bollinger | 0.228 | 0.745 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| regression_bands | 0.001 | 0.067 | 0.395 | 1.000 | 1.000 | 1.000 | 1.000 |
| todas | 0.002 | 0.115 | 0.610 | 1.000 | 1.000 | 1.000 | 1.000 |

## RC/SPA, família completa, P2

| cost_bps_per_side | n_candidates | n_days | best_candidate | best_mean_bps_per_day | rc_pvalue | spa_pvalue_lower | spa_pvalue_consistent | spa_pvalue_upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 18 | 1820 | SOLUSDT|regression_bands|1 | 18.108 | 0.007 | 0.010 | 0.010 | 0.010 |
| 1 | 18 | 1820 | SOLUSDT|regression_bands|2 | 8.278 | 0.256 | 0.177 | 0.392 | 0.431 |
| 2 | 18 | 1820 | SOLUSDT|regression_bands|2 | -0.824 | 0.999 | 1.000 | 1.000 | 1.000 |
| 3 | 18 | 1820 | SOLUSDT|bollinger|3 | -8.754 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 18 | 1820 | BTCUSDT|regression_bands|3 | -13.882 | 1.000 | 1.000 | 1.000 | 1.000 |
| 6 | 18 | 1820 | BTCUSDT|regression_bands|3 | -22.287 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 18 | 1820 | BTCUSDT|regression_bands|3 | -30.691 | 1.000 | 1.000 | 1.000 | 1.000 |

## SPA consistente por família, P2

| family | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 0.239 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ETHUSDT | 0.661 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SOLUSDT | 0.004 | 0.176 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| bollinger | 0.223 | 0.759 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| regression_bands | 0.005 | 0.243 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| todas | 0.010 | 0.392 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Sensibilidade ao bloco do bootstrap

| period | mean_block_days | cost_bps_per_side | n_days | rc_pvalue | spa_pvalue_consistent |
| --- | --- | --- | --- | --- | --- |
| P1 | 1 | 0.000 | 1456 | 0.001 | 0.001 |
| P1 | 1 | 4.000 | 1456 | 1.000 | 1.000 |
| P1 | 5 | 0.000 | 1456 | 0.002 | 0.002 |
| P1 | 5 | 4.000 | 1456 | 1.000 | 1.000 |
| P1 | 10 | 0.000 | 1456 | 0.003 | 0.002 |
| P1 | 10 | 4.000 | 1456 | 1.000 | 1.000 |
| P1 | 20 | 0.000 | 1456 | 0.006 | 0.007 |
| P1 | 20 | 4.000 | 1456 | 1.000 | 1.000 |
| P1 | 40 | 0.000 | 1456 | 0.003 | 0.005 |
| P1 | 40 | 4.000 | 1456 | 1.000 | 1.000 |
| P2 | 1 | 0.000 | 1820 | 0.003 | 0.001 |
| P2 | 1 | 4.000 | 1820 | 1.000 | 1.000 |
| P2 | 5 | 0.000 | 1820 | 0.004 | 0.004 |
| P2 | 5 | 4.000 | 1820 | 1.000 | 1.000 |
| P2 | 10 | 0.000 | 1820 | 0.007 | 0.010 |
| P2 | 10 | 4.000 | 1820 | 1.000 | 1.000 |
| P2 | 20 | 0.000 | 1820 | 0.011 | 0.012 |
| P2 | 20 | 4.000 | 1820 | 1.000 | 1.000 |
| P2 | 40 | 0.000 | 1820 | 0.010 | 0.009 |
| P2 | 40 | 4.000 | 1820 | 1.000 | 1.000 |

## Decisão (regra primária pré-registrada, P1)

- CBE >= 4 bps em >= 2 lookbacks de um mesmo método: **não atendido**
- SPA consistente (família completa, 4 bps, p = 1.000) <= 0.1: **não atendido**
- **Estratégia lucrativa depois de custos: NÃO**

## Consistência com o piloto contínuo

```
{
  "period": [
    "2025-01-06",
    "2025-06-29"
  ],
  "rows_full": 3150,
  "rows_pilot": 1050,
  "rows_matched": 1050,
  "sum_ret": {
    "max_abs_diff": 0.0,
    "share_equal": 1.0
  },
  "sum_gross_ret": {
    "max_abs_diff": 0.0,
    "share_equal": 1.0
  },
  "turnover_units": {
    "max_abs_diff": 0.0,
    "share_equal": 1.0
  },
  "n_bars": {
    "max_abs_diff": 0.0,
    "share_equal": 1.0
  },
  "days_different": 0,
  "days_different_by_candidate": {}
}
```
