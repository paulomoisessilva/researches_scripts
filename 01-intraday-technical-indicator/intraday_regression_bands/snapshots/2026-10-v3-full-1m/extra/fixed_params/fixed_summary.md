# Diagnóstico de parâmetros fixos, 1 minuto, 5 anos (descritivo)

3 ativos, 96 combinações por ativo, custo 2+2 bps por lado, fechamento diário UTC. P1 = até 2024-12-31; P2 = período completo.

> **Ressalva de data snooping:** todas as combinações foram avaliadas no MESMO período. A melhor delas é otimista por construção. O RC/SPA abaixo corrige isso para a família inteira; nada aqui foi confirmado fora da amostra.

## BTCUSDT: 10 combinações com maior retorno bruto (P2)

| strategy | window | k | gross_pct | net_pct | cbe_bps | trades_per_day |
| --- | --- | --- | --- | --- | --- | --- |
| bollinger | 25 | 2.00 | 84941.86 | -100.00 | 0.42 | 44.14 |
| bollinger | 25 | 1.50 | 16741.79 | -100.00 | 0.23 | 61.73 |
| bollinger | 50 | 2.50 | 10125.64 | -100.00 | 0.73 | 17.42 |
| regression_bands | 50 | 2.00 | 9705.97 | -100.00 | 0.30 | 42.02 |
| bollinger | 50 | 2.00 | 9086.63 | -100.00 | 0.47 | 26.40 |
| bollinger | 25 | 2.50 | 7979.51 | -100.00 | 0.49 | 24.39 |
| regression_bands | 50 | 1.50 | 6563.91 | -100.00 | 0.19 | 60.64 |
| regression_bands | 25 | 1.50 | 5771.41 | -100.00 | 0.11 | 99.13 |
| regression_bands | 25 | 2.00 | 5421.39 | -100.00 | 0.18 | 60.55 |
| regression_bands | 50 | 2.50 | 4608.71 | -100.00 | 0.43 | 24.34 |

## BTCUSDT: fração de combinações com retorno bruto > 0, por ano

| period | share_gross_positive | max_cbe_bps | median_cbe_bps |
| --- | --- | --- | --- |
| 2021 | 0.60 | 104.86 | 0.74 |
| 2022 | 0.35 | 2.20 | 0.00 |
| 2023 | 0.54 | 29.46 | 0.06 |
| 2024 | 0.58 | 22.84 | 0.08 |
| 2025 | 0.57 | 23.35 | 0.15 |

## ETHUSDT: 10 combinações com maior retorno bruto (P2)

| strategy | window | k | gross_pct | net_pct | cbe_bps | trades_per_day |
| --- | --- | --- | --- | --- | --- | --- |
| bollinger | 25 | 2.00 | 5014.88 | -100.00 | 0.25 | 43.89 |
| bollinger | 25 | 1.50 | 1860.03 | -100.00 | 0.13 | 61.59 |
| regression_bands | 25 | 2.00 | 1359.41 | -100.00 | 0.12 | 60.59 |
| regression_bands | 25 | 1.50 | 1330.06 | -100.00 | 0.07 | 99.65 |
| regression_bands | 50 | 2.00 | 1111.46 | -100.00 | 0.16 | 42.02 |
| regression_bands | 75 | 2.00 | 1032.01 | -100.00 | 0.21 | 31.55 |
| regression_bands | 75 | 2.50 | 780.89 | -100.00 | 0.31 | 19.58 |
| bollinger | 25 | 2.50 | 667.91 | -100.00 | 0.23 | 24.33 |
| regression_bands | 50 | 1.50 | 580.32 | -100.00 | 0.09 | 60.66 |
| regression_bands | 100 | 2.50 | 547.61 | -100.00 | 0.31 | 16.37 |

## ETHUSDT: fração de combinações com retorno bruto > 0, por ano

| period | share_gross_positive | max_cbe_bps | median_cbe_bps |
| --- | --- | --- | --- |
| 2021 | 0.60 | 91.70 | 0.34 |
| 2022 | 0.17 | 18.93 | 0.00 |
| 2023 | 0.78 | 25.97 | 0.34 |
| 2024 | 0.19 | 9.21 | 0.00 |
| 2025 | 0.28 | 2.10 | 0.00 |

## SOLUSDT: 10 combinações com maior retorno bruto (P2)

| strategy | window | k | gross_pct | net_pct | cbe_bps | trades_per_day |
| --- | --- | --- | --- | --- | --- | --- |
| regression_bands | 25 | 1.50 | 14658.12 | -100.00 | 0.14 | 99.43 |
| regression_bands | 25 | 2.00 | 13294.63 | -100.00 | 0.22 | 61.10 |
| bollinger | 25 | 2.00 | 9993.74 | -100.00 | 0.29 | 43.49 |
| bollinger | 25 | 1.50 | 4537.75 | -100.00 | 0.17 | 60.95 |
| regression_bands | 75 | 2.00 | 3504.81 | -100.00 | 0.31 | 31.56 |
| bollinger | 25 | 2.50 | 2617.86 | -100.00 | 0.39 | 23.34 |
| regression_bands | 75 | 2.50 | 1483.93 | -100.00 | 0.39 | 19.51 |
| regression_bands | 50 | 2.00 | 1385.12 | -100.00 | 0.18 | 42.26 |
| regression_bands | 25 | 2.50 | 1233.92 | -100.00 | 0.28 | 25.05 |
| regression_bands | 50 | 2.50 | 1214.70 | -100.00 | 0.29 | 24.15 |

## SOLUSDT: fração de combinações com retorno bruto > 0, por ano

| period | share_gross_positive | max_cbe_bps | median_cbe_bps |
| --- | --- | --- | --- |
| 2021 | 0.64 | 77.74 | 0.51 |
| 2022 | 0.42 | 88.83 | 0.00 |
| 2023 | 0.77 | 151.79 | 0.72 |
| 2024 | 0.61 | 59.87 | 0.26 |
| 2025 | 0.38 | 16.95 | 0.00 |

## SPA consistente por família, P1

| family | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 0.001 | 0.592 | 0.607 | 0.625 | 0.630 | 0.616 | 0.614 |
| ETHUSDT | 0.158 | 0.323 | 0.353 | 0.362 | 0.448 | 0.713 | 0.832 |
| SOLUSDT | 0.096 | 0.110 | 0.128 | 0.157 | 0.174 | 0.258 | 0.344 |
| todas | 0.002 | 0.251 | 0.248 | 0.272 | 0.299 | 0.418 | 0.522 |

## RC por família, P1

| family | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 0.000 | 0.987 | 0.991 | 0.991 | 0.993 | 0.994 | 0.995 |
| ETHUSDT | 0.030 | 0.976 | 0.988 | 0.996 | 1.000 | 1.000 | 1.000 |
| SOLUSDT | 0.298 | 0.907 | 0.914 | 0.926 | 0.931 | 0.945 | 0.956 |
| todas | 0.050 | 0.996 | 0.997 | 0.998 | 0.998 | 0.999 | 1.000 |

## SPA consistente por família, P2

| family | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 0.000 | 0.526 | 0.456 | 0.507 | 0.599 | 0.607 | 0.609 |
| ETHUSDT | 0.087 | 0.342 | 0.287 | 0.332 | 0.415 | 0.638 | 0.820 |
| SOLUSDT | 0.189 | 0.424 | 0.441 | 0.475 | 0.505 | 0.620 | 0.695 |
| todas | 0.000 | 0.708 | 0.667 | 0.699 | 0.739 | 0.838 | 0.887 |

## RC por família, P2

| family | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 0.000 | 0.988 | 0.989 | 0.991 | 0.992 | 0.995 | 0.996 |
| ETHUSDT | 0.015 | 0.990 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| SOLUSDT | 0.118 | 0.924 | 0.935 | 0.944 | 0.954 | 0.974 | 0.988 |
| todas | 0.019 | 0.997 | 0.998 | 0.999 | 0.999 | 0.999 | 1.000 |
