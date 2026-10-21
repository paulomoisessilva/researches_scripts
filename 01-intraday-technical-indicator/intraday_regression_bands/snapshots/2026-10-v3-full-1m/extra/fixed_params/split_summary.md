# Persistência fora da amostra de parâmetros fixos (exploratório)

S = antes de 2024-01-01; T = a partir de 2024-01-01. Custo 2+2 bps por lado. Elegível = pelo menos 30 operações em S. **Pós-hoc:** a divisão foi escolhida depois de ver os resultados do período completo; não corrige a busca múltipla.

## Resumo

| symbol | combos | eligible | net_pos_S | of_those_net_pos_T | mean_net_pct_T_of_those | median_net_pct_T_of_those | spearman_net_S_vs_T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 96 | 81 | 2 | 2 | 4.67 | 4.67 | 0.99 |
| ETHUSDT | 96 | 81 | 2 | 0 | -12.93 | -12.93 | 0.98 |
| SOLUSDT | 96 | 81 | 13 | 4 | -1.44 | -4.12 | 0.93 |

## BTCUSDT: 5 melhores elegíveis em S e o que aconteceu em T

| strategy | window | k | net_pct_S | cbe_S | trades_S | net_pct_T | cbe_T | trades_T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression_bands | 75 | 7.00 | 8.22 | 9.81 | 68.00 | 4.63 | 19.09 | 15.00 |
| regression_bands | 50 | 6.00 | 5.25 | 7.37 | 76.00 | 4.70 | 16.76 | 18.00 |
| bollinger | 50 | 6.00 | -0.61 | 3.57 | 71.00 | 0.17 | 4.60 | 14.00 |
| bollinger | 75 | 7.00 | -4.21 | 0.00 | 44.00 | -1.71 | 0.00 | 12.00 |
| bollinger | 25 | 4.50 | -4.53 | 1.95 | 113.00 | -3.98 | 0.00 | 33.00 |

## ETHUSDT: 5 melhores elegíveis em S e o que aconteceu em T

| strategy | window | k | net_pct_S | cbe_S | trades_S | net_pct_T | cbe_T | trades_T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression_bands | 50 | 5.00 | 4.79 | 4.46 | 509.00 | -16.82 | 0.16 | 240.00 |
| regression_bands | 50 | 6.00 | 1.93 | 5.39 | 69.00 | -9.04 | 0.00 | 23.00 |
| regression_bands | 75 | 7.00 | -0.51 | 3.63 | 69.00 | -11.93 | 0.00 | 22.00 |
| bollinger | 50 | 6.00 | -3.46 | 1.21 | 63.00 | -18.27 | 0.00 | 24.00 |
| bollinger | 25 | 4.50 | -5.33 | 0.89 | 88.00 | -8.49 | 0.00 | 38.00 |

## SOLUSDT: 5 melhores elegíveis em S e o que aconteceu em T

| strategy | window | k | net_pct_S | cbe_S | trades_S | net_pct_T | cbe_T | trades_T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression_bands | 100 | 6.50 | 55.26 | 20.17 | 136.00 | -2.58 | 1.10 | 45.00 |
| regression_bands | 100 | 7.00 | 47.08 | 28.73 | 78.00 | -11.71 | 0.00 | 25.00 |
| regression_bands | 100 | 6.00 | 38.15 | 11.38 | 219.00 | 18.79 | 13.90 | 87.00 |
| regression_bands | 75 | 6.00 | 30.06 | 12.21 | 160.00 | 9.41 | 12.99 | 50.00 |
| bollinger | 25 | 4.50 | 26.49 | 25.76 | 54.00 | -4.12 | 0.00 | 12.00 |
