# Reality Check e SPA (bootstrap estacionário)

27 candidatas (ativo x estratégia x lookback), 1766 dias, benchmark = caixa, 5000 repetições, bloco médio 10 dias, runs com custo 2+2 bps.

## p-valor SPA (consistente) por família e custo (bps por lado)

| family | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 0.199 | 0.817 | 0.878 | 0.857 | 1.000 | 1.000 | 1.000 |
| symbol=BTCUSDT | 0.661 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| symbol=ETHUSDT | 0.823 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| symbol=SOLUSDT | 0.086 | 0.586 | 0.718 | 0.855 | 1.000 | 1.000 | 1.000 |
| strategy=bollinger | 0.770 | 0.934 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| strategy=regression_bands | 0.310 | 0.510 | 0.520 | 0.724 | 1.000 | 1.000 | 1.000 |
| strategy=return_residuals | 0.065 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=7 | 0.137 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=14 | 0.193 | 0.914 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=28 | 0.085 | 0.543 | 0.626 | 0.699 | 1.000 | 1.000 | 1.000 |

## p-valor Reality Check por família e custo

| family | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 0.021 | 0.763 | 0.960 | 0.999 | 1.000 | 1.000 | 1.000 |
| symbol=BTCUSDT | 0.572 | 0.981 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| symbol=ETHUSDT | 0.819 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| symbol=SOLUSDT | 0.021 | 0.610 | 0.820 | 0.945 | 0.990 | 1.000 | 1.000 |
| strategy=bollinger | 0.694 | 0.923 | 0.992 | 1.000 | 1.000 | 1.000 | 1.000 |
| strategy=regression_bands | 0.191 | 0.436 | 0.755 | 0.948 | 0.997 | 1.000 | 1.000 |
| strategy=return_residuals | 0.021 | 0.997 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=7 | 0.020 | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=14 | 0.036 | 0.959 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=28 | 0.012 | 0.560 | 0.863 | 0.989 | 1.000 | 1.000 | 1.000 |

## Melhor candidata (família `all`) por custo

| cost_bps_per_side | best_candidate | best_mean_bps_per_day | best_annualized_log_return_pct | spa_pvalue_consistent |
| --- | --- | --- | --- | --- |
| 0 | SOLUSDT|return_residuals|val28 | 25.170 | 91.872 | 0.199 |
| 1 | SOLUSDT|regression_bands|val28 | 6.925 | 25.278 | 0.817 |
| 2 | SOLUSDT|regression_bands|val28 | 3.503 | 12.786 | 0.878 |
| 3 | SOLUSDT|regression_bands|val28 | 0.081 | 0.294 | 0.857 |
| 4 | SOLUSDT|regression_bands|val28 | -3.342 | -12.198 | 1.000 |
| 6 | SOLUSDT|regression_bands|val28 | -10.187 | -37.181 | 1.000 |
| 8 | SOLUSDT|regression_bands|val28 | -17.032 | -62.165 | 1.000 |

## Sensibilidade ao bloco (família `all`, custo 4 bps)

| mean_block_days | cost_bps_per_side | rc_pvalue | spa_pvalue_consistent | spa_pvalue_upper |
| --- | --- | --- | --- | --- |
| 1.000 | 4.000 | 1.000 | 1.000 | 1.000 |
| 5.000 | 4.000 | 1.000 | 1.000 | 1.000 |
| 10.000 | 4.000 | 1.000 | 1.000 | 1.000 |
| 20.000 | 4.000 | 1.000 | 1.000 | 1.000 |
| 40.000 | 4.000 | 1.000 | 1.000 | 1.000 |
