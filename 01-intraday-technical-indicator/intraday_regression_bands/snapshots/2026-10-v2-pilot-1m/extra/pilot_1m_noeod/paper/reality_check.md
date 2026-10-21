# Reality Check e SPA (bootstrap estacionário)

6 candidatas (ativo x estratégia x lookback), 175 dias, benchmark = caixa, 5000 repetições, bloco médio 10 dias, runs com custo 2+2 bps.

## p-valor SPA (consistente) por família e custo (bps por lado)

| family | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 0.806 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| strategy=bollinger | 0.695 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| strategy=regression_bands | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=1 | 0.614 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## p-valor Reality Check por família e custo

| family | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 0.809 | 0.985 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| strategy=bollinger | 0.698 | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| strategy=regression_bands | 0.790 | 0.925 | 0.981 | 0.997 | 1.000 | 1.000 | 1.000 |
| val=1 | 0.625 | 0.934 | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=2 | 0.900 | 0.979 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=3 | 0.805 | 0.952 | 0.993 | 0.999 | 1.000 | 1.000 | 1.000 |

## Melhor candidata (família `all`) por custo

| cost_bps_per_side | best_candidate | best_mean_bps_per_day | best_annualized_log_return_pct | spa_pvalue_consistent |
| --- | --- | --- | --- | --- |
| 0 | BTCUSDT|bollinger|val1 | 1.218 | 4.444 | 0.806 |
| 1 | BTCUSDT|regression_bands|val3 | -5.039 | -18.392 | 1.000 |
| 2 | BTCUSDT|regression_bands|val3 | -8.490 | -30.990 | 1.000 |
| 3 | BTCUSDT|regression_bands|val3 | -11.942 | -43.588 | 1.000 |
| 4 | BTCUSDT|regression_bands|val3 | -15.393 | -56.185 | 1.000 |
| 6 | BTCUSDT|regression_bands|val3 | -22.296 | -81.381 | 1.000 |
| 8 | BTCUSDT|regression_bands|val3 | -29.199 | -106.576 | 1.000 |

## Sensibilidade ao bloco (família `all`, custo 4 bps)

| mean_block_days | cost_bps_per_side | rc_pvalue | spa_pvalue_consistent | spa_pvalue_upper |
| --- | --- | --- | --- | --- |
| 1.000 | 4.000 | 1.000 | 1.000 | 1.000 |
| 5.000 | 4.000 | 1.000 | 1.000 | 1.000 |
| 10.000 | 4.000 | 1.000 | 1.000 | 1.000 |
| 20.000 | 4.000 | 1.000 | 1.000 | 1.000 |
| 40.000 | 4.000 | 1.000 | 1.000 | 1.000 |
