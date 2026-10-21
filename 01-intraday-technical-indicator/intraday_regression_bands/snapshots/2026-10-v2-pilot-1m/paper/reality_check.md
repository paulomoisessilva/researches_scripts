# Reality Check e SPA (bootstrap estacionário)

6 candidatas (ativo x estratégia x lookback), 175 dias, benchmark = caixa, 5000 repetições, bloco médio 10 dias, runs com custo 2+2 bps.

## p-valor SPA (consistente) por família e custo (bps por lado)

| family | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 0.671 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| strategy=bollinger | 0.545 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| strategy=regression_bands | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=1 | 0.466 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## p-valor Reality Check por família e custo

| family | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 0.689 | 0.971 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| strategy=bollinger | 0.557 | 0.943 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| strategy=regression_bands | 0.802 | 0.926 | 0.981 | 0.997 | 1.000 | 1.000 | 1.000 |
| val=1 | 0.490 | 0.896 | 0.992 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=2 | 0.895 | 0.977 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| val=3 | 0.804 | 0.950 | 0.992 | 0.999 | 1.000 | 1.000 | 1.000 |

## Melhor candidata (família `all`) por custo

| cost_bps_per_side | best_candidate | best_mean_bps_per_day | best_annualized_log_return_pct | spa_pvalue_consistent |
| --- | --- | --- | --- | --- |
| 0 | BTCUSDT|bollinger|val1 | 2.869 | 10.473 | 0.671 |
| 1 | BTCUSDT|bollinger|val1 | -3.999 | -14.597 | 1.000 |
| 2 | BTCUSDT|regression_bands|val3 | -8.630 | -31.500 | 1.000 |
| 3 | BTCUSDT|regression_bands|val3 | -12.093 | -44.139 | 1.000 |
| 4 | BTCUSDT|regression_bands|val3 | -15.556 | -56.778 | 1.000 |
| 6 | BTCUSDT|regression_bands|val3 | -22.481 | -82.057 | 1.000 |
| 8 | BTCUSDT|regression_bands|val3 | -29.407 | -107.336 | 1.000 |

## Sensibilidade ao bloco (família `all`, custo 4 bps)

| mean_block_days | cost_bps_per_side | rc_pvalue | spa_pvalue_consistent | spa_pvalue_upper |
| --- | --- | --- | --- | --- |
| 1.000 | 4.000 | 1.000 | 1.000 | 1.000 |
| 5.000 | 4.000 | 1.000 | 1.000 | 1.000 |
| 10.000 | 4.000 | 1.000 | 1.000 | 1.000 |
| 20.000 | 4.000 | 1.000 | 1.000 | 1.000 |
| 40.000 | 4.000 | 1.000 | 1.000 | 1.000 |
