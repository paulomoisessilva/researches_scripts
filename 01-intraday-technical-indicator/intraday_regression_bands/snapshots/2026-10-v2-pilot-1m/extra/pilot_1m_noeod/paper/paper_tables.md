# Tabelas do paper (walk-forward OOS)

## Sharpe OOS por mês — validação 2d, custo 2+2 bps/lado

| strategy | 2025-01 | 2025-02 | 2025-03 | 2025-04 | 2025-05 | 2025-06 | ALL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bollinger | 1.66 | -8.36 | -8.20 | -1.69 | -4.70 | -6.09 | -4.58 |
| regression_bands | -2.35 | -6.83 | -9.96 | -7.37 | -3.90 | -8.90 | -6.19 |

## Custo de break-even por mês (bps por lado que zeram o retorno)

| strategy | 2025-01 | 2025-02 | 2025-03 | 2025-04 | 2025-05 | 2025-06 | ALL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bollinger | 5.88 | 0.00 | 0.00 | 2.36 | 0.00 | 0.00 | 0.00 |
| regression_bands | 2.31 | 0.00 | 0.00 | 0.00 | 1.55 | 0.00 | 0.00 |

## Sharpe OOS por janela de validação = lookback de seleção (dias) — custo 2+2 bps/lado

| strategy | 1 | 2 | 3 |
| --- | --- | --- | --- |
| bollinger | -4.66 | -4.58 | -4.82 |
| regression_bands | -6.83 | -6.19 | -4.21 |

## CBE OOS por janela de validação (dias, bps por lado)

| strategy | 1 | 2 | 3 |
| --- | --- | --- | --- |
| bollinger | 0.18 | 0.00 | 0.00 |
| regression_bands | 0.00 | 0.00 | 0.00 |
