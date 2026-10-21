# Tabelas do paper (walk-forward OOS)

## Sharpe OOS por mês — validação 2d, custo 2+2 bps/lado

| strategy | 2025-01 | 2025-02 | 2025-03 | 2025-04 | 2025-05 | 2025-06 | ALL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bollinger | 2.12 | -8.38 | -8.27 | -1.77 | -4.77 | -6.19 | -4.60 |
| regression_bands | -2.40 | -6.83 | -9.96 | -7.39 | -3.72 | -8.90 | -6.19 |

## Custo de break-even por mês (bps por lado que zeram o retorno)

| strategy | 2025-01 | 2025-02 | 2025-03 | 2025-04 | 2025-05 | 2025-06 | ALL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bollinger | 6.60 | 0.00 | 0.00 | 2.33 | 0.00 | 0.00 | 0.00 |
| regression_bands | 2.27 | 0.00 | 0.00 | 0.00 | 1.50 | 0.00 | 0.00 |

## Sharpe OOS por janela de validação = lookback de seleção (dias) — custo 2+2 bps/lado

| strategy | 1 | 2 | 3 |
| --- | --- | --- | --- |
| bollinger | -4.40 | -4.60 | -4.56 |
| regression_bands | -6.49 | -6.19 | -4.21 |

## CBE OOS por janela de validação (dias, bps por lado)

| strategy | 1 | 2 | 3 |
| --- | --- | --- | --- |
| bollinger | 0.42 | 0.00 | 0.00 |
| regression_bands | 0.00 | 0.00 | 0.00 |
