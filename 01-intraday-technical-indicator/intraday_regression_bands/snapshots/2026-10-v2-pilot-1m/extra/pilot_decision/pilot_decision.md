# Decisão do piloto

Regra (fixada antes): CBE >= 3 bps em >= 2 lookbacks de um mesmo método **e** SPA consistente <= 0.1 a 2 bps/lado.

12 candidatas, 175 dias (2025-01-06 a 2025-06-29), bootstrap 5000 x bloco 10.

## CBE total por método e lookback (bps por lado)

| variant | strategy | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| config.pilot_1m | bollinger | 0.42 | 0.00 | 0.00 |
| config.pilot_1m | regression_bands | 0.00 | 0.00 | 0.00 |
| config.pilot_1m_noeod | bollinger | 0.18 | 0.00 | 0.00 |
| config.pilot_1m_noeod | regression_bands | 0.00 | 0.00 | 0.00 |

## SPA da família por custo

| cost_bps_per_side | n_candidates | n_days | best_candidate | best_mean_bps_per_day | rc_pvalue | spa_pvalue_consistent |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 12 | 175 | config.pilot_1m|bollinger|1 | 2.869 | 0.711 | 0.696 |
| 1 | 12 | 175 | config.pilot_1m|bollinger|1 | -3.999 | 0.976 | 1.000 |
| 2 | 12 | 175 | config.pilot_1m_noeod|regression_bands|3 | -8.490 | 0.999 | 1.000 |
| 3 | 12 | 175 | config.pilot_1m_noeod|regression_bands|3 | -11.942 | 1.000 | 1.000 |
| 4 | 12 | 175 | config.pilot_1m_noeod|regression_bands|3 | -15.393 | 1.000 | 1.000 |
| 6 | 12 | 175 | config.pilot_1m_noeod|regression_bands|3 | -22.296 | 1.000 | 1.000 |
| 8 | 12 | 175 | config.pilot_1m_noeod|regression_bands|3 | -29.199 | 1.000 | 1.000 |

## Resultado

- Critério de CBE: **não atendido**
- Critério de SPA (p = 1.000 a 2 bps): **não atendido**
- **Escalar para o período maior: NÃO**
