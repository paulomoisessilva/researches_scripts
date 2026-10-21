# Intraday Regression Bands

Projeto de pesquisa acadêmica para backtesting intradiário em **Binance USD-M Futures** usando bandas de regressão, indicadores técnicos clássicos e validação walk-forward.

---

## Objetivo

Comparar estratégias intradiárias baseadas em:

- Bandas de Bollinger clássicas
- Bandas ajustadas por regressão rolling
- Bandas de resíduos de retorno
- RSI (Índice de Força Relativa)
- Estocástico
- Indicador de confluência técnica composto
- Estratégia aleatória multi-seed (placebo)

Sobre os ativos `BTCUSDT` e `ETHUSDT` nas frequências `1m`, `5m` e `15m`.

---

## Estrutura

```
intraday_regression_bands/
  config.yaml           # Parâmetros globais
  requirements.txt      # Dependências Python
  pyproject.toml        # Metadados do projeto

  data/
    raw/binance/        # Dados brutos baixados (ZIP/CSV)
    processed/          # Parquet limpo por símbolo/intervalo
    features/           # Parquet com indicadores calculados
    backtests/          # Equity curves e trades por estratégia

  results/
    tables/             # CSV com métricas e resumos
    figures/            # PNG com gráficos
    reports/            # Relatório MD e HTML

  src/                  # Módulos reutilizáveis
  scripts/              # Scripts executáveis em ordem
  notebooks/            # Análise exploratória e revisão
  tests/                # Testes de integridade
```

---

## Instalação

```bash
pip install -r requirements.txt
```

Ou com pip editável:

```bash
pip install -e ".[dev]"
```

---

## Execução completa

Execute os scripts na ordem:

```bash
# 1. Baixar dados históricos da Binance
python scripts/01_download_data.py

# 2. Preparar e limpar os dados
python scripts/02_prepare_data.py

# 3. Calcular indicadores técnicos
python scripts/03_run_features.py

# 4. Rodar backtests por estratégia e custo
python scripts/04_run_backtests.py

# 5. Validação walk-forward (cada run grava em results/walk_forward/<run_id>/)
python scripts/05_run_walk_forward.py --config config.paper.yaml          # run primária
python scripts/05_run_walk_forward.py --config config.paper.yaml --study  # janelas x custos

# 6. Gerar gráficos, tabelas e relatório do backtest fixo (script 04)
python scripts/06_generate_results.py

# 7. Tabelas do paper (por ano, por lookback de validação) a partir das runs do passo 5
python scripts/07_paper_tables.py --config config.paper.yaml

# 8. (opcional) equivalência entre o motor de seleção fatiado e o de passagem única
python scripts/08_check_engine_equivalence.py --config config.paper.yaml --n-windows 40

# 9. Reality Check (White) e SPA (Hansen) sobre as candidatas ativo x estratégia x lookback
python scripts/09_reality_check.py --config config.paper.yaml
```

### Reality Check / SPA (`scripts/09_reality_check.py`)

Testa se **alguma** das candidatas (ativo x estratégia x `validation_days`) tem retorno líquido esperado
positivo depois de corrigir por data snooping. Usa os retornos diários das runs, bootstrap estacionário
(blocos geométricos, média de 10 dias por padrão) e benchmark = caixa. Como as posições não dependem do custo,
o retorno líquido é `bruto - turnover x custo`, então o teste é repetido numa grade de custos (0 = sinal bruto).
Saídas em `results/paper/`: `reality_check.csv`, `reality_check_block_sensitivity.csv` e `reality_check.md`.
O p-valor a reportar é o **SPA consistente** (`spa_pvalue_consistent`); o RC é a versão sem studentizar.

### Runs do walk-forward (`run_id`)

Cada combinação janela de treino × cenário de custo é uma *run* com pasta própria em
`results/walk_forward/<run_id>/` (`walk_forward_results.csv`, `best_parameters_by_window.csv`,
`walk_forward_oos_summary.csv`, `walk_forward_equity.parquet`, `run_meta.json`). O `run_id` padrão
codifica janelas, custo e período, ex.: `tr21_va7_te1_st1_fee2_slip2_20210101-20251231_oos20210301`.

- `--costs 2:2 5:5` — cenários fee:slippage (bps por lado); sem isso usa o primeiro de `study.cost_scenarios`.
  O custo da run é o da **seleção** de parâmetros; a sensibilidade a custo é lida no custo de break-even
  (CBE, `script 07`), então o estudo do paper roda um único custo (2+2 bps).
- `--validation-days 7 14 28` — uma run por janela de validação (o **lookback de seleção**). `--study` usa `study.*` do config (3 runs no `config.paper.yaml`). `--train-days` existe, mas o treino só aquece a estratégia e não altera os resultados.
- `--fresh` apaga a run antes; sem ele a execução retoma/mescla. Trocar o grid ou as janelas numa pasta
  existente é recusado (impressão digital do config em `run_meta.json`).
- `walk_forward.selection_engine`: `sliced` (padrão; refaz o backtest de treino+validação de cada janela) ou
  `single_pass` (cada combo roda uma vez na série inteira e a pontuação de validação sai de somas acumuladas;
  cerca de 40× menos CPU). A equivalência entre os dois é medida por
  `scripts/08_check_engine_equivalence.py`. Empates de pontuação (< 1e-9) vão para o primeiro combo do grid.
- **Baseline aleatório (`random`)**: roda em paralelo por seed e grava em blocos (placebo de 100 seeds x 1766 janelas: ~5 min com 14 workers, contra ~23 min antes). Com `save_random_equity: false`, o resultado é guardado em `results/cache/random/` e **reaproveitado** por qualquer run com o mesmo `oos_start`, custo, seeds e dados (o placebo não depende do lookback); `walk_forward.random_cache: false` desliga.
- Só a janela de **validação** entra na pontuação de seleção; `train_days` apenas aquece a estratégia (não altera a escolha). Por isso o estudo de robustez varia `validation_days`.
- `walk_forward.oos_start` fixa o primeiro dia de teste, então todas as janelas de treino avaliam o **mesmo** período OOS.
- Sem custo configurado o padrão é fee=0/slippage=0 (resultado bruto); o script avisa.

### Flags disponíveis

| Flag | Descrição |
|------|-----------|
| `--symbols BTCUSDT` | Filtrar símbolos |
| `--intervals 5m 15m` | Filtrar intervalos |
| `--force` | Forçar redownload/reprocessamento |
| `--config outro.yaml` | Usar config alternativa |

---

## Configuração

Edite `config.yaml` para alterar:

- Período (`start_date`, `end_date`)
- Ativos (`symbols`)
- Frequências (`intervals`)
- Capital inicial e custos de transação
- Janelas dos indicadores
- Parâmetros do walk-forward

---

## Metodologia

### Execução causal

- Sinal calculado no candle `t`
- Execução no `open` do candle `t+1`
- Posição: `-1` (short), `0` (neutro), `+1` (long)

### Custos

```
strategy_return_t = position_{t-1} × log_return_t − fee_t − slippage_t
```

Custos testados: 0, 2, 5, 10 bps por lado.

### Walk-forward

Config do paper (`config.paper.yaml`):

- Treino: 21 dias (só aquece a estratégia)
- Validação: 7 dias (estudo de robustez: 7, 14 e 28 = lookback de seleção)
- Teste: 1 dia, passo de 1 dia, primeiro dia de teste em `oos_start` (2021-03-01)
- Seleção por Sharpe líquido (`net_sharpe`) apenas no período de validação; posição fechada no fim do dia UTC
- Sharpe e retorno anualizados com 365 dias/ano (cripto 24/7)
- Tabelas por ano: `scripts/07_paper_tables.py` reagrega os momentos por janela (`sum_ret`, `sum_sq_ret`, ...)
- `turnover_units` conta o |Δposição| cobrado em cada barra, inclusive a entrada na 1ª barra do dia (`trade_units` no backtest); `positions.diff()` na fatia perdia essa unidade e inflava o CBE por dia em ~1 bps.

### Baseline aleatório multi-seed (placebo)

O walk-forward inclui uma estratégia `random` usada como **placebo**: trading direcional aleatório (long/short 50/50) com taxa fixa de eventos por candle (`random_baseline.trade_prob`). Não é otimizada na validação.

Com `random_baseline.n_seeds` (ex.: 100), o pipeline gera uma **distribuição** de resultados OOS — um caminho por seed — e resume percentis em `walk_forward_random_seed_distribution.csv`. Isso permite verificar se o desempenho das estratégias reais excede o que seria esperado por acaso direcional.

Configuração:

```yaml
random_baseline:
  n_seeds: 100
  seed_base: 0
  trade_prob: 0.05
```

Comparações do paper: Bollinger, Regression Bands, Return Residual Bands, buy-and-hold BTC e placebo aleatório multi-seed.

---

## Métricas reportadas

Retorno total, retorno anualizado, volatilidade, Sharpe, Sortino, drawdown máximo, Calmar, hit rate, profit factor, número de trades, tempo médio de holding, exposição, custo de break-even.

---

## Testes

```bash
pytest tests/ -v
```

Cobrem:

- Ausência de look-ahead bias
- Validade dos indicadores (RSI ∈ [0,100], bandas superiores > inferiores)
- Integridade do backtest (custos reduzem retorno, posição ∈ {-1,0,1})

---

## Fontes de dados

- **Primária**: [data.binance.vision](https://data.binance.vision) — arquivos históricos de klines USD-M Futures
- **Fallback**: [Binance REST API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data) — endpoint `/fapi/v1/klines`

---

## Restrições

- Não usa yfinance
- Não executa trade no candle do sinal
- Não otimiza parâmetros no período de teste
- Indicadores técnicos não são chamados de probabilidade
- Resultados in-sample e out-of-sample são separados
