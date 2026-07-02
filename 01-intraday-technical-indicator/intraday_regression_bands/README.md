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
- Estratégia aleatória (baseline)

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

# 5. Validação walk-forward
python scripts/05_run_walk_forward.py

# 6. Gerar gráficos, tabelas e relatório
python scripts/06_generate_results.py
```

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

- Treino: 180 dias
- Validação: 60 dias
- Teste: 60 dias
- Passo: 60 dias
- Seleção por Sharpe líquido no período de validação

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
