# Pré-registro: estudo de 1 minuto em 5 anos e 3 ativos

Escrito antes de baixar ou analisar qualquer dado de 1 minuto fora do piloto de jan-jun/2025 e da
fumaça de nov-dez/2025 do BTC. O SHA-256 deste arquivo e do `config.full_1m.yaml` é registrado em
`results/full_1m/PREREG_HASHES.txt` antes do primeiro download.

## 1. Origem e desvio da regra anterior

O piloto de 1 minuto (BTC, jan-jun/2025) tinha uma regra de decisão registrada antes dos
resultados (`src/pilot_decision.py`, commit d28f73e). O resultado foi NÃO escalar. Este estudo
**não** cumpre essa regra: foi pedido pelo usuário como extensão exploratória, com 5 anos e 3
ativos, e fica registrado como desvio. A conclusão do piloto sobre a regra antiga continua valendo.

A pergunta deste estudo é outra: com cerca de 10 vezes mais dados e dois ativos a mais, alguma
estratégia de bandas em 1 minuto, escolhida por walk-forward, é lucrativa depois de 4 bps por lado?

## 2. Desenho (idêntico ao piloto, sem ajustes depois de ver resultados)

- Ativos: BTCUSDT, ETHUSDT, SOLUSDT (Binance USD-M perpétuos), 1 minuto, 2021-01-01 a 2025-12-31.
- Estratégias: `bollinger` e `regression_bands` (log-preço); janelas 25, 50, 75, 100 barras;
  k de 1,5 a 7,0 em passos de 0,5; limiar de entrada z = 1,0; reversão confirmada; saída em z = 0.
- Walk-forward diário: treino de 2 dias (só aquece), lookback de seleção de 1, 2 e 3 dias, teste de
  1 dia, primeiro dia de teste 2021-01-06, fechamento da posição no fim do dia UTC, custo de
  2 + 2 bps por lado, seleção por Sharpe líquido na validação.
- Execução em blocos semestrais, cada um com pelo menos 6 dias de histórico antes do primeiro dia
  de teste, por limite de memória (ver `config.full_1m.yaml`). Os blocos não alteram a escolha de
  parâmetros, que só olha 1 a 3 dias para trás.
- Não serão executados neste estudo: variante de 24 h sem fechamento diário, baseline aleatório,
  `return_residuals`.
- Candidatas: 3 ativos x 2 estratégias x 3 lookbacks = 18.

## 3. Dados já vistos e janelas de análise

Já vistos em 1 minuto (BTC): 2025-01-06 a 2025-06-29 (piloto e diagnóstico de parâmetros fixos) e
2025-11-05 a 2025-12-30 (fumaça). ETH e SOL em 1 minuto nunca foram analisados.

- **P1, confirmatória:** 2021-01-06 a 2024-12-31, nunca vista em 1 minuto por nenhum dos ativos.
- **P2, completa:** 2021-01-06 a 2025-12-30 (último dia de teste possível), descritiva, pois inclui
  dias do BTC já vistos.

## 4. Regra primária de decisão (avaliada em P1)

Há evidência de estratégia lucrativa depois de custos se, simultaneamente:

1. algum método (ativo x estratégia) tem CBE >= 4 bps por lado em pelo menos 2 dos 3 lookbacks; e
2. o p-valor SPA consistente da família das 18 candidatas, com retorno líquido a 4 bps por lado,
   é <= 0,10.

Caso contrário, o resultado é negativo e será relatado como tal.

## 5. Desfechos secundários (descritivos, sem decisão)

- CBE, retorno bruto e líquido, operações por dia e Sharpe, por candidata, em P1 e em P2.
- SPA (consistente), RC e SPA inferior/superior da família das 18 e por ativo, com custos de 0, 1,
  2, 3, 4, 6 e 8 bps por lado, em P1 e em P2. Bootstrap estacionário, 5000 repetições, bloco médio
  de 10 dias, semente 0.
- CBE e retorno por ativo e por ano, e fração de candidatas com retorno bruto > 0 por ano.
- A regra do piloto (CBE >= 3 bps em >= 2 lookbacks e SPA consistente <= 0,10 a 2 bps) aplicada a
  P1 e P2, para continuidade com o piloto.
- Consistência: os dias de 2025-01-06 a 2025-06-29 do BTC no estudo completo devem reproduzir o
  piloto; diferenças serão relatadas.

## 6. Compromissos

- Nenhuma mudança de grid, janelas, k, limiar, custo ou lookback depois de ver resultados.
- Resultados relatados qualquer que seja o sinal; análises não previstas aqui serão rotuladas como
  exploratórias.
- Limitações já conhecidas: sem spread, funding, impacto nem preenchimento de ordens a limite;
  CBE não mede risco; um só provedor de dados e uma só exchange.
