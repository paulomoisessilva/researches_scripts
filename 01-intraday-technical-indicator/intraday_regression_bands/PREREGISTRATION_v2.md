# Pré-registro v2: bandas de reversão à média em 1, 5 e 15 minutos, 2021-2026

**Rascunho, ainda sem hash.** Este arquivo e os três configs (`config.v2_1m.yaml`, `config.v2_5m.yaml`,
`config.v2_15m.yaml`, a escrever depois da aprovação) têm o SHA-256 e o horário UTC gravados em
`results/v2/PREREG_HASHES.txt`, e o protocolo é commitado e enviado **antes** de baixar qualquer dado
novo e de rodar qualquer coisa. Depois disso, só mudanças registradas na seção "Desvios".

## 1. Pergunta

Algum sistema de bandas (Bollinger ou regressão, em 1, 5 ou 15 minutos, em BTC, ETH ou SOL), com
parâmetros escolhidos por walk-forward, tem retorno esperado positivo depois de custos, mesmo
corrigindo pela busca entre todos os sistemas testados (data snooping)? E, se tem, até que custo por
lado ele aguenta?

Não se fixa nenhum custo. O resultado é uma curva em função do custo e um número que a resume.

## 2. Dados

- Ativos: BTCUSDT, ETHUSDT e SOLUSDT, perpétuos USDⓈ-M da Binance, dados públicos
  (`data.binance.vision`).
- Período dos dados: 2021-01-04 (primeiro dia útil de 2021) a 2026-07-01 00:00 UTC. A barra de
  2026-07-01 00:00 fecha o último dia de teste.
- Frequências: 1, 5 e 15 minutos. O 5 e o 15 minutos saem do 1 minuto por agregação (open da primeira
  barra, máxima, mínima e close da última), uma só fonte de dados. Conferência: a agregação de 15
  minutos reproduz os arquivos nativos de 15 minutos nos períodos em que existem.
- O mercado negocia 24 h por dia, 7 dias por semana. Buracos nos arquivos são preenchidos pela API
  REST da Binance (o SOL tem dois em 1 minuto: 26-28/02/2022 e 01-02/04/2022).

## 3. Sistemas

Duas estratégias, ambas por reversão confirmada: `bollinger` (média e desvio padrão da janela) e
`regression_bands` (regressão por mínimos quadrados do log-preço na janela, banda de ±k desvios dos
resíduos). Define-se z = (preço − centro) / (banda superior − centro). Entra comprado quando z cruza
de baixo para cima −1,0 e vendido quando cruza de cima para baixo +1,0; sai quando z volta a 0.
Execução no open da barra seguinte. Sem alavancagem (posição −1, 0 ou +1).

- **k**: 1,5 a 7,0 em passos de 0,5 (12 valores), igual nas três frequências.
- **Limiar de entrada**: z = 1,0, igual nas três frequências.
- **Janelas** definidas em tempo de relógio, para que a frequência e o tamanho da janela não se
  misturem. Conjunto comum de 2, 4, 8 e 16 horas, mais valores próprios:

| Frequência | Barras por hora | Janelas comuns (barras) | Janelas próprias (barras) | Janelas | Combinações por ativo (2 estratégias x janelas x 12 k) |
|---|---|---|---|---|---|
| 1 min | 60 | 120, 240, 480, 960 | 30 e 60 min (30, 60) | 6 | 144 |
| 5 min | 12 | 24, 48, 96, 192 | 1 h e 24 h (12, 288) | 6 | 144 |
| 15 min | 4 | 8, 16, 32, 64 | 24, 36, 48 e 60 h (96, 144, 192, 240) | 8 | 192 |

## 4. Walk-forward

- Passo diário: a cada dia UTC escolhe-se um par (janela, k) para a estratégia e opera-se o dia seguinte.
- **Lookback de seleção** (dias olhados para trás): 1, 2, 3, 7, 14, 21 e 28. Cada lookback é uma rodada
  separada, e todas avaliam os mesmos dias. O treino de 2 dias só aquece a estratégia.
- **Seleção a custo 0**: maior Sharpe bruto no lookback (taxa livre de risco 0, 365 dias por ano).
  Empate dentro de 1e-9: o primeiro da grade. Assim a trajetória escolhida não depende do custo.
- **Posição contínua com estado carregado**: não há fechamento diário. Há uma só trajetória de posição
  para o período de teste inteiro. Na troca de parâmetros de um dia para o outro a posição real
  continua; a regra nova só sai quando o z dela cruza a saída e só entra numa nova reversão
  confirmada. A trajetória começa zerada na primeira barra do primeiro dia de teste. O custo é cobrado
  sobre o movimento real de posição.
- **Período de teste**: de 2021-02-03 a 2026-06-30, 1.974 dias. O primeiro dia vem 30 dias depois do
  início dos dados (2 de treino e 28 do maior lookback). Os mesmos dias valem para todos os lookbacks.
- Uma única execução contínua por ativo, frequência e lookback (sem divisão em blocos, que cortaria a
  posição no meio).
- Rodada a custo bruto (taxa 0, derrapagem 0). O retorno líquido a custo c por lado é
  `bruto − c · unidades negociadas / 10^4` (unidade = 1 de posição; virar de comprado a vendido conta 2).

## 5. Candidatas e famílias

Uma candidata é uma série diária de retorno líquido da trajetória walk-forward de (ativo, estratégia,
lookback, frequência): 3 x 2 x 7 x 3 = **126**.

- **Família primária: as 126 juntas** (tudo o que foi buscado).
- Famílias secundárias, **descritivas e sem correção entre elas**: por frequência (42), por ativo (42),
  por estratégia (63).
- **Diagnóstico de parâmetros fixos** (seção 7): família própria, bem maior, declarada como tal.

## 6. Resultado primário e o que se reporta

Teste SPA de Hansen (2005) com bootstrap estacionário (5.000 repetições, bloco médio de 10 dias,
semente 0) para H0: nenhuma candidata tem retorno líquido esperado positivo, contra zero (caixa).
Versão consistente como principal; RC de White e SPA inferior e superior também são reportados.

1. **Curva de p-valor por custo**: p do SPA global consistente para c de 0 a 8 bps por lado, em passos
   de 0,25 bps. A grade de custos serve só ao SPA; o CBE não precisa de grade (é uma conta por
   candidata).
2. **Custo-limite c\***: maior c da grade com p ≤ 0,05. Se p > 0,05 já a custo 0, o relatório diz que o
   SPA global não rejeita nem a custo zero. Verifica-se se a curva é praticamente monótona.
3. **Maior CBE** por frequência (e por ativo e por estratégia nas tabelas descritivas).
4. Sensibilidade ao bloco do bootstrap (5 e 20 dias) para a curva primária.

**Não há regra de aprovado ou reprovado.** Os resultados são lidos contra a tabela de custos reais
(seção 10). Qualquer conclusão no texto é uma leitura dessa tabela, e não um corte fixado antes.

## 7. Diagnóstico de parâmetros fixos (descritivo)

Cada combinação (estratégia, janela, k) roda com parâmetros fixos, sem walk-forward, na trajetória
contínua do período inteiro, por ativo e frequência: 3 x (144 + 144 + 192) = **1.440** candidatas.
Reporta CBE, retorno bruto, operações por dia, e a curva de p-valor do SPA dessa família, para
responder "alguém com custo c consegue lucrar com algum parâmetro fixo?". Responde à pergunta que o
walk-forward a custo 0 não responde. Não entra no teste primário.

## 8. Tabelas descritivas e apêndice

Por candidata: CBE, retorno bruto, operações por dia e Sharpe bruto. Por ano (2021, a partir de 3 de
fevereiro, a 2026, até 30 de junho): CBE, retorno bruto e fração de candidatas com retorno bruto positivo. Apêndice:
retorno líquido a 2,0 (maker regular), 4,5 (taker com BNB) e 5,0 (taker regular) bps por lado.

## 9. A métrica e como ler

- **CBE** (custo de break-even) = 10^4 · Σ bruto / Σ unidades negociadas, em bps por lado; vale 0 se
  o retorno bruto acumulado é ≤ 0. É o custo por lado que zera o retorno acumulado da amostra.
- **Leitura.** CBE abaixo do custo c significa retorno acumulado negativo a esse custo. CBE é uma
  média da amostra e **não mede risco**.
- **O que o walk-forward a custo 0 responde**: se a melhor estratégia bruta de cada dia aguenta um
  custo. A pergunta "quem tem custo c consegue lucrar?" é respondida pelo diagnóstico da seção 7.
- **O que não se pode concluir**: que nenhuma estratégia fora desta família lucra, ou que não há lucro
  com ordens a limite, preenchimento parcial ou outro horizonte.

## 10. Tabela de custos reais (para ler os resultados)

Taxas de negociação dos contratos perpétuos USDⓈ-M (USDT) da Binance, em bps por lado, lidas da
página pública de taxas da Binance em 2026-10-21 (`binance.com/en/fee/futureFee`). A tabela é a
vigente nessa data; as taxas de anos anteriores podem ter sido diferentes.

| Nível | Volume 30 dias | Maker | Maker com BNB (−10%) | Taker | Taker com BNB (−10%) |
|---|---|---|---|---|---|
| Regular | < 5 mi USD | 2,00 | 1,80 | 5,00 | 4,50 |
| VIP 1 | ≥ 5 mi | 1,80 | 1,62 | 5,00 | 4,50 |
| VIP 2 | ≥ 10 mi | 1,60 | 1,44 | 4,00 | 3,60 |
| VIP 3 | ≥ 50 mi | 1,20 | 1,08 | 3,20 | 2,88 |
| VIP 4 | ≥ 600 mi | 1,00 | 0,90 | 3,00 | 2,70 |
| VIP 5 | ≥ 1 bi | 0,80 | 0,72 | 2,70 | 2,43 |
| VIP 6 | ≥ 2,5 bi | 0,60 | 0,54 | 2,50 | 2,25 |
| VIP 7 | ≥ 5 bi | 0,40 | 0,36 | 2,20 | 1,98 |
| VIP 8 | ≥ 12,5 bi | 0,20 | 0,18 | 2,00 | 1,80 |
| VIP 9 | ≥ 25 bi | 0,00 | 0,00 | 1,70 | 1,53 |

São só taxas. Spread, derrapagem e impacto de mercado ficam de fora e somam a essas taxas; dependem do
ativo, do tamanho e da hora, e não foram medidos (o bid/ask público só existe de 2023-05-16 a
2024-03-30, e fica para trabalho futuro).

## 11. Limitações e compromissos

- Limitações conhecidas: o **funding** é ignorado (estimativa de pior caso de efeito no CBE de cerca de
  1% em 1 minuto e até 8% em 15 minutos; em 2021 houve taxas extremas); modelo de custo constante por
  lado; execução a mercado no open da barra seguinte; uma exchange e um provedor de dados; o CBE não
  mede risco; a família testada é a que está aqui e nada fora dela.
- Nenhuma mudança de grade, janelas, k, limiar, lookbacks, período ou famílias depois de ver
  resultados. Análises não previstas serão rotuladas como exploratórias.
- Todos os resultados são relatados, qualquer que seja o sinal.
- O motor é o NumPy por eventos (commit `196c847`), validado por testes de equivalência com o laço de
  referência (resultado idêntico bit a bit); a trajetória contínua é a de `src/carried_walk_forward.py`.

## 12. Ordem de execução

1. Aprovar este rascunho e escrever os configs. Gravar SHA-256 e horário UTC em
   `results/v2/PREREG_HASHES.txt`. Commit e push.
2. Baixar 2026 (janeiro a junho, mais a barra de 01/07), preparar os dados e gravar o SHA-256 dos
   arquivos processados.
3. Rodar um bloco pequeno só para medir tempo e memória, sem olhar desempenho, e decidir onde rodar.
4. Rodar tudo; depois a análise do SPA global, as famílias secundárias e o diagnóstico de parâmetros
   fixos.

## 13. Desvios

(vazio)
