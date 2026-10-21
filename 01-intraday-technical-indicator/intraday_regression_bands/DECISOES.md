# Decisões do projeto (para retomar entre sessões)

Atualizado em 2026-10-21. Quem decide: Paulo. Este arquivo registra o que foi decidido, o que está em
aberto e o porquê, para uma sessão nova continuar sem reconstruir a conversa. Não tem código; os números
vêm de `snapshots/` e do relatório.

## Como retomar

1. Leia este arquivo, a memória do projeto e o relatório (Claude Docs):
   https://claude.ai/artifact/9XLhNRGJWW2akUuSGNpd59
2. Não rode nada longo, não baixe dados e não faça commit/push sem pedido explícito de Paulo.
3. Responda em português do Brasil, relate resultados negativos como negativos e diga quando um erro foi nosso.

## Estado em 2026-10-21

- Último commit na `main`: `3bb1dfb` (estudo de 1 minuto em 5 anos, scripts 13 a 16, snapshot v3). Testes: 264 passam, 4 pulados.
- Resultados já obtidos (4 bps por lado = taxa 2 + derrapagem 2):
  - 15 min, BTC/ETH/SOL 2021-2025, 27 candidatas: nenhuma lucra; maior CBE 3,02 bps (SOL, regression_bands, 28 dias).
  - 1 min, piloto BTC jan-jun/2025: falhou a regra pré-registrada (maior CBE 0,42 bps).
  - 1 min, estudo completo 2021-2025, 3 ativos, 18 candidatas, lookbacks 1/2/3: nenhuma lucra; maior CBE 2,43 bps; SPA a 4 bps p = 1,00; edge bruto significativo sem custo (p = 0,002) e já não a 1 bp (p = 0,115); concentrado no SOL e em 2021.
- Esse estudo de 1 minuto foi um desvio da regra do piloto (que mandava não escalar). O protocolo (`PREREGISTRATION_full_1m.md`) foi escrito antes dos dados, mas só commitado depois.

## Decisões tomadas

| # | Decisão | Motivo e observações |
|---|---|---|
| D1 | Remover `return_residuals` da família principal do 15 min. | Só existe em 15 min, CBE < 0,8 bps, Sharpe de −5 a −11, e torna as famílias de 15 e 1 min incomparáveis. Todos os números do 15 min mudam (SPA com 18 candidatas, figuras 1 e 3, tabelas, resumo). Refazer junto com a nova rodada, para editar o relatório uma vez só. |
| D2 | Remover a fumaça do trabalho (seção 8.1, a tabela, a linha na tabela de dados e as menções). | Decisão de Paulo: foi só checagem de bugs e o teto k = 7 foi arbitrário. **Registro fiel:** nesta conversa o teto de k = 7 veio depois da fumaça (k testado até 10; nenhuma janela escolheu k acima de 6). Paulo preferiu não descrever isso, e a fumaça (nov-dez/2025) fica fora de 2021-2024. No texto, a grade é "k de 1,5 a 7,0, fixada antes do piloto"; não chamar de "arbitrário". O registro completo fica aqui. |
| D3 | Lookbacks comuns às duas frequências: **{1, 2, 3, 7, 14, 21, 28} dias**. | Hoje 15 min usa 7/14/28 e 1 min usa 1/2/3. Nos dois casos o CBE subiu com o lookback e o melhor valor foi o maior testado, então o "NÃO" do 1 min pode estar truncado. Cada lookback é uma run separada: os resultados de 1/2/3 e 7/14/28 não mudam por existirem os outros (só o tamanho da família do SPA). |
| D4 | Período de análise único: **2021-01 a 2026-09** (dados de 2026 incluídos). | Decisão de Paulo: mesma metodologia, só mais cenários, então não há razão para separar um período confirmatório. A linha de 2026 aparece na tabela por ano. A extensão foi decidida depois de ver o resultado negativo; isso entra em uma frase nas limitações. O SPA sobre a família ampliada já corrige a busca extra. |
| D5 | Sem fechamento diário no UTC: **posição contínua**, com **funding real a cada 8 h** (00h, 08h, 16h UTC). | Decisão de Paulo: fechar à meia-noite corta operações em momentos arbitrários, e em escala grande isso pode distorcer. Conferido no código: a janela de teste roda sobre o contexto treino+validação+teste e a posição é herdada, então o modo contínuo não muda a estrutura do walk-forward. Falta implementar o termo de funding no backtest. |
| D6 | Janelas das bandas **também definidas em tempo de relógio**, com um conjunto comum {2, 4, 8, 16 h} em todas as frequências, mais valores próprios: 1 min: 30 e 60 min; 15 min: 24, 36, 48 e 60 h. | Hoje as janelas não se sobrepõem em tempo (1 min: 25-100 min; 15 min: 5-50 h), então frequência e tamanho da janela se misturam. Em barras: 1 min = 120/240/480/960 para 2/4/8/16 h; 5 min = 24/48/96/192; 15 min = 8/16/32/64. **Ver O1.** |
| D7 | Bid/ask (bookTicker) fica **fora** deste trabalho; entra como trabalho futuro. | Só há dado público de 2023-05-16 a 2024-03-30 (BTC, ETH, SOL), ~120 GB compactados. |
| D8 | O CBE (com a curva de SPA por custo) é a medida principal; o retorno líquido a 4 bps vai para apêndice. | Argumento de Paulo, que aceitei: o CBE responde "a que custo vale a pena" para qualquer investidor, e o líquido a 4 bps só vale para um custo. O CBE é uma média da amostra e não mede risco; dizer isso. |

## Em aberto (precisam de decisão)

| # | Questão | Recomendação de Claude |
|---|---|---|
| O1 | Adicionar janelas muda o espaço de seleção, então **todos** os resultados do walk-forward mudam (inclusive lookbacks 1/2/3). Manter a grade antiga como robustez ou só a nova? | Rodar a grade nova como principal e, se for barato, a antiga como linha de robustez. Dizer no texto que os números mudaram por causa do desenho, não por erro. |
| O2 | Custo usado na seleção: um só ou {0, 2, 4}? Paulo teme excesso de informação. | Um só: 4 bps (2+2), como no pré-registro. Seleção a custo 0 como robustez em apêndice, uma linha. O motor permite vários custos quase de graça (3 somas acumuladas a mais), então não é custo de máquina, é custo de clareza. Dizer no texto que a varredura de custo reavalia a mesma trajetória escolhida a 4 bps. |
| O3 | Incluir 5 minutos? | Só se o grid comum de janelas e lookbacks mostrar que a frequência importa. Aumenta a família e o tempo. |
| O4 | Manter o placebo aleatório? Paulo entendeu 50%. | Tirar do texto principal. Placebo mede só o bruto (régua para o timing do sinal); o CBE contra o custo já responde se vale a pena, e o SPA contra zero já corrige a busca. Refazer exigiria rodar de novo por causa do modo contínuo. |
| O5 | Título e estrutura (comprimir piloto, mover detalhes para apêndice, paper curto). | Título sugerido: "Bandas de reversão à média em criptomoedas: o edge bruto sobrevive aos custos e ao data snooping?". Manter o piloto em uma subseção curta (é a regra pré-registrada e a variante de 24 h). |
| O6 | Onde rodar: notebook local ou AWS. | AWS compensa (ver estimativas). |

## Notas técnicas verificadas

- **Funding (Binance USDⓈ-M).** Checado em BTCUSDT, ETHUSDT e SOLUSDT nos meses 2021-05, 2023-06 e 2026-09: intervalo de 8 h, às 00h, 08h e 16h UTC. Taxa típica de 0,3 a 1 bp por 8 h (média 0,5 bp em 2026; em mai/2021 médias de 2,4 a 2,8 bps com extremos de −54 a +17 bps no SOL). Histórico mensal público até 2026-09. Conferir o intervalo mês a mês antes de rodar (a Binance pode mudar para 4 h).
- **Dados de preço.** Candles de 1 min mensais existem até 2026-09. Cripto negocia 24 h por dia, 7 dias por semana; BTC e ETH não tinham nenhuma lacuna em 5 anos, e o SOL tinha dois buracos (26-28/02/2022 e 01-02/04/2022) preenchidos pela API REST.
- **Efeito esperado do funding.** Só paga quem segura posição no instante do funding. Em 1 min as posições duram minutos: estimativa grosseira de ~0,1 bp por dia contra ~40 bps por dia de custo de negociação (desprezível). Em 15 min as posições duram horas e o efeito é maior (ordem de 1 bp por dia contra ~12), mas long e short se compensam em parte. São estimativas; a simulação dirá.
- **Bid/ask.** Pela tick size (conta, não medida), o spread de BTC e ETH deve ficar abaixo de 0,1 bp e o de SOL em 0,1 a 0,5 bp; o custo real é quase só a taxa. Taxa base da Binance (da memória, conferir): taker 0,04% e maker 0,02% nos primeiros anos.
- **Lookback e treino.** `train_days` só aquece e não altera a escolha (19 de 19 janelas idênticas). Estudo varia `validation_days`.
- **Seleção a um custo.** O walk-forward escolhe parâmetros por Sharpe líquido a 4 bps. O CBE e a grade de custos reavaliam as mesmas posições; com outro custo de seleção, as escolhas seriam outras.

## Estimativas de tempo (extrapoladas do notebook Ryzen 7 5700U, 8 núcleos; não medidas na AWS)

- 1 min: um bloco (ativo x semestre) leva 3,8 min com 3 lookbacks (≈ 0,7 min fixos + 1,0 a 1,2 por lookback). Com 7 lookbacks ≈ 9 min x ~35 blocos (2021-01 a 2026-09) ≈ 5 a 6 h.
- 15 min: 1,5 a 2 h com 7 lookbacks. 5 min, se entrar: +1,5 h. Total local ≈ 8 a 10 h.
- AWS (c6i.32xlarge, 128 vCPU, 256 GB, blocos em paralelo): ≈ 45 a 75 min de cálculo, ~2 h com preparo, ~US$ 5 a 10. Cada worker usa ~0,8 GB em 1 min. Calibrar com um bloco antes.

## Glossário curto

- **CBE (custo de break-even):** custo por lado, em bps, que zera o retorno acumulado: 10^4 x soma do bruto / soma das unidades negociadas.
- **RC/SPA:** testes de que alguma candidata tem retorno esperado positivo depois de corrigir pela busca entre muitas (White 2000; Hansen 2005), com bootstrap estacionário (5.000 repetições, bloco médio de 10 dias).
- **Funding:** em contratos perpétuos (sem vencimento), pagamento periódico entre quem está comprado e quem está vendido, para o preço do perpétuo ficar perto do spot. Taxa positiva: compradores pagam vendedores; negativa: o contrário. Pago sobre o valor da posição, só a quem a segura no instante. Aqui: 00h, 08h e 16h UTC.
- **Pós-hoc:** decidido ou analisado depois de ver os resultados. **Pré-registro:** decisão e regra gravadas antes dos dados.
- **P1:** 2021-01-06 a 2024-12-31. **P2:** período completo 2021-2025 (descritivo). Com D4, o novo período único é 2021-2026.

## Próximos passos propostos

1. Paulo decide O1 a O6.
2. Claude escreve o pré-registro v2 (arquivo novo, antes de baixar dados): lookbacks, janelas, período 2021-2026, modo contínuo com funding, família sem `return_residuals`, regra de decisão, o que é primário.
3. Implementar o funding e o modo contínuo no backtest, com testes; baixar 2026 e o histórico de funding.
4. Rodar (local ou AWS); depois a análise geral e o diagnóstico de parâmetros fixos.
5. Atualizar o relatório (remover `return_residuals` e fumaça, números novos, nova versão do resumo) e, só se Paulo pedir, commit e push.

## Onde estão as coisas

- Código e resultados em `intraday_regression_bands/`: scripts 13 a 16 (estudo de 1 min em blocos, análise geral, parâmetros fixos, persistência), `src/full_study.py`, `src/fixed_params.py`, snapshots `2026-10-v1`, `v2-pilot-1m` e `v3-full-1m`.
- Runs completas fora do git: `results/full_1m/` e a cópia em `C:\Users\Paulo\Documents\Pessoal\backups\`.
- Dados de 5 anos em 1 min: `data/processed_full_1m/` (ignorado pelo git).
