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
| D2 | Remover a fumaça do trabalho (seção 8.1, a tabela, a linha na tabela de dados e as menções). | A fumaça (56 dias de BTC, nov-dez/2025) foi checagem de bugs. O teto k = 7 foi **escolha arbitrária de Paulo** (pensou em 5, subiu para 10, achou demais e voltou para 7); o miniteste de 56 dias não é base de decisão. Registro dos fatos: nele nenhuma janela escolheu k acima de 6. Como o período de análise agora é 2021-2026, a fumaça está dentro dele, mas o efeito é desprezível (o teto só afeta combinações quase inativas). No texto: "k de 1,5 a 7,0, em passos de 0,5", sem justificativa. |
| D3 | Lookbacks comuns às duas frequências: **{1, 2, 3, 7, 14, 21, 28} dias**. | Hoje 15 min usa 7/14/28 e 1 min usa 1/2/3. Nos dois casos o CBE subiu com o lookback e o melhor valor foi o maior testado, então o "NÃO" do 1 min pode estar truncado. Cada lookback é uma run separada: os resultados de 1/2/3 e 7/14/28 não mudam por existirem os outros (só o tamanho da família do SPA). |
| D4 | Período de análise único: **2021-01 a 2026-09** (dados de 2026 incluídos). | Mesma metodologia, só mais cenários, então não há razão para separar um período confirmatório. A linha de 2026 aparece na tabela por ano. Motivo da extensão, nas palavras de Paulo: a análise anterior estava confusa e as frequências não eram comparáveis (lookbacks diferentes), e ele decidiu corrigir o desenho; não foi o resultado negativo. Para registro: os resultados com lookbacks 1 a 3 já eram conhecidos quando a extensão foi proposta, e todos os cenários entram na família do SPA. |
| D5 | Sem fechamento diário no UTC: **posição contínua**. O **funding é ignorado** e citado como limitação (D14). | Decisão de Paulo: fechar à meia-noite corta operações em momentos arbitrários, e em escala grande isso pode distorcer. Conferido no código: a janela de teste roda sobre o contexto treino+validação+teste e a posição é herdada, então o modo contínuo não muda a estrutura do walk-forward; basta `close_position_end_of_day: false`. |
| D14 | **Ignorar o funding** (sem termo no retorno, sem pseudo-operações) e citar como limitação. | Decisão de Paulo, por simplicidade. Para o texto: o efeito estimado no CBE é de ~1% no 1 min e até ~8% no 15 min (conta: custo esperado de funding por dia ÷ retorno bruto por dia; supõe 1 bp por 8 h sempre contra nós, é um limite superior; ex.: SOL 15 min, CBE 3,0 → ~2,8; SOL 1 min, CBE 2,4 → ~2,4). Em 2021 houve taxas extremas (−54 a +17 bps). A taxa real é pública (arquivos mensais) e pode ser incluída em trabalho futuro. |
| D6 | Janelas das bandas **também definidas em tempo de relógio**, com um conjunto comum {2, 4, 8, 16 h} em todas as frequências, mais valores próprios: 1 min: 30 e 60 min; 15 min: 24, 36, 48 e 60 h. | Hoje as janelas não se sobrepõem em tempo (1 min: 25-100 min; 15 min: 5-50 h), então frequência e tamanho da janela se misturam. Em barras: 1 min = 120/240/480/960 para 2/4/8/16 h; 5 min = 24/48/96/192; 15 min = 8/16/32/64. Extras do 5 min: 1 h e 24 h; grade de k e limiar iguais nas três frequências (D19). |
| D7 | Bid/ask (bookTicker) fica **fora** deste trabalho; entra como trabalho futuro. | Só há dado público de 2023-05-16 a 2024-03-30 (BTC, ETH, SOL), ~120 GB compactados. |
| D8 | O CBE (com a curva de SPA por custo) é a medida principal; o retorno líquido a 4 bps vai para apêndice. | Argumento de Paulo, que aceitei: o CBE responde "a que custo vale a pena" para qualquer investidor, e o líquido a 4 bps só vale para um custo. O CBE é uma média da amostra e não mede risco; dizer isso. |
| D9 | Remover o **placebo aleatório** do trabalho (seções 5.5, 6.5, 9.2 e o que citar). | Decisão de Paulo. Ele mede só o bruto; o CBE contra o custo e o SPA contra zero respondem o que importa. |
| D10 | **Seleção de parâmetros a custo 0** (Sharpe bruto, taxa livre de risco 0), em vez de 4 bps. | Preferência de Paulo: uma única trajetória, independente do custo; o CBE é um número só e a grade de custos do SPA vale exatamente. **Consequências a escrever com clareza:** (i) a seleção bruta escolhe as combinações mais ativas, de CBE baixo (0,3 a 0,7 bps no diagnóstico de parâmetros fixos), então os CBEs do walk-forward devem ficar menores que os atuais (2,4 bps, escolhidos a 4 bps); (ii) o walk-forward passa a responder "a melhor estratégia bruta aguenta custo?" e não "um investidor com custo c consegue lucrar?"; para isso serve o diagnóstico de parâmetros fixos (independe do custo, SPA sobre todas as combinações). Os dois juntos sustentam a conclusão. |
| D11 | Não usar "piloto" nem "fumaça" no trabalho: há um estudo, e o foco é o que importa. | Decisão de Paulo. O histórico (piloto, fumaça, primeira rodada de 5 anos) fica neste arquivo e no git (`d28f73e`, `3bb1dfb`). |
| D12 | **Só a grade nova de janelas** (em tempo de relógio, D6); a grade antiga não é rodada de novo. | Decisão de Paulo. Os números antigos ficam no git (`3bb1dfb`). No texto, os números novos não são "correções": vêm de outro desenho. |
| D13 | O trabalho deve **explicar a métrica, como lê-la e quais conclusões ela permite**. | Decisão de Paulo, com Claude. Subseção curta: (i) definição do CBE; (ii) leitura: CBE abaixo do custo c significa retorno acumulado negativo a esse custo, e CBE é uma média da amostra que não mede risco; (iii) como o walk-forward a custo 0 responde "a melhor estratégia bruta aguenta o custo?"; (iv) a pergunta "quem tem custo c consegue lucrar?" é respondida pelo diagnóstico de parâmetros fixos com SPA; (v) o que não se pode concluir (p. ex., que nenhuma estratégia fora da família lucra, ou que não há lucro com ordens a limite). |
| D15 | **Incluir 5 minutos**: três frequências (1, 5 e 15 min). | Decisão de Paulo. Dá um eixo de frequência com três pontos. Aumenta a família e o tempo (≈ +15%). Grade definida em D19. |
| D16 | **Posição contínua com estado carregado**: uma única trajetória de posição, sem salto na troca de parâmetros. | Aprovado por Paulo (já era a ideia dele). Achado de Claude no código: hoje cada dia de teste roda com os parâmetros escolhidos e começa na posição que esses parâmetros teriam sozinhos (`_run_one_window` em `walk_forward.py`). Com fechamento diário isso não aparece (todo dia começa zerado); em modo contínuo, se ontem os parâmetros A terminaram comprados e hoje os B começam zerados, a posição saltaria sem custo e sem unidades negociadas, o que tende a inflar o CBE. Regra nova: ao trocar de parâmetros a posição real continua; a regra nova só manda sair quando z cruzar a saída e só entra com nova reversão confirmada. Precisa de teste próprio. Os resultados antigos (fechamento diário) e o diagnóstico de parâmetros fixos (não há troca) não são afetados. |
| D17 | **Teste primário: SPA global**, com todas as candidatas (3 ativos x 2 estratégias x 7 lookbacks x 3 frequências = 126), reportado como **curva de p-valor por custo** e resumido pelo **custo-limite** (maior custo por lado, em bps, em que o SPA consistente ainda rejeita a 5%). | Aprovado por Paulo. Não assume nenhum custo. Conferir na grade de custos se a curva é praticamente monótona. Famílias por frequência, por ativo e por estratégia: **secundárias e descritivas**. O diagnóstico de parâmetros fixos tem **família própria**, bem maior, declarada como tal. Lógica: para afirmar que existe edge, a família global (tudo o que foi buscado) é a proteção correta; famílias menores têm mais poder e mostram onde estaria a evidência. Muda a recomendação anterior de Claude (uma família por frequência como principal). |
| D18 | **Sem teste de controle positivo** (edge plantado nos dados para medir o poder do SPA). | Decisão de Paulo: informação demais. Fica como ideia para trabalho futuro. |
| D19 | **Grade do 5 min e grade comum**: 5 min com janelas {2, 4, 8, 16 h} (24, 48, 96, 192 barras) mais extras de **1 h e 24 h** (12 e 288 barras). **Mesma grade de k (1,5 a 7,0, passo 0,5; 12 valores) e mesmo limiar de entrada (z = 1,0) nas três frequências.** | Aprovado por Paulo. Só a barra muda entre as frequências. As rodadas antigas não eram comparáveis também nisto: o 15 min usou k de 1,5 a 3,5 e limiares 0,7 a 1,0 (`config.paper.yaml`), o 1 min usou k até 7,0 e limiar 1,0. Janelas por frequência (barras): 1 min = 30, 60, 120, 240, 480, 960; 5 min = 12, 24, 48, 96, 192, 288; 15 min = 8, 16, 32, 64, 96, 144, 192, 240 (D6). Combinações por ativo (2 estratégias x janelas x 12 k): 144, 144 e 192. |
| D20 | **Sem regra de aprovado/reprovado** no pré-registro v2: valem resultados reportados (curva do SPA global, custo-limite, maior CBE por frequência, além das tabelas descritivas por ativo e estratégia), lidos contra uma **tabela de custos reais** (taxas taker e maker). | Aprovado por Paulo. A regra antiga usava 4 bps, que era um custo assumido. A tabela de custos precisa ser conferida na fonte (O5). |
| D21 | **Otimização do motor: NumPy vetorizado por eventos**, sem numba e sem C++. Numba fica como plano B. Pré-cálculos que valem estão em "Notas técnicas". | Medido em 2026-10-21 (1 processo, 300 mil barras de 1 min): laço do sinal 2,49 → 0,05 µs/barra (0,16 com muitas operações); laço do backtest 2,02 → 0,03 µs (0,08 com fechamento diário); saídas idênticas ao motor atual em 300 casos aleatórios (NaN, saltos, empates nos limiares) e em dados realistas. Lógica: operações são raras (0,3 a 3 por 100 barras), então o estado só muda em eventos; acham-se entradas e saídas com `searchsorted` e a posição sai de uma soma acumulada. Paulo apontou o NumPy; Claude havia dito que o laço de estado não vetorizava, e estava errado. Falta implementar no repositório com testes. |
| D22 | **Onde rodar:** notebook primeiro; se precisar de AWS, **uma máquina EC2** (sobe, roda, copia, desliga), não ECS/Fargate. | Preferência de Paulo (execução única, mais simples). Preço por núcleo é parecido nos dois (Fargate ≈ US$ 0,049 por vCPU-hora com 2 GB; c6i.32xlarge ≈ US$ 0,0425; valores de sites de terceiros, conferir na AWS). Uma lista de versões fixas ou `Dockerfile` dá reprodutibilidade sem custo. Decisão final depois de medir um bloco real. |

## Em aberto (precisam de decisão)

| # | Questão | Recomendação de Claude |
|---|---|---|
| O4 | Título e estrutura. | Adiado: Paulo prefere discutir depois. Título sugerido por Claude: "Bandas de reversão à média em criptomoedas: o edge bruto sobrevive aos custos e ao data snooping?". |
| O5 | Tabela de custos reais (taxas taker e maker por nível da Binance, desconto, derrapagem típica) para ler o CBE e a curva do SPA (D20). | Conferir na fonte antes do pré-registro v2; os valores de memória de Claude não servem. É dado de entrada, não precisa de rodada. |

Resolvidas em 2026-10-21: O1 (SPA global primário, D17), O2 (NumPy por eventos, D21) e O3 (notebook primeiro; EC2 se for preciso, D22).

## Notas técnicas verificadas

- **Funding (Binance USDⓈ-M).** Checado em BTCUSDT, ETHUSDT e SOLUSDT nos meses 2021-05, 2023-06 e 2026-09: intervalo de 8 h, às 00h, 08h e 16h UTC. Taxa típica de 0,3 a 1 bp por 8 h (média 0,5 bp em 2026; em mai/2021 médias de 2,4 a 2,8 bps com extremos de −54 a +17 bps no SOL). Histórico mensal público até 2026-09 (útil se o funding entrar em trabalho futuro; a Binance pode mudar o intervalo para 4 h em alguns contratos).
- **Dados de preço.** Candles de 1 min mensais existem até 2026-09. Cripto negocia 24 h por dia, 7 dias por semana; BTC e ETH não tinham nenhuma lacuna em 5 anos, e o SOL tinha dois buracos (26-28/02/2022 e 01-02/04/2022) preenchidos pela API REST.
- **O que o funding muda nos retornos.** É um fluxo de caixa: no instante do funding, quem está com posição paga ou recebe `posição x taxa`, e isso entra como um termo no retorno daquela barra. Não muda sinais, posições nem número de operações, só o retorno líquido (e o "bruto após funding"). Estimativa pelas runs atuais, supondo taxa de 1 bp por 8 h sempre contra nós (limite superior; long e short se compensam em parte): 1 min, exposição de 4 a 8% do tempo, ≈ 0,1 a 0,2 bp por dia contra 18 a 27 bps por dia de custo (0,4 a 1,3%); 15 min, exposição de ~29% do tempo, ≈ 0,9 bp por dia contra 11 a 14 bps (6 a 8%). Em 2021 houve taxas extremas (−54 a +17 bps), que podem pesar em dias específicos. Não deve mudar a conclusão; entra por realismo.
- **Bid/ask.** Pela tick size (conta, não medida), o spread de BTC e ETH deve ficar abaixo de 0,1 bp e o de SOL em 0,1 a 0,5 bp; o custo real é quase só a taxa. Taxa base da Binance (da memória, conferir): taker 0,04% e maker 0,02% nos primeiros anos.
- **Lookback e treino.** `train_days` só aquece e não altera a escolha (19 de 19 janelas idênticas). Estudo varia `validation_days`.
- **Seleção a um custo.** Nas rodadas já feitas, o walk-forward escolheu por Sharpe líquido a 4 bps; o CBE e a grade de custos reavaliam as mesmas posições, e com outro custo de seleção as escolhas seriam outras. Pela decisão D10 a próxima rodada seleciona a custo 0, o que elimina essa dependência (e muda os números).
- **Desempenho do motor atual** (medido em 2026-10-21; 1 processo, 300 mil barras de 1 min, dados sintéticos com o código do projeto). Uma combinação custa 4,5 µs/barra, 99% nos dois laços de Python: sinal 2,46, backtest 2,02, z 0,01, pontuação 0,02. Os 17 µs das estimativas anteriores vieram de 8 processos em paralelo no notebook.
- **NumPy por eventos (D21), medido.** Sinal: 2,49 → 0,05 µs/barra (0,16 com muitas operações). Backtest: 2,02 → 0,03 µs (0,08 com fechamento diário). Saídas idênticas ao motor atual em 300 casos aleatórios e em dados realistas. O protótipo e as medições estão em `scripts/proto_numpy_events.py` (não commitado; roda em segundos, só com o código do projeto).
- **List comprehension não é C++.** Mede 39 ns por elemento contra 50 do laço com `append` (1,3x); `numpy.multiply.outer` faz 1,7 ns. O ganho do NumPy vem de o laço rodar em código compilado sobre arrays contíguos; a comprehension continua no interpretador.
- **Pré-cálculo que vale.** (i) Estatísticas diárias por combinação (soma de r, soma de r², n, bruto, unidades, operações): a pontuação de qualquer lookback vira soma de L dias (exato, a menos de arredondamento) e o diagnóstico de parâmetros fixos sai de graça. (ii) Guardar só `mid` e `std` por janela e calcular `upper = mid + k·std` na hora, com a mesma ordem de operações (hoje há uma coluna por k, até 12x mais memória). (iii) Bootstrap do SPA: as médias reamostradas saem de (matriz de contagens) @ (retornos diários) e, como o líquido é linear no custo, de duas multiplicações (bruto e unidades) para todos os custos; hoje `spa_grid` reaproveita só os índices e refaz a coleta `d[idx]` a cada custo.
- **Pré-cálculo que não vale.** Regressão rolante por somas acumuladas: custo constante por barra (0,12 µs contra 1,1 µs com janela 100 e 10 µs com janela 960 na versão atual), mas o erro relativo do desvio chega a 6e-6 (cancelamento numérico) e as features custam cerca de 1 min por ativo no 1 min (extrapolado); não compensa. Programação dinâmica no sentido de Bellman não tem encaixe: a regra é fixa e não há otimização de caminho; o análogo aqui são as somas acumuladas.

## Estimativas de tempo e custo (extrapoladas; não medidas na AWS)

**Base medida no notebook** (Ryzen 7 5700U, 8 núcleos, 8 processos): cada combinação (janela x k x limiar) custa ≈ 17 µs por barra por passagem, no 1 min e no 15 min. Um bloco de 1 min (ativo x semestre, 96 combinações, 3 lookbacks) levou 3,8 min.

**Premissas (ainda por fechar no pré-registro v2):** 3 ativos; 3 frequências (1, 5 e 15 min); 2 estratégias; 7 lookbacks; 2021-01 a 2026-09 (≈ 3,0 milhões de barras por ativo em 1 min); grade assumida de 72 combinações por estratégia em 1 e 5 min (6 janelas x 12 valores de k) e 160 em 15 min (8 janelas x 5 k x 4 limiares).

| Cenário | Núcleo-horas | Notebook (≈ 8 núcleo-h por hora) | AWS c6i.32xlarge (≈ 100 processos úteis) |
|---|---|---|---|
| Motor como está (cada lookback recalcula as combinações) | ≈ 70 a 80 | ≈ 9 a 10 h | ≈ 30 a 60 min de cálculo |
| Motor otimizado (O2: cada combinação uma vez, reaproveitada por todos os lookbacks e pelo diagnóstico de parâmetros fixos) | ≈ 10 a 12 | ≈ 1,5 h | ≈ 10 a 20 min |
| NumPy por eventos (D21), extrapolado de medições por etapa | < 1 | minutos a ≈ 30 min (sem contar download) | desnecessária |

- **AWS, custo:** ≈ US$ 5,4 por hora sob demanda (valor de memória; conferir; spot é mais barato). Com ~0,5 a 1 h de preparo (criar a máquina, copiar o código, baixar e preparar os dados), o total fica em ≈ US$ 4 a 12.
- **Memória:** ≈ 1 GB por processo em 1 min; 100 processos ≈ 100 GB, cabe nos 256 GB da c6i.32xlarge. O notebook tem 23 GB (≈ 13 livres): 8 processos cabem, um bloco por vez.
- **Sensibilidades:** dobrar a grade dobra o tempo; o 5 min soma ≈ 15%; selecionar a mais de um custo é quase de graça no motor; o RC/SPA com famílias grandes leva alguns minutos.
- **Calibração:** rodar um bloco de 1 min (ativo x semestre) com a grade nova e 7 lookbacks no notebook e extrapolar, antes de decidir onde rodar.
- **Leitura:** com o motor otimizado, o estudo inteiro cabe no notebook em ≈ 1,5 a 2 h e a AWS provavelmente não compensa; sem a otimização, a AWS evita quase um dia de máquina ocupada.
- **Atualização depois de medir (2026-10-21).** Com o NumPy por eventos (D21) o custo por combinação cai de 4,5 para ≈ 0,1 a 0,25 µs/barra (soma das etapas medidas separadamente, não ponta a ponta). No 1 min (3 ativos, 144 combinações por ativo, ≈ 3,0 milhões de barras por ativo): ≈ 2 a 6 min por passagem em um núcleo; recalculando os 7 lookbacks sem compartilhar, ≈ 15 a 40 min; compartilhando as estatísticas diárias, uma passagem. O 5 min e o 15 min somam ≈ 25% (D19: 144, 144 e 192 combinações por ativo). Ordem de grandeza: menos de 1 núcleo-hora, contra 70 a 80 do motor atual, então o estudo cabe no notebook. Ainda não medido e pode virar gargalo: features (≈ 1 min por ativo no 1 min, extrapolado), leitura dos arquivos, SPA e download. Calibrar com um bloco real.

## Glossário curto

- **CBE (custo de break-even):** custo por lado, em bps, que zera o retorno acumulado: 10^4 x soma do bruto / soma das unidades negociadas.
- **RC/SPA:** testes de que alguma candidata tem retorno esperado positivo depois de corrigir pela busca entre muitas (White 2000; Hansen 2005), com bootstrap estacionário (5.000 repetições, bloco médio de 10 dias).
- **Funding:** em contratos perpétuos (sem vencimento), pagamento periódico entre quem está comprado e quem está vendido, para o preço do perpétuo ficar perto do spot. Taxa positiva: compradores pagam vendedores; negativa: o contrário. Pago sobre o valor da posição, só a quem a segura no instante. Aqui: 00h, 08h e 16h UTC.
- **Pós-hoc:** decidido ou analisado depois de ver os resultados. **Pré-registro:** decisão e regra gravadas antes dos dados.
- **P1:** 2021-01-06 a 2024-12-31. **P2:** período completo 2021-2025 (descritivo). Com D4, o novo período único é 2021-2026.
- **Custo-limite do SPA:** maior custo por lado (bps) em que o SPA global (versão consistente) ainda rejeita a 5%; resume a curva de p-valor por custo (D17). Não confundir com o CBE, que é por estratégia.

## Próximos passos propostos

1. Claude implementa o motor NumPy por eventos (D21), a trajetória contínua com estado carregado (D16) e as estatísticas diárias por combinação, com testes de equivalência contra o motor atual e um teste próprio da regra de troca de parâmetros. Nada longo é rodado.
2. Paulo confere a tabela de custos reais (O5); título e estrutura (O4) ficam para depois.
3. Claude escreve o pré-registro v2 (arquivo novo, antes de baixar dados): 3 frequências (1, 5, 15 min), lookbacks {1, 2, 3, 7, 14, 21, 28}, janelas e grade de D6 e D19, período 2021-2026, seleção a custo 0, modo contínuo com estado carregado (sem funding), SPA global primário com curva e custo-limite (D17), sem `return_residuals` e sem placebo, resultados reportados no lugar de regra de aprovação (D20).
4. Ajustar as configs, baixar os dados de 1, 5 e 15 min até 2026-09 (só com OK de Paulo) e conferir com um bloco pequeno antes de rodar tudo.
5. Medir um bloco real e decidir onde rodar (D22); depois rodar, com análise geral e diagnóstico de parâmetros fixos.
6. Atualizar o relatório (remover `return_residuals`, placebo, fumaça e a palavra "piloto", números novos, subseção da métrica e limitação do funding) e, só se Paulo pedir, commit e push.

## Onde estão as coisas

- Código e resultados em `intraday_regression_bands/`: scripts 13 a 16 (estudo de 1 min em blocos, análise geral, parâmetros fixos, persistência), `src/full_study.py`, `src/fixed_params.py`, snapshots `2026-10-v1`, `v2-pilot-1m` e `v3-full-1m`.
- Runs completas fora do git: `results/full_1m/` e a cópia em `C:\Users\Paulo\Documents\Pessoal\backups\`.
- Dados de 5 anos em 1 min: `data/processed_full_1m/` (ignorado pelo git).
