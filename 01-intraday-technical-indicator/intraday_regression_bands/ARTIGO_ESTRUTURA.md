# Decisões de escrita e estrutura do artigo

Registro das decisões sobre como o artigo (relatório no Claude Docs, https://claude.ai/artifact/9XLhNRGJWW2akUuSGNpd59) é construído. As decisões de estudo (dados, métodos, testes) ficam em `DECISOES.md`; aqui ficam só as de redação e estrutura.

Método de trabalho: revisão etapa por etapa, uma parte do artigo de cada vez (começando pela Introdução). O texto só é aplicado ao documento depois que o Paulo aprova a versão proposta. Resumo e Conclusão são revisados só depois da Introdução.

## Decididas

| # | Data | Decisão | Origem |
|---|------|---------|--------|
| E1 | 2026-10-27 | Linguagem calibrada nas conclusões: "com as metodologias empregadas, não se encontrou evidência de ..." em vez de "responde que não". Fatos da amostra (nenhuma candidata lucra depois das taxas; CBE máximo 1,18 bps) continuam diretos; o que não se afirma é a generalização para qualquer banda de reversão. Evitar também "distingue do acaso" para p = 0,046 (Reality Check 0,105): dizer que a correção por busca múltipla rejeita a hipótese nula só a custo zero. | Comentário na Introdução |
| E2 | 2026-10-27 | Sem motivação pessoal no artigo (a observação do autor sobre janelas de 50 candles e o contrato de dólar). A motivação é a de mercado (24 h, custos baixos) e a da literatura (data snooping e custos). Sem autoelogio ("inspeção estatística rigorosa"). | Comentário na Introdução |
| E3 | 2026-10-27 | Texto corrido, em prosa, no corpo do artigo: sem listas numeradas ou de marcadores e sem rótulos em negrito ("Perguntas.", "Contribuições.", "Organização.") onde um artigo teria parágrafos. Tabelas continuam para números. | Comentário na Introdução |
| E4 | 2026-10-27 | Parágrafo das perguntas abre com "O estudo se propõe a responder cinco perguntas". | Resposta ao texto proposto |
| E5 | 2026-10-27 | Parágrafo das contribuições sem contagem ("As contribuições são cinco" é repetição desnecessária). | Resposta ao texto proposto |
| E6 | 2026-10-27 | Reprodutibilidade (SHA-256 gravado antes da execução, hashes dos dados, testes automatizados) não entra na Introdução como contribuição. | Resposta ao texto proposto |
| E7 | 2026-10-27 | As perguntas de pesquisa são abertas ("em que medida", "como", "quanto", "a que custo"), nunca de sim ou não: a resposta é um achado, não um veredito. | Resposta ao texto proposto |
| E8 | 2026-10-27 | Estrutura de referência: Introdução, Revisão da literatura, Metodologia, Resultados e discussão, Conclusão, com subseções dentro de cada uma. O corpo atual tem 14 seções de primeiro nível (uma por análise) e deve ser reorganizado nesse esquema; o mapeamento exato (onde ficam Dados e Discussão) o mapeamento adotado está em E9. | Pergunta sobre a estrutura geral |
| E9 | 2026-10-27 | Estrutura adotada, cinco seções de primeiro nível: Introdução; Revisão da literatura; Metodologia (Dados como primeira subseção, depois estratégias e grade, execução e custos, walk-forward, métricas e CBE, SPA, parâmetros fixos, análises descritivas); Resultados e discussão (resultado primário, ganhos e perdas, calendário e regime, regras de saída, parâmetros fixos e persistência e, por último, "Discussão e limitações"); Conclusão. Resumo sem número; a Introdução é a seção 1. | Aprovação do mapeamento |
| E10 | 2026-10-27 | Material de engenharia fora do artigo: saem a seção 11 (Validação, 458 testes) e o Apêndice C (Reprodução), porque o Paulo não costuma ver isso em artigos. Motor de seleção (5.4) e Protocolo (5.9) viram, no máximo, uma frase cada na Metodologia. Ficam os Apêndices A (configuração dos experimentos) e B (tabelas complementares). | Pergunta sobre a necessidade de apêndice |
| E11 | 2026-10-27 | A Introdução não traz resultados numéricos (1,18 bps, p = 0,046 e afins são conclusão e ficam no Resumo e na Conclusão). Ordem dos parágrafos: motivação e literatura; objetivo seguido das perguntas abertas (a frase de E4 vem depois do objetivo); contribuições; organização. O motor de seleção não entra como contribuição (coerente com E10). | Comentários sobre os parágrafos 1 e 2 |
| E12 | 2026-10-27 | Na prosa, evitar dois pontos e enumerações longas com ponto e vírgula; preferir frases fluidas. A frase de organização fica numa só frase corrida ("Além desta introdução, o artigo revisa a literatura na Seção 2, ... e conclui na Seção 5"). | Comentário sobre a frase de organização |
| E13 | 2026-10-28 | Siglas e termos: no primeiro uso, nome completo seguido da sigla entre parênteses, por exemplo Bitcoin (BTC), teste de capacidade preditiva superior (SPA, de Superior Predictive Ability), custo de break-even (CBE), pontos-base (bps). Vale no Resumo e de novo no corpo do texto. | Comentário sobre "BTC, ETH e SOL" |
| E14 | 2026-10-28 | Estilo de redação inspirado no STE (Simplified Technical English): frases curtas, uma ideia por frase, palavras simples. Em trechos complexos, explicar em passos; fora deles, não deixar o texto mecânico. | Comentário sobre a frase das contribuições |
| E15 | 2026-10-28 | O artigo é de backtest, não de engenharia: o motor de seleção não entra nem como contribuição nem como frase na Metodologia. Substitui a ressalva de E10 sobre uma frase para Motor e Protocolo. | Resposta sobre o motor de seleção |
| E16 | 2026-10-28 | Aplicado no documento: a Introdução (quatro parágrafos: motivação e literatura; objetivo e perguntas; contribuições; organização) e a reorganização em cinco seções (E9), com referências cruzadas renumeradas. Saíram o antigo 5.4 (Motor de seleção), o antigo 5.9 (Protocolo), a antiga seção 11 (Validação) e o Apêndice C (Reprodução). Backup em `scratchpad/backup_blocos_removidos_rev113.xml` e no histórico do documento. | Pedido "faça a reorganização dos tópicos" |
| E17 | 2026-10-28 | A motivação da Introdução parte de Wei et al. (2024): a literatura de criptomoedas está dividida, o estudo mais completo com correção de busca múltipla não acha valor nas regras tradicionais, e os estudos citados usam retornos diários. Entraram na Revisão e nas Referências (agora 39): Grobys et al. (2020), Wei et al. (2024) e Hsu, Taylor e Wang (2016). | "inclusive dá para usar ele como base da motivação" |
| E18 | 2026-10-28 | Resumo e Conclusão calibrados: "com as metodologias empregadas, não se encontrou evidência de ...", rejeição "da hipótese nula" só a custo zero, sem dois pontos, siglas por extenso no primeiro uso. A Conclusão virou texto corrido (quatro parágrafos mais os próximos passos em frases). | "calibra" |
| E19 | 2026-10-28 | Rótulos em negrito e listas do corpo convertidos em texto corrido (3.1 Dados, 3.2 regras, 3.3 premissas, 4.2, 4.6.3 e Conclusão). Ficam o negrito de termos definidos (CBE, teste primário, custo-limite) e os rótulos B.1 a B.3 dos apêndices. | "pode alterar" |
| E20 | 2026-10-28 | Artigos recebidos em PDF são lidos por completo e resumidos em `LITERATURA_LIDA.md` (desenho, números, limites, uso e cuidados), para consulta futura. | "criar um documento de referências" |
| E21 | 2026-10-28 | A Introdução apresenta o assunto, o objetivo e as perguntas. Não traz conclusões nem afirmações sobre o trabalho e não tem trecho de lacuna. A lacuna da literatura (dados diários contra intradiários, regras de tendência contra reversão à média, regressão contra Bollinger) vai para a Revisão, na seção 2.7, escrita como pergunta em aberto. Trabalhos futuros vêm depois da Conclusão. | Comentário sobre a lacuna na Introdução |
| E22 | 2026-10-28 | Aplicado E21 no documento (rev 191). Saíram as duas frases finais do parágrafo 1 da Introdução; a seção 2.7 passou a descrever o que foi testado (regras de tendência em dados diários, osciladores como uma classe entre cinco) e as três perguntas em aberto. Contribuições: o que a Introdução propõe levantar são o objetivo e as perguntas; contribuições realizadas vão para a Discussão ou a Conclusão, conforme o conteúdo. O parágrafo de contribuições saiu da Introdução e entrou na Conclusão, depois da interpretação prática e antes dos trabalhos futuros, no passado ("apresentou", "mediu", "descreveu"). | Respostas ao texto proposto |
| E23 | 2026-10-28 | Estilo de frases curtas aplicado à Revisão (2.1 a 2.6), à Metodologia (3.1 a 3.8) e aos Resultados e discussão (4.1 a 4.6), antes da leitura do Paulo (rev 196). Números e afirmações ficam iguais. Cortes de processo: a frase sobre o corpus de 19 trabalhos "fichados na fase 1" (Revisão) e a frase do SHA-256 dos arquivos (Dados), coerentes com E6 e E10. Siglas definidas no primeiro uso (OHLCV, UTC, GLM, ATR, MAE, MFE, LSTM, TWAP, VWAP). Referências cruzadas conferidas. A seção 4.6.1 passou a comparar com Hudson e Urquhart pelos números do texto lido (custos de break-even de 7,88 a 147,56 bps por classe, contra cerca de 50 bps no mercado à vista) em vez do resumo. | "vale aplicar isso antes de eu ler tudo" |
| E24 | 2026-10-28 | Literatura: os 4 PDFs lidos foram copiados para `papers/phase-2/` e entraram como F20 a F23 no `Fichamento phase 1.docx` (contador de fontes 23, mapa temático, seção 6, lista de referências). Das 39 referências do relatório, 18 têm PDF em `papers/phase-1/` (o arquivo que se chamava "Optimal Execution of Portfolio Transactions.pdf" era na verdade o Papanicolaou et al. 2023 e foi renomeado em 2026-10-28), 4 estão em `phase-2` e 17 não têm PDF (DOIs passados ao Paulo). | Pedido sobre artigos não lidos e fichamento |
| E25 | 2026-10-28 | Os 16 PDFs de `papers/phase-3` (Bollinger 2001 fica de fora, livro) foram lidos por agentes e viraram as fichas F24 a F39, em `papers/Fichamento phase 3.txt` (passagem para docx pendente), com a conferência do relatório em `papers/Conferencia do relatorio phase 3.txt`. Nada no relatório foi contradito. Cinco ajustes sugeridos, ainda não aplicados: Poterba e Summers (não rejeitam o passeio aleatório em níveis convencionais), "médias móveis e momentum" na Introdução (Detzel e Grobys testam só médias móveis), Cont (a inferência sobre o bootstrap i.i.d. é do relatório), Lo 2002 (a anualização por raiz de 365 supõe retornos independentes), Park e Irwin (redação de data snooping e custos). He et al. confirma o funding de 0,01% por 8 horas na Binance (cerca de 1 bp), compatível com a hipótese de 1 bp por 8 horas do relatório; o PDF é a v7 de setembro de 2026. | Pedido de análise dos artigos novos |

## Em aberto

* **Frase de organização da Introdução**: já em prosa corrida (E12); falta só decidir se fica, pois com cinco seções ela é opcional.
* **Renumeração**: ao reorganizar o corpo, o Resumo perde o número e as referências cruzadas mudam; fazer seção por seção, na ordem de revisão.
* **Revisão do Paulo, ponto por ponto**: Introdução, Resumo, Conclusão e as conversões de E19 já estão aplicados e aguardam a leitura dele.
* **Trabalhos futuros**: seguem como o último parágrafo da Conclusão, sem seção própria. A resposta do Paulo ("Não, para continuar igual isso") foi lida como manter a estrutura de cinco seções; se ele quis uma seção 6, é só separar.
* **Hudson e Urquhart (2021)**: lido por inteiro em 2026-10-28. A descrição do relatório está de acordo com o texto, e a leitura de Wei et al. sobre ele não (ver `LITERATURA_LIDA.md`).
* **Estilo (E13, E14, E23)**: aplicado ao documento todo. Falta só o ajuste fino que o Paulo pedir ao ler. Ficaram sem mudança os blocos de tabelas, fórmulas, legendas e os Apêndices.
* **Diagrama do walk-forward**: ainda falta a versão v2.
* **Periódico ou conferência-alvo**: define o limite de páginas, o modelo e se há "Contribuições" explícitas.


## Pendências da literatura recente (anotado em 01/11/2026)

Fichas F40 a F48 estão em `papers/Fichamento phase 4.txt` (papers/ é ignorado pelo Git; ainda em txt, não em docx). PDFs em `papers/phase-4`.

**Ainda sem PDF (a obter e fichar):**
- Wen, Bouri, Xu e Zhao (2022), Intraday return predictability in the cryptocurrency markets: Momentum, reversal, or both. DOI 10.1016/j.najef.2022.101733. É o mais importante, por tratar de momentum e reversão intradiária no Bitcoin (2013 a 2020).
- Zaremba, Bilgin, Long e Mercik (2021), Up or down? Short-term reversal, momentum, and liquidity effects in cryptocurrency markets. DOI 10.1016/j.irfa.2021.101908.
- Tzouvanas, Kizys e Tsend-Ayush (2020), Momentum trading in cryptocurrencies. DOI 10.1016/j.econlet.2019.108728.
- Grobys e Sapkota (2019), Cryptocurrencies and momentum. DOI 10.1016/j.econlet.2019.03.028.
- Fil e Kristoufek (2020), Pairs trading in cryptocurrency markets (barras de 5 minutos). DOI 10.1109/access.2020.3024619.
- Gao, Han, Li e Zhou (2018), Market intraday momentum. DOI 10.1016/j.jfineco.2018.05.009. O arquivo gao_et_al_intramom.pdf era na verdade Baltussen et al. (2021), já renomeado.

**Lidos só em parte (conferir os números antes de citar):**
- Liu e Tsyvinski (2021), RFS. DOI 10.1093/rfs/hhaa113. Lida a versão de trabalho NBER 24877.
- Liu, Tsyvinski e Wu (2022), Journal of Finance. DOI 10.1111/jofi.13119. Lida a versão de trabalho NBER 25882.
- Baltussen, Da, Lammers e Martens (2021), JFE. DOI 10.1016/j.jfineco.2021.04.029.

**Outros:**
- Kitron e Wengrowicz (2026), arXiv 2608.21888: preprint sem revisão, autores com interesse comercial. Ler os apêndices e ver se saiu versão publicada, com DOI.
- Eross et al. (2019), DOI 10.1016/j.ribaf.2019.01.008, e Shen, Urquhart e Wang (2022), DOI 10.1111/fire.12290: lidos as versões aceitas, que podem diferir das publicadas.
- Descartada como fonte: tese de mestrado sobre Bollinger em barras horárias (Ekström, 2025), por não ser revisada.

**Decisões aprovadas e ainda não aplicadas no Docs:** Poterba e Summers (nova redação), tirar "momentum" da frase de Grobys e Detzel, Cont mantido na seção do bootstrap (o estudo usa bootstrap estacionário no SPA), ressalva de Lo (2002) sobre a anualização, redação de Park e Irwin. Faltam aprovar: parágrafo novo sobre cripto em horizontes curtos, nova lacuna da seção 2.7 e as seis referências novas.

**Outras pendências:** diagrama do walk-forward v2; conferir se a tabela "Sharpe bruto" do relatório mostra números anualizados (decide se a ressalva de Lo entra na metodologia); fichas das fases 3 e 4 só em txt (a fase 3 já tem docx).

**Aplicado no Docs em 01/11/2026 (rev 211):** Introdução (Grobys e Detzel sem "momentum"; definição de data snooping com Park e Irwin); 2.1 (Park e Irwin com período e ressalva de lucro líquido; Poterba e Summers); 2.2 (parágrafo novo sobre cripto em horizontes curtos, com Liu e Tsyvinski, Shen et al., De Nicola, Corbet e Katsiampa, Caporale e Plastun e Kitron e Wengrowicz; Wei et al. com a razão de Detzel e a queda do Sharpe fora da amostra); 2.3 (Lo); 2.4 (Cont); 2.7 (lacuna reescrita); 3.4 (ressalva da anualização, Lo 2002); seis referências novas. Fica sem aplicar apenas o que está na lista de pendências acima (leituras e PDFs que faltam). Conferir depois: o parágrafo novo cita Liu e Tsyvinski só de forma qualitativa (versão de trabalho lida em parte).
