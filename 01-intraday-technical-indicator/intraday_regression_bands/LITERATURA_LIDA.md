# Literatura lida: fichas para consulta

Fichas dos artigos que o Paulo anexou em 2026-10-28 e que foram lidos para o artigo. Cada ficha traz o desenho do estudo, os resultados com números, o que o texto não faz e como usar no nosso trabalho. Os números vêm do texto dos PDFs; quando algo é leitura minha, está marcado como "leitura nossa".

Cobertura da leitura: texto corrido, notas, legendas e conclusões dos quatro PDFs foram lidos por inteiro. As células numéricas das tabelas (coeficientes por regra e por moeda) foram vistas só nos cabeçalhos, nas notas e nas linhas citadas abaixo. O Online Appendix de Wei et al. e de Hsu et al. não veio nos PDFs. O Hudson e Urquhart (2021) foi lido em 2026-10-28, depois dos outros três.

## Resumo comparativo

| | Grobys, Ahmed e Sapkota (2020) | Wei et al. (2024) | Hsu, Taylor e Wang (2016) | Hudson e Urquhart (2021) |
|---|---|---|---|---|
| Mercado | 11 criptomoedas | 12 criptomoedas | 30 moedas contra o dólar | Bitcoin (2 fontes), Litecoin, Ripple, Ethereum |
| Período e frequência | 2016 a 2018, diário | 2015 a 2022, diário | 1971 a 2015, diário | 2010 a 2017 (conforme a moeda), diário |
| Regras | média móvel (1, n), 5 versões | 7.846 regras STW + 5 razões PMA = 7.851 | 21.195 regras | 14.919 regras em 5 classes |
| Custos | não considera | média da Binance, 0,044% | estimados dos spreads (cerca de 4,5 bps desenvolvidas, 21 bps emergentes) | custo de break-even, comparado a cerca de 50 bps do Bitcoin |
| Data snooping | teste SUR conjunto entre moedas, sem correção da busca | Lucky Factors e SPA sobre as regras com previsibilidade | SPA em etapas | bootstrap estacionário com Bonferroni, Holm, BH e BY |
| Fora da amostra | não tem | sim, três cortes (50%, 75%, 90%) | sim, 2012 a 2015 | sim, só janeiro a junho de 2018 |
| Conclusão | mercado ineficiente; (1,20) rende 8,76% ao ano acima de comprar e manter | regras tradicionais sem valor genuíno; só PMA1 e hashrate sobrevivem | lucro real, mas em queda nas moedas desenvolvidas; custos não eliminam o lucro | rentabilidade significativa mesmo após os ajustes, mas o Bitcoin não mantém previsibilidade fora da amostra |

## 1. Grobys, Ahmed e Sapkota (2020)

**Referência.** GROBYS, Klaus; AHMED, Shaker; SAPKOTA, Niranjan. Technical trading rules in the cryptocurrency market. *Finance Research Letters*, v. 32, art. 101396, 2020. DOI 10.1016/j.frl.2019.101396. O PDF lido é a versão journal pre-proof (aceito em 5/12/2019).

**Desenho.** Preços diários de 11 criptomoedas (coinmarketcap.com), de 1/1/2016 a 31/12/2018, 1.095 observações por moeda. A amostra são as maiores por capitalização em 3/1/2016, sem a Dash. A amostra principal tem 10 moedas, sem o Bitcoin; a robustez inclui o Bitcoin. A regra é o oscilador de médias móveis variáveis (VMA), com média curta de 1 dia e média longa de 20, 50, 100, 150 ou 200 dias. Só posição comprada (não dá para vender a descoberto no período, segundo os autores). Compra quando a média curta passa acima da longa e fica até o sinal de venda. O teste é um Wald de SUR (regressões aparentemente não relacionadas) conjunto entre as moedas, para tratar a correlação contemporânea.

**Resultados.**
- A regra (1,20) rende 45,63% ao ano nas 10 moedas, contra 36,87% de comprar e manter, ou seja, 8,76% ao ano de excesso. Com o Bitcoin, 45,46% contra 36,50%, ou 8,96%.
- O teste conjunto é significativo para (1,20), (1,50) e (1,100). Na (1,20), 5 das 10 moedas têm retorno significativo a 5%; na (1,50), 3; na (1,100), só a Ethereum.
- O excesso cai para 3,65% ao ano na (1,50) e as regras mais longas não passam de comprar e manter. Os autores leem isso como processos de memória curta.
- Conclusão dos autores: os mercados de criptomoedas não são eficientes na forma fraca.

**O que o artigo não faz.** Não inclui custos de transação, não corrige a escolha entre as cinco regras (cita White, 2000, mas não aplica), não tem amostra fora da amostra e cobre três anos de alta (comprar e manter rende 36,87% ao ano). O artigo é uma *Letter*, curto.

**Uso no nosso trabalho.** Exemplo de resultado favorável com uma regra simples e poucos controles. Citar como "regras de médias móveis com ganho sobre comprar e manter em dados diários de 2016 a 2018, sem custos". Não citar como evidência de lucro líquido.

## 2. Wei, Kyriakou, Sermpinis e Stasinakis (2024)

**Referência.** WEI, Mingzhe; KYRIAKOU, Ioannis; SERMPINIS, Georgios; STASINAKIS, Charalampos. Cryptocurrencies and Lucky Factors: The value of technical and fundamental analysis. *International Journal of Finance & Economics*, v. 29, n. 4, p. 4073-4104, 2024. DOI 10.1002/ijfe.2863. Recebido em 29/1/2023, aceito em 3/7/2023, acesso aberto (CC BY). A ordem dos autores na capa do PDF extraído saiu embaralhada; a ordem acima é a do Crossref.

**Desenho.**
- **Ativos.** Doze criptomoedas. As principais são BTC, ETH, XRP e DASH, de 8/8/2015 a 5/3/2022 (2.460 observações no BTC). As outras oito (ADA, AVAX, BNB, DOGE, DOT, LTC, LUNA, SOL) entram como robustez e começam mais tarde. Preços da Bitstamp e da Binance, diários.
- **Regras técnicas.** 7.846 regras do universo de Sullivan, Timmermann e White (1999): médias móveis, suporte e resistência, rompimento de canal, volume (OBV) e filtro. Mais cinco razões entre o log-preço e sua média móvel (PMA, de Detzel et al., 2021), com 1, 2, 4, 10 e 20 semanas. Total de 7.851 regras por série.
- **Fatores fundamentais.** 59, em cinco grupos (fatores tradicionais, câmbio, índices de ações, tecnologia blockchain, tendências de busca e fóruns).
- **Custos.** O desempenho das regras é calculado com o custo médio da Binance, 0,044% (nota 6 do texto). Benchmark: comprar e manter.
- **Seleção.** As 15 melhores regras por Sharpe (também por Sortino) dentro da amostra, em três janelas: 50%, 75% e 90% da amostra como dentro da amostra.
- **Previsibilidade.** Regressões preditivas bivariadas (Neely et al., 2014) com wild bootstrap unilateral.
- **Lucro genuíno.** Lucky Factors (Harvey e Liu, 2021) e SPA (Hansen, 2005), corte de 5%, aplicados ao conjunto de fatores que já mostraram previsibilidade dentro da amostra.
- **Fora da amostra.** Janela expansiva, R² fora da amostra e MSFE-adj. contra a média histórica, mais Sharpe fora da amostra.

**Resultados.**
- Dentro da amostra, as 15 melhores regras batem comprar e manter (Sharpe do BTC até 1,350 na janela de 75%). Mas só uma regra técnica, a PMA1 (razão de 1 semana), e um fator fundamental, o hashrate (HSH), têm previsibilidade e lucro genuínos dentro e fora da amostra.
- Os autores dizem que as regras tradicionais não têm valor nas criptomoedas e que só a PMA, introduzida recentemente, parece ter.
- Fora da amostra, o desempenho das regras técnicas cai muito nas janelas de 75% e 90%. O texto conclui que o valor do período 1 "não é genuíno" e que os benefícios da análise técnica se corroem rápido e dependem de mudanças de momentum.
- Fatores de notícias, sentimento e finanças tradicionais são fracos ou inexistentes; 3mBill, NSQ e OIL aparecem só para BTC e ETH na janela de 90%.
- Sobre a literatura: a maioria dos estudos de criptomoedas usa poucas regras, poucos ativos ou períodos curtos, e o controle de data snooping é raro (Tabela 1 do artigo). Defendem a necessidade de testar regras técnicas contra sorte e data snooping.

**Observações (leitura nossa).**
- O SPA e o Lucky Factors são aplicados ao conjunto reduzido (top 15 por Sharpe e fatores com previsibilidade dentro da amostra), não ao universo completo de 7.851 regras. O nosso SPA cobre a família inteira de 126 candidatas.
- Todo o desenho é diário. Não há barras de minutos.
- O artigo descreve o Hudson e Urquhart (2021) como se, depois da correção de data snooping, não houvesse lucro genuíno. Isso difere do resumo de Hudson e Urquhart e do que o nosso relatório já diz (ver Cuidados).

**Uso no nosso trabalho.** É a base da motivação: a literatura de criptomoedas está dividida, e o estudo mais completo com correção de busca múltipla não encontra valor nas regras tradicionais. Frases seguras: "testam 7.846 regras técnicas em 12 criptomoedas com dados diários, custos da Binance e correção por Lucky Factors e SPA" e "concluem que as regras tradicionais não têm valor genuíno".

## 3. Hsu, Taylor e Wang (2016)

**Referência.** HSU, Po-Hsuan; TAYLOR, Mark P.; WANG, Zigan. Technical trading: Is it still beating the foreign exchange market? *Journal of International Economics*, v. 102, p. 188-208, 2016. DOI 10.1016/j.jinteco.2016.03.012. O PDF lido é o manuscrito aceito (59 páginas com tabelas e figuras).

**Desenho.**
- **Dados.** Cotações diárias de meio-dia em Londres (cedidas pelo BlackRock) de 30 moedas contra o dólar: 9 desenvolvidas e 21 emergentes. Desenvolvidas de 4/1/1971 a 11/9/2015 (11.660 dias); emergentes a partir de datas variadas (Israel desde 1978).
- **Regras.** 21.195, de cinco classes: 2.835 filtros, 12.870 médias móveis, 1.890 suporte e resistência, 3.000 rompimento de canal e 600 osciladores (RSI).
- **Métricas.** Retorno excedente médio e Sharpe, e as versões ajustadas por "tilt" (parte que seria obtida só comprando e mantendo). Os autores chamam de previsibilidade o retorno médio significativamente positivo e de lucratividade o Sharpe significativamente positivo.
- **Custos.** Um terço do spread cotado do forward de um mês (Bloomberg), seguindo Neely e Weller (2013). Antes disso, 5, 4 e 3 bps nas décadas de 1970, 1980 e 1990 nas desenvolvidas. Custo médio de ida de cerca de 4,5 bps nas desenvolvidas e 21 bps nas emergentes.
- **Teste.** SPA em etapas (Hsu, Hsu e Kuan, 2010) com bootstrap estacionário, nível de 10%, 500 reamostragens. O texto resume a linhagem: Reality Check de White (2000), SPA de Hansen (2005), StepM de Romano e Wolf (2005).

**Resultados.**
- Retorno médio: 15 das 30 moedas são previsíveis a 10% (11 a 5%). Nas desenvolvidas, o melhor retorno anual fica entre 6,5% (marco/euro) e 7,7% (iene), média de 6,9%. Nas emergentes, entre 5,2% (rupia) e 16,2% (rublo), média de 9,5%.
- Sharpe: 3 das 9 desenvolvidas (0,652 a 0,752) e 9 das 21 emergentes (até 1,279 no peso colombiano) são significativas a 10%.
- Custo de break-even: o custo de ida que zera o lucro. Em múltiplos do custo estimado, vai de 6,6 a cerca de 43 nas desenvolvidas (regras escolhidas por retorno) e de cerca de 3 a cerca de 170 nas emergentes. Com a seleção por Sharpe, de 10 a 100 nas desenvolvidas e de 13 a mais de 500 nas emergentes, exceto o won coreano (cerca de 3, regra de alta frequência com cerca de 2.500 operações). Conclusão: custos não eliminam necessariamente o lucro.
- Subperíodos: nas desenvolvidas a previsibilidade some desde 1992-1996 (os autores avisam que o poder do teste é baixo em subperíodos curtos); nas emergentes também cai, mas segue maior que nas desenvolvidas.
- Volatilidade: o melhor retorno médio de cada moeda tem correlação de 0,81 com a volatilidade (um ponto a mais de desvio padrão anual vem com 4,17 pontos a mais de retorno anual). Mas o Sharpe dessas regras não se relaciona com a volatilidade (correlação de -0,13). Com alavancagem adequada, retornos iguais com o mesmo risco seriam possíveis em moedas menos voláteis.
- Fora da amostra (regras escolhidas com dados até 2011, avaliadas de 1/1/2012 a 11/9/2015, 1.817 dias): carteira de desenvolvidas com Sharpe 0,24 e 1,35% ao ano (não significativo); emergentes com Sharpe 1,81 e 7,3% ao ano (t de 3,5); combinada com Sharpe 1,2 e 4,6% ao ano. Os autores avisam que o poder do teste é baixo com menos de quatro anos.
- Explicação favorecida: comportamento temporariamente não racional e imaturidade do mercado, em linha com a Hipótese dos Mercados Adaptativos (Lo, 2004). Descartam prêmio de risco constante, crises e intervenção de bancos centrais como causas únicas.

**Uso no nosso trabalho.**
- Precedente de método: precisa de teste de busca múltipla em escala, com custos reais e fora da amostra. É o mesmo caminho que o nosso SPA global.
- Precedente do CBE: o "break-even transaction cost" é o mesmo conceito do nosso CBE, expresso por ida em bps e em múltiplos do custo estimado.
- Precedente para a seção de volume e volatilidade (4.3.2): o retorno médio cresce com a volatilidade, mas o ganho por unidade de risco não. É parecido com o nosso resultado de que o efeito da volatilidade é de escala.
- Precedente para o decaimento no tempo: a previsibilidade cai com a eficiência do mercado.

## 4. Hudson e Urquhart (2021)

**Referência.** HUDSON, Robert; URQUHART, Andrew. Technical trading and cryptocurrencies. *Annals of Operations Research*, v. 297, n. 1-2, p. 191-220, 2021 (publicado online em 30/8/2019). DOI 10.1007/s10479-019-03357-1. Acesso aberto (CC BY 4.0), edição especial "Networks and Risk Management".

**Desenho.**
- **Dados.** Diários, até 31/12/2017. Bitcoin de duas fontes: CoinDesk (desde 18/7/2010, 2.735 observações) e Bitstamp (desde 1/12/2012, 2.217). Mais Litecoin (desde 28/4/2013, 1.710), Ripple (desde 4/8/2013, 1.624) e Ethereum (desde 7/8/2015, 877), do coinmarketcap.com.
- **Regras.** 14.919, em cinco classes seguindo Hsu et al. (2016): médias móveis (7.980), filtros (963), suporte e resistência, osciladores (RSI) e rompimento de canal (4.536). Posição comprada, vendida ou em caixa; retornos simples.
- **Métricas.** Retorno médio, Sharpe, Sortino, Calmar e custo de break-even. O custo de referência é cerca de 50 bps para o Bitcoin (Lintilhac e Tourin, 2017).
- **Data snooping.** P-valor individual de cada regra por bootstrap estacionário (1.000 repetições, bloco 0,1) e ajustes de Bonferroni e Holm (taxa de erro por família, FWER) e de Benjamini-Hochberg (BH) e Benjamini-Yekutieli (BY) (taxa de falsas descobertas, FDR). Não usa Reality Check nem SPA.

**Resultados.**
- As cinco classes têm retorno médio anualizado significativo a 5% em todas as moedas. Filtro e rompimento de canal são as melhores (até 16,45% ao ano no Ripple, média da classe). O ganho vem dos sinais de compra; os de venda são em geral negativos.
- A melhor regra de cada moeda chega a 13,42% ao ano no CoinDesk e a 22,15% no Ethereum; a pior chega a -3,90%.
- Custo de break-even médio por classe, de 7,88 bps (suporte e resistência no Litecoin) a 147,56 bps (filtro no Ethereum). No CoinDesk e no Bitstamp, até 66,41 e 57,51 bps. Até 47,35% das regras de uma classe passam de 50 bps (rompimento de canal no Ethereum); o suporte e resistência quase nunca passa.
- Contra comprar e manter, só de 5% a 16% das regras superam o retorno anualizado, mas de 27% a 51% superam no Sortino e de 28% a 58% no Calmar (menos quedas longas e fortes).
- Depois dos ajustes, ficam significativas a 5% de 20,35% a 33,61% das regras com Bonferroni e até 50,41% com BH (CoinDesk). Só com o bootstrap, de 27,96% a 58,64%.
- Fora da amostra, as melhores regras até 31/12/2017 foram avaliadas de janeiro a junho de 2018, em mercado de queda. O Bitcoin tem retorno, Sharpe e Sortino negativos nas duas fontes. Litecoin, Ripple e Ethereum têm Sharpe positivo (de 0,74 a 1,36).
- Os autores citam que o desempenho das regras tende a diminuir com o tempo e depois da publicação (Menkhoff, 2007; McLean e Pontiff, 2016).

**O que o artigo não faz.** Não usa Reality Check nem SPA, e não repete a seleção a cada dia como um walk-forward. O fora da amostra tem só seis meses, um único corte e cai num mercado de baixa. Os dados terminam em 2017, depois de uma alta forte. As médias por classe misturam regras boas e ruins. É um estudo diário. A soma do apêndice (14.919) usa 630 regras de suporte e resistência, embora liste 945; por isso, citar "quase 15.000" regras.

**Uso no nosso trabalho.**
- Precedente do custo de break-even em criptomoedas. O custo de referência de 50 bps é de mercado à vista; o nosso é de perpétuos (2 a 5 bps por lado).
- Mostra o outro lado de Wei et al.: em dados diários, com ajuste FWER e FDR, boa parte das regras segue significativa. A diferença entre os dois estudos (método de correção, amostra e período) ajuda a justificar um desenho com SPA global e walk-forward.
- Fonte para a frase de que o Bitcoin perde previsibilidade fora da amostra.

**Pistas de leitura sobre dados intradiários (citadas por eles e ainda não abertas).** Hudson, McGroarty e Urquhart (2017), "Sampling frequency and the performance of different types of technical trading rules", *Finance Research Letters*, 22: 136-139. Batten et al. (2018), análise técnica intradiária em metais preciosos, *Journal of International Financial Markets, Institutions and Money*, 52: 102-113. Marshall, Cahan e Cahan (2008b), análise técnica intradiária em ações dos EUA, *Journal of Empirical Finance*, 15: 199-210. Yamamoto (2012), análise técnica intradiária na Bolsa de Tóquio, *Journal of Banking and Finance*, 36: 3033-3047. Podem servir de base para o argumento sobre frequência.

**Mais uma pista sobre dados intradiários (verificada no Crossref, não aberta).** Neely e Weller (2003), "Intraday technical trading in the foreign exchange market", *Journal of International Money and Finance*, 22: 223-237, DOI 10.1016/s0261-5606(02)00101-8. Hsu, Taylor e Wang (2016) a citam junto com Bessembinder e Chan (1995) a propósito do custo de break-even.

**Fichamento formal.** As fichas de 9 campos dos quatro artigos (F20 a F23) estão em `papers/Fichamento phase 1.docx`, e os PDFs em `papers/phase-2/`. Foram redigidas por quatro agentes que leram cada texto por inteiro e conferidas aqui contra esta leitura.

## Cuidados e discrepâncias

1. **Wei et al. sobre Hudson e Urquhart (resolvida).** Wei et al. escrevem que, para Hudson e Urquhart (2021), depois da correção de data snooping não há lucro genuíno em análise técnica de criptomoedas. O texto de Hudson e Urquhart diz outra coisa: de 20% a 50% das regras seguem significativas depois dos ajustes, e só o Bitcoin falha fora da amostra (seis meses). A descrição que o nosso relatório faz de Hudson e Urquhart está de acordo com o texto. Não repetir a leitura de Wei et al. sobre esse artigo.
2. **Idades diferentes.** Grobys, Wei, Hudson e Urquhart e Hsu et al. usam dados diários. Nenhum dos quatro usa barras de minutos. Não afirmar que "ninguém testou" frequências altas; dizer só que os estudos citados usam dados diários.
3. **Grobys não tem custos.** Qualquer menção a lucro deve dizer "sem custos".
4. **Detzel et al. (2021) e Gerritsen et al. (2020).** Descritos aqui só pelo que Wei et al. e Grobys et al. dizem deles; os textos não foram lidos.
5. **Ordem dos autores de Wei et al.** A do Crossref (Wei, Kyriakou, Sermpinis, Stasinakis) é a que está no relatório.

## Referências vistas nessas listas e ainda não conferidas

Candidatas a leitura se o artigo precisar de mais base. Nenhuma foi aberta nem verificada no Crossref.

- Hsu, Hsu e Kuan (2010), *Journal of Empirical Finance*, 17: 471-484. SPA em etapas.
- Bajgrowicz e Scaillet (2012), *Journal of Financial Economics*, 106: 473-491. Descobertas falsas, testes de persistência e custos.
- Neely e Weller (2013), *Journal of Banking & Finance*, 37: 3783-3798. Estimativa de custos e evolução das estratégias.
- Timmermann e Granger (2004), *International Journal of Forecasting*, 20: 15-27. Previsibilidade e autodestruição das regras.
- Lo (2004), *Journal of Portfolio Management*, 30: 15-29. Hipótese dos Mercados Adaptativos.
- Menkhoff e Taylor (2007), *Journal of Economic Literature*, 45: 936-972. Revisão de análise técnica no câmbio.
- Corbet, Eraslan, Lucey e Sensoy (2019), *Finance Research Letters*, 31: 32-37. Regras técnicas em criptomoedas.
- Harvey e Liu (2021), Lucky Factors. A referência de Wei et al. para o método.
- Clark e West (2007) e Campbell e Thompson (2008). Estatísticas fora da amostra usadas por Wei et al.
