# Equação de Drake como probabilidade: resultados

Gerado em 2026-10-28 por `estagios.py` (semente 1, 10⁶ amostras no prior do Sandberg e 4×10⁶ nos passos difíceis). Os decis da probabilidade, para todos os estágios, estão em [decis.md](decis.md), e os decis do número esperado de civilizações (log₁₀ N) em [decis_n.md](decis_n.md).

Cada célula mostra **já existiu / existe agora**: a probabilidade de existir ao menos um *outro* planeta com o estágio, em qualquer época e no presente. "—" significa que o prior não define o estágio.

## Colunas

| Coluna | O que é |
|---|---|
| Sandberg | Priors de Sandberg, Drexler & Ord (2018) para `fl`, `fi`, `fc` e L, com `pHZ` log-uniforme de 0,16 a 0,88. Não leva em conta que existimos. |
| Sandberg, ingênuo | Pesa cada amostra por P(a própria região tem ao menos uma civilização). Limite superior da atualização. |
| Sandberg, SSA | Pesa por P(o universo de 2×10²² estrelas tem ao menos uma civilização), com os mesmos parâmetros em toda parte. É o condicionamento em existirmos que eu defendo. |
| Passos difíceis 10¹⁰ / 10²⁰ | Posterior de Snyder-Beattie et al. (2021) com a linha do tempo da Terra, para limites superiores de 10¹⁰ e 10²⁰ Ga no tempo esperado de cada transição. Corrige o viés de amostragem do planeta, mas não usa o fato de existir ao menos uma civilização no universo. |
| Passos difíceis, SSA | A posterior acima multiplicada pelo mesmo peso SSA do Sandberg. |

## Via Láctea (2×10¹¹ estrelas, T = 10¹⁰ anos)

| Estágio | Sandberg | Sandberg, ingênuo | Sandberg, SSA | PD 10¹⁰ | PD 10¹⁰, SSA | PD 10²⁰ | PD 10²⁰, SSA |
|---|---|---|---|---|---|---|---|
| Vida (qualquer) | 0,589 / 0,587 | 1 / 1 | 0,902 / 0,898 | 1 / 1 | 1 / 1 | 0,620 / 0,605 | 0,955 / 0,950 |
| Eucariotos | — | — | — | 0,729 / 0,689 | 0,863 / 0,830 | 0,207 / 0,190 | 0,745 / 0,713 |
| Eucariotos com sexo | — | — | — | 0,380 / 0,335 | 0,532 / 0,470 | 0,0539 / 0,0466 | 0,435 / 0,390 |
| Inteligência | 0,577 / 0,545 | 0,999 / 0,954 | 0,884 / 0,834 | 0,143 / 0,0501 | 0,201 / 0,0703 | 0,00995 / 0,00351 | 0,118 / 0,0418 |
| Tecnologia comunicante | 0,570 / 0,523 | 0,996 / 0,918 | 0,872 / 0,801 | 0,109 / 0,0355 | 0,153 / 0,0498 | 0,00751 / 0,00244 | 0,0895 / 0,0290 |

## Universo observável (2×10²² estrelas, T = 1,38×10¹⁰ anos)

Aqui "ingênuo" e "SSA" do Sandberg coincidem, porque a região é o próprio universo.

| Estágio | Sandberg | Sandberg, SSA | PD 10¹⁰ | PD 10¹⁰, SSA | PD 10²⁰ | PD 10²⁰, SSA |
|---|---|---|---|---|---|---|
| Vida (qualquer) | 0,672 / 0,668 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Eucariotos | — | — | 1 / 1 | 1 / 1 | 0,668 / 0,640 | 1 / 1 |
| Eucariotos com sexo | — | — | 0,961 / 0,945 | 1 / 0,999 | 0,301 / 0,275 | 0,999 / 0,996 |
| Inteligência | 0,661 / 0,630 | 0,999 / 0,963 | 0,766 / 0,521 | 0,995 / 0,724 | 0,0981 / 0,0515 | 0,986 / 0,597 |
| Tecnologia comunicante | 0,653 / 0,622 | 0,997 / 0,953 | 0,712 / 0,461 | 0,977 / 0,646 | 0,0840 / 0,0429 | 0,951 / 0,509 |

Estamos sozinhos no universo observável? É 1 menos a probabilidade de existir outra civilização comunicante que já existiu: 0,35 (Sandberg), 0,29 (PD 10¹⁰) e 0,92 (PD 10²⁰) sem condicionar, e 0,003, 0,023 e 0,049 com SSA.

Com 2×10²³ estrelas, os valores sem condicionar mudam menos de 0,05 em valor absoluto.

## Sensibilidade ao limite superior do prior (universo, já existiu)

| Limite (Ga) | Vida | | Inteligência | | Tecnologia | |
|---|---|---|---|---|---|---|
| | posterior | + SSA | posterior | + SSA | posterior | + SSA |
| 10¹⁰ | 1 | 1 | 0,764 | 0,995 | 0,711 | 0,977 |
| 10¹² | 1 | 1 | 0,527 | 0,991 | 0,473 | 0,966 |
| 10¹⁴ | 1 | 1 | 0,343 | 0,989 | 0,302 | 0,959 |
| 10¹⁶ | 1 | 1 | 0,221 | 0,988 | 0,192 | 0,955 |
| 10¹⁸ | 1 | 1 | 0,146 | 0,987 | 0,125 | 0,952 |
| 10²⁰ | 1 | 1 | 0,0988 | 0,986 | 0,0845 | 0,950 |
| 10²² | 1 | 1 | 0,0677 | 0,986 | 0,0578 | 0,950 |
| 10²⁵ | 0,918 | 1 | 0,0414 | 0,986 | 0,0354 | 0,950 |

Sem condicionar, a probabilidade de inteligência cai de 0,76 para 0,04 quando o limite sobe. Com SSA ela fica entre 0,986 e 0,995.

## Entradas

| Variável | Valor usado |
|---|---|
| `pfp × pne` (`pHZ`) | 0,16 a 0,88, log-uniforme (central ≈ 1 em 3) |
| `fl` | Sandberg: mediana 0,63, bimodal (log₁₀ de λVt ~ Normal(0, 50)). Passos difíceis: passo de abiogênese |
| `fi` | 1 em 1 a 1 em 1000, log-uniforme, mediana 1 em 32 (só Sandberg) |
| `fc` | 1 em 1 a 1 em 100, log-uniforme, mediana 1 em 10 |
| L | 100 anos a 10¹⁰ anos, log-uniforme |
| Durações para "agora" | Vida 4,8 Ga, eucariotos 2,8 Ga, com sexo 2,2 Ga (linha do tempo da Terra); inteligência e tecnologia usam L |
| Passos difíceis | Datas da Terra: habitabilidade a partir de 4,2 Gya, abiogênese 3,8, eucariotos 1,8, sexo 1,2, inteligência hoje; janela de 5,2 Ga; tempo esperado de cada transição log-uniforme de 10⁻¹⁴ Ga até o limite superior |

## Limites

- O prior de passos difíceis não reproduz a Tabela 2 de Snyder-Beattie et al. com o limite declarado de 10¹⁰ Ga. Os números do artigo só se aproximam com limites entre 10¹⁹ e 10²², e mesmo assim não em todas as colunas. A rotina da CDF bate com as fórmulas exatas.
- Os "1" de vida e de eucariotos sem condicionar são efeito do teto do prior: com o limite 10¹⁰ na galáxia e 10²⁰ no universo, o passo mais lento permitido ainda produz centenas de planetas com vida.
- O condicionamento SSA assume que "existimos" é evidência apenas de que o universo tem ao menos uma civilização, com a mesma probabilidade por planeta em toda parte. Outras posições, como o SIA, dão pesos maiores a mundos com mais civilizações e empurram os resultados para cima.
- O modelo não tem passo para animais. "Eucariotos com sexo" é o que mais se aproxima de vida não microbiana, mas é um pré-requisito, não o resultado.
- "Existe agora" assume início uniforme das civilizações em T e durações tiradas da Terra. No universo, "agora" não é observável nem tem simultaneidade bem definida, então usei o tempo cósmico. A tabela trata de existência, não de contato.
- No universo o resultado condicionado diz que há outras civilizações em algum lugar, não que haja alguma na Via Láctea: nos passos difíceis, a mediana de N para inteligência é 10⁴ no universo e da ordem de 10⁻⁶ na galáxia.
- A probabilidade das tabelas é a média entre os mundos sorteados, e em cada mundo ela costuma ser quase 0 ou quase 1 (ver [decis.md](decis.md)). Nos passos difíceis com limite 10²⁰, a mediana da probabilidade de inteligência no universo, sem condicionar, é 2×10⁻¹⁶ e a média é 0,098, porque cerca de uma décima parte dos mundos concentra a probabilidade.

## Reprodução

```bash
python estagios.py --decis-md decis.md --decis-n-md decis_n.md --titulo "Via Láctea"
python estagios.py --estrelas 2e22 --T 1.38e10 --decis-md decis.md --decis-n-md decis_n.md --titulo "Universo observável"
python estagios.py --estrelas 2e22 --T 1.38e10 --varredura
```

## Fontes

- Bryson et al., [arXiv 2010.14812](https://arxiv.org/abs/2010.14812)
- Dressing & Charbonneau 2015, [arXiv 1501.01623](https://arxiv.org/abs/1501.01623)
- Cassan et al. 2012, [arXiv 1202.0903](https://arxiv.org/abs/1202.0903)
- Sandberg, Drexler & Ord 2018, [arXiv 1806.02404](https://arxiv.org/abs/1806.02404)
- Snyder-Beattie et al. 2021, [Astrobiology](https://pmc.ncbi.nlm.nih.gov/articles/PMC7997718)
- Frank & Sullivan, [arXiv 1510.08837](https://arxiv.org/abs/1510.08837)
- Lineweaver & Davis 2002, [astro-ph/0205014](https://arxiv.org/abs/astro-ph/0205014)
- Spiegel & Turner, [arXiv 1107.3835](https://arxiv.org/abs/1107.3835)
- Kipping 2020, [arXiv 2005.09008](https://arxiv.org/abs/2005.09008)
