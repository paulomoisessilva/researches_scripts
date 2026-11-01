"""Probabilidade de existir ao menos UM OUTRO planeta com cada estágio, na galáxia ou no universo.

Estágios aninhados: vida (qualquer) > eucariotos > eucariotos com sexo > inteligência >
tecnologia comunicante. Duas famílias de prior:
  - Sandberg, Drexler & Ord (2018): só define fl, fi e fc, então não tem estágios intermediários.
  - Passos difíceis (Snyder-Beattie et al. 2021): posterior dos tempos observados na Terra, dividida
    pela chance de a transição ocorrer na janela. Isso corrige o viés de amostragem do planeta, mas
    não usa o fato de existir ao menos uma civilização no universo.
"Agora" usa min(1, D/T), com D a duração do estágio numa linha do tempo como a da Terra.

Condicionar em existirmos tem duas versões:
  - ingênuo: pesa cada amostra por P(a própria região tem >= 1 civilização);
  - SSA universo: pesa por P(o universo tem >= 1 civilização), com os mesmos parâmetros em toda parte.
Nos passos difíceis, a coluna "| SSA" multiplica o peso da posterior pelo peso SSA.

Cada coluna traz P(>= 1 outro), a mediana e os decis, entre os mundos sorteados, da probabilidade
1 - exp(-N) e de log10 N, com N o número esperado de outros planetas que tiveram o estágio. Os decis
mostram a bimodalidade que a média das probabilidades esconde.

Uso: python estagios.py [--estrelas 2e11] [--estrelas-universo 2e22] [--T 1e10] [--beta-max 1e10 1e20]
                        [--decis-md decis.md] [--decis-n-md decis_n.md] [--titulo "Via Láctea"] [--varredura]
"""
import argparse
import sys

import numpy as np

import passos_dificeis as pdf
from drake import N_ESTRELAS, fl_sandberg, log_uniforme, prob_alguma

ESTAGIOS = ["vida (qualquer)", "eucariotos", "eucariotos com sexo", "inteligência", "tecnologia comunicante"]
FIM_JANELA = pdf.JANELA - pdf.INICIO                       # Ga que faltam para a janela fechar
DURACAO_GA = {                                             # Ga que o estágio dura numa Terra-análoga
    "vida (qualquer)": 3.8 + FIM_JANELA,
    "eucariotos": 1.8 + FIM_JANELA,
    "eucariotos com sexo": 1.2 + FIM_JANELA,
}
DECIS = np.arange(1, 10) / 10
VARREDURA_BETA = (1e10, 1e12, 1e14, 1e16, 1e18, 1e20, 1e22, 1e25)


def quantis_ponderados(x, w, qs):
    ordem = np.argsort(x)
    acum = np.cumsum(w[ordem])
    return x[ordem][np.minimum(np.searchsorted(acum, qs * acum[-1]), len(x) - 1)]


def tamanho_efetivo(w):
    return w.sum() ** 2 / (w**2).sum()


def contagens_agora(ever, L, T):
    agora = {}
    for nome, n in ever.items():
        fator = np.minimum(1.0, (DURACAO_GA[nome] * 1e9 / T) if nome in DURACAO_GA else L / T)
        agora[nome] = n * fator
    return agora


def prior_sandberg(rng, m, T, n_estrelas):
    pHZ = log_uniforme(rng, 0.16, 0.88, m)
    fl = fl_sandberg(rng, m)
    fi = log_uniforme(rng, 1e-3, 1, m)
    fc = log_uniforme(rng, 1e-2, 1, m)
    L = log_uniforme(rng, 1e2, 1e10, m)
    nHZ = n_estrelas * pHZ
    ever = {"vida (qualquer)": nHZ * fl, "inteligência": nHZ * fl * fi,
            "tecnologia comunicante": nHZ * fl * fi * fc}
    return ever, contagens_agora(ever, L, T)


def prior_passos(rng, m, T, beta_max, n_estrelas):
    _, F, w = pdf.posterior(rng, 4, m, beta_max)
    k = len(w)
    nHZ = n_estrelas * log_uniforme(rng, 0.16, 0.88, k)
    fc = log_uniforme(rng, 1e-2, 1, k)
    L = log_uniforme(rng, 1e2, 1e10, k)
    ever = dict(zip(ESTAGIOS[:4], (nHZ * F[:, j] for j in range(4))))
    ever["tecnologia comunicante"] = nHZ * F[:, 3] * fc
    return ever, contagens_agora(ever, L, T), w


def coluna(ever, agora, w):
    return {nome: dict(
        p_ever=np.average(prob_alguma(ever[nome]), weights=w),
        p_agora=np.average(prob_alguma(agora[nome]), weights=w),
        dec_ever=quantis_ponderados(np.log10(ever[nome]), w, DECIS),
        dec_agora=quantis_ponderados(np.log10(agora[nome]), w, DECIS)) for nome in ever}


def imprimir(colunas):
    largura = 17
    print(f"\n{'':26s}" + "".join(f"{nome:>{2 * largura}s}" for nome in colunas))
    print(f"{'estágio':26s}" + "".join(f"{'já existiu':>{largura}s}{'existe agora':>{largura}s}" for _ in colunas))
    for est in ESTAGIOS:
        linha = f"{est:26s}"
        for c in colunas.values():
            linha += (f"{c[est]['p_ever']:>{largura}.3g}{c[est]['p_agora']:>{largura}.3g}" if est in c
                      else f"{'-':>{largura}s}{'-':>{largura}s}")
        print(linha)
    print("\nmediana de N (outros planetas que já tiveram o estágio)")
    for est in ESTAGIOS:
        print(f"{est:26s}" + "".join(
            f"{10 ** c[est]['dec_ever'][4]:>{2 * largura}.2e}" if est in c else f"{'-':>{2 * largura}s}"
            for c in colunas.values()))


def escrever_decis(colunas, caminho, titulo, probabilidade):
    """Decis entre os mundos sorteados. A probabilidade 1 - exp(-N) é crescente em N, então o decil
    da probabilidade é a probabilidade calculada no decil de N."""
    cab = "| Estágio | " + " | ".join(f"D{int(q * 100)}" for q in DECIS) + " |\n|---|" + "---|" * len(DECIS) + "\n"
    if probabilidade:
        fmt = lambda v: f"{prob_alguma(10.0 ** v):.3g}"
        texto = ("Decis da probabilidade de existir ao menos um outro planeta com o estágio, entre os mundos "
                 "sorteados. Cada mundo é um conjunto de parâmetros. A média sobre todos os mundos é o número "
                 "das tabelas de resultados. D50 é a mediana.")
    else:
        fmt = lambda v: f"{v:.1f}"
        texto = ("Decis de log₁₀ N, com N o número esperado de outros planetas que tiveram o estágio. D50 é a "
                 "mediana. Valores negativos não são civilizações negativas: log₁₀ N = −3 quer dizer N = 0,001.")
    with open(caminho, "a", encoding="utf-8") as f:
        f.write(f"\n## {titulo}\n\n{texto}\n")
        for nome, c in colunas.items():
            for chave, rotulo in (("dec_ever", "já existiu"), ("dec_agora", "existe agora")):
                f.write(f"\n### {nome}, {rotulo}\n\n{cab}")
                for est in ESTAGIOS:
                    if est in c:
                        f.write(f"| {est} | " + " | ".join(fmt(v) for v in c[est][chave]) + " |\n")


def varredura(rng, a):
    razao = a.estrelas_universo / a.estrelas
    print(f"\nP(>= 1 outro, já existiu) por limite superior do prior (N estrelas = {a.estrelas:.1e})")
    print(f"{'beta_max':>9s}" + "".join(f"{e[:14]:>30s}" for e in ("vida", "inteligência", "tecnologia")))
    print(f"{'':9s}" + "".join(f"{'posterior':>15s}{'+ SSA':>15s}" for _ in range(3)))
    for bmax in VARREDURA_BETA:
        ever, _, w = prior_passos(rng, a.amostras_passos, a.T, bmax, a.estrelas)
        w_ssa = w * prob_alguma(ever["tecnologia comunicante"] * razao)
        linha = f"{bmax:9.0e}"
        for est in ("vida (qualquer)", "inteligência", "tecnologia comunicante"):
            p = prob_alguma(ever[est])
            linha += f"{np.average(p, weights=w):15.3g}{np.average(p, weights=w_ssa):15.3g}"
        print(linha + f"   (tam. efetivo SSA = {tamanho_efetivo(w_ssa):,.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amostras", type=int, default=1_000_000)
    ap.add_argument("--amostras-passos", type=int, default=4_000_000)
    ap.add_argument("--semente", type=int, default=1)
    ap.add_argument("--T", type=float, default=1e10)
    ap.add_argument("--estrelas", type=float, default=N_ESTRELAS)
    ap.add_argument("--estrelas-universo", type=float, default=2e22)
    ap.add_argument("--beta-max", type=float, nargs="+", default=[1e10, 1e20])
    ap.add_argument("--decis-md", help="acrescenta os decis da probabilidade a este arquivo markdown")
    ap.add_argument("--decis-n-md", help="acrescenta os decis de log10 N a este arquivo markdown")
    ap.add_argument("--titulo", default="Resultados")
    ap.add_argument("--varredura", action="store_true", help="só varre o limite superior do prior")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(a.semente)

    print(f"N estrelas = {a.estrelas:.1e}, T = {a.T:.2e} anos")
    if a.varredura:
        varredura(rng, a)
        return

    razao = a.estrelas_universo / a.estrelas
    ever, agora = prior_sandberg(rng, a.amostras, a.T, a.estrelas)
    colunas = {"Sandberg": coluna(ever, agora, np.ones(a.amostras))}
    # ingênuo: a própria região precisa conter uma civilização; SSA: o universo inteiro precisa conter uma
    for nome, escala in (("Sandberg | ingênuo", 1.0), ("Sandberg | SSA universo", razao)):
        peso = prob_alguma(ever["tecnologia comunicante"] * escala)
        colunas[nome] = coluna(ever, agora, peso)
        print(f"{nome}: tamanho efetivo da amostra = {tamanho_efetivo(peso):,.0f}")

    for bmax in a.beta_max:
        ever, agora, w = prior_passos(rng, a.amostras_passos, a.T, bmax, a.estrelas)
        w_ssa = w * prob_alguma(ever["tecnologia comunicante"] * razao)
        print(f"Passos difíceis {bmax:.0e}: tamanho efetivo = {tamanho_efetivo(w):,.0f}; com SSA = {tamanho_efetivo(w_ssa):,.0f}")
        colunas[f"Passos difíceis {bmax:.0e}"] = coluna(ever, agora, w)
        colunas[f"Passos difíceis {bmax:.0e} | SSA"] = coluna(ever, agora, w_ssa)

    imprimir(colunas)
    if a.decis_md:
        escrever_decis(colunas, a.decis_md, a.titulo, probabilidade=True)
    if a.decis_n_md:
        escrever_decis(colunas, a.decis_n_md, a.titulo, probabilidade=False)


if __name__ == "__main__":
    main()
