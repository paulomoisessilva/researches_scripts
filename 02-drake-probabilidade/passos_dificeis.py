"""Modelo de passos difíceis de Snyder-Beattie, Sandberg, Drexler & Bonsall (2021).

Cada transição i tem tempo esperado beta_i (Ga) com prior log-uniforme em [1e-14, beta_max].
Verossimilhança: produto de exponenciais dos tempos observados na Terra, dividida pela
probabilidade de todas as transições ocorrerem dentro da janela de habitabilidade.
A CDF da soma de exponenciais usa uniformização, que só soma termos positivos e por isso
mantém precisão relativa mesmo com F ~ 1e-30.

Uso: python passos_dificeis.py   (compara com a Tabela 2 do artigo)
"""
import math
import sys

import numpy as np

JANELA = 5.2      # Ga de habitabilidade total
INICIO = 4.2      # Gya em que a habitabilidade começa
DATAS = {         # Gya: abiogênese, eucariotos, sexo, inteligência (hoje = 0)
    2: [3.8, 0.0],
    3: [3.8, 1.8, 0.0],
    4: [3.8, 1.8, 1.2, 0.0],
}
BETA_MIN = 1e-14
BETA_RAPIDO = 0.02    # piso de beta só dentro da CDF: passos mais rápidos não mudam F
CORTE_EXPOENTE = 40   # t/beta acima disso dá peso < e^-40 e é descartado

# Tabela 2 do artigo (L = 5,2 Ga): fração do peso com P(último passo) acima de 1e-2, 1e-12, 1e-24
LIMIARES = (1e-2, 1e-12, 1e-24)
ARTIGO_PRIOR = {2: (0.42, 0.66, 0.94), 3: (0.27, 0.51, 0.84), 4: (0.18, 0.39, 0.71)}
ARTIGO_POSTERIOR = {2: (0.068, 0.40, 0.83), 3: (0.0022, 0.059, 0.30), 4: (0.0003, 0.014, 0.11)}


def tempos_observados(n):
    return -np.diff([INICIO] + DATAS[n])


def _uniformizacao(lam, q, ql, kmax):
    m, n = lam.shape
    avanco = lam / q[:, None]
    s = np.zeros((m, n + 1))
    s[:, 0] = 1.0
    acc = np.zeros((m, n))
    for k in range(kmax + 1):
        pmf = np.exp(-ql + k * np.log(ql) - math.lgamma(k + 1))
        cauda = np.cumsum(s[:, ::-1], axis=1)[:, ::-1]   # P(>= c transições concluídas)
        acc += pmf[:, None] * cauda[:, 1:]
        mov = s[:, :n] * avanco
        s[:, :n] -= mov
        s[:, 1:] += mov
    return acc


def cdf_etapas(beta, L=JANELA, bloco=20000):
    """F[:, j-1] = P(as j primeiras transições terminam em até L Ga)."""
    lam = 1.0 / np.maximum(beta, BETA_RAPIDO)
    q = lam.max(axis=1)
    ql = q * L
    kmax = np.ceil(ql + 12 * np.sqrt(ql) + 40).astype(int)
    ordem = np.argsort(kmax)
    F = np.empty(beta.shape)
    for ini in range(0, len(beta), bloco):
        idx = ordem[ini:ini + bloco]
        F[idx] = _uniformizacao(lam[idx], q[idx], ql[idx], int(kmax[idx].max()))
    return F


def sortear_beta(rng, n, m, beta_max):
    return 10 ** rng.uniform(math.log10(BETA_MIN), math.log10(beta_max), (m, n))


def posterior(rng, n, m, beta_max=1e10):
    """Amostras da posterior por importância. Retorna beta, F (por etapa) e pesos normalizados."""
    t = tempos_observados(n)
    beta = sortear_beta(rng, n, m, beta_max)
    beta = beta[(t / beta).max(axis=1) < CORTE_EXPOENTE]
    F = cdf_etapas(beta)
    logw = np.sum(-np.log(beta) - t / beta, axis=1) - np.log(F[:, -1])
    w = np.exp(logw - logw.max())
    return beta, F, w / w.sum()


def fracoes(F_ultimo, w=None):
    return tuple(np.average(F_ultimo > lim, weights=w) for lim in LIMIARES)


def validar(m=300_000, semente=1):
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(semente)
    print("Fração do peso com P(último passo) > 1e-2 / 1e-12 / 1e-24 (L = 5,2 Ga, beta_max = 1e10)")
    print(f"{'n':>2s} {'':10s} {'meu código':>26s}   {'artigo':>22s}")
    for n in (2, 3, 4):
        F = cdf_etapas(sortear_beta(rng, n, m, 1e10))[:, -1]
        _, Fp, w = posterior(rng, n, 20 * m)
        for nome, meu, art in (("prior", fracoes(F), ARTIGO_PRIOR[n]),
                               ("posterior", fracoes(Fp[:, -1], w), ARTIGO_POSTERIOR[n])):
            print(f"{n:>2d} {nome:10s} {' '.join(f'{x:8.4f}' for x in meu):>26s}   "
                  f"{' '.join(f'{x:7.4f}' for x in art):>22s}")


if __name__ == "__main__":
    validar()
