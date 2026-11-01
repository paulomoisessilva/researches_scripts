"""Equação de Drake como probabilidade por estrela, sem subtrair de 1.

Cenários pontuais e Monte Carlo com as distribuições de Sandberg, Drexler & Ord (2018).
Toda probabilidade de "ao menos uma" usa -expm1(-n), que não perde precisão para n ~ 1e-18.

Uso: python drake.py [--amostras 1000000] [--semente 1] [--T 1e10]
"""
import argparse
import math
import sys

import numpy as np

N_ESTRELAS = 2e11
L_GRADE = (1e4, 1e6, 1e8)

# pHZ = pfp * pne (medido pela Kepler); fbt = pfl * pfi * pfc (não medido)
CENARIOS = {
    "Valores atuais da planilha": dict(pHZ=1e-5 * 1e-6, fbt=1e-6**3),
    "Sandberg, abiogenese facil": dict(pHZ=1 / 3, fbt=-math.expm1(-1) * 10**-1.5 * 0.1),
    "Ponto de virada (Frank-Sullivan)": dict(pHZ=1 / 3, fbt=3 / N_ESTRELAS),
    "Snyder-Beattie": dict(pHZ=1 / 3, fbt=1e-12),
}


def prob_alguma(n):
    return -np.expm1(-n)


def contagens(p_estrela, N, L, T):
    n_ever = N * p_estrela
    n_agora = n_ever * np.minimum(1.0, L / T)
    return n_ever, n_agora


def log_uniforme(rng, lo, hi, n):
    return 10 ** rng.uniform(math.log10(lo), math.log10(hi), n)


def fl_sandberg(rng, n):
    # fl = 1 - exp(-lambda*V*t), log10(lambda*V*t) ~ Normal(0, 50): mediana 1 evento por planeta
    z = np.clip(rng.normal(0.0, 50.0, n), -250, 250)
    return -np.expm1(-(10.0**z))


def replica_sandberg(rng, n):
    """Mesmas distribuições da Tabela 1 de Sandberg et al., para validar o código."""
    return (
        log_uniforme(rng, 1, 100, n)
        * log_uniforme(rng, 0.1, 1, n)
        * log_uniforme(rng, 0.1, 1, n)
        * fl_sandberg(rng, n)
        * log_uniforme(rng, 1e-3, 1, n)
        * log_uniforme(rng, 1e-2, 1, n)
        * log_uniforme(rng, 1e2, 1e10, n)
    )


def monte_carlo(rng, n, T):
    pHZ = log_uniforme(rng, 0.16, 0.88, n)
    fbt = fl_sandberg(rng, n) * log_uniforme(rng, 1e-3, 1, n) * log_uniforme(rng, 1e-2, 1, n)
    L = log_uniforme(rng, 1e2, 1e10, n)
    return contagens(pHZ * fbt, N_ESTRELAS, L, T)


def tabela_cenarios(T):
    print(f"Cenários pontuais (N_estrelas = {N_ESTRELAS:.1e}, T = {T:.1e} anos)")
    cab = f"{'cenário':34s} {'p/estrela':>10s} {'N_ever':>10s} {'P(>=1 ever)':>12s}"
    cab += "".join(f" {'N_agora L=' + format(L, '.0e'):>17s}" for L in L_GRADE)
    print(cab)
    for nome, c in CENARIOS.items():
        p = c["pHZ"] * c["fbt"]
        linha = f"{nome:34s} {p:10.2e} {N_ESTRELAS * p:10.2e} {prob_alguma(N_ESTRELAS * p):12.2e}"
        for L in L_GRADE:
            linha += f" {contagens(p, N_ESTRELAS, L, T)[1]:17.2e}"
        print(linha)


def resumir(nome, x):
    q5, q25, q50, q75, q95 = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(
        f"{nome:10s} P(N<1)={np.mean(x < 1):.3f}  mediana={q50:.2e}  média={x.mean():.2e}  "
        f"q05={q5:.1e}  q25={q25:.1e}  q75={q75:.1e}  q95={q95:.1e}  P(>=1)={prob_alguma(x).mean():.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amostras", type=int, default=1_000_000)
    ap.add_argument("--semente", type=int, default=1)
    ap.add_argument("--T", type=float, default=1e10)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(a.semente)

    tabela_cenarios(a.T)

    print("\nValidação contra Sandberg et al. 2018 (Pr[N<1] = 0,52; média = 2,7e7; mediana = 0,32)")
    resumir("réplica", replica_sandberg(rng, a.amostras))

    print(f"\nMonte Carlo deste modelo ({a.amostras:,} amostras)")
    n_ever, n_agora = monte_carlo(rng, a.amostras, a.T)
    resumir("N_ever", n_ever)
    resumir("N_agora", n_agora)


if __name__ == "__main__":
    main()
