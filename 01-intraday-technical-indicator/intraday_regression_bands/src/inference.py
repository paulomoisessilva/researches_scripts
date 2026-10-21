"""Inferência com múltiplas estratégias: Reality Check (White, 2000) e SPA (Hansen, 2005).

Pergunta: dentre ``K`` estratégias candidatas, alguma tem retorno esperado positivo depois
de corrigir pelo fato de termos olhado para todas (data snooping)?

* H0: ``E[d_k] <= 0`` para todo ``k``, onde ``d_k,t`` é o retorno líquido da estratégia ``k``
  no período ``t`` em excesso de um benchmark (aqui, caixa: retorno zero).
* O bootstrap estacionário (Politis & Romano, 1994) preserva a dependência serial
  (volatilidade agrupada) reamostrando blocos de comprimento geométrico.
* **Reality Check**: ``V = max_k sqrt(n) * dbar_k``; distribuição sob H0 por
  ``V*_b = max_k sqrt(n) * (dbar*_k,b - dbar_k)``.
* **SPA**: estatística studentizada ``T = max(max_k sqrt(n) dbar_k / omega_k, 0)`` e
  recentragem que descarta candidatas claramente ruins, o que dá mais poder que o RC
  quando há muitas estratégias fracas. Reporta os três p-valores de Hansen:
  inferior (``p_l``), consistente (``p_c``, o recomendado) e superior (``p_u``, equivale
  a um RC studentizado).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bootstrap estacionário
# ---------------------------------------------------------------------------


def stationary_bootstrap_indices(
    n: int,
    n_boot: int,
    mean_block: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Índices (n_boot, n) de reamostragem estacionária com blocos de média ``mean_block``.

    Em cada posição, com probabilidade ``p = 1/mean_block`` começa um novo bloco numa posição
    uniforme; caso contrário continua o bloco anterior (índice + 1, circular).
    """
    if n < 2:
        raise ValueError("São necessárias ao menos 2 observações.")
    if mean_block < 1:
        raise ValueError("mean_block deve ser >= 1.")
    p = 1.0 / mean_block
    starts = rng.integers(0, n, size=(n_boot, n))
    new_block = rng.random((n_boot, n)) < p
    new_block[:, 0] = True

    pos = np.arange(n)
    # posição (no caminho reamostrado) em que começou o bloco corrente
    block_start_pos = np.maximum.accumulate(np.where(new_block, pos[None, :], 0), axis=1)
    block_origin = np.take_along_axis(starts, block_start_pos, axis=1)
    return (block_origin + (pos[None, :] - block_start_pos)) % n


def bootstrap_means(
    d: np.ndarray,
    indices: np.ndarray,
    chunk: int = 250,
) -> np.ndarray:
    """Médias reamostradas, shape (n_boot, K), sem materializar (n_boot, n, K) inteiro."""
    n_boot = indices.shape[0]
    out = np.empty((n_boot, d.shape[1]))
    for i in range(0, n_boot, chunk):
        out[i:i + chunk] = d[indices[i:i + chunk]].mean(axis=1)
    return out


# ---------------------------------------------------------------------------
# Reality Check e SPA
# ---------------------------------------------------------------------------


@dataclass
class SpaResult:
    n_obs: int
    n_candidates: int
    best_candidate: int            # índice da melhor candidata (maior média)
    best_mean: float               # retorno médio por período da melhor candidata
    rc_statistic: float
    rc_pvalue: float
    spa_statistic: float
    spa_pvalue_lower: float
    spa_pvalue_consistent: float
    spa_pvalue_upper: float


def reality_check_spa(
    d: np.ndarray,
    boot_means: np.ndarray,
) -> SpaResult:
    """Calcula RC e SPA para a matriz ``d`` (n períodos x K candidatas) e médias bootstrap.

    P-valores usam ``>=`` (caudas com empate): com todas as candidatas ruins ou com retornos
    degenerados (zeros) o resultado é p = 1, nunca uma rejeição espúria.
    """
    d = np.asarray(d, dtype=float)
    n, k = d.shape
    dbar = d.mean(axis=0)
    root_n = np.sqrt(n)

    # --- Reality Check (White): sem studentizar, centrado em dbar (todas com média 0) ---
    rc_stat = float(np.max(root_n * dbar))
    rc_boot = np.max(root_n * (boot_means - dbar[None, :]), axis=1)
    rc_p = float(np.mean(rc_boot >= rc_stat))

    # --- SPA (Hansen): studentizado e com recentragem das candidatas ruins ---
    omega = (root_n * boot_means).std(axis=0, ddof=1)
    degenerate = ~(omega > 0)
    omega_safe = np.where(degenerate, 1.0, omega)
    t_ratio = np.where(degenerate, 0.0, root_n * dbar / omega_safe)
    spa_stat = float(max(t_ratio.max(), 0.0))

    a_n = np.sqrt(2.0 * np.log(np.log(max(n, 3))))
    centers = {
        # p_l: candidatas com média negativa mantêm a deriva negativa (H0 mais favorável)
        "lower": np.maximum(dbar, 0.0),
        # p_c: só as claramente ruins (t <= -A_n) saem do máximo; as demais ficam em dbar
        "consistent": np.where(t_ratio <= -a_n, 0.0, dbar),
        # p_u: todas com média 0 (equivale a um RC studentizado; o mais conservador)
        "upper": dbar,
    }
    pvals: dict[str, float] = {}
    for name, center in centers.items():
        z = root_n * (boot_means - center[None, :]) / omega_safe[None, :]
        z[:, degenerate] = 0.0
        t_boot = np.maximum(z.max(axis=1), 0.0)
        pvals[name] = float(np.mean(t_boot >= spa_stat))

    best = int(np.argmax(dbar))
    return SpaResult(
        n_obs=n,
        n_candidates=k,
        best_candidate=best,
        best_mean=float(dbar[best]),
        rc_statistic=rc_stat,
        rc_pvalue=rc_p,
        spa_statistic=spa_stat,
        spa_pvalue_lower=pvals["lower"],
        spa_pvalue_consistent=pvals["consistent"],
        spa_pvalue_upper=pvals["upper"],
    )


def spa_test(
    d: np.ndarray,
    n_boot: int = 5000,
    mean_block: float = 10.0,
    seed: int = 0,
) -> SpaResult:
    """RC + SPA com bootstrap estacionário (uma chamada)."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(d.shape[0], n_boot, mean_block, rng)
    return reality_check_spa(d, bootstrap_means(d, idx))


# ---------------------------------------------------------------------------
# Dados: matriz diária de retornos líquidos por custo
# ---------------------------------------------------------------------------


def net_returns_at_cost(
    gross: pd.Series | np.ndarray,
    turnover_units: pd.Series | np.ndarray,
    cost_bps_per_side: float,
) -> np.ndarray:
    """Retorno log líquido por período para qualquer custo (fee+slippage por lado, em bps).

    O custo é linear em ``|Δposição|`` e a trajetória de posições não depende do custo,
    então ``net = bruto - turnover * custo / 10_000`` é exato para posições fixas.
    """
    return np.asarray(gross, dtype=float) - np.asarray(turnover_units, dtype=float) * cost_bps_per_side / 10_000.0


def candidate_panels(
    windows: pd.DataFrame,
    candidate_cols: list[str],
) -> dict[tuple, pd.DataFrame]:
    """Separa a tabela de janelas em painéis por candidata, indexados por ``test_date``.

    ``windows`` deve ter ``test_date``, ``sum_gross_ret``, ``turnover_units`` e as colunas que
    identificam a candidata (por exemplo ``symbol``, ``strategy``, ``validation_days``).
    """
    panels: dict[tuple, pd.DataFrame] = {}
    for key, grp in windows.groupby(candidate_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        panels[key] = grp.sort_values("test_date").set_index("test_date")[["sum_gross_ret", "turnover_units"]]
    return panels


def daily_matrix(
    panels: dict[tuple, pd.DataFrame],
    cost_bps_per_side: float,
    keys: list[tuple] | None = None,
) -> tuple[np.ndarray, list[tuple], pd.Index]:
    """Matriz (dias x candidatas) de retornos líquidos ao custo dado, nas datas em comum."""
    keys = keys or list(panels)
    common = None
    for key in keys:
        idx = panels[key].index
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values()
    cols = [
        net_returns_at_cost(
            panels[k].loc[common, "sum_gross_ret"], panels[k].loc[common, "turnover_units"], cost_bps_per_side
        )
        for k in keys
    ]
    return np.column_stack(cols), keys, common
