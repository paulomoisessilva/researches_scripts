"""Verificações de robustez do estudo v2 (descritivas).

* ``leave_one_out_spa``: o SPA global sem cada ano, ativo, frequência, estratégia ou lookback, para ver se o resultado vem de uma parte;
* ``fixed_param_persistence``: as combinações de parâmetro fixo que deram retorno líquido positivo a um custo, o repetem nas duas
  metades do período e em quantos anos?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .v2_study import bootstrap_counts, spa_curve, threshold_cost

READING_COSTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.5, 5.0)


def leave_one_out_spa(
    dates: pd.DatetimeIndex,
    gross: np.ndarray,
    units: np.ndarray,
    keys: list[tuple],
    key_names: list[str],
    costs: tuple[float, ...] = READING_COSTS,
    reps: int = 5000,
    mean_block: float = 10.0,
    seed: int = 0,
) -> pd.DataFrame:
    """SPA (consistente) da família sem cada parte: um ano de dias de teste, ou um valor de cada coluna de identificação.

    Tirar um ano emenda os dias vizinhos (o bootstrap em blocos trata a série restante como contígua). A linha ``all`` é a família inteira.
    """
    keys_arr = np.array(keys, dtype=object)
    cost_grid = np.array(costs, dtype=float)

    def run(day_mask: np.ndarray, col_mask: np.ndarray) -> dict[str, float]:
        g, u = gross[np.ix_(day_mask, col_mask)], units[np.ix_(day_mask, col_mask)]
        counts = bootstrap_counts(g.shape[0], reps, mean_block, seed)
        curve = spa_curve(g, u, cost_grid, counts=counts)
        row = {f"p_cost_{c:g}": float(p) for c, p in zip(curve["cost_bps_per_side"], curve["spa_pvalue_consistent"])}
        thr = threshold_cost(curve)
        row.update({"threshold_cost_bps": thr["threshold_cost_bps"], "n_days": int(day_mask.sum()), "n_candidates": int(col_mask.sum())})
        return row

    all_days, all_cols = np.ones(len(dates), bool), np.ones(len(keys), bool)
    rows = [{"left_out": "all", "kind": "all", **run(all_days, all_cols)}]
    for year in sorted(set(dates.year)):
        rows.append({"left_out": f"year={year}", "kind": "year", **run(np.asarray(dates.year != year), all_cols)})
    for pos, name in enumerate(key_names):
        for value in sorted(set(keys_arr[:, pos])):
            rows.append({"left_out": f"{name}={value}", "kind": name, **run(all_days, keys_arr[:, pos] != value)})
    return pd.DataFrame(rows)


def fixed_param_persistence(
    gross: np.ndarray,
    units: np.ndarray,
    trades: np.ndarray,
    keys: list[tuple],
    key_names: list[str],
    split_day: int,
    year_of_day: np.ndarray,
    cost_bps: float = 5.0,
    min_trades: int = 30,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Por combinação fixa: CBE, retorno líquido a ``cost_bps`` e operações em cada metade e no total, e em quantos anos o líquido é positivo.

    Resumo: das combinações com líquido positivo e pelo menos ``min_trades`` operações na 1ª metade, quantas ficam positivas na 2ª; a
    correlação de Spearman do CBE entre as metades (combinações com operações suficientes nas duas).
    """
    from scipy.stats import spearmanr

    first, second = slice(0, split_day), slice(split_day, None)
    table = pd.DataFrame(keys, columns=key_names)
    for tag, sl in (("h1", first), ("h2", second), ("all", slice(None))):
        g, u, t = gross[sl].sum(axis=0), units[sl].sum(axis=0), trades[sl].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            table[f"cbe_{tag}"] = np.where(g <= 0, 0.0, np.where(u > 0, 1e4 * g / np.where(u > 0, u, 1.0), np.nan))
        table[f"net_{tag}_bps"] = (g - cost_bps * u / 1e4) * 1e4
        table[f"trades_{tag}"] = t
    years = sorted(set(year_of_day))
    pos_years = np.zeros(gross.shape[1])
    for y in years:
        m = year_of_day == y
        pos_years += (gross[m].sum(axis=0) - cost_bps * units[m].sum(axis=0) / 1e4) > 0
    table["years_positive"] = pos_years
    table["n_years"] = len(years)

    pos1 = (table["net_h1_bps"] > 0) & (table["trades_h1"] >= min_trades)
    both = (table["trades_h1"] >= min_trades) & (table["trades_h2"] >= min_trades)
    rho, p = spearmanr(table.loc[both, "cbe_h1"], table.loc[both, "cbe_h2"])
    summary = {
        "n_combinations": len(table),
        "positive_net_all": int((table["net_all_bps"] > 0).sum()),
        "positive_net_h1_with_min_trades": int(pos1.sum()),
        "of_those_positive_net_h2": int(((table["net_h2_bps"] > 0) & pos1).sum()),
        "of_those_positive_net_h2_with_min_trades": int(((table["net_h2_bps"] > 0) & pos1 & (table["trades_h2"] >= min_trades)).sum()),
        "positive_net_h2_overall_with_min_trades": int(((table["net_h2_bps"] > 0) & (table["trades_h2"] >= min_trades)).sum()),
        "n_with_min_trades_both_halves": int(both.sum()),
        "spearman_cbe_h1_h2": float(rho), "spearman_p": float(p),
        "cost_bps": cost_bps, "min_trades": min_trades,
    }
    return table, summary
