"""Regra de decisão do piloto, fixada ANTES de olhar os resultados.

Escalar para um período maior (e para a confirmação fora da amostra) só se, simultaneamente:

1. **CBE:** algum método (variante de fechamento x estratégia) tem custo de break-even
   >= ``min_cbe_bps`` em pelo menos ``min_lookbacks`` dos lookbacks testados; e
2. **SPA:** o p-valor SPA consistente da família inteira de candidatas, com o retorno líquido
   a ``spa_cost_bps`` por lado, é <= ``max_spa_p``.

Caso contrário o piloto vira um resultado negativo documentado.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecisionRule:
    min_cbe_bps: float = 3.0
    min_lookbacks: int = 2
    spa_cost_bps: float = 2.0
    max_spa_p: float = 0.10


def cbe_by_candidate(
    windows: pd.DataFrame,
    keys: tuple[str, ...] = ("variant", "strategy", "validation_days"),
) -> pd.DataFrame:
    """CBE total (bps por lado) de cada candidata: retorno bruto / unidades negociadas."""
    g = windows.groupby(list(keys)).agg(
        gross_log=("sum_gross_ret", "sum"),
        units=("turnover_units", "sum"),
        days=("test_date", "nunique"),
    ).reset_index()
    cbe = np.where(
        g["gross_log"] <= 0, 0.0,
        np.where(g["units"] > 0, g["gross_log"] * 1e4 / g["units"].where(g["units"] > 0, 1.0), np.nan),
    )
    g["cbe_bps"] = cbe
    return g


def evaluate_rule(cbe: pd.DataFrame, spa_pvalue: float, rule: DecisionRule) -> dict:
    """Aplica a regra. ``cbe`` precisa de ``variant``, ``strategy``, ``validation_days``, ``cbe_bps``."""
    methods = []
    for (variant, strategy), grp in cbe.groupby(["variant", "strategy"]):
        n_ok = int((grp["cbe_bps"] >= rule.min_cbe_bps).sum())
        methods.append({
            "variant": variant,
            "strategy": strategy,
            "lookbacks_tested": int(len(grp)),
            "lookbacks_with_cbe_ok": n_ok,
            "cbe_by_lookback": {int(r.validation_days): float(r.cbe_bps) for r in grp.itertuples()},
            "cbe_criterion_met": n_ok >= rule.min_lookbacks,
        })
    cbe_met = any(m["cbe_criterion_met"] for m in methods)
    spa_met = bool(spa_pvalue <= rule.max_spa_p)
    return {
        "rule": asdict(rule),
        "methods": methods,
        "cbe_criterion_met": cbe_met,
        "spa_pvalue_at_rule_cost": float(spa_pvalue),
        "spa_criterion_met": spa_met,
        "scale_up": bool(cbe_met and spa_met),
    }
