"""Análise agregada do estudo de 1 minuto em blocos (BTC, ETH, SOL; 2021-2025).

Reúne as runs semestrais de ``13_run_full_1m.py`` em uma tabela de janelas diárias por
candidata (ativo x estratégia x lookback) e calcula, sobre ela, as visões do estudo: métricas
por candidata e por período, CBE agregado, Reality Check/SPA por família e custo, e a
comparação com o piloto. Tudo parte dos momentos por janela (``sum_ret``, ``sum_gross_ret``,
``turnover_units``...), como em ``wf_analysis``.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .inference import (
    bootstrap_means,
    candidate_panels,
    daily_matrix,
    reality_check_spa,
    stationary_bootstrap_indices,
)
from .wf_analysis import ALL_PERIOD, _with_period, list_runs, metrics_from_moments

WINDOW_COLS = [
    "strategy", "symbol", "interval", "test_date", "n_bars",
    "sum_ret", "sum_sq_ret", "sum_gross_ret", "turnover_units", "number_of_trades",
]
CANDIDATE = ["symbol", "strategy", "validation_days"]
_RANDOM = "random"


# ---------------------------------------------------------------------------
# Blocos semestrais
# ---------------------------------------------------------------------------


def make_chunks(
    data_start: date,
    data_end: date,
    first_oos: date,
    history_days: int,
) -> list[dict]:
    """Semestres de ``first_oos`` até ``data_end``: rótulo, início dos dados, 1º dia de teste, fim.

    O ``end_date`` de um bloco é o primeiro dia de teste do próximo (um dia de teste só existe se
    o dia inteiro estiver nos dados), e o início dos dados fica ``history_days`` antes do 1º dia
    de teste (limitado ao início dos dados).
    """
    boundaries: list[date] = []
    for year in range(first_oos.year, data_end.year + 1):
        for month in (1, 7):
            d = date(year, month, 1)
            if first_oos < d <= data_end:
                boundaries.append(d)
    firsts = [first_oos, *boundaries]
    chunks = []
    for i, first_test in enumerate(firsts):
        end = firsts[i + 1] if i + 1 < len(firsts) else data_end
        start = max(data_start, first_test - timedelta(days=history_days))
        label = f"{first_test.year}H{1 if first_test.month < 7 else 2}"
        chunks.append({
            "label": label, "start_date": start.isoformat(),
            "oos_start": first_test.isoformat(), "end_date": end.isoformat(),
        })
    return chunks


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


def load_windows(
    runs_root: str | Path,
    train_days: int,
    run_cost: tuple[float, float],
) -> pd.DataFrame:
    """Janelas diárias de todas as runs (blocos) que casam com treino e custo.

    Cada candidata (ativo, estratégia, lookback) deve ter no máximo uma linha por ``test_date``;
    linhas repetidas idênticas (mesmo dia em duas runs) são descartadas e divergentes dão erro.
    """
    frames = []
    for run_dir in list_runs(runs_root):
        meta = json.loads((Path(run_dir) / "run_meta.json").read_text(encoding="utf-8"))
        if meta["train_days"] != train_days or (meta["fee_bps"], meta["slippage_bps"]) != tuple(run_cost):
            continue
        w = pd.read_csv(Path(run_dir) / "walk_forward_results.csv", usecols=WINDOW_COLS)
        w = w[w["strategy"] != _RANDOM].copy()
        w["validation_days"] = int(meta["validation_days"])
        w["run"] = Path(run_dir).name
        frames.append(w)
    if not frames:
        raise SystemExit("Nenhuma run casa com o treino/custo pedidos. Rode o script 13 antes.")
    out = pd.concat(frames, ignore_index=True)
    out["test_date"] = out["test_date"].astype(str)
    key = [*CANDIDATE, "test_date"]
    dup = out.duplicated(key, keep=False)
    if dup.any():
        cols = ["sum_ret", "sum_gross_ret", "turnover_units"]
        spread = out[dup].groupby(key)[cols].agg(lambda s: s.max() - s.min()).abs().to_numpy().max()
        if spread > 1e-12:
            raise ValueError(f"{int(dup.sum())} linhas repetidas por (candidata, dia) com valores diferentes.")
        out = out.drop_duplicates(key, keep="first")
    return out.sort_values([*CANDIDATE, "test_date"]).reset_index(drop=True)


def restrict(windows: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Janelas com ``start <= test_date <= end`` (datas ISO, comparação textual)."""
    mask = pd.Series(True, index=windows.index)
    if start:
        mask &= windows["test_date"] >= start
    if end:
        mask &= windows["test_date"] <= end
    return windows[mask]


# ---------------------------------------------------------------------------
# Métricas por candidata e por período
# ---------------------------------------------------------------------------


def _row_from_group(g: pd.DataFrame, n_candidates: int = 1) -> dict:
    """Métricas de um grupo de janelas; com ``n_candidates > 1`` o retorno é o da carteira igual-pesos."""
    m = metrics_from_moments(g, str(g["interval"].iloc[0]))
    gross_log = g["sum_gross_ret"].sum() / n_candidates
    net_log = g["sum_ret"].sum() / n_candidates
    n_days = max(m["n_days"], 1)
    return {
        "n_days": m["n_days"],
        "gross_pct": float((np.exp(gross_log) - 1.0) * 100.0),
        "net_pct": float((np.exp(net_log) - 1.0) * 100.0),
        "cbe_bps": m["break_even_bps"],
        "trades_per_day": m["number_of_trades"] / n_days / n_candidates,
        "turnover_units_per_day": m["turnover_units"] / n_days / n_candidates,
        # somar linhas de candidatas diferentes não dá o Sharpe da carteira
        "sharpe_net": m["sharpe"] if n_candidates == 1 else float("nan"),
    }


def candidate_table(windows: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por candidata (ativo, estratégia, lookback) sobre as janelas dadas."""
    rows = [
        {"symbol": s, "strategy": st, "validation_days": v, **_row_from_group(g)}
        for (s, st, v), g in windows.groupby(CANDIDATE, sort=True)
    ]
    return pd.DataFrame(rows)


def candidate_table_by_period(windows: pd.DataFrame, freq: str = "year") -> pd.DataFrame:
    """Candidata x período (``year``, ``quarter`` ou ``month``)."""
    df = _with_period(windows, freq)
    rows = []
    for (s, st, v, p), g in df.groupby([*CANDIDATE, "_period"], sort=True):
        rows.append({"symbol": s, "strategy": st, "validation_days": v, "period": p, **_row_from_group(g)})
    return pd.DataFrame(rows)


def pooled_table(windows: pd.DataFrame, by: list[str], freq: str | None = None) -> pd.DataFrame:
    """Carteira igual-pesos das candidatas de cada grupo (por exemplo, por ativo e ano).

    O CBE agregado é a razão das somas (bruto / unidades), igual para qualquer peso igual.
    """
    df = _with_period(windows, freq) if freq else windows.copy()
    keys = list(by) + (["_period"] if freq else [])
    rows = []
    for key_vals, g in df.groupby(keys, sort=True):
        key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        n_cand = g.groupby(CANDIDATE).ngroups
        label = dict(zip(by + (["period"] if freq else []), key_vals))
        rows.append({**label, "n_candidates": n_cand, **_row_from_group(g, n_cand)})
    return pd.DataFrame(rows)


def share_gross_positive(windows: pd.DataFrame, freq: str = "year") -> pd.DataFrame:
    """Fração de candidatas com retorno bruto > 0 em cada período (e no total)."""
    df = candidate_table_by_period(windows, freq)
    tot = candidate_table(windows).assign(period=ALL_PERIOD)
    both = pd.concat([df, tot], ignore_index=True)
    return (
        both.groupby("period")
        .agg(n_candidates=("gross_pct", "size"), share_gross_positive=("gross_pct", lambda s: float((s > 0).mean())),
             median_cbe_bps=("cbe_bps", "median"), max_cbe_bps=("cbe_bps", "max"))
        .reset_index()
    )


def cumulative_series(windows: pd.DataFrame, keys: list[tuple] | None = None) -> pd.DataFrame:
    """Retorno acumulado (%) bruto e líquido da carteira igual-pesos das candidatas dadas."""
    w = windows
    if keys is not None:
        idx = pd.MultiIndex.from_frame(w[CANDIDATE])
        w = w[idx.isin(keys)]
    n_cand = w.groupby(CANDIDATE).ngroups
    daily = w.groupby("test_date")[["sum_gross_ret", "sum_ret"]].sum() / n_cand
    out = pd.DataFrame({
        "test_date": daily.index,
        "gross_pct": (np.exp(daily["sum_gross_ret"].cumsum()) - 1.0).to_numpy() * 100.0,
        "net_pct": (np.exp(daily["sum_ret"].cumsum()) - 1.0).to_numpy() * 100.0,
    })
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reality Check / SPA por família e custo
# ---------------------------------------------------------------------------


def make_families(symbols: Iterable[str], strategies: Iterable[str]) -> dict[str, Callable[[tuple], bool]]:
    """Famílias de candidatas: todas, por ativo e por estratégia (chave = (ativo, estratégia, lookback))."""
    fams: dict[str, Callable[[tuple], bool]] = {"todas": lambda k: True}
    for s in symbols:
        fams[s] = lambda k, s=s: k[0] == s
    for st in strategies:
        fams[st] = lambda k, st=st: k[1] == st
    return fams


def spa_grid(
    windows: pd.DataFrame,
    families: dict[str, Callable[[tuple], bool]],
    costs: list[float],
    reps: int = 5000,
    block: float = 10.0,
    seed: int = 0,
) -> pd.DataFrame:
    """RC e SPA (inferior, consistente, superior) por família e custo, nas janelas dadas."""
    panels = candidate_panels(windows, CANDIDATE)
    rows = []
    for fam, pred in families.items():
        keys = [k for k in sorted(panels) if pred(k)]
        if not keys:
            continue
        d0, _, dates = daily_matrix(panels, costs[0], keys)
        idx = stationary_bootstrap_indices(d0.shape[0], reps, block, np.random.default_rng(seed))
        for cost in costs:
            d, _, _ = daily_matrix(panels, cost, keys)
            r = reality_check_spa(d, bootstrap_means(d, idx))
            rows.append({
                "family": fam, "cost_bps_per_side": cost, "n_candidates": r.n_candidates, "n_days": r.n_obs,
                "best_candidate": "|".join(str(x) for x in keys[r.best_candidate]),
                "best_mean_bps_per_day": r.best_mean * 1e4,
                "rc_pvalue": r.rc_pvalue,
                "spa_pvalue_lower": r.spa_pvalue_lower,
                "spa_pvalue_consistent": r.spa_pvalue_consistent,
                "spa_pvalue_upper": r.spa_pvalue_upper,
            })
    return pd.DataFrame(rows)


def block_sensitivity(
    windows: pd.DataFrame,
    costs: list[float],
    blocks: list[float],
    reps: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    """SPA consistente da família completa para vários comprimentos médios de bloco."""
    rows = []
    for b in blocks:
        g = spa_grid(windows, {"todas": lambda k: True}, costs, reps=reps, block=b, seed=seed)
        g.insert(0, "mean_block_days", b)
        rows.append(g[["mean_block_days", "cost_bps_per_side", "n_days", "rc_pvalue", "spa_pvalue_consistent"]])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Comparação com o piloto
# ---------------------------------------------------------------------------


def compare_with_pilot(
    full: pd.DataFrame,
    pilot: pd.DataFrame,
    start: str,
    end: str,
    atol: float = 1e-12,
) -> dict:
    """Diferença dia a dia entre o estudo em blocos e o piloto contínuo no período em comum."""
    f = restrict(full, start, end)
    p = restrict(pilot, start, end)
    keys = [*CANDIDATE, "test_date"]
    m = f.merge(p, on=keys, suffixes=("_full", "_pilot"))
    out: dict = {
        "period": [start, end],
        "rows_full": int(len(f)), "rows_pilot": int(len(p)), "rows_matched": int(len(m)),
    }
    for col in ("sum_ret", "sum_gross_ret", "turnover_units", "n_bars"):
        diff = (m[f"{col}_full"] - m[f"{col}_pilot"]).abs()
        out[col] = {
            "max_abs_diff": float(diff.max()) if len(diff) else float("nan"),
            "share_equal": float((diff <= atol).mean()) if len(diff) else float("nan"),
        }
    if len(m):
        bad = m[(m["sum_ret_full"] - m["sum_ret_pilot"]).abs() > atol]
        out["days_different"] = int(len(bad))
        out["days_different_by_candidate"] = {
            "|".join(str(x) for x in k): int(len(g)) for k, g in bad.groupby(CANDIDATE)
        }
    return out
