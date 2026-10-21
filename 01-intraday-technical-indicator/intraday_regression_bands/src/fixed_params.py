"""Backtests de parâmetro fixo (janela x k) sobre um bloco de dados, para o diagnóstico descritivo.

Cada combinação do grid roda como uma estratégia de parâmetro único numa série contínua; os
retornos brutos, as unidades negociadas e as operações são agregados por dia UTC dentro de
``[start, end)``. Os blocos de 5 anos de 1 minuto são rodados um a um (memória) e as séries
diárias concatenadas por quem chama.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .fast_select import band_grid
from .strategies import BAND_STRATEGIES, compute_band_z, signal_from_z

_state: dict = {}


def _init(df_feat: pd.DataFrame, bt_cfg: BacktestConfig, start: pd.Timestamp, end: pd.Timestamp) -> None:
    _state.update(df=df_feat, bt=bt_cfg, start=start, end=end)


def _eval(task: tuple[str, dict, dict]) -> dict:
    strategy, struct, thresh = task
    st = _state
    z = compute_band_z(st["df"], strategy, struct["window"], struct["k"])
    signal = signal_from_z(z, name=strategy, **thresh)
    res = run_backtest(st["df"], signal, st["bt"])

    ts = pd.DatetimeIndex(st["df"]["timestamp"])
    mask = (ts >= st["start"]) & (ts < st["end"])
    day = ts[mask].normalize()
    gross = pd.Series(res.gross_returns.to_numpy()[mask], index=day).groupby(level=0).sum()
    units = pd.Series(res.trade_units.to_numpy()[mask], index=day).groupby(level=0).sum()
    trades = pd.Series(0.0, index=gross.index)
    if not res.trades.empty:
        exit_ts = pd.to_datetime(res.trades["exit_timestamp"], utc=True)
        exit_ts = exit_ts[(exit_ts >= st["start"]) & (exit_ts < st["end"])]
        counts = exit_ts.dt.normalize().value_counts()
        trades = trades.add(counts.reindex(trades.index).fillna(0.0), fill_value=0.0)
    return {
        "strategy": strategy, "window": struct["window"], "k": struct["k"],
        "gross": gross, "units": units, "trades": trades,
    }


def grid_tasks(strategies: list[str], cfg: dict) -> list[tuple[str, dict, dict]]:
    return [(s, struct, thresh) for s in strategies if s in BAND_STRATEGIES for struct, thresh in band_grid(s, cfg)]


def evaluate_chunk(
    df_feat: pd.DataFrame,
    strategies: list[str],
    cfg: dict,
    bt_cfg: BacktestConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    workers: int = 1,
) -> dict:
    """Roda todo o grid em ``df_feat`` e devolve matrizes (dias x combos) em ``[start, end)``.

    Retorna ``{"days", "combos", "gross", "units", "trades"}``; ``combos`` é uma lista de
    ``(estratégia, janela, k)`` na ordem das colunas.
    """
    tasks = grid_tasks(strategies, cfg)
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), initializer=_init, initargs=(df_feat, bt_cfg, start, end)
        ) as ex:
            results = list(ex.map(_eval, tasks, chunksize=2))
    else:
        _init(df_feat, bt_cfg, start, end)
        results = [_eval(t) for t in tasks]

    days = results[0]["gross"].index
    def col(key: str) -> np.ndarray:
        return np.column_stack([r[key].reindex(days).fillna(0.0).to_numpy() for r in results])
    return {
        "days": days,
        "combos": [(r["strategy"], r["window"], r["k"]) for r in results],
        "gross": col("gross"), "units": col("units"), "trades": col("trades"),
    }


def summarize(
    days: pd.DatetimeIndex,
    combos: list[tuple],
    gross: np.ndarray,
    units: np.ndarray,
    trades: np.ndarray,
    cost_bps: float,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Uma linha por (combo, período): ``ALL`` (limitado por ``start``/``end``) e um por ano.

    CBE = 10^4 * soma(bruto) / soma(unidades), 0 quando o bruto é <= 0 (como em ``metrics``).
    """
    days = pd.DatetimeIndex(days)
    keep = np.ones(len(days), dtype=bool)
    if start:
        keep &= days >= pd.Timestamp(start, tz=days.tz)
    if end:
        keep &= days <= pd.Timestamp(end, tz=days.tz)
    years = days.year.to_numpy()
    periods = [("ALL", keep)] + [(str(y), keep & (years == y)) for y in sorted(set(years[keep]))]

    rows = []
    for label, mask in periods:
        n_days = int(mask.sum())
        if n_days == 0:
            continue
        g, u, t = gross[mask].sum(axis=0), units[mask].sum(axis=0), trades[mask].sum(axis=0)
        net = g - u * cost_bps / 1e4
        with np.errstate(divide="ignore", invalid="ignore"):
            cbe = np.where(g <= 0, 0.0, np.where(u > 0, g * 1e4 / np.where(u > 0, u, 1.0), np.nan))
        for j, (strategy, window, k) in enumerate(combos):
            rows.append({
                "strategy": strategy, "window": window, "k": k, "period": label, "n_days": n_days,
                "gross_pct": float((np.exp(g[j]) - 1.0) * 100.0), "net_pct": float((np.exp(net[j]) - 1.0) * 100.0),
                "cbe_bps": float(cbe[j]), "trades_per_day": float(t[j] / n_days),
            })
    return pd.DataFrame(rows)


def split_persistence(
    days: pd.DatetimeIndex,
    combos: list[tuple],
    gross: np.ndarray,
    units: np.ndarray,
    trades: np.ndarray,
    cost_bps: float,
    split_date: str,
    min_trades: int = 30,
) -> pd.DataFrame:
    """Uma linha por combo com as métricas antes e depois de ``split_date`` (exploratório).

    ``net_*`` é o retorno log líquido do período; ``elegivel`` marca os combos com pelo menos
    ``min_trades`` operações no período anterior, para a escolha não cair em amostras minúsculas.
    """
    days = pd.DatetimeIndex(days)
    cut = pd.Timestamp(split_date, tz=days.tz)
    parts = {"S": days < cut, "T": days >= cut}
    rows = []
    cols: dict[str, np.ndarray] = {}
    for tag, mask in parts.items():
        g, u, t = gross[mask].sum(axis=0), units[mask].sum(axis=0), trades[mask].sum(axis=0)
        net = g - u * cost_bps / 1e4
        with np.errstate(divide="ignore", invalid="ignore"):
            cbe = np.where(g <= 0, 0.0, np.where(u > 0, g * 1e4 / np.where(u > 0, u, 1.0), np.nan))
        cols.update({f"gross_{tag}": g, f"net_{tag}": net, f"cbe_{tag}": cbe, f"trades_{tag}": t})
    for j, (strategy, window, k) in enumerate(combos):
        rows.append({"strategy": strategy, "window": window, "k": k,
                     **{name: float(arr[j]) for name, arr in cols.items()},
                     "elegivel": bool(cols["trades_S"][j] >= min_trades)})
    return pd.DataFrame(rows)
