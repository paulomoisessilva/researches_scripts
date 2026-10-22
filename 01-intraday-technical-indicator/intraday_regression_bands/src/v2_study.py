"""Análise do estudo v2 (1, 5 e 15 minutos, posição contínua, seleção a custo 0).

Blocos:

* leitura das runs (``load_windows``) e matrizes diárias de retorno bruto e unidades negociadas por
  candidata (``daily_matrices``);
* SPA/RC em família, com a curva de p-valor por custo e o custo-limite (``spa_curve``,
  ``threshold_cost``). Como o líquido é linear no custo, as médias do bootstrap saem de duas
  multiplicações de matrizes (contagens do bootstrap x retorno bruto e x unidades) e valem para
  todos os custos;
* tabelas por candidata e por ano (``candidate_table``, ``by_year_table``);
* parâmetros fixos (``fixed_param_daily``): cada combinação (janela, k) numa trajetória contínua;
* perfil das entradas por hora e dia da semana (``accumulate_profile``, ``profile_tables``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fast_core import backtest_arrays, reversal_position_array
from .inference import reality_check_spa, stationary_bootstrap_indices
from .strategies import compute_band_z

FREQS = ("1m", "5m", "15m")
COST_GRID = np.round(np.arange(0.0, 8.0 + 1e-9, 0.25), 2)
READING_COSTS = (2.0, 4.5, 5.0)  # maker regular, taker com BNB, taker regular (bps por lado)
WINDOW_USECOLS = [
    "strategy", "symbol", "interval", "test_date", "n_bars",
    "sum_gross_ret", "turnover_units", "number_of_trades",
]
KEY = ["freq", "symbol", "strategy", "lookback"]


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Tabela em Markdown sem depender do ``tabulate``."""
    def fmt(x) -> str:
        if isinstance(x, (float, np.floating)):
            return "" if np.isnan(x) else format(x, floatfmt)
        return "" if x is None else str(x)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------


def load_windows(results_root: str | Path) -> pd.DataFrame:
    """Janelas diárias de todas as runs de ``results_root/{1m,5m,15m}/walk_forward``."""
    root = Path(results_root)
    frames = []
    for freq in FREQS:
        for run in sorted((root / freq / "walk_forward").iterdir()):
            meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
            df = pd.read_csv(run / "walk_forward_results.csv", usecols=WINDOW_USECOLS)
            df["freq"] = freq
            df["lookback"] = int(meta["validation_days"])
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def daily_matrices(win: pd.DataFrame) -> dict:
    """Matrizes (dias x candidatas) de bruto, unidades e operações, na mesma ordem de colunas."""
    value_cols = {"gross": "sum_gross_ret", "units": "turnover_units", "trades": "number_of_trades"}
    out = {}
    for name, col in value_cols.items():
        piv = win.pivot_table(index="test_date", columns=KEY, values=col, aggfunc="sum").sort_index()
        piv = piv.reindex(sorted(piv.columns), axis=1)
        if piv.isna().any().any():
            raise ValueError("Há candidatas sem algum dia de teste: as janelas não coincidem.")
        out[name] = piv
    keys = list(out["gross"].columns)
    return {
        "dates": pd.DatetimeIndex(out["gross"].index),
        "keys": keys,
        "gross": out["gross"].to_numpy(float),
        "units": out["units"].to_numpy(float),
        "trades": out["trades"].to_numpy(float),
    }


# ---------------------------------------------------------------------------
# SPA e curva por custo
# ---------------------------------------------------------------------------


def bootstrap_counts(n: int, reps: int = 5000, mean_block: float = 10.0, seed: int = 0) -> np.ndarray:
    """Quantas vezes cada dia entra em cada reamostra do bootstrap estacionário: (reps, n)."""
    idx = stationary_bootstrap_indices(n, reps, mean_block, np.random.default_rng(seed))
    counts = np.empty((reps, n))
    for i in range(reps):
        counts[i] = np.bincount(idx[i], minlength=n)
    return counts


def spa_curve(
    gross: np.ndarray,
    units: np.ndarray,
    costs: np.ndarray = COST_GRID,
    counts: np.ndarray | None = None,
    reps: int = 5000,
    mean_block: float = 10.0,
    seed: int = 0,
    keys: list | None = None,
) -> pd.DataFrame:
    """RC e SPA (inferior, consistente e superior) da família ``gross``/``units`` para cada custo.

    A candidata k, a custo c por lado, tem retorno diário ``gross - c * units / 1e4``. As médias do
    bootstrap vêm de ``(contagens @ bruto - c/1e4 * contagens @ unidades) / n``.
    """
    n, k = gross.shape
    if counts is None:
        counts = bootstrap_counts(n, reps, mean_block, seed)
    cg = counts @ gross
    cu = counts @ units
    rows = []
    for c in costs:
        d = gross - c * units / 1e4
        r = reality_check_spa(d, (cg - c * cu / 1e4) / n)
        best = keys[r.best_candidate] if keys is not None else r.best_candidate
        rows.append({
            "cost_bps_per_side": float(c), "n_candidates": k, "n_days": n,
            "rc_pvalue": r.rc_pvalue, "spa_pvalue_lower": r.spa_pvalue_lower,
            "spa_pvalue_consistent": r.spa_pvalue_consistent, "spa_pvalue_upper": r.spa_pvalue_upper,
            "best_candidate": "|".join(str(x) for x in best) if isinstance(best, (tuple, list)) else str(best),
            "best_mean_bps_per_day": r.best_mean * 1e4,
        })
    return pd.DataFrame(rows)


def threshold_cost(curve: pd.DataFrame, alpha: float = 0.05, column: str = "spa_pvalue_consistent") -> dict:
    """Custo-limite: maior custo da grade com p <= alpha (``None`` se nem a custo 0 rejeita)."""
    c = curve.sort_values("cost_bps_per_side")
    p = c[column].to_numpy()
    cost = c["cost_bps_per_side"].to_numpy()
    rejected = p <= alpha
    return {
        "threshold_cost_bps": float(cost[rejected].max()) if rejected.any() else None,
        "p_at_zero_cost": float(p[0]),
        "rejects_at_zero_cost": bool(rejected[0]),
        "monotone": bool(np.all(np.diff(p) >= -1e-12)),
        "n_decreases": int((np.diff(p) < -1e-12).sum()),
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Tabelas por candidata e por ano
# ---------------------------------------------------------------------------


def _cbe(gross_sum: np.ndarray, units_sum: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(gross_sum <= 0, 0.0, np.where(units_sum > 0, 1e4 * gross_sum / np.where(units_sum > 0, units_sum, 1.0), np.nan))


def candidate_table(
    gross: np.ndarray,
    units: np.ndarray,
    trades: np.ndarray,
    keys: list,
    key_names: list[str],
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Uma linha por candidata: CBE, bruto, líquido em custos de referência, operações e Sharpe bruto."""
    m = np.ones(len(gross), bool) if mask is None else mask
    g, u, t = gross[m], units[m], trades[m]
    n = int(m.sum())
    gs, us = g.sum(axis=0), u.sum(axis=0)
    sd = g.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(sd > 0, g.mean(axis=0) / sd * np.sqrt(365.0), 0.0)
    out = pd.DataFrame(keys, columns=key_names)
    out["n_days"] = n
    out["gross_pct"] = (np.exp(gs) - 1.0) * 100.0
    out["cbe_bps"] = _cbe(gs, us)
    out["trades_per_day"] = t.sum(axis=0) / n
    out["units_per_day"] = us / n
    out["gross_sharpe"] = sharpe
    for c in READING_COSTS:
        out[f"net_pct_at_{c:g}bps"] = (np.exp(gs - c * us / 1e4) - 1.0) * 100.0
    return out


def by_year_table(
    dates: pd.DatetimeIndex, gross: np.ndarray, units: np.ndarray, trades: np.ndarray, keys: list, key_names: list[str]
) -> pd.DataFrame:
    frames = []
    for year in sorted(set(dates.year)):
        tab = candidate_table(gross, units, trades, keys, key_names, mask=np.asarray(dates.year == year))
        tab.insert(0, "year", year)
        frames.append(tab)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Parâmetros fixos (cada combinação numa trajetória contínua)
# ---------------------------------------------------------------------------


def fixed_param_daily(
    df_feat: pd.DataFrame,
    strategy: str,
    window: int,
    k: float,
    entry_z: float,
    first_idx: int,
    n_days: int,
    bars_per_day: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bruto, unidades e entradas por dia de uma combinação fixa, a custo 0 e sem fechamento diário.

    A posição corre desde o início dos dados; só os ``n_days`` dias a partir de ``first_idx`` contam.
    """
    z = compute_band_z(df_feat, strategy, window, k).to_numpy(dtype=float)
    position_signal = reversal_position_array(z, entry_z, entry_z, 0.0, 0.0)
    signal = np.zeros(len(z))
    signal[1:] = position_signal[:-1]
    arr = backtest_arrays(df_feat["open"].to_numpy(float), signal, 0.0, 0.0, 1.0, True, None)
    sl = slice(first_idx, first_idx + n_days * bars_per_day)
    entered = ((arr["position"] != arr["prev"]) & (arr["position"] != 0))[sl].astype(float)

    def per_day(x: np.ndarray) -> np.ndarray:
        return x[sl].reshape(n_days, bars_per_day).sum(axis=1)

    return per_day(arr["gross"]), per_day(arr["units"]), entered.reshape(n_days, bars_per_day).sum(axis=1)


# ---------------------------------------------------------------------------
# Perfil das entradas por hora e dia da semana
# ---------------------------------------------------------------------------

N_CELLS = 7 * 24  # dia da semana (segunda = 0) x hora UTC
ROUND_TRIP_UNITS = 2.0  # uma operação = 1 unidade na entrada + 1 na saída


def accumulate_profile(
    entry_ts: pd.DatetimeIndex,
    trade_gross: np.ndarray,
    first_day: pd.Timestamp,
    n_days: int,
    costs: tuple[float, ...] = READING_COSTS,
) -> dict[str, np.ndarray]:
    """Contagens por (dia de teste, dia da semana x hora UTC) das entradas e das que deram lucro.

    ``trade_gross`` é o retorno log bruto da operação; a custo c por lado o líquido é
    ``bruto - 2c/1e4``. Devolve matrizes (n_days, 168): ``n``, ``pos_gross``, ``pos_c<custo>``, ``sum_gross``.
    """
    ts = pd.DatetimeIndex(entry_ts)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")  # a tabela de operações guarda UTC sem fuso
    first = pd.Timestamp(first_day)
    first = first.tz_localize("UTC") if first.tz is None else first.tz_convert("UTC")
    day = np.asarray((ts.normalize() - first.normalize()).days)
    keep = (day >= 0) & (day < n_days)
    cell = (ts.dayofweek.to_numpy() * 24 + ts.hour.to_numpy())
    g = np.asarray(trade_gross, dtype=float)
    key = (day * N_CELLS + cell)[keep]
    g = g[keep]

    def acc(weights: np.ndarray | None) -> np.ndarray:
        return np.bincount(key, weights=weights, minlength=n_days * N_CELLS).reshape(n_days, N_CELLS)

    out = {"n": acc(None), "pos_gross": acc((g > 0).astype(float)), "sum_gross": acc(g)}
    for c in costs:
        out[f"pos_c{c:g}"] = acc((g - ROUND_TRIP_UNITS * c / 1e4 > 0).astype(float))
    return out


def _marginals(arr: np.ndarray) -> dict[str, np.ndarray]:
    """(dias, 168) -> hora (dias, 24), dia da semana (dias, 7) e célula (dias, 168)."""
    cube = arr.reshape(arr.shape[0], 7, 24)
    return {"hour": cube.sum(axis=1), "weekday": cube.sum(axis=2), "cell": arr}


def profile_tables(
    arrays: dict[str, np.ndarray],
    counts: np.ndarray | None = None,
    min_n: int = 30,
    ci: float = 95.0,
) -> dict[str, pd.DataFrame]:
    """Tabelas por hora, dia da semana e célula: entradas, % com lucro, bruto médio e CBE da faixa.

    Com ``counts`` (contagens do bootstrap por dias, de ``bootstrap_counts``) acrescenta o intervalo
    de confiança da % com lucro e do bruto médio por operação.
    """
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    marg = {name: _marginals(a) for name, a in arrays.items()}
    pos_cols = [c for c in arrays if c.startswith("pos_")]
    result = {}
    for level, labels in (
        ("hour", [f"{h:02d}" for h in range(24)]),
        ("weekday", ["seg", "ter", "qua", "qui", "sex", "sab", "dom"]),
        ("cell", [f"{d}-{h:02d}" for d in ("seg", "ter", "qua", "qui", "sex", "sab", "dom") for h in range(24)]),
    ):
        tot = {name: marg[name][level].sum(axis=0) for name in arrays}
        n = tot["n"]
        with np.errstate(divide="ignore", invalid="ignore"):
            df = pd.DataFrame({"bin": labels, "n_entries": n})
            for col in pos_cols:
                df[f"share_{col}"] = np.where(n >= min_n, tot[col] / n, np.nan)
            mean_g = np.where(n >= min_n, 1e4 * tot["sum_gross"] / n, np.nan)
            df["mean_gross_bps"] = mean_g
            df["cbe_bps"] = np.where(n >= min_n, np.where(tot["sum_gross"] > 0, mean_g / ROUND_TRIP_UNITS, 0.0), np.nan)
            if counts is not None:
                cn = counts @ marg["n"][level]
                for col, name in (("pos_gross", "share_pos_gross"), ("sum_gross", "mean_gross_bps")):
                    boot = (counts @ marg[col][level]) / cn
                    if col == "sum_gross":
                        boot = boot * 1e4
                    df[f"{name}_ci_low"] = np.where(n >= min_n, np.nanpercentile(boot, lo, axis=0), np.nan)
                    df[f"{name}_ci_high"] = np.where(n >= min_n, np.nanpercentile(boot, hi, axis=0), np.nan)
        result[level] = df
    return result


def half_split_agreement(arrays: dict[str, np.ndarray], split_day: int) -> pd.DataFrame:
    """Correlação de Spearman do perfil (hora e dia da semana) entre a 1ª e a 2ª metade dos dias."""
    from scipy.stats import spearmanr

    rows = []
    marg = {name: _marginals(a) for name, a in arrays.items()}
    for level in ("hour", "weekday"):
        for stat in ("share_pos_gross", "mean_gross_bps"):
            vals = []
            for part in (slice(0, split_day), slice(split_day, None)):
                n = marg["n"][level][part].sum(axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    if stat == "share_pos_gross":
                        vals.append(marg["pos_gross"][level][part].sum(axis=0) / n)
                    else:
                        vals.append(1e4 * marg["sum_gross"][level][part].sum(axis=0) / n)
            ok = np.isfinite(vals[0]) & np.isfinite(vals[1])
            rho, p = spearmanr(vals[0][ok], vals[1][ok]) if ok.sum() >= 4 else (np.nan, np.nan)
            rows.append({"level": level, "statistic": stat, "n_bins": int(ok.sum()), "spearman": float(rho), "p_value": float(p)})
    return pd.DataFrame(rows)
