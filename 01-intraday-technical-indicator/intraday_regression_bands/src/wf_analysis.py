"""Tabelas do paper a partir das runs walk-forward (``results/walk_forward/<run_id>/``).

Tudo é reagregado a partir dos momentos por janela gravados em
``walk_forward_results.csv`` (``sum_ret``, ``sum_sq_ret``, ``sum_gross_ret``,
``turnover_units``, ``n_bars``, ``test_date``), então não depende das curvas de
equity (que podem ser omitidas para o baseline aleatório).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import annualization_factor

MOMENT_COLUMNS = ("test_date", "n_bars", "sum_ret", "sum_sq_ret", "sum_gross_ret", "turnover_units")
ALL_PERIOD = "ALL"
_RANDOM = "random"


# ---------------------------------------------------------------------------
# Momentos -> métricas
# ---------------------------------------------------------------------------


def sharpe_from_moments(n: float, total: float, total_sq: float, interval: str) -> float:
    """Sharpe anualizado igual a ``metrics.sharpe_ratio`` (std com ddof=1)."""
    if n < 2:
        return float("nan")
    mean = total / n
    var = (total_sq - total * total / n) / (n - 1)
    if var <= 0:
        return 0.0
    factor = annualization_factor(interval)
    return float(mean * np.sqrt(factor) / np.sqrt(var))


def metrics_from_moments(grp: pd.DataFrame, interval: str) -> dict[str, float]:
    """Métricas OOS de um conjunto de janelas (soma dos momentos)."""
    n = float(grp["n_bars"].sum())
    s = float(grp["sum_ret"].sum())
    ss = float(grp["sum_sq_ret"].sum())
    gross = float(grp["sum_gross_ret"].sum())
    units = float(grp["turnover_units"].sum())
    if gross <= 0:
        betc = 0.0
    elif units == 0:
        betc = float("nan")
    else:
        betc = gross * 10_000 / units
    return {
        "n_days": int(grp["test_date"].nunique()),
        "n_bars": int(n),
        "total_return": float(np.exp(s) - 1.0),
        "annualized_return": float(s / n * annualization_factor(interval)) if n else float("nan"),
        "sharpe": sharpe_from_moments(n, s, ss, interval),
        "turnover_units": units,
        "number_of_trades": int(grp["number_of_trades"].sum()) if "number_of_trades" in grp else 0,
        "break_even_bps": betc,
    }


def _with_period(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = pd.to_datetime(out["test_date"]).dt.year.astype("Int64")
    return out


def _grouped_metrics(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Métricas por (keys..., period) com o período ``ALL`` além de um por ano."""
    rows: list[dict] = []
    for key_vals, grp in df.groupby(keys, dropna=False):
        key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        base = dict(zip(keys, key_vals))
        interval = str(grp["interval"].iloc[0])
        rows.append({**base, "period": ALL_PERIOD, **metrics_from_moments(grp, interval)})
        for year, ygrp in grp.groupby("year"):
            rows.append({**base, "period": str(int(year)), **metrics_from_moments(ygrp, interval)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Uma run
# ---------------------------------------------------------------------------


def load_run(run_dir: str | Path) -> tuple[dict, pd.DataFrame]:
    """Lê ``run_meta.json`` e as métricas por janela de uma run."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    df = pd.read_csv(run_dir / "walk_forward_results.csv")
    missing = [c for c in MOMENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{run_dir.name}: faltam colunas {missing} em walk_forward_results.csv "
            "(run gerada por versão antiga). Rode de novo o script 05 com --fresh."
        )
    return meta, df


def run_tables(df_windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabelas de uma run: estratégias reais por período e distribuição do random.

    Retorna ``(real, random_dist)``:
      - ``real``: uma linha por (strategy, symbol, interval, period) com as métricas,
        mais a posição frente ao placebo (``random_sharpe_p05/p50/p95`` e
        ``pct_random_below_sharpe`` = fração de seeds com Sharpe menor).
      - ``random_dist``: por (symbol, interval, period), distribuição entre seeds.
    """
    df = _with_period(df_windows)
    keys = ["symbol", "interval"]

    real = df[df["strategy"] != _RANDOM]
    real_tbl = _grouped_metrics(real, ["strategy", *keys]) if not real.empty else pd.DataFrame()

    rnd = df[df["strategy"] == _RANDOM]
    if rnd.empty or "seed" not in rnd.columns or not rnd["seed"].notna().any():
        return real_tbl, pd.DataFrame()

    per_seed = _grouped_metrics(rnd, [*keys, "seed"])
    dist_rows: list[dict] = []
    for (symbol, interval, period), grp in per_seed.groupby([*keys, "period"]):
        row = {"symbol": symbol, "interval": interval, "period": period, "n_seeds": len(grp)}
        for col in ("sharpe", "total_return", "number_of_trades", "turnover_units"):
            vals = grp[col].astype(float)
            row[f"{col}_mean"] = float(vals.mean())
            for q in (0.05, 0.50, 0.95):
                row[f"{col}_p{int(q * 100):02d}"] = float(vals.quantile(q))
        dist_rows.append(row)
    dist_tbl = pd.DataFrame(dist_rows)

    if not real_tbl.empty:
        merged = real_tbl.merge(
            dist_tbl[[
                *keys, "period", "n_seeds",
                "sharpe_p05", "sharpe_p50", "sharpe_p95",
                "total_return_p50", "total_return_p95",
            ]].rename(columns={
                "sharpe_p05": "random_sharpe_p05",
                "sharpe_p50": "random_sharpe_p50",
                "sharpe_p95": "random_sharpe_p95",
                "total_return_p50": "random_total_return_p50",
                "total_return_p95": "random_total_return_p95",
            }),
            on=[*keys, "period"],
            how="left",
        )
        pct = []
        for _, r in merged.iterrows():
            seeds = per_seed[
                (per_seed["symbol"] == r["symbol"])
                & (per_seed["interval"] == r["interval"])
                & (per_seed["period"] == r["period"])
            ]
            pct.append(float((seeds["sharpe"] < r["sharpe"]).mean()) if len(seeds) else np.nan)
        merged["pct_random_below_sharpe"] = pct
        real_tbl = merged

    return real_tbl, dist_tbl


# ---------------------------------------------------------------------------
# Várias runs
# ---------------------------------------------------------------------------

_META_COLUMNS = ("run_id", "train_days", "validation_days", "test_days", "step_days",
                 "fee_bps", "slippage_bps")


def list_runs(runs_root: str | Path) -> list[Path]:
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return []
    return sorted(
        p for p in runs_root.iterdir()
        if p.is_dir() and (p / "run_meta.json").exists() and (p / "walk_forward_results.csv").exists()
    )


def collect_runs(run_dirs: Iterable[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatena as tabelas de todas as runs, anotando janela de treino e custo."""
    reals: list[pd.DataFrame] = []
    dists: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        meta, df = load_run(run_dir)
        real, dist = run_tables(df)
        for tbl, bucket in ((real, reals), (dist, dists)):
            if tbl.empty:
                continue
            tbl = tbl.copy()
            for i, col in enumerate(_META_COLUMNS):
                tbl.insert(i, col, meta.get(col))
            bucket.append(tbl)
    real_all = pd.concat(reals, ignore_index=True) if reals else pd.DataFrame()
    dist_all = pd.concat(dists, ignore_index=True) if dists else pd.DataFrame()
    return real_all, dist_all


def sharpe_pivot(
    real: pd.DataFrame,
    *,
    index: str = "strategy",
    columns: str = "period",
    values: str = "sharpe",
    filters: dict | None = None,
) -> pd.DataFrame:
    """Pivot de uma métrica (por padrão Sharpe: estratégia x período), com ``ALL`` por último."""
    sub = real
    for col, val in (filters or {}).items():
        sub = sub[sub[col] == val]
    if sub.empty:
        return pd.DataFrame()
    pv = sub.pivot_table(index=index, columns=columns, values=values, aggfunc="first")
    if ALL_PERIOD in pv.columns:
        pv = pv[[c for c in pv.columns if c != ALL_PERIOD] + [ALL_PERIOD]]
    return pv


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Tabela Markdown simples (evita depender de ``tabulate``)."""
    if df.empty:
        return "_(sem dados)_"
    frame = df.reset_index()
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    lines = [header, sep]
    for _, row in frame.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
