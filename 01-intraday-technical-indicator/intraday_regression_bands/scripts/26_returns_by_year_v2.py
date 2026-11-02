#!/usr/bin/env python3
"""Script 26 — Estatísticas descritivas dos retornos por ativo, frequência e ano (estudo v2).

Lê ``data/processed_v2`` (candles de 1 minuto) e, para cada ativo, calcula o log-retorno de abertura a abertura,
``ln(open_t / open_{t-1})`` em pontos-base, nas barras de 1, 5 e 15 minutos (as de 5 e 15 saem da agregação das de 1 minuto, como no estudo).
Grava em ``results/v2/analysis/data_descriptive/``:

* ``returns_by_year.csv``  por frequência x ativo x ano: barras, mínimo, percentis 1, 25, 50, 75 e 99, média, máximo, desvio padrão,
                           assimetria e curtose em excesso;
* ``returns_by_year.md``   as mesmas tabelas, uma por frequência.

O período é o do estudo (4 de janeiro de 2021 a 1º de julho de 2026, UTC). Descreve a base; nada é inferido.

Uso: python scripts/26_returns_by_year_v2.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.utils import setup_logger
from src.v2_study import markdown_table

logger = setup_logger("26_returns_by_year_v2")
ROOT = Path(__file__).parent.parent
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
FREQS = {"1m": 1, "5m": 5, "15m": 15}
START, END = pd.Timestamp("2021-01-04", tz="UTC"), pd.Timestamp("2026-07-02", tz="UTC")
COLUMNS = ["bars", "min", "p01", "p25", "p50", "mean", "p75", "p99", "max", "std", "skew", "kurt"]


def describe(r: pd.Series) -> dict:
    q = r.quantile([.01, .25, .5, .75, .99])
    return {"bars": len(r), "min": r.min(), "p01": q[.01], "p25": q[.25], "p50": q[.5], "mean": r.mean(), "p75": q[.75], "p99": q[.99],
            "max": r.max(), "std": r.std(), "skew": r.skew(), "kurt": r.kurt()}


def returns_bps(opens: pd.Series, minutes: int) -> pd.Series:
    """Log-retorno de abertura a abertura, em bps, na barra de ``minutes`` minutos (a abertura da barra agregada é a da primeira barra de 1 minuto)."""
    o = opens if minutes == 1 else opens.resample(f"{minutes}min").first()
    return 1e4 * np.log(o / o.shift(1)).dropna()


def main() -> None:
    t0 = time.time()
    out = ROOT / "results" / "v2" / "analysis" / "data_descriptive"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for asset in ASSETS:
        df = pd.read_parquet(ROOT / "data" / "processed_v2" / f"{asset}_1m.parquet", columns=["timestamp", "open"])
        opens = df[(df.timestamp >= START) & (df.timestamp < END)].set_index("timestamp")["open"]
        for freq, minutes in FREQS.items():
            r = returns_bps(opens, minutes)
            for year, ry in r.groupby(r.index.year):
                rows.append({"freq": freq, "asset": asset, "year": int(year), **describe(ry)})
        logger.info("%s pronto (%.0f s).", asset, time.time() - t0)

    table = pd.DataFrame(rows)
    table.to_csv(out / "returns_by_year.csv", index=False)
    lines = ["# Log-retornos de abertura a abertura, por ativo e ano (bps; curtose em excesso)", ""]
    for freq in FREQS:
        t = table[table.freq == freq].drop(columns="freq")
        lines += [f"## Barras de {freq}", "", markdown_table(t[["asset", "year"] + COLUMNS], ".2f"), ""]
    (out / "returns_by_year.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Pronto em %.0f s: %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
