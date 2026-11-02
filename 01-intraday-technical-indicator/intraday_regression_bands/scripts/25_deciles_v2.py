#!/usr/bin/env python3
"""Script 25 — Acerto e bruto por decil de volume e de volatilidade da última hora (estudo v2, descritivo).

Refaz, com decis, a análise por tercil do script 24. Lê ``results/v2/analysis/trades.parquet`` (script 20) e os minutos de
``data/processed_v2`` e grava em ``results/v2/analysis/gross_regime/``:

* ``deciles_by_group.csv``  por frequência x ativo x estratégia x variável x decil: operações, acerto, ganho e perda médios, bruto médio
                            (erro-padrão agrupado por semana), mediana e volatilidade média;
* ``deciles_pooled.csv``    o mesmo por frequência x variável x decil, juntando os seis grupos ativo x estratégia de cada frequência
                            (os decis são calculados dentro de cada grupo);
* ``deciles_extremes.csv``  por grupo e variável, a diferença do bruto médio entre o decil 10 e o decil 1 e o p-valor;
* ``deciles_summary.md``.

Erros-padrão agrupados por semana. Descreve a amostra; nada é inferido para fora dela.

Uso: python scripts/25_deciles_v2.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.utils import setup_logger
from src.v2_behavior import extreme_bins_difference, gross_by_quantile, prepare_trades
from src.v2_study import markdown_table
from src.v2_trades import market_features

logger = setup_logger("25_deciles_v2")
ROOT = Path(__file__).parent.parent
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
VARIABLES = {"volume": "z_qv", "volatility": "z_rv"}
BINS = 10


def pooled(by_group: pd.DataFrame) -> pd.DataFrame:
    """Junta os grupos de cada frequência, por decil, com pesos pelo número de operações (acerto, bruto, volatilidade), de ganhos (ganho
    médio) e de perdas (perda média)."""
    rows = []
    for (freq, var, b), s in by_group.groupby(["freq", "variable", "bin"]):
        n, nw, nl = s["n"].sum(), s["n_win"].sum(), s["n_loss"].sum()
        rows.append({"freq": freq, "variable": var, "bin": b, "n": n, "hit_pct": 100 * nw / n,
                     "mean_win": float((s["mean_win"] * s["n_win"]).sum() / nw), "mean_loss": float((s["mean_loss"] * s["n_loss"]).sum() / nl),
                     "mean_gross": float((s["mean_gross"] * s["n"]).sum() / n), "mean_rv_bps": float((s["mean_rv_bps"] * s["n"]).sum() / n)})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    out = ROOT / "results" / "v2" / "analysis" / "gross_regime"
    out.mkdir(parents=True, exist_ok=True)
    trades = pd.read_parquet(ROOT / "results" / "v2" / "analysis" / "trades.parquet")
    trades["strategy"] = trades["strategy"].astype(str)
    feats = {a: market_features(pd.read_parquet(ROOT / "data" / "processed_v2" / f"{a}_1m.parquet", columns=["timestamp", "close", "quote_volume"]))
             for a in ASSETS}

    by_group, extremes = [], []
    for (freq, asset), t in trades.groupby(["freq", "asset"]):
        g = prepare_trades(t, feats[asset], require_features=True)
        for strategy, gs in g.groupby("strategy"):
            for var, col in VARIABLES.items():
                tab = gross_by_quantile(gs, col, BINS)
                by_group.append(tab.assign(freq=freq, asset=asset, strategy=strategy, variable=var))
                extremes.append({"freq": freq, "asset": asset, "strategy": strategy, "variable": var,
                                 "hit_d1": float(tab["hit_pct"].iloc[0]), "hit_d10": float(tab["hit_pct"].iloc[-1]),
                                 "gross_d1": float(tab["mean_gross"].iloc[0]), "gross_d10": float(tab["mean_gross"].iloc[-1]),
                                 **extreme_bins_difference(gs, col, BINS)})
        logger.info("%s %s pronto (%.0f s).", freq, asset, time.time() - t0)

    bg = pd.concat(by_group, ignore_index=True)
    pl = pooled(bg)
    ex = pd.DataFrame(extremes)
    bg.to_csv(out / "deciles_by_group.csv", index=False)
    pl.to_csv(out / "deciles_pooled.csv", index=False)
    ex.to_csv(out / "deciles_extremes.csv", index=False)

    lines = ["# Acerto e bruto por decil de volume e de volatilidade (descritivo)", "",
             f"{len(trades):,} operações; decis calculados dentro de cada grupo ativo x frequência x estratégia; erros-padrão agrupados por semana.".replace(",", "."), ""]
    for var in VARIABLES:
        p = pl[pl.variable == var]
        lines += [f"## {var}: grupos juntos por frequência (D1 = menor)", "", markdown_table(p[["freq", "bin", "n", "hit_pct", "mean_win", "mean_loss", "mean_gross", "mean_rv_bps"]], ".2f"), ""]
        e = ex[ex.variable == var]
        lines += [f"Em {len(e)} grupos: o acerto do D10 é menor que o do D1 em {int((e.hit_d10 < e.hit_d1).sum())}; o bruto médio do D10 é maior que o do D1 em "
                  f"{int((e.diff_bps > 0).sum())} (significativo em {int(((e.diff_bps > 0) & (e.p_value < .05)).sum())}) e menor em {int((e.diff_bps < 0).sum())} "
                  f"(significativo em {int(((e.diff_bps < 0) & (e.p_value < .05)).sum())}); mediana da diferença {e.diff_bps.median():+.2f} bps.", ""]
    (out / "deciles_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Pronto em %.0f s: %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
