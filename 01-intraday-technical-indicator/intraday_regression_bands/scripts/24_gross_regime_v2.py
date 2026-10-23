#!/usr/bin/env python3
"""Script 24 — O bruto em bps (e não só o acerto) contra volume e volatilidade da última hora (estudo v2, descritivo).

Lê ``results/v2/analysis/trades.parquet`` (script 20) e os minutos de ``data/processed_v2`` e grava em
``results/v2/analysis/gross_regime/``:

* ``by_group.csv``     MQO do bruto contra volume e volatilidade, por ativo x frequência (as duas estratégias, efeitos fixos de lookback e
                       estratégia), em bps bruto, bps cortado (percentis 1 e 99) e unidades de volatilidade; metades e R2 fora da amostra;
* ``by_strategy.csv``  o mesmo por ativo x frequência x estratégia;
* ``deciles.csv``      bruto médio, mediano e médio cortado por decil de volume e de volatilidade, por ativo x frequência;
* ``terciles.csv``     acerto, ganho e perda médios e bruto médio por tercil, por ativo x frequência x estratégia;
* ``summary.md``.

Erros-padrão agrupados por semana. Descreve a amostra; nada é inferido para fora dela.

Uso: python scripts/24_gross_regime_v2.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.utils import setup_logger
from src.v2_behavior import gross_by_tercile, gross_deciles, gross_regime_effects, prepare_trades
from src.v2_study import markdown_table
from src.v2_trades import market_features

logger = setup_logger("24_gross_regime_v2")
ROOT = Path(__file__).parent.parent
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
VARIABLES = {"volume": "z_qv", "volatility": "z_rv"}


def main() -> None:
    t0 = time.time()
    out = ROOT / "results" / "v2" / "analysis" / "gross_regime"
    out.mkdir(parents=True, exist_ok=True)
    trades = pd.read_parquet(ROOT / "results" / "v2" / "analysis" / "trades.parquet")
    trades["strategy"] = trades["strategy"].astype(str)
    feats = {a: market_features(pd.read_parquet(ROOT / "data" / "processed_v2" / f"{a}_1m.parquet", columns=["timestamp", "close", "quote_volume"]))
             for a in ASSETS}

    by_group, by_strategy, deciles, terciles = [], [], [], []
    for (freq, asset), t in trades.groupby(["freq", "asset"]):
        tag = {"freq": freq, "asset": asset}
        g = prepare_trades(t, feats[asset], require_features=True)
        by_group.append({**tag, **gross_regime_effects(g)})
        for var, col in VARIABLES.items():
            deciles.append({**tag, "variable": var, **gross_deciles(g, col)})
        for strategy, gs in g.groupby("strategy"):
            by_strategy.append({"freq": freq, "asset": asset, "strategy": strategy, **gross_regime_effects(gs)})
            for var, col in VARIABLES.items():
                for _, row in gross_by_tercile(gs, col).iterrows():
                    terciles.append({"freq": freq, "asset": asset, "strategy": strategy, "variable": var, **row.to_dict()})
        logger.info("%s %s pronto (%.0f s).", freq, asset, time.time() - t0)

    frames = {"by_group": pd.DataFrame(by_group), "by_strategy": pd.DataFrame(by_strategy), "deciles": pd.DataFrame(deciles),
              "terciles": pd.DataFrame(terciles)}
    for name, df in frames.items():
        df.to_csv(out / f"{name}.csv", index=False)

    bg, bs, tc = frames["by_group"], frames["by_strategy"], frames["terciles"]
    n_pos = lambda df, b, p: f"{int(((df[p] < .05) & (df[b] > 0)).sum())} positivos, {int(((df[p] < .05) & (df[b] < 0)).sum())} negativos"
    vol = tc[tc.variable == "volume"].pivot_table(index=["freq", "asset", "strategy"], columns="tercile", values=["hit_pct", "mean_gross", "mean_win", "mean_loss"])
    vol.columns = [f"{a}_T{int(b)}" for a, b in vol.columns]
    lines = ["# Bruto em bps contra volume e volatilidade (descritivo)", "",
             f"{len(trades):,} operações; erros-padrão agrupados por semana. Coeficientes por desvio padrão do z da última hora.".replace(",", "."), "",
             "## Por ativo x frequência (as duas estratégias): coeficiente do volume e da volatilidade", "",
             markdown_table(bg[["freq", "asset", "n", "b_volume_w", "p_volume_w", "b_vol_w", "p_vol_w", "b_volume_pv", "p_volume_pv", "b_vol_pv", "p_vol_pv",
                                "b_volume_w_h1", "b_volume_w_h2", "oos_r2_w"]], ".4f"), "",
             f"Volume, bruto cortado, p < 0,05 (de {len(bg)}): {n_pos(bg, 'b_volume_w', 'p_volume_w')}. Volatilidade: {n_pos(bg, 'b_vol_w', 'p_vol_w')}. "
             f"Em unidades de volatilidade, volume: {n_pos(bg, 'b_volume_pv', 'p_volume_pv')}; volatilidade: {n_pos(bg, 'b_vol_pv', 'p_vol_pv')}.", "",
             "## Por tercil de volume (T1 = menor): acerto, ganho médio, perda média e bruto médio (bps), por grupo", "",
             markdown_table(vol.reset_index()[["freq", "asset", "strategy", "hit_pct_T1", "hit_pct_T2", "hit_pct_T3", "mean_win_T1", "mean_win_T3",
                                              "mean_loss_T1", "mean_loss_T3", "mean_gross_T1", "mean_gross_T2", "mean_gross_T3"]], ".2f"), "",
             "## Por estratégia: coeficiente do volume (bruto cortado)", "",
             markdown_table(bs[["freq", "asset", "strategy", "n", "b_volume_w", "p_volume_w", "b_vol_w", "p_vol_w", "oos_r2_w"]], ".4f"), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Pronto em %.0f s: %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
