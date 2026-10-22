#!/usr/bin/env python3
"""Script 21 — O que explica a chance de lucro das operações do estudo v2 (descritivo): calendário, volume, volatilidade e persistência.

Lê ``results/v2/analysis/trades.parquet`` (script 20) e os minutos de ``data/processed_v2`` e grava em ``results/v2/analysis/behavior/``:

* ``calendar_by_group.csv``       dia da semana, hora e sessões contra o lucro bruto, por ativo x frequência x estratégia;
* ``calendar_weekend_by_year.csv`` fim de semana contra dias úteis por ano;
* ``regime_by_group.csv``          volume e volatilidade (nível e surpresa), mediação do fim de semana e escala dos percentis;
* ``regime_deciles.csv``           % com lucro por decil de volume e de volatilidade;
* ``regime_threshold.csv``         forma da relação (ponto de quebra);
* ``regime_by_year.csv`` e ``regime_by_strategy.csv``;
* ``block_contributions.csv``      quanto explicam volume/volatilidade, dia da semana e hora, juntos;
* ``persistence_acf.csv``, ``persistence_seasonal.csv``, ``persistence_recent.csv``;
* ``summary.md``.

Grupos: ativo x frequência (as duas estratégias e os 7 lookbacks juntos, com efeitos fixos) ou ativo x frequência x estratégia (os 7
lookbacks juntos). Erros-padrão agrupados por semana. Tudo descreve a amostra; nada é inferido para fora dela.

Uso: python scripts/21_behavior_v2.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.utils import setup_logger
from src.v2_behavior import (
    acf_overlap_summary, block_contributions, calendar_summary, decile_table, prepare_trades, recent_performance_tests, regime_effects,
    seasonal_lag_tests, threshold_shape, weekend_by_year, weekend_mediation,
)
from src.v2_study import markdown_table
from src.v2_trades import market_features

logger = setup_logger("21_behavior_v2")
ROOT = Path(__file__).parent.parent
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
N_HOURS = 1974 * 24


def main() -> None:
    t0 = time.time()
    out = ROOT / "results" / "v2" / "analysis" / "behavior"
    out.mkdir(parents=True, exist_ok=True)
    trades = pd.read_parquet(ROOT / "results" / "v2" / "analysis" / "trades.parquet")
    trades["strategy"] = trades["strategy"].astype(str)
    feats = {a: market_features(pd.read_parquet(ROOT / "data" / "processed_v2" / f"{a}_1m.parquet", columns=["timestamp", "close", "quote_volume"]))
             for a in ASSETS}
    logger.info("%d operações; features de mercado calculadas (%.0f s).", len(trades), time.time() - t0)

    tables: dict[str, list[dict]] = {k: [] for k in (
        "calendar", "weekend_year", "regime", "deciles", "threshold", "block", "regime_year", "regime_strategy", "acf", "seasonal", "recent")}
    for (freq, asset), t in trades.groupby(["freq", "asset"]):
        tag = {"freq": freq, "asset": asset}
        g = prepare_trades(t, feats[asset], require_features=True)
        # ----- calendário, por estratégia
        for strategy, gs in g.groupby("strategy"):
            tables["calendar"].append({**tag, "strategy": strategy, **calendar_summary(gs)})
            for _, r in weekend_by_year(gs).iterrows():
                tables["weekend_year"].append({**tag, "strategy": strategy, **r.to_dict()})
            tables["regime_strategy"].append({**tag, "strategy": strategy, **regime_effects(gs)})
        # ----- volume e volatilidade, estratégias juntas
        tables["regime"].append({**tag, **regime_effects(g), **weekend_mediation(g),
                                 "qv_p10_millions": float(np.percentile(g["qv"], 10) / 1e6), "qv_p50_millions": float(np.percentile(g["qv"], 50) / 1e6),
                                 "qv_p90_millions": float(np.percentile(g["qv"], 90) / 1e6), "rv_p10_bps": float(np.percentile(g["rv"], 10)),
                                 "rv_p50_bps": float(np.percentile(g["rv"], 50)), "rv_p90_bps": float(np.percentile(g["rv"], 90))})
        for var, col in (("volume", "z_qv"), ("volatility", "z_rv")):
            tables["deciles"].append({**tag, "variable": var, **decile_table(g, col).to_dict()})
        for var, col, other in (("volume", "z_qv", "z_rv"), ("volatility", "z_rv", "z_qv")):
            tables["threshold"].append({**tag, "variable": var, **threshold_shape(g, col, other)})
        tables["block"].append({**tag, **block_contributions(g)})
        for year, gy in g.groupby("year"):
            if len(gy) >= 500:
                tables["regime_year"].append({**tag, "year": int(year), **regime_effects(gy)})
        # ----- persistência
        tables["acf"].append({**tag, **acf_overlap_summary(t, N_HOURS)})
        tables["seasonal"].append({**tag, **seasonal_lag_tests(g, N_HOURS)})
        tables["recent"].append({**tag, **recent_performance_tests(g, g["exit_ts"])})
        logger.info("%s %s pronto (%.0f s).", freq, asset, time.time() - t0)

    names = {"calendar": "calendar_by_group", "weekend_year": "calendar_weekend_by_year", "regime": "regime_by_group", "deciles": "regime_deciles",
             "threshold": "regime_threshold", "block": "block_contributions", "regime_year": "regime_by_year", "regime_strategy": "regime_by_strategy",
             "acf": "persistence_acf", "seasonal": "persistence_seasonal", "recent": "persistence_recent"}
    frames = {k: pd.DataFrame(v) for k, v in tables.items()}
    for k, df in frames.items():
        df.to_csv(out / f"{names[k]}.csv", index=False)

    cal, blk = frames["calendar"], frames["block"]
    lines = ["# Comportamento das operações (descritivo)", "",
             f"{len(trades):,} operações, 126 trajetórias, 1.974 dias de teste. Erros-padrão agrupados por semana.".replace(",", "."), "",
             "## Calendário por grupo (p de Wald; sábado contra segunda)", "",
             markdown_table(cal[["freq", "asset", "strategy", "n", "win_pct", "p_weekday", "p_hour", "p_weekday_given_hour", "p_hour_given_weekday",
                                 "OR_saturday", "p_saturday", "sat_minus_mon_bps"]], ".4f"), "",
             f"Grupos com p < 0,05 (de {len(cal)}): dia da semana {int((cal.p_weekday < .05).sum())}, hora {int((cal.p_hour < .05).sum())}, "
             f"sábado contra segunda {int((cal.p_saturday < .05).sum())}.", "",
             "## Volume e volatilidade (razão de chances por desvio padrão)", "",
             markdown_table(frames["regime"][["freq", "asset", "n", "OR_vol_level", "p_vol_level", "OR_volume_level", "p_volume_level",
                                              "OR_weekend_raw", "OR_weekend_controlled", "p_weekend_controlled"]], ".4f"), "",
             "## Contribuição única de cada bloco (% do ganho total)", "",
             markdown_table(blk[["freq", "asset", "unique_V_pct", "unique_W_pct", "unique_H_pct", "p_V", "p_W", "p_H",
                                 "oos_dLL_V", "oos_dLL_W", "oos_dLL_H"]], ".3f"), "",
             "## Persistência: autocorrelação do acerto por hora (14 trajetórias somadas, sem duplicatas, trajetória isolada)", "",
             markdown_table(frames["acf"][["freq", "asset", "pooled_1h", "dedup_1h", "single_1h_mean", "pooled_24h", "pooled_168h"]], ".3f"), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Pronto em %.0f s: %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
