#!/usr/bin/env python3
"""Script 18 — Diagnóstico de parâmetros fixos do estudo v2 (PREREGISTRATION_v2.md, seção 7).

Cada combinação (estratégia, janela, k) roda com parâmetros fixos numa trajetória contínua, a custo
0, por ativo e frequência: 3 x (144 + 144 + 192) = 1.440 candidatas, nos mesmos dias de teste do
walk-forward. A posição corre desde o início dos dados; só os dias de teste contam.

Saídas em ``results/v2/analysis/``: ``fixed_candidates.csv`` (CBE, bruto, líquido, operações),
``fixed_spa_curves.csv`` e ``fixed_spa_threshold.csv`` (família inteira e por frequência, por
ativo e por estratégia), ``fixed_daily.npz`` (matrizes diárias).

Uso: python scripts/18_fixed_params_v2.py [--reps 5000]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.config import filter_dataframe_by_date_range, load_config
from src.fixed_params import grid_tasks
from src.indicators import compute_all_features
from src.utils import setup_logger
from src.v2_study import (
    COST_GRID, FREQS, bootstrap_counts, candidate_table, fixed_param_daily, markdown_table, spa_curve, threshold_cost,
)

logger = setup_logger("18_fixed_params_v2")
ROOT = Path(__file__).parent.parent
MINUTES = {"1m": 1, "5m": 5, "15m": 15}
KEY_FIXED = ["freq", "symbol", "strategy", "window", "k"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results/v2/analysis")
    parser.add_argument("--reps", type=int, default=5000)
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    keys, gross_cols, unit_cols, trade_cols = [], [], [], []
    t0 = time.time()
    for freq in FREQS:
        cfg = load_config(ROOT / f"config.v2_{freq}.yaml")
        wf = cfg["walk_forward"]
        bpd = 1440 // MINUTES[freq]
        for symbol in cfg["data"]["symbols"]:
            df = pd.read_parquet(ROOT / cfg["data"]["processed_dir"] / f"{symbol}_{freq}.parquet")
            df, _ = filter_dataframe_by_date_range(df, cfg["data"]["start_date"], cfg["data"]["end_date"])
            df_feat = compute_all_features(df, cfg, strategy_names=cfg["strategies"])
            ts = pd.DatetimeIndex(df_feat["timestamp"])
            first_idx = int(ts.searchsorted(pd.Timestamp(wf["oos_start"], tz="UTC")))
            last_day = pd.Timestamp(cfg["data"]["end_date"], tz="UTC") - pd.Timedelta(days=1)
            n_days = int((last_day - pd.Timestamp(wf["oos_start"], tz="UTC")).days) + 1
            assert ts[first_idx].hour == 0 and ts[first_idx].minute == 0
            assert first_idx + n_days * bpd <= len(df_feat)
            for strategy, struct, thresh in grid_tasks(cfg["strategies"], cfg):
                g, u, t = fixed_param_daily(
                    df_feat, strategy, struct["window"], struct["k"], thresh["entry_long_z"], first_idx, n_days, bpd
                )
                keys.append((freq, symbol, strategy, int(struct["window"]), float(struct["k"])))
                gross_cols.append(g)
                unit_cols.append(u)
                trade_cols.append(t)
            logger.info("%s %s: %d combinações acumuladas (%.0f s).", freq, symbol, len(keys), time.time() - t0)
            del df, df_feat

    gross, units, trades = (np.column_stack(c) for c in (gross_cols, unit_cols, trade_cols))
    n = gross.shape[0]
    np.savez_compressed(out / "fixed_daily.npz", gross=gross, units=units, trades=trades,
                        keys=np.array([list(map(str, k)) for k in keys]))
    cand = candidate_table(gross, units, trades, keys, KEY_FIXED)
    cand.to_csv(out / "fixed_candidates.csv", index=False)

    counts = bootstrap_counts(n, args.reps, 10.0, 0)
    arr = np.array(keys, dtype=object)
    fam = {"all": np.arange(len(keys))}
    for pos, name in ((0, "freq"), (1, "asset"), (2, "strategy")):
        for value in sorted(set(arr[:, pos])):
            fam[f"{name}={value}"] = np.flatnonzero(arr[:, pos] == value)
    curves, thresholds = [], []
    for name, cols in fam.items():
        curve = spa_curve(gross[:, cols], units[:, cols], COST_GRID, counts=counts, keys=[keys[i] for i in cols])
        curve.insert(0, "family", name)
        curves.append(curve)
        thresholds.append({"family": name, "n_candidates": len(cols), **threshold_cost(curve)})
    pd.concat(curves, ignore_index=True).to_csv(out / "fixed_spa_curves.csv", index=False)
    thr = pd.DataFrame(thresholds)
    thr.to_csv(out / "fixed_spa_threshold.csv", index=False)

    lines = ["# Parâmetros fixos v2: resumo", "", f"{len(keys)} combinações, {n} dias de teste.", "",
             "## Custo-limite por família", "", markdown_table(thr, ".4f"), "",
             "## 10 maiores CBE", "", markdown_table(cand.nlargest(10, "cbe_bps"), ".3f"), "",
             "## Maior CBE por frequência e estratégia", "",
             markdown_table(cand.groupby(["freq", "strategy"])["cbe_bps"].agg(["max", "mean"]).reset_index(), ".3f"), ""]
    (out / "fixed_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Pronto em %.0f s.", time.time() - t0)


if __name__ == "__main__":
    main()
