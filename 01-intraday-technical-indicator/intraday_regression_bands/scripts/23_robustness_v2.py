#!/usr/bin/env python3
"""Script 23 — Robustez do estudo v2 (descritivo).

1. O SPA global das 126 candidatas sem cada ano de teste, ativo, frequência, estratégia ou lookback (``spa_leave_one_out.csv``).
2. Os parâmetros fixos (1.440 combinações, ``fixed_daily.npz`` do script 18): quais têm retorno líquido positivo a 5 bps, e se isso se repete
   na 1ª e na 2ª metade dos dias e em quantos anos (``fixed_persistence.csv`` e ``fixed_persistence_summary.json``).

Uso: python scripts/23_robustness_v2.py [--reps 5000]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.utils import setup_logger
from src.v2_robustness import fixed_param_persistence, leave_one_out_spa
from src.v2_study import KEY, daily_matrices, load_windows, markdown_table

logger = setup_logger("23_robustness_v2")
ROOT = Path(__file__).parent.parent
HALF = 987


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=5000)
    args = parser.parse_args()
    t0 = time.time()
    out = ROOT / "results" / "v2" / "analysis" / "robustness"
    out.mkdir(parents=True, exist_ok=True)

    win = load_windows(ROOT / "results" / "v2")
    m = daily_matrices(win)
    loo = leave_one_out_spa(m["dates"], m["gross"], m["units"], m["keys"], KEY, reps=args.reps)
    loo.to_csv(out / "spa_leave_one_out.csv", index=False)
    logger.info("SPA sem cada parte pronto (%.0f s).", time.time() - t0)

    fixed = np.load(ROOT / "results" / "v2" / "analysis" / "fixed_daily.npz", allow_pickle=True)
    keys = [tuple(k) for k in fixed["keys"].tolist()]
    key_names = ["freq", "symbol", "strategy", "window", "k"]
    keys = [(a, b, c, int(d), float(e)) for a, b, c, d, e in keys]
    dates = m["dates"]
    table, summary = fixed_param_persistence(fixed["gross"], fixed["units"], fixed["trades"], keys, key_names, HALF, np.asarray(dates.year))
    table.to_csv(out / "fixed_persistence.csv", index=False)
    (out / "fixed_persistence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pos = table[table["net_all_bps"] > 0]
    lines = ["# Robustez (descritivo)", "", "## SPA global (126 candidatas) sem cada parte: p consistente por custo (bps por lado)", "",
             markdown_table(loo[["left_out", "n_days", "n_candidates", "p_cost_0", "p_cost_0.25", "p_cost_0.5", "p_cost_1", "threshold_cost_bps"]], ".4f"), "",
             "## Parâmetros fixos com retorno líquido positivo a 5 bps", "", json.dumps(summary, indent=2), "",
             "Por frequência e ativo, entre as que têm líquido positivo no período inteiro:", "",
             markdown_table(pos.groupby(["freq", "symbol"]).agg(n=("window", "size"), median_trades=("trades_all", "median"),
                                                                 pos_h1=("net_h1_bps", lambda s: int((s > 0).sum())), pos_h2=("net_h2_bps", lambda s: int((s > 0).sum())),
                                                                 median_years_positive=("years_positive", "median")).reset_index(), ".1f"), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Pronto em %.0f s: %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
