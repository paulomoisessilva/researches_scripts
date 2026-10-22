#!/usr/bin/env python3
"""Script 17 — Análise principal do estudo v2 (PREREGISTRATION_v2.md, seções 5, 6 e 8).

Lê as runs de ``results/v2/{1m,5m,15m}/walk_forward`` e escreve em ``results/v2/analysis/``:

* ``candidates.csv``      126 candidatas: CBE, bruto, líquido em custos de referência, operações, Sharpe bruto;
* ``by_year.csv``         as mesmas métricas por ano;
* ``cbe_by_group.csv``    maior CBE por frequência, ativo, estratégia e lookback;
* ``spa_curves.csv``      p-valor do RC e do SPA por custo (0 a 8 bps em passos de 0,25), por família;
* ``spa_threshold.csv``   custo-limite de cada família;
* ``spa_block_sensitivity.csv``  curva primária com bloco médio de 5 e 20 dias;
* ``summary.md``          resumo.

Família primária: as 126 candidatas. Famílias por frequência, ativo e estratégia: descritivas.

Uso: python scripts/17_analyze_v2.py [--reps 5000]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.utils import setup_logger
from src.v2_study import (
    COST_GRID, KEY, READING_COSTS, bootstrap_counts, by_year_table, candidate_table, daily_matrices, markdown_table,
    load_windows, spa_curve, threshold_cost,
)

logger = setup_logger("17_analyze_v2")
ROOT = Path(__file__).parent.parent


def families(keys: list) -> dict[str, np.ndarray]:
    """Posições das colunas de cada família (chave = (freq, ativo, estratégia, lookback))."""
    arr = np.array(keys, dtype=object)
    fam = {"all": np.arange(len(keys))}
    for pos, name in ((0, "freq"), (1, "asset"), (2, "strategy")):
        for value in sorted(set(arr[:, pos])):
            fam[f"{name}={value}"] = np.flatnonzero(arr[:, pos] == value)
    return fam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results/v2")
    parser.add_argument("--out", default="results/v2/analysis")
    parser.add_argument("--reps", type=int, default=5000)
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    win = load_windows(ROOT / args.results)
    m = daily_matrices(win)
    keys, dates = m["keys"], m["dates"]
    gross, units, trades = m["gross"], m["units"], m["trades"]
    logger.info("%d candidatas, %d dias de teste (%s a %s).", len(keys), len(dates), dates[0].date(), dates[-1].date())

    cand = candidate_table(gross, units, trades, keys, KEY)
    cand.to_csv(out / "candidates.csv", index=False)
    year = by_year_table(dates, gross, units, trades, keys, KEY)
    year.to_csv(out / "by_year.csv", index=False)

    group_rows = []
    for col in ("freq", "symbol", "strategy", "lookback"):
        g = cand.groupby(col).agg(n_candidates=("cbe_bps", "size"), max_cbe_bps=("cbe_bps", "max"),
                                  mean_cbe_bps=("cbe_bps", "mean"), share_gross_positive=("gross_pct", lambda s: float((s > 0).mean())))
        g.insert(0, "group", col)
        group_rows.append(g.reset_index().rename(columns={col: "value"}))
    pd.concat(group_rows, ignore_index=True).to_csv(out / "cbe_by_group.csv", index=False)

    logger.info("Bootstrap (%d repetições, bloco médio de 10 dias)...", args.reps)
    counts = bootstrap_counts(len(dates), args.reps, 10.0, 0)
    curves, thresholds = [], []
    for name, cols in families(keys).items():
        sub_keys = [keys[i] for i in cols]
        curve = spa_curve(gross[:, cols], units[:, cols], COST_GRID, counts=counts, keys=sub_keys)
        curve.insert(0, "family", name)
        curves.append(curve)
        thresholds.append({"family": name, "n_candidates": len(cols), **threshold_cost(curve)})
    pd.concat(curves, ignore_index=True).to_csv(out / "spa_curves.csv", index=False)
    thr = pd.DataFrame(thresholds)
    thr.to_csv(out / "spa_threshold.csv", index=False)

    sens = []
    for block in (5.0, 20.0):
        c = spa_curve(gross, units, COST_GRID, counts=bootstrap_counts(len(dates), args.reps, block, 0), keys=keys)
        c.insert(0, "mean_block_days", block)
        sens.append(c)
    sens = pd.concat(sens, ignore_index=True)
    sens.to_csv(out / "spa_block_sensitivity.csv", index=False)

    # ----- resumo
    allc = pd.concat(curves, ignore_index=True)
    prim = allc[allc["family"] == "all"].set_index("cost_bps_per_side")
    lines = ["# Análise v2: resumo", "", f"{len(keys)} candidatas, {len(dates)} dias de teste "
             f"({dates[0].date()} a {dates[-1].date()}). Bootstrap estacionário, {args.reps} repetições, bloco médio de 10 dias.", ""]
    row = thr[thr["family"] == "all"].iloc[0]
    lines += ["## Teste primário (SPA global, 126 candidatas)", "",
              f"- p-valor SPA consistente a custo 0: {row['p_at_zero_cost']:.4f}",
              f"- custo-limite (maior custo da grade com p <= 0,05): "
              f"{'nenhum (não rejeita nem a custo zero)' if row['threshold_cost_bps'] is None or pd.isna(row['threshold_cost_bps']) else str(row['threshold_cost_bps']) + ' bps por lado'}",
              f"- curva monótona: {row['monotone']} (quedas: {row['n_decreases']})", "",
              "| custo (bps/lado) | RC | SPA inferior | SPA consistente | SPA superior | melhor candidata | melhor média (bps/dia) |",
              "|---|---|---|---|---|---|---|"]
    for c in sorted({0.0, 0.5, 1.0, 1.5, *READING_COSTS, 3.0, 4.0, 6.0, 8.0}):
        r = prim.loc[c]
        lines.append(f"| {c:g} | {r['rc_pvalue']:.4f} | {r['spa_pvalue_lower']:.4f} | {r['spa_pvalue_consistent']:.4f} | "
                     f"{r['spa_pvalue_upper']:.4f} | {r['best_candidate']} | {r['best_mean_bps_per_day']:.3f} |")
    lines += ["", "## Custo-limite por família (descritivas)", "", markdown_table(thr, ".4f"), "",
              "## Maior CBE por grupo", "", markdown_table(pd.concat(group_rows), ".3f"), "",
              "## 10 maiores CBE", "", markdown_table(cand.nlargest(10, "cbe_bps"), ".3f"), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Pronto em %.0f s. Saídas em %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
