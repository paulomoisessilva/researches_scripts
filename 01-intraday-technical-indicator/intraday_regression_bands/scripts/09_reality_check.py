#!/usr/bin/env python3
"""Script 09 — Reality Check (White) e SPA (Hansen) sobre as candidatas das runs.

Candidatas: cada combinação (ativo, estratégia, lookback = validation_days) das runs em
``walk_forward.runs_dir`` com o custo primário. Os retornos são diários, líquidos de custo,
em log; o benchmark é caixa (retorno zero). H0: nenhuma candidata tem retorno esperado > 0.

Como as posições não dependem do custo, o retorno líquido a qualquer custo é
``bruto - turnover * custo``; por isso o teste é repetido numa grade de custos (0 = sinal
bruto). A distribuição nula vem de um bootstrap estacionário (blocos geométricos).

Gera em ``<pasta das runs>/../paper/``:
  reality_check.csv                    p-valores por família x custo
  reality_check_block_sensitivity.csv  família ``all`` no custo primário, por comprimento de bloco
  reality_check.md                     resumo em Markdown

Uso:
    python scripts/09_reality_check.py --config config.paper.yaml
    python scripts/09_reality_check.py --config config.paper.yaml --reps 10000 --block 20
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.config import load_config
from src.inference import (
    bootstrap_means,
    candidate_panels,
    daily_matrix,
    reality_check_spa,
    stationary_bootstrap_indices,
)
from src.utils import ensure_dir, setup_logger
from src.wf_analysis import list_runs, load_candidate_windows, to_markdown
from src.wf_runner import resolve_cost_scenarios, resolve_runs_root, resolve_train_days

logger = setup_logger("09_reality_check")

CANDIDATE_COLS = ["symbol", "strategy", "validation_days"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reality Check e SPA com bootstrap estacionário.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--runs-root", type=str, default=None)
    parser.add_argument("--train-days", type=int, default=None, help="Treino das runs (default: do config).")
    parser.add_argument("--costs-run", type=str, default=None, metavar="FEE:SLIP",
                        help="Custo das runs usadas (default: primeiro de study.cost_scenarios).")
    parser.add_argument("--costs", nargs="+", type=float, default=[0, 1, 2, 3, 4, 6, 8],
                        help="Grade de custos (fee+slippage por lado, bps) para reavaliar o retorno líquido.")
    parser.add_argument("--reps", type=int, default=5000, help="Repetições do bootstrap.")
    parser.add_argument("--block", type=float, default=10.0, help="Comprimento médio de bloco (dias).")
    parser.add_argument("--blocks-sensitivity", nargs="+", type=float, default=[1, 5, 10, 20, 40])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def label(key: tuple) -> str:
    return "|".join(f"{v}" if i < 2 else f"val{v}" for i, v in enumerate(key))


def families(keys: list[tuple]) -> dict[str, list[int]]:
    """Subconjuntos de candidatas testados: todas, por ativo, por estratégia e por lookback."""
    out: dict[str, list[int]] = {"all": list(range(len(keys)))}
    for pos, name in enumerate(("symbol", "strategy", "val")):
        values = sorted({k[pos] for k in keys})
        if len(values) < 2:
            continue
        for v in values:
            tag = f"{name}={v}"
            out[tag] = [i for i, k in enumerate(keys) if k[pos] == v]
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runs_root = Path(args.runs_root) if args.runs_root else resolve_runs_root(cfg)
    out_dir = runs_root.parent / "paper"
    ensure_dir(out_dir)

    train_days = resolve_train_days(cfg, [args.train_days] if args.train_days else None)[0]
    run_cost = resolve_cost_scenarios(cfg, [args.costs_run] if args.costs_run else None)[0]
    run_cost_total = run_cost[0] + run_cost[1]

    windows = load_candidate_windows(list_runs(runs_root), train_days, run_cost)
    panels = candidate_panels(windows, CANDIDATE_COLS)
    keys = sorted(panels)
    fams = families(keys)
    logger.info("%d candidatas (%s); treino=%s; custo das runs=%s+%s bps; %d reps; bloco médio %g",
                len(keys), ", ".join(sorted({k[0] for k in keys})), train_days, run_cost[0], run_cost[1],
                args.reps, args.block)

    rows = []
    t0 = time.time()
    n_days = None
    idx = None
    for cost in args.costs:
        d, _, dates = daily_matrix(panels, cost, keys)
        if idx is None:
            n_days = d.shape[0]
            idx = stationary_bootstrap_indices(
                n_days, args.reps, args.block, np.random.default_rng(args.seed)
            )
            logger.info("%d dias em comum (%s -> %s)", n_days, dates[0], dates[-1])
        boot = bootstrap_means(d, idx)
        for fam, cols in fams.items():
            r = reality_check_spa(d[:, cols], boot[:, cols])
            best = keys[cols[r.best_candidate]]
            rows.append({
                "family": fam,
                "cost_bps_per_side": cost,
                "n_candidates": r.n_candidates,
                "n_days": r.n_obs,
                "best_candidate": label(best),
                "best_mean_bps_per_day": r.best_mean * 1e4,
                "best_annualized_log_return_pct": r.best_mean * 365 * 100,
                "rc_statistic": r.rc_statistic,
                "rc_pvalue": r.rc_pvalue,
                "spa_statistic": r.spa_statistic,
                "spa_pvalue_lower": r.spa_pvalue_lower,
                "spa_pvalue_consistent": r.spa_pvalue_consistent,
                "spa_pvalue_upper": r.spa_pvalue_upper,
            })
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "reality_check.csv", index=False)
    logger.info("Testes por família x custo em %.0f s", time.time() - t0)

    # Sensibilidade ao comprimento do bloco (todas as candidatas, custo das runs).
    d_run, _, _ = daily_matrix(panels, run_cost_total, keys)
    sens_rows = []
    for block in args.blocks_sensitivity:
        bidx = stationary_bootstrap_indices(d_run.shape[0], args.reps, block, np.random.default_rng(args.seed))
        r = reality_check_spa(d_run, bootstrap_means(d_run, bidx))
        sens_rows.append({
            "mean_block_days": block, "cost_bps_per_side": run_cost_total,
            "rc_pvalue": r.rc_pvalue, "spa_pvalue_consistent": r.spa_pvalue_consistent,
            "spa_pvalue_upper": r.spa_pvalue_upper,
        })
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(out_dir / "reality_check_block_sensitivity.csv", index=False)

    # Markdown
    fam_order = list(fams)
    pv = result.pivot_table(index="family", columns="cost_bps_per_side", values="spa_pvalue_consistent").loc[fam_order]
    pv_rc = result.pivot_table(index="family", columns="cost_bps_per_side", values="rc_pvalue").loc[fam_order]
    best = result[result.family == "all"][
        ["cost_bps_per_side", "best_candidate", "best_mean_bps_per_day", "best_annualized_log_return_pct", "spa_pvalue_consistent"]
    ].set_index("cost_bps_per_side")
    md = [
        "# Reality Check e SPA (bootstrap estacionário)\n",
        f"{len(keys)} candidatas (ativo x estratégia x lookback), {n_days} dias, benchmark = caixa, "
        f"{args.reps} repetições, bloco médio {args.block:g} dias, runs com custo {run_cost[0]:g}+{run_cost[1]:g} bps.\n",
        "## p-valor SPA (consistente) por família e custo (bps por lado)\n",
        to_markdown(pv, floatfmt="{:.3f}"),
        "",
        "## p-valor Reality Check por família e custo\n",
        to_markdown(pv_rc, floatfmt="{:.3f}"),
        "",
        "## Melhor candidata (família `all`) por custo\n",
        to_markdown(best, floatfmt="{:.3f}"),
        "",
        f"## Sensibilidade ao bloco (família `all`, custo {run_cost_total:g} bps)\n",
        to_markdown(sens.set_index("mean_block_days"), floatfmt="{:.3f}"),
        "",
    ]
    (out_dir / "reality_check.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("Resultados em %s", out_dir)

    pd.set_option("display.width", 200)
    print("\nSPA (consistente) — p-valor por família e custo")
    print(pv.round(3).to_string())
    print("\nMelhor candidata (all):")
    print(best.round(3).to_string())


if __name__ == "__main__":
    main()
