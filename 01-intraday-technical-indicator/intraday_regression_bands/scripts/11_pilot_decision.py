#!/usr/bin/env python3
"""Script 11 — Critério de decisão do piloto (regra fixada antes dos resultados).

Reúne as runs de uma ou mais configs (por exemplo, fechamento no UTC e 24 h contínuo),
forma a família de candidatas (variante x estratégia x lookback), calcula o CBE de cada uma
e o SPA da família inteira, e aplica a regra de ``src/pilot_decision.py``.

Uso:
    python scripts/11_pilot_decision.py --configs config.pilot_1m.yaml config.pilot_1m_noeod.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.config import get_project_root, load_config
from src.inference import (
    bootstrap_means,
    candidate_panels,
    daily_matrix,
    reality_check_spa,
    stationary_bootstrap_indices,
)
from src.pilot_decision import DecisionRule, cbe_by_candidate, evaluate_rule
from src.utils import ensure_dir, setup_logger
from src.wf_analysis import list_runs, load_candidate_windows, to_markdown
from src.wf_runner import resolve_cost_scenarios, resolve_runs_root, resolve_train_days

logger = setup_logger("11_pilot_decision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aplica a regra de decisão do piloto.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--min-cbe", type=float, default=3.0, help="CBE mínimo (bps por lado).")
    parser.add_argument("--min-lookbacks", type=int, default=2, help="Lookbacks com CBE >= mínimo.")
    parser.add_argument("--spa-cost", type=float, default=2.0, help="Custo (bps/lado) em que o SPA é lido.")
    parser.add_argument("--max-p", type=float, default=0.10, help="Maior p-valor SPA aceito.")
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--block", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--costs", nargs="+", type=float, default=[0, 1, 2, 3, 4, 6, 8])
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rule = DecisionRule(args.min_cbe, args.min_lookbacks, args.spa_cost, args.max_p)

    frames = []
    first_cfg = None
    for cfg_path in args.configs:
        cfg = load_config(cfg_path)
        first_cfg = first_cfg or cfg
        train_days = resolve_train_days(cfg)[0]
        run_cost = resolve_cost_scenarios(cfg)[0]
        w = load_candidate_windows(list_runs(resolve_runs_root(cfg)), train_days, run_cost)
        w["variant"] = Path(cfg_path).stem
        frames.append(w)
    windows = pd.concat(frames, ignore_index=True)

    out_dir = Path(args.out_dir) if args.out_dir else resolve_runs_root(first_cfg).parent.parent / "pilot_decision"
    ensure_dir(out_dir)

    cbe = cbe_by_candidate(windows)
    cols = ["variant", "strategy", "validation_days"]
    panels = candidate_panels(windows, cols)
    keys = sorted(panels)

    rows = []
    d0, _, dates = daily_matrix(panels, args.costs[0], keys)
    idx = stationary_bootstrap_indices(d0.shape[0], args.reps, args.block, np.random.default_rng(args.seed))
    for cost in args.costs:
        d, _, _ = daily_matrix(panels, cost, keys)
        r = reality_check_spa(d, bootstrap_means(d, idx))
        best = keys[r.best_candidate]
        rows.append({
            "cost_bps_per_side": cost, "n_candidates": r.n_candidates, "n_days": r.n_obs,
            "best_candidate": "|".join(str(x) for x in best),
            "best_mean_bps_per_day": r.best_mean * 1e4,
            "rc_pvalue": r.rc_pvalue, "spa_pvalue_consistent": r.spa_pvalue_consistent,
        })
    spa = pd.DataFrame(rows)
    spa_at_rule = float(spa.loc[np.isclose(spa["cost_bps_per_side"], rule.spa_cost_bps), "spa_pvalue_consistent"].iloc[0])

    decision = evaluate_rule(cbe, spa_at_rule, rule)
    decision["n_days"] = int(d0.shape[0])
    decision["period"] = [str(dates[0]), str(dates[-1])]
    decision["bootstrap"] = {"reps": args.reps, "mean_block_days": args.block, "seed": args.seed}
    (out_dir / "pilot_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    cbe.to_csv(out_dir / "pilot_cbe_by_candidate.csv", index=False)
    spa.to_csv(out_dir / "pilot_spa_by_cost.csv", index=False)

    cbe_pv = cbe.pivot_table(index=["variant", "strategy"], columns="validation_days", values="cbe_bps")
    md = [
        "# Decisão do piloto\n",
        f"Regra (fixada antes): CBE >= {rule.min_cbe_bps:g} bps em >= {rule.min_lookbacks} lookbacks "
        f"de um mesmo método **e** SPA consistente <= {rule.max_spa_p:g} a {rule.spa_cost_bps:g} bps/lado.\n",
        f"{len(keys)} candidatas, {d0.shape[0]} dias ({dates[0]} a {dates[-1]}), bootstrap {args.reps} x bloco {args.block:g}.\n",
        "## CBE total por método e lookback (bps por lado)\n",
        to_markdown(cbe_pv, floatfmt="{:.2f}"),
        "",
        "## SPA da família por custo\n",
        to_markdown(spa.set_index("cost_bps_per_side"), floatfmt="{:.3f}"),
        "",
        "## Resultado\n",
        f"- Critério de CBE: **{'atendido' if decision['cbe_criterion_met'] else 'não atendido'}**",
        f"- Critério de SPA (p = {spa_at_rule:.3f} a {rule.spa_cost_bps:g} bps): "
        f"**{'atendido' if decision['spa_criterion_met'] else 'não atendido'}**",
        f"- **Escalar para o período maior: {'SIM' if decision['scale_up'] else 'NÃO'}**",
        "",
    ]
    (out_dir / "pilot_decision.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("Resultados em %s", out_dir)
    print("\n".join(md))


if __name__ == "__main__":
    main()
