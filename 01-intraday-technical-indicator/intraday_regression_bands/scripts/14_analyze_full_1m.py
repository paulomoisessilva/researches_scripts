#!/usr/bin/env python3
"""Script 14 — Análise geral do estudo de 1 minuto (BTC, ETH, SOL; 2021-2025).

Lê as runs semestrais de ``13_run_full_1m.py`` e gera, em ``results/full_1m/analysis/``:

* ``candidates_P1.csv`` / ``candidates_P2.csv``: métricas das 18 candidatas (ativo x estratégia x
  lookback) na janela confirmatória P1 (2021-01-06 a 2024-12-31) e na completa P2;
* ``candidates_by_year.csv`` e ``pooled_by_symbol_year.csv``: por ano, por candidata e agregado;
* ``spa_by_family_P1.csv`` / ``spa_by_family_P2.csv`` e ``spa_block_sensitivity.csv``;
* ``decision.json``: a regra primária pré-registrada (``PREREGISTRATION_full_1m.md``) e a do piloto;
* ``pilot_consistency.json``: dias do BTC em comum com o piloto contínuo;
* ``cumulative_pooled.csv`` e ``full_1m_summary.md``.

Uso:
    python scripts/14_analyze_full_1m.py --config config.full_1m.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.config import get_project_root, load_config
from src.full_study import (
    CANDIDATE,
    block_sensitivity,
    candidate_table,
    candidate_table_by_period,
    compare_with_pilot,
    cumulative_series,
    load_windows,
    make_families,
    pooled_table,
    restrict,
    share_gross_positive,
    spa_grid,
)
from src.pilot_decision import DecisionRule, cbe_by_candidate, evaluate_rule
from src.utils import ensure_dir, setup_logger
from src.wf_analysis import to_markdown
from src.wf_runner import resolve_cost_scenarios, resolve_runs_root, resolve_train_days

logger = setup_logger("14_analyze_full_1m")

P1 = ("2021-01-06", "2024-12-31")
PILOT_DAYS = ("2025-01-06", "2025-06-29")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Análise geral do estudo de 1m completo.")
    parser.add_argument("--config", type=str, default="config.full_1m.yaml")
    parser.add_argument("--pilot-config", type=str, default="config.pilot_1m.yaml")
    parser.add_argument("--costs", nargs="+", type=float, default=[0, 1, 2, 3, 4, 6, 8])
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--block", type=float, default=10.0)
    parser.add_argument("--blocks-sensitivity", nargs="+", type=float, default=[1, 5, 10, 20, 40])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def decision_for(windows: pd.DataFrame, spa: pd.DataFrame, rule: DecisionRule) -> dict:
    """Aplica uma regra de decisão; o 'método' é ativo x estratégia (cada um com seus lookbacks)."""
    w = windows.assign(variant=windows["symbol"])
    cbe = cbe_by_candidate(w, keys=("variant", "strategy", "validation_days"))
    fam = spa[(spa["family"] == "todas") & np.isclose(spa["cost_bps_per_side"], rule.spa_cost_bps)]
    if fam.empty:
        raise SystemExit(f"SPA da família 'todas' a {rule.spa_cost_bps:g} bps não calculado; inclua em --costs.")
    return evaluate_rule(cbe, float(fam["spa_pvalue_consistent"].iloc[0]), rule)


def main() -> None:
    args = parse_args()
    root = get_project_root()
    cfg = load_config(args.config)
    train_days = resolve_train_days(cfg)[0]
    run_cost = resolve_cost_scenarios(cfg)[0]
    out_dir = Path(args.out_dir) if args.out_dir else resolve_runs_root(cfg).parent / "analysis"
    ensure_dir(out_dir)

    windows = load_windows(resolve_runs_root(cfg), train_days, run_cost)
    symbols = sorted(windows["symbol"].unique())
    strategies = sorted(windows["strategy"].unique())
    first, last = windows["test_date"].min(), windows["test_date"].max()
    logger.info("%d janelas | ativos %s | %s a %s", len(windows), symbols, first, last)

    p1 = restrict(windows, *P1)
    p2 = windows
    periods = {"P1": p1, "P2": p2}

    # --- insumos para auditoria: janelas diárias das candidatas e proveniência ---
    windows.drop(columns=["run"]).to_csv(out_dir / "candidate_windows.csv", index=False)
    prov = ensure_dir(out_dir / "provenance")
    runs_parent = resolve_runs_root(cfg).parent
    for src in (root / "PREREGISTRATION_full_1m.md", root / args.config, runs_parent / "PREREG_HASHES.txt",
                runs_parent / "data_hashes.txt", runs_parent / "progress.json"):
        if Path(src).exists():
            shutil.copy2(src, prov / Path(src).name)

    # --- candidatas e períodos ---
    for name, w in periods.items():
        candidate_table(w).to_csv(out_dir / f"candidates_{name}.csv", index=False)
    by_year = candidate_table_by_period(p2, "year")
    by_year.to_csv(out_dir / "candidates_by_year.csv", index=False)
    pooled_sy = pooled_table(p2, ["symbol"], "year")
    pooled_sy.to_csv(out_dir / "pooled_by_symbol_year.csv", index=False)
    pooled_y = pooled_table(p2.assign(family="todas"), ["family"], "year")
    pooled_y.to_csv(out_dir / "pooled_by_year.csv", index=False)
    pooled_m = pooled_table(p2, ["symbol"], "month")
    pooled_m.to_csv(out_dir / "pooled_by_symbol_month.csv", index=False)
    share = share_gross_positive(p2, "year")
    share.to_csv(out_dir / "share_gross_positive_by_year.csv", index=False)
    cumulative_series(p2).to_csv(out_dir / "cumulative_pooled.csv", index=False)
    for s in symbols:
        cumulative_series(p2[p2["symbol"] == s]).to_csv(out_dir / f"cumulative_pooled_{s}.csv", index=False)

    # --- RC / SPA ---
    families = make_families(symbols, strategies)
    spa = {}
    for name, w in periods.items():
        spa[name] = spa_grid(w, families, args.costs, reps=args.reps, block=args.block, seed=args.seed)
        spa[name].to_csv(out_dir / f"spa_by_family_{name}.csv", index=False)
    sens_costs = [c for c in (0.0, 4.0) if c in args.costs] or args.costs[:1]
    sens = pd.concat(
        [block_sensitivity(w, sens_costs, args.blocks_sensitivity, reps=args.reps, seed=args.seed).assign(period=n)
         for n, w in periods.items()],
        ignore_index=True,
    )
    sens.to_csv(out_dir / "spa_block_sensitivity.csv", index=False)

    # --- decisão ---
    primary = DecisionRule(min_cbe_bps=4.0, min_lookbacks=2, spa_cost_bps=4.0, max_spa_p=0.10)
    pilot_rule = DecisionRule(min_cbe_bps=3.0, min_lookbacks=2, spa_cost_bps=2.0, max_spa_p=0.10)
    decision = {
        "period_P1": list(P1),
        "period_P2": [first, last],
        "primary_rule_on_P1": decision_for(p1, spa["P1"], primary),
        "pilot_rule_on_P1": decision_for(p1, spa["P1"], pilot_rule),
        "pilot_rule_on_P2": decision_for(p2, spa["P2"], pilot_rule),
        "bootstrap": {"reps": args.reps, "mean_block_days": args.block, "seed": args.seed},
    }
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- consistência com o piloto contínuo ---
    consistency = None
    try:
        pcfg = load_config(args.pilot_config)
        pilot = load_windows(resolve_runs_root(pcfg), resolve_train_days(pcfg)[0], resolve_cost_scenarios(pcfg)[0])
        consistency = compare_with_pilot(windows, pilot, *PILOT_DAYS)
        (out_dir / "pilot_consistency.json").write_text(json.dumps(consistency, indent=2), encoding="utf-8")
    except (SystemExit, FileNotFoundError, KeyError) as exc:
        logger.warning("Comparação com o piloto não feita: %s", exc)

    # --- resumo em Markdown ---
    cand2 = candidate_table(p2)
    cand1 = candidate_table(p1)
    key = ["symbol", "strategy", "validation_days"]
    show = ["gross_pct", "net_pct", "cbe_bps", "trades_per_day", "sharpe_net"]
    md = [
        "# Estudo de 1 minuto completo (BTC, ETH, SOL)\n",
        f"Dias de teste {first} a {last}; {windows.groupby(CANDIDATE).ngroups} candidatas; custo "
        f"{run_cost[0]:g}+{run_cost[1]:g} bps por lado. P1 = {P1[0]} a {P1[1]} (nunca vista); P2 = período completo.\n",
        "## Candidatas em P1\n", to_markdown(cand1.set_index(key)[show], floatfmt="{:.2f}"), "",
        "## Candidatas em P2\n", to_markdown(cand2.set_index(key)[show], floatfmt="{:.2f}"), "",
        "## Carteira igual-pesos por ativo e ano\n", to_markdown(pooled_sy.set_index(["symbol", "period"]), floatfmt="{:.2f}"), "",
        "## Fração de candidatas com retorno bruto > 0\n", to_markdown(share.set_index("period"), floatfmt="{:.2f}"), "",
    ]
    for name in ("P1", "P2"):
        s = spa[name]
        piv = s[s["family"] == "todas"].set_index("cost_bps_per_side")[
            ["n_candidates", "n_days", "best_candidate", "best_mean_bps_per_day", "rc_pvalue",
             "spa_pvalue_lower", "spa_pvalue_consistent", "spa_pvalue_upper"]]
        md += [f"## RC/SPA, família completa, {name}\n", to_markdown(piv, floatfmt="{:.3f}"), ""]
        md += [f"## SPA consistente por família, {name}\n",
               to_markdown(s.pivot(index="family", columns="cost_bps_per_side", values="spa_pvalue_consistent"),
                           floatfmt="{:.3f}"), ""]
    md += ["## Sensibilidade ao bloco do bootstrap\n", to_markdown(sens.set_index(["period", "mean_block_days", "cost_bps_per_side"]), floatfmt="{:.3f}"), ""]
    pr = decision["primary_rule_on_P1"]
    md += [
        "## Decisão (regra primária pré-registrada, P1)\n",
        f"- CBE >= {primary.min_cbe_bps:g} bps em >= {primary.min_lookbacks} lookbacks de um mesmo método: "
        f"**{'atendido' if pr['cbe_criterion_met'] else 'não atendido'}**",
        f"- SPA consistente (família completa, {primary.spa_cost_bps:g} bps, p = {pr['spa_pvalue_at_rule_cost']:.3f}) "
        f"<= {primary.max_spa_p:g}: **{'atendido' if pr['spa_criterion_met'] else 'não atendido'}**",
        f"- **Estratégia lucrativa depois de custos: {'SIM' if pr['scale_up'] else 'NÃO'}**",
        "",
    ]
    if consistency:
        md += ["## Consistência com o piloto contínuo\n", "```", json.dumps(consistency, indent=2), "```", ""]
    (out_dir / "full_1m_summary.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("Resultados em %s", out_dir)
    print("\n".join(md))


if __name__ == "__main__":
    main()
