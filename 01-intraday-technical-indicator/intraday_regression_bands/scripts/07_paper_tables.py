#!/usr/bin/env python3
"""Script 07 — Tabelas do paper a partir das runs em results/walk_forward/.

Gera em ``<pasta das runs>/../paper/`` (por padrão ``results/paper/``):
  oos_by_period_all_runs.csv   estratégias reais por ano (+ ALL), com posição frente ao random
  random_by_period_all_runs.csv distribuição do baseline aleatório entre seeds, por ano (+ ALL)
  sharpe_by_year_primary.csv   Sharpe estratégia x ano da run primária
  cbe_by_year_primary.csv      custo de break-even (bps/lado) estratégia x ano da run primária
  cbe_vs_random_primary.csv    Sharpe e CBE da estratégia frente ao placebo, por ano
  lookback_comparison.csv      Sharpe OOS (ALL) por janela de validação (lookback de seleção)
  lookback_comparison_cbe.csv  CBE OOS (ALL) por janela de validação
  cost_sensitivity.csv         Sharpe OOS (ALL) por cenário de custo (só com >1 custo rodado)
  paper_tables.md              as tabelas acima em Markdown

O CBE (retorno bruto / turnover, em bps por lado) resume a sensibilidade a custo: para
posições fixas o retorno líquido é linear no custo, então não é preciso varrer cenários.

A janela de validação é o lookback de seleção de parâmetros (só ela entra na pontuação;
o treino apenas aquece a estratégia). A run primária é a de ``walk_forward.train_days`` e
``validation_days`` do config e do primeiro cenário de ``study.cost_scenarios`` (ou use
--train-days/--validation-days/--costs).

Uso:
    python scripts/07_paper_tables.py --config config.paper.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils import ensure_dir, setup_logger
from src.wf_analysis import (
    ALL_PERIOD,
    collect_runs,
    list_runs,
    sharpe_pivot,
    to_markdown,
)
from src.wf_runner import (
    resolve_cost_scenarios,
    resolve_runs_root,
    resolve_train_days,
    resolve_validation_days,
)

logger = setup_logger("07_paper_tables")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera tabelas do paper (por ano e por run).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--runs-root", type=str, default=None,
                        help="Pasta com as runs (default: walk_forward.runs_dir do config).")
    parser.add_argument("--train-days", type=int, default=None,
                        help="Janela de treino da run primária (default: do config).")
    parser.add_argument("--validation-days", type=int, default=None,
                        help="Janela de validação (lookback de seleção) da run primária (default: do config).")
    parser.add_argument("--period", choices=["year", "quarter", "month"], default="year",
                        help="Granularidade das tabelas por período (default: year).")
    parser.add_argument("--costs", type=str, default=None, metavar="FEE:SLIP",
                        help="Custo da run primária (default: primeiro de study.cost_scenarios).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runs_root = Path(args.runs_root) if args.runs_root else resolve_runs_root(cfg)
    out_dir = runs_root.parent / "paper"  # results/paper (ou results/smoke/paper no smoke)
    ensure_dir(out_dir)

    run_dirs = list_runs(runs_root)
    if not run_dirs:
        raise SystemExit(f"Nenhuma run em {runs_root}. Rode scripts/05_run_walk_forward.py primeiro.")
    logger.info("%d run(s) encontradas em %s", len(run_dirs), runs_root)

    real, dist = collect_runs(run_dirs, period=args.period)
    period_label = {"year": "ano", "quarter": "trimestre", "month": "mês"}[args.period]
    if real.empty:
        raise SystemExit("Runs sem estratégias reais para tabular.")

    primary_train = resolve_train_days(cfg, [args.train_days] if args.train_days else None)[0]
    primary_val = resolve_validation_days(
        cfg, [args.validation_days] if args.validation_days else None
    )[0]
    primary_fee, primary_slip = resolve_cost_scenarios(
        cfg, [args.costs] if args.costs else None
    )[0]
    logger.info(
        "Run primária: treino=%s validação=%s | fee=%g slip=%g",
        primary_train, primary_val, primary_fee, primary_slip,
    )

    real.to_csv(out_dir / "oos_by_period_all_runs.csv", index=False)
    if not dist.empty:
        dist.to_csv(out_dir / "random_by_period_all_runs.csv", index=False)

    primary = {
        "train_days": primary_train,
        "validation_days": primary_val,
        "fee_bps": primary_fee,
        "slippage_bps": primary_slip,
    }
    cost_label = f"{primary_fee:g}+{primary_slip:g} bps/lado"
    multi_asset = real["symbol"].nunique() > 1
    if multi_asset:
        logger.info("Multi-ativo: %s", sorted(real["symbol"].unique()))
    md: list[str] = ["# Tabelas do paper (walk-forward OOS)\n"]

    by_year = sharpe_pivot(real, filters=primary)
    if by_year.empty:
        logger.warning("Run primária não encontrada (%s); tabela por ano omitida.", primary)
    else:
        by_year.to_csv(out_dir / f"sharpe_by_{args.period}_primary.csv")
        md += [
            f"## Sharpe OOS por {period_label} — validação {primary_val}d, custo {cost_label}\n",
            to_markdown(by_year),
            "",
        ]
        if not dist.empty:
            rnd = dist
            for col, val in primary.items():
                rnd = rnd[rnd[col] == val]
            sym = ["symbol"] if multi_asset else []
            rnd = rnd.assign(_all=rnd["period"] == ALL_PERIOD).sort_values(
                [*sym, "_all", "period"]
            ).set_index([*sym, "period"])[[
                "n_seeds", "sharpe_p05", "sharpe_p50", "sharpe_p95",
                "break_even_bps_p05", "break_even_bps_p50", "break_even_bps_p95",
            ]]
            md += ["### Placebo `random` (entre seeds; CBE em bps por lado)\n", to_markdown(rnd), ""]

    cbe_by_year = sharpe_pivot(real, values="break_even_bps", filters=primary)
    if not cbe_by_year.empty:
        cbe_by_year.to_csv(out_dir / f"cbe_by_{args.period}_primary.csv")
        md += [
            f"## Custo de break-even por {period_label} (bps por lado que zeram o retorno)\n",
            to_markdown(cbe_by_year),
            "",
        ]

    id_cols = [*(["symbol"] if multi_asset else []), "strategy", "period"]
    vs_random_cols = [
        *id_cols, "sharpe", "random_sharpe_p50", "pct_random_below_sharpe",
        "break_even_bps", "random_cbe_p50", "random_cbe_p95", "pct_random_below_cbe",
    ]
    prim_rows = real
    for col, val in primary.items():
        prim_rows = prim_rows[prim_rows[col] == val]
    if "pct_random_below_cbe" in prim_rows.columns and not prim_rows.empty:
        vs_random = prim_rows[vs_random_cols].sort_values(id_cols)
        vs_random.to_csv(out_dir / "cbe_vs_random_primary.csv", index=False)
        md += [
            "## Estratégias frente ao placebo (fração de seeds abaixo da estratégia)\n",
            to_markdown(vs_random.set_index(id_cols), floatfmt="{:.3f}"),
            "",
        ]

    all_period = real[real["period"] == ALL_PERIOD]

    # Lookback de seleção: uma coluna por janela de validação (treino e custo fixos).
    lookback_filters = {
        "train_days": primary_train, "fee_bps": primary_fee, "slippage_bps": primary_slip,
    }
    lookback_sharpe = sharpe_pivot(all_period, columns="validation_days", filters=lookback_filters)
    if not lookback_sharpe.empty:
        lookback_sharpe.to_csv(out_dir / "lookback_comparison.csv")
        md += [
            f"## Sharpe OOS por janela de validação = lookback de seleção (dias) — custo {cost_label}\n",
            to_markdown(lookback_sharpe),
            "",
        ]

    lookback_cbe = sharpe_pivot(
        all_period, columns="validation_days", values="break_even_bps", filters=lookback_filters
    )
    if not lookback_cbe.empty:
        lookback_cbe.to_csv(out_dir / "lookback_comparison_cbe.csv")
        md += [
            "## CBE OOS por janela de validação (dias, bps por lado)\n",
            to_markdown(lookback_cbe),
            "",
        ]

    cost_cmp = all_period[
        (all_period["train_days"] == primary_train) & (all_period["validation_days"] == primary_val)
    ].copy()
    if cost_cmp[["fee_bps", "slippage_bps"]].drop_duplicates().shape[0] < 2:
        cost_cmp = cost_cmp.iloc[0:0]  # um único custo: o CBE já resume a sensibilidade
    if not cost_cmp.empty:
        cost_cmp["custo_por_lado_bps"] = cost_cmp["fee_bps"] + cost_cmp["slippage_bps"]
        cost_pv = cost_cmp.pivot_table(
            index=["symbol", "strategy"] if multi_asset else "strategy",
            columns="custo_por_lado_bps", values="sharpe", aggfunc="first",
        )
        cost_pv.to_csv(out_dir / "cost_sensitivity.csv")
        md += [
            f"## Sharpe OOS por custo (fee+slippage por lado, bps) — validação {primary_val}d\n",
            to_markdown(cost_pv),
            "",
        ]

    (out_dir / "paper_tables.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("Tabelas salvas em %s", out_dir)


if __name__ == "__main__":
    main()
