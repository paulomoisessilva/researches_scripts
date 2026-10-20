#!/usr/bin/env python3
"""Script 05 — Validação walk-forward out-of-sample.

Seleciona parâmetros pontuando só a janela de validação (o treino apenas aquece a
estratégia) e aplica no teste. Cada execução (janelas x cenário de custo) grava em
``results/walk_forward/<run_id>/`` (tabelas, equity e run_meta.json), sem
sobrescrever outras execuções.

Uso:
    # execução primária do config (primeiro cenário de study.cost_scenarios)
    python scripts/05_run_walk_forward.py --config config.paper.yaml

    # estudo de lookback: study.validation_days (x study.cost_scenarios)
    python scripts/05_run_walk_forward.py --config config.paper.yaml --study

    # lookbacks (janela de validação) e custos explícitos (fee:slippage em bps por lado)
    python scripts/05_run_walk_forward.py --config config.paper.yaml \
        --validation-days 7 14 28 --costs 2:2 5:5

    python scripts/05_run_walk_forward.py --strategies bollinger regression_bands
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    load_config,
    resolve_date_range,
    resolve_strategies,
)
from src.utils import setup_logger
from src.wf_runner import (
    prepare_inputs,
    resolve_cost_scenarios,
    resolve_train_days,
    resolve_validation_days,
    run_experiment,
)

logger = setup_logger("05_run_walk_forward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa validação walk-forward.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--intervals", nargs="+", default=None)
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--fee-bps", type=float, default=None,
                        help="Fee por lado em bps (um único cenário).")
    parser.add_argument("--slippage-bps", type=float, default=None,
                        help="Slippage por lado em bps (um único cenário).")
    parser.add_argument("--costs", nargs="+", default=None, metavar="FEE:SLIP",
                        help="Cenários de custo por lado em bps, ex.: 2:2 0:0 5:5.")
    parser.add_argument("--train-days", nargs="+", type=int, default=None,
                        help="Uma ou mais janelas de treino (dias); uma run por valor.")
    parser.add_argument("--validation-days", nargs="+", type=int, default=None,
                        help="Uma ou mais janelas de validação (dias) = lookback da seleção de "
                             "parâmetros; uma run por valor.")
    parser.add_argument("--study", action="store_true",
                        help="Usa study.validation_days (e train_days/cost_scenarios, se houver) do config.")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Nome da pasta da run (só com uma combinação janela x custo).")
    parser.add_argument("--tag", type=str, default=None,
                        help="Sufixo opcional anexado ao run_id gerado.")
    parser.add_argument("--fresh", action="store_true",
                        help="Apaga a pasta da run antes de executar (default: retoma/mescla).")
    parser.add_argument("--force-features", action="store_true",
                        help="Recalcula features mesmo se o parquet existir.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Workers para grid search (default: walk_forward.parallel_workers).")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Início do período (default: config.data.start_date).")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Fim do período (default: config.data.end_date).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.start_date:
        cfg["data"]["start_date"] = args.start_date
    if args.end_date:
        cfg["data"]["end_date"] = args.end_date
    data_cfg = cfg["data"]

    syms = args.symbols or data_cfg["symbols"]
    ints = args.intervals or data_cfg["intervals"]
    strats = resolve_strategies(cfg, args.strategies)
    start_date, end_date = resolve_date_range(cfg)

    train_days_list = resolve_train_days(cfg, args.train_days, args.study)
    validation_days_list = resolve_validation_days(cfg, args.validation_days, args.study)
    cost_scenarios = resolve_cost_scenarios(
        cfg, args.costs, args.fee_bps, args.slippage_bps, args.study
    )
    n_runs = len(train_days_list) * len(validation_days_list) * len(cost_scenarios)
    if args.run_id and n_runs > 1:
        raise SystemExit("--run-id só vale para uma única combinação de janelas e custo.")

    logger.info("Período: %s -> %s | estratégias: %s", start_date, end_date, strats)
    logger.info(
        "Treino: %s dias | validação (lookback de seleção): %s dias | custos (fee:slip): %s",
        train_days_list, validation_days_list, cost_scenarios,
    )
    logger.info("%d run(s) a executar.", n_runs)

    inputs = prepare_inputs(
        cfg, syms, ints, strats, start_date, end_date, force_features=args.force_features
    )

    run_dirs = []
    for train_days in train_days_list:
        for validation_days in validation_days_list:
            for fee_bps, slippage_bps in cost_scenarios:
                run_dir = run_experiment(
                    cfg,
                    inputs,
                    strats,
                    fee_bps,
                    slippage_bps,
                    train_days=train_days,
                    validation_days=validation_days,
                    run_id=args.run_id,
                    tag=args.tag,
                    workers=args.workers,
                    fresh=args.fresh,
                    config_path=args.config,
                )
                run_dirs.append(run_dir)

    logger.info("Concluído. Runs:")
    for run_dir in run_dirs:
        logger.info("  %s", run_dir)
    logger.info("Próximo passo: python scripts/07_paper_tables.py")


if __name__ == "__main__":
    main()
