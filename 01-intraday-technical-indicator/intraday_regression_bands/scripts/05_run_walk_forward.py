#!/usr/bin/env python3
"""Script 05 — Validação walk-forward out-of-sample.

Usa treino+validação para selecionar parâmetros e aplica no teste.
Resultados salvos em results/tables/ e data/backtests/.

Uso:
    python scripts/05_run_walk_forward.py
    python scripts/05_run_walk_forward.py --symbols BTCUSDT --intervals 5m
    python scripts/05_run_walk_forward.py --strategies bollinger regression_bands
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import (
    filter_dataframe_by_date_range,
    load_config,
    get_project_root,
    resolve_cost_bps,
    resolve_date_range,
    resolve_strategies,
)
from src.indicators import features_parquet_path, load_or_compute_features
from src.strategies import STRATEGY_REGISTRY
from src.utils import ensure_dir, setup_logger
from src.walk_forward import run_walk_forward, save_walk_forward_results

logger = setup_logger("05_run_walk_forward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa validação walk-forward.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--intervals", nargs="+", default=None)
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--fee-bps", type=float, default=None,
                        help="Fee por lado em bps (default: config.backtest.fee_bps_per_side[0]).")
    parser.add_argument("--slippage-bps", type=float, default=None,
                        help="Slippage por lado em bps (default: config.backtest.slippage_bps_per_side[0]).")
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Recalcula features mesmo se o parquet existir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Workers para grid search (default: walk_forward.parallel_workers no config).",
    )
    parser.add_argument("--start-date", type=str, default=None,
                        help="Início do período (default: config.data.start_date).")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Fim do período (default: config.data.end_date).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    root = get_project_root()

    processed_dir = root / data_cfg["processed_dir"]
    backtests_dir = root / "data" / "backtests"
    tables_dir = root / "results" / "tables"
    ensure_dir(backtests_dir)
    ensure_dir(tables_dir)

    syms = args.symbols or data_cfg["symbols"]
    ints = args.intervals or data_cfg["intervals"]
    strats = resolve_strategies(cfg, args.strategies)
    start_date, end_date = resolve_date_range(cfg, args.start_date, args.end_date)
    fee_bps, slippage_bps = resolve_cost_bps(cfg, args.fee_bps, args.slippage_bps)

    logger.info("=" * 60)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("Período: %s → %s", start_date, end_date)
    logger.info("Estratégias: %s", strats)
    logger.info("Fee: %.1f bps | Slippage: %.1f bps", fee_bps, slippage_bps)
    wf_cfg = cfg.get("walk_forward", {})
    logger.info(
        "WF janelas: train=%s + validation=%s + test=%s | step=%s | metric=%s | EOD=%s",
        wf_cfg.get("train_days"),
        wf_cfg.get("validation_days", wf_cfg.get("val_days")),
        wf_cfg.get("test_days"),
        wf_cfg.get("step_days"),
        wf_cfg.get("selection_metric", "net_sharpe"),
        wf_cfg.get("close_position_end_of_day", True),
    )
    workers = args.workers if args.workers is not None else wf_cfg.get("parallel_workers", 1)
    logger.info("Grid workers: %d", workers)
    logger.info("=" * 60)

    all_wf_results = []

    for symbol in syms:
        for interval in ints:
            proc_path = processed_dir / f"{symbol}_{interval}.parquet"
            if not proc_path.exists():
                logger.warning("Dados processados não encontrados: %s", proc_path.name)
                continue

            df = pd.read_parquet(proc_path)

            df, date_filtered = filter_dataframe_by_date_range(df, start_date, end_date)
            logger.info("Candles no período: %d", len(df))

            feat_path = features_parquet_path(root, symbol, interval)
            df_feat = load_or_compute_features(
                df,
                cfg,
                features_path=feat_path,
                force_compute=args.force_features or date_filtered,
                strategy_names=strats,
            )
            logger.info("Walk-forward: %s %s (%d candles)", symbol, interval, len(df_feat))

            for strategy in strats:
                logger.info("  Estratégia: %s", strategy)
                results = run_walk_forward(
                    df=df,
                    strategy_name=strategy,
                    cfg=cfg,
                    symbol=symbol,
                    interval=interval,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                    df_feat=df_feat,
                    n_workers=workers,
                )
                all_wf_results.extend(results)
                logger.info("  -> %d janelas concluídas.", len(results))

    save_walk_forward_results(all_wf_results, tables_dir, backtests_dir)
    logger.info("Walk-forward concluído. %d janelas totais.", len(all_wf_results))


if __name__ == "__main__":
    main()
