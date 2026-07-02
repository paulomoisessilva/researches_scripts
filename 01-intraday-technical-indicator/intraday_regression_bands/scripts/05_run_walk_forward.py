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

from src.config import load_config, get_project_root
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
    parser.add_argument("--fee-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    root = get_project_root()

    features_dir = root / "data" / "features"
    backtests_dir = root / "data" / "backtests"
    tables_dir = root / "results" / "tables"
    ensure_dir(backtests_dir)
    ensure_dir(tables_dir)

    syms = args.symbols or data_cfg["symbols"]
    ints = args.intervals or data_cfg["intervals"]
    strats = args.strategies or list(STRATEGY_REGISTRY.keys())

    logger.info("=" * 60)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("Estratégias: %s", strats)
    logger.info("Fee: %.1f bps | Slippage: %.1f bps", args.fee_bps, args.slippage_bps)
    logger.info("=" * 60)

    all_wf_results = []

    for symbol in syms:
        for interval in ints:
            feat_path = features_dir / f"{symbol}_{interval}_features.parquet"
            if not feat_path.exists():
                logger.warning("Features não encontradas: %s", feat_path.name)
                continue

            df = pd.read_parquet(feat_path)
            logger.info("Walk-forward: %s %s (%d candles)", symbol, interval, len(df))

            for strategy in strats:
                logger.info("  Estratégia: %s", strategy)
                results = run_walk_forward(
                    df=df,
                    strategy_name=strategy,
                    cfg=cfg,
                    symbol=symbol,
                    interval=interval,
                    fee_bps=args.fee_bps,
                    slippage_bps=args.slippage_bps,
                )
                all_wf_results.extend(results)
                logger.info("  -> %d janelas concluídas.", len(results))

    save_walk_forward_results(all_wf_results, tables_dir, backtests_dir)
    logger.info("Walk-forward concluído. %d janelas totais.", len(all_wf_results))


if __name__ == "__main__":
    main()
