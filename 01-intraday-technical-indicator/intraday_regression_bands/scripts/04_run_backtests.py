#!/usr/bin/env python3
"""Script 04 — Execução dos backtests para todas as estratégias × custos.

Lê features de data/features/ e salva resultados em data/backtests/.

Uso:
    python scripts/04_run_backtests.py
    python scripts/04_run_backtests.py --symbols BTCUSDT --intervals 5m
    python scripts/04_run_backtests.py --strategies bollinger rsi
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.backtest import run_all_backtests, BacktestConfig, run_backtest, save_backtest_results
from src.config import load_config, get_project_root
from src.metrics import compute_metrics, aggregate_metrics, save_metrics, save_performance_by_symbol_interval
from src.strategies import compute_all_signals, STRATEGY_REGISTRY
from src.utils import ensure_dir, setup_logger

logger = setup_logger("04_run_backtests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa backtests para todas as estratégias.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--intervals", nargs="+", default=None)
    parser.add_argument("--strategies", nargs="+", default=None,
                        help=f"Estratégias: {list(STRATEGY_REGISTRY.keys())}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    bt_cfg = cfg["backtest"]
    root = get_project_root()

    features_dir = root / "data" / "features"
    backtests_dir = root / "data" / "backtests"
    tables_dir = root / "results" / "tables"
    ensure_dir(backtests_dir)
    ensure_dir(tables_dir)

    syms = args.symbols or data_cfg["symbols"]
    ints = args.intervals or data_cfg["intervals"]
    strategy_filter = set(args.strategies) if args.strategies else set(STRATEGY_REGISTRY.keys())

    fee_list = bt_cfg.get("fee_bps_per_side", [0, 2, 5, 10])
    slip_list = bt_cfg.get("slippage_bps_per_side", [0, 2, 5, 10])

    logger.info("=" * 60)
    logger.info("BACKTESTS")
    logger.info("Estratégias: %s", list(strategy_filter))
    logger.info("Custos fee × slippage: %s × %s bps", fee_list, slip_list)
    logger.info("=" * 60)

    all_metrics: list[dict] = []

    for symbol in syms:
        for interval in ints:
            feat_path = features_dir / f"{symbol}_{interval}_features.parquet"
            if not feat_path.exists():
                logger.warning("Features não encontradas: %s — rode o script 03 primeiro.", feat_path.name)
                continue

            logger.info("Carregando features: %s %s...", symbol, interval)
            df = pd.read_parquet(feat_path)

            signals_df = compute_all_signals(df, cfg)
            logger.info("Sinais calculados para %d estratégias.", len(signals_df.columns) - 1)

            for strategy_name in strategy_filter:
                if strategy_name not in signals_df.columns:
                    logger.warning("Sinal '%s' não disponível.", strategy_name)
                    continue

                signal = signals_df[strategy_name]

                for fee in fee_list:
                    for slip in slip_list:
                        cfg_bt = BacktestConfig(
                            strategy_name=strategy_name,
                            symbol=symbol,
                            interval=interval,
                            initial_capital=bt_cfg.get("initial_capital", 10_000),
                            fee_bps=fee,
                            slippage_bps=slip,
                            allow_short=bt_cfg.get("allow_short", True),
                            max_position=bt_cfg.get("max_position", 1.0),
                            close_eod=bt_cfg.get("close_position_end_of_day", False),
                            max_holding_bars=bt_cfg.get("max_holding_bars"),
                        )

                        result = run_backtest(df, signal, cfg_bt)
                        save_backtest_results(result, backtests_dir)

                        m = compute_metrics(
                            result.equity,
                            result.returns,
                            result.positions,
                            result.trades,
                            interval=interval,
                            strategy_name=strategy_name,
                            symbol=symbol,
                            fee_bps=fee,
                            slippage_bps=slip,
                        )
                        all_metrics.append(m)

    if all_metrics:
        df_metrics = aggregate_metrics(all_metrics)
        save_metrics(df_metrics, tables_dir, "performance_summary.csv")
        save_performance_by_symbol_interval(df_metrics, tables_dir)
        logger.info("Métricas salvas: %d entradas.", len(df_metrics))

        # Resumo de trades
        trades_path = tables_dir / "trades_summary.csv"
        df_metrics[["strategy", "symbol", "interval", "fee_bps", "slippage_bps",
                    "number_of_trades", "hit_rate", "profit_factor",
                    "average_trade_return", "average_holding_time"]].to_csv(trades_path, index=False)
        logger.info("Resumo de trades salvo: %s", trades_path.name)

    logger.info("Backtests concluídos.")


if __name__ == "__main__":
    main()
