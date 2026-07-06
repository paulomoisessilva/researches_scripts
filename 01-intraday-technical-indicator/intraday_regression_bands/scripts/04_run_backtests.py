#!/usr/bin/env python3
"""Script 04 — Execução dos backtests para todas as estratégias × custos.

Lê OHLCV de data/processed/, recalcula features conforme o config ativo
e salva resultados em data/backtests/.

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
from src.config import (
    filter_dataframe_by_date_range,
    load_config,
    get_project_root,
    resolve_date_range,
    resolve_strategies,
)
from src.indicators import compute_all_features
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
    parser.add_argument("--start-date", type=str, default=None,
                        help="Início do período (default: config.data.start_date).")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Fim do período (default: config.data.end_date).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    bt_cfg = cfg["backtest"]
    root = get_project_root()

    processed_dir = root / data_cfg["processed_dir"]
    backtests_dir = root / "data" / "backtests"
    tables_dir = root / "results" / "tables"
    ensure_dir(backtests_dir)
    ensure_dir(tables_dir)

    syms = args.symbols or data_cfg["symbols"]
    ints = args.intervals or data_cfg["intervals"]
    strategy_filter = resolve_strategies(cfg, args.strategies)
    start_date, end_date = resolve_date_range(cfg, args.start_date, args.end_date)

    fee_list = bt_cfg.get("fee_bps_per_side", [0, 2, 5, 10])
    slip_list = bt_cfg.get("slippage_bps_per_side", [0, 2, 5, 10])

    logger.info("=" * 60)
    logger.info("BACKTESTS")
    logger.info("Período: %s → %s", start_date, end_date)
    logger.info("Estratégias: %s", list(strategy_filter))
    logger.info("Custos fee × slippage: %s × %s bps", fee_list, slip_list)
    logger.info("=" * 60)

    all_metrics: list[dict] = []

    for symbol in syms:
        for interval in ints:
            proc_path = processed_dir / f"{symbol}_{interval}.parquet"
            if not proc_path.exists():
                logger.warning(
                    "Dados processados não encontrados: %s — rode o script 02 primeiro.",
                    proc_path.name,
                )
                continue

            logger.info("Carregando OHLCV: %s %s...", symbol, interval)
            df = pd.read_parquet(proc_path)

            df, _ = filter_dataframe_by_date_range(df, start_date, end_date)
            logger.info("Candles no período: %d", len(df))

            logger.info("Calculando features conforme config ativo...")
            df = compute_all_features(df, cfg, strategy_names=list(strategy_filter))

            signals_df = compute_all_signals(df, cfg)
            available = [s for s in strategy_filter if s in signals_df.columns]
            missing = [s for s in strategy_filter if s not in signals_df.columns]
            if missing:
                logger.warning("Sinais indisponíveis (features/params): %s", missing)
            logger.info("Sinais prontos: %s", available)

            for strategy_name in strategy_filter:
                if strategy_name not in signals_df.columns:
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
                            gross_returns=result.gross_returns,
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
