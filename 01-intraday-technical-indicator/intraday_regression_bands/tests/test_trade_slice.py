"""Testes de filtragem de trades ao fatiar resultados walk-forward."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest
from src.metrics import compute_metrics
from src.walk_forward import _slice_backtest_result


def _make_df(n: int = 10) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    log_rets = np.zeros(n)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": np.full(n, 100.0),
        "close": np.full(n, 100.0),
        "log_return": log_rets,
    })


def _run_with_signal(signal_values: list[float]) -> tuple:
    df = _make_df(len(signal_values))
    signal = pd.Series(signal_values, index=df.index)
    cfg = BacktestConfig(
        strategy_name="test",
        symbol="X",
        interval="1m",
        fee_bps=0.0,
        slippage_bps=0.0,
    )
    return df, run_backtest(df, signal, cfg)


class TestTradeSliceFiltering:
    def test_removes_trades_closed_before_window(self):
        # Trade entra t=1, sai t=2; janela começa em t=3
        df, result = _run_with_signal([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        start, end = df["timestamp"].iloc[3], df["timestamp"].iloc[7]
        sliced = _slice_backtest_result(result, start, end)
        assert sliced.trades.empty

    def test_keeps_trades_closed_inside_window(self):
        # Trade entra t=2, sai t=5; janela [t=3, t=7)
        df, result = _run_with_signal([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
        start, end = df["timestamp"].iloc[3], df["timestamp"].iloc[7]
        sliced = _slice_backtest_result(result, start, end)
        assert len(sliced.trades) == 1
        assert pd.to_datetime(sliced.trades["exit_timestamp"].iloc[0], utc=True) == pd.to_datetime(
            df["timestamp"].iloc[5], utc=True
        )

    def test_removes_trades_closed_after_window(self):
        # Trade 1: sai t=5 (dentro); trade 2: sai t=8 (fora)
        signal = [0, 0, 1, 1, 1, 0, 1, 1, 0, 0]
        df, result = _run_with_signal(signal)
        start, end = df["timestamp"].iloc[3], df["timestamp"].iloc[7]
        sliced = _slice_backtest_result(result, start, end)
        assert len(result.trades) == 2
        assert len(sliced.trades) == 1
        assert pd.to_datetime(sliced.trades["exit_timestamp"].iloc[0], utc=True) == pd.to_datetime(
            df["timestamp"].iloc[5], utc=True
        )

    def test_metrics_use_filtered_trades_only(self):
        signal = [0, 0, 1, 1, 1, 0, 1, 1, 0, 0]
        df, result = _run_with_signal(signal)
        start, end = df["timestamp"].iloc[3], df["timestamp"].iloc[7]
        sliced = _slice_backtest_result(result, start, end)

        full_metrics = compute_metrics(
            result.equity, result.returns, result.positions, result.trades, "1m"
        )
        slice_metrics = compute_metrics(
            sliced.equity, sliced.returns, sliced.positions, sliced.trades, "1m"
        )

        assert full_metrics["number_of_trades"] == 2
        assert slice_metrics["number_of_trades"] == 1
        assert slice_metrics["number_of_trades"] < full_metrics["number_of_trades"]
