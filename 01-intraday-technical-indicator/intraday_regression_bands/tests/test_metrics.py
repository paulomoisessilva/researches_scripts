"""Testes de métricas de performance."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest
from src.indicators import compute_all_features
from src.metrics import break_even_total_return_bps, total_return
from src.strategies import strategy_rsi


def _make_bt_cfg(fee_bps: float = 0.0, slippage_bps: float = 0.0) -> BacktestConfig:
    return BacktestConfig(
        strategy_name="rsi",
        symbol="BTCUSDT",
        interval="1m",
        initial_capital=10_000.0,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )


class TestBreakEvenTotalReturn:
    def test_zeros_total_return_at_break_even(self, sample_ohlcv, sample_config):
        df = compute_all_features(sample_ohlcv, sample_config)
        signal = strategy_rsi(df, window=14)
        result_zero = run_backtest(df, signal, _make_bt_cfg())

        bec = break_even_total_return_bps(result_zero.gross_returns, result_zero.positions)
        assert bec > 0

        result_be = run_backtest(df, signal, _make_bt_cfg(fee_bps=bec, slippage_bps=0.0))
        assert total_return(result_be.equity) == pytest.approx(0.0, abs=1e-4)

    def test_non_positive_gross_returns_zero(self):
        gross = pd.Series([0.0, -0.01, -0.02])
        positions = pd.Series([0.0, 1.0, 0.0])
        assert break_even_total_return_bps(gross, positions) == 0.0

    def test_no_position_changes_is_nan(self):
        gross = pd.Series([0.01, 0.01, 0.01])
        positions = pd.Series([0.0, 0.0, 0.0])
        assert np.isnan(break_even_total_return_bps(gross, positions))

    def test_one_side_cost_model(self):
        """CBE usa |Δpos| sem multiplicador ×2 automático."""
        gross = pd.Series([0.0, 0.01, 0.0])
        positions = pd.Series([0.0, 1.0, 0.0])  # 2 lados, gross = 1%
        bec = break_even_total_return_bps(gross, positions)
        assert bec == pytest.approx(0.01 * 10_000 / 2)
