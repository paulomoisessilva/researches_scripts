"""Testes de execução next_open com preços open executáveis."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest
from src.metrics import break_even_total_return_bps, total_return
from src.strategies import _shift_signal


def _make_ohlc(opens: list[float], closes: list[float] | None = None) -> pd.DataFrame:
    n = len(opens)
    closes = opens if closes is None else closes
    timestamps = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    close_s = pd.Series(closes, dtype=float)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "close": closes,
        "log_return": np.log(close_s / close_s.shift(1)).fillna(0.0).values,
    })


class TestNextOpenExecution:
    def test_entry_at_open_t_plus_one(self):
        """Decisão em t=1 → entrada no open de t=2."""
        df = _make_ohlc([100.0, 100.0, 102.0, 105.0, 103.0])
        raw = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=df.index)
        signal = _shift_signal(raw)

        result = run_backtest(
            df, signal,
            BacktestConfig(strategy_name="test", symbol="X", interval="1m"),
        )

        assert result.positions.iloc[1] == 0.0
        assert result.positions.iloc[2] == 1.0
        assert result.trades.iloc[0]["entry_price"] == pytest.approx(102.0)
        assert result.trades.iloc[0]["entry_bar"] == 2

    def test_exit_at_open_after_signal(self):
        """Saída confirmada em t=3 → execução no open de t=4."""
        df = _make_ohlc([100.0, 100.0, 102.0, 105.0, 103.0])
        raw = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=df.index)
        signal = _shift_signal(raw)

        result = run_backtest(
            df, signal,
            BacktestConfig(strategy_name="test", symbol="X", interval="1m"),
        )

        assert result.trades.iloc[0]["exit_price"] == pytest.approx(103.0)
        assert result.trades.iloc[0]["exit_bar"] == 4

    def test_long_trade_return_from_open_prices(self):
        df = _make_ohlc([100.0, 100.0, 102.0, 105.0, 103.0])
        signal = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0], index=df.index)

        result = run_backtest(
            df, signal,
            BacktestConfig(strategy_name="test", symbol="X", interval="1m"),
        )

        expected = np.log(103.0 / 102.0)
        assert result.trades.iloc[0]["trade_return"] == pytest.approx(expected, abs=1e-12)
        assert result.gross_returns.sum() == pytest.approx(expected, abs=1e-12)

    def test_short_trade_return_from_open_prices(self):
        df = _make_ohlc([100.0, 100.0, 102.0, 98.0, 101.0])
        signal = pd.Series([0.0, 0.0, -1.0, -1.0, 0.0], index=df.index)

        result = run_backtest(
            df, signal,
            BacktestConfig(strategy_name="test", symbol="X", interval="1m"),
        )

        expected = np.log(102.0 / 101.0)
        assert result.trades.iloc[0]["trade_return"] == pytest.approx(expected, abs=1e-12)

    def test_no_return_before_execution_candle(self):
        df = _make_ohlc([100.0, 100.0, 110.0, 120.0, 120.0])
        signal = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0], index=df.index)

        result = run_backtest(
            df, signal,
            BacktestConfig(strategy_name="test", symbol="X", interval="1m"),
        )

        assert result.gross_returns.iloc[2] == pytest.approx(0.0, abs=1e-12)
        assert result.gross_returns.iloc[3] == pytest.approx(np.log(120.0 / 110.0), abs=1e-12)

    def test_no_double_lag(self):
        """Primeiro retorno capturado no candle após a entrada, não dois candles depois."""
        df = _make_ohlc([100.0, 100.0, 100.0, 110.0, 110.0])
        raw = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=df.index)
        signal = _shift_signal(raw)

        result = run_backtest(
            df, signal,
            BacktestConfig(strategy_name="test", symbol="X", interval="1m"),
        )

        assert result.gross_returns.iloc[2] == pytest.approx(0.0, abs=1e-12)
        assert result.gross_returns.iloc[3] == pytest.approx(np.log(110.0 / 100.0), abs=1e-12)

    def test_per_bar_gross_matches_trade_decomposition(self):
        df = _make_ohlc([100.0, 100.0, 102.0, 105.0, 103.0])
        signal = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0], index=df.index)

        result = run_backtest(
            df, signal,
            BacktestConfig(strategy_name="test", symbol="X", interval="1m"),
        )

        entry, exit_ = 2, 4
        bar_sum = result.gross_returns.iloc[entry + 1: exit_ + 1].sum()
        assert bar_sum == pytest.approx(result.trades.iloc[0]["trade_return"], abs=1e-12)

    def test_transaction_costs_one_side_each(self):
        df = _make_ohlc([100.0, 100.0, 100.0, 100.0])
        signal = pd.Series([0.0, 1.0, 1.0, 0.0], index=df.index)
        cfg = BacktestConfig(
            strategy_name="test", symbol="X", interval="1m",
            fee_bps=10.0, slippage_bps=0.0,
        )
        result = run_backtest(df, signal, cfg)
        assert result.returns.sum() == pytest.approx(-2 * (10.0 / 10_000), abs=1e-12)

    def test_flip_costs_two_sides(self):
        df = _make_ohlc([100.0, 100.0, 100.0, 100.0])
        signal = pd.Series([0.0, 1.0, -1.0, -1.0], index=df.index)
        cfg = BacktestConfig(
            strategy_name="test", symbol="X", interval="1m",
            fee_bps=10.0, slippage_bps=0.0,
        )
        result = run_backtest(df, signal, cfg)
        flip_cost = 2 * (10.0 / 10_000)
        assert result.returns.iloc[2] == pytest.approx(-flip_cost, abs=1e-12)

    def test_betc_zeros_net_profit(self):
        df = _make_ohlc([100.0, 100.0, 110.0, 120.0, 120.0])
        signal = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0], index=df.index)
        gross_result = run_backtest(
            df, signal,
            BacktestConfig(strategy_name="test", symbol="X", interval="1m"),
        )
        bec = break_even_total_return_bps(gross_result.gross_returns, gross_result.positions)
        net_result = run_backtest(
            df, signal,
            BacktestConfig(
                strategy_name="test", symbol="X", interval="1m",
                fee_bps=bec, slippage_bps=0.0,
            ),
        )
        assert total_return(net_result.equity) == pytest.approx(0.0, abs=1e-6)

    def test_eod_closes_at_last_bar_open(self):
        """EOD: saída forçada no open do último candle UTC (não no dia seguinte)."""
        n = 5
        timestamps = pd.date_range("2025-01-01 12:00", periods=n, freq="1min", tz="UTC")
        opens = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "close": opens,
            "log_return": np.zeros(n),
        })
        signal = pd.Series(1.0, index=df.index)
        cfg = BacktestConfig(
            strategy_name="test", symbol="X", interval="1m", close_eod=True,
        )
        result = run_backtest(df, signal, cfg)

        assert result.positions.iloc[-1] == 0.0
        assert result.trades.iloc[-1]["exit_bar"] == n - 1
        assert result.trades.iloc[-1]["exit_price"] == pytest.approx(104.0)

    def test_ignores_close_to_close_when_open_differs(self):
        """Retorno deve vir de open, não de close-to-close."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="1min", tz="UTC"),
            "open": [100.0, 100.0, 110.0, 110.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "log_return": [0.0, 0.0, 0.05, 0.0],
        })
        signal = pd.Series([0.0, 0.0, 1.0, 0.0], index=df.index)
        result = run_backtest(
            df, signal,
            BacktestConfig(strategy_name="test", symbol="X", interval="1m"),
        )
        assert result.gross_returns.iloc[3] == pytest.approx(0.0, abs=1e-12)
        assert result.gross_returns.sum() == pytest.approx(0.0, abs=1e-12)
