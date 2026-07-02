"""Fixtures compartilhadas entre os testes."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Gera DataFrame OHLCV sintético com 500 candles de 1 minuto."""
    rng = np.random.default_rng(42)
    n = 500

    # Preço simulado como random walk
    log_ret = rng.normal(0, 0.001, n)
    close = 30_000 * np.exp(np.cumsum(log_ret))

    high = close * (1 + rng.uniform(0, 0.002, n))
    low = close * (1 - rng.uniform(0, 0.002, n))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    volume = rng.uniform(1, 100, n)

    timestamps = pd.date_range("2023-01-01", periods=n, freq="1min", tz="UTC")

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "close_time": timestamps + pd.Timedelta("1min"),
        "quote_volume": volume * close,
        "number_of_trades": rng.integers(10, 500, n),
        "taker_buy_base_volume": volume * 0.5,
        "taker_buy_quote_volume": volume * 0.5 * close,
        "symbol": "BTCUSDT",
        "interval": "1m",
    })


@pytest.fixture
def sample_config() -> dict:
    """Config mínima para os testes."""
    return {
        "project": {"name": "test", "timezone": "UTC"},
        "data": {
            "source": "test",
            "market": "futures_um",
            "symbols": ["BTCUSDT"],
            "intervals": ["1m"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "raw_dir": "data/raw/binance",
            "processed_dir": "data/processed",
        },
        "backtest": {
            "initial_capital": 10_000,
            "allow_short": True,
            "execution": "next_open",
            "fee_bps_per_side": [0, 2],
            "slippage_bps_per_side": [0, 2],
            "max_position": 1.0,
            "close_position_end_of_day": False,
            "max_holding_bars": None,
        },
        "features": {
            "rsi_windows": [14],
            "stochastic_windows": [14],
            "bollinger_windows": [20],
            "bollinger_k": [2.0],
            "regression_windows": [50],
            "regression_k": [2.0],
        },
        "walk_forward": {
            "train_days": 10,
            "validation_days": 5,
            "test_days": 5,
            "step_days": 5,
            "selection_metric": "net_sharpe",
        },
    }
