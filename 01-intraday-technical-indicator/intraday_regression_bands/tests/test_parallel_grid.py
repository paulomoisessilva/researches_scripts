"""Testes de equivalência entre grid sequencial e paralelo."""

import pandas as pd
import pytest

from src.backtest import BacktestConfig
from src.indicators import compute_all_features
from src.walk_forward import (
    _resolve_n_workers,
    optimize_params_on_validation,
)


@pytest.fixture
def df_with_features(sample_ohlcv, sample_config):
    return compute_all_features(sample_ohlcv, sample_config)


@pytest.fixture
def bt_cfg():
    return BacktestConfig(
        strategy_name="bollinger",
        symbol="BTCUSDT",
        interval="1m",
        initial_capital=10_000,
        fee_bps=2.0,
        slippage_bps=2.0,
    )


@pytest.fixture
def df_train_val(df_with_features):
    train_end = df_with_features["timestamp"].iloc[400]
    return df_with_features[df_with_features["timestamp"] < train_end].reset_index(drop=True)


def _val_bounds(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Última fração do slice como período de validação."""
    ts = df["timestamp"]
    split = int(len(df) * train_frac)
    return ts.iloc[split], ts.iloc[-1] + pd.Timedelta(minutes=1)


@pytest.fixture
def val_period(df_train_val):
    return _val_bounds(df_train_val)


class TestResolveWorkers:
    def test_explicit_workers(self, sample_config):
        assert _resolve_n_workers(sample_config, 8) == 8

    def test_zero_means_sequential(self, sample_config):
        assert _resolve_n_workers(sample_config, 0) == 0

    def test_from_config_default(self, sample_config):
        assert _resolve_n_workers(sample_config, None) == 1


class TestParallelGridEquivalence:
    def test_parallel_matches_sequential_bollinger(
        self, sample_ohlcv, sample_config, df_train_val, bt_cfg, val_period
    ):
        val_start, val_end = val_period
        kwargs = dict(
            df_train_val=sample_ohlcv,
            strategy_name="bollinger",
            cfg=sample_config,
            bt_cfg_template=bt_cfg,
            df_feat_train_val=df_train_val,
            val_start=val_start,
            val_end=val_end,
        )
        seq_params, seq_score = optimize_params_on_validation(**kwargs, n_workers=1)
        par_params, par_score = optimize_params_on_validation(**kwargs, n_workers=2)

        assert seq_params == par_params
        assert seq_score == par_score

    def test_zero_workers_matches_one_bollinger(
        self, sample_ohlcv, sample_config, df_train_val, bt_cfg, val_period
    ):
        val_start, val_end = val_period
        kwargs = dict(
            df_train_val=sample_ohlcv,
            strategy_name="bollinger",
            cfg=sample_config,
            bt_cfg_template=bt_cfg,
            df_feat_train_val=df_train_val,
            val_start=val_start,
            val_end=val_end,
        )
        one_params, one_score = optimize_params_on_validation(**kwargs, n_workers=1)
        zero_params, zero_score = optimize_params_on_validation(**kwargs, n_workers=0)

        assert one_params == zero_params
        assert one_score == zero_score

    def test_parallel_matches_sequential_rsi(
        self, sample_ohlcv, sample_config, df_train_val, val_period
    ):
        val_start, val_end = val_period
        bt_cfg = BacktestConfig(
            strategy_name="rsi",
            symbol="BTCUSDT",
            interval="1m",
        )
        kwargs = dict(
            df_train_val=sample_ohlcv,
            strategy_name="rsi",
            cfg=sample_config,
            bt_cfg_template=bt_cfg,
            df_feat_train_val=df_train_val,
            val_start=val_start,
            val_end=val_end,
        )
        seq_params, seq_score = optimize_params_on_validation(**kwargs, n_workers=1)
        par_params, par_score = optimize_params_on_validation(**kwargs, n_workers=2)

        assert seq_params == par_params
        assert seq_score == par_score
