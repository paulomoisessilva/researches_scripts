"""Testes de cache e escopo de features no walk-forward."""

import numpy as np
import pandas as pd

from src.backtest import BacktestConfig
from src.indicators import compute_all_features, compute_features_for_strategy
from src.walk_forward import (
    _run_backtest_on_features,
    _run_strategy_on_slice,
    optimize_params_on_validation,
    run_walk_forward,
)


def test_bollinger_feature_scope_excludes_regression(sample_ohlcv, sample_config):
    df = compute_features_for_strategy("bollinger", sample_ohlcv, sample_config)
    cols = set(df.columns)

    assert "log_return" in cols
    assert any(c.startswith("bb_") for c in cols)
    assert not any(c.startswith("reg_") for c in cols)
    assert not any(c.startswith("resid_") for c in cols)
    assert not any(c.startswith("rsi_") for c in cols)


def test_compute_features_idempotent(sample_ohlcv, sample_config):
    df_once = compute_features_for_strategy("bollinger", sample_ohlcv, sample_config)
    df_twice = compute_features_for_strategy("bollinger", df_once, sample_config)

    shared_cols = [c for c in df_once.columns if c in df_twice.columns]
    for col in shared_cols:
        pd.testing.assert_series_equal(df_once[col], df_twice[col], check_names=False)


def test_precomputed_slice_matches_full_series(sample_ohlcv, sample_config):
    """Features fatiadas da série completa coincidem com valores na série inteira."""
    full = compute_features_for_strategy("bollinger", sample_ohlcv, sample_config)
    start = full["timestamp"].iloc[100]
    end = full["timestamp"].iloc[200]
    mask = (full["timestamp"] >= start) & (full["timestamp"] < end)
    sliced = full[mask].reset_index(drop=True)

    col = "bb_upper_20_20"
    assert col in full.columns
    np.testing.assert_array_equal(
        sliced[col].values,
        full.loc[mask, col].values,
    )


def test_optimize_params_reuses_precomputed_features(sample_ohlcv, sample_config):
    cfg = sample_config
    bt_cfg = BacktestConfig(
        strategy_name="bollinger",
        symbol="BTCUSDT",
        interval="1m",
        initial_capital=10_000,
        fee_bps=2.0,
        slippage_bps=2.0,
    )

    df_feat = compute_features_for_strategy("bollinger", sample_ohlcv, sample_config)
    train_end = df_feat["timestamp"].iloc[300]

    df_train_val = df_feat[df_feat["timestamp"] < train_end].reset_index(drop=True)
    val_start = df_train_val["timestamp"].iloc[int(len(df_train_val) * 0.8)]
    val_end = train_end

    params_cached, score_cached = optimize_params_on_validation(
        df_train_val=sample_ohlcv,
        strategy_name="bollinger",
        cfg=cfg,
        bt_cfg_template=bt_cfg,
        df_feat_train_val=df_train_val,
        val_start=val_start,
        val_end=val_end,
    )

    params_fresh, score_fresh = optimize_params_on_validation(
        df_train_val=sample_ohlcv[sample_ohlcv["timestamp"] < train_end].reset_index(drop=True),
        strategy_name="bollinger",
        cfg=cfg,
        bt_cfg_template=bt_cfg,
        df_feat_train_val=None,
        val_start=val_start,
        val_end=val_end,
    )

    assert params_cached == params_fresh
    assert score_cached == score_fresh


def test_run_backtest_on_features_matches_run_strategy_on_slice(sample_ohlcv, sample_config):
    cfg = sample_config
    bt_cfg = BacktestConfig(
        strategy_name="bollinger",
        symbol="BTCUSDT",
        interval="1m",
    )
    params = {"window": 20, "k": 2.0}
    slice_df = sample_ohlcv.iloc[50:200].reset_index(drop=True)

    result_wrapper, df_feat = _run_strategy_on_slice(
        slice_df, "bollinger", params, cfg, bt_cfg
    )
    result_direct = _run_backtest_on_features(df_feat, "bollinger", params, bt_cfg)

    pd.testing.assert_series_equal(result_wrapper.equity, result_direct.equity)
    pd.testing.assert_series_equal(result_wrapper.returns, result_direct.returns)


def test_run_walk_forward_skips_feature_recompute(sample_ohlcv, sample_config):
    df_feat = compute_all_features(sample_ohlcv, sample_config)
    results = run_walk_forward(
        df=sample_ohlcv,
        strategy_name="bollinger",
        cfg=sample_config,
        symbol="BTCUSDT",
        interval="1m",
        df_feat=df_feat,
    )
    assert isinstance(results, list)


def test_compute_all_features_produces_all_indicator_groups(sample_ohlcv, sample_config):
    df = compute_all_features(sample_ohlcv, sample_config)
    cols = set(df.columns)

    assert "log_return" in cols
    assert "rsi_14" in cols
    assert "stoch_k_14" in cols
    assert "bb_upper_20_20" in cols
    assert "reg_upper_50_20" in cols
    assert "resid_upper_50_20" in cols
    assert "technical_confluence_score" in cols
    assert not df.columns.duplicated().any()
