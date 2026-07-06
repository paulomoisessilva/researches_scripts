"""Testes de cache de features e grid walk-forward em dois níveis."""

import pandas as pd
import pytest

from src.backtest import BacktestConfig
from src.indicators import compute_all_features, load_or_compute_features
from src.strategies import (
    compute_band_z,
    signal_from_z,
    strategy_bollinger,
    strategy_regression_bands,
    strategy_return_residuals,
)
from src.walk_forward import optimize_params_on_validation, run_walk_forward


@pytest.fixture
def df_with_features(sample_ohlcv, sample_config):
    return compute_all_features(sample_ohlcv, sample_config)


class TestComputeBandZEquivalence:
    @pytest.mark.parametrize(
        "strategy_name,strategy_func,window,k",
        [
            ("bollinger", strategy_bollinger, 20, 2.0),
            ("regression_bands", strategy_regression_bands, 50, 2.0),
            ("return_residuals", strategy_return_residuals, 50, 2.0),
        ],
    )
    def test_signal_from_z_matches_strategy(
        self, df_with_features, strategy_name, strategy_func, window, k
    ):
        thresh = {
            "entry_long_z": 0.8,
            "entry_short_z": 0.8,
            "exit_long_z": 0.0,
            "exit_short_z": 0.0,
        }
        z = compute_band_z(df_with_features, strategy_name, window, k)
        expected = strategy_func(df_with_features, window=window, k=k, **thresh)
        actual = signal_from_z(z, name=strategy_name, **thresh)

        pd.testing.assert_series_equal(actual, expected)


class TestLoadOrComputeFeatures:
    def test_loads_from_parquet(self, sample_ohlcv, sample_config, tmp_path):
        df_feat = compute_all_features(sample_ohlcv, sample_config)
        parquet_path = tmp_path / "BTCUSDT_1m_features.parquet"
        df_feat.to_parquet(parquet_path, index=False)

        loaded = load_or_compute_features(
            sample_ohlcv,
            sample_config,
            features_path=parquet_path,
        )

        pd.testing.assert_frame_equal(loaded, df_feat)

    def test_computes_when_parquet_missing(self, sample_ohlcv, sample_config, tmp_path):
        missing_path = tmp_path / "missing.parquet"
        computed = load_or_compute_features(
            sample_ohlcv,
            sample_config,
            features_path=missing_path,
        )
        expected = compute_all_features(sample_ohlcv, sample_config)

        pd.testing.assert_frame_equal(computed, expected)

    def test_force_compute_ignores_parquet(self, sample_ohlcv, sample_config, tmp_path):
        df_feat = compute_all_features(sample_ohlcv, sample_config)
        parquet_path = tmp_path / "features.parquet"
        df_feat.to_parquet(parquet_path, index=False)

        recomputed = load_or_compute_features(
            sample_ohlcv,
            sample_config,
            features_path=parquet_path,
            force_compute=True,
        )
        expected = compute_all_features(sample_ohlcv, sample_config)

        pd.testing.assert_frame_equal(recomputed, expected)


class TestTwoLevelGridOptimization:
    def test_optimize_band_strategy_returns_valid_params(self, sample_ohlcv, sample_config):
        bt_cfg = BacktestConfig(
            strategy_name="bollinger",
            symbol="BTCUSDT",
            interval="1m",
            initial_capital=10_000,
            fee_bps=2.0,
            slippage_bps=2.0,
        )
        df_feat = compute_all_features(sample_ohlcv, sample_config)
        train_end = df_feat["timestamp"].iloc[400]
        df_train_val = df_feat[df_feat["timestamp"] < train_end].reset_index(drop=True)
        val_start = df_train_val["timestamp"].iloc[int(len(df_train_val) * 0.8)]
        val_end = train_end

        best_params, best_score = optimize_params_on_validation(
            df_train_val=sample_ohlcv,
            strategy_name="bollinger",
            cfg=sample_config,
            bt_cfg_template=bt_cfg,
            df_feat_train_val=df_train_val,
            val_start=val_start,
            val_end=val_end,
        )

        assert "window" in best_params
        assert "k" in best_params
        assert "entry_long_z" in best_params
        assert isinstance(best_score, float)


class TestRunWalkForwardWithPreloadedFeatures:
    def test_run_walk_forward_accepts_df_feat(self, sample_ohlcv, sample_config):
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
