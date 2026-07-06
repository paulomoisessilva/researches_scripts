"""Testes do baseline aleatório multi-seed (metodologia principal)."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest
from src.indicators import compute_features_for_strategy
from src.strategies import (
    position_exposure_stats,
    resolve_random_seeds,
    strategy_random,
)
from src.walk_forward import (
    WFResult,
    aggregate_random_seed_distribution,
    aggregate_walk_forward_oos,
    optimize_params_on_validation,
    run_walk_forward,
)


class TestRandomNotOptimized:
    def test_param_grid_is_singleton(self, sample_config):
        from src.walk_forward import _build_param_grid

        grid = _build_param_grid("random", sample_config)
        assert grid == [{}]

    def test_optimize_does_not_select_among_seeds(self, sample_ohlcv, sample_config):
        df_feat = compute_features_for_strategy("random", sample_ohlcv, sample_config)
        ts = df_feat["timestamp"]
        bt_cfg = BacktestConfig(strategy_name="random", symbol="X", interval="1m")

        params, score = optimize_params_on_validation(
            df_train_val=sample_ohlcv.iloc[:400],
            strategy_name="random",
            cfg=sample_config,
            bt_cfg_template=bt_cfg,
            val_start=ts.iloc[300],
            val_end=ts.iloc[380],
            df_feat_train_val=df_feat.iloc[:400],
        )
        assert params == {}
        assert np.isfinite(score)


class TestRandomReproducibility:
    def test_same_seed_same_signal(self, sample_ohlcv):
        s1 = strategy_random(sample_ohlcv, seed=123, trade_prob=0.1)
        s2 = strategy_random(sample_ohlcv, seed=123, trade_prob=0.1)
        pd.testing.assert_series_equal(s1, s2)

    def test_different_seed_different_signal(self, sample_ohlcv):
        s1 = strategy_random(sample_ohlcv, seed=1, trade_prob=0.1)
        s2 = strategy_random(sample_ohlcv, seed=2, trade_prob=0.1)
        assert not s1.equals(s2)


class TestRandomExposureBalance:
    def test_balanced_long_short_over_large_sample(self):
        n = 50_000
        df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=n, freq="1min")})
        sig = strategy_random(df, seed=99, trade_prob=0.05)

        entries = sig[sig != 0]
        long_frac = (entries > 0).mean()
        assert 0.45 <= long_frac <= 0.55

    def test_position_stats_keys(self, sample_ohlcv):
        sig = strategy_random(sample_ohlcv, seed=42, trade_prob=0.1)
        stats = position_exposure_stats(sig)
        assert set(stats) >= {"long_pct", "short_pct", "flat_pct", "avg_position", "position_changes"}
        assert abs(stats["avg_position"]) < 0.15


class TestRandomWalkForwardBypass:
    def test_random_skips_validation_optimization(self, sample_config):
        n = 30 * 96
        rng = np.random.default_rng(0)
        close = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC"),
            "open": close * (1 + rng.normal(0, 0.0002, n)),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.uniform(1, 100, n),
        })
        sample_config["random_baseline"] = {"n_seeds": 3, "seed_base": 0, "trade_prob": 0.05}
        results = run_walk_forward(
            df,
            strategy_name="random",
            cfg=sample_config,
            symbol="BTCUSDT",
            interval="15m",
            fee_bps=0.0,
            slippage_bps=0.0,
        )
        assert len(results) > 0
        seeds = {r.best_params["seed"] for r in results}
        assert seeds <= {0, 1, 2}
        assert all(np.isnan(r.val_metric) for r in results)


class TestMultiSeedAggregation:
    def _make_seed_result(self, strategy: str, seed: int, sharpe_proxy: float) -> WFResult:
        ret = pd.Series([sharpe_proxy * 0.001, sharpe_proxy * 0.001])
        return WFResult(
            window_id=seed,
            strategy=strategy,
            symbol="BTCUSDT",
            interval="15m",
            best_params={"seed": seed},
            val_metric=float("nan"),
            test_equity=pd.Series([1.0, 1.0 + sharpe_proxy * 0.001]),
            test_returns=ret,
            test_gross_returns=ret,
            test_positions=pd.Series([1.0, 0.0]),
            test_trades=pd.DataFrame(),
            test_metrics={},
        )

    def test_oos_summary_one_row_per_seed(self):
        results = [
            self._make_seed_result("random", 0, 1.0),
            self._make_seed_result("random", 1, -1.0),
            self._make_seed_result("bollinger", None, 2.0),
        ]
        results[2].best_params = {"window": 20}
        summary = aggregate_walk_forward_oos(results)
        random_rows = summary[summary["strategy"] == "random"]
        assert len(random_rows) == 2

    def test_seed_distribution_excludes_matched(self):
        summary = aggregate_walk_forward_oos([
            self._make_seed_result("random", i, float(i)) for i in range(5)
        ] + [
            WFResult(
                window_id=0,
                strategy="random_matched_bollinger",
                symbol="BTCUSDT",
                interval="15m",
                best_params={"seed": 0},
                val_metric=float("nan"),
                test_equity=pd.Series([1.0]),
                test_returns=pd.Series([0.01]),
                test_gross_returns=pd.Series([0.01]),
                test_positions=pd.Series([1.0]),
                test_trades=pd.DataFrame(),
                test_metrics={},
            )
        ])
        dist = aggregate_random_seed_distribution(summary)
        assert len(dist) == 1
        assert dist.iloc[0]["strategy"] == "random"
        assert dist.iloc[0]["n_seeds"] == 5

    def test_seed_distribution_percentiles(self):
        summary = aggregate_walk_forward_oos([
            self._make_seed_result("random", i, float(i - 5)) for i in range(10)
        ])
        dist = aggregate_random_seed_distribution(summary)
        assert len(dist) == 1
        assert dist.iloc[0]["n_seeds"] == 10
        assert dist.iloc[0]["oos_sharpe_p50"] == pytest.approx(
            summary["oos_sharpe"].median(), rel=1e-6
        )


class TestRandomSameExecution:
    def test_uses_next_open_backtest(self, sample_ohlcv):
        df = sample_ohlcv.iloc[:20].copy()
        signal = strategy_random(df, seed=1, trade_prob=0.5)
        cfg = BacktestConfig(
            strategy_name="random", symbol="X", interval="1m",
            fee_bps=10.0, slippage_bps=0.0,
        )
        result = run_backtest(df, signal, cfg)
        assert len(result.positions) == len(df)
        assert result.trades.shape[0] >= 0


class TestResolveRandomSeeds:
    def test_n_seeds_range(self):
        cfg = {"random_baseline": {"n_seeds": 5, "seed_base": 10}}
        assert resolve_random_seeds(cfg) == [10, 11, 12, 13, 14]

    def test_explicit_seeds(self):
        cfg = {"random_baseline": {"seeds": [1, 3, 5]}}
        assert resolve_random_seeds(cfg) == [1, 3, 5]
