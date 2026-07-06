"""Testes metodológicos do walk-forward (janelas, validação, custos, CBE)."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest
from src.config import load_config, resolve_cost_bps, resolve_strategies
from src.indicators import compute_features_for_strategy, log_return
from src.metrics import (
    _position_change_units,
    break_even_total_return_bps,
    selection_metric_score,
)
from src.walk_forward import (
    _get_wf_int,
    _run_backtest_on_features,
    _score_backtest_on_period,
    _slice_backtest_result,
    build_windows,
    optimize_params_on_validation,
)


def _make_daily_df(n_days: int = 40, bars_per_day: int = 96) -> pd.DataFrame:
    """OHLCV sintético com barras 15m e dias UTC completos."""
    rng = np.random.default_rng(7)
    n = n_days * bars_per_day
    log_ret = rng.normal(0, 0.0005, n)
    close = 30_000 * np.exp(np.cumsum(log_ret))
    timestamps = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close * (1 + rng.normal(0, 0.0002, n)),
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": rng.uniform(1, 100, n),
        "log_return": log_return(pd.Series(close)),
    })


class TestWalkForwardConfigKeys:
    def test_accepts_validation_days(self):
        cfg = {"validation_days": 7, "val_days": 99}
        assert _get_wf_int(cfg, "validation_days", "val_days") == 7

    def test_accepts_val_days_alias(self):
        cfg = {"val_days": 5}
        assert _get_wf_int(cfg, "validation_days", "val_days") == 5

    def test_config_dev_window_sizes(self):
        root = __import__("pathlib").Path(__file__).parent.parent
        cfg = load_config(root / "config.dev.yaml")
        wf = cfg["walk_forward"]
        assert wf["train_days"] == 21
        assert wf["validation_days"] == 7
        assert wf["test_days"] == 1
        assert wf["step_days"] == 1
        assert wf["selection_metric"] == "net_sharpe"
        assert wf["close_position_end_of_day"] is True


class TestBuildWindowsOneDayStep:
    def test_one_day_test_non_overlapping(self):
        df = _make_daily_df(n_days=40)
        windows = build_windows(
            df, train_days=21, val_days=7, test_days=1, step_days=1
        )
        assert len(windows) >= 5

        test_starts = [w.test_start for w in windows]
        test_ends = [w.test_end for w in windows]
        for i in range(len(windows) - 1):
            assert test_ends[i] == test_starts[i + 1]
            assert (test_ends[i] - test_starts[i]) == pd.Timedelta(days=1)

    def test_step_advances_train_by_one_day(self):
        df = _make_daily_df(n_days=40)
        windows = build_windows(
            df, train_days=21, val_days=7, test_days=1, step_days=1
        )
        for i in range(len(windows) - 1):
            delta = windows[i + 1].train_start - windows[i].train_start
            assert delta == pd.Timedelta(days=1)


class TestValidationOnlySelection:
    def test_score_uses_validation_period_only(self, sample_ohlcv, sample_config):
        df_feat = compute_features_for_strategy("bollinger", sample_ohlcv, sample_config)
        ts = df_feat["timestamp"]
        val_start = ts.iloc[350]
        val_end = ts.iloc[450]

        bt_cfg = BacktestConfig(
            strategy_name="bollinger",
            symbol="BTCUSDT",
            interval="1m",
            fee_bps=0.0,
            slippage_bps=0.0,
        )

        from src.metrics import sharpe_ratio

        params = {"window": 20, "k": 2.0, "entry_long_z": 0.8, "entry_short_z": 0.8,
                  "exit_long_z": 0.0, "exit_short_z": 0.0}
        result = _run_backtest_on_features(df_feat.iloc[:450], "bollinger", params, bt_cfg)

        full_sharpe = sharpe_ratio(result.returns, "1m")
        val_sharpe = _score_backtest_on_period(
            result, val_start, val_end, "1m", "net_sharpe"
        )

        assert val_sharpe != full_sharpe or len(result.returns) == len(
            result.returns.loc[(ts.iloc[:450] >= val_start) & (ts.iloc[:450] < val_end)]
        )


class TestStrategiesFromConfig:
    def test_only_configured_strategies(self, sample_config):
        sample_config["strategies"] = [
            "bollinger", "regression_bands", "return_residuals", "random"
        ]
        names = resolve_strategies(sample_config)
        assert set(names) == {
            "bollinger", "regression_bands", "return_residuals", "random"
        }
        assert "rsi" not in names
        assert "stochastic" not in names
        assert "technical_confluence" not in names


class TestHistoricalContext:
    def test_test_slice_uses_full_context_signals(self, sample_ohlcv, sample_config):
        """Sinais stateful devem usar histórico train+val+test, métricas só no teste."""
        df_feat = compute_features_for_strategy("bollinger", sample_ohlcv, sample_config)
        ts = df_feat["timestamp"]
        test_start = ts.iloc[400]
        test_end = ts.iloc[450]
        params = {
            "window": 20, "k": 2.0,
            "entry_long_z": 0.8, "entry_short_z": 0.8,
            "exit_long_z": 0.0, "exit_short_z": 0.0,
        }
        bt_cfg = BacktestConfig(
            strategy_name="bollinger", symbol="BTCUSDT", interval="1m",
        )

        context = df_feat[(ts >= ts.iloc[0]) & (ts < test_end)].reset_index(drop=True)
        isolated = df_feat[(ts >= test_start) & (ts < test_end)].reset_index(drop=True)

        full_result = _run_backtest_on_features(context, "bollinger", params, bt_cfg)
        iso_result = _run_backtest_on_features(isolated, "bollinger", params, bt_cfg)

        sliced = _slice_backtest_result(full_result, test_start, test_end)
        assert not sliced.positions.equals(iso_result.positions)


class TestSelectionMetric:
    def test_net_sharpe_default(self, sample_ohlcv, sample_config):
        df_feat = compute_features_for_strategy("bollinger", sample_ohlcv, sample_config)
        ts = df_feat["timestamp"]
        bt_cfg = BacktestConfig(strategy_name="bollinger", symbol="X", interval="1m")
        params = {"window": 20, "k": 2.0, "entry_long_z": 0.8, "entry_short_z": 0.8,
                  "exit_long_z": 0.0, "exit_short_z": 0.0}
        result = _run_backtest_on_features(df_feat.iloc[:300], "bollinger", params, bt_cfg)
        val_start, val_end = ts.iloc[200], ts.iloc[280]
        score = _score_backtest_on_period(result, val_start, val_end, "1m", "net_sharpe")
        assert np.isfinite(score) or score == -np.inf

    def test_net_total_return_metric(self, sample_ohlcv, sample_config):
        df_feat = compute_features_for_strategy("bollinger", sample_ohlcv, sample_config)
        ts = df_feat["timestamp"]
        bt_cfg = BacktestConfig(strategy_name="bollinger", symbol="X", interval="1m")
        params = {"window": 20, "k": 2.0, "entry_long_z": 0.8, "entry_short_z": 0.8,
                  "exit_long_z": 0.0, "exit_short_z": 0.0}
        result = _run_backtest_on_features(df_feat.iloc[:300], "bollinger", params, bt_cfg)
        val_start, val_end = ts.iloc[200], ts.iloc[280]
        score = _score_backtest_on_period(result, val_start, val_end, "1m", "net_total_return")
        assert isinstance(score, float)
        assert score != -np.inf or len(result.returns) < 2

    def test_unknown_metric_raises(self, sample_ohlcv, sample_config):
        df_feat = compute_features_for_strategy("bollinger", sample_ohlcv, sample_config)
        bt_cfg = BacktestConfig(strategy_name="bollinger", symbol="X", interval="1m")
        params = {"window": 20, "k": 2.0, "entry_long_z": 0.8, "entry_short_z": 0.8,
                  "exit_long_z": 0.0, "exit_short_z": 0.0}
        result = _run_backtest_on_features(df_feat.iloc[:200], "bollinger", params, bt_cfg)
        ts = df_feat["timestamp"]
        with pytest.raises(ValueError, match="desconhecida"):
            _score_backtest_on_period(
                result, ts.iloc[100], ts.iloc[150], "1m", "invalid_metric"
            )


class TestCostConfigResolution:
    def test_resolve_cost_from_config_lists(self):
        root = __import__("pathlib").Path(__file__).parent.parent
        cfg = load_config(root / "config.dev.yaml")
        fee, slip = resolve_cost_bps(cfg)
        assert fee == cfg["backtest"]["fee_bps_per_side"][0]
        assert slip == cfg["backtest"]["slippage_bps_per_side"][0]


class TestEndOfDayClosing:
    def test_forces_flat_at_utc_day_end(self):
        n = 10
        timestamps = pd.date_range("2025-01-01 23:50", periods=n, freq="1min", tz="UTC")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": np.linspace(100, 101, n),
            "close": np.linspace(100, 101, n),
            "log_return": np.zeros(n),
        })
        signal = pd.Series(1.0, index=df.index)
        cfg = BacktestConfig(
            strategy_name="test",
            symbol="X",
            interval="1m",
            close_eod=True,
        )
        result = run_backtest(df, signal, cfg)

        last_day = timestamps[-1].normalize()
        day_mask = pd.to_datetime(result.positions.index).normalize() == last_day
        assert (result.positions.loc[day_mask].iloc[-1] == 0.0)

    def test_eod_close_includes_cost(self):
        n = 5
        timestamps = pd.date_range("2025-01-01 23:56", periods=n, freq="1min", tz="UTC")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": np.full(n, 100.0),
            "close": np.full(n, 100.0),
            "log_return": np.zeros(n),
        })
        signal = pd.Series(1.0, index=df.index)
        cfg_no_eod = BacktestConfig(
            strategy_name="test", symbol="X", interval="1m",
            fee_bps=10.0, slippage_bps=0.0, close_eod=False,
        )
        cfg_eod = BacktestConfig(
            strategy_name="test", symbol="X", interval="1m",
            fee_bps=10.0, slippage_bps=0.0, close_eod=True,
        )
        r_no = run_backtest(df, signal, cfg_no_eod)
        r_eod = run_backtest(df, signal, cfg_eod)
        assert r_eod.returns.sum() < r_no.returns.sum()

    def test_eod_close_increases_turnover(self):
        n = 5
        timestamps = pd.date_range("2025-01-01 23:56", periods=n, freq="1min", tz="UTC")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": np.full(n, 100.0),
            "close": np.full(n, 100.0),
            "log_return": np.zeros(n),
        })
        signal = pd.Series(1.0, index=df.index)
        cfg_no = BacktestConfig(strategy_name="t", symbol="X", interval="1m", close_eod=False)
        cfg_yes = BacktestConfig(strategy_name="t", symbol="X", interval="1m", close_eod=True)
        r_no = run_backtest(df, signal, cfg_no)
        r_yes = run_backtest(df, signal, cfg_yes)
        assert _position_change_units(r_yes.positions) >= _position_change_units(r_no.positions)


class TestPerSideCosts:
    def test_zero_to_one_costs_one_side(self):
        n = 4
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC"),
            "open": np.full(n, 100.0),
            "close": np.full(n, 100.0),
            "log_return": np.zeros(n),
        })
        signal = pd.Series([0.0, 1.0, 1.0, 0.0], index=df.index)
        cfg = BacktestConfig(
            strategy_name="test", symbol="X", interval="1m",
            fee_bps=10.0, slippage_bps=0.0,
        )
        result = run_backtest(df, signal, cfg)
        expected_cost = 2 * (10.0 / 10_000)  # entrada + saída, 1 lado cada
        assert result.returns.sum() == pytest.approx(-expected_cost, abs=1e-10)

    def test_one_to_minus_one_costs_two_sides(self):
        n = 4
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC"),
            "open": np.full(n, 100.0),
            "close": np.full(n, 100.0),
            "log_return": np.zeros(n),
        })
        signal = pd.Series([0.0, 1.0, -1.0, -1.0], index=df.index)
        cfg = BacktestConfig(
            strategy_name="test", symbol="X", interval="1m",
            fee_bps=10.0, slippage_bps=0.0,
        )
        result = run_backtest(df, signal, cfg)
        flip_cost = 2 * (10.0 / 10_000)  # |1 - (-1)| = 2 lados no candle t=2
        assert result.returns.iloc[2] == pytest.approx(-flip_cost, abs=1e-10)


class TestBreakEvenFromGross:
    def test_betc_from_gross_not_net(self):
        gross = pd.Series([0.0, 0.02, 0.0])
        positions = pd.Series([0.0, 1.0, 0.0])
        bec = break_even_total_return_bps(gross, positions)
        assert bec == pytest.approx(100.0)  # 0.02 * 10000 / 2

    def test_betc_zero_when_gross_non_positive(self):
        gross = pd.Series([0.0, -0.01, 0.0])
        positions = pd.Series([0.0, 1.0, 0.0])
        assert break_even_total_return_bps(gross, positions) == 0.0

    def test_betc_nan_when_zero_turnover(self):
        gross = pd.Series([0.01, 0.01])
        positions = pd.Series([0.0, 0.0])
        assert np.isnan(break_even_total_return_bps(gross, positions))
