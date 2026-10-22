"""Trajetória contínua com estado carregado entre os dias (D16)."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest, run_backtest_loop
from src.carried_walk_forward import carried_backtest, carried_signal_inputs, run_carried_walk_forward
from src.indicators import compute_all_features
from src.strategies import STRATEGY_REGISTRY, compute_band_z
from src.walk_forward import build_windows, run_walk_forward
from tests.test_fast_core import _assert_same_result, _reference_pair_machine

PARAMS_A = {"window": 20, "k": 1.5, "entry_long_z": 0.8, "entry_short_z": 0.8,
            "exit_long_z": 0.0, "exit_short_z": 0.0}
PARAMS_B = {"window": 50, "k": 2.0, "entry_long_z": 1.0, "entry_short_z": 1.0,
            "exit_long_z": 0.0, "exit_short_z": 0.0}


def _setup(days: int = 45, seed: int = 4):
    """Série de 15 min que reverte à média (posições duram horas) e config contínua."""
    rng = np.random.default_rng(seed)
    n = days * 96
    x = np.zeros(n)
    for i in range(1, n):  # OU: z passa de um dia para o outro com posição aberta
        x[i] = 0.995 * x[i - 1] + rng.normal(0, 3e-3)
    close = 30_000 * np.exp(x)
    ts = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open": close * (1 + rng.normal(0, 2e-4, n)),
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": rng.uniform(1, 100, n),
    })
    cfg = {
        "backtest": {"initial_capital": 10_000, "allow_short": True, "max_position": 1.0},
        "features": {
            "rsi_windows": [14], "stochastic_windows": [14],
            "bollinger_windows": [20, 50], "bollinger_k": [1.5, 2.0, 3.0],
            "regression_windows": [50], "regression_k": [2.0, 3.0],
        },
        "strategy": {"entry_long_z": [0.8, 1.0], "entry_short_z": [0.8, 1.0],
                     "exit_long_z": [0.0], "exit_short_z": [0.0]},
        "walk_forward": {
            "train_days": 3, "validation_days": 3, "test_days": 1, "step_days": 1,
            "selection_metric": "net_sharpe", "close_position_end_of_day": False,
            "selection_engine": "single_pass",
        },
    }
    return df, compute_all_features(df, cfg), cfg


def _windows(df, cfg):
    wf = cfg["walk_forward"]
    return build_windows(df, wf["train_days"], wf["validation_days"], wf["test_days"], wf["step_days"])


def _bt(**kw):
    base = dict(strategy_name="bollinger", symbol="X", interval="15m", fee_bps=2.0, slippage_bps=2.0, close_eod=False)
    base.update(kw)
    return BacktestConfig(**base)


def _bounds(df_feat, windows):
    ts = pd.DatetimeIndex(df_feat["timestamp"])
    starts = ts.searchsorted([w.test_start for w in windows], side="left")
    ends = ts.searchsorted([w.test_end for w in windows], side="left")
    return np.asarray(starts), np.asarray(ends)


class TestConstantParameters:
    def test_equals_one_plain_continuous_run(self):
        df, df_feat, cfg = _setup()
        windows = _windows(df, cfg)
        params = [PARAMS_B] * len(windows)
        result, inp = carried_backtest(df_feat, "bollinger", windows, params, _bt())

        span = df_feat.iloc[inp["s0"]:inp["s1"]].reset_index(drop=True)
        plain_signal = STRATEGY_REGISTRY["bollinger"](span, **PARAMS_B)
        plain = run_backtest(span[["timestamp", "open"]], plain_signal, _bt())
        _assert_same_result(result, plain)
        assert result.trade_units.sum() > 0  # o teste não é trivial

    def test_trajectory_starts_flat_on_the_first_test_bar(self):
        df, df_feat, cfg = _setup()
        windows = _windows(df, cfg)
        result, inp = carried_backtest(df_feat, "bollinger", windows, [PARAMS_A] * len(windows), _bt())
        assert result.positions.iloc[0] == 0.0
        assert result.trade_units.iloc[0] == 0.0
        assert pd.Timestamp(result.positions.index[0]) == windows[0].test_start


class TestParameterSwitch:
    @staticmethod
    def _alternating(n_windows, seed=0):
        rng = np.random.default_rng(seed)
        return [PARAMS_A if rng.random() < 0.5 else PARAMS_B for _ in range(n_windows)]

    def test_matches_a_stepwise_reference_with_switching_parameters(self):
        df, df_feat, cfg = _setup()
        windows = _windows(df, cfg)
        params = self._alternating(len(windows))
        result, inp = carried_backtest(df_feat, "bollinger", windows, params, _bt())

        starts, ends = _bounds(df_feat, windows)
        s0, s1 = int(starts[0]), int(ends[-1])
        length = s1 - s0
        z_full = {
            (p["window"], p["k"]): compute_band_z(df_feat, "bollinger", p["window"], p["k"]).to_numpy()
            for p in (PARAMS_A, PARAMS_B)
        }
        zp, zc = np.full(length, np.nan), np.full(length, np.nan)
        el, es, xl, xs = (np.zeros(length) for _ in range(4))
        for i, p in enumerate(params):
            a, b = int(starts[i]) - s0, int(ends[i]) - s0
            zf = z_full[(p["window"], p["k"])]
            zc[a:b] = zf[s0 + a:s0 + b]
            zp[a:b] = zf[s0 + a - 1:s0 + b - 1]
            el[a:b], es[a:b] = p["entry_long_z"], p["entry_short_z"]
            xl[a:b], xs[a:b] = p["exit_long_z"], p["exit_short_z"]
        ref_pos = _reference_pair_machine(zp, zc, el, es, xl, xs)
        ref_signal = pd.Series(np.r_[0.0, ref_pos[:-1]])
        span = df_feat[["timestamp", "open"]].iloc[s0:s1].reset_index(drop=True)
        ref = run_backtest_loop(span, ref_signal, _bt())
        _assert_same_result(result, ref)

        # o teste só vale se houve posição aberta atravessando uma troca de parâmetros
        carried_over_switch = 0
        for i in range(1, len(windows)):
            boundary = int(starts[i]) - s0
            if params[i] is not params[i - 1] and result.positions.iloc[boundary - 1] != 0.0:
                carried_over_switch += 1
        assert carried_over_switch >= 1

    def test_pair_on_the_first_bar_of_a_day_uses_the_new_rule_on_both_sides(self):
        df, df_feat, cfg = _setup()
        windows = _windows(df, cfg)
        params = [PARAMS_A, PARAMS_B] * (len(windows) // 2) + [PARAMS_A] * (len(windows) % 2)
        inp = carried_signal_inputs(df_feat, "bollinger", windows, params)
        z_b = compute_band_z(df_feat, "bollinger", PARAMS_B["window"], PARAMS_B["k"]).to_numpy()
        first = int(inp["starts"][1])  # primeiro dia com PARAMS_B
        a = first - inp["s0"]
        assert inp["z_curr"][a] == z_b[first]
        assert inp["z_prev"][a] == z_b[first - 1]

    def test_carried_position_is_continuous_and_every_change_is_charged(self):
        df, df_feat, cfg = _setup()
        windows = _windows(df, cfg)
        params = self._alternating(len(windows), seed=3)
        result, _ = carried_backtest(df_feat, "bollinger", windows, params, _bt())
        pos = result.positions.to_numpy()
        assert result.trade_units.sum() == pytest.approx(np.abs(np.diff(pos)).sum(), abs=1e-12)
        assert set(np.unique(pos)) <= {-1.0, 0.0, 1.0}

    def test_per_window_mode_leaves_position_jumps_uncharged_and_carried_mode_does_not(self):
        df, df_feat, cfg = _setup()
        kwargs = dict(df=df, strategy_name="bollinger", symbol="X", interval="15m",
                      fee_bps=2.0, slippage_bps=2.0, df_feat=df_feat, n_workers=1)
        legacy = run_walk_forward(cfg=cfg, **kwargs)
        carried_cfg = copy.deepcopy(cfg)
        carried_cfg["walk_forward"]["carry_position_state"] = True
        carried = run_walk_forward(cfg=carried_cfg, **kwargs)

        def uncharged(results):
            rs = sorted(results, key=lambda r: r.window_id)
            pos = pd.concat([r.test_positions for r in rs])
            units = pd.concat([r.test_trade_units for r in rs])
            return pos.diff().abs().fillna(pos.abs()).sum() - units.sum()

        assert uncharged(legacy) > 0.5  # o modo antigo salta de posição sem custo
        assert uncharged(carried) == pytest.approx(0.0, abs=1e-12)


class TestRunWalkForwardIntegration:
    def test_carried_results_are_one_trajectory_cut_into_days(self):
        df, df_feat, cfg = _setup()
        cfg["walk_forward"]["carry_position_state"] = True
        results = run_walk_forward(df=df, strategy_name="bollinger", cfg=cfg, symbol="X",
                                   interval="15m", fee_bps=2.0, slippage_bps=2.0, df_feat=df_feat, n_workers=1)
        windows = _windows(df, cfg)
        assert [r.window_id for r in results] == [w.window_id for w in windows]

        # mesma escolha de parâmetros que o motor em passagem única sem trajetória contínua
        plain_cfg = copy.deepcopy(cfg)
        plain_cfg["walk_forward"]["carry_position_state"] = False
        plain = run_walk_forward(df=df, strategy_name="bollinger", cfg=plain_cfg, symbol="X",
                                 interval="15m", fee_bps=2.0, slippage_bps=2.0, df_feat=df_feat, n_workers=1)
        for a, b in zip(results, plain):
            assert a.best_params == b.best_params
            assert a.val_metric == pytest.approx(b.val_metric, abs=1e-12)

        for r, w in zip(results, windows):
            idx = pd.DatetimeIndex(r.test_returns.index)
            assert idx.min() >= w.test_start and idx.max() < w.test_end
            assert len(r.test_returns) == len(r.test_positions) == len(r.test_trade_units) > 0
        for a, b in zip(results, plain):  # mesmas métricas por janela que o modo antigo
            assert set(a.test_metrics) == set(b.test_metrics)

        # as janelas juntas reproduzem a trajetória única
        params = [r.best_params for r in results]
        full, _ = carried_backtest(df_feat, "bollinger", windows, params, _bt())
        joined = pd.concat([r.test_returns for r in results])
        pd.testing.assert_series_equal(joined, full.returns, check_exact=True)

        # operações: cada uma aparece em exatamente uma janela (a da saída)
        assert sum(len(r.test_trades) for r in results) == len(full.trades)

    @pytest.mark.parametrize("strategy", ["bollinger", "regression_bands"])
    def test_works_for_both_band_strategies(self, strategy):
        df, df_feat, cfg = _setup()
        cfg["walk_forward"]["carry_position_state"] = True
        results = run_walk_forward(df=df, strategy_name=strategy, cfg=cfg, symbol="X",
                                   interval="15m", fee_bps=2.0, slippage_bps=2.0, df_feat=df_feat, n_workers=1)
        assert len(results) > 10
        assert all(r.strategy == strategy for r in results)


class TestRequirements:
    @staticmethod
    def _run(cfg, strategy="bollinger"):
        df, df_feat, _ = _setup()
        return run_walk_forward(df=df, strategy_name=strategy, cfg=cfg, symbol="X", interval="15m",
                                fee_bps=2.0, slippage_bps=2.0, df_feat=df_feat, n_workers=1)

    def _cfg(self, **wf):
        _, _, cfg = _setup()
        cfg["walk_forward"]["carry_position_state"] = True
        cfg["walk_forward"].update(wf)
        return cfg

    def test_rejects_end_of_day_close(self):
        with pytest.raises(ValueError, match="close_position_end_of_day"):
            self._run(self._cfg(close_position_end_of_day=True))

    def test_rejects_the_sliced_engine(self):
        with pytest.raises(ValueError, match="single_pass"):
            self._run(self._cfg(selection_engine="sliced"))

    def test_rejects_gaps_between_test_days(self):
        with pytest.raises(ValueError, match="step_days"):
            self._run(self._cfg(step_days=2))

    def test_rejects_the_random_strategy(self):
        with pytest.raises(ValueError, match="random"):
            self._run(self._cfg(), strategy="random")

    def test_non_contiguous_windows_are_refused_by_the_path_builder(self):
        df, df_feat, cfg = _setup()
        windows = _windows(df, cfg)
        del windows[3]
        with pytest.raises(ValueError, match="contíguos"):
            carried_signal_inputs(df_feat, "bollinger", windows, [PARAMS_A] * len(windows))

    def test_path_builder_refuses_eod_close_and_non_band_strategies(self):
        df, df_feat, cfg = _setup()
        windows = _windows(df, cfg)
        with pytest.raises(ValueError, match="fechamento diário"):
            carried_backtest(df_feat, "bollinger", windows, [PARAMS_A] * len(windows), _bt(close_eod=True))
        with pytest.raises(ValueError, match="bandas|cobre"):
            carried_signal_inputs(df_feat, "rsi", windows, [PARAMS_A] * len(windows))
