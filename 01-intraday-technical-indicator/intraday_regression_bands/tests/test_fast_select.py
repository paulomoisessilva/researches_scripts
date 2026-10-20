"""Testes do motor de seleção em passagem única (somas acumuladas)."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig
from src.fast_select import (
    compare_engines,
    summarize_comparison,
    window_scores_from_returns,
)
from src.metrics import annualization_factor, selection_metric_score
from src.walk_forward import build_windows, run_walk_forward
from tests.test_wf_analysis import TestWindowParallelism


class TestScoresFromReturns:
    @pytest.mark.parametrize("metric", ["net_sharpe", "net_total_return", "net_profit_factor"])
    def test_matches_selection_metric_score(self, metric):
        rng = np.random.default_rng(5)
        r = rng.normal(2e-5, 1e-3, 3000)
        r[rng.integers(0, 3000, 400)] = 0.0  # barras sem posição
        starts = np.array([0, 100, 700, 1500, 2990, 50])
        ends = np.array([672, 772, 1372, 2172, 2995, 51])  # inclui janela curta (n=1)
        got = window_scores_from_returns(r, starts, ends, metric, annualization_factor("15m"))

        series = pd.Series(r)
        for i, (a, b) in enumerate(zip(starts, ends)):
            mask = pd.Series(False, index=series.index)
            mask.iloc[a:b] = True
            expected = selection_metric_score(series, pd.DataFrame(), "15m", metric, index_mask=mask)
            assert got[i] == pytest.approx(expected, rel=1e-9, abs=1e-12), (metric, a, b)

    def test_flat_window_has_zero_sharpe_and_neg_inf_profit_factor(self):
        r = np.zeros(1000)
        f = annualization_factor("15m")
        assert window_scores_from_returns(r, np.array([10]), np.array([300]), "net_sharpe", f)[0] == 0.0
        assert window_scores_from_returns(r, np.array([10]), np.array([300]), "net_profit_factor", f)[0] == -np.inf

    def test_only_gains_gives_inf_profit_factor(self):
        r = np.concatenate([np.full(100, -1e-3), np.full(100, 1e-3)])
        f = annualization_factor("15m")
        got = window_scores_from_returns(r, np.array([100]), np.array([200]), "net_profit_factor", f)
        assert got[0] == np.inf  # perdas anteriores à janela não contam


class TestEngineEquivalenceSynthetic:
    @staticmethod
    def _windows(df, cfg):
        wf = cfg["walk_forward"]
        return build_windows(df, wf["train_days"], wf["validation_days"], wf["test_days"], wf["step_days"])

    @pytest.mark.parametrize("strategy", ["bollinger", "regression_bands", "return_residuals"])
    def test_same_scores_and_choices(self, strategy):
        df, df_feat, cfg = TestWindowParallelism._setup()
        windows = self._windows(df, cfg)
        bt = BacktestConfig(strategy_name=strategy, symbol="X", interval="15m",
                            fee_bps=2.0, slippage_bps=2.0, close_eod=True)
        cmp = compare_engines(df_feat, strategy, cfg, bt, windows, "net_sharpe", n_workers=1)
        s = summarize_comparison(cmp)

        assert s["same_best_params_rate"] == 1.0
        assert s["score_equal_rate_same_state"] == 1.0
        assert s["max_abs_score_diff_same_state"] < 1e-9

    @pytest.mark.parametrize("strategy", ["bollinger", "return_residuals"])
    def test_run_walk_forward_same_with_both_engines(self, strategy):
        df, df_feat, cfg = TestWindowParallelism._setup()
        kwargs = dict(df=df, strategy_name=strategy, symbol="X", interval="15m",
                      fee_bps=2.0, slippage_bps=2.0, df_feat=df_feat, n_workers=1)

        sliced_cfg = {**cfg, "walk_forward": {**cfg["walk_forward"], "selection_engine": "sliced"}}
        fast_cfg = {**cfg, "walk_forward": {**cfg["walk_forward"], "selection_engine": "single_pass"}}
        sliced = run_walk_forward(cfg=sliced_cfg, **kwargs)
        fast = run_walk_forward(cfg=fast_cfg, **kwargs)

        assert len(sliced) == len(fast) > 3
        for a, b in zip(sliced, fast):
            assert a.window_id == b.window_id
            assert a.best_params == b.best_params
            assert a.val_metric == pytest.approx(b.val_metric, abs=1e-9)
            pd.testing.assert_series_equal(a.test_returns, b.test_returns)

    def test_invalid_engine_name_raises(self):
        df, df_feat, cfg = TestWindowParallelism._setup()
        bad = {**cfg, "walk_forward": {**cfg["walk_forward"], "selection_engine": "turbo"}}
        with pytest.raises(ValueError, match="selection_engine"):
            run_walk_forward(df=df, strategy_name="bollinger", cfg=bad, symbol="X",
                             interval="15m", df_feat=df_feat, n_workers=1)
