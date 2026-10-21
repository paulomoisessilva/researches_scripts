"""Testes do diagnóstico de parâmetros fixos por bloco (src/fixed_params.py)."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest
from src.fixed_params import evaluate_chunk, grid_tasks
from src.strategies import compute_band_z, signal_from_z
from tests.test_wf_analysis import TestWindowParallelism


def _bt():
    return BacktestConfig(strategy_name="fixed", symbol="X", interval="15m",
                          fee_bps=2.0, slippage_bps=2.0, close_eod=True)


def test_grid_tasks_cover_every_combination():
    _, _, cfg = TestWindowParallelism._setup()
    tasks = grid_tasks(["bollinger", "regression_bands", "random"], cfg)
    # 1 janela x 2 k x 2 limiares de entrada, por estratégia de bandas; "random" fica de fora
    assert len(tasks) == 2 * (1 * 2 * 2)
    assert {t[0] for t in tasks} == {"bollinger", "regression_bands"}


def test_daily_sums_match_a_direct_backtest_and_parallel_equals_sequential():
    df, df_feat, cfg = TestWindowParallelism._setup()
    ts = pd.DatetimeIndex(df_feat["timestamp"])
    start, end = ts[0].normalize() + pd.Timedelta(days=5), ts[-1].normalize()
    seq = evaluate_chunk(df_feat, ["bollinger", "regression_bands"], cfg, _bt(), start, end, workers=1)
    par = evaluate_chunk(df_feat, ["bollinger", "regression_bands"], cfg, _bt(), start, end, workers=2)

    assert seq["combos"] == par["combos"]
    for key in ("gross", "units", "trades"):
        assert np.array_equal(seq[key], par[key])

    strategy, window, k = seq["combos"][0]
    thresh = {"entry_long_z": 0.8, "entry_short_z": 0.8, "exit_long_z": 0.0, "exit_short_z": 0.0}
    res = run_backtest(df_feat, signal_from_z(compute_band_z(df_feat, strategy, window, k), name=strategy, **thresh), _bt())
    mask = (ts >= start) & (ts < end)
    assert seq["gross"][:, 0].sum() == pytest.approx(res.gross_returns.to_numpy()[mask].sum(), abs=1e-12)
    assert seq["units"][:, 0].sum() == pytest.approx(res.trade_units.to_numpy()[mask].sum(), abs=1e-12)
    assert len(seq["days"]) == (end - start).days
    assert (seq["trades"] >= 0).all()


def test_summarize_cbe_net_and_year_split():
    from src.fixed_params import summarize

    days = pd.date_range("2021-12-30", periods=4, freq="D", tz="UTC")           # 2 dias em 2021, 2 em 2022
    gross = np.array([[0.01, -0.02], [0.01, -0.01], [0.02, 0.0], [0.00, 0.01]])
    units = np.array([[2.0, 2.0], [2.0, 2.0], [4.0, 0.0], [0.0, 2.0]])
    trades = np.array([[1, 1], [1, 1], [2, 0], [0, 1]], dtype=float)
    t = summarize(days, [("bollinger", 25, 2.0), ("bollinger", 25, 3.0)], gross, units, trades, cost_bps=4.0)

    a = t[(t.k == 2.0) & (t.period == "ALL")].iloc[0]
    assert a.cbe_bps == pytest.approx(0.04 * 1e4 / 8.0)                          # 50 bps
    assert a.net_pct == pytest.approx((np.exp(0.04 - 8.0 * 4.0 / 1e4) - 1) * 100)
    assert a.trades_per_day == pytest.approx(1.0)
    b = t[(t.k == 3.0) & (t.period == "ALL")].iloc[0]
    assert b.cbe_bps == 0.0                                                      # bruto <= 0
    y22 = t[(t.k == 2.0) & (t.period == "2022")].iloc[0]
    assert y22.n_days == 2 and y22.gross_pct == pytest.approx((np.exp(0.02) - 1) * 100)
    cut = summarize(days, [("bollinger", 25, 2.0)], gross[:, :1], units[:, :1], trades[:, :1], 4.0, end="2021-12-31")
    assert set(cut.period) == {"ALL", "2021"}


def test_split_persistence_splits_by_date_and_flags_eligibility():
    from src.fixed_params import split_persistence

    days = pd.date_range("2021-01-01", periods=6, freq="D", tz="UTC")
    gross = np.array([[0.01, 0.0]] * 3 + [[-0.01, 0.02]] * 3)
    units = np.array([[2.0, 0.0]] * 6)
    trades = np.array([[1, 0]] * 6, dtype=float)
    t = split_persistence(days, [("bollinger", 25, 2.0), ("bollinger", 25, 3.0)], gross, units, trades,
                          cost_bps=10.0, split_date="2021-01-04", min_trades=3)
    a = t.iloc[0]
    assert a.gross_S == pytest.approx(0.03) and a.gross_T == pytest.approx(-0.03)
    assert a.net_S == pytest.approx(0.03 - 6.0 * 10.0 / 1e4)
    assert a.cbe_S == pytest.approx(0.03 * 1e4 / 6.0) and a.cbe_T == 0.0
    assert a.trades_S == 3 and bool(a.elegivel) is True
    assert bool(t.iloc[1].elegivel) is False
