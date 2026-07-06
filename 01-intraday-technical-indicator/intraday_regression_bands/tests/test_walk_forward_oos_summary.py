"""Testes do resumo OOS walk-forward + CBE."""

import pandas as pd

from src.walk_forward import WFResult, aggregate_walk_forward_oos


def _make_result(
    strategy: str,
    window_id: int,
    returns: list[float],
    positions: list[float],
) -> WFResult:
    idx = pd.RangeIndex(len(returns))
    ret = pd.Series(returns, index=idx)
    gross = pd.Series(returns, index=idx)
    pos = pd.Series(positions, index=idx)
    equity = pd.Series((1 + ret).cumprod(), index=idx)
    return WFResult(
        window_id=window_id,
        strategy=strategy,
        symbol="BTCUSDT",
        interval="15m",
        best_params={"window": 20, "k": 2.0},
        val_metric=1.0,
        test_equity=equity,
        test_returns=ret,
        test_gross_returns=gross,
        test_positions=pos,
        test_trades=pd.DataFrame({"entry_idx": [0], "exit_idx": [1]}),
        test_metrics={"sharpe": 1.0},
    )


def test_aggregate_walk_forward_oos_concatenates_windows():
    r1 = _make_result("bollinger", 0, [0.001, 0.001], [1.0, 1.0])
    r2 = _make_result("bollinger", 1, [0.002, -0.0005], [1.0, 0.0])
    summary = aggregate_walk_forward_oos([r1, r2])

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["strategy"] == "bollinger"
    assert row["n_windows"] == 2
    assert row["number_of_trades"] == 2
    assert row["oos_total_return"] > 0
    assert row["break_even_total_return_bps"] > 0


def test_aggregate_walk_forward_oos_zero_cbe_when_losing():
    r1 = _make_result("random", 0, [-0.001, -0.0005], [1.0, 1.0])
    summary = aggregate_walk_forward_oos([r1])

    assert summary.iloc[0]["oos_total_return"] < 0
    assert summary.iloc[0]["break_even_total_return_bps"] == 0.0
