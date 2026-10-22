"""Testes de src/v2_robustness.py (SPA sem cada parte e persistência dos parâmetros fixos)."""

import numpy as np
import pandas as pd
import pytest

from src.v2_robustness import fixed_param_persistence, leave_one_out_spa


@pytest.fixture(scope="module")
def planted():
    """730 dias, 4 candidatas (g x h); só ('a','x') tem retorno positivo, em todos os dias."""
    rng = np.random.default_rng(3)
    n = 730
    dates = pd.date_range("2021-01-01", periods=n, tz="UTC")
    gross = rng.normal(0.0, 0.002, size=(n, 4))
    gross[:, 0] += 0.002
    units = np.full((n, 4), 0.5)
    keys = [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")]
    return dates, gross, units, keys


def test_leave_one_out_rows(planted):
    dates, gross, units, keys = planted
    out = leave_one_out_spa(dates, gross, units, keys, ["g", "h"], costs=(0.0, 1.0), reps=500)
    assert list(out["left_out"]) == ["all", "year=2021", "year=2022", "g=a", "g=b", "h=x", "h=y"]
    assert out.loc[out.left_out == "all", "n_candidates"].item() == 4
    assert out.loc[out.left_out == "g=a", "n_candidates"].item() == 2
    assert out.loc[out.left_out == "year=2021", "n_days"].item() == 365


def test_leave_one_out_finds_the_planted_candidate(planted):
    dates, gross, units, keys = planted
    out = leave_one_out_spa(dates, gross, units, keys, ["g", "h"], costs=(0.0, 1.0), reps=500).set_index("left_out")
    assert out.loc["all", "p_cost_0"] < 0.01
    assert out.loc["g=b", "p_cost_0"] < 0.01          # a candidata plantada continua
    assert out.loc["h=y", "p_cost_0"] < 0.01
    assert out.loc["year=2021", "p_cost_0"] < 0.05    # o efeito está nos dois anos
    assert out.loc["g=a", "p_cost_0"] > 0.05          # sem ela, nada
    assert out.loc["h=x", "p_cost_0"] > 0.05
    assert out.loc["all", "threshold_cost_bps"] >= 0.0


def test_leave_one_out_p_grows_with_cost(planted):
    dates, gross, units, keys = planted
    out = leave_one_out_spa(dates, gross, units, keys, ["g", "h"], costs=(0.0, 4.0, 40.0), reps=500).set_index("left_out")
    row = out.loc["all"]
    assert row["p_cost_0"] <= row["p_cost_4"] <= row["p_cost_40"]
    assert row["p_cost_40"] > 0.5


@pytest.fixture(scope="module")
def persistence():
    """100 dias (metade em 2021, metade em 2022). Cada operação = 2 unidades; custo de 5 bps por lado = 1e-3 por operação."""
    n, half = 100, 50
    gross = np.zeros((n, 4))
    gross[:, 0] = 0.002                     # A: líquido positivo nas duas metades
    gross[:half, 1], gross[half:, 1] = 0.002, 0.0005   # B: positivo só na 1ª metade
    gross[:, 2] = 0.0005                    # C: negativo nas duas
    gross[:, 3] = 0.002                     # D: positivo, mas poucas operações
    units = np.tile([2.0, 2.0, 2.0, 0.2], (n, 1))
    trades = np.tile([1.0, 1.0, 1.0, 0.1], (n, 1))
    keys = [("1m", "BTCUSDT", "boll", 20, 2.0), ("1m", "BTCUSDT", "boll", 40, 2.0), ("1m", "BTCUSDT", "boll", 60, 2.0), ("1m", "BTCUSDT", "boll", 80, 2.0)]
    years = np.array([2021] * half + [2022] * half)
    names = ["freq", "symbol", "strategy", "window", "k"]
    return fixed_param_persistence(gross, units, trades, keys, names, half, years, cost_bps=5.0, min_trades=30)


def test_persistence_per_half_values(persistence):
    table, _ = persistence
    a, b, c = table.iloc[0], table.iloc[1], table.iloc[2]
    assert a["cbe_h1"] == pytest.approx(10.0) and a["cbe_h2"] == pytest.approx(10.0)
    assert a["net_h1_bps"] == pytest.approx(500.0) and a["net_h2_bps"] == pytest.approx(500.0)
    assert b["cbe_h2"] == pytest.approx(2.5)
    assert b["net_h1_bps"] > 0 > b["net_h2_bps"]
    assert c["net_h1_bps"] < 0 and c["net_h2_bps"] < 0
    assert a["trades_h1"] == 50 and a["trades_all"] == 100


def test_persistence_years_and_summary(persistence):
    table, summary = persistence
    assert list(table["years_positive"]) == [2, 1, 0, 2]
    assert summary["n_combinations"] == 4
    assert summary["positive_net_all"] == 3                       # A, B (soma positiva) e D
    assert summary["positive_net_h1_with_min_trades"] == 2        # A e B (D tem só 5 operações por metade)
    assert summary["of_those_positive_net_h2"] == 1               # só A repete
    assert summary["n_with_min_trades_both_halves"] == 3
    assert np.isfinite(summary["spearman_cbe_h1_h2"])


def test_persistence_zero_gross_has_zero_cbe():
    gross, units, trades = np.zeros((20, 1)), np.ones((20, 1)), np.ones((20, 1))
    table, _ = fixed_param_persistence(gross, units, trades, [("1m", "X", "boll", 20, 2.0)], ["freq", "symbol", "strategy", "window", "k"], 10,
                                       np.array([2021] * 10 + [2022] * 10), min_trades=5)
    assert table["cbe_all"].item() == 0.0
    assert table["years_positive"].item() == 0
