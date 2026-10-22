"""Funções da análise v2: SPA por custo, tabelas, parâmetros fixos e perfil das entradas."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest
from src.indicators import compute_all_features
from src.inference import bootstrap_means, reality_check_spa, stationary_bootstrap_indices
from src.strategies import compute_band_z, signal_from_z
from src.v2_study import (
    COST_GRID,
    KEY,
    N_CELLS,
    accumulate_profile,
    bootstrap_counts,
    by_year_table,
    candidate_table,
    daily_matrices,
    fixed_param_daily,
    half_split_agreement,
    profile_tables,
    spa_curve,
    threshold_cost,
)


def _panel(n=300, k=6, seed=0, edge=0.0):
    rng = np.random.default_rng(seed)
    units = rng.integers(2, 12, size=(n, k)).astype(float)
    gross = rng.normal(edge, 4e-4, size=(n, k)) + units * 1e-4 * rng.uniform(0.2, 1.2, k)
    return gross, units


class TestSpaCurve:
    def test_bootstrap_means_from_counts_equal_the_resampled_means(self):
        n, reps = 200, 400
        gross, units = _panel(n)
        counts = bootstrap_counts(n, reps, 10.0, 3)
        idx = stationary_bootstrap_indices(n, reps, 10.0, np.random.default_rng(3))
        for c in (0.0, 2.5, 8.0):
            d = gross - c * units / 1e4
            assert np.allclose((counts @ gross - c * (counts @ units) / 1e4) / n, bootstrap_means(d, idx), atol=1e-14)

    def test_p_values_match_the_direct_route_at_every_cost(self):
        n, reps = 250, 600
        gross, units = _panel(n, edge=2e-5)
        counts = bootstrap_counts(n, reps, 10.0, 1)
        idx = stationary_bootstrap_indices(n, reps, 10.0, np.random.default_rng(1))
        costs = np.array([0.0, 1.0, 3.0, 5.0])
        curve = spa_curve(gross, units, costs, counts=counts)
        for row in curve.itertuples():
            d = gross - row.cost_bps_per_side * units / 1e4
            ref = reality_check_spa(d, bootstrap_means(d, idx))
            assert row.spa_pvalue_consistent == pytest.approx(ref.spa_pvalue_consistent, abs=1e-12)
            assert row.rc_pvalue == pytest.approx(ref.rc_pvalue, abs=1e-12)
            assert row.spa_pvalue_lower == pytest.approx(ref.spa_pvalue_lower, abs=1e-12)
            assert row.spa_pvalue_upper == pytest.approx(ref.spa_pvalue_upper, abs=1e-12)

    def test_p_value_grows_with_cost_and_a_planted_edge_is_found_at_low_cost(self):
        n = 400
        rng = np.random.default_rng(5)
        units = np.full((n, 3), 4.0)
        gross = rng.normal(0, 3e-4, (n, 3))
        gross[:, 0] += 4 * 3.0e-4 / 4 * 4 / 4 + 0.0  # candidata 0: bruto médio de 3e-4 por dia (~0,75 bps por unidade)
        curve = spa_curve(gross, units, COST_GRID, reps=800, mean_block=5.0, seed=2)
        p = curve["spa_pvalue_consistent"].to_numpy()
        assert p[0] < 0.05 and p[-1] > 0.5
        assert np.all(np.diff(p) >= -0.02)  # praticamente monótona
        thr = threshold_cost(curve)
        assert thr["rejects_at_zero_cost"] and 0.0 <= thr["threshold_cost_bps"] < 8.0

    def test_best_candidate_label_uses_the_keys(self):
        gross, units = _panel(100)
        keys = [("1m", "BTC", "bollinger", i) for i in range(gross.shape[1])]
        curve = spa_curve(gross, units, np.array([0.0]), reps=200, keys=keys)
        assert curve["best_candidate"].iloc[0].startswith("1m|BTC|bollinger|")


class TestThreshold:
    @staticmethod
    def _curve(ps):
        return pd.DataFrame({"cost_bps_per_side": np.arange(len(ps)) * 0.25, "spa_pvalue_consistent": ps})

    def test_largest_cost_with_p_below_alpha(self):
        thr = threshold_cost(self._curve([0.001, 0.01, 0.03, 0.049, 0.06, 0.4]))
        assert thr["threshold_cost_bps"] == 0.75 and thr["monotone"] and thr["rejects_at_zero_cost"]

    def test_no_rejection_even_at_zero_cost(self):
        thr = threshold_cost(self._curve([0.2, 0.4, 0.9]))
        assert thr["threshold_cost_bps"] is None and not thr["rejects_at_zero_cost"]

    def test_non_monotone_curve_is_flagged(self):
        thr = threshold_cost(self._curve([0.01, 0.2, 0.15, 0.6]))
        assert not thr["monotone"] and thr["n_decreases"] == 1


class TestTables:
    def test_cbe_and_net_returns_by_hand(self):
        gross = np.array([[0.01, -0.02], [0.01, -0.01]])
        units = np.array([[10.0, 10.0], [10.0, 10.0]])
        trades = np.array([[5.0, 5.0], [5.0, 5.0]])
        tab = candidate_table(gross, units, trades, [("a",), ("b",)], ["id"])
        assert tab["cbe_bps"].tolist() == pytest.approx([1e4 * 0.02 / 20, 0.0])
        assert tab["trades_per_day"].tolist() == [5.0, 5.0]
        assert tab["gross_pct"].iloc[0] == pytest.approx((np.exp(0.02) - 1) * 100)
        assert tab["net_pct_at_2bps"].iloc[0] == pytest.approx((np.exp(0.02 - 2 * 20 / 1e4) - 1) * 100)

    def test_by_year_splits_the_days(self):
        dates = pd.date_range("2021-12-30", periods=5, freq="D")
        gross = np.array([[0.01]] * 5)
        out = by_year_table(dates, gross, np.ones((5, 1)), np.ones((5, 1)), [("x",)], ["id"])
        assert out["year"].tolist() == [2021, 2022] and out["n_days"].tolist() == [2, 3]

    def test_daily_matrices_align_candidates_and_reject_missing_days(self):
        days = [f"2021-02-0{d}" for d in (1, 2, 3)]
        rows = []
        for sym in ("BTC", "ETH"):
            for i, d in enumerate(days):
                rows.append({"freq": "15m", "symbol": sym, "strategy": "bollinger", "lookback": 3, "test_date": d,
                             "sum_gross_ret": 0.001 * (i + 1), "turnover_units": 4.0, "number_of_trades": 2})
        win = pd.DataFrame(rows)
        m = daily_matrices(win)
        assert m["gross"].shape == (3, 2) and m["keys"][0][:2] == ("15m", "BTC")
        with pytest.raises(ValueError, match="coincidem"):
            daily_matrices(win.iloc[:-1])


class TestFixedParams:
    def test_daily_sums_match_the_backtest_engine(self):
        rng = np.random.default_rng(9)
        n_days, bpd = 30, 96
        n = n_days * bpd
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.995 * x[i - 1] + rng.normal(0, 3e-3)
        close = 30_000 * np.exp(x)
        ts = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
        df = pd.DataFrame({"timestamp": ts, "open": close * (1 + rng.normal(0, 2e-4, n)), "high": close * 1.001,
                           "low": close * 0.999, "close": close, "volume": 1.0})
        cfg = {"features": {"bollinger_windows": [24], "bollinger_k": [1.5], "compact_bands": True}}
        feat = compute_all_features(df, cfg, strategy_names=["bollinger"])
        first_idx = 5 * bpd
        g, u, t = fixed_param_daily(feat, "bollinger", 24, 1.5, 1.0, first_idx, 20, bpd)

        sig = signal_from_z(compute_band_z(feat, "bollinger", 24, 1.5), entry_long_z=1.0, entry_short_z=1.0, name="b")
        res = run_backtest(df, sig, BacktestConfig(strategy_name="b", symbol="X", interval="15m", close_eod=False))
        day = pd.DatetimeIndex(df["timestamp"]).normalize()
        sl = slice(first_idx, first_idx + 20 * bpd)
        ref_g = res.gross_returns.iloc[sl].groupby(day[sl]).sum().to_numpy()
        ref_u = res.trade_units.iloc[sl].groupby(day[sl]).sum().to_numpy()
        assert np.allclose(g, ref_g, atol=1e-15, rtol=0) and np.array_equal(u, ref_u)
        assert u.sum() > 0
        # operações = entradas: cada entrada soma 1 unidade (e cada saída, outra)
        assert t.sum() == pytest.approx(u.sum() / 2, abs=1.0)


def _trade_frame():
    # segunda 10:05 (bruto +5 bps), segunda 10:40 (+3 bps), terça 23:59 (−4 bps), domingo 00:00 (+1 bp)
    ts = pd.DatetimeIndex(["2021-02-01 10:05", "2021-02-01 10:40", "2021-02-02 23:59", "2021-02-07 00:00"], tz="UTC")
    return ts, np.array([5e-4, 3e-4, -4e-4, 1e-4])


class TestEntryProfile:
    def test_entries_are_binned_by_day_of_week_and_utc_hour(self):
        ts, g = _trade_frame()
        a = accumulate_profile(ts, g, pd.Timestamp("2021-02-01", tz="UTC"), 7)
        assert a["n"].shape == (7, N_CELLS) and a["n"].sum() == 4
        mon10 = 0 * 24 + 10
        assert a["n"][0, mon10] == 2 and a["pos_gross"][0, mon10] == 2
        assert a["n"][1, 1 * 24 + 23] == 1 and a["pos_gross"][1, 1 * 24 + 23] == 0
        assert a["n"][6, 6 * 24 + 0] == 1
        assert a["sum_gross"].sum() == pytest.approx(5e-4 + 3e-4 - 4e-4 + 1e-4)

    def test_naive_utc_timestamps_give_the_same_counts(self):
        ts, g = _trade_frame()
        aware = accumulate_profile(ts, g, pd.Timestamp("2021-02-01", tz="UTC"), 7)
        naive = accumulate_profile(ts.tz_localize(None), g, pd.Timestamp("2021-02-01"), 7)
        for key in aware:
            assert np.array_equal(aware[key], naive[key])

    def test_net_positive_counts_use_a_round_trip_cost(self):
        ts, g = _trade_frame()
        a = accumulate_profile(ts, g, pd.Timestamp("2021-02-01", tz="UTC"), 7, costs=(1.0, 2.0))
        # líquido = bruto − 2c: a 1 bp, 5 e 3 bps ainda ganham (3 e 1) e +1 bp zera; a 2 bps só o de 5 bps ganha (+1)
        assert a["pos_c1"].sum() == 2 and a["pos_c2"].sum() == 1

    def test_trades_outside_the_test_days_are_dropped(self):
        ts = pd.DatetimeIndex(["2021-01-30 10:00", "2021-02-02 10:00", "2021-02-03 10:00"], tz="UTC")
        a = accumulate_profile(ts, np.array([1e-4, 1e-4, 1e-4]), pd.Timestamp("2021-02-01", tz="UTC"), 2)
        assert a["n"].sum() == 1 and a["n"][1].sum() == 1  # só 2021-02-02 está nos 2 dias de teste

    def test_tables_totals_and_confidence_intervals(self):
        rng = np.random.default_rng(4)
        n_days = 120
        ts = pd.DatetimeIndex(pd.Timestamp("2021-02-01", tz="UTC") + pd.to_timedelta(rng.integers(0, n_days * 1440, 5000), unit="m"))
        g = rng.normal(2e-5, 6e-4, 5000)
        a = accumulate_profile(ts, g, pd.Timestamp("2021-02-01", tz="UTC"), n_days)
        tabs = profile_tables(a, counts=bootstrap_counts(n_days, 300, 5.0, 0), min_n=5)
        assert tabs["hour"]["n_entries"].sum() == 5000 == tabs["weekday"]["n_entries"].sum() == tabs["cell"]["n_entries"].sum()
        h = tabs["hour"]
        assert len(h) == 24 and len(tabs["weekday"]) == 7 and len(tabs["cell"]) == 168
        assert ((h["share_pos_gross_ci_low"] <= h["share_pos_gross"]) & (h["share_pos_gross"] <= h["share_pos_gross_ci_high"])).all()
        assert h["share_pos_gross"].between(0.3, 0.7).all()  # ruído em torno de 50%
        row = h.iloc[10]
        assert row["mean_gross_bps"] == pytest.approx(1e4 * g[ts.hour == 10].mean(), rel=1e-9)
        assert row["cbe_bps"] == pytest.approx(max(row["mean_gross_bps"], 0.0) / 2)

    def test_bins_with_few_entries_are_blank(self):
        ts, g = _trade_frame()
        a = accumulate_profile(ts, g, pd.Timestamp("2021-02-01", tz="UTC"), 7)
        tab = profile_tables(a, min_n=30)["hour"]
        assert tab["share_pos_gross"].isna().all() and tab["n_entries"].sum() == 4

    def test_half_split_agreement_is_perfect_for_identical_halves_and_weak_for_noise(self):
        rng = np.random.default_rng(1)
        base = rng.random((1, N_CELLS)) * 50
        same = {"n": np.vstack([base, base]) + 10, "pos_gross": np.vstack([base * 0.5, base * 0.5]),
                "sum_gross": np.vstack([base * 1e-5, base * 1e-5])}
        out = half_split_agreement(same, 1)
        assert (out["spearman"] > 0.999).all()
        noise = {"n": np.full((2, N_CELLS), 40.0), "pos_gross": rng.binomial(40, 0.5, (2, N_CELLS)).astype(float),
                 "sum_gross": rng.normal(0, 1e-3, (2, N_CELLS))}
        assert (half_split_agreement(noise, 1)["p_value"] > 0.001).all()
