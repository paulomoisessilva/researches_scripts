"""Testes do bootstrap estacionário, Reality Check e SPA."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest
from src.inference import (
    bootstrap_means,
    candidate_panels,
    daily_matrix,
    net_returns_at_cost,
    reality_check_spa,
    spa_test,
    stationary_bootstrap_indices,
)


class TestStationaryBootstrap:
    def test_shape_and_range(self):
        idx = stationary_bootstrap_indices(300, 50, 8.0, np.random.default_rng(0))
        assert idx.shape == (50, 300)
        assert idx.min() >= 0 and idx.max() < 300

    @pytest.mark.parametrize("mean_block", [3.0, 10.0, 25.0])
    def test_mean_block_length(self, mean_block):
        n = 4000
        idx = stationary_bootstrap_indices(n, 40, mean_block, np.random.default_rng(1))
        # um novo bloco quebra a sequência idx[t] == idx[t-1] + 1 (exceto coincidências ~1/n)
        breaks = idx[:, 1:] != (idx[:, :-1] + 1) % n
        measured = n / (breaks.sum(axis=1).mean() + 1)
        assert measured == pytest.approx(mean_block, rel=0.15)

    def test_block_one_is_iid_like(self):
        idx = stationary_bootstrap_indices(1000, 10, 1.0, np.random.default_rng(2))
        assert (idx[:, 1:] == (idx[:, :-1] + 1) % 1000).mean() < 0.01

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            stationary_bootstrap_indices(1, 10, 5.0, np.random.default_rng(0))
        with pytest.raises(ValueError):
            stationary_bootstrap_indices(100, 10, 0.5, np.random.default_rng(0))

    def test_bootstrap_means_chunking_is_equivalent(self):
        rng = np.random.default_rng(3)
        d = rng.normal(size=(120, 4))
        idx = stationary_bootstrap_indices(120, 37, 5.0, rng)
        full = d[idx].mean(axis=1)
        np.testing.assert_allclose(bootstrap_means(d, idx, chunk=10), full)


class TestRealityCheckSpa:
    def test_strong_edge_is_detected(self):
        rng = np.random.default_rng(10)
        d = rng.normal(0, 1, (1500, 12))
        d[:, 3] += 0.2
        r = spa_test(d, n_boot=1500, mean_block=5, seed=1)
        assert r.best_candidate == 3
        assert r.rc_pvalue < 0.02 and r.spa_pvalue_consistent < 0.02

    def test_all_bad_candidates_give_p_one(self):
        rng = np.random.default_rng(11)
        d = rng.normal(-0.1, 1, (1500, 12))
        r = spa_test(d, n_boot=1000, mean_block=5, seed=1)
        assert r.rc_pvalue > 0.9 and r.spa_pvalue_consistent == 1.0 and r.spa_statistic == 0.0

    def test_degenerate_zero_returns_give_p_one(self):
        r = spa_test(np.zeros((200, 5)), n_boot=200, mean_block=5, seed=1)
        assert r.rc_pvalue == 1.0 and r.spa_pvalue_consistent == 1.0

    def test_pvalue_ordering_lower_consistent_upper(self):
        rng = np.random.default_rng(12)
        for _ in range(8):
            d = rng.normal(rng.uniform(-0.1, 0.05, 15), 1, (600, 15))
            r = spa_test(d, n_boot=400, mean_block=5, seed=int(rng.integers(1000)))
            assert r.spa_pvalue_lower <= r.spa_pvalue_consistent + 1e-12
            assert r.spa_pvalue_consistent <= r.spa_pvalue_upper + 1e-12

    def test_deterministic_given_seed(self):
        d = np.random.default_rng(13).normal(size=(300, 6))
        a = spa_test(d, n_boot=300, seed=7)
        b = spa_test(d, n_boot=300, seed=7)
        assert a == b

    def test_null_rejection_rate_is_controlled_iid(self):
        rng = np.random.default_rng(14)
        rej_rc = rej_spa = 0
        sims = 200
        for s in range(sims):
            d = rng.normal(0, 1, (400, 15))  # média verdadeira 0: fronteira de H0
            r = spa_test(d, n_boot=299, mean_block=3, seed=s)
            rej_rc += r.rc_pvalue < 0.05
            rej_spa += r.spa_pvalue_consistent < 0.05
        assert rej_rc / sims <= 0.11 and rej_spa / sims <= 0.11

    def test_null_rejection_rate_is_controlled_with_serial_dependence(self):
        rng = np.random.default_rng(15)
        n, k, rho, sims = 600, 10, 0.5, 150
        rej = 0
        for s in range(sims):
            e = rng.normal(size=(n, k))
            x = np.zeros((n, k))
            for t in range(1, n):
                x[t] = rho * x[t - 1] + e[t]
            r = spa_test(x, n_boot=299, mean_block=10, seed=s)
            rej += r.spa_pvalue_consistent < 0.05
        assert rej / sims <= 0.12

    def test_reality_check_spa_accepts_precomputed_bootstrap(self):
        rng = np.random.default_rng(16)
        d = rng.normal(0, 1, (200, 5))
        idx = stationary_bootstrap_indices(200, 100, 4.0, np.random.default_rng(5))
        r = reality_check_spa(d, bootstrap_means(d, idx))
        assert r.n_obs == 200 and r.n_candidates == 5 and 0 <= r.rc_pvalue <= 1


class TestDailyReturns:
    def test_net_return_is_linear_in_cost_and_matches_backtest(self):
        rng = np.random.default_rng(20)
        n = 96 * 3
        close = 30_000 * np.exp(np.cumsum(rng.normal(0, 6e-4, n)))
        df = pd.DataFrame({
            "timestamp": pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC"),
            "open": close,
        })
        signal = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.2, 0.6, 0.2]))

        def run(fee, slip):
            return run_backtest(
                df, signal,
                BacktestConfig(strategy_name="x", symbol="S", interval="15m", fee_bps=fee, slippage_bps=slip),
            )

        gross = run(0.0, 0.0)
        net = run(2.0, 3.0)
        turnover = net.positions.diff().abs().fillna(0.0).sum()
        got = net_returns_at_cost(gross.returns.sum(), turnover, 5.0)
        assert got == pytest.approx(net.returns.sum(), rel=1e-12, abs=1e-12)

    def test_daily_matrix_uses_common_dates(self):
        d1 = pd.DataFrame({"test_date": ["2022-01-01", "2022-01-02", "2022-01-03"],
                           "symbol": "A", "strategy": "s", "sum_gross_ret": [0.01, 0.02, 0.03],
                           "turnover_units": [1.0, 2.0, 3.0]})
        d2 = pd.DataFrame({"test_date": ["2022-01-02", "2022-01-03", "2022-01-04"],
                           "symbol": "B", "strategy": "s", "sum_gross_ret": [0.0, 0.0, 0.0],
                           "turnover_units": [1.0, 1.0, 1.0]})
        panels = candidate_panels(pd.concat([d1, d2]), ["symbol", "strategy"])
        mat, keys, dates = daily_matrix(panels, cost_bps_per_side=10.0)
        assert list(dates) == ["2022-01-02", "2022-01-03"]
        assert keys == [("A", "s"), ("B", "s")]
        np.testing.assert_allclose(mat[:, 0], [0.02 - 2 * 0.001, 0.03 - 3 * 0.001])
        np.testing.assert_allclose(mat[:, 1], [-0.001, -0.001])


class TestTurnoverUnits:
    """A entrada na 1ª barra do dia tem custo e precisa entrar no turnover da janela."""

    @staticmethod
    def _day_two_slice():
        from src.walk_forward import _slice_backtest_result, window_moments, WFResult

        n = 96 * 3
        rng = np.random.default_rng(31)
        close = 30_000 * np.exp(np.cumsum(rng.normal(0, 6e-4, n)))
        df = pd.DataFrame({
            "timestamp": pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC"),
            "open": close,
        })
        signal = pd.Series(0.0, index=range(n))
        signal.iloc[96:140] = 1.0  # posição aberta já na 1ª barra do dia 2
        bt = BacktestConfig(strategy_name="x", symbol="S", interval="15m",
                            fee_bps=2.0, slippage_bps=2.0, close_eod=True)
        full = run_backtest(df, signal, bt)
        day2 = _slice_backtest_result(
            full, pd.Timestamp("2022-01-02", tz="UTC"), pd.Timestamp("2022-01-03", tz="UTC")
        )
        result = WFResult(
            window_id=1, strategy="x", symbol="S", interval="15m", best_params={}, val_metric=0.0,
            test_equity=day2.equity, test_returns=day2.returns, test_gross_returns=day2.gross_returns,
            test_positions=day2.positions, test_trades=day2.trades, test_metrics={},
            test_trade_units=day2.trade_units,
        )
        return day2, window_moments(result)

    def test_first_bar_entry_counts_as_traded_units(self):
        day2, moments = self._day_two_slice()
        assert day2.positions.iloc[0] == 1.0
        assert day2.positions.diff().abs().fillna(0).sum() == 1.0   # só a saída: perde a entrada
        assert day2.trade_units.sum() == 2.0                         # entrada + saída
        assert moments["turnover_units"] == 2.0

    def test_net_equals_gross_minus_units_times_cost(self):
        day2, moments = self._day_two_slice()
        expected = day2.gross_returns.sum() - moments["turnover_units"] * 4.0 / 10_000
        assert day2.returns.sum() == pytest.approx(expected, rel=1e-12, abs=1e-14)
