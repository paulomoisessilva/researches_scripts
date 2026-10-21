"""Testes dos momentos por janela, do salvamento e das tabelas por ano."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import sharpe_ratio
from src.walk_forward import WFResult, save_walk_forward_results, window_moments
from src.wf_analysis import (
    ALL_PERIOD,
    collect_runs,
    metrics_from_moments,
    percentile_rank,
    run_tables,
    sharpe_from_moments,
    sharpe_pivot,
)


def _result(strategy: str, window_id: int, day: str, rets: np.ndarray, seed=None) -> WFResult:
    idx = pd.date_range(day, periods=len(rets), freq="15min", tz="UTC")
    ret = pd.Series(rets, index=idx)
    pos = pd.Series(np.where(np.arange(len(rets)) % 8 < 4, 1.0, 0.0), index=idx)
    params = {"window": 20, "k": 2.0}
    if seed is not None:
        params = {"seed": seed, "trade_prob": 0.05}
    return WFResult(
        window_id=window_id,
        strategy=strategy,
        symbol="BTCUSDT",
        interval="15m",
        best_params=params,
        val_metric=1.0,
        test_equity=pd.Series(np.exp(ret.cumsum()), index=idx),
        test_returns=ret,
        test_gross_returns=ret,
        test_positions=pos,
        test_trades=pd.DataFrame({"entry_idx": [0], "exit_idx": [1]}),
        test_metrics={
            "strategy": strategy, "symbol": "BTCUSDT", "interval": "15m",
            "sharpe": 1.0, "number_of_trades": 1,
        },
    )


def _fake_run(tmp_path, with_random_equity=True):
    rng = np.random.default_rng(3)
    days = ["2021-12-30", "2021-12-31", "2022-01-01", "2022-01-02"]
    results = []
    for wid, day in enumerate(days):
        results.append(_result("bollinger", wid, day, rng.normal(2e-5, 1e-3, 96)))
        for seed in range(5):
            results.append(_result("random", wid, day, rng.normal(0, 1e-3, 96), seed=seed))
    save_walk_forward_results(results, tmp_path, tmp_path, save_random_equity=with_random_equity)
    return results


class TestMoments:
    def test_sharpe_from_moments_matches_sharpe_ratio(self):
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(1e-5, 1e-3, 500))
        got = sharpe_from_moments(len(r), r.sum(), (r ** 2).sum(), "15m")
        assert got == pytest.approx(sharpe_ratio(r, "15m"))

    def test_window_moments(self):
        res = _result("bollinger", 0, "2022-01-01", np.full(96, 1e-4))
        m = window_moments(res)
        assert m["test_date"] == "2022-01-01"
        assert m["n_bars"] == 96
        assert m["sum_ret"] == pytest.approx(96e-4)
        assert m["turnover_units"] == 23.0  # 12 descidas + 11 subidas em 96 barras

    def test_zero_variance_gives_zero_sharpe(self):
        assert sharpe_from_moments(10, 0.0, 0.0, "15m") == 0.0


class TestSaveAndTables:
    def test_save_writes_moment_columns(self, tmp_path):
        _fake_run(tmp_path)
        df = pd.read_csv(tmp_path / "walk_forward_results.csv")
        for col in ("test_date", "n_bars", "sum_ret", "sum_sq_ret", "sum_gross_ret", "turnover_units"):
            assert col in df.columns

    def test_random_equity_can_be_omitted(self, tmp_path):
        _fake_run(tmp_path, with_random_equity=False)
        eq = pd.read_parquet(tmp_path / "walk_forward_equity.parquet")
        assert set(eq["strategy"]) == {"bollinger"}

    def test_random_equity_kept_by_default(self, tmp_path):
        _fake_run(tmp_path, with_random_equity=True)
        eq = pd.read_parquet(tmp_path / "walk_forward_equity.parquet")
        assert "random" in set(eq["strategy"])

    def test_yearly_table_splits_by_year_and_matches_overall(self, tmp_path):
        results = _fake_run(tmp_path)
        df = pd.read_csv(tmp_path / "walk_forward_results.csv")
        real, dist = run_tables(df)

        periods = set(real["period"])
        assert periods == {ALL_PERIOD, "2021", "2022"}
        by = real.set_index(["strategy", "period"])
        assert by.loc[("bollinger", "2021"), "n_days"] == 2
        assert by.loc[("bollinger", "2022"), "n_days"] == 2

        # Retorno composto do ALL = produto dos anos
        tot = (1 + by.loc[("bollinger", "2021"), "total_return"]) * (
            1 + by.loc[("bollinger", "2022"), "total_return"]
        )
        assert tot - 1 == pytest.approx(by.loc[("bollinger", ALL_PERIOD), "total_return"])

        # Sharpe do ALL bate com o Sharpe das barras concatenadas
        rets = pd.concat([r.test_returns for r in results if r.strategy == "bollinger"])
        assert by.loc[("bollinger", ALL_PERIOD), "sharpe"] == pytest.approx(
            sharpe_ratio(rets, "15m")
        )

    def test_random_distribution_and_percentile(self, tmp_path):
        _fake_run(tmp_path)
        df = pd.read_csv(tmp_path / "walk_forward_results.csv")
        real, dist = run_tables(df)
        row_all = dist[dist["period"] == ALL_PERIOD].iloc[0]
        assert row_all["n_seeds"] == 5
        assert row_all["sharpe_p05"] <= row_all["sharpe_p50"] <= row_all["sharpe_p95"]
        pct = real.loc[real["period"] == ALL_PERIOD, "pct_random_below_sharpe"].iloc[0]
        assert 0.0 <= pct <= 1.0

    def test_pivot(self, tmp_path):
        _fake_run(tmp_path)
        df = pd.read_csv(tmp_path / "walk_forward_results.csv")
        real, _ = run_tables(df)
        pv = sharpe_pivot(real)
        assert list(pv.columns)[-1] == ALL_PERIOD
        assert "bollinger" in pv.index


def test_metrics_from_moments_break_even():
    grp = pd.DataFrame({
        "n_bars": [96], "sum_ret": [0.01], "sum_sq_ret": [0.0001],
        "sum_gross_ret": [0.02], "turnover_units": [10.0],
        "test_date": ["2022-01-01"], "number_of_trades": [3],
    })
    m = metrics_from_moments(grp, "15m")
    assert m["break_even_bps"] == pytest.approx(0.02 * 10_000 / 10.0)


class TestWindowParallelism:
    """Paralelizar entre janelas deve dar exatamente o mesmo resultado que o modo sequencial."""

    @staticmethod
    def _setup():
        from src.indicators import compute_all_features

        rng = np.random.default_rng(11)
        n = 40 * 96
        close = 30_000 * np.exp(np.cumsum(rng.normal(0, 6e-4, n)))
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
                "bollinger_windows": [20], "bollinger_k": [2.0, 3.0],
                "regression_windows": [50], "regression_k": [2.0, 3.0],
            },
            "strategy": {
                "entry_long_z": [0.8, 1.0], "entry_short_z": [0.8], "exit_long_z": [0.0],
                "exit_short_z": [0.0],
            },
            "walk_forward": {
                "train_days": 10, "validation_days": 3, "test_days": 1, "step_days": 4,
                "selection_metric": "net_sharpe", "close_position_end_of_day": True,
            },
        }
        return df, compute_all_features(df, cfg), cfg

    @pytest.mark.parametrize("strategy", ["bollinger", "regression_bands", "return_residuals"])
    def test_parallel_matches_sequential(self, strategy):
        from src.walk_forward import run_walk_forward

        df, df_feat, cfg = self._setup()
        kwargs = dict(df=df, strategy_name=strategy, cfg=cfg, symbol="BTCUSDT",
                      interval="15m", fee_bps=2.0, slippage_bps=2.0, df_feat=df_feat)
        seq = run_walk_forward(**kwargs, n_workers=1)
        par = run_walk_forward(**kwargs, n_workers=2)

        assert len(seq) == len(par) > 3
        for a, b in zip(sorted(seq, key=lambda r: r.window_id), sorted(par, key=lambda r: r.window_id)):
            assert a.window_id == b.window_id
            assert a.best_params == b.best_params
            pd.testing.assert_series_equal(a.test_returns, b.test_returns)


class TestCbeVsRandom:
    def test_percentile_rank_counts_ties_as_half(self):
        sample = pd.Series([0.0, 0.0, 0.0, 2.0])
        assert percentile_rank(sample, 0.0) == pytest.approx(0.375)  # (0 + 0.5*3) / 4
        assert percentile_rank(sample, 5.0) == 1.0
        assert percentile_rank(sample, -1.0) == 0.0
        assert np.isnan(percentile_rank(sample, float("nan")))

    def test_real_table_has_cbe_columns(self, tmp_path):
        _fake_run(tmp_path)
        df = pd.read_csv(tmp_path / "walk_forward_results.csv")
        real, dist = run_tables(df)
        for col in ("random_cbe_p05", "random_cbe_p50", "random_cbe_p95", "pct_random_below_cbe"):
            assert col in real.columns
        assert "break_even_bps_p50" in dist.columns
        pct = real["pct_random_below_cbe"].dropna()
        assert ((pct >= 0) & (pct <= 1)).all()

    def test_collect_runs_annotates_meta(self, tmp_path):
        import json

        run = tmp_path / "run_a"
        run.mkdir()
        _fake_run(run)
        (run / "run_meta.json").write_text(json.dumps({
            "run_id": "run_a", "train_days": 21, "validation_days": 7, "test_days": 1,
            "step_days": 1, "fee_bps": 2.0, "slippage_bps": 2.0,
        }))
        real, dist = collect_runs([run])
        assert set(real["train_days"]) == {21} and set(dist["fee_bps"]) == {2.0}
        assert real.columns[0] == "run_id"


class TestMultiAsset:
    """Vários ativos na mesma run não podem sobrescrever uns aos outros."""

    def test_save_keeps_rows_of_both_symbols(self, tmp_path):
        def res(symbol, wid, day):
            r = _result("bollinger", wid, day, np.random.default_rng(wid).normal(0, 1e-3, 96))
            r.symbol = symbol
            r.test_metrics["symbol"] = symbol
            return r

        # Mesmos window_id/strategy para dois ativos, salvos em chamadas separadas.
        save_walk_forward_results([res("AAA", 0, "2022-01-01"), res("AAA", 1, "2022-01-02")], tmp_path, tmp_path)
        save_walk_forward_results([res("BBB", 0, "2022-01-01"), res("BBB", 1, "2022-01-02")], tmp_path, tmp_path)

        wf = pd.read_csv(tmp_path / "walk_forward_results.csv")
        params = pd.read_csv(tmp_path / "best_parameters_by_window.csv")
        eq = pd.read_parquet(tmp_path / "walk_forward_equity.parquet")
        assert len(wf) == 4 and set(wf["symbol"]) == {"AAA", "BBB"}
        assert len(params) == 4 and set(params["symbol"]) == {"AAA", "BBB"}
        assert set(eq["symbol"]) == {"AAA", "BBB"}

    def test_resave_replaces_instead_of_duplicating(self, tmp_path):
        def res(wid, day):
            return _result("bollinger", wid, day, np.random.default_rng(wid).normal(0, 1e-3, 96))

        batch = [res(0, "2022-01-01"), res(1, "2022-01-02")]
        save_walk_forward_results(batch, tmp_path, tmp_path)
        save_walk_forward_results(batch, tmp_path, tmp_path)  # retomada da mesma run
        assert len(pd.read_csv(tmp_path / "walk_forward_results.csv")) == 2

    def test_run_experiment_two_symbols_end_to_end(self, tmp_path):
        import json

        from src.wf_runner import run_experiment

        df, df_feat, cfg = TestWindowParallelism._setup()
        cfg["walk_forward"].update({
            "runs_dir": str(tmp_path / "wf"), "oos_start": "2022-01-20", "step_days": 6,
            "save_random_equity": False, "parallel_workers": 1,
        })
        cfg["random_baseline"] = {"n_seeds": 3, "seed_base": 0, "trade_prob": 0.05}
        cfg["data"] = {"start_date": "2022-01-01", "end_date": "2022-02-09"}
        inputs = {("AAA", "15m"): (df, df_feat), ("BBB", "15m"): (df, df_feat)}

        run_dir = run_experiment(cfg, inputs, ["bollinger", "random"], 2.0, 2.0, fresh=True)
        wf = pd.read_csv(run_dir / "walk_forward_results.csv")
        for symbol in ("AAA", "BBB"):
            sub = wf[wf["symbol"] == symbol]
            assert set(sub["strategy"]) == {"bollinger", "random"}
            assert sub.loc[sub["strategy"] == "random", "seed"].nunique() == 3
        n_boll = wf[wf["strategy"] == "bollinger"].groupby("symbol").size()
        assert n_boll["AAA"] == n_boll["BBB"] > 1

        meta = json.loads((run_dir / "run_meta.json").read_text())
        assert meta["symbols"] == ["AAA", "BBB"]

        real, dist = run_tables(wf)
        assert set(real["symbol"]) == {"AAA", "BBB"} and set(dist["symbol"]) == {"AAA", "BBB"}
        pv = sharpe_pivot(real)
        assert list(pv.index.names) == ["symbol", "strategy"]

    def test_extending_a_run_with_another_symbol_keeps_both(self, tmp_path):
        import json

        from src.wf_runner import run_experiment

        df, df_feat, cfg = TestWindowParallelism._setup()
        cfg["walk_forward"].update({
            "runs_dir": str(tmp_path / "wf"), "oos_start": "2022-01-20", "step_days": 6,
            "save_random_equity": False, "parallel_workers": 1, "random_cache": False,
        })
        cfg["random_baseline"] = {"n_seeds": 2, "seed_base": 0, "trade_prob": 0.05}
        cfg["data"] = {"start_date": "2022-01-01", "end_date": "2022-02-09"}

        first = run_experiment(cfg, {("AAA", "15m"): (df, df_feat)}, ["bollinger", "random"], 2.0, 2.0)
        second = run_experiment(cfg, {("BBB", "15m"): (df, df_feat)}, ["bollinger", "random"], 2.0, 2.0)
        assert first == second  # mesma pasta: o run_id não depende do ativo

        meta = json.loads((second / "run_meta.json").read_text())
        assert meta["symbols"] == ["AAA", "BBB"]
        assert set(meta["n_windows"]) >= {"AAA_15m_bollinger", "BBB_15m_bollinger", "AAA_15m_random", "BBB_15m_random"}
        wf = pd.read_csv(second / "walk_forward_results.csv")
        assert set(wf["symbol"]) == {"AAA", "BBB"}
