"""Equivalência do baseline aleatório otimizado (paralelo, em blocos, com cache)."""

import copy
import json

import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest
from src.metrics import compute_metrics
from src.random_baseline import random_cache_key, run_random_baseline_to_disk
from src.strategies import resolve_random_seeds, strategy_random
from src.walk_forward import (
    WFResult,
    _random_baseline_params,
    build_windows,
    save_walk_forward_results,
)
from src.wf_runner import run_experiment
from tests.test_wf_analysis import TestWindowParallelism

TABLES = ("walk_forward_results.csv", "best_parameters_by_window.csv", "walk_forward_oos_summary.csv")


def _setup(n_seeds: int = 5):
    df, df_feat, cfg = TestWindowParallelism._setup()
    cfg["walk_forward"].update({
        "train_days": 5, "validation_days": 3, "test_days": 1, "step_days": 1,
        "oos_start": "2022-01-12", "save_random_equity": False, "parallel_workers": 1,
    })
    cfg["random_baseline"] = {"n_seeds": n_seeds, "seed_base": 7, "trade_prob": 0.05}
    cfg["data"] = {"start_date": "2022-01-01", "end_date": "2022-02-09"}
    wf = cfg["walk_forward"]
    windows = build_windows(df, wf["train_days"], wf["validation_days"], wf["test_days"],
                            wf["step_days"], oos_start=wf["oos_start"])
    bt = BacktestConfig(strategy_name="random", symbol="AAA", interval="15m",
                        fee_bps=2.0, slippage_bps=2.0, close_eod=True)
    return df, df_feat, cfg, windows, bt


def _reference_random(df_feat_full, cfg, symbol, interval, windows, bt_cfg, wf_cfg):
    """Cópia do laço ORIGINAL (máscara de datas por janela, tudo em memória)."""
    results = []
    for seed in resolve_random_seeds(cfg):
        params = _random_baseline_params(cfg, seed)
        full_signal = strategy_random(df_feat_full, **params)
        for win in windows:
            ts = df_feat_full["timestamp"]
            test_mask = (ts >= win.test_start) & (ts < win.test_end)
            df_feat_test = df_feat_full.loc[test_mask].reset_index(drop=True)
            min_test_bars = 1 if wf_cfg.get("test_days", 1) == 1 else 10
            if len(df_feat_test) < min_test_bars:
                continue
            signal_test = full_signal.loc[test_mask].reset_index(drop=True)
            test_result = run_backtest(df_feat_test, signal_test, bt_cfg)
            test_metrics = compute_metrics(
                test_result.equity, test_result.returns, test_result.positions,
                test_result.trades, interval=interval, strategy_name="random", symbol=symbol,
                fee_bps=bt_cfg.fee_bps, slippage_bps=bt_cfg.slippage_bps,
                gross_returns=test_result.gross_returns,
            )
            results.append(WFResult(
                window_id=win.window_id, strategy="random", symbol=symbol, interval=interval,
                best_params=params, val_metric=float("nan"),
                test_equity=test_result.equity, test_returns=test_result.returns,
                test_gross_returns=test_result.gross_returns, test_positions=test_result.positions,
                test_trades=test_result.trades, test_metrics=test_metrics,
            ))
    return results


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("symbol", "seed", "param_seed", "window_id") if c in df.columns]
    return df.sort_values(keys).reset_index(drop=True)


def _assert_tables_equal(dir_a, dir_b):
    for name in TABLES:
        a = _sorted(pd.read_csv(dir_a / name))
        b = _sorted(pd.read_csv(dir_b / name))
        # Inteiros/ids/strings exatos; floats até 1 ulp (a soma do pandas muda de ordem
        # conforme o layout em memória da fatia: iloc em vez de máscara booleana).
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-15, obj=name)


class TestRandomEquivalence:
    @pytest.mark.parametrize("workers,block", [(1, 2), (2, 1), (2, 10)])
    def test_matches_original_loop(self, tmp_path, workers, block):
        df, df_feat, cfg, windows, bt = _setup()
        ref_dir, new_dir = tmp_path / "ref", tmp_path / "new"

        ref = _reference_random(df_feat, cfg, "AAA", "15m", windows, bt, cfg["walk_forward"])
        save_walk_forward_results(ref, ref_dir, ref_dir, save_random_equity=False)

        n = run_random_baseline_to_disk(
            df_feat, cfg, "AAA", "15m", windows, bt, cfg["walk_forward"], new_dir,
            n_workers=workers, save_random_equity=False, block_seeds=block,
        )
        assert n == 5 and len(ref) == 5 * len(windows)
        _assert_tables_equal(ref_dir, new_dir)

    def test_equity_curves_match_when_saved(self, tmp_path):
        df, df_feat, cfg, windows, bt = _setup(n_seeds=2)
        ref_dir, new_dir = tmp_path / "ref", tmp_path / "new"
        ref = _reference_random(df_feat, cfg, "AAA", "15m", windows, bt, cfg["walk_forward"])
        save_walk_forward_results(ref, ref_dir, ref_dir, save_random_equity=True)
        run_random_baseline_to_disk(
            df_feat, cfg, "AAA", "15m", windows, bt, cfg["walk_forward"], new_dir,
            n_workers=2, save_random_equity=True, block_seeds=1,
        )
        a = pd.read_parquet(ref_dir / "walk_forward_equity.parquet")
        b = pd.read_parquet(new_dir / "walk_forward_equity.parquet")
        keys = ["seed", "window_id", "timestamp"]
        pd.testing.assert_frame_equal(
            a.sort_values(keys).reset_index(drop=True),
            b.sort_values(keys).reset_index(drop=True), check_exact=False, rtol=1e-12,
        )


class TestRandomCache:
    def test_key_ignores_lookback_but_not_cost_or_oos(self):
        df, df_feat, cfg, windows, bt = _setup()
        base = random_cache_key(cfg, "AAA", "15m", 2.0, 2.0, df_feat)

        other_val = copy.deepcopy(cfg)
        other_val["walk_forward"]["validation_days"] = 9
        other_val["walk_forward"]["train_days"] = 30
        assert random_cache_key(other_val, "AAA", "15m", 2.0, 2.0, df_feat) == base

        assert random_cache_key(cfg, "AAA", "15m", 5.0, 2.0, df_feat) != base
        assert random_cache_key(cfg, "BBB", "15m", 2.0, 2.0, df_feat) != base
        other_oos = copy.deepcopy(cfg)
        other_oos["walk_forward"]["oos_start"] = "2022-01-13"
        assert random_cache_key(other_oos, "AAA", "15m", 2.0, 2.0, df_feat) != base
        other_seed = copy.deepcopy(cfg)
        other_seed["random_baseline"]["trade_prob"] = 0.1
        assert random_cache_key(other_seed, "AAA", "15m", 2.0, 2.0, df_feat) != base

    def test_key_depends_on_lookback_without_oos_start(self):
        df, df_feat, cfg, windows, bt = _setup()
        cfg["walk_forward"].pop("oos_start")
        a = random_cache_key(cfg, "AAA", "15m", 2.0, 2.0, df_feat)
        cfg["walk_forward"]["validation_days"] = 9
        assert random_cache_key(cfg, "AAA", "15m", 2.0, 2.0, df_feat) != a

    def test_second_lookback_reuses_random(self, tmp_path, monkeypatch):
        import src.wf_runner as runner

        df, df_feat, cfg, windows, bt = _setup()
        cfg["walk_forward"]["runs_dir"] = str(tmp_path / "wf")
        inputs = {("AAA", "15m"): (df, df_feat)}

        run_a = run_experiment(cfg, inputs, ["random"], 2.0, 2.0, validation_days=3, fresh=True)
        meta_a = json.loads((run_a / "run_meta.json").read_text())
        assert meta_a["random_cache"]["AAA_15m"].startswith("miss")

        def boom(*args, **kwargs):
            raise AssertionError("o placebo deveria vir do cache")

        monkeypatch.setattr(runner, "run_random_baseline_to_disk", boom)
        run_b = run_experiment(cfg, inputs, ["random"], 2.0, 2.0, validation_days=5, fresh=True)
        meta_b = json.loads((run_b / "run_meta.json").read_text())
        assert run_a != run_b and meta_b["random_cache"]["AAA_15m"] == "hit"
        _assert_tables_equal(run_a, run_b)

    def test_cache_off_recomputes(self, tmp_path):
        df, df_feat, cfg, windows, bt = _setup(n_seeds=2)
        cfg["walk_forward"]["runs_dir"] = str(tmp_path / "wf")
        cfg["walk_forward"]["random_cache"] = False
        inputs = {("AAA", "15m"): (df, df_feat)}
        run = run_experiment(cfg, inputs, ["random"], 2.0, 2.0, fresh=True)
        meta = json.loads((run / "run_meta.json").read_text())
        assert meta["random_cache"]["AAA_15m"] == "desligado"
        assert not (tmp_path / "cache").exists()
