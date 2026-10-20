"""Testes de run_id, cenários de custo, janelas com oos_start e anualização 365."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.metrics import annualization_factor, sharpe_ratio
from src.wf_runner import (
    _check_meta_compatible,
    config_fingerprint,
    make_run_id,
    parse_cost_scenarios,
    resolve_cost_scenarios,
    resolve_train_days,
)
from src.walk_forward import build_windows, validate_grid_features

ROOT = Path(__file__).parent.parent


def _paper_cfg() -> dict:
    return load_config(ROOT / "config.paper.yaml")


def _daily_df(n_days: int) -> pd.DataFrame:
    ts = pd.date_range("2021-01-01", periods=n_days * 96, freq="15min", tz="UTC")
    return pd.DataFrame({"timestamp": ts})


class TestAnnualization:
    def test_crypto_uses_365_days(self):
        assert annualization_factor("15m") == 96 * 365
        assert annualization_factor("1h") == 24 * 365

    def test_sharpe_scales_with_365(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(1e-5, 1e-3, 2000))
        expected = r.mean() * 96 * 365 / (r.std(ddof=1) * np.sqrt(96 * 365))
        assert sharpe_ratio(r, "15m") == pytest.approx(expected)


class TestCostScenarios:
    def test_parse(self):
        assert parse_cost_scenarios(["2:2", "0:5"]) == [(2.0, 2.0), (0.0, 5.0)]

    def test_parse_invalid(self):
        with pytest.raises(ValueError):
            parse_cost_scenarios(["2-2"])

    def test_primary_only_without_study(self):
        cfg = _paper_cfg()
        assert resolve_cost_scenarios(cfg) == [(2.0, 2.0)]

    def test_study_returns_all(self):
        cfg = _paper_cfg()
        got = resolve_cost_scenarios(cfg, study=True)
        assert got[0] == (2.0, 2.0) and len(got) == 4

    def test_cli_overrides(self):
        cfg = _paper_cfg()
        assert resolve_cost_scenarios(cfg, ["5:5"], study=True) == [(5.0, 5.0)]
        assert resolve_cost_scenarios(cfg, None, 1.0, 3.0) == [(1.0, 3.0)]

    def test_train_days(self):
        cfg = _paper_cfg()
        assert resolve_train_days(cfg) == [21]
        assert resolve_train_days(cfg, study=True) == [14, 21, 42]
        assert resolve_train_days(cfg, [30, 60], study=True) == [30, 60]


class TestRunId:
    def test_run_id_encodes_window_cost_and_period(self):
        cfg = _paper_cfg()
        rid = make_run_id(cfg, 2.0, 2.0, "2021-01-01", "2025-12-31")
        assert rid == "tr21_va7_te1_st1_fee2_slip2_20210101-20251231_oos20210301"

    def test_run_id_differs_by_train_days_and_cost(self):
        cfg = _paper_cfg()
        a = make_run_id(cfg, 2, 2, "2021-01-01", "2025-12-31")
        cfg2 = copy.deepcopy(cfg)
        cfg2["walk_forward"]["train_days"] = 42
        b = make_run_id(cfg2, 2, 2, "2021-01-01", "2025-12-31")
        c = make_run_id(cfg, 5, 5, "2021-01-01", "2025-12-31")
        assert len({a, b, c}) == 3


class TestFingerprint:
    def test_ignores_costs_and_workers(self):
        cfg = _paper_cfg()
        cfg2 = copy.deepcopy(cfg)
        cfg2["walk_forward"]["parallel_workers"] = 1
        cfg2["backtest"]["fee_bps_per_side"] = [9]
        assert config_fingerprint(cfg) == config_fingerprint(cfg2)

    def test_changes_with_grid_or_window(self):
        cfg = _paper_cfg()
        base = config_fingerprint(cfg)
        g = copy.deepcopy(cfg)
        g["features"]["bollinger_k"] = [2.0]
        w = copy.deepcopy(cfg)
        w["walk_forward"]["train_days"] = 42
        assert config_fingerprint(g) != base
        assert config_fingerprint(w) != base

    def test_meta_mismatch_raises(self, tmp_path):
        meta_path = tmp_path / "run_meta.json"
        meta_path.write_text(json.dumps({"fingerprint": "a", "fee_bps": 2.0, "slippage_bps": 2.0}))
        _check_meta_compatible(meta_path, {"fingerprint": "a", "fee_bps": 2.0, "slippage_bps": 2.0})
        with pytest.raises(ValueError, match="fingerprint"):
            _check_meta_compatible(meta_path, {"fingerprint": "b", "fee_bps": 2.0, "slippage_bps": 2.0})


class TestOosStart:
    def test_same_oos_period_across_train_days(self):
        df = _daily_df(120)
        a = build_windows(df, 14, 7, 1, 1, oos_start="2021-03-01")
        b = build_windows(df, 42, 7, 1, 1, oos_start="2021-03-01")
        assert len(a) == len(b)
        assert [w.test_start for w in a] == [w.test_start for w in b]
        assert a[0].test_start == pd.Timestamp("2021-03-01", tz="UTC")
        assert a[0].window_id == b[0].window_id == 0

    def test_train_precedes_validation_and_test(self):
        df = _daily_df(120)
        w = build_windows(df, 21, 7, 1, 1, oos_start="2021-03-01")[0]
        assert w.train_end == w.val_start and w.val_end == w.test_start
        assert w.train_start == pd.Timestamp("2021-03-01", tz="UTC") - pd.Timedelta(days=28)

    def test_raises_when_not_enough_history(self):
        df = _daily_df(120)
        with pytest.raises(ValueError, match="oos_start"):
            build_windows(df, 42, 7, 1, 1, oos_start="2021-01-20")


class TestValidateGridFeatures:
    def test_detects_missing_columns(self):
        cfg = {"features": {"bollinger_windows": [20], "bollinger_k": [2.0]}}
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0], "bb_mid_20": [1.0, 1.0, 1.0]})
        with pytest.raises(ValueError, match="bollinger"):
            validate_grid_features(df, cfg, ["bollinger"])

    def test_passes_when_present(self):
        cfg = {"features": {"bollinger_windows": [20], "bollinger_k": [2.0]}}
        df = pd.DataFrame({
            "close": [1.0, 2.0, 3.0],
            "bb_mid_20": [1.0, 1.0, 1.0],
            "bb_upper_20_20": [2.0, 2.0, 2.0],
        })
        validate_grid_features(df, cfg, ["bollinger"])

    def test_paper_grid_matches_cached_features(self):
        """O grid do config.paper.yaml deve existir no cache de features (se houver)."""
        feat = ROOT / "data" / "features" / "BTCUSDT_15m_features.parquet"
        if not feat.exists():
            pytest.skip("cache de features ausente")
        import pyarrow.parquet as pq

        cols = pq.ParquetFile(feat).schema.names
        cfg = _paper_cfg()
        df = pd.DataFrame({c: [1.0, 2.0] for c in cols})
        validate_grid_features(df, cfg, ["bollinger", "regression_bands", "return_residuals"])
