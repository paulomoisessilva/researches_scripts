"""Modo compacto de features (centro e desvio por janela): mesmo z, mesma escolha, menos colunas."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.indicators import compute_all_features
from src.strategies import compute_band_z
from src.walk_forward import run_walk_forward, validate_grid_features
from tests.test_carried_walk_forward import _setup

FEATURES = {
    "bollinger_windows": [20, 50], "bollinger_k": [1.5, 2.0, 3.5],
    "regression_windows": [30, 60], "regression_k": [1.5, 2.0, 7.0],
}


def _frame(n=3000, seed=2):
    rng = np.random.default_rng(seed)
    close = 30_000 * np.exp(np.cumsum(rng.normal(0, 8e-4, n)))
    ts = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": close * (1 + rng.normal(0, 2e-4, n)),
        "high": close * 1.001, "low": close * 0.999, "close": close, "volume": rng.uniform(1, 100, n),
    })


def _cfgs():
    full = {"features": dict(FEATURES)}
    compact = {"features": {**FEATURES, "compact_bands": True}}
    return full, compact


class TestSameZ:
    @pytest.mark.parametrize("strategy,windows", [("bollinger", FEATURES["bollinger_windows"]),
                                                  ("regression_bands", FEATURES["regression_windows"])])
    def test_compact_z_is_bit_identical_to_full_z(self, strategy, windows):
        df = _frame()
        full_cfg, compact_cfg = _cfgs()
        full = compute_all_features(df, full_cfg, strategy_names=[strategy])
        compact = compute_all_features(df, compact_cfg, strategy_names=[strategy])
        ks = FEATURES["bollinger_k"] if strategy == "bollinger" else FEATURES["regression_k"]
        for w in windows:
            for k in ks:
                pd.testing.assert_series_equal(
                    compute_band_z(compact, strategy, w, k), compute_band_z(full, strategy, w, k),
                    check_exact=True,
                )

    def test_compact_keeps_two_columns_per_window_and_no_band_columns(self):
        df = _frame()
        _, compact_cfg = _cfgs()
        full_cfg, _ = _cfgs()
        compact = compute_all_features(df, compact_cfg, strategy_names=["bollinger", "regression_bands"])
        full = compute_all_features(df, full_cfg, strategy_names=["bollinger", "regression_bands"])
        assert not any(c.startswith(("bb_upper_", "bb_lower_", "bb_pct_", "reg_upper_", "reg_lower_")) for c in compact)
        for w in FEATURES["bollinger_windows"]:
            assert {f"bb_mid_{w}", f"bb_std_{w}"} <= set(compact.columns)
        for w in FEATURES["regression_windows"]:
            assert {f"reg_fitted_{w}", f"reg_std_{w}"} <= set(compact.columns)
        base_only = compute_all_features(df, compact_cfg, strategy_names=["random"])
        n_windows = len(FEATURES["bollinger_windows"]) + len(FEATURES["regression_windows"])
        assert compact.shape[1] == base_only.shape[1] + 2 * n_windows
        assert compact.shape[1] < full.shape[1]

    def test_grid_validation_accepts_compact_and_still_catches_missing_windows(self):
        df = _frame()
        _, compact_cfg = _cfgs()
        compact = compute_all_features(df, compact_cfg, strategy_names=["bollinger"])
        validate_grid_features(compact, compact_cfg, ["bollinger"])
        wider = {"features": {**compact_cfg["features"], "bollinger_windows": [20, 50, 77]}}
        with pytest.raises(ValueError, match="ausentes"):
            validate_grid_features(compact, wider, ["bollinger"])


class TestSameWalkForward:
    @pytest.mark.parametrize("strategy", ["bollinger", "regression_bands"])
    @pytest.mark.parametrize("carry", [False, True])
    def test_results_do_not_change_with_compact_features(self, strategy, carry):
        df, df_feat_full, cfg = _setup()
        cfg = copy.deepcopy(cfg)
        cfg["walk_forward"]["carry_position_state"] = carry
        compact_cfg = copy.deepcopy(cfg)
        compact_cfg["features"]["compact_bands"] = True
        df_feat_compact = compute_all_features(df, compact_cfg, strategy_names=[strategy])
        kwargs = dict(df=df, strategy_name=strategy, symbol="X", interval="15m",
                      fee_bps=0.0, slippage_bps=0.0, n_workers=1)
        a = run_walk_forward(cfg=cfg, df_feat=df_feat_full, **kwargs)
        b = run_walk_forward(cfg=compact_cfg, df_feat=df_feat_compact, **kwargs)
        assert len(a) == len(b) > 10
        for x, y in zip(a, b):
            assert x.best_params == y.best_params
            pd.testing.assert_series_equal(x.test_returns, y.test_returns, check_exact=True)
            pd.testing.assert_series_equal(x.test_positions, y.test_positions, check_exact=True)
