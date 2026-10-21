"""A regressão rolling vetorizada deve reproduzir a versão original (lstsq por janela)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    _compute_regression_features,
    regression_bands,
    regression_bands_multi_k,
    rolling_ols_last,
)

ROOT = Path(__file__).parent.parent


def _reference_regression_bands(log_price: pd.Series, window: int, k: float):
    """Cópia da implementação ORIGINAL (um lstsq por janela)."""
    n = len(log_price)
    fitted = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    values = log_price.values
    for i in range(window - 1, n):
        segment = values[i - window + 1: i + 1]
        t = np.arange(window, dtype=float)
        A = np.column_stack([np.ones(window), t])
        try:
            coef, *_ = np.linalg.lstsq(A, segment, rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = A @ coef
        residuals = segment - pred
        std_res = residuals.std(ddof=2) if window > 2 else 0.0
        fitted[i] = pred[-1]
        upper[i] = fitted[i] + k * std_res
        lower[i] = fitted[i] - k * std_res
    idx = log_price.index
    return pd.Series(upper, idx), pd.Series(fitted, idx), pd.Series(lower, idx)


def _series(n=1500, seed=0, with_nan=False):
    rng = np.random.default_rng(seed)
    lp = pd.Series(np.log(30_000) + np.cumsum(rng.normal(0, 6e-4, n)),
                   index=pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC"))
    if with_nan:
        lp.iloc[400] = np.nan
    return lp


class TestRollingOls:
    @pytest.mark.parametrize("window", [3, 25, 50, 75, 100])
    @pytest.mark.parametrize("k", [1.5, 3.5, 5.0])
    def test_matches_original(self, window, k):
        lp = _series()
        u_ref, f_ref, l_ref = _reference_regression_bands(lp, window, k)
        u, f, l = regression_bands(lp, window=window, k=k)
        for got, ref in ((u, u_ref), (f, f_ref), (l, l_ref)):
            np.testing.assert_allclose(got.to_numpy(), ref.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)

    def test_nan_in_window_gives_nan_like_original(self):
        lp = _series(with_nan=True)
        u_ref, f_ref, _ = _reference_regression_bands(lp, 50, 2.0)
        u, f, _ = regression_bands(lp, window=50, k=2.0)
        np.testing.assert_array_equal(np.isnan(f.to_numpy()), np.isnan(f_ref.to_numpy()))
        np.testing.assert_allclose(f.to_numpy(), f_ref.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)

    def test_window_two_has_zero_residual_std(self):
        fitted, std = rolling_ols_last(np.array([1.0, 2.0, 4.0, 3.0]), 2)
        assert np.isnan(fitted[0]) and np.allclose(std[1:], 0.0)
        assert fitted[1] == pytest.approx(2.0) and fitted[2] == pytest.approx(4.0)

    def test_shorter_than_window_is_all_nan(self):
        fitted, std = rolling_ols_last(np.arange(10.0), 50)
        assert np.isnan(fitted).all() and np.isnan(std).all()

    def test_chunking_does_not_change_result(self):
        values = _series(3000, seed=3).to_numpy()
        a = rolling_ols_last(values, 50, chunk=7)
        b = rolling_ols_last(values, 50, chunk=100_000)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_multi_k_reuses_one_fit(self):
        lp = _series()
        fitted, bands = regression_bands_multi_k(lp, 50, [1.5, 2.0, 5.0])
        for k, (u, l) in bands.items():
            u1, f1, l1 = regression_bands(lp, window=50, k=k)
            pd.testing.assert_series_equal(u, u1)
            pd.testing.assert_series_equal(l, l1)
            pd.testing.assert_series_equal(fitted, f1)

    def test_feature_columns_unchanged(self):
        lp = _series(400)
        feats = _compute_regression_features(lp, {"regression_windows": [25, 50], "regression_k": [1.5, 2.0]})
        assert set(feats) == {
            "reg_fitted_25", "reg_upper_25_15", "reg_lower_25_15", "reg_upper_25_20", "reg_lower_25_20",
            "reg_fitted_50", "reg_upper_50_15", "reg_lower_50_15", "reg_upper_50_20", "reg_lower_50_20",
        }


def test_matches_cached_features_from_original_code():
    """As colunas reg_* do cache de 5 anos (geradas pelo código antigo) batem com as novas."""
    cache = ROOT / "data" / "features" / "BTCUSDT_15m_features.parquet"
    if not cache.exists():
        pytest.skip("cache de features ausente")
    cols = ["log_price", "reg_fitted_50", "reg_upper_50_20", "reg_lower_50_20"]
    df = pd.read_parquet(cache, columns=cols)
    fitted, bands = regression_bands_multi_k(df["log_price"], 50, [2.0])
    np.testing.assert_allclose(fitted.to_numpy(), df["reg_fitted_50"].to_numpy(), rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(bands[2.0][0].to_numpy(), df["reg_upper_50_20"].to_numpy(), rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(bands[2.0][1].to_numpy(), df["reg_lower_50_20"].to_numpy(), rtol=0, atol=1e-9, equal_nan=True)
