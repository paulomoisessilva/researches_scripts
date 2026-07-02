"""Testes de integridade dos indicadores técnicos."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    bollinger_bands,
    compute_all_features,
    log_return,
    regression_bands,
    return_residual_bands,
    rsi,
    stochastic,
    technical_confluence_score,
)


class TestLogReturn:
    def test_first_value_is_nan(self, sample_ohlcv):
        ret = log_return(sample_ohlcv["close"])
        assert pd.isna(ret.iloc[0])

    def test_values_are_finite_after_first(self, sample_ohlcv):
        ret = log_return(sample_ohlcv["close"])
        assert ret.iloc[1:].notna().all()

    def test_cumulative_return(self, sample_ohlcv):
        close = sample_ohlcv["close"]
        ret = log_return(close).fillna(0)
        reconstructed = close.iloc[0] * np.exp(ret.cumsum())
        np.testing.assert_allclose(reconstructed.iloc[-1], close.iloc[-1], rtol=1e-6)


class TestRSI:
    def test_range_0_to_100(self, sample_ohlcv):
        r = rsi(sample_ohlcv["close"], window=14)
        valid = r.dropna()
        assert (valid >= 0).all(), f"RSI abaixo de 0: min={valid.min()}"
        assert (valid <= 100).all(), f"RSI acima de 100: max={valid.max()}"

    def test_first_window_values_are_nan(self, sample_ohlcv):
        r = rsi(sample_ohlcv["close"], window=14)
        assert r.iloc[:13].isna().all()

    def test_different_windows(self, sample_ohlcv):
        r14 = rsi(sample_ohlcv["close"], window=14)
        r21 = rsi(sample_ohlcv["close"], window=21)
        # Janelas diferentes devem produzir valores diferentes
        assert not np.allclose(r14.dropna().values[-100:], r21.dropna().values[-100:], rtol=1e-3)

    def test_constant_prices_rsi_undefined(self):
        close = pd.Series([100.0] * 50)
        r = rsi(close, window=14)
        # Com preços constantes, delta=0, RS seria 0/0; resultado pode ser NaN ou 50
        assert r.dropna().isin([0, 50, 100]).all() or r.dropna().isna().all()


class TestStochastic:
    def test_range_0_to_100(self, sample_ohlcv):
        k, d = stochastic(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            k_window=14,
        )
        valid_k = k.dropna()
        valid_d = d.dropna()
        assert (valid_k >= 0).all(), f"Stoch K abaixo de 0: min={valid_k.min()}"
        assert (valid_k <= 100).all(), f"Stoch K acima de 100: max={valid_k.max()}"
        assert (valid_d >= 0).all()
        assert (valid_d <= 100).all()

    def test_initial_nans(self, sample_ohlcv):
        k, d = stochastic(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            k_window=14,
            d_window=3,
        )
        # Primeiros 13 valores de %K devem ser NaN
        assert k.iloc[:13].isna().all()


class TestBollingerBands:
    def test_upper_greater_than_lower(self, sample_ohlcv):
        upper, mid, lower = bollinger_bands(sample_ohlcv["close"], window=20, k=2.0)
        valid = upper.notna() & lower.notna()
        assert (upper[valid] > lower[valid]).all()

    def test_mid_between_bands(self, sample_ohlcv):
        upper, mid, lower = bollinger_bands(sample_ohlcv["close"], window=20, k=2.0)
        valid = upper.notna() & mid.notna() & lower.notna()
        assert (upper[valid] >= mid[valid]).all()
        assert (mid[valid] >= lower[valid]).all()

    def test_initial_nans(self, sample_ohlcv):
        upper, _, _ = bollinger_bands(sample_ohlcv["close"], window=20, k=2.0)
        assert upper.iloc[:19].isna().all()

    def test_symmetric_around_midline(self, sample_ohlcv):
        upper, mid, lower = bollinger_bands(sample_ohlcv["close"], window=20, k=2.0)
        valid = upper.notna()
        diff_up = (upper[valid] - mid[valid]).values
        diff_lo = (mid[valid] - lower[valid]).values
        np.testing.assert_allclose(diff_up, diff_lo, rtol=1e-10)


class TestRegressionBands:
    def test_upper_greater_than_lower(self, sample_ohlcv):
        log_price = np.log(sample_ohlcv["close"])
        upper, fitted, lower = regression_bands(log_price, window=50, k=2.0)
        valid = upper.notna() & lower.notna()
        assert (upper[valid] > lower[valid]).all()

    def test_no_lookahead(self, sample_ohlcv):
        """Verifica que a banda em t não usa dados futuros."""
        log_price = np.log(sample_ohlcv["close"])
        upper1, fitted1, lower1 = regression_bands(log_price, window=50, k=2.0)

        # Truncar série em t e recalcular
        t = 200
        log_price_truncated = log_price.iloc[:t]
        upper2, fitted2, lower2 = regression_bands(log_price_truncated, window=50, k=2.0)

        # O valor no ponto t-1 deve ser idêntico nas duas séries
        assert abs(fitted1.iloc[t - 1] - fitted2.iloc[t - 1]) < 1e-10

    def test_initial_nans(self, sample_ohlcv):
        log_price = np.log(sample_ohlcv["close"])
        upper, _, _ = regression_bands(log_price, window=50, k=2.0)
        assert upper.iloc[:49].isna().all()


class TestReturnResidualBands:
    def test_upper_positive_lower_negative(self, sample_ohlcv):
        log_ret = log_return(sample_ohlcv["close"])
        upper, resid, lower = return_residual_bands(log_ret, window=50, k=2.0)
        valid = upper.notna()
        assert (upper[valid] > 0).all()
        assert (lower[valid] < 0).all()


class TestTechnicalConfluenceScore:
    def test_range_minus3_to_plus3(self, sample_ohlcv):
        score = technical_confluence_score(
            sample_ohlcv["close"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
        )
        valid = score.dropna()
        assert (valid >= -3).all()
        assert (valid <= 3).all()

    def test_integer_values(self, sample_ohlcv):
        score = technical_confluence_score(
            sample_ohlcv["close"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
        )
        valid = score.dropna()
        assert (valid == valid.round()).all()


class TestComputeAllFeatures:
    def test_output_has_expected_columns(self, sample_ohlcv, sample_config):
        df = compute_all_features(sample_ohlcv, sample_config)
        assert "log_return" in df.columns
        assert "rsi_14" in df.columns
        assert "stoch_k_14" in df.columns
        assert "bb_upper_20_20" in df.columns
        assert "technical_confluence_score" in df.columns

    def test_no_extra_rows(self, sample_ohlcv, sample_config):
        df = compute_all_features(sample_ohlcv, sample_config)
        assert len(df) == len(sample_ohlcv)

    def test_original_columns_preserved(self, sample_ohlcv, sample_config):
        df = compute_all_features(sample_ohlcv, sample_config)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns
