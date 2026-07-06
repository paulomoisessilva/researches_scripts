"""Testes da lógica de reversão confirmada via z."""

import numpy as np
import pandas as pd
import pytest

from src.strategies import (
    _reversal_confirmed_position,
    strategy_bollinger,
    strategy_regression_bands,
    strategy_return_residuals,
)
from src.indicators import compute_all_features


class TestReversalConfirmedPosition:
    def test_long_entry_requires_prior_extreme_and_cross_back(self):
        z = pd.Series([np.nan, -1.2, -1.1, -0.7, -0.5, 0.1])
        pos = _reversal_confirmed_position(z, entry_long_z=0.8, entry_short_z=0.8)

        assert pos.iloc[3] == 1.0
        assert pos.iloc[2] == 0.0

    def test_short_entry_requires_prior_extreme_and_cross_back(self):
        z = pd.Series([np.nan, 1.2, 1.1, 0.7, 0.5, -0.1])
        pos = _reversal_confirmed_position(z, entry_long_z=0.8, entry_short_z=0.8)

        assert pos.iloc[3] == -1.0
        assert pos.iloc[2] == 0.0

    def test_no_entry_on_first_cross_into_extreme(self):
        z = pd.Series([np.nan, 0.0, -0.5, -0.9, -1.1])
        pos = _reversal_confirmed_position(z, entry_long_z=0.8, entry_short_z=0.8)

        assert (pos == 0.0).all()

    def test_exit_at_center_default(self):
        z = pd.Series([np.nan, -1.2, -0.7, -0.3, 0.0, 0.2])
        pos = _reversal_confirmed_position(z, entry_long_z=0.8, entry_short_z=0.8)

        assert pos.iloc[3] == 1.0
        assert pos.iloc[4] == 0.0


class TestBandStrategiesReversal:
    @pytest.fixture
    def df_with_features(self, sample_ohlcv, sample_config):
        return compute_all_features(sample_ohlcv, sample_config)

    def test_bollinger_produces_valid_signal(self, df_with_features):
        signal = strategy_bollinger(
            df_with_features,
            window=20,
            k=2.0,
            entry_long_z=0.8,
            entry_short_z=0.8,
        )
        assert signal.index.equals(df_with_features.index)
        assert signal.isin([-1.0, 0.0, 1.0]).all()

    def test_regression_bands_produces_valid_signal(self, df_with_features):
        signal = strategy_regression_bands(
            df_with_features,
            window=50,
            k=2.0,
            entry_long_z=0.8,
            entry_short_z=0.8,
        )
        assert signal.isin([-1.0, 0.0, 1.0]).all()

    def test_return_residuals_produces_valid_signal(self, df_with_features):
        signal = strategy_return_residuals(
            df_with_features,
            window=50,
            k=2.0,
            entry_long_z=0.8,
            entry_short_z=0.8,
        )
        assert signal.isin([-1.0, 0.0, 1.0]).all()
