"""Testes dev-only para baseline matched (fora da metodologia principal)."""

import pytest

from src.indicators import compute_features_for_strategy
from src.random_baseline_dev import expand_random_strategies_dev
from src.strategies import strategy_bollinger, strategy_random_matched


@pytest.mark.skip(reason="Dev-only: matched-random não faz parte da metodologia principal")
class TestRandomMatchedTurnover:
    def test_same_position_change_count(self, sample_ohlcv, sample_config):
        df_feat = compute_features_for_strategy("bollinger", sample_ohlcv, sample_config)
        target = strategy_bollinger(
            df_feat,
            window=20,
            k=2.0,
            entry_long_z=0.8,
            entry_short_z=0.8,
            exit_long_z=0.0,
            exit_short_z=0.0,
        )
        matched = strategy_random_matched(df_feat, target, seed=7)

        target_changes = (target.diff().abs() > 0).sum()
        matched_changes = (matched.diff().abs() > 0).sum()
        assert matched_changes == target_changes

    def test_matched_differs_from_target_direction(self, sample_ohlcv, sample_config):
        df_feat = compute_features_for_strategy("bollinger", sample_ohlcv, sample_config)
        target = strategy_bollinger(
            df_feat, window=20, k=2.0,
            entry_long_z=0.8, entry_short_z=0.8,
            exit_long_z=0.0, exit_short_z=0.0,
        )
        matched = strategy_random_matched(df_feat, target, seed=42)
        assert not matched.equals(target)


@pytest.mark.skip(reason="Dev-only: matched-random não faz parte da metodologia principal")
class TestExpandRandomStrategiesDev:
    def test_expands_only_when_enabled(self, sample_config):
        sample_config["random_baseline"] = {
            "enable_matched_random": True,
            "matched_strategies": ["bollinger", "regression_bands"],
        }
        expanded = expand_random_strategies_dev(["bollinger", "random"], sample_config)
        assert "random_matched_bollinger" in expanded

    def test_no_expansion_by_default(self, sample_config):
        expanded = expand_random_strategies_dev(["bollinger", "random"], sample_config)
        assert expanded == ["bollinger", "random"]
