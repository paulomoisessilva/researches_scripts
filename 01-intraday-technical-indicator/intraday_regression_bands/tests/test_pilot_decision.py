"""Testes da regra de decisão do piloto."""

import pandas as pd
import pytest

from src.pilot_decision import DecisionRule, cbe_by_candidate, evaluate_rule


def _cbe(rows):
    return pd.DataFrame(rows, columns=["variant", "strategy", "validation_days", "cbe_bps"])


class TestCbeByCandidate:
    def test_cbe_is_gross_over_units(self):
        w = pd.DataFrame({
            "variant": "v", "strategy": "s", "validation_days": 2,
            "test_date": ["2025-01-01", "2025-01-02"],
            "sum_gross_ret": [0.0010, 0.0020], "turnover_units": [2.0, 4.0],
        })
        out = cbe_by_candidate(w)
        assert out["cbe_bps"].iloc[0] == pytest.approx(0.0030 * 1e4 / 6.0)

    def test_negative_gross_gives_zero_and_no_units_gives_nan(self):
        w = pd.DataFrame({
            "variant": ["v", "v"], "strategy": ["a", "b"], "validation_days": [2, 2],
            "test_date": ["2025-01-01", "2025-01-01"],
            "sum_gross_ret": [-0.01, 0.01], "turnover_units": [3.0, 0.0],
        })
        out = cbe_by_candidate(w).set_index("strategy")
        assert out.loc["a", "cbe_bps"] == 0.0 and pd.isna(out.loc["b", "cbe_bps"])


class TestRule:
    rule = DecisionRule(min_cbe_bps=3.0, min_lookbacks=2, spa_cost_bps=2.0, max_spa_p=0.10)

    def test_scale_up_when_both_criteria_hold(self):
        cbe = _cbe([("v", "boll", 1, 3.5), ("v", "boll", 2, 4.0), ("v", "boll", 3, 1.0)])
        d = evaluate_rule(cbe, 0.05, self.rule)
        assert d["cbe_criterion_met"] and d["spa_criterion_met"] and d["scale_up"]

    def test_needs_two_lookbacks_for_the_same_method(self):
        # um lookback de cada método passa, mas nenhum método passa em 2 lookbacks
        cbe = _cbe([("v", "a", 1, 5.0), ("v", "a", 2, 1.0), ("v", "b", 1, 1.0), ("v", "b", 2, 6.0)])
        d = evaluate_rule(cbe, 0.01, self.rule)
        assert not d["cbe_criterion_met"] and not d["scale_up"]

    def test_variants_are_separate_methods(self):
        cbe = _cbe([("utc", "a", 1, 4.0), ("noeod", "a", 2, 4.0)])
        assert not evaluate_rule(cbe, 0.01, self.rule)["cbe_criterion_met"]

    def test_spa_gate_blocks_even_with_good_cbe(self):
        cbe = _cbe([("v", "a", 1, 4.0), ("v", "a", 2, 4.0)])
        d = evaluate_rule(cbe, 0.30, self.rule)
        assert d["cbe_criterion_met"] and not d["spa_criterion_met"] and not d["scale_up"]

    def test_boundaries_are_inclusive(self):
        cbe = _cbe([("v", "a", 1, 3.0), ("v", "a", 2, 3.0)])
        assert evaluate_rule(cbe, 0.10, self.rule)["scale_up"]
