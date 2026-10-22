"""Regras de saída (stop, alvo, tempo, trailing, break-even): casos montados à mão e comparação com uma versão barra a barra."""

import numpy as np
import pandas as pd
import pytest

from src.v2_exits import (
    BIG,
    Bars,
    Rule,
    atr_bps,
    bar_indices,
    combine,
    excursions,
    level_return,
    rule_outcomes,
    trade_return,
)


def _bars(rows, atr=None):
    o, h, l, c = (np.array([r[i] for r in rows], dtype=float) for i in range(4))
    return Bars(o, h, l, c, None if atr is None else np.asarray(atr, dtype=float))


def _one(bars, rule, e=0, x=None, sign=1.0):
    x = len(bars.open) - 1 if x is None else x
    out = rule_outcomes(bars, np.array([e]), np.array([x]), np.array([sign]), [rule])[rule]
    return int(out[0][0]), float(out[1][0])


# barras (open, high, low, close) de uma operação comprada em 100 que cai até 98,5 e volta
PATH = _bars([
    (100.0, 100.52, 99.8, 100.2),   # 0: contra 20, a favor 52
    (100.2, 100.6, 99.5, 99.7),     # 1: contra 50, a favor 60
    (99.7, 100.0, 98.8, 99.0),      # 2: contra 120
    (99.0, 99.5, 98.5, 99.2),       # 3: contra 150
    (99.2, 100.4, 99.0, 100.3),     # 4: contra 100
    (100.3, 100.5, 100.0, 100.4),   # 5: saída na abertura (preço 100,3)
])


class TestExcursions:
    def test_mae_and_mfe_of_a_long_trade(self):
        mae, mfe = excursions(PATH, np.array([0]), np.array([5]), np.array([1.0]))
        assert mae[0] == pytest.approx(150.0) and mfe[0] == pytest.approx(60.0)

    def test_short_trade_swaps_the_sides(self):
        mae, mfe = excursions(PATH, np.array([0]), np.array([5]), np.array([-1.0]))
        assert mae[0] == pytest.approx(60.0) and mfe[0] == pytest.approx(150.0)

    def test_trade_return_matches_open_to_open(self):
        r = trade_return(PATH, np.array([0]), np.array([5]), np.array([1.0]))
        assert r[0] == pytest.approx(np.log(100.3 / 100.0) * 1e4)


class TestStopLossAndTarget:
    def test_stop_loss_exits_at_the_level_on_the_first_bar_that_touches_it(self):
        first, ret = _one(PATH, Rule("SL", 100), x=5)
        assert first == 2 and ret == pytest.approx(np.log(0.99) * 1e4)
        first, ret = _one(PATH, Rule("SL", 30), x=5)
        assert first == 1 and ret == pytest.approx(np.log(0.997) * 1e4)

    def test_stop_never_touched_leaves_the_trade_alone(self):
        first, ret = _one(PATH, Rule("SL", 500), x=5)
        assert first == BIG and ret == 0.0

    def test_gap_through_the_stop_exits_at_the_open(self):
        bars = _bars([(100, 100.3, 99.9, 100.1), (100.1, 100.2, 99.8, 99.9), (98.0, 98.4, 97.8, 98.1), (98.1, 98.5, 98.0, 98.3)])
        first, ret = _one(bars, Rule("SL", 100), x=3)
        assert first == 2 and ret == pytest.approx(np.log(98.0 / 100.0) * 1e4)   # pior que o nível de -100 bps

    def test_short_trade_stop_is_above_the_entry(self):
        bars = _bars([(100, 100.2, 99.9, 100.1), (100.1, 101.2, 100.0, 101.0), (101.0, 101.1, 100.9, 101.0)])
        first, ret = _one(bars, Rule("SL", 100), x=2, sign=-1.0)
        assert first == 1 and ret == pytest.approx(-np.log(1.01) * 1e4)

    def test_target_exits_at_the_level_even_if_the_bar_opens_beyond_it(self):
        first, ret = _one(PATH, Rule("TP", 50), x=5)
        assert first == 0 and ret == pytest.approx(np.log(1.005) * 1e4)

    def test_stop_wins_when_both_levels_are_touched_in_the_same_bar(self):
        bars = _bars([(100, 100.6, 99.0, 100.0), (100.0, 100.1, 99.9, 100.0)])
        o = rule_outcomes(bars, np.array([0]), np.array([1]), np.array([1.0]), [Rule("SL", 50), Rule("TP", 50)])
        ret, hit = combine(o, [Rule("SL", 50), Rule("TP", 50)], np.array([7.0]))
        assert hit[0] and ret[0] == pytest.approx(np.log(0.995) * 1e4)

    def test_target_wins_when_it_is_touched_before_the_stop(self):
        rules = [Rule("SL", 100), Rule("TP", 50)]
        o = rule_outcomes(PATH, np.array([0]), np.array([5]), np.array([1.0]), rules)
        ret, hit = combine(o, rules, np.array([30.0]))
        assert hit[0] and ret[0] == pytest.approx(np.log(1.005) * 1e4)

    def test_no_rule_triggered_keeps_the_original_return(self):
        rules = [Rule("SL", 900), Rule("TP", 900)]
        o = rule_outcomes(PATH, np.array([0]), np.array([5]), np.array([1.0]), rules)
        ret, hit = combine(o, rules, np.array([30.0]))
        assert not hit[0] and ret[0] == 30.0
        ret, hit = combine(o, [], np.array([30.0]))
        assert not hit[0] and ret[0] == 30.0


class TestTimeStop:
    def test_exit_at_the_open_of_the_nth_bar(self):
        first, ret = _one(PATH, Rule("TIME", 2), x=5)
        assert first == 2 and ret == pytest.approx(np.log(99.7 / 100.0) * 1e4)

    def test_trades_shorter_than_the_limit_are_not_touched(self):
        first, ret = _one(PATH, Rule("TIME", 5), x=5)
        assert first == BIG and ret == 0.0

    def test_time_stop_beats_a_stop_touched_in_the_same_bar_and_loses_to_an_earlier_one(self):
        rules = [Rule("TIME", 2), Rule("SL", 100)]       # o stop de 100 bps só é tocado na barra 2
        o = rule_outcomes(PATH, np.array([0]), np.array([5]), np.array([1.0]), rules)
        ret, _ = combine(o, rules, np.array([0.0]))
        assert ret[0] == pytest.approx(np.log(99.7 / 100.0) * 1e4)
        rules = [Rule("TIME", 2), Rule("SL", 30)]        # o stop de 30 bps é tocado na barra 1
        o = rule_outcomes(PATH, np.array([0]), np.array([5]), np.array([1.0]), rules)
        ret, _ = combine(o, rules, np.array([0.0]))
        assert ret[0] == pytest.approx(np.log(0.997) * 1e4)


class TestTrailingAndBreakEven:
    TRAIL = _bars([
        (100.0, 101.0, 99.9, 100.8),     # 0: pico a favor de 100
        (100.8, 101.5, 100.8, 101.4),    # 1: novo pico de 150
        (101.4, 101.5, 100.8, 100.9),    # 2: cai do pico de 150 para 80 a favor
        (100.9, 101.0, 100.7, 100.8),
    ])

    def test_trailing_stop_follows_the_peak(self):
        first, ret = _one(self.TRAIL, Rule("TR", 50), x=3)
        assert first == 2 and ret == pytest.approx(np.log(1.01) * 1e4)    # pico 150 - 50 = nível de 100 bps a favor

    def test_trailing_stop_acts_as_a_plain_stop_before_any_gain(self):
        bars = _bars([(100.0, 100.0, 99.9, 99.95), (99.95, 99.96, 99.5, 99.6), (99.6, 99.7, 99.4, 99.5)])
        first, ret = _one(bars, Rule("TR", 30), x=2)
        assert first == 1 and ret == pytest.approx(np.log(0.997) * 1e4)

    def test_trailing_stop_locks_in_part_of_an_early_gain(self):
        first, ret = _one(PATH, Rule("TR", 30), x=5)       # pico de 52 na barra 0: o stop sobe a +22, mas a barra 1 abre em +20: sai na abertura
        assert first == 1 and ret == pytest.approx(np.log(1.002) * 1e4)

    def test_break_even_exits_at_the_entry_after_the_activation(self):
        bars = _bars([(100.0, 100.8, 99.95, 100.4), (100.5, 100.6, 99.9, 100.2), (100.2, 100.3, 100.0, 100.1)])
        first, ret = _one(bars, Rule("BE", 50), x=2)
        assert first == 1 and ret == pytest.approx(0.0)

    def test_break_even_with_a_gap_below_the_entry_exits_at_the_open(self):
        bars = _bars([(100.0, 100.8, 99.95, 100.4), (99.8, 100.0, 99.7, 99.9), (99.9, 100.0, 99.8, 99.9)])
        first, ret = _one(bars, Rule("BE", 50), x=2)
        assert first == 1 and ret == pytest.approx(np.log(99.8 / 100.0) * 1e4)

    def test_break_even_is_not_armed_before_the_activation_level(self):
        first, _ = _one(PATH, Rule("BE", 500), x=5)
        assert first == BIG


class TestAtrRules:
    def test_atr_stop_and_target_scale_with_the_atr_of_the_entry_bar(self):
        atr = np.full(6, 20.0)
        bars = _bars([(100.0, 100.5, 99.8, 100.2), (100.2, 100.7, 99.5, 99.7), (99.7, 100.0, 98.8, 99.0),
                      (99.0, 99.5, 98.5, 99.2), (99.2, 100.4, 99.0, 100.3), (100.3, 100.5, 100.0, 100.4)], atr=atr)
        first, ret = _one(bars, Rule("ASL", 2), x=5)       # 2 x 20 = 40 bps: tocado na barra 1 (contra 50)
        assert first == 1 and ret == pytest.approx(np.log(1 - 0.0040) * 1e4)
        first, ret = _one(bars, Rule("ATP", 3), x=5)       # 3 x 20 = 60 bps: tocado na barra 1 (a favor 70)
        assert first == 1 and ret == pytest.approx(np.log(1.006) * 1e4)

    def test_atr_rules_require_the_atr(self):
        for kind in ("ASL", "ATP", "DSL", "DTR"):
            with pytest.raises(ValueError, match="ATR"):
                _one(PATH, Rule(kind, 2), x=5)

    def test_dynamic_atr_stop_follows_the_atr_of_each_bar(self):
        atr = np.array([60.0, 20.0, 20.0, 20.0, 20.0, 20.0])
        bars = _bars([(100.0, 100.5, 99.8, 100.2), (100.2, 100.7, 99.5, 99.7), (99.7, 100.0, 98.8, 99.0),
                      (99.0, 99.5, 98.5, 99.2), (99.2, 100.4, 99.0, 100.3), (100.3, 100.5, 100.0, 100.4)], atr=atr)
        first, ret = _one(bars, Rule("DSL", 2), x=5)       # barra 0: 2 x 60 = 120 não é tocado; barra 1: 2 x 20 = 40 e contra 50
        assert first == 1 and ret == pytest.approx(np.log(1 - 0.0040) * 1e4)
        first, ret = _one(bars, Rule("ASL", 1.5), x=5)     # o ATR da entrada (60) dá 90 bps: só a barra 2 (contra 120)
        assert first == 2 and ret == pytest.approx(np.log(1 - 0.0090) * 1e4)

    def test_dynamic_atr_trailing_uses_the_current_atr_and_the_prior_peak(self):
        atr = np.array([20.0, 20.0, 5.0, 5.0, 5.0, 5.0])
        bars = _bars([(100.0, 100.5, 99.9, 100.4), (100.4, 100.8, 100.3, 100.7), (100.7, 100.8, 100.2, 100.3),
                      (100.3, 100.4, 100.0, 100.1), (100.1, 100.2, 99.9, 100.0), (100.0, 100.1, 99.9, 100.0)], atr=atr)
        # pico a favor de 80 bps até a barra 1 (distância 4 x 20 = 80 nas barras 0 e 1: nada). Na barra 2 o ATR cai para 5: distância 4 x 5 = 20,
        # nível = 80 - 20 = 60 bps; a mínima 100,2 está 20 bps acima da entrada, abaixo do nível: sai em +60 (a abertura, +70, está acima)
        first, ret = _one(bars, Rule("DTR", 4), x=5)
        assert first == 2 and ret == pytest.approx(np.log(1 + 0.0060) * 1e4)

    def test_constant_atr_reduces_dynamic_rules_to_the_fixed_ones(self):
        bars, e, x, sign = TestInputsAndVectorization._random_case(11)
        const = Bars(bars.open, bars.high, bars.low, bars.close, np.full(len(bars.open), 10.0))
        out = rule_outcomes(const, e, x, sign, [Rule("DSL", 3), Rule("SL", 30), Rule("DTR", 2), Rule("TR", 20), Rule("ASL", 3)])
        for dyn, fixed in ((Rule("DSL", 3), Rule("SL", 30)), (Rule("DTR", 2), Rule("TR", 20)), (Rule("ASL", 3), Rule("SL", 30))):
            assert np.array_equal(out[dyn][0], out[fixed][0]) and np.allclose(out[dyn][1], out[fixed][1])

    def test_atr_is_causal_and_in_bps(self):
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 50)))
        high, low = close * 1.001, close * 0.999
        a = atr_bps(high, low, close, 5)
        assert np.isnan(a[:6]).all() and np.isfinite(a[6:]).all()
        high2 = high.copy(); high2[20] *= 1.1                # uma barra muda: o ATR só reage na barra seguinte
        a2 = atr_bps(high2, low, close, 5)
        assert np.allclose(a[:21], a2[:21], equal_nan=True) and a2[21] != a[21]


class TestInputsAndVectorization:
    def test_unknown_rule_is_rejected(self):
        with pytest.raises(ValueError, match="desconhecida"):
            Rule("XX", 1)

    def test_bar_indices_requires_exact_times_and_exits_after_entries(self):
        ts = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
        e, x = bar_indices(ts, ts[[1, 3]], ts[[2, 8]])
        assert e.tolist() == [1, 3] and x.tolist() == [2, 8]
        with pytest.raises(ValueError, match="não existe"):
            bar_indices(ts, [pd.Timestamp("2025-01-01", tz="UTC")], ts[[2]])
        with pytest.raises(ValueError, match="saída"):
            bar_indices(ts, ts[[3]], ts[[3]])

    @staticmethod
    def _random_case(seed, n_bars=400, n_trades=80):
        rng = np.random.default_rng(seed)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 8e-4, n_bars)))
        op = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 2e-4, n_bars))
        high = np.maximum(op, close) * (1 + rng.uniform(0, 8e-4, n_bars))
        low = np.minimum(op, close) * (1 - rng.uniform(0, 8e-4, n_bars))
        bars = Bars(op, high, low, close, atr_bps(high, low, close, 10))
        e = np.sort(rng.integers(20, n_bars - 40, n_trades))
        x = e + rng.integers(1, 30, n_trades)
        sign = rng.choice([-1.0, 1.0], n_trades)
        return bars, e, x, sign

    @staticmethod
    def _reference(bars, e, x, sign, rule):
        """Versão barra a barra da mesma regra."""
        p0 = bars.open[e]
        peak = 0.0
        for off in range(x - e):
            j = e + off
            adv = sign * 0 + (p0 - bars.low[j]) / p0 * 1e4 if sign > 0 else (bars.high[j] - p0) / p0 * 1e4
            fav = (bars.high[j] - p0) / p0 * 1e4 if sign > 0 else (p0 - bars.low[j]) / p0 * 1e4
            fav_open = sign * (bars.open[j] - p0) / p0 * 1e4
            level, gap, hit = None, True, False
            if rule.kind == "SL":
                hit, level = adv >= rule.param, -rule.param
            elif rule.kind == "TP":
                hit, level, gap = fav >= rule.param, rule.param, False
            elif rule.kind == "TR":
                hit, level = peak + adv >= rule.param, peak - rule.param
            elif rule.kind == "BE":
                hit, level = (peak >= rule.param) and (adv >= 0), 0.0
            elif rule.kind == "ASL":
                d = rule.param * bars.atr[e]; hit, level = adv >= d, -d
            elif rule.kind == "ATP":
                d = rule.param * bars.atr[e]; hit, level, gap = fav >= d, d, False
            elif rule.kind == "DSL":
                d = rule.param * bars.atr[j]; hit, level = adv >= d, -d
            elif rule.kind == "DTR":
                d = rule.param * bars.atr[j]; hit, level = peak + adv >= d, peak - d
            if hit:
                lv = min(level, fav_open) if gap else level
                return off, float(level_return(lv, sign))
            peak = max(peak, fav)
        return BIG, 0.0

    @pytest.mark.parametrize("rule", [Rule("SL", 20), Rule("SL", 80), Rule("TP", 15), Rule("TR", 10), Rule("TR", 40), Rule("BE", 10),
                                      Rule("BE", 40), Rule("ASL", 2), Rule("ATP", 1), Rule("DSL", 2), Rule("DSL", 5), Rule("DTR", 1), Rule("DTR", 3)])
    def test_matches_a_bar_by_bar_reference_on_random_paths(self, rule):
        bars, e, x, sign = self._random_case(3)
        first, ret = rule_outcomes(bars, e, x, sign, [rule])[rule]
        for i in range(len(e)):
            ref_first, ref_ret = self._reference(bars, e[i], x[i], sign[i], rule)
            assert first[i] == ref_first, (rule, i)
            assert ret[i] == pytest.approx(ref_ret, abs=1e-9), (rule, i)

    def test_chunked_evaluation_gives_the_same_result(self):
        bars, e, x, sign = self._random_case(5)
        rules = [Rule("SL", 30), Rule("TP", 20), Rule("TR", 15), Rule("BE", 20), Rule("TIME", 3)]
        whole = rule_outcomes(bars, e, x, sign, rules, max_rows=10 ** 9)
        small = rule_outcomes(bars, e, x, sign, rules, max_rows=25)
        for r in rules:
            assert np.array_equal(whole[r][0], small[r][0]) and np.allclose(whole[r][1], small[r][1])
        a = excursions(bars, e, x, sign, max_rows=10 ** 9); b = excursions(bars, e, x, sign, max_rows=25)
        assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])

    def test_a_looser_stop_never_triggers_earlier(self):
        bars, e, x, sign = self._random_case(9)
        o = rule_outcomes(bars, e, x, sign, [Rule("SL", 20), Rule("SL", 60)])
        assert (o[Rule("SL", 60)][0] >= o[Rule("SL", 20)][0]).all()
