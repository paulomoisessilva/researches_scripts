"""Equivalência do núcleo vetorizado (sinal por eventos e backtest em arrays) com os laços de referência."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest, run_backtest_loop
from src.fast_core import backtest_arrays, positions_from_pairs, reversal_position_array
from src.strategies import (
    _reversal_confirmed_position,
    _reversal_confirmed_position_loop,
    compute_band_z,
    signal_from_z,
)


def _random_z(rng: np.random.Generator, n: int, kind: int) -> np.ndarray:
    if kind == 0:  # passeio aleatório suave: poucos cruzamentos
        z = np.cumsum(rng.normal(0, 0.4, n))
    elif kind == 1:  # saltos grandes: saída de uma posição e entrada na outra na mesma barra
        z = rng.normal(0, 2.0, n)
    else:  # valores em múltiplos de 0,5: empates exatos nos limiares
        z = np.round(rng.normal(0, 1.5, n) * 2) / 2
    return z


class TestReversalSignalEquivalence:
    @staticmethod
    def _check(z, **kw):
        series = pd.Series(z)
        ref = _reversal_confirmed_position_loop(series, **kw)
        got = _reversal_confirmed_position(series, **kw)
        pd.testing.assert_series_equal(got, ref, check_exact=True)

    @pytest.mark.parametrize("kind", [0, 1, 2])
    @pytest.mark.parametrize("with_nan", [False, True])
    def test_random_series_and_thresholds(self, kind, with_nan):
        rng = np.random.default_rng(100 + kind + 10 * with_nan)
        for _ in range(120):
            n = int(rng.integers(2, 500))
            z = _random_z(rng, n, kind)
            if with_nan:
                z[rng.random(n) < 0.1] = np.nan
            self._check(
                z,
                entry_long_z=float(rng.choice([0.5, 0.8, 1.0, 1.5])),
                entry_short_z=float(rng.choice([0.5, 0.8, 1.0, 1.5])),
                exit_long_z=float(rng.choice([0.0, 0.0, -0.5, 0.3])),
                exit_short_z=float(rng.choice([0.0, 0.0, 0.5, -0.3])),
            )

    def test_realistic_bollinger_z(self):
        rng = np.random.default_rng(3)
        n = 20_000
        price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n))))
        mid, sd = price.rolling(100).mean(), price.rolling(100).std()
        for k in (1.0, 2.0, 3.0):
            z = ((price - mid) / ((mid + k * sd) - mid)).to_numpy()
            self._check(z, entry_long_z=1.0, entry_short_z=1.0)

    @pytest.mark.parametrize("z", [[], [0.3], [0.3, -2.0], [np.nan] * 5, [0.0] * 8, [-3.0, -3.0, 0.5, 0.2, 0.0]])
    def test_degenerate_inputs(self, z):
        self._check(np.asarray(z, dtype=float), entry_long_z=1.0, entry_short_z=1.0)

    def test_signal_from_z_matches_loop(self):
        rng = np.random.default_rng(8)
        z = pd.Series(_random_z(rng, 600, 1), index=pd.date_range("2024-01-01", periods=600, freq="15min"))
        got = signal_from_z(z, entry_long_z=1.0, entry_short_z=1.0, name="s")
        ref_pos = _reversal_confirmed_position_loop(z, entry_long_z=1.0, entry_short_z=1.0)
        ref = ref_pos.shift(1).fillna(0).rename("s")
        pd.testing.assert_series_equal(got, ref, check_exact=True)

    def test_long_then_short_in_the_same_bar(self):
        # entra long (z cruza -1 para cima com z=3), na barra seguinte sai (z=0.5 >= 0) e entra short
        # (z_prev=3 > 1 e z=0.5 <= 1) na mesma barra.
        z = np.array([-2.0, 3.0, 0.5, 0.5])
        got = reversal_position_array(z, 1.0, 1.0)
        assert got.tolist() == [0.0, 1.0, -1.0, -1.0]
        self._check(z, entry_long_z=1.0, entry_short_z=1.0)

    def test_exit_and_reentry_on_the_same_bar(self):
        # long aberto; em t=2 o z sobe a 0.5 (sai) vindo de -2 (re-entra na mesma barra)
        z = np.array([-2.0, -0.5, -2.0, 0.5, 0.5])
        self._check(z, entry_long_z=1.0, entry_short_z=1.0)


def _reference_pair_machine(zp, zc, el, es, xl, xs):
    """Máquina de estados barra a barra sobre pares (z anterior, z atual) com limiares por barra."""
    n = len(zc)
    el, es, xl, xs = (np.broadcast_to(np.asarray(v, dtype=float), (n,)) for v in (el, es, xl, xs))
    pos = np.zeros(n)
    cur = 0.0
    for i in range(1, n):
        a, b = zp[i], zc[i]
        if np.isnan(a) or np.isnan(b):
            pos[i] = cur
            continue
        if cur == 1.0:
            if b >= xl[i]:
                cur = 0.0
        elif cur == -1.0:
            if b <= xs[i]:
                cur = 0.0
        if cur == 0.0:
            if a < -el[i] and b >= -el[i]:
                cur = 1.0
            elif a > es[i] and b <= es[i]:
                cur = -1.0
        pos[i] = cur
    return pos


class TestPairsWithPerBarThresholds:
    def test_matches_reference_with_independent_pairs_and_piecewise_thresholds(self):
        rng = np.random.default_rng(21)
        for _ in range(150):
            n = int(rng.integers(2, 400))
            zc = _random_z(rng, n, int(rng.integers(0, 3)))
            zp = _random_z(rng, n, int(rng.integers(0, 3)))  # z anterior independente (troca de regra)
            zp[rng.random(n) < 0.05] = np.nan
            zc[rng.random(n) < 0.05] = np.nan
            cuts = np.sort(rng.integers(0, n, 3))
            def piecewise(choices):
                out = np.empty(n)
                for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, n]):
                    out[lo:hi] = rng.choice(choices)
                return out
            el, es = piecewise([0.5, 1.0, 1.5]), piecewise([0.5, 1.0, 1.5])
            xl, xs = piecewise([0.0, -0.3, 0.3]), piecewise([0.0, 0.3, -0.3])
            got = positions_from_pairs(zp, zc, el, es, xl, xs)
            ref = _reference_pair_machine(zp, zc, el, es, xl, xs)
            assert np.array_equal(got, ref)

    def test_scalar_and_vector_thresholds_agree(self):
        rng = np.random.default_rng(5)
        z = _random_z(rng, 300, 1)
        zp = np.r_[np.nan, z[:-1]]
        a = positions_from_pairs(zp, z, 1.0, 0.8, 0.0, 0.0)
        b = positions_from_pairs(zp, z, np.full(300, 1.0), np.full(300, 0.8), np.zeros(300), np.zeros(300))
        assert np.array_equal(a, b)

    def test_rejects_mismatched_sizes(self):
        with pytest.raises(ValueError, match="tamanhos"):
            positions_from_pairs(np.zeros(3), np.zeros(4))
        with pytest.raises(ValueError, match="por barra"):
            positions_from_pairs(np.zeros(4), np.zeros(4), entry_long_z=np.ones(3))


def _random_frame_and_signal(rng, n, freq="1h", nan_opens=False, kind="int"):
    ts = pd.date_range("2024-03-01", periods=n, freq=freq, tz="UTC")
    opens = 100 * np.exp(np.cumsum(rng.normal(0, 3e-3, n)))
    if nan_opens:
        opens[rng.random(n) < 0.03] = np.nan
        opens[rng.random(n) < 0.01] = 0.0
    df = pd.DataFrame({"timestamp": ts, "open": opens, "close": opens})
    runs = rng.integers(1, 12, n)
    if kind == "int":
        values = rng.choice([-1.0, 0.0, 0.0, 1.0], n)
    else:  # frações e valores fora de [-max_position, max_position]
        values = rng.choice([-2.0, -0.5, 0.0, 0.5, 1.0, 2.0], n)
    signal = pd.Series(np.repeat(values, runs)[:n])
    return df, signal


def _assert_same_result(got, ref):
    pd.testing.assert_series_equal(got.positions, ref.positions, check_exact=True)
    pd.testing.assert_series_equal(got.gross_returns, ref.gross_returns, check_exact=True)
    pd.testing.assert_series_equal(got.returns, ref.returns, check_exact=True)
    pd.testing.assert_series_equal(got.trade_units, ref.trade_units, check_exact=True)
    pd.testing.assert_series_equal(got.equity, ref.equity, check_exact=True)
    pd.testing.assert_frame_equal(got.trades, ref.trades, check_exact=True)


class TestBacktestEquivalence:
    @pytest.mark.parametrize("close_eod", [False, True])
    @pytest.mark.parametrize("allow_short", [True, False])
    @pytest.mark.parametrize("max_position", [1.0, 0.5])
    @pytest.mark.parametrize("kind", ["int", "fractional"])
    def test_matches_loop(self, close_eod, allow_short, max_position, kind):
        rng = np.random.default_rng(
            1000 + 8 * close_eod + 4 * allow_short + 2 * (max_position == 0.5) + (kind == "int")
        )
        for _ in range(8):
            df, signal = _random_frame_and_signal(rng, int(rng.integers(30, 400)), kind=kind)
            cfg = BacktestConfig(strategy_name="s", symbol="X", interval="1h", fee_bps=2.0, slippage_bps=1.5,
                                 close_eod=close_eod, allow_short=allow_short, max_position=max_position)
            _assert_same_result(run_backtest(df, signal, cfg), run_backtest_loop(df, signal, cfg))

    @pytest.mark.parametrize("close_eod", [False, True])
    def test_matches_loop_with_missing_and_zero_prices(self, close_eod):
        rng = np.random.default_rng(77)
        for _ in range(10):
            df, signal = _random_frame_and_signal(rng, 300, nan_opens=True)
            cfg = BacktestConfig(strategy_name="s", symbol="X", interval="1h", close_eod=close_eod)
            _assert_same_result(run_backtest(df, signal, cfg), run_backtest_loop(df, signal, cfg))

    def test_falls_back_to_loop_for_nan_signal_and_max_holding(self):
        rng = np.random.default_rng(9)
        df, signal = _random_frame_and_signal(rng, 200)
        nan_signal = signal.copy()
        nan_signal.iloc[[5, 6, 50]] = np.nan
        cfg = BacktestConfig(strategy_name="s", symbol="X", interval="1h")
        _assert_same_result(run_backtest(df, nan_signal, cfg), run_backtest_loop(df, nan_signal, cfg))
        held = BacktestConfig(strategy_name="s", symbol="X", interval="1h", max_holding_bars=4)
        _assert_same_result(run_backtest(df, signal, held), run_backtest_loop(df, signal, held))

    def test_tiny_inputs(self):
        for n in (0, 1, 2):
            ts = pd.date_range("2024-03-01", periods=n, freq="1h", tz="UTC")
            df = pd.DataFrame({"timestamp": ts, "open": np.linspace(100, 101, n)})
            signal = pd.Series(np.ones(n))
            cfg = BacktestConfig(strategy_name="s", symbol="X", interval="1h")
            _assert_same_result(run_backtest(df, signal, cfg), run_backtest_loop(df, signal, cfg))

    def test_collect_trades_false_keeps_every_array(self):
        rng = np.random.default_rng(12)
        df, signal = _random_frame_and_signal(rng, 300)
        cfg = BacktestConfig(strategy_name="s", symbol="X", interval="1h")
        full = run_backtest(df, signal, cfg)
        lean = run_backtest(df, signal, cfg, collect_trades=False)
        assert lean.trades.empty and not full.trades.empty
        for name in ("positions", "gross_returns", "returns", "trade_units", "equity"):
            pd.testing.assert_series_equal(getattr(lean, name), getattr(full, name), check_exact=True)

    def test_units_are_total_absolute_position_change(self):
        rng = np.random.default_rng(14)
        df, signal = _random_frame_and_signal(rng, 500)
        out = backtest_arrays(df["open"].to_numpy(), signal.to_numpy(), 2.0, 2.0)
        assert out["units"].sum() == pytest.approx(np.abs(np.diff(out["position"])).sum(), abs=1e-12)
        assert out["position"][0] == 0.0 and out["units"][0] == 0.0


class TestEndToEndWithRealZ:
    def test_strategy_signal_and_backtest_match_loops(self):
        rng = np.random.default_rng(31)
        n = 6000
        ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
        s = pd.Series(close)
        mid, sd = s.rolling(50).mean(), s.rolling(50).std()
        df = pd.DataFrame({
            "timestamp": ts, "open": close, "close": close,
            "bb_mid_50": mid, "bb_upper_50_20": mid + 2.0 * sd,
        })
        z = compute_band_z(df, "bollinger", 50, 2.0)
        sig_new = signal_from_z(z, entry_long_z=1.0, entry_short_z=1.0, name="bollinger")
        sig_ref = _reversal_confirmed_position_loop(z, entry_long_z=1.0, entry_short_z=1.0).shift(1).fillna(0)
        assert np.array_equal(sig_new.to_numpy(), sig_ref.to_numpy())
        for eod in (False, True):
            cfg = BacktestConfig(strategy_name="b", symbol="X", interval="15m", fee_bps=2.0, slippage_bps=2.0,
                                 close_eod=eod)
            _assert_same_result(run_backtest(df, sig_new, cfg), run_backtest_loop(df, sig_ref.rename("bollinger"), cfg))
