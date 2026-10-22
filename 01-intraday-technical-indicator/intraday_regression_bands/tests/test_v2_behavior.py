"""Análises descritivas do estudo v2: calendário, volume e volatilidade, persistência e distribuição de ganhos e perdas."""

import types

import numpy as np
import pandas as pd
import pytest

from src.v2_behavior import (
    HALF_DAYS,
    acf_overlap_summary,
    autocorrelation,
    block_contributions,
    calendar_summary,
    decile_table,
    half_agreement,
    hourly_acf,
    hourly_counts,
    lagged_hour_rate,
    pearson_residuals,
    prepare_trades,
    rank_auc,
    recent_closed_rate,
    recent_performance_tests,
    regime_effects,
    seasonal_lag_tests,
    shrunk_rate,
    sorted_decile_contribution,
    tail_deciles,
    threshold_shape,
    weekend_by_year,
    weekend_mediation,
    win_loss_summary,
)
from src.v2_trades import OOS_START, market_features, trades_frame

N_DAYS = 1974


def _trades(n=40_000, seed=0, p_fn=None, vol_fn=None, lookbacks=(28,), strategies=("bollinger",)):
    """Operações sintéticas: entradas espalhadas pelos 1.974 dias de teste, com a chance de lucro dada por ``p_fn`` (dia da semana, hora, z_qv)."""
    rng = np.random.default_rng(seed)
    minutes = rng.integers(0, N_DAYS * 1440, n)
    ts = pd.DatetimeIndex(OOS_START + pd.to_timedelta(minutes, unit="m"))
    wd, hr = ts.dayofweek.to_numpy(), ts.hour.to_numpy()
    zq = vol_fn(wd, hr, rng) if vol_fn else rng.normal(size=n)
    p = p_fn(wd, hr, zq) if p_fn else np.full(n, 0.6)
    win = rng.random(n) < p
    gross = np.where(win, rng.exponential(20, n), -rng.exponential(30, n))
    df = pd.DataFrame({"freq": "1m", "strategy": rng.choice(strategies, n), "asset": "BTCUSDT", "lookback": rng.choice(lookbacks, n),
                       "entry_ts": ts, "exit_ts": ts + pd.to_timedelta(rng.integers(1, 120, n), unit="m"), "side": "long",
                       "gross_bps": gross, "holding_min": rng.integers(1, 120, n).astype(float)})
    feats = pd.DataFrame({"z_qv": zq, "z_rv": rng.normal(size=n), "s_rv": rng.normal(size=n), "s_qv": rng.normal(size=n)}, index=ts)
    feats = feats[~feats.index.duplicated()]
    return df, feats


def _prepared(**kw):
    df, feats = _trades(**kw)
    return prepare_trades(df, feats, require_features=True)


class TestPrepare:
    def test_calendar_columns_week_clusters_and_halves(self):
        ts = pd.DatetimeIndex(["2021-02-03 00:00", "2021-02-06 23:59", "2021-02-08 12:00", OOS_START + pd.Timedelta(days=HALF_DAYS)], tz="UTC")
        df = pd.DataFrame({"entry_ts": ts, "gross_bps": [5.0, -2.0, 0.0, 3.0], "strategy": ["bollinger", "regression_bands", "bollinger", "bollinger"],
                           "lookback": 3})
        g = prepare_trades(df)
        assert g["wd"].tolist() == [2, 5, 0, ts[3].dayofweek]               # 03/02/2021 foi quarta-feira
        assert g["hr"].tolist() == [0, 23, 12, 0] and g["year"].tolist() == [2021, 2021, 2021, 2023]
        assert g["win"].tolist() == [1.0, 0.0, 0.0, 1.0]                      # bruto zero não conta como lucro
        assert g["week"].iloc[0] == g["week"].iloc[1] != g["week"].iloc[2]    # 03 e 06/02 na mesma semana ISO; 08/02 na seguinte
        assert g["second_half"].tolist() == [False, False, False, True] and g["strat"].tolist() == [0.0, 1.0, 0.0, 0.0]

    def test_features_are_taken_at_the_entry_minute_and_missing_ones_can_be_dropped(self):
        ts = pd.DatetimeIndex(["2021-02-03 10:00", "2021-02-03 10:01"], tz="UTC")
        feats = pd.DataFrame({"z_rv": [np.nan, 2.0], "z_qv": [1.0, 3.0], "s_rv": [0.0, 0.0], "s_qv": [0.0, 0.0]}, index=ts)
        df = pd.DataFrame({"entry_ts": ts, "gross_bps": [1.0, 1.0], "strategy": "bollinger", "lookback": 1})
        assert len(prepare_trades(df, feats)) == 2
        g = prepare_trades(df, feats, require_features=True)
        assert len(g) == 1 and g["z_qv"].iloc[0] == 3.0


class TestCalendar:
    def test_planted_saturday_effect_is_found_and_hours_are_not(self):
        g = _prepared(seed=1, p_fn=lambda wd, hr, zq: np.where(wd == 5, 0.75, 0.60))
        s = calendar_summary(g)
        assert s["p_weekday"] < 1e-6 and s["p_weekday_given_hour"] < 1e-6
        assert 1.7 < s["OR_saturday"] < 2.4 and s["p_saturday"] < 1e-6        # exp(ln(3) - ln(1,5)) = 2
        assert s["p_hour"] > 0.01 and s["p_hour_given_weekday"] > 0.01
        assert s["AUC_oos_weekday"] > 0.5 and s["win_pct"] == pytest.approx(100 * g["win"].mean())

    def test_no_effect_gives_large_p_values_and_small_ors(self):
        s = calendar_summary(_prepared(seed=2))
        assert s["p_weekday"] > 0.01 and s["p_hour"] > 0.01 and s["p_session"] > 0.01
        assert 0.8 < s["OR_saturday"] < 1.25 and abs(s["sat_minus_mon_bps"]) < 6

    def test_planted_hour_effect_is_found(self):
        g = _prepared(seed=3, p_fn=lambda wd, hr, zq: np.where(hr == 12, 0.72, 0.60))
        s = calendar_summary(g)
        assert s["p_hour"] < 1e-6 and s["p_hour_given_weekday"] < 1e-6 and s["p_weekday"] > 0.01

    def test_weekend_by_year_returns_one_row_per_year_with_enough_trades(self):
        g = _prepared(seed=4, p_fn=lambda wd, hr, zq: np.where(wd >= 5, 0.70, 0.60))
        t = weekend_by_year(g)
        assert t["year"].tolist() == [2021, 2022, 2023, 2024, 2025, 2026] and (t["weekend_pp"] > 4).all() and (t["p"] < 0.05).all()


class TestRegime:
    VOL = staticmethod(lambda wd, hr, rng: rng.normal(size=len(wd)) - 0.8 * (wd >= 5))   # volume menor no fim de semana

    def test_volume_effect_has_the_planted_sign_and_deciles_are_monotone(self):
        g = _prepared(seed=5, vol_fn=self.VOL, p_fn=lambda wd, hr, zq: 1 / (1 + np.exp(-(0.4 - 0.25 * zq))))
        e = regime_effects(g)
        assert 0.7 < e["OR_volume_level"] < 0.85 and e["p_volume_level"] < 1e-6
        d = decile_table(g, "z_qv")
        assert len(d) == 10 and d["D1"] > d["D10"] and d.diff().dropna().lt(1.5).all()

    def test_weekend_effect_is_explained_by_volume(self):
        g = _prepared(seed=6, vol_fn=self.VOL, p_fn=lambda wd, hr, zq: 1 / (1 + np.exp(-(0.4 - 0.4 * zq))))
        m = weekend_mediation(g)
        assert m["OR_weekend_raw"] > 1.15 and m["p_weekend_raw"] < 0.01
        assert abs(np.log(m["OR_weekend_controlled"])) < abs(np.log(m["OR_weekend_raw"])) / 2

    def test_threshold_shape_finds_a_change_of_slope(self):
        g = _prepared(seed=7, p_fn=lambda wd, hr, zq: 1 / (1 + np.exp(-(0.4 - 0.7 * np.maximum(0, zq - 0.5)))))
        t = threshold_shape(g, "z_qv", "z_rv")
        assert t["OR_above"] < t["OR_below"] and t["p_change"] < 1e-3 and t["knot_percentile"] in (60, 75, 50)

    def test_block_contributions_attribute_a_volume_only_effect_to_volume(self):
        g = _prepared(seed=8, p_fn=lambda wd, hr, zq: 1 / (1 + np.exp(-(0.4 - 0.3 * zq))))
        b = block_contributions(g)
        assert b["unique_V_pct"] > 70 and b["p_V"] < 1e-6 and b["p_W"] > 0.01 and b["p_H"] > 0.01
        assert b["oos_dLL_V"] > 0 and b["oos_dLL_H"] < b["oos_dLL_V"]


class TestPersistence:
    def test_autocorrelation_of_independent_hours_is_near_zero_and_detects_planted_adjacency(self):
        rng = np.random.default_rng(0)
        n_hours = 24 * 300
        z = rng.normal(size=n_hours)
        ac = autocorrelation(z)
        assert all(abs(v) < 0.08 for v in ac.values())
        shared = np.convolve(rng.normal(size=n_hours + 1), [1, 1], mode="valid") / np.sqrt(2)     # horas vizinhas partilham metade do ruído
        assert autocorrelation(shared)[1] > 0.35 and abs(autocorrelation(shared)[24]) < 0.1

    def test_hourly_counts_and_pearson_residuals(self):
        ts = pd.DatetimeIndex(OOS_START + pd.to_timedelta([5, 20, 70, 130, 135], unit="m"))
        h, n, w = hourly_counts(ts, np.array([1.0, 0.0, 1.0, 1.0, 1.0]), 4)
        assert h.tolist() == [0, 0, 1, 2, 2] and n.tolist() == [2, 1, 2, 0] and w.tolist() == [1, 1, 2, 0]
        z = pearson_residuals(n, w, 0.5)
        assert z[0] == pytest.approx(0.0) and z[2] == pytest.approx(1 / np.sqrt(2 * 0.25)) and np.isnan(z[3])    # (2 - 2 x 0,5) / raiz(2 x 0,25)

    def test_duplicated_trades_across_lookbacks_are_dropped_in_the_dedup_view(self):
        df, _ = _trades(n=5_000, seed=9, lookbacks=(1,))
        twin = df.assign(lookback=2)
        both = pd.concat([df, twin], ignore_index=True)
        row = acf_overlap_summary(both, N_DAYS * 24)
        assert row["dedup_share_pct"] == pytest.approx(50.0) and np.isfinite(row["single_1h_mean"])
        assert hourly_acf(df, N_DAYS * 24)[1] == pytest.approx(hourly_acf(twin, N_DAYS * 24)[1])

    def test_lagged_rate_uses_the_hour_that_many_hours_before_and_nothing_earlier(self):
        rate = np.arange(100, dtype=float)
        got = lagged_hour_rate(rate, np.array([3, 30, 99]), 24)
        assert np.isnan(got[0]) and got[1] == 6.0 and got[2] == 75.0
        r = shrunk_rate(np.array([0.0, 10.0]), np.array([0.0, 10.0]), 0.6, prior=10.0)
        assert r[0] == pytest.approx(0.6) and r[1] == pytest.approx((10 + 6) / 20)

    def test_recent_closed_rate_counts_only_trades_that_have_already_closed(self):
        t0 = np.datetime64("2024-01-01T00:00")
        entry = np.array([t0 + np.timedelta64(300, "m"), t0 + np.timedelta64(300, "m")])
        exit_ = np.array([t0 + np.timedelta64(m, "m") for m in (200, 290, 301, 400)][:2])
        win = np.array([1.0, 0.0])
        # operações do conjunto: saem em 200 (ganho), 290 (perda), 301 (ganho, depois da entrada: não conta), 400 (não conta)
        ent_all = np.array([t0, t0, t0, t0]); ex_all = np.array([t0 + np.timedelta64(m, "m") for m in (200, 290, 301, 400)])
        w_all = np.array([1.0, 0.0, 1.0, 1.0])
        w1, n1 = recent_closed_rate(np.array([t0 + np.timedelta64(300, "m")]), ex_all, w_all, np.timedelta64(1, "h"))
        assert n1.tolist() == [1.0] and w1.tolist() == [0.0]               # só a de 290 (a janela de 1 h começa em 240)
        w3, n3 = recent_closed_rate(np.array([t0 + np.timedelta64(300, "m")]), ex_all, w_all, np.timedelta64(3, "h"))
        assert n3.tolist() == [2.0] and w3.tolist() == [1.0]

    def test_independent_outcomes_show_no_seasonal_or_recent_persistence(self):
        df, feats = _trades(n=40_000, seed=10)
        feats = feats.assign(z_rv3=np.random.default_rng(1).normal(size=len(feats)), z_qv3=np.random.default_rng(2).normal(size=len(feats)),
                             z_rv24=np.random.default_rng(3).normal(size=len(feats)), z_qv24=np.random.default_rng(4).normal(size=len(feats)))
        g = prepare_trades(df, feats, require_features=True)
        s = seasonal_lag_tests(g, N_DAYS * 24)
        assert 0.9 < s["OR_lag24_lags_only"] < 1.1 and s["p_lag168_lags_only"] > 0.01
        r = recent_performance_tests(g, g["exit_ts"])
        assert 0.9 < r["OR_z_r3_alone"] < 1.1 and 0.9 < r["OR_z_r1_regime_calendar"] < 1.1 and r["p_recent_joint"] > 0.01
        assert any(k.startswith("win_pct_Q") for k in r)

    def test_half_agreement_and_rank_auc(self):
        a = np.arange(24, dtype=float)
        rho, p = half_agreement(a, a * 2 + 1)
        assert rho == pytest.approx(1.0) and p < 1e-6
        assert np.isnan(half_agreement(np.array([1.0, 2.0]), np.array([1.0, 2.0]))[0])
        y = np.array([0, 0, 1, 1]); assert rank_auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0 and rank_auc(np.array([0.5] * 4), y) == 0.5


class TestDistribution:
    GROSS = np.array([10.0, 20.0, 30.0, -10.0, -20.0, -60.0, -200.0, 0.0])

    def test_summary_numbers(self):
        s = win_loss_summary(self.GROSS, np.array([5, 5, 5, 20, 20, 20, 20, 1.0]))
        assert s["hit_pct"] == pytest.approx(100 * 3 / 8) and s["mean_win"] == 20.0 and s["mean_loss"] == pytest.approx(-72.5)
        assert s["payoff"] == pytest.approx(20 / 72.5) and s["median_hold_min_wins"] == 5.0 and s["median_hold_min_losses"] == 20.0
        assert s["worst10pct_losers_share_of_loss"] == pytest.approx(100 * 200 / 290)

    def test_deciles_and_sorted_contribution(self):
        r = np.arange(-50, 50, dtype=float) + 0.5
        w, l = tail_deciles(r)
        assert w["p50"] == pytest.approx(np.percentile(r[r > 0], 50)) and l["p50"] == pytest.approx(np.percentile(-r[r < 0], 50))
        c = sorted_decile_contribution(r)
        assert len(c) == 10 and c.sum() == pytest.approx(0.0, abs=1e-9) and c[0] < 0 < c[-1]


class TestTradesAndMarket:
    def test_trades_frame_converts_the_backtest_table(self):
        tr = pd.DataFrame({"entry_timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00"]), "exit_timestamp": pd.to_datetime(["2024-01-01 01:00", "2024-01-01 03:00"]),
                           "side": ["long", "short"], "trade_return": [0.001, -0.002], "holding_bars": [4, 4]})
        f = trades_frame(types.SimpleNamespace(trades=tr), "15m", "bollinger", "BTCUSDT", 28)
        assert f["gross_bps"].tolist() == pytest.approx([10.0, -20.0]) and f["holding_min"].tolist() == [60.0, 60.0]
        assert str(f["entry_ts"].dt.tz) == "UTC" and f["lookback"].tolist() == [28, 28] and f["freq"].iloc[0] == "15m"
        assert trades_frame(types.SimpleNamespace(trades=pd.DataFrame()), "1m", "x", "y", 1).empty

    def test_market_features_are_causal_and_standardised(self):
        rng = np.random.default_rng(0)
        n = 4 * 1440
        close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
        bars = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"), "close": close,
                             "quote_volume": rng.uniform(1e5, 2e5, n)})
        f = market_features(bars)
        ret = np.log(close[1:] / close[:-1]) * 1e4
        t = 3000
        assert f["rv"].iloc[t] == pytest.approx(np.sqrt((ret[t - 61:t - 1] ** 2).sum()))   # retornos das barras t-60 a t-1 (fechamento de t-1 já é conhecido)
        assert f["qv"].iloc[t] == pytest.approx(bars["quote_volume"].iloc[t - 60:t].sum())
        bars2 = bars.copy(); bars2.loc[t, "close"] *= 1.05
        f2 = market_features(bars2)
        assert np.allclose(f["rv"].iloc[: t + 1], f2["rv"].iloc[: t + 1], equal_nan=True) and f["rv"].iloc[t + 1] != f2["rv"].iloc[t + 1]
        for c in ("z_rv", "z_qv", "z_rv3", "z_qv24", "s_rv", "s_qv"):
            v = f[c].dropna()
            assert abs(v.mean()) < 0.1 and abs(v.std() - 1) < 0.05, c
        assert f["rv"].iloc[:60].isna().all() and f["z_qv24"].iloc[:1440].isna().all()
