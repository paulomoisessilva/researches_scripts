"""Testes da análise agregada do estudo de 1m em blocos (src/full_study.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.full_study import (
    CANDIDATE,
    candidate_table,
    candidate_table_by_period,
    compare_with_pilot,
    cumulative_series,
    load_windows,
    make_families,
    pooled_table,
    restrict,
    share_gross_positive,
    spa_grid,
)


def _chunk(dates, symbols=("AAA", "BBB"), strategies=("bollinger", "regression_bands"),
           gross=0.001, units=2.0, cost_bps=4.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in symbols:
        for st in strategies:
            for d in dates:
                g = gross + rng.normal(0, 1e-4)
                u = units
                rows.append({
                    "strategy": st, "symbol": s, "interval": "1m", "test_date": d, "n_bars": 1440,
                    "sum_ret": g - u * cost_bps / 1e4, "sum_sq_ret": 1e-6, "sum_gross_ret": g,
                    "turnover_units": u, "number_of_trades": 1,
                })
    return pd.DataFrame(rows)


def _write_run(root, name, df, validation_days, train_days=2, fee=2.0, slip=2.0):
    run = root / name
    run.mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({
        "train_days": train_days, "validation_days": validation_days, "fee_bps": fee, "slippage_bps": slip,
    }), encoding="utf-8")
    df.to_csv(run / "walk_forward_results.csv", index=False)


@pytest.fixture
def runs_root(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    h1 = [d.strftime("%Y-%m-%d") for d in pd.date_range("2021-01-06", "2021-06-30")]
    h2 = [d.strftime("%Y-%m-%d") for d in pd.date_range("2021-07-01", "2021-12-30")]
    for v in (1, 2):
        _write_run(root, f"c1_va{v}", _chunk(h1, seed=v), v)
        _write_run(root, f"c2_va{v}", _chunk(h2, seed=10 + v), v)
    # run de outro treino: deve ser ignorada
    _write_run(root, "other_train", _chunk(h1), 1, train_days=5)
    return root


def test_load_windows_concatenates_chunks_and_filters_train(runs_root):
    w = load_windows(runs_root, 2, (2.0, 2.0))
    assert w.groupby(CANDIDATE).ngroups == 2 * 2 * 2
    assert w["test_date"].min() == "2021-01-06" and w["test_date"].max() == "2021-12-30"
    assert not w.duplicated([*CANDIDATE, "test_date"]).any()


def test_load_windows_identical_duplicates_dropped_divergent_raise(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    base = _chunk(["2021-01-06", "2021-01-07"])
    _write_run(root, "a", base, 1)
    _write_run(root, "b", base, 1)
    assert len(load_windows(root, 2, (2.0, 2.0))) == len(base)
    bad = base.copy()
    bad["sum_ret"] = bad["sum_ret"] + 0.01
    _write_run(root, "c", bad, 1)
    with pytest.raises(ValueError):
        load_windows(root, 2, (2.0, 2.0))


def test_candidate_cbe_is_gross_over_units(runs_root):
    w = load_windows(runs_root, 2, (2.0, 2.0))
    t = candidate_table(w)
    row = t.iloc[0]
    g = w[(w.symbol == row.symbol) & (w.strategy == row.strategy) & (w.validation_days == row.validation_days)]
    assert row.cbe_bps == pytest.approx(g.sum_gross_ret.sum() * 1e4 / g.turnover_units.sum())
    assert row.n_days == len(g)
    assert row.trades_per_day == pytest.approx(1.0)


def test_pooled_cbe_is_ratio_of_sums_and_sharpe_is_nan(runs_root):
    w = load_windows(runs_root, 2, (2.0, 2.0))
    p = pooled_table(w.assign(family="todas"), ["family"], None)
    assert p["cbe_bps"].iloc[0] == pytest.approx(w.sum_gross_ret.sum() * 1e4 / w.turnover_units.sum())
    assert p["n_candidates"].iloc[0] == 8
    assert np.isnan(p["sharpe_net"].iloc[0])


def test_by_period_splits_years_and_share_positive(runs_root):
    w = load_windows(runs_root, 2, (2.0, 2.0))
    t = candidate_table_by_period(w, "year")
    assert set(t["period"]) == {"2021"}
    s = share_gross_positive(w, "year")
    assert (s["share_gross_positive"] == 1.0).all()


def test_cumulative_series_matches_manual(runs_root):
    w = load_windows(runs_root, 2, (2.0, 2.0))
    c = cumulative_series(w)
    daily = w.groupby("test_date")["sum_gross_ret"].sum() / 8
    assert c["gross_pct"].iloc[-1] == pytest.approx((np.exp(daily.sum()) - 1) * 100)
    assert (c["net_pct"] < c["gross_pct"]).all()


def test_restrict_is_inclusive(runs_root):
    w = load_windows(runs_root, 2, (2.0, 2.0))
    r = restrict(w, "2021-03-01", "2021-03-31")
    assert r["test_date"].min() == "2021-03-01" and r["test_date"].max() == "2021-03-31"


def test_spa_grid_positive_edge_rejects_and_negative_does_not(tmp_path):
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2021-01-06", periods=300)]
    good = _chunk(dates, gross=0.004, units=1.0)           # líquido positivo a 4 bps
    bad = _chunk(dates, gross=0.0001, units=2.0, seed=1)    # líquido negativo
    fams = make_families(["AAA", "BBB"], ["bollinger", "regression_bands"])
    g = spa_grid(restrict(good.assign(validation_days=1), None, None), fams, [0.0, 4.0], reps=500, seed=0)
    b = spa_grid(bad.assign(validation_days=1), fams, [0.0, 4.0], reps=500, seed=0)
    assert g[(g.family == "todas") & (g.cost_bps_per_side == 4.0)]["spa_pvalue_consistent"].iloc[0] < 0.05
    assert b[(b.family == "todas") & (b.cost_bps_per_side == 4.0)]["spa_pvalue_consistent"].iloc[0] > 0.5


def test_compare_with_pilot_identical_and_perturbed(runs_root):
    w = load_windows(runs_root, 2, (2.0, 2.0))
    btc = w[w.symbol == "AAA"]
    res = compare_with_pilot(btc, btc, "2021-02-01", "2021-02-28")
    assert res["rows_matched"] == res["rows_full"] > 0
    assert res["sum_ret"]["share_equal"] == 1.0 and res["days_different"] == 0
    pert = btc.copy()
    pert.loc[pert.index[40], "sum_ret"] += 1e-6
    res2 = compare_with_pilot(btc, pert, "2021-01-06", "2021-12-30")
    assert res2["days_different"] == 1


def test_make_chunks_covers_every_test_day_once():
    from datetime import date

    from src.full_study import make_chunks

    chunks = make_chunks(date(2021, 1, 1), date(2025, 12, 31), date(2021, 1, 6), 7)
    assert [c["label"] for c in chunks][:3] == ["2021H1", "2021H2", "2022H1"]
    assert len(chunks) == 10
    assert chunks[0]["start_date"] == "2021-01-01" and chunks[0]["oos_start"] == "2021-01-06"
    for a, b in zip(chunks, chunks[1:]):
        assert a["end_date"] == b["oos_start"]                      # nenhum dia de teste fica sem bloco
        assert date.fromisoformat(b["oos_start"]) - date.fromisoformat(b["start_date"]) == __import__("datetime").timedelta(days=7)
    assert chunks[-1]["end_date"] == "2025-12-31"
