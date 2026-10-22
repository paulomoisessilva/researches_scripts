"""Os configs do estudo v2 (1, 5 e 15 minutos) seguem o PREREGISTRATION_v2.md."""

from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.fixed_params import grid_tasks
from src.walk_forward import build_windows

ROOT = Path(__file__).parent.parent
CONFIGS = {"1m": "config.v2_1m.yaml", "5m": "config.v2_5m.yaml", "15m": "config.v2_15m.yaml"}
MINUTES = {"1m": 1, "5m": 5, "15m": 15}

# janelas em horas: comuns (2, 4, 8, 16) e próprias de cada frequência
EXPECTED_WINDOW_HOURS = {
    "1m": [0.5, 1, 2, 4, 8, 16],
    "5m": [1, 2, 4, 8, 16, 24],
    "15m": [2, 4, 8, 16, 24, 36, 48, 60],
}
EXPECTED_COMBOS_PER_ASSET = {"1m": 144, "5m": 144, "15m": 192}
K_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
LOOKBACKS = [1, 2, 3, 7, 14, 21, 28]


def _cfg(freq: str) -> dict:
    return load_config(ROOT / CONFIGS[freq])


@pytest.fixture(params=list(CONFIGS))
def freq(request):
    return request.param


class TestEachConfig:
    def test_windows_are_defined_in_clock_time(self, freq):
        cfg = _cfg(freq)
        bars_per_hour = 60 / MINUTES[freq]
        for key in ("bollinger_windows", "regression_windows"):
            hours = [w / bars_per_hour for w in cfg["features"][key]]
            assert hours == EXPECTED_WINDOW_HOURS[freq]
        assert cfg["features"]["bollinger_windows"] == cfg["features"]["regression_windows"]

    def test_combinations_per_asset(self, freq):
        cfg = _cfg(freq)
        assert len(grid_tasks(cfg["strategies"], cfg)) == EXPECTED_COMBOS_PER_ASSET[freq]

    def test_common_grid_and_rules(self, freq):
        cfg = _cfg(freq)
        assert cfg["strategies"] == ["bollinger", "regression_bands"]
        assert cfg["features"]["bollinger_k"] == K_GRID and cfg["features"]["regression_k"] == K_GRID
        assert cfg["features"]["compact_bands"] is True
        s = cfg["strategy"]
        assert s["entry_mode"] == "reversal_confirmed"
        assert s["entry_long_z"] == [1.0] and s["entry_short_z"] == [1.0]
        assert s["exit_long_z"] == [0.0] and s["exit_short_z"] == [0.0]
        assert cfg["data"]["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        assert cfg["data"]["intervals"] == [freq]

    def test_walk_forward_setup(self, freq):
        wf = _cfg(freq)["walk_forward"]
        assert (wf["train_days"], wf["test_days"], wf["step_days"]) == (2, 1, 1)
        assert wf["close_position_end_of_day"] is False
        assert wf["carry_position_state"] is True
        assert wf["selection_metric"] == "net_sharpe"
        assert wf["selection_engine"] == "single_pass"
        assert wf["parallel_workers"] == 1
        assert wf["runs_dir"] == f"results/v2/{freq}/walk_forward"

    def test_lookbacks_and_cost_zero_selection(self, freq):
        cfg = _cfg(freq)
        assert cfg["study"]["validation_days"] == LOOKBACKS
        assert cfg["study"]["cost_scenarios"] == [[0, 0]]
        assert cfg["backtest"]["close_position_end_of_day"] is False
        assert cfg["backtest"]["max_holding_bars"] is None
        assert cfg["backtest"]["fee_bps_per_side"] == [0] and cfg["backtest"]["slippage_bps_per_side"] == [0]

    def test_dates_and_test_days(self, freq):
        cfg = _cfg(freq)
        data, wf = cfg["data"], cfg["walk_forward"]
        assert data["start_date"] == "2021-01-04" and data["end_date"] == "2026-07-01"
        # primeiro teste 30 dias depois do início dos dados (2 de treino + 28 do maior lookback)
        first_test = pd.Timestamp(wf["oos_start"])
        assert first_test - pd.Timedelta(days=wf["train_days"] + max(LOOKBACKS)) == pd.Timestamp(data["start_date"])
        assert first_test == pd.Timestamp("2021-02-03")
        last_test = pd.Timestamp(data["end_date"]) - pd.Timedelta(days=1)
        assert last_test == pd.Timestamp("2026-06-30")
        assert len(pd.date_range(first_test, last_test)) == 1974

    def test_windows_cover_the_planned_test_days_for_every_lookback(self, freq):
        cfg = _cfg(freq)
        minutes = MINUTES[freq]
        ts = pd.date_range("2021-01-04", "2026-07-01 23:59", freq=f"{minutes}min", tz="UTC")
        df = pd.DataFrame({"timestamp": ts})
        wf = cfg["walk_forward"]
        for lookback in LOOKBACKS:
            windows = build_windows(df, wf["train_days"], lookback, wf["test_days"], wf["step_days"],
                                    oos_start=wf["oos_start"])
            assert len(windows) == 1974
            assert windows[0].test_start == pd.Timestamp("2021-02-03", tz="UTC")
            assert windows[-1].test_start == pd.Timestamp("2026-06-30", tz="UTC")
            assert all(a.test_end == b.test_start for a, b in zip(windows[:-1], windows[1:]))


class TestAcrossConfigs:
    def test_same_rules_in_the_three_frequencies(self):
        cfgs = {f: _cfg(f) for f in CONFIGS}
        for key in ("strategy", "study"):
            assert cfgs["1m"][key] == cfgs["5m"][key] == cfgs["15m"][key]
        for f in CONFIGS:
            wf = {k: v for k, v in cfgs[f]["walk_forward"].items() if k != "runs_dir"}
            assert wf == {k: v for k, v in cfgs["1m"]["walk_forward"].items() if k != "runs_dir"}
        assert cfgs["1m"]["features"]["bollinger_k"] == cfgs["5m"]["features"]["bollinger_k"] \
            == cfgs["15m"]["features"]["bollinger_k"]

    def test_5m_and_15m_come_from_the_1m_files(self):
        cfgs = {f: _cfg(f) for f in CONFIGS}
        assert "aggregate_from" not in cfgs["1m"]["data"]
        assert cfgs["5m"]["data"]["aggregate_from"] == "1m" and cfgs["15m"]["data"]["aggregate_from"] == "1m"
        assert len({c["data"]["processed_dir"] for c in cfgs.values()}) == 1
        assert cfgs["1m"]["data"]["processed_dir"] == "data/processed_v2"
        assert len({c["walk_forward"]["runs_dir"] for c in cfgs.values()}) == 3

    def test_candidate_counts_match_the_protocol(self):
        n_assets = len(_cfg("1m")["data"]["symbols"])
        n_strategies = len(_cfg("1m")["strategies"])
        # candidatas do teste primário: ativo x estratégia x lookback x frequência
        assert n_assets * n_strategies * len(LOOKBACKS) * len(CONFIGS) == 126
        # diagnóstico de parâmetros fixos: todas as combinações por ativo e frequência
        assert n_assets * sum(EXPECTED_COMBOS_PER_ASSET.values()) == 1440
