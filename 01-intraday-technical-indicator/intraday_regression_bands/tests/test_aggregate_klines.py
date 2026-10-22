"""Agregação do 1 minuto em 5 e 15 minutos (D24)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.data_download as data_download
from src.data_prepare import (
    _FINAL_COLUMNS,
    aggregate_klines,
    interval_minutes,
    prepare_aggregated_interval,
    prepare_all,
)

ROOT = Path(__file__).parent.parent


def _minutes(n=45, start="2024-01-01 00:00", symbol="BTCUSDT"):
    ts = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    i = np.arange(n, dtype=float)
    return pd.DataFrame({
        "timestamp": ts,
        "open": i, "high": i + 0.5, "low": i - 0.5, "close": i + 0.25,
        "volume": np.ones(n), "close_time": ts + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1),
        "quote_volume": np.full(n, 10.0), "number_of_trades": pd.array(np.full(n, 2), dtype="Int64"),
        "taker_buy_base_volume": np.full(n, 0.4), "taker_buy_quote_volume": np.full(n, 4.0),
        "symbol": symbol, "interval": "1m",
    })


class TestAggregation:
    def test_values_of_15_minute_bars(self):
        out = aggregate_klines(_minutes(45), "15m", symbol="BTCUSDT")
        assert list(out.columns) == _FINAL_COLUMNS
        assert len(out) == 3
        assert out["timestamp"].tolist() == list(pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"))
        assert out["open"].tolist() == [0.0, 15.0, 30.0]
        assert out["close"].tolist() == [14.25, 29.25, 44.25]
        assert out["high"].tolist() == [14.5, 29.5, 44.5]
        assert out["low"].tolist() == [-0.5, 14.5, 29.5]
        assert out["volume"].tolist() == [15.0] * 3
        assert out["quote_volume"].tolist() == [150.0] * 3
        assert out["number_of_trades"].tolist() == [30] * 3
        assert str(out["number_of_trades"].dtype) == "Int64"
        assert out["taker_buy_base_volume"].tolist() == pytest.approx([6.0] * 3)
        assert out["taker_buy_quote_volume"].tolist() == pytest.approx([60.0] * 3)
        assert (out["symbol"] == "BTCUSDT").all() and (out["interval"] == "15m").all()

    def test_close_time_follows_the_binance_convention(self):
        out = aggregate_klines(_minutes(30), "5m", symbol="BTCUSDT")
        expected = out["timestamp"] + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1)
        assert out["close_time"].equals(expected)
        assert len(out) == 6

    def test_bars_are_aligned_to_the_utc_clock_even_when_the_data_starts_mid_bar(self):
        out = aggregate_klines(_minutes(12, start="2024-01-01 00:03"), "5m", symbol="BTCUSDT")
        assert out["timestamp"].tolist() == list(pd.date_range("2024-01-01 00:00", periods=3, freq="5min", tz="UTC"))
        assert out["volume"].tolist() == [2.0, 5.0, 5.0]  # a primeira barra só tem 00:03 e 00:04

    def test_a_missing_minute_leaves_the_bar_built_from_the_minutes_that_exist(self):
        df = _minutes(15).drop(index=[3, 14]).reset_index(drop=True)
        out = aggregate_klines(df, "15m", symbol="BTCUSDT")
        assert len(out) == 1
        assert out["volume"].iloc[0] == 13.0
        assert out["open"].iloc[0] == 0.0 and out["close"].iloc[0] == 13.25  # último minuto presente
        assert out["number_of_trades"].iloc[0] == 26

    def test_unsorted_input_is_handled(self):
        df = _minutes(30).sample(frac=1.0, random_state=1)
        out = aggregate_klines(df, "15m", symbol="BTCUSDT")
        assert out["open"].tolist() == [0.0, 15.0] and out["close"].tolist() == [14.25, 29.25]

    def test_interval_minutes(self):
        assert [interval_minutes(x) for x in ("1m", "5m", "15m", "1h", "4h", "1d")] == [1, 5, 15, 60, 240, 1440]
        for bad in ("m", "xm", "5x", ""):
            with pytest.raises(ValueError, match="não suportado"):
                interval_minutes(bad)


class TestPrepareFromConfig:
    def _cfg(self, tmp_path, intervals):
        return {"data": {"symbols": ["BTCUSDT"], "intervals": intervals, "aggregate_from": "1m",
                         "start_date": "2024-01-01", "end_date": "2024-01-02",
                         "raw_dir": str(tmp_path / "raw"), "processed_dir": str(tmp_path / "processed")}}

    def test_prepare_all_builds_5m_and_15m_from_the_processed_1m(self, tmp_path):
        processed = tmp_path / "processed"
        processed.mkdir()
        _minutes(120).to_parquet(processed / "BTCUSDT_1m.parquet", index=False)
        paths = prepare_all(self._cfg(tmp_path, ["5m", "15m"]))
        assert [p.name for p in paths] == ["BTCUSDT_5m.parquet", "BTCUSDT_15m.parquet"]
        five = pd.read_parquet(paths[0])
        assert len(five) == 24 and (five["interval"] == "5m").all()
        assert len(pd.read_parquet(paths[1])) == 8

    def test_existing_output_is_kept_unless_forced(self, tmp_path):
        processed = tmp_path / "processed"
        processed.mkdir()
        _minutes(60).to_parquet(processed / "BTCUSDT_1m.parquet", index=False)
        path = prepare_aggregated_interval("BTCUSDT", "15m", "1m", processed)
        marker = path.stat().st_mtime_ns
        assert prepare_aggregated_interval("BTCUSDT", "15m", "1m", processed).stat().st_mtime_ns == marker
        _minutes(120).to_parquet(processed / "BTCUSDT_1m.parquet", index=False)
        assert len(pd.read_parquet(prepare_aggregated_interval("BTCUSDT", "15m", "1m", processed))) == 4
        assert len(pd.read_parquet(prepare_aggregated_interval("BTCUSDT", "15m", "1m", processed, force=True))) == 8

    def test_missing_source_interval_is_an_error(self, tmp_path):
        (tmp_path / "processed").mkdir()
        with pytest.raises(FileNotFoundError, match="1m"):
            prepare_aggregated_interval("BTCUSDT", "15m", "1m", tmp_path / "processed")

    def test_download_is_skipped_for_aggregated_intervals(self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("não deveria baixar nada")
        monkeypatch.setattr(data_download, "download_monthly_klines", boom)
        monkeypatch.setattr(data_download, "download_daily_klines", boom)
        assert data_download.download_all(self._cfg(tmp_path, ["5m"])) is None


@pytest.mark.skipif(
    not (ROOT / "data/processed_full_1m/BTCUSDT_1m.parquet").exists()
    or not (ROOT / "data/processed/BTCUSDT_15m.parquet").exists(),
    reason="dados locais (não versionados) ausentes",
)
def test_aggregated_15m_reproduces_the_native_files_on_real_btc_data():
    one = pd.read_parquet(ROOT / "data/processed_full_1m/BTCUSDT_1m.parquet")
    native = pd.read_parquet(ROOT / "data/processed/BTCUSDT_15m.parquet").set_index("timestamp")
    agg = aggregate_klines(one, "15m", symbol="BTCUSDT").set_index("timestamp")
    assert agg.index.equals(native.index)
    ohlc = ["open", "high", "low", "close"]
    differing = (agg[ohlc] != native[ohlc]).any(axis=1).sum()
    assert differing / len(agg) < 5e-4  # medido: 18 de 175.296 barras (0,010%)
    assert (agg["close_time"] == native["close_time"]).all()
