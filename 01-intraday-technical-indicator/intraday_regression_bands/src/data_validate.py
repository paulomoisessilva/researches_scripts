"""Validação de integridade dos dados brutos e processados."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from .utils import setup_logger, bars_per_day

logger = setup_logger(__name__)


class ValidationResult(NamedTuple):
    symbol: str
    interval: str
    ok: bool
    n_rows: int
    n_duplicates: int
    n_gaps: int
    n_nulls: int
    messages: list[str]


def validate_dataframe(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
) -> ValidationResult:
    """Valida um DataFrame de candles processados."""
    messages: list[str] = []
    ok = True

    required_cols = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        messages.append(f"Colunas ausentes: {missing}")
        ok = False

    n_rows = len(df)
    if n_rows == 0:
        messages.append("DataFrame vazio.")
        return ValidationResult(symbol, interval, False, 0, 0, 0, 0, messages)

    # Duplicatas
    n_dups = int(df["timestamp"].duplicated().sum())
    if n_dups > 0:
        messages.append(f"{n_dups} timestamps duplicados.")
        ok = False

    # Nulos em colunas críticas
    critical = ["open", "high", "low", "close", "volume"]
    n_nulls = int(df[critical].isnull().sum().sum())
    if n_nulls > 0:
        messages.append(f"{n_nulls} valores nulos em colunas OHLCV.")
        ok = False

    # Verificação OHLC
    invalid_ohlc = (
        (df["high"] < df["low"]) |
        (df["close"] > df["high"]) |
        (df["close"] < df["low"]) |
        (df["open"] > df["high"]) |
        (df["open"] < df["low"])
    ).sum()
    if invalid_ohlc > 0:
        messages.append(f"{invalid_ohlc} candles com OHLC inconsistente.")
        ok = False

    # Verificação de buracos temporais
    n_gaps = _count_temporal_gaps(df, interval)
    if n_gaps > 0:
        messages.append(f"{n_gaps} buracos temporais detectados.")

    if ok and not messages:
        messages.append("OK")

    return ValidationResult(symbol, interval, ok, n_rows, n_dups, n_gaps, n_nulls, messages)


def _count_temporal_gaps(df: pd.DataFrame, interval: str) -> int:
    """Conta buracos temporais maiores que 2× o período esperado."""
    interval_ms_map = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    expected_ms = interval_ms_map.get(interval)
    if expected_ms is None:
        return 0

    ts = df["timestamp"].sort_values().reset_index(drop=True)
    diffs = ts.diff().dropna()
    threshold = pd.Timedelta(milliseconds=expected_ms * 2)
    gaps = (diffs > threshold).sum()
    return int(gaps)


def validate_parquet_file(
    path: str | Path,
    symbol: str,
    interval: str,
) -> ValidationResult:
    """Carrega e valida um arquivo Parquet processado."""
    path = Path(path)
    if not path.exists():
        return ValidationResult(
            symbol, interval, False, 0, 0, 0, 0, [f"Arquivo não encontrado: {path}"]
        )
    df = pd.read_parquet(path)
    return validate_dataframe(df, symbol, interval)


def print_validation_report(results: list[ValidationResult]) -> None:
    failed = [r for r in results if not r.ok]
    passed = [r for r in results if r.ok]

    logger.info("=" * 60)
    logger.info("RELATÓRIO DE VALIDAÇÃO")
    logger.info("Total: %d | OK: %d | Falha: %d", len(results), len(passed), len(failed))
    logger.info("=" * 60)

    for r in results:
        status = "OK" if r.ok else "FALHA"
        logger.info(
            "[%s] %s %s — %d linhas, %d dups, %d gaps, %d nulls",
            status, r.symbol, r.interval, r.n_rows, r.n_duplicates, r.n_gaps, r.n_nulls,
        )
        for msg in r.messages:
            if msg != "OK":
                logger.warning("  -> %s", msg)
