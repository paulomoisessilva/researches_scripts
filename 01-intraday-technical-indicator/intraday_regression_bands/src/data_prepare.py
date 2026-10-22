"""Transformação dos dados brutos (ZIP/CSV) em Parquet limpo."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd

from .config import get_project_root, load_config
from .data_validate import validate_dataframe, print_validation_report
from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

_RAW_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

_FINAL_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "symbol",
    "interval",
]

_FLOAT_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
]
_INT_COLS = ["number_of_trades"]

# Binance passou a incluir cabeçalho e renomeou algumas colunas a partir de 2022.
_COLUMN_RENAMES = {
    "count": "number_of_trades",
    "taker_buy_volume": "taker_buy_base_volume",
}


def _read_klines_csv(content: str) -> pd.DataFrame:
    """Lê CSV de klines da Binance (com ou sem cabeçalho)."""
    first_line = content.split("\n", 1)[0].strip()
    if first_line.startswith("open_time"):
        df = pd.read_csv(io.StringIO(content))
    else:
        df = pd.read_csv(
            io.StringIO(content),
            header=None,
            names=_RAW_COLUMNS,
        )
    return df.rename(columns=_COLUMN_RENAMES)


def load_zip_file(path: Path) -> pd.DataFrame:
    """Lê um arquivo ZIP contendo CSV de klines."""
    with zipfile.ZipFile(path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            logger.warning("Nenhum CSV dentro do ZIP: %s", path)
            return pd.DataFrame()
        with zf.open(csv_names[0]) as f:
            content = f.read().decode("utf-8")

    return _read_klines_csv(content)


def load_csv_file(path: Path) -> pd.DataFrame:
    """Lê um arquivo CSV de klines (REST API ou extraído)."""
    content = path.read_text(encoding="utf-8")
    return _read_klines_csv(content)


def _clean_dataframe(df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """Aplica limpeza e normalização ao DataFrame bruto."""
    if df.empty:
        return df

    # Colunas numéricas
    for col in _FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in _INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Timestamp em UTC (ms → datetime)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Remover coluna ignore e open_time original
    drop_cols = [c for c in ["ignore", "open_time"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Adicionar metadados
    df["symbol"] = symbol
    df["interval"] = interval

    # Remover duplicatas (manter primeira ocorrência)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Remover linhas com preços nulos
    df = df.dropna(subset=["open", "high", "low", "close"])

    return df[_FINAL_COLUMNS]


def prepare_symbol_interval(
    symbol: str,
    interval: str,
    raw_dir: Path,
    processed_dir: Path,
    force: bool = False,
) -> Path:
    """Consolida todos os arquivos brutos de um par symbol/interval em um único Parquet."""
    output_path = processed_dir / f"{symbol}_{interval}.parquet"

    if output_path.exists() and not force:
        logger.info("Já processado: %s", output_path.name)
        return output_path

    ensure_dir(processed_dir)

    # Coletar todos os arquivos disponíveis
    symbol_dir = raw_dir / symbol / interval
    if not symbol_dir.exists():
        logger.warning("Diretório não encontrado: %s", symbol_dir)
        return output_path

    all_files = sorted(symbol_dir.rglob("*.zip")) + sorted(symbol_dir.rglob("*.csv"))
    if not all_files:
        logger.warning("Nenhum arquivo encontrado em: %s", symbol_dir)
        return output_path

    logger.info("Processando %s %s — %d arquivos", symbol, interval, len(all_files))

    frames: list[pd.DataFrame] = []
    for file_path in all_files:
        try:
            if file_path.suffix == ".zip":
                df = load_zip_file(file_path)
            else:
                df = load_csv_file(file_path)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.warning("Erro ao ler %s: %s", file_path.name, exc)

    if not frames:
        logger.error("Nenhum dado válido para %s %s", symbol, interval)
        return output_path

    combined = pd.concat(frames, ignore_index=True)
    cleaned = _clean_dataframe(combined, symbol, interval)

    result = validate_dataframe(cleaned, symbol, interval)
    if not result.ok:
        for msg in result.messages:
            logger.warning("[%s %s] %s", symbol, interval, msg)

    cleaned.to_parquet(output_path, index=False, compression="snappy")
    logger.info(
        "Salvo: %s (%d linhas, %s → %s)",
        output_path.name,
        len(cleaned),
        cleaned["timestamp"].min(),
        cleaned["timestamp"].max(),
    )
    return output_path


def interval_minutes(interval: str) -> int:
    """Minutos de um intervalo da Binance ("1m", "15m", "1h", "1d")."""
    factor = {"m": 1, "h": 60, "d": 1440}.get(interval[-1:])
    if factor is None or not interval[:-1].isdigit():
        raise ValueError(f"Intervalo não suportado: '{interval}'.")
    return int(interval[:-1]) * factor


def aggregate_klines(df: pd.DataFrame, interval: str, symbol: str | None = None) -> pd.DataFrame:
    """Agrega candles de menor intervalo em candles de ``interval`` (ex.: 1m -> 5m e 15m).

    Barras alinhadas ao relógio UTC, como na Binance: open da primeira barra, máxima, mínima, close
    da última, e soma de volume, volume em USDT, número de negociações e volumes do lado comprador.
    Se faltar um minuto dentro da barra, a barra sai das barras disponíveis (como a barra nativa, que
    sai das negociações que existem); ``close_time`` segue a convenção da Binance (abertura + intervalo
    − 1 ms).
    """
    minutes = interval_minutes(interval)
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    bins = pd.DatetimeIndex(ordered["timestamp"]).floor(f"{minutes}min")
    grouped = ordered.groupby(bins, sort=True)
    out = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
        number_of_trades=("number_of_trades", "sum"),
        taker_buy_base_volume=("taker_buy_base_volume", "sum"),
        taker_buy_quote_volume=("taker_buy_quote_volume", "sum"),
    )
    source_minutes = (
        interval_minutes(str(ordered["interval"].iloc[0])) if len(ordered) and "interval" in ordered.columns else 1
    )
    n_incomplete = int((grouped.size() < max(minutes // source_minutes, 1)).sum())
    out.index.name = "timestamp"
    out = out.reset_index()
    out["close_time"] = out["timestamp"] + pd.Timedelta(minutes=minutes) - pd.Timedelta(milliseconds=1)
    out["symbol"] = symbol if symbol is not None else (ordered["symbol"].iloc[0] if len(ordered) else "")
    out["interval"] = interval
    if n_incomplete:
        logger.info("%s %s: %d barras agregadas com minutos faltando.", out["symbol"].iloc[0], interval, n_incomplete)
    return out[_FINAL_COLUMNS]


def prepare_aggregated_interval(
    symbol: str,
    interval: str,
    source_interval: str,
    processed_dir: Path,
    force: bool = False,
) -> Path:
    """Gera ``{symbol}_{interval}.parquet`` agregando o Parquet processado de ``source_interval``."""
    output_path = processed_dir / f"{symbol}_{interval}.parquet"
    if output_path.exists() and not force:
        logger.info("Já processado: %s", output_path.name)
        return output_path

    source_path = processed_dir / f"{symbol}_{source_interval}.parquet"
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} não existe; prepare primeiro o intervalo de origem ({source_interval})."
        )
    source = pd.read_parquet(source_path)
    aggregated = aggregate_klines(source, interval, symbol=symbol)

    result = validate_dataframe(aggregated, symbol, interval)
    if not result.ok:
        for msg in result.messages:
            logger.warning("[%s %s] %s", symbol, interval, msg)

    aggregated.to_parquet(output_path, index=False, compression="snappy")
    logger.info(
        "Salvo: %s (%d linhas, %s → %s), agregado de %s",
        output_path.name, len(aggregated), aggregated["timestamp"].min(), aggregated["timestamp"].max(),
        source_interval,
    )
    return output_path


def prepare_all(
    cfg: dict,
    force: bool = False,
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
) -> list[Path]:
    """Prepara todos os pares symbol/interval conforme configuração.

    Com ``data.aggregate_from`` (ex.: "1m"), os intervalos do config saem da agregação do Parquet
    processado desse intervalo, que deve estar na mesma ``processed_dir``.
    """
    data_cfg = cfg["data"]
    syms = symbols or data_cfg["symbols"]
    ints = intervals or data_cfg["intervals"]
    root = get_project_root()
    raw_dir = root / data_cfg["raw_dir"]
    processed_dir = root / data_cfg["processed_dir"]
    source_interval = data_cfg.get("aggregate_from")

    results: list[Path] = []
    for symbol in syms:
        for interval in ints:
            if source_interval:
                path = prepare_aggregated_interval(symbol, interval, source_interval, processed_dir, force=force)
            else:
                path = prepare_symbol_interval(symbol, interval, raw_dir, processed_dir, force=force)
            results.append(path)

    return results
