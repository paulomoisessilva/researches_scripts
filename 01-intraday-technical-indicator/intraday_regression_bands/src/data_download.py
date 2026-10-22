"""Download de dados históricos da Binance USD-M Futures.

Fonte primária : data.binance.vision (arquivos ZIP mensais/diários)
Fallback       : Binance REST API /fapi/v1/klines
"""

from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import requests

from .config import load_config, get_project_root
from .utils import ensure_dir, setup_logger, sha256_file, date_range_months, date_range_days

logger = setup_logger(__name__)

BASE_VISION = "https://data.binance.vision/data/futures/um/monthly/klines"
BASE_VISION_DAILY = "https://data.binance.vision/data/futures/um/daily/klines"
BASE_REST = "https://fapi.binance.com/fapi/v1/klines"

_COLUMNS = [
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_monthly_klines(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    raw_dir: str | Path,
    force: bool = False,
) -> list[Path]:
    """Baixa arquivos mensais de klines do data.binance.vision.

    Returns
    -------
    Lista de caminhos dos arquivos salvos.
    """
    raw_dir = Path(raw_dir)
    saved: list[Path] = []

    for year, month in date_range_months(start_date, end_date):
        month_str = f"{year}-{month:02d}"
        filename = f"{symbol}-{interval}-{month_str}.zip"
        dest = raw_dir / symbol / interval / filename

        if dest.exists() and not force:
            logger.debug("Já existe (mensal): %s", dest)
            saved.append(dest)
            continue

        url = f"{BASE_VISION}/{symbol}/{interval}/{filename}"
        result = _download_file(url, dest)
        if result:
            saved.append(result)
            verify_checksum_if_available(dest, url)

    return saved


def download_daily_klines(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    raw_dir: str | Path,
    force: bool = False,
) -> list[Path]:
    """Baixa arquivos diários de klines do data.binance.vision."""
    raw_dir = Path(raw_dir)
    saved: list[Path] = []

    for dt in date_range_days(start_date, end_date):
        date_str = dt.strftime("%Y-%m-%d")
        filename = f"{symbol}-{interval}-{date_str}.zip"
        dest = raw_dir / symbol / interval / "daily" / filename

        if dest.exists() and not force:
            logger.debug("Já existe (diário): %s", dest)
            saved.append(dest)
            continue

        url = f"{BASE_VISION_DAILY}/{symbol}/{interval}/{filename}"
        result = _download_file(url, dest)
        if result:
            saved.append(result)
            verify_checksum_if_available(dest, url)

    return saved


def download_from_rest_api(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    raw_dir: str | Path,
    force: bool = False,
) -> Path:
    """Baixa dados via Binance REST API /fapi/v1/klines e salva em CSV.

    Usado como fallback quando data.binance.vision não tem o arquivo.
    """
    raw_dir = Path(raw_dir)
    dest = raw_dir / symbol / interval / f"{symbol}_{interval}_rest.csv"

    if dest.exists() and not force:
        logger.info("Arquivo REST já existe: %s", dest)
        return dest

    ensure_dir(dest.parent)
    rows: list[list] = []

    start_ms = _date_to_ms(start_date)
    end_ms = _date_to_ms(end_date) + 86_400_000  # inclusivo

    logger.info("REST API: %s %s %s → %s", symbol, interval, start_date, end_date)
    limit = 1000
    current_ms = start_ms

    while current_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_ms,
            "endTime": end_ms - 1,
            "limit": limit,
        }
        try:
            resp = requests.get(BASE_REST, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Erro na REST API: %s. Tentando novamente...", exc)
            time.sleep(2)
            continue

        if not data:
            break

        rows.extend(data)
        last_close_time = data[-1][6]
        current_ms = last_close_time + 1

        if len(data) < limit:
            break

        time.sleep(0.1)

    if not rows:
        logger.warning("Nenhum dado retornado pela REST API para %s %s", symbol, interval)
        return dest

    import csv

    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_COLUMNS)
        writer.writerows(rows)

    logger.info("Salvo via REST: %s (%d linhas)", dest, len(rows))
    return dest


def verify_checksum_if_available(file_path: str | Path, base_url: str = "") -> bool:
    """Verifica checksum SHA256 se o arquivo .CHECKSUM estiver disponível.

    Returns
    -------
    True se verificado com sucesso ou se checksum não disponível, False se divergir.
    """
    file_path = Path(file_path)
    checksum_url = base_url + ".CHECKSUM" if base_url else ""

    if not file_path.exists():
        return False

    if not checksum_url:
        return True

    try:
        resp = requests.get(checksum_url, timeout=10)
        if resp.status_code != 200:
            return True  # checksum não disponível
        expected_hash = resp.text.strip().split()[0].lower()
        actual_hash = sha256_file(file_path)
        if expected_hash != actual_hash:
            logger.error(
                "Checksum divergente para %s: esperado=%s, calculado=%s",
                file_path.name,
                expected_hash,
                actual_hash,
            )
            return False
        logger.debug("Checksum OK: %s", file_path.name)
        return True
    except Exception as exc:
        logger.debug("Não foi possível verificar checksum: %s", exc)
        return True


def download_all(
    cfg: dict,
    force: bool = False,
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
) -> None:
    """Baixa todos os dados conforme a configuração."""
    data_cfg = cfg["data"]
    if data_cfg.get("aggregate_from"):
        logger.info(
            "Intervalos %s saem da agregação de %s (data.aggregate_from): nada a baixar.",
            intervals or data_cfg["intervals"], data_cfg["aggregate_from"],
        )
        return
    syms = symbols or data_cfg["symbols"]
    ints = intervals or data_cfg["intervals"]
    start = data_cfg["start_date"]
    end = data_cfg["end_date"]
    raw_dir = get_project_root() / data_cfg["raw_dir"]

    log_entries: list[dict] = []

    for symbol in syms:
        for interval in ints:
            logger.info("Baixando %s %s (%s → %s)", symbol, interval, start, end)

            monthly = download_monthly_klines(symbol, interval, start, end, raw_dir, force=force)
            log_entries.append({
                "symbol": symbol,
                "interval": interval,
                "source": "monthly",
                "files": len(monthly),
            })

            # Dados do mês atual — tenta diário (mais recente que mensal)
            today = datetime.utcnow()
            last_complete_month = (
                datetime(today.year, today.month, 1) - timedelta(days=1)
            ).strftime("%Y-%m-%d")
            if end > last_complete_month:
                daily = download_daily_klines(
                    symbol, interval, last_complete_month, end, raw_dir, force=force
                )
                log_entries.append({
                    "symbol": symbol,
                    "interval": interval,
                    "source": "daily",
                    "files": len(daily),
                })

    # Salva log
    log_path = raw_dir / "download_log.json"
    ensure_dir(log_path.parent)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2)
    logger.info("Log de download salvo em: %s", log_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path) -> Optional[Path]:
    ensure_dir(dest.parent)
    try:
        resp = requests.get(url, timeout=60, stream=True)
        if resp.status_code == 404:
            logger.debug("Não encontrado (404): %s", url)
            return None
        resp.raise_for_status()

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        logger.info("Baixado: %s", dest.name)
        return dest

    except requests.RequestException as exc:
        logger.warning("Falha ao baixar %s: %s", url, exc)
        return None


def _date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)
