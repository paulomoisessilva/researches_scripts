"""Utilitários gerais do projeto."""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configura e retorna um logger com handler de console."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(path: str | Path) -> Path:
    """Garante que o diretório existe, criando-o se necessário."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | Path) -> str:
    """Calcula SHA-256 de um arquivo."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def date_range_months(start: str, end: str) -> Generator[tuple[int, int], None, None]:
    """Gera pares (ano, mês) entre start e end (inclusivo).

    Parameters
    ----------
    start, end:
        Datas no formato 'YYYY-MM-DD'.
    """
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    current = datetime(s.year, s.month, 1)
    while current <= e:
        yield current.year, current.month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)


def date_range_days(start: str, end: str) -> Generator[datetime, None, None]:
    """Gera datas diárias entre start e end (inclusivo)."""
    from datetime import timedelta

    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    current = s
    while current <= e:
        yield current
        current += timedelta(days=1)


def retry_request(func, max_retries: int = 5, backoff: float = 1.5):
    """Decorator/wrapper para retentar chamadas HTTP com backoff exponencial."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = backoff ** attempt
            time.sleep(wait)
    return None


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def bars_per_day(interval: str) -> float:
    """Retorna o número aproximado de candles por dia para um dado intervalo."""
    mapping = {
        "1m": 1440,
        "3m": 480,
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "2h": 12,
        "4h": 6,
        "6h": 4,
        "8h": 3,
        "12h": 2,
        "1d": 1,
    }
    return mapping.get(interval, 1)
