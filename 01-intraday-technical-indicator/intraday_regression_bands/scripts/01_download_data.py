#!/usr/bin/env python3
"""Script 01 — Download de dados históricos da Binance USD-M Futures.

Uso:
    python scripts/01_download_data.py
    python scripts/01_download_data.py --symbols BTCUSDT --intervals 5m 15m
    python scripts/01_download_data.py --force
    python scripts/01_download_data.py --config config.yaml
"""

import argparse
import sys
from pathlib import Path

# Adiciona raiz do projeto ao path para importar src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data_download import download_all
from src.utils import setup_logger

logger = setup_logger("01_download_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download de dados históricos da Binance USD-M Futures."
    )
    parser.add_argument("--config", type=str, default=None, help="Caminho para config.yaml")
    parser.add_argument("--symbols", nargs="+", default=None, help="Símbolos a baixar (ex: BTCUSDT)")
    parser.add_argument("--intervals", nargs="+", default=None, help="Intervalos (ex: 1m 5m)")
    parser.add_argument("--force", action="store_true", help="Forçar redownload mesmo se arquivo existe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logger.info("=" * 60)
    logger.info("DOWNLOAD DE DADOS — Binance USD-M Futures")
    logger.info("=" * 60)
    logger.info("Símbolos: %s", args.symbols or cfg["data"]["symbols"])
    logger.info("Intervalos: %s", args.intervals or cfg["data"]["intervals"])
    logger.info("Período: %s → %s", cfg["data"]["start_date"], cfg["data"]["end_date"])
    logger.info("Force: %s", args.force)

    download_all(
        cfg=cfg,
        force=args.force,
        symbols=args.symbols,
        intervals=args.intervals,
    )

    logger.info("Download concluído.")


if __name__ == "__main__":
    main()
