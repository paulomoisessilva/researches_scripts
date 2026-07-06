#!/usr/bin/env python3
"""Script 02 — Preparação e limpeza dos dados brutos em Parquet.

Consolida arquivos ZIP/CSV baixados, limpa, normaliza e salva em Parquet.

Uso:
    python scripts/02_prepare_data.py
    python scripts/02_prepare_data.py --symbols BTCUSDT --intervals 1m
    python scripts/02_prepare_data.py --force
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, get_project_root
from src.data_prepare import prepare_all
from src.data_validate import validate_parquet_file, print_validation_report
from src.utils import setup_logger

logger = setup_logger("02_prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara dados brutos em Parquet limpo.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--intervals", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-validation", action="store_true", help="Pula validação pós-processamento")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    logger.info("=" * 60)
    logger.info("PREPARAÇÃO DE DADOS")
    logger.info("=" * 60)

    paths = prepare_all(
        cfg=cfg,
        force=args.force,
        symbols=args.symbols,
        intervals=args.intervals,
    )

    if not args.skip_validation:
        logger.info("Validando arquivos processados...")
        syms = args.symbols or data_cfg["symbols"]
        ints = args.intervals or data_cfg["intervals"]
        root = get_project_root()
        processed_dir = root / data_cfg["processed_dir"]

        val_results = []
        for symbol in syms:
            for interval in ints:
                p = processed_dir / f"{symbol}_{interval}.parquet"
                r = validate_parquet_file(p, symbol, interval)
                val_results.append(r)

        print_validation_report(val_results)

    logger.info("Preparação concluída. %d arquivos gerados.", len(paths))


if __name__ == "__main__":
    main()
