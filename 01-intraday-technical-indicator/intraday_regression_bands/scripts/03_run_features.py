#!/usr/bin/env python3
"""Script 03 — Cálculo de indicadores técnicos e features.

Lê dados processados e salva features em data/features/.

Uso:
    python scripts/03_run_features.py
    python scripts/03_run_features.py --symbols BTCUSDT --intervals 5m
    python scripts/03_run_features.py --force
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import load_config, get_project_root
from src.indicators import compute_all_features
from src.utils import ensure_dir, setup_logger

logger = setup_logger("03_run_features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calcula indicadores técnicos.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--intervals", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    root = get_project_root()

    processed_dir = root / data_cfg["processed_dir"]
    features_dir = root / "data" / "features"
    ensure_dir(features_dir)

    syms = args.symbols or data_cfg["symbols"]
    ints = args.intervals or data_cfg["intervals"]

    logger.info("=" * 60)
    logger.info("CÁLCULO DE INDICADORES / FEATURES")
    logger.info("=" * 60)

    total = 0
    for symbol in syms:
        for interval in ints:
            src_path = processed_dir / f"{symbol}_{interval}.parquet"
            dst_path = features_dir / f"{symbol}_{interval}_features.parquet"

            if dst_path.exists() and not args.force:
                logger.info("Já calculado: %s", dst_path.name)
                total += 1
                continue

            if not src_path.exists():
                logger.warning("Arquivo processado não encontrado: %s", src_path)
                continue

            logger.info("Calculando features: %s %s...", symbol, interval)
            df = pd.read_parquet(src_path)
            df_feat = compute_all_features(df, cfg)
            df_feat.to_parquet(dst_path, index=False, compression="snappy")

            n_features = len([c for c in df_feat.columns if c not in df.columns])
            logger.info(
                "Salvo: %s (%d linhas, %d features novas)",
                dst_path.name, len(df_feat), n_features,
            )
            total += 1

    logger.info("Features concluídas: %d arquivos.", total)


if __name__ == "__main__":
    main()
