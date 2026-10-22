#!/usr/bin/env python3
"""Script 20 — Base de operações do estudo v2 (uma linha por entrada).

Refaz as 126 trajetórias contínuas do walk-forward (3 ativos x 2 estratégias x 7 lookbacks x 3 frequências) a partir dos parâmetros
gravados em ``results/v2/<freq>/walk_forward`` e grava as operações em ``results/v2/analysis/trades.parquet``. O bruto refeito é
conferido contra o gravado nas runs (diferença máxima abaixo de 1e-8, senão o script falha).

Uso: python scripts/20_build_trades_v2.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logger
from src.v2_trades import rebuild_trades

logger = setup_logger("20_build_trades_v2")
ROOT = Path(__file__).parent.parent


def main() -> None:
    t0 = time.time()
    trades, worst = rebuild_trades(ROOT, ROOT / "results" / "v2")
    logger.info("%d operações; maior diferença entre o bruto refeito e o gravado: %.2e", len(trades), worst)
    if worst > 1e-8:
        raise SystemExit(f"A trajetória refeita não bate com a gravada (diferença {worst:.2e}).")
    out = ROOT / "results" / "v2" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(out / "trades.parquet", index=False)
    summary = trades.groupby(["freq", "strategy"], observed=True).size().rename("n_trades").reset_index()
    logger.info("\n%s", summary.to_string(index=False))
    logger.info("Pronto em %.0f s: %s", time.time() - t0, out / "trades.parquet")


if __name__ == "__main__":
    main()
