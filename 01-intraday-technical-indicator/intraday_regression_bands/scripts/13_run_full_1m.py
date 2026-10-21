#!/usr/bin/env python3
"""Script 13 — Estudo de 1 minuto em blocos semestrais (BTC, ETH, SOL, 2021-2025).

Um ano de 1m tem ~525 mil barras e o DataFrame de features passa de 1 GB por processo, então o
período é dividido em semestres. Cada bloco usa uma cópia do config com ``start_date``,
``end_date`` e ``oos_start`` próprios e roda ``05_run_walk_forward.py --study`` por ativo.

* Cada bloco começa ``--history-days`` antes do primeiro dia de teste (o mínimo exigido é
  ``train_days + max(validation_days)``), então a escolha de parâmetros de cada dia vê o mesmo
  histórico de uma run contínua.
* O ``end_date`` de um bloco é o primeiro dia de teste do próximo: uma janela de teste só existe
  se o dia inteiro estiver nos dados, então assim nenhum dia fica sem teste entre blocos. No
  último bloco o fim dos dados é 2025-12-31 e o último dia de teste é 2025-12-30 (como no 15m).
* O progresso fica em ``results/full_1m/progress.json``; rodar de novo pula o que já terminou.

Uso:
    python scripts/13_run_full_1m.py --config config.full_1m.yaml --workers 8
    python scripts/13_run_full_1m.py --config config.full_1m.yaml --symbols BTCUSDT --chunks 2021H1
    python scripts/13_run_full_1m.py --config config.full_1m.yaml --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.config import get_project_root, load_config
from src.full_study import make_chunks
from src.utils import ensure_dir, setup_logger
from src.wf_runner import resolve_runs_root, resolve_train_days, resolve_validation_days

logger = setup_logger("13_run_full_1m")


def chunk_config(base: dict, chunk: dict) -> dict:
    cfg = copy.deepcopy(base)
    cfg["data"]["start_date"] = chunk["start_date"]
    cfg["data"]["end_date"] = chunk["end_date"]
    cfg["walk_forward"]["oos_start"] = chunk["oos_start"]
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estudo de 1m em blocos semestrais.")
    parser.add_argument("--config", type=str, default="config.full_1m.yaml")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--chunks", nargs="+", default=None, help="Rótulos, ex.: 2021H1 2021H2.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--history-days", type=int, default=7)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = get_project_root()
    base = load_config(args.config)
    symbols = args.symbols or base["data"]["symbols"]

    need = resolve_train_days(base)[0] + max(resolve_validation_days(base, None, True))
    if args.history_days < need:
        raise SystemExit(f"--history-days {args.history_days} < train_days + max(validation_days) = {need}.")

    chunks = make_chunks(
        date.fromisoformat(base["data"]["start_date"]),
        date.fromisoformat(base["data"]["end_date"]),
        date.fromisoformat(base["walk_forward"]["oos_start"]),
        args.history_days,
    )
    if args.chunks:
        chunks = [c for c in chunks if c["label"] in args.chunks]

    out_root = resolve_runs_root(base).parent
    cfg_dir = ensure_dir(out_root / "chunk_configs")
    progress_path = out_root / "progress.json"
    progress: dict = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}

    logger.info("%d blocos x %d ativos | workers %d", len(chunks), len(symbols), args.workers)
    for c in chunks:
        logger.info("  %s: dados %s -> %s | 1º teste %s", c["label"], c["start_date"], c["end_date"], c["oos_start"])
    if args.dry_run:
        return

    failures: list[str] = []
    t_all = time.time()
    for chunk in chunks:
        cfg_path = cfg_dir / f"chunk_{chunk['label']}.yaml"
        cfg_path.write_text(yaml.safe_dump(chunk_config(base, chunk), sort_keys=False, allow_unicode=True), encoding="utf-8")
        for symbol in symbols:
            key = f"{symbol}|{chunk['label']}"
            status = progress.get(key, {}).get("status")
            if status == "done" or (status == "failed" and not args.retry_failed):
                logger.info("%s: %s, pulando.", key, status)
                continue
            logger.info("%s: iniciando.", key)
            t0 = time.time()
            cmd = [
                sys.executable, str(root / "scripts" / "05_run_walk_forward.py"),
                "--config", str(cfg_path), "--symbols", symbol, "--study", "--workers", str(args.workers),
            ]
            proc = subprocess.run(cmd, cwd=root, check=False)
            elapsed = time.time() - t0
            progress[key] = {
                "status": "done" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
                "elapsed_s": round(elapsed, 1),
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "workers": args.workers,
            }
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            if proc.returncode != 0:
                failures.append(key)
                logger.error("%s: falhou (código %d).", key, proc.returncode)
            else:
                logger.info("%s: ok em %.1f min.", key, elapsed / 60)

    logger.info("Fim em %.1f min. Falhas: %s", (time.time() - t_all) / 60, failures or "nenhuma")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
