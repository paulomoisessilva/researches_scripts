#!/usr/bin/env python3
"""Script 10 — Snapshot versionável dos resultados pequenos de uma etapa.

Copia para ``snapshots/<nome>/`` (que vai para o git) apenas o que é pequeno e suficiente
para auditar a etapa: tabelas do paper, Reality Check, checagens de equivalência, o
``run_meta.json`` de cada run e o config usado. As runs completas (centenas de MB) ficam
em ``results/`` e podem ganhar uma cópia local com ``--backup-dir``.

O ``MANIFEST.json`` registra o commit do código, se havia alterações não commitadas, o SHA-256
dos Parquets processados e as versões das bibliotecas, para a etapa ser reproduzível.

Uso:
    python scripts/10_make_snapshot.py --name 2026-10-v1 --config config.paper.yaml
    python scripts/10_make_snapshot.py --name 2026-10-v1 --backup-dir D:/backups/intraday_v1
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.config import get_project_root, load_config
from src.utils import ensure_dir, setup_logger
from src.wf_runner import resolve_runs_root

logger = setup_logger("10_make_snapshot")

SMALL_FILE_LIMIT = 5 * 1024 * 1024  # 5 MB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cria um snapshot versionável dos resultados.")
    parser.add_argument("--name", required=True, help="Nome da pasta em snapshots/, ex.: 2026-10-v1.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--backup-dir", type=str, default=None,
                        help="Se informado, copia results/ inteiro para lá (fora do git).")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="Pastas adicionais (relativas à raiz do projeto) cujos arquivos pequenos "
                             "são copiados, recursivamente, para extra/<nome>/.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(root: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "has_uncommitted_changes": bool(run("status", "--porcelain")),
        "last_commit_message": run("log", "-1", "--pretty=%s"),
    }


def copy_small(src: Path, dst: Path, copied: list[str]) -> None:
    if not src.exists():
        return
    if src.stat().st_size > SMALL_FILE_LIMIT:
        logger.warning("Ignorado (>5 MB): %s", src)
        return
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    copied.append(str(dst))


def main() -> None:
    args = parse_args()
    root = get_project_root()
    cfg = load_config(args.config)
    results = root / "results"
    out = root / "snapshots" / args.name
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} já existe. Use --overwrite para refazer.")
        shutil.rmtree(out)
    ensure_dir(out)

    copied: list[str] = []
    paper_dir = resolve_runs_root(cfg).parent / "paper"
    for p in sorted(paper_dir.glob("*")):
        copy_small(p, out / "paper" / p.name, copied)
    for p in sorted((resolve_runs_root(cfg).parent / "engine_check").glob("*summary*.csv")):
        copy_small(p, out / "engine_check" / p.name, copied)
    for run_dir in sorted(resolve_runs_root(cfg).glob("*")):
        copy_small(run_dir / "run_meta.json", out / "runs" / run_dir.name / "run_meta.json", copied)
    cfg_path = Path(args.config) if args.config else root / "config.yaml"
    copy_small(cfg_path if cfg_path.is_absolute() else root / cfg_path, out / "config" / cfg_path.name, copied)

    for extra in args.extra:
        src_dir = root / extra
        for f in sorted(p for p in src_dir.rglob("*") if p.is_file() and p.suffix != ".parquet"):
            copy_small(f, out / "extra" / src_dir.name / f.relative_to(src_dir), copied)

    data_hashes = {
        p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted((root / cfg["data"]["processed_dir"]).glob("*.parquet"))
    }
    manifest = {
        "name": args.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": git_info(root),
        "data_processed": data_hashes,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "files": sorted(str(Path(c).relative_to(out)).replace("\\", "/") for c in copied),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "README.md").write_text(
        f"# Snapshot {args.name}\n\n"
        "Resultados pequenos e metadados de reprodutibilidade desta etapa (ver `MANIFEST.json`).\n\n"
        "- `paper/`: tabelas do paper, Reality Check/SPA.\n"
        "- `runs/*/run_meta.json`: configuração e impressão digital de cada run.\n"
        "- `engine_check/`: equivalência entre os motores de seleção.\n"
        "- `config/`: config usado.\n"
        "- `extra/`: pastas adicionais pedidas com `--extra` (arquivos pequenos, sem Parquets).\n\n"
        "As runs completas ficam em `results/` (fora do git).\n",
        encoding="utf-8",
    )
    logger.info("Snapshot em %s (%d arquivos)", out, len(copied))

    if args.backup_dir:
        dst = Path(args.backup_dir)
        ensure_dir(dst.parent)
        shutil.copytree(results, dst, dirs_exist_ok=True)
        logger.info("Cópia completa de results/ em %s", dst)


if __name__ == "__main__":
    main()
