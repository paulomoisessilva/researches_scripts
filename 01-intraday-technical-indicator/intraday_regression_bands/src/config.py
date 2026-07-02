"""Carregamento e validação de configurações do projeto."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Carrega o arquivo YAML de configuração.

    Parameters
    ----------
    path:
        Caminho para o arquivo config.yaml. Se None, usa o padrão do projeto.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    required_sections = ["project", "data", "backtest", "features", "walk_forward"]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Seção obrigatória ausente no config.yaml: '{section}'")

    data = cfg["data"]
    if not data.get("symbols"):
        raise ValueError("config.data.symbols não pode ser vazio.")
    if not data.get("intervals"):
        raise ValueError("config.data.intervals não pode ser vazio.")
    if not data.get("start_date") or not data.get("end_date"):
        raise ValueError("config.data.start_date e end_date são obrigatórios.")


def resolve_path(cfg: dict[str, Any], relative_path: str) -> Path:
    """Resolve um caminho relativo à raiz do projeto."""
    project_root = Path(__file__).parent.parent
    return project_root / relative_path


def get_project_root() -> Path:
    return Path(__file__).parent.parent
