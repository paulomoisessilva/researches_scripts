"""Carregamento e validação de configurações do projeto."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
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


def resolve_strategies(
    cfg: dict[str, Any],
    cli_strategies: list[str] | None = None,
) -> list[str]:
    """Resolve a lista de estratégias: CLI > config.strategies > todas."""
    from .strategies import STRATEGY_REGISTRY

    all_strategies = list(STRATEGY_REGISTRY.keys())

    if cli_strategies:
        names = cli_strategies
    else:
        configured = cfg.get("strategies")
        if configured is None:
            return all_strategies
        if isinstance(configured, list):
            names = configured
        elif isinstance(configured, dict):
            active = configured.get("active")
            if not active:
                return all_strategies
            names = active
        else:
            raise ValueError(
                "config.strategies deve ser uma lista de nomes ou um dict com chave 'active'."
            )

    unknown = sorted(set(names) - set(all_strategies))
    if unknown:
        raise ValueError(
            f"Estratégias desconhecidas: {unknown}. Disponíveis: {all_strategies}"
        )

    return names


def resolve_date_range(
    cfg: dict[str, Any],
    cli_start: str | None = None,
    cli_end: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve período de análise: CLI > config.data.start_date/end_date."""
    data = cfg.get("data", {})
    start = cli_start if cli_start is not None else data.get("start_date")
    end = cli_end if cli_end is not None else data.get("end_date")
    return start, end


def filter_dataframe_by_date_range(
    df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, bool]:
    """Filtra OHLCV/features por período. Retorna (df, reduziu_linhas)."""
    if not start_date and not end_date:
        return df, False
    if timestamp_col not in df.columns:
        return df, False

    n_before = len(df)
    ts = pd.to_datetime(df[timestamp_col])
    tz = ts.dt.tz
    mask = pd.Series(True, index=df.index)
    if start_date:
        mask &= ts >= pd.Timestamp(start_date, tz=tz)
    if end_date:
        mask &= ts <= pd.Timestamp(end_date, tz=tz)
    filtered = df.loc[mask].reset_index(drop=True)
    return filtered, len(filtered) < n_before


def resolve_cost_bps(
    cfg: dict[str, Any],
    cli_fee: float | None = None,
    cli_slippage: float | None = None,
) -> tuple[float, float]:
    """Resolve custo por lado: CLI > primeiro valor de config.backtest."""
    bt = cfg.get("backtest", {})
    default_fee = float(bt.get("fee_bps_per_side", [2.0])[0])
    default_slip = float(bt.get("slippage_bps_per_side", [2.0])[0])
    fee = default_fee if cli_fee is None else cli_fee
    slip = default_slip if cli_slippage is None else cli_slippage
    return fee, slip
