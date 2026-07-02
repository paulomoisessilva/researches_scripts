#!/usr/bin/env python3
"""Script 06 — Geração de gráficos, tabelas e relatório final.

Lê métricas de results/tables/ e equity curves de data/backtests/,
gera todos os gráficos em results/figures/ e o relatório em results/reports/.

Uso:
    python scripts/06_generate_results.py
    python scripts/06_generate_results.py --top-fee 0 --top-slip 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import load_config, get_project_root
from src.plots import generate_all_plots
from src.report import generate_all_reports
from src.utils import ensure_dir, setup_logger

logger = setup_logger("06_generate_results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera gráficos, tabelas e relatório final.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--top-fee", type=float, default=2.0,
                        help="Fee (bps) para selecionar curvas de equity para gráficos")
    parser.add_argument("--top-slip", type=float, default=2.0,
                        help="Slippage (bps) para selecionar curvas de equity para gráficos")
    return parser.parse_args()


def _load_equity_curves(
    backtests_dir: Path,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[dict, dict, dict]:
    """Carrega equity curves dos arquivos Parquet de backtest."""
    equity_dict: dict = {}
    returns_dict: dict = {}
    positions_dict: dict = {}

    fee_tag = f"fee{int(fee_bps)}_slip{int(slippage_bps)}"
    pattern = f"*_{fee_tag}.parquet"

    for p in backtests_dir.glob(pattern):
        if not p.name.startswith("equity_"):
            continue
        try:
            df = pd.read_parquet(p)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            # Extrair nome da estratégia do filename: equity_{strategy}_{symbol}_{interval}_{fee_tag}
            parts = p.stem.replace(f"_{fee_tag}", "").split("_")
            label = "_".join(parts[1:])  # remove prefixo "equity_"

            if "equity" in df.columns:
                equity_dict[label] = df["equity"]
            if "strategy_return" in df.columns:
                returns_dict[label] = df["strategy_return"]
            if "position" in df.columns:
                positions_dict[label] = df["position"]
        except Exception as exc:
            logger.warning("Erro ao carregar %s: %s", p.name, exc)

    return equity_dict, returns_dict, positions_dict


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = get_project_root()

    backtests_dir = root / "data" / "backtests"
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    reports_dir = root / "results" / "reports"

    for d in [figures_dir, reports_dir]:
        ensure_dir(d)

    logger.info("=" * 60)
    logger.info("GERAÇÃO DE RESULTADOS")
    logger.info("=" * 60)

    # Carregar métricas
    metrics_path = tables_dir / "performance_summary.csv"
    if metrics_path.exists():
        df_metrics = pd.read_csv(metrics_path)
        logger.info("Métricas carregadas: %d entradas.", len(df_metrics))
    else:
        logger.warning("performance_summary.csv não encontrado — rode script 04 primeiro.")
        df_metrics = pd.DataFrame()

    # Carregar equity curves
    equity_dict, returns_dict, positions_dict = _load_equity_curves(
        backtests_dir, args.top_fee, args.top_slip
    )
    logger.info("Equity curves carregadas: %d estratégias.", len(equity_dict))

    if not equity_dict:
        logger.warning(
            "Nenhuma equity curve encontrada para fee=%s slip=%s bps. "
            "Verifique se o script 04 foi executado.",
            args.top_fee, args.top_slip,
        )

    # Gráficos
    logger.info("Gerando gráficos...")
    figure_paths = generate_all_plots(
        df_metrics=df_metrics,
        equity_dict=equity_dict,
        returns_dict=returns_dict,
        positions_dict=positions_dict,
        figures_dir=figures_dir,
    )
    logger.info("%d gráficos gerados.", len(figure_paths))

    # Relatório
    logger.info("Gerando relatório...")
    md_path, html_path = generate_all_reports(
        cfg=cfg,
        df_metrics=df_metrics,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        reports_dir=reports_dir,
    )
    logger.info("Relatório Markdown: %s", md_path)
    logger.info("Relatório HTML: %s", html_path)
    logger.info("=" * 60)
    logger.info("Geração de resultados concluída.")


if __name__ == "__main__":
    main()
