"""Geração automática de relatórios em Markdown e HTML."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #222; }}
    h1 {{ color: #1a1a2e; border-bottom: 3px solid #0f3460; padding-bottom: 10px; }}
    h2 {{ color: #0f3460; margin-top: 40px; }}
    h3 {{ color: #16213e; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 13px; }}
    th {{ background: #0f3460; color: white; padding: 10px 12px; text-align: left; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #ddd; }}
    tr:nth-child(even) {{ background: #f5f7fa; }}
    tr:hover {{ background: #e8eaf6; }}
    .metric-good {{ color: #27ae60; font-weight: bold; }}
    .metric-bad {{ color: #c0392b; font-weight: bold; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }}
    .info-box {{ background: #e8f4f8; border-left: 4px solid #0f3460; padding: 12px 16px; margin: 16px 0; border-radius: 2px; }}
    footer {{ margin-top: 60px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
  </style>
</head>
<body>
{body}
<footer>
  <p>Gerado automaticamente em {generated_at} | intraday_regression_bands</p>
</footer>
</body>
</html>
"""


def _df_to_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_Sem dados._\n"
    df_show = df.head(max_rows).copy()
    for col in df_show.select_dtypes(include="float").columns:
        df_show[col] = df_show[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    header = "| " + " | ".join(str(c) for c in df_show.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df_show.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df_show.itertuples(index=False)
    ]
    return "\n".join([header, sep] + rows) + "\n"


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "<p><em>Sem dados.</em></p>"
    df_show = df.head(max_rows).copy()
    for col in df_show.select_dtypes(include="float").columns:
        df_show[col] = df_show[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    return df_show.to_html(index=False, border=0, classes="")


def _img_link_md(path: str | Path, caption: str = "") -> str:
    p = Path(path)
    return f"![{caption}]({p})\n"


def _img_tag_html(path: str | Path, caption: str = "") -> str:
    return f'<figure><img src="{path}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def generate_markdown_report(
    cfg: dict,
    df_metrics: pd.DataFrame,
    figures_dir: str | Path,
    tables_dir: str | Path,
    reports_dir: str | Path,
) -> Path:
    """Gera relatório em Markdown."""
    figures_dir = Path(figures_dir)
    tables_dir = Path(tables_dir)
    reports_dir = Path(reports_dir)
    ensure_dir(reports_dir)

    data_cfg = cfg["data"]
    bt_cfg = cfg["backtest"]
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []

    # Cabeçalho
    lines += [
        "# Relatório de Backtesting Intradiário",
        "",
        f"**Gerado em:** {now_str}",
        "",
        "---",
        "",
        "## 1. Configuração do Estudo",
        "",
        f"- **Período analisado:** {data_cfg['start_date']} → {data_cfg['end_date']}",
        f"- **Ativos:** {', '.join(data_cfg['symbols'])}",
        f"- **Frequências:** {', '.join(data_cfg['intervals'])}",
        f"- **Capital inicial:** USD {bt_cfg['initial_capital']:,.0f}",
        f"- **Custos testados (fee, bps/lado):** {bt_cfg['fee_bps_per_side']}",
        f"- **Slippage testado (bps/lado):** {bt_cfg['slippage_bps_per_side']}",
        f"- **Execução:** {bt_cfg['execution']}",
        "",
        "## 2. Estratégias Testadas",
        "",
        "| Estratégia | Descrição |",
        "|---|---|",
        "| bollinger | Reversão à média com Bandas de Bollinger clássicas |",
        "| regression_bands | Bandas ajustadas por regressão OLS rolling sobre log-preço |",
        "| return_residuals | Bandas sobre resíduos de regressão de retornos |",
        "| rsi | RSI com limiares de sobrecompra/sobrevenda |",
        "| stochastic | Estocástico %K com limiares |",
        "| technical_confluence | Score de confluência técnica composto (RSI + Estoc. + Bollinger) |",
        "| random | Placebo multi-seed: trading direcional aleatório (taxa fixa por candle) |",
        "",
    ]

    # Melhores por Sharpe
    if not df_metrics.empty and "sharpe" in df_metrics.columns:
        lines += ["## 3. Melhores Estratégias por Sharpe Líquido", ""]
        best_sharpe = (
            df_metrics.sort_values("sharpe", ascending=False)
            .drop_duplicates(subset=["strategy", "symbol", "interval"])
            .head(10)[["strategy", "symbol", "interval", "fee_bps", "slippage_bps",
                       "sharpe", "annualized_return", "max_drawdown", "number_of_trades"]]
        )
        lines.append(_df_to_markdown_table(best_sharpe))

    # Melhores por drawdown
    if not df_metrics.empty and "max_drawdown" in df_metrics.columns:
        lines += ["## 4. Melhores Estratégias por Drawdown Máximo", ""]
        best_dd = (
            df_metrics.sort_values("max_drawdown", ascending=False)
            .drop_duplicates(subset=["strategy", "symbol", "interval"])
            .head(10)[["strategy", "symbol", "interval", "fee_bps",
                       "max_drawdown", "sharpe", "calmar"]]
        )
        lines.append(_df_to_markdown_table(best_dd))

    # Comparação com random
    if not df_metrics.empty and "strategy" in df_metrics.columns:
        lines += ["## 5. Comparação com Placebo Aleatório Multi-Seed", ""]
        random_metrics = df_metrics[df_metrics["strategy"] == "random"]
        other_metrics = df_metrics[df_metrics["strategy"] != "random"]

        if not random_metrics.empty:
            rand_sharpe = random_metrics["sharpe"].mean()
            lines.append(f"- **Sharpe médio da estratégia aleatória:** {rand_sharpe:.4f}")

        if not other_metrics.empty:
            best_vs_random = (
                other_metrics.groupby("strategy")["sharpe"]
                .mean()
                .reset_index()
                .sort_values("sharpe", ascending=False)
            )
            lines.append("")
            lines.append(_df_to_markdown_table(best_vs_random))

    # Métricas completas
    if not df_metrics.empty:
        lines += ["## 6. Tabela Completa de Métricas", ""]
        lines.append(_df_to_markdown_table(df_metrics, max_rows=50))

    # Gráficos
    lines += ["## 7. Gráficos", ""]
    figure_files = [
        ("equity_curves.png", "Curvas de Equity Normalizadas"),
        ("drawdowns.png", "Drawdowns por Estratégia"),
        ("returns_distribution.png", "Distribuição de Retornos"),
        ("performance_heatmap_by_cost.png", "Heatmap de Sharpe por Custo"),
        ("break_even_costs.png", "Custo de Break-even"),
        ("trades_over_time.png", "Trades ao Longo do Tempo"),
    ]
    for fname, caption in figure_files:
        fpath = figures_dir / fname
        if fpath.exists():
            lines.append(_img_link_md(fpath, caption))

    # Links para tabelas
    lines += ["## 8. Arquivos de Resultados", ""]
    table_files = [
        "performance_summary.csv",
        "performance_by_symbol_interval.csv",
        "walk_forward_results.csv",
        "walk_forward_oos_summary.csv",
        "best_parameters_by_window.csv",
    ]
    for tname in table_files:
        tp = tables_dir / tname
        if tp.exists():
            lines.append(f"- [{tname}]({tp})")

    content = "\n".join(lines)
    md_path = reports_dir / "summary_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Relatório Markdown salvo: %s", md_path)
    return md_path


def generate_html_report(
    cfg: dict,
    df_metrics: pd.DataFrame,
    figures_dir: str | Path,
    tables_dir: str | Path,
    reports_dir: str | Path,
) -> Path:
    """Gera relatório em HTML."""
    figures_dir = Path(figures_dir)
    tables_dir = Path(tables_dir)
    reports_dir = Path(reports_dir)
    ensure_dir(reports_dir)

    data_cfg = cfg["data"]
    bt_cfg = cfg["backtest"]
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    sections: list[str] = []

    sections.append(f"<h1>Relatório de Backtesting Intradiário</h1>")
    sections.append(
        f'<div class="info-box">'
        f"<strong>Período:</strong> {data_cfg['start_date']} → {data_cfg['end_date']}<br>"
        f"<strong>Ativos:</strong> {', '.join(data_cfg['symbols'])}<br>"
        f"<strong>Frequências:</strong> {', '.join(data_cfg['intervals'])}<br>"
        f"<strong>Capital inicial:</strong> USD {bt_cfg['initial_capital']:,.0f}"
        f"</div>"
    )

    if not df_metrics.empty and "sharpe" in df_metrics.columns:
        sections.append("<h2>Melhores Estratégias por Sharpe</h2>")
        best = (
            df_metrics.sort_values("sharpe", ascending=False)
            .drop_duplicates(subset=["strategy", "symbol", "interval"])
            .head(10)
        )
        sections.append(_df_to_html_table(best))

    if not df_metrics.empty and "max_drawdown" in df_metrics.columns:
        sections.append("<h2>Melhores Estratégias por Drawdown</h2>")
        best_dd = df_metrics.sort_values("max_drawdown", ascending=False).head(10)
        sections.append(_df_to_html_table(best_dd))

    sections.append("<h2>Gráficos</h2>")
    figure_files = [
        ("equity_curves.png", "Curvas de Equity"),
        ("drawdowns.png", "Drawdowns"),
        ("returns_distribution.png", "Distribuição de Retornos"),
        ("performance_heatmap_by_cost.png", "Heatmap por Custo"),
        ("break_even_costs.png", "Break-even Costs"),
        ("trades_over_time.png", "Trades ao Longo do Tempo"),
    ]
    for fname, caption in figure_files:
        fpath = figures_dir / fname
        if fpath.exists():
            sections.append(_img_tag_html(fpath, caption))

    body = "\n".join(sections)
    html_content = _HTML_TEMPLATE.format(
        title="Backtest Report — Intraday Regression Bands",
        body=body,
        generated_at=now_str,
    )

    html_path = reports_dir / "summary_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info("Relatório HTML salvo: %s", html_path)
    return html_path


def generate_all_reports(
    cfg: dict,
    df_metrics: pd.DataFrame,
    figures_dir: str | Path,
    tables_dir: str | Path,
    reports_dir: str | Path,
) -> tuple[Path, Path]:
    """Gera relatório em MD e HTML."""
    md = generate_markdown_report(cfg, df_metrics, figures_dir, tables_dir, reports_dir)
    html = generate_html_report(cfg, df_metrics, figures_dir, tables_dir, reports_dir)
    return md, html
