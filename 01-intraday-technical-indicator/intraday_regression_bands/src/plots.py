"""Geração de gráficos para análise dos resultados de backtest.

Usa matplotlib exclusivamente (sem seaborn).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .utils import ensure_dir, setup_logger

matplotlib.use("Agg")  # backend sem display

logger = setup_logger(__name__)

_FIGSIZE = (14, 7)
_DPI = 150
_STYLE = {
    "axes.facecolor": "#f8f9fa",
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _apply_style(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.35, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# 1. Curvas de equity
# ---------------------------------------------------------------------------


def plot_equity_curves(
    equity_dict: dict[str, pd.Series],
    figures_dir: str | Path,
    title: str = "Equity Curves",
) -> Path:
    """Plota curvas de equity normalizadas para múltiplas estratégias."""
    figures_dir = Path(figures_dir)
    ensure_dir(figures_dir)

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)

    colors = plt.cm.tab10.colors
    for i, (name, equity) in enumerate(equity_dict.items()):
        normalized = equity / equity.iloc[0]
        ax.plot(equity.index, normalized, label=name, color=colors[i % len(colors)], linewidth=1.2)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Retorno Acumulado (base=1)")
    ax.set_xlabel("Data")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    _apply_style(ax)
    fig.tight_layout()

    path = figures_dir / "equity_curves.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    logger.info("Salvo: %s", path.name)
    return path


# ---------------------------------------------------------------------------
# 2. Drawdowns
# ---------------------------------------------------------------------------


def plot_drawdowns(
    equity_dict: dict[str, pd.Series],
    figures_dir: str | Path,
) -> Path:
    figures_dir = Path(figures_dir)
    ensure_dir(figures_dir)

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    colors = plt.cm.tab10.colors

    for i, (name, equity) in enumerate(equity_dict.items()):
        dd = (equity - equity.cummax()) / equity.cummax() * 100
        ax.fill_between(equity.index, dd, 0, alpha=0.25, color=colors[i % len(colors)], label=name)
        ax.plot(equity.index, dd, color=colors[i % len(colors)], linewidth=0.8)

    ax.set_title("Drawdowns por Estratégia (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Data")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    _apply_style(ax)
    fig.tight_layout()

    path = figures_dir / "drawdowns.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    logger.info("Salvo: %s", path.name)
    return path


# ---------------------------------------------------------------------------
# 3. Distribuição de retornos
# ---------------------------------------------------------------------------


def plot_returns_distribution(
    returns_dict: dict[str, pd.Series],
    figures_dir: str | Path,
    bins: int = 100,
) -> Path:
    figures_dir = Path(figures_dir)
    ensure_dir(figures_dir)

    n = len(returns_dict)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=_DPI)
    axes = np.array(axes).flatten() if n > 1 else [axes]

    colors = plt.cm.tab10.colors
    for i, (name, ret) in enumerate(returns_dict.items()):
        ax = axes[i]
        clean = ret.dropna()
        ax.hist(clean, bins=bins, color=colors[i % len(colors)], alpha=0.75, edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(clean.mean(), color="red", linewidth=1.2, linestyle="-", label=f"Média: {clean.mean():.5f}")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Retorno Log")
        ax.set_ylabel("Frequência")
        ax.legend(fontsize=7)
        _apply_style(ax)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribuição de Retornos por Estratégia", fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = figures_dir / "returns_distribution.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    logger.info("Salvo: %s", path.name)
    return path


# ---------------------------------------------------------------------------
# 4. Heatmap de performance por custo
# ---------------------------------------------------------------------------


def plot_performance_heatmap_by_cost(
    df_metrics: pd.DataFrame,
    strategy: str,
    metric: str = "sharpe",
    figures_dir: str | Path = "results/figures",
) -> Path:
    figures_dir = Path(figures_dir)
    ensure_dir(figures_dir)

    sub = df_metrics[df_metrics["strategy"] == strategy].copy()
    if sub.empty:
        logger.warning("Sem dados para heatmap: %s", strategy)
        return figures_dir / "performance_heatmap_by_cost.png"

    pivot = sub.pivot_table(
        index="fee_bps", columns="slippage_bps", values=metric, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(8, 6), dpi=_DPI)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
    fig.colorbar(im, ax=ax, label=metric)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c} bps" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r} bps" for r in pivot.index])
    ax.set_xlabel("Slippage por lado")
    ax.set_ylabel("Fee por lado")
    ax.set_title(f"Heatmap {metric} — {strategy}", fontsize=13, fontweight="bold")

    for ii in range(len(pivot.index)):
        for jj in range(len(pivot.columns)):
            val = pivot.values[ii, jj]
            if not np.isnan(val):
                ax.text(jj, ii, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    path = figures_dir / "performance_heatmap_by_cost.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    logger.info("Salvo: %s", path.name)
    return path


# ---------------------------------------------------------------------------
# 5. Break-even costs
# ---------------------------------------------------------------------------


def plot_break_even_costs(
    df_metrics: pd.DataFrame,
    figures_dir: str | Path,
) -> Path:
    figures_dir = Path(figures_dir)
    ensure_dir(figures_dir)

    sub = df_metrics[df_metrics["fee_bps"] == 0].copy()
    if sub.empty or "break_even_cost_bps" not in sub.columns:
        logger.warning("Sem dados para break-even costs.")
        return figures_dir / "break_even_costs.png"

    sub = sub.dropna(subset=["break_even_cost_bps"])
    sub = sub.sort_values("break_even_cost_bps", ascending=False)

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    colors = ["#2ecc71" if v > 5 else "#e74c3c" for v in sub["break_even_cost_bps"]]
    bars = ax.barh(sub["strategy"] + " | " + sub["symbol"] + " " + sub["interval"],
                   sub["break_even_cost_bps"], color=colors, alpha=0.8)
    ax.axvline(2.0, color="orange", linewidth=1.5, linestyle="--", label="Custo típico (2 bps)")
    ax.axvline(5.0, color="red", linewidth=1.5, linestyle="--", label="Custo alto (5 bps)")
    ax.set_xlabel("Break-even cost (bps por lado)")
    ax.set_title("Custo de Break-even por Estratégia", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    _apply_style(ax)
    fig.tight_layout()

    path = figures_dir / "break_even_costs.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    logger.info("Salvo: %s", path.name)
    return path


# ---------------------------------------------------------------------------
# 6. Trades ao longo do tempo
# ---------------------------------------------------------------------------


def plot_trades_over_time(
    equity_dict: dict[str, pd.Series],
    positions_dict: dict[str, pd.Series],
    figures_dir: str | Path,
) -> Path:
    figures_dir = Path(figures_dir)
    ensure_dir(figures_dir)

    fig, axes = plt.subplots(2, 1, figsize=_FIGSIZE, dpi=_DPI, sharex=True)
    colors = plt.cm.tab10.colors

    for i, (name, equity) in enumerate(equity_dict.items()):
        normalized = equity / equity.iloc[0]
        axes[0].plot(equity.index, normalized, label=name, color=colors[i % len(colors)], linewidth=1.0)

    axes[0].set_ylabel("Equity (normalizada)")
    axes[0].set_title("Equity e Posições ao Longo do Tempo", fontsize=13, fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=7, ncol=3)
    _apply_style(axes[0])

    for i, (name, pos) in enumerate(positions_dict.items()):
        trade_bars = pos[pos.diff().abs() > 0]
        axes[1].scatter(trade_bars.index, trade_bars.values + i * 0.05,
                        s=5, alpha=0.4, color=colors[i % len(colors)], label=name)

    axes[1].set_ylabel("Posição")
    axes[1].set_xlabel("Data")
    axes[1].set_yticks([-1, 0, 1])
    _apply_style(axes[1])

    fig.tight_layout()
    path = figures_dir / "trades_over_time.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    logger.info("Salvo: %s", path.name)
    return path


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------


def generate_all_plots(
    df_metrics: pd.DataFrame,
    equity_dict: dict[str, pd.Series],
    returns_dict: dict[str, pd.Series],
    positions_dict: dict[str, pd.Series],
    figures_dir: str | Path,
) -> list[Path]:
    """Gera todos os gráficos obrigatórios."""
    paths: list[Path] = []

    paths.append(plot_equity_curves(equity_dict, figures_dir))
    paths.append(plot_drawdowns(equity_dict, figures_dir))
    paths.append(plot_returns_distribution(returns_dict, figures_dir))

    if not df_metrics.empty and "strategy" in df_metrics.columns:
        top_strategy = df_metrics.groupby("strategy")["sharpe"].mean().idxmax()
        paths.append(plot_performance_heatmap_by_cost(df_metrics, top_strategy, figures_dir=figures_dir))

    paths.append(plot_break_even_costs(df_metrics, figures_dir))
    paths.append(plot_trades_over_time(equity_dict, positions_dict, figures_dir))

    return paths
