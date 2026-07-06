"""Cálculo de métricas de performance para backtests."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import ensure_dir, setup_logger, bars_per_day

logger = setup_logger(__name__)

_TRADING_DAYS_PER_YEAR = 252


def _annualization_factor(interval: str) -> float:
    bpd = bars_per_day(interval)
    return bpd * _TRADING_DAYS_PER_YEAR


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def annualized_return(returns: pd.Series, interval: str) -> float:
    factor = _annualization_factor(interval)
    mean_ret = returns.mean()
    return float(mean_ret * factor)


def annualized_volatility(returns: pd.Series, interval: str) -> float:
    factor = _annualization_factor(interval)
    return float(returns.std(ddof=1) * np.sqrt(factor))


def sharpe_ratio(returns: pd.Series, interval: str, risk_free: float = 0.0) -> float:
    ann_ret = annualized_return(returns, interval) - risk_free
    ann_vol = annualized_volatility(returns, interval)
    if ann_vol == 0:
        return 0.0
    return float(ann_ret / ann_vol)


def sortino_ratio(returns: pd.Series, interval: str, risk_free: float = 0.0) -> float:
    factor = _annualization_factor(interval)
    ann_ret = annualized_return(returns, interval) - risk_free
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.inf
    downside_vol = float(downside.std(ddof=1) * np.sqrt(factor))
    if downside_vol == 0:
        return 0.0
    return float(ann_ret / downside_vol)


def max_drawdown(equity: pd.Series) -> float:
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, equity: pd.Series, interval: str) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    ann_ret = annualized_return(returns, interval)
    return float(ann_ret / mdd)


def hit_rate(trades: pd.DataFrame) -> float:
    """Proporção de trades com retorno positivo."""
    if trades.empty or "trade_return" not in trades.columns:
        return np.nan
    winners = (trades["trade_return"] > 0).sum()
    return float(winners / len(trades))


def profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty or "trade_return" not in trades.columns:
        return np.nan
    gains = trades.loc[trades["trade_return"] > 0, "trade_return"].sum()
    losses = abs(trades.loc[trades["trade_return"] < 0, "trade_return"].sum())
    if losses == 0:
        return np.inf
    return float(gains / losses)


def _compute_trade_returns(trades: pd.DataFrame) -> pd.DataFrame:
    """Calcula retorno de cada trade com base em preços de entrada/saída."""
    if trades.empty:
        return trades
    df = trades.copy()
    if "entry_price" in df.columns and "exit_price" in df.columns:
        valid = df["entry_price"].notna() & df["exit_price"].notna() & (df["entry_price"] != 0)
        df.loc[valid, "trade_return"] = (
            df.loc[valid, "side"].map({"long": 1, "short": -1}) *
            np.log(df.loc[valid, "exit_price"] / df.loc[valid, "entry_price"])
        )
    return df


def exposure_time(positions: pd.Series) -> float:
    """Proporção de candles com posição não-nula."""
    if len(positions) == 0:
        return 0.0
    return float((positions != 0).sum() / len(positions))


def average_holding_time(trades: pd.DataFrame) -> float:
    if trades.empty or "holding_bars" not in trades.columns:
        return np.nan
    return float(trades["holding_bars"].mean())


def turnover(positions: pd.Series) -> float:
    """Soma das mudanças absolutas de posição / número de candles."""
    changes = positions.diff().abs().fillna(0)
    return float(changes.sum() / max(len(positions), 1))


def _position_change_units(positions: pd.Series) -> float:
    """Soma de |Δposição| — unidades sobre as quais o backtest cobra custo."""
    return float(positions.diff().abs().fillna(0).sum())


SELECTION_METRICS = frozenset({"net_sharpe", "net_total_return", "net_profit_factor"})


def selection_metric_score(
    returns: pd.Series,
    trades: pd.DataFrame,
    interval: str,
    metric: str,
    *,
    index_mask: pd.Series | None = None,
) -> float:
    """Calcula métrica de seleção walk-forward no subperíodo indicado.

    Métricas suportadas:
      - net_sharpe: Sharpe anualizado dos retornos líquidos
      - net_total_return: retorno total composto no período
      - net_profit_factor: soma de ganhos / soma de perdas (retornos por barra)
    """
    if metric not in SELECTION_METRICS:
        raise ValueError(
            f"Métrica de seleção desconhecida: '{metric}'. "
            f"Suportadas: {sorted(SELECTION_METRICS)}"
        )

    if index_mask is not None:
        period_returns = returns.loc[index_mask]
    else:
        period_returns = returns

    if len(period_returns) < 2:
        return -np.inf

    if metric == "net_sharpe":
        return sharpe_ratio(period_returns, interval)

    if metric == "net_total_return":
        return float(np.exp(period_returns.fillna(0).sum()) - 1.0)

    # net_profit_factor — baseado em retornos líquidos por barra
    gains = period_returns[period_returns > 0].sum()
    losses = abs(period_returns[period_returns < 0].sum())
    if losses == 0:
        return np.inf if gains > 0 else -np.inf
    return float(gains / losses)


def break_even_total_return_bps(
    gross_returns: pd.Series,
    positions: pd.Series,
) -> float:
    """Custo combinado (fee + slippage) por lado que zera o lucro acumulado.

    Usa retorno bruto (sem custos) e turnover total:
        custo_bar = |Δposição| × (fee_bps + slippage_bps) / 10_000

    Retorna o valor de (fee_bps + slippage_bps) por lado necessário para
    exp(sum(retornos_brutos)) - 1 = 0.
    """
    gross_log = float(gross_returns.fillna(0).sum())
    if gross_log <= 0:
        return 0.0

    change_units = _position_change_units(positions)
    if change_units == 0:
        return np.nan  # convenção: sem turnover, BETC indefinido

    return float(gross_log * 10_000 / change_units)


def break_even_cost_bps(
    gross_returns: pd.Series,
    positions: pd.Series,
    interval: str | None = None,
) -> float:
    """Alias de break_even_total_return_bps (CBE por turnover, não anualizado)."""
    _ = interval  # mantido por compatibilidade de assinatura
    return break_even_total_return_bps(gross_returns, positions)


def compute_metrics(
    equity: pd.Series,
    returns: pd.Series,
    positions: pd.Series,
    trades: pd.DataFrame,
    interval: str,
    strategy_name: str = "",
    symbol: str = "",
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    gross_returns: pd.Series | None = None,
) -> dict:
    """Calcula todas as métricas de performance de um backtest."""
    trades_with_ret = _compute_trade_returns(trades)
    gross = returns if gross_returns is None else gross_returns

    metrics = {
        "strategy": strategy_name,
        "symbol": symbol,
        "interval": interval,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "total_return": total_return(equity),
        "annualized_return": annualized_return(returns, interval),
        "annualized_volatility": annualized_volatility(returns, interval),
        "sharpe": sharpe_ratio(returns, interval),
        "sortino": sortino_ratio(returns, interval),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(returns, equity, interval),
        "hit_rate": hit_rate(trades_with_ret),
        "profit_factor": profit_factor(trades_with_ret),
        "number_of_trades": len(trades_with_ret),
        "average_trade_return": (
            trades_with_ret["trade_return"].mean()
            if "trade_return" in trades_with_ret.columns and not trades_with_ret.empty
            else np.nan
        ),
        "median_trade_return": (
            trades_with_ret["trade_return"].median()
            if "trade_return" in trades_with_ret.columns and not trades_with_ret.empty
            else np.nan
        ),
        "exposure_time": exposure_time(positions),
        "average_holding_time": average_holding_time(trades_with_ret),
        "turnover": turnover(positions),
        "break_even_cost_bps": break_even_cost_bps(gross, positions, interval),
        "break_even_total_return_bps": break_even_total_return_bps(gross, positions),
    }
    return metrics


def aggregate_metrics(results_list: list[dict]) -> pd.DataFrame:
    """Agrega lista de dicionários de métricas em DataFrame."""
    return pd.DataFrame(results_list)


def save_metrics(
    df_metrics: pd.DataFrame,
    tables_dir: str | Path,
    filename: str = "performance_summary.csv",
) -> Path:
    tables_dir = Path(tables_dir)
    ensure_dir(tables_dir)
    path = tables_dir / filename
    df_metrics.to_csv(path, index=False)
    logger.info("Métricas salvas: %s", path)
    return path


def save_performance_by_symbol_interval(
    df_metrics: pd.DataFrame,
    tables_dir: str | Path,
) -> Path:
    """Exporta resumo de performance agrupado por símbolo e intervalo."""
    tables_dir = Path(tables_dir)
    ensure_dir(tables_dir)

    agg = (
        df_metrics.groupby(["strategy", "symbol", "interval"])
        .agg(
            mean_sharpe=("sharpe", "mean"),
            best_sharpe=("sharpe", "max"),
            mean_return=("annualized_return", "mean"),
            mean_drawdown=("max_drawdown", "mean"),
            mean_trades=("number_of_trades", "mean"),
        )
        .reset_index()
    )
    path = tables_dir / "performance_by_symbol_interval.csv"
    agg.to_csv(path, index=False)
    logger.info("Métricas por símbolo/intervalo salvas: %s", path)
    return path
