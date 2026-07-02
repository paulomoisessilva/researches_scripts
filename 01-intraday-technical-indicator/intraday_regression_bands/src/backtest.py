"""Motor de backtest causal para estratégias intradiárias.

Regras de execução:
  - Sinal calculado no candle t
  - Execução no open do candle t+1
  - Posição: -1 (short), 0 (neutro), +1 (long)
  - Custos e slippage aplicados por lado

Fórmula de retorno:
  strategy_return_t = position_{t-1} × log_return_t − cost_t − slippage_t
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses de configuração e resultado
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    strategy_name: str
    symbol: str
    interval: str
    initial_capital: float = 10_000.0
    fee_bps: float = 2.0           # bps por lado
    slippage_bps: float = 2.0      # bps por lado
    max_position: float = 1.0
    allow_short: bool = True
    close_eod: bool = False
    max_holding_bars: Optional[int] = None


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Motor principal
# ---------------------------------------------------------------------------


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cfg_bt: BacktestConfig,
) -> BacktestResult:
    """Executa backtest completo para uma série de sinais.

    Parameters
    ----------
    df     : DataFrame com colunas 'timestamp', 'open', 'close', 'log_return'.
    signal : Série com posição desejada no candle t (já deslocada 1 período).
    cfg_bt : Configuração do backtest.
    """
    _validate_inputs(df, signal)

    log_ret = df["log_return"].values
    pos_desired = signal.values.copy()

    # Clipar posição ao máximo permitido
    pos_desired = np.clip(pos_desired, -cfg_bt.max_position, cfg_bt.max_position)
    if not cfg_bt.allow_short:
        pos_desired = np.clip(pos_desired, 0, cfg_bt.max_position)

    n = len(df)
    position = np.zeros(n)
    strategy_ret = np.zeros(n)
    holding_bars = 0

    fee_rate = cfg_bt.fee_bps / 10_000.0
    slip_rate = cfg_bt.slippage_bps / 10_000.0
    total_cost_rate = (fee_rate + slip_rate) * 2  # 2 lados

    trade_records: list[dict] = []
    entry_price = np.nan
    entry_bar = 0

    for t in range(1, n):
        prev_pos = position[t - 1]
        new_pos = pos_desired[t]

        # Fechar posição por limite de holding
        if cfg_bt.max_holding_bars and holding_bars >= cfg_bt.max_holding_bars:
            new_pos = 0.0

        # Detectar mudança de posição
        changed = new_pos != prev_pos
        cost_this_bar = 0.0

        if changed:
            # Custo proporcional ao tamanho do trade
            trade_size = abs(new_pos - prev_pos)
            cost_this_bar = trade_size * total_cost_rate

            # Registrar trade
            if prev_pos != 0:
                trade_records.append({
                    "entry_bar": entry_bar,
                    "exit_bar": t,
                    "entry_price": entry_price,
                    "exit_price": df["open"].iloc[t] if "open" in df.columns else np.nan,
                    "side": "long" if prev_pos > 0 else "short",
                    "position": prev_pos,
                    "holding_bars": t - entry_bar,
                    "cost_bps": cfg_bt.fee_bps,
                    "slippage_bps": cfg_bt.slippage_bps,
                })

            if new_pos != 0:
                entry_price = df["open"].iloc[t] if "open" in df.columns else np.nan
                entry_bar = t
                holding_bars = 0
        else:
            holding_bars += 1

        position[t] = new_pos

        ret = prev_pos * log_ret[t] if not np.isnan(log_ret[t]) else 0.0
        strategy_ret[t] = ret - cost_this_bar

    # Equity curve
    equity = _build_equity(strategy_ret, cfg_bt.initial_capital)

    timestamps = df["timestamp"] if "timestamp" in df.columns else df.index
    eq_series = pd.Series(equity, index=timestamps, name="equity")
    ret_series = pd.Series(strategy_ret, index=timestamps, name="strategy_return")
    pos_series = pd.Series(position, index=timestamps, name="position")

    trades_df = pd.DataFrame(trade_records)
    if not trades_df.empty and "entry_bar" in trades_df.columns:
        trades_df["entry_timestamp"] = df["timestamp"].iloc[trades_df["entry_bar"].values].values
        trades_df["exit_timestamp"] = df["timestamp"].iloc[trades_df["exit_bar"].values].values

    return BacktestResult(
        config=cfg_bt,
        equity=eq_series,
        returns=ret_series,
        positions=pos_series,
        trades=trades_df,
    )


def _build_equity(returns: np.ndarray, initial: float) -> np.ndarray:
    """Constrói curva de equity a partir de retornos logarítmicos."""
    cum_log = np.cumsum(np.nan_to_num(returns))
    return initial * np.exp(cum_log)


def _validate_inputs(df: pd.DataFrame, signal: pd.Series) -> None:
    if len(df) != len(signal):
        raise ValueError(
            f"DataFrame ({len(df)} linhas) e signal ({len(signal)} linhas) têm tamanhos diferentes."
        )
    if "log_return" not in df.columns:
        raise KeyError("Coluna 'log_return' ausente no DataFrame.")


# ---------------------------------------------------------------------------
# Exportação de resultados
# ---------------------------------------------------------------------------


def save_backtest_results(
    result: BacktestResult,
    backtests_dir: str | Path,
) -> tuple[Path, Path]:
    """Salva equity curve em Parquet e trades em CSV."""
    backtests_dir = Path(backtests_dir)
    ensure_dir(backtests_dir)

    cfg = result.config
    prefix = f"{cfg.strategy_name}_{cfg.symbol}_{cfg.interval}"
    fee_tag = f"fee{int(cfg.fee_bps)}_slip{int(cfg.slippage_bps)}"

    # Equity parquet
    equity_df = pd.DataFrame({
        "timestamp": result.equity.index,
        "equity": result.equity.values,
        "strategy_return": result.returns.values,
        "position": result.positions.values,
    })
    equity_path = backtests_dir / f"equity_{prefix}_{fee_tag}.parquet"
    equity_df.to_parquet(equity_path, index=False, compression="snappy")

    # Trades CSV
    trades_path = backtests_dir / f"trades_{prefix}_{fee_tag}.csv"
    result.trades.to_csv(trades_path, index=False)

    logger.info("Backtest salvo: %s | %s", equity_path.name, trades_path.name)
    return equity_path, trades_path


def run_all_backtests(
    df: pd.DataFrame,
    signals_df: pd.DataFrame,
    cfg: dict,
    backtests_dir: str | Path,
    symbol: str,
    interval: str,
) -> list[BacktestResult]:
    """Roda backtests para todas as combinações de estratégia × custo × slippage."""
    bt_cfg = cfg["backtest"]
    fee_list = bt_cfg.get("fee_bps_per_side", [0, 2, 5, 10])
    slip_list = bt_cfg.get("slippage_bps_per_side", [0, 2, 5, 10])

    strategy_cols = [c for c in signals_df.columns if c != "timestamp"]
    results: list[BacktestResult] = []

    for strategy_name in strategy_cols:
        signal = signals_df[strategy_name]
        for fee in fee_list:
            for slip in slip_list:
                cfg_bt = BacktestConfig(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    interval=interval,
                    initial_capital=bt_cfg.get("initial_capital", 10_000),
                    fee_bps=fee,
                    slippage_bps=slip,
                    allow_short=bt_cfg.get("allow_short", True),
                    max_position=bt_cfg.get("max_position", 1.0),
                    close_eod=bt_cfg.get("close_position_end_of_day", False),
                    max_holding_bars=bt_cfg.get("max_holding_bars"),
                )
                result = run_backtest(df, signal, cfg_bt)
                save_backtest_results(result, backtests_dir)
                results.append(result)

    return results
