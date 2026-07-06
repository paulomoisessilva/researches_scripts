"""Motor de backtest causal para estratégias intradiárias.

Execução next_open (único modo suportado):
  - Sinal calculado no candle t (estratégias entregam posição desejada já deslocada)
  - Entrada/saída no open do candle t
  - Retorno bruto bar-a-bar: posição vigente × ln(open_t / open_{t-1})
  - Long: ln(exit_open / entry_open); short: ln(entry_open / exit_open)

Fechamento EOD:
  - No último candle UTC do dia, força flat no open desse candle
  - Não carrega posição para o dia seguinte

Custos:
  - Fee + slippage por lado, proporcional a |Δposição|
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
    gross_returns: pd.Series
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Motor principal
# ---------------------------------------------------------------------------


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cfg_bt: BacktestConfig,
) -> BacktestResult:
    """Executa backtest com execução next_open em preços de abertura.

    Parameters
    ----------
    df     : DataFrame com colunas 'timestamp', 'open' (e opcionalmente 'close').
    signal : Série com posição desejada no open do candle t (já deslocada 1 barra).
    cfg_bt : Configuração do backtest.
    """
    _validate_inputs(df, signal)

    opens = df["open"].values.astype(float)
    open_log_ret = _open_log_returns(opens)
    pos_desired = signal.values.copy()

    pos_desired = np.clip(pos_desired, -cfg_bt.max_position, cfg_bt.max_position)
    if not cfg_bt.allow_short:
        pos_desired = np.clip(pos_desired, 0, cfg_bt.max_position)

    n = len(df)
    position = np.zeros(n)
    strategy_ret = np.zeros(n)
    gross_ret = np.zeros(n)
    holding_bars = 0

    fee_rate = cfg_bt.fee_bps / 10_000.0
    slip_rate = cfg_bt.slippage_bps / 10_000.0
    per_side_rate = fee_rate + slip_rate

    utc_dates = _utc_dates(df["timestamp"]) if cfg_bt.close_eod else None

    trade_records: list[dict] = []
    entry_price = np.nan
    entry_bar = 0

    for t in range(1, n):
        prev_pos = position[t - 1]
        new_pos = pos_desired[t]

        if cfg_bt.max_holding_bars and holding_bars >= cfg_bt.max_holding_bars:
            new_pos = 0.0

        # EOD: fecha no open do último candle UTC do dia (não no open do dia seguinte).
        if cfg_bt.close_eod and utc_dates is not None and _is_last_bar_of_utc_day(t, n, utc_dates):
            new_pos = 0.0

        changed = new_pos != prev_pos
        cost_this_bar = 0.0

        if changed:
            trade_size = abs(new_pos - prev_pos)
            cost_this_bar = trade_size * per_side_rate

            if prev_pos != 0:
                exit_price = opens[t]
                side = "long" if prev_pos > 0 else "short"
                trade_records.append({
                    "entry_bar": entry_bar,
                    "exit_bar": t,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "side": side,
                    "position": prev_pos,
                    "holding_bars": t - entry_bar,
                    "cost_bps": cfg_bt.fee_bps,
                    "slippage_bps": cfg_bt.slippage_bps,
                    "trade_return": _trade_log_return(entry_price, exit_price, side),
                })

            if new_pos != 0:
                entry_price = opens[t]
                entry_bar = t
                holding_bars = 0
        else:
            holding_bars += 1

        position[t] = new_pos

        # Retorno open→open da posição vigente ao entrar no candle t.
        bar_log_ret = open_log_ret[t]
        price_ret = prev_pos * bar_log_ret if not np.isnan(bar_log_ret) else 0.0
        gross_ret[t] = price_ret
        strategy_ret[t] = price_ret - cost_this_bar

    equity = _build_equity(strategy_ret, cfg_bt.initial_capital)

    timestamps = df["timestamp"] if "timestamp" in df.columns else df.index
    eq_series = pd.Series(equity, index=timestamps, name="equity")
    ret_series = pd.Series(strategy_ret, index=timestamps, name="strategy_return")
    gross_series = pd.Series(gross_ret, index=timestamps, name="gross_return")
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
        gross_returns=gross_series,
    )


def _open_log_returns(opens: np.ndarray) -> np.ndarray:
    """ln(open_t / open_{t-1}); índice 0 = 0."""
    out = np.zeros(len(opens))
    if len(opens) < 2:
        return out
    ratio = opens[1:] / opens[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = np.log(ratio)
    out[~np.isfinite(out)] = np.nan
    return out


def _trade_log_return(entry_price: float, exit_price: float, side: str) -> float:
    """Retorno logarítmico do trade com preços open de entrada/saída."""
    if not np.isfinite(entry_price) or not np.isfinite(exit_price):
        return np.nan
    if entry_price <= 0 or exit_price <= 0:
        return np.nan
    if side == "long":
        return float(np.log(exit_price / entry_price))
    return float(np.log(entry_price / exit_price))


def _utc_dates(timestamps: pd.Series) -> np.ndarray:
    """Normaliza timestamps para datas UTC (numpy datetime64[D])."""
    ts = pd.to_datetime(timestamps, utc=True)
    return ts.dt.normalize().values.astype("datetime64[D]")


def _is_last_bar_of_utc_day(t: int, n: int, utc_dates: np.ndarray) -> bool:
    """True se o candle t é o último do dia UTC."""
    if t == n - 1:
        return True
    return utc_dates[t] != utc_dates[t + 1]


def _build_equity(returns: np.ndarray, initial: float) -> np.ndarray:
    """Constrói curva de equity a partir de retornos logarítmicos."""
    cum_log = np.cumsum(np.nan_to_num(returns))
    return initial * np.exp(cum_log)


def _validate_inputs(df: pd.DataFrame, signal: pd.Series) -> None:
    if len(df) != len(signal):
        raise ValueError(
            f"DataFrame ({len(df)} linhas) e signal ({len(signal)} linhas) têm tamanhos diferentes."
        )
    if "open" not in df.columns:
        raise KeyError("Coluna 'open' ausente no DataFrame (obrigatória para next_open).")


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

    equity_df = pd.DataFrame({
        "timestamp": result.equity.index,
        "equity": result.equity.values,
        "strategy_return": result.returns.values,
        "position": result.positions.values,
    })
    equity_path = backtests_dir / f"equity_{prefix}_{fee_tag}.parquet"
    equity_df.to_parquet(equity_path, index=False, compression="snappy")

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
