"""Base de operações do estudo v2 (uma linha por entrada) e features de mercado (volume e volatilidade), para as análises descritivas.

``rebuild_trades`` refaz cada trajetória contínua do walk-forward a partir dos parâmetros gravados (``best_parameters_by_window.csv``) e
extrai as operações; o bruto refeito é conferido contra o gravado nas tabelas de cada run. ``market_features`` calcula, só com barras
ANTERIORES a cada minuto, a volatilidade (raiz da soma dos quadrados dos retornos de 1 min, em bps) e o volume (USDT) das últimas 1 h,
3 h e 24 h.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult
from .carried_walk_forward import carried_backtest
from .config import filter_dataframe_by_date_range, load_config
from .indicators import compute_all_features
from .walk_forward import build_windows

FREQS = ("1m", "5m", "15m")
LOOKBACKS = (1, 2, 3, 7, 14, 21, 28)
BAR_MINUTES = {"1m": 1, "5m": 5, "15m": 15}
OOS_START = pd.Timestamp("2021-02-03", tz="UTC")
TRADE_COLUMNS = ["freq", "strategy", "asset", "lookback", "entry_ts", "exit_ts", "side", "gross_bps", "holding_min"]


def find_run_dir(results_root: Path, freq: str, lookback: int) -> Path:
    """Pasta da run de ``freq`` com o lookback dado (``results_root/<freq>/walk_forward/*_va<lookback>_*``)."""
    for d in sorted((Path(results_root) / freq / "walk_forward").iterdir()):
        if f"_va{lookback}_" in d.name:
            return d
    raise FileNotFoundError(f"Run de {freq} com lookback {lookback} não encontrada em {results_root}.")


def trades_frame(result: BacktestResult, freq: str, strategy: str, asset: str, lookback: int) -> pd.DataFrame:
    """Operações de um backtest (trajetória contínua) no formato da base, com horários em UTC e retorno bruto em bps."""
    tr = result.trades
    if tr.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.DataFrame({
        "freq": freq,
        "strategy": strategy,
        "asset": asset,
        "lookback": lookback,
        "entry_ts": pd.DatetimeIndex(tr["entry_timestamp"]).tz_localize("UTC") if pd.DatetimeIndex(tr["entry_timestamp"]).tz is None
        else pd.DatetimeIndex(tr["entry_timestamp"]).tz_convert("UTC"),
        "exit_ts": pd.DatetimeIndex(tr["exit_timestamp"]).tz_localize("UTC") if pd.DatetimeIndex(tr["exit_timestamp"]).tz is None
        else pd.DatetimeIndex(tr["exit_timestamp"]).tz_convert("UTC"),
        "side": tr["side"].astype(str).to_numpy(),
        "gross_bps": tr["trade_return"].to_numpy(dtype=float) * 1e4,
        "holding_min": tr["holding_bars"].to_numpy(dtype=float) * BAR_MINUTES[freq],
    })[TRADE_COLUMNS]


def rebuild_trades(
    project_root: Path,
    results_root: Path,
    lookbacks: tuple[int, ...] = LOOKBACKS,
    freqs: tuple[str, ...] = FREQS,
) -> tuple[pd.DataFrame, float]:
    """Refaz as 126 trajetórias e devolve (operações, maior diferença entre o bruto refeito e o gravado nas runs)."""
    frames: list[pd.DataFrame] = []
    worst = 0.0
    for freq in freqs:
        cfg = load_config(project_root / f"config.v2_{freq}.yaml")
        wf = cfg["walk_forward"]
        for symbol in cfg["data"]["symbols"]:
            df = pd.read_parquet(project_root / cfg["data"]["processed_dir"] / f"{symbol}_{freq}.parquet")
            df, _ = filter_dataframe_by_date_range(df, cfg["data"]["start_date"], cfg["data"]["end_date"])
            feat = compute_all_features(df, cfg, strategy_names=cfg["strategies"])
            bt = BacktestConfig(strategy_name="", symbol=symbol, interval=freq, fee_bps=0.0, slippage_bps=0.0, close_eod=False)
            for lookback in lookbacks:
                run = find_run_dir(results_root, freq, lookback)
                params_all = pd.read_csv(run / "best_parameters_by_window.csv")
                saved_all = pd.read_csv(run / "walk_forward_results.csv", usecols=["strategy", "symbol", "sum_gross_ret"])
                windows = build_windows(df[["timestamp"]], wf["train_days"], lookback, wf["test_days"], wf["step_days"],
                                        oos_start=wf["oos_start"])
                for strategy in cfg["strategies"]:
                    sel = params_all[(params_all["symbol"] == symbol) & (params_all["strategy"] == strategy)].sort_values("window_id")
                    params = [
                        {"window": int(r.param_window), "k": float(r.param_k), "entry_long_z": float(r.param_entry_long_z),
                         "entry_short_z": float(r.param_entry_short_z), "exit_long_z": float(r.param_exit_long_z),
                         "exit_short_z": float(r.param_exit_short_z)}
                        for r in sel.itertuples()
                    ]
                    bt.strategy_name = strategy
                    result, _ = carried_backtest(feat, strategy, windows, params, bt)
                    saved = float(saved_all[(saved_all["symbol"] == symbol) & (saved_all["strategy"] == strategy)]["sum_gross_ret"].sum())
                    worst = max(worst, abs(saved - float(result.gross_returns.sum())))
                    frames.append(trades_frame(result, freq, strategy, symbol, lookback))
    trades = pd.concat(frames, ignore_index=True)
    trades["strategy"] = trades["strategy"].astype("category")
    return trades, worst


# ---------------------------------------------------------------------------
# Features de mercado
# ---------------------------------------------------------------------------

_SUFFIX = {60: "", 180: "3", 1440: "24"}


def market_features(minute_bars: pd.DataFrame, windows: tuple[int, ...] = (60, 180, 1440)) -> pd.DataFrame:
    """Volatilidade e volume das últimas ``w`` barras de 1 minuto, conhecidos ANTES de cada minuto (índice = horário UTC).

    Colunas: ``rv``/``qv`` (1 h, na escala original: bps e USDT); ``z_rv``/``z_qv`` (log padronizado dentro do ativo) e, para 3 h e 24 h,
    ``z_rv3``/``z_qv3`` e ``z_rv24``/``z_qv24``; ``s_rv``/``s_qv`` = "surpresa" da 1 h (o log menos a média do mesmo dia da semana e hora,
    padronizado). As padronizações usam o arquivo inteiro de 1 minuto do ativo.
    """
    ret_bps = np.log(minute_bars["close"].astype(float)).diff() * 1e4
    sq = ret_bps ** 2
    qv = minute_bars["quote_volume"].astype(float)
    ts = pd.DatetimeIndex(minute_bars["timestamp"])
    out = pd.DataFrame(index=ts)
    cell = None
    for w in windows:
        s = _SUFFIX[w]
        rv = pd.Series(np.sqrt(sq.rolling(w, min_periods=w).sum().shift(1).to_numpy()), index=ts)
        vol = pd.Series(qv.rolling(w, min_periods=w).sum().shift(1).to_numpy(), index=ts)
        if w == 60:
            out["rv"], out["qv"] = rv, vol
        for name, series in (("rv", rv), ("qv", vol)):
            lg = np.log(series.clip(lower=1e-9))
            out[f"z_{name}{s}"] = (lg - lg.mean()) / lg.std()
            if w == 60:
                if cell is None:
                    cell = [ts.dayofweek, ts.hour]
                mean_cell = lg.groupby(cell).transform("mean")
                sur = lg - mean_cell
                out[f"s_{name}"] = sur / sur.std()
    return out
