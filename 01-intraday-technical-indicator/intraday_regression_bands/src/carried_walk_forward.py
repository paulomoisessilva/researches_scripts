"""Walk-forward com uma única trajetória de posição (estado carregado entre os dias).

No modo por janela (``walk_forward._run_one_window``) cada dia de teste roda com os parâmetros
escolhidos e começa na posição que esses parâmetros teriam sozinhos. Com fechamento diário isso não
aparece, porque todo dia começa zerado. Sem fechamento, quando os parâmetros mudam de um dia para o
outro a posição salta, sem custo e sem unidades negociadas, da posição de ontem para a de hoje.

Aqui há uma só trajetória para o período de teste inteiro. A cada dia vale o z dos parâmetros
escolhidos para ele; a posição real continua de um dia para o outro. Na troca de parâmetros a regra
nova sai quando o z dela cruza a saída e só entra numa nova reversão confirmada. Custos e unidades
saem da trajetória real.

Convenções:

* a trajetória começa zerada na primeira barra do primeiro dia de teste (sem histórico de posição);
* os dias de teste precisam ser contíguos (``step_days == test_days``);
* no par (z anterior, z atual) da primeira barra de um dia, os dois z são da regra nova;
* o fechamento diário não é aceito (a trajetória é contínua por definição).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_backtest
from .fast_core import positions_from_pairs
from .metrics import compute_metrics
from .strategies import BAND_STRATEGIES, compute_band_z
from .utils import setup_logger
from .walk_forward import WFResult, WFWindow

logger = setup_logger(__name__)

_THRESHOLD_DEFAULTS = {
    "entry_long_z": 0.8,
    "entry_short_z": 0.8,
    "exit_long_z": 0.0,
    "exit_short_z": 0.0,
}


def _window_bounds(df_feat: pd.DataFrame, windows: list[WFWindow]) -> tuple[np.ndarray, np.ndarray]:
    """Índices (barra inicial, barra final exclusiva) do teste de cada janela."""
    ts = pd.DatetimeIndex(df_feat["timestamp"])
    starts = np.asarray(ts.searchsorted([w.test_start for w in windows], side="left"), dtype=int)
    ends = np.asarray(ts.searchsorted([w.test_end for w in windows], side="left"), dtype=int)
    return starts, ends


def _check_contiguous(windows: list[WFWindow]) -> None:
    for prev, nxt in zip(windows[:-1], windows[1:]):
        if prev.test_end != nxt.test_start:
            raise ValueError(
                "A trajetória contínua exige dias de teste contíguos (step_days == test_days): "
                f"o teste da janela {prev.window_id} termina em {prev.test_end} e o da janela "
                f"{nxt.window_id} começa em {nxt.test_start}."
            )


def carried_signal_inputs(
    df_feat: pd.DataFrame,
    strategy_name: str,
    windows: list[WFWindow],
    params_by_window: list[dict],
) -> dict:
    """Pares (z anterior, z atual) e limiares por barra da trajetória única, de ``s0`` a ``s1``.

    Cada janela contribui com o z dos seus parâmetros em ``[starts[i], ends[i])``; o z anterior de
    cada barra é o z da MESMA regra na barra anterior.
    """
    if strategy_name not in BAND_STRATEGIES:
        raise ValueError(f"Trajetória contínua só cobre {sorted(BAND_STRATEGIES)}.")
    if len(windows) != len(params_by_window):
        raise ValueError("Uma lista de parâmetros por janela é necessária.")
    _check_contiguous(windows)

    starts, ends = _window_bounds(df_feat, windows)
    s0, s1 = int(starts[0]), int(ends[-1])
    length = s1 - s0
    z_curr = np.full(length, np.nan)
    z_prev = np.full(length, np.nan)

    per_window = {
        key: np.array([float(p.get(key, default)) for p in params_by_window])
        for key, default in _THRESHOLD_DEFAULTS.items()
    }
    thresholds: dict[str, float | np.ndarray] = {}
    for key, vals in per_window.items():
        thresholds[key] = float(vals[0]) if np.all(vals == vals[0]) else np.full(length, np.nan)

    groups: dict[tuple[int, float], list[int]] = {}
    for i, p in enumerate(params_by_window):
        groups.setdefault((int(p["window"]), float(p["k"])), []).append(i)

    for (window, k), members in groups.items():
        z_full = compute_band_z(df_feat, strategy_name, window, k).to_numpy(dtype=float)
        for i in members:
            first, last = int(starts[i]), int(ends[i])
            if last <= first:
                continue
            a, b = first - s0, last - s0
            z_curr[a:b] = z_full[first:last]
            if first >= 1:
                z_prev[a:b] = z_full[first - 1:last - 1]
            else:  # sem barra anterior no conjunto de dados
                z_prev[a + 1:b] = z_full[first:last - 1]
            for key, arr in thresholds.items():
                if isinstance(arr, np.ndarray):
                    arr[a:b] = per_window[key][i]
    return {"s0": s0, "s1": s1, "starts": starts, "ends": ends, "z_prev": z_prev, "z_curr": z_curr, **thresholds}


def carried_backtest(
    df_feat: pd.DataFrame,
    strategy_name: str,
    windows: list[WFWindow],
    params_by_window: list[dict],
    bt_cfg: BacktestConfig,
) -> tuple[BacktestResult, dict]:
    """Backtest da trajetória única de ``s0`` a ``s1`` (índices posicionais de ``df_feat``)."""
    if bt_cfg.close_eod:
        raise ValueError("A trajetória contínua não aceita fechamento diário (close_eod=True).")
    if bt_cfg.max_holding_bars:
        raise ValueError("A trajetória contínua não aceita max_holding_bars.")

    inp = carried_signal_inputs(df_feat, strategy_name, windows, params_by_window)
    position_signal = positions_from_pairs(
        inp["z_prev"], inp["z_curr"],
        entry_long_z=inp["entry_long_z"], entry_short_z=inp["entry_short_z"],
        exit_long_z=inp["exit_long_z"], exit_short_z=inp["exit_short_z"],
    )
    # posição decidida no fechamento da barra t é assumida no open de t+1
    signal = np.zeros(len(position_signal))
    signal[1:] = position_signal[:-1]

    df_span = df_feat[["timestamp", "open"]].iloc[inp["s0"]:inp["s1"]].reset_index(drop=True)
    result = run_backtest(df_span, pd.Series(signal), bt_cfg)
    return result, inp


def run_carried_walk_forward(
    df_feat: pd.DataFrame,
    strategy_name: str,
    windows: list[WFWindow],
    params_by_window: list[dict],
    val_scores: list[float],
    bt_cfg: BacktestConfig,
    symbol: str,
    interval: str,
    fee_bps: float,
    slippage_bps: float,
) -> list[WFResult]:
    """Um ``WFResult`` por dia de teste, fatiado de uma única trajetória de posição."""
    if not windows:
        return []
    logger.info(
        "Trajetória contínua com estado carregado: %s %s, %d janelas.", strategy_name, symbol, len(windows)
    )
    result, inp = carried_backtest(df_feat, strategy_name, windows, params_by_window, bt_cfg)
    s0 = inp["s0"]
    trades = result.trades
    exit_bars = trades["exit_bar"].to_numpy() if not trades.empty else np.array([], dtype=int)

    out: list[WFResult] = []
    for i, win in enumerate(windows):
        a, b = int(inp["starts"][i]) - s0, int(inp["ends"][i]) - s0
        if b <= a:
            logger.warning("[WF %d] Sem barras no dia de teste, janela ignorada.", win.window_id)
            continue
        lo, hi = np.searchsorted(exit_bars, [a, b], side="left")
        trades_w = trades.iloc[lo:hi].reset_index(drop=True) if not trades.empty else trades
        equity = result.equity.iloc[a:b]
        returns = result.returns.iloc[a:b]
        positions = result.positions.iloc[a:b]
        gross = result.gross_returns.iloc[a:b]
        units = result.trade_units.iloc[a:b]
        metrics = compute_metrics(
            equity, returns, positions, trades_w,
            interval=interval, strategy_name=strategy_name, symbol=symbol,
            fee_bps=fee_bps, slippage_bps=slippage_bps,
            gross_returns=gross, trade_units=units,
        )
        out.append(WFResult(
            window_id=win.window_id,
            strategy=strategy_name,
            symbol=symbol,
            interval=interval,
            best_params=params_by_window[i],
            val_metric=val_scores[i],
            test_equity=equity,
            test_returns=returns,
            test_gross_returns=gross,
            test_positions=positions,
            test_trades=trades_w,
            test_metrics=metrics,
            test_trade_units=units,
        ))
    return out
