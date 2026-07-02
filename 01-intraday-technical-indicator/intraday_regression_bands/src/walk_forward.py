"""Validação walk-forward (expanding/rolling window).

Processo:
  1. Separar treino, validação e teste
  2. Escolher parâmetros no treino/validação
  3. Aplicar APENAS no período de teste
  4. Rolar a janela step_days para frente
  5. Concatenar todos os resultados out-of-sample

Restrição fundamental: parâmetros NÃO podem ser escolhidos usando dados de teste.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .backtest import run_backtest, BacktestConfig, BacktestResult
from .indicators import compute_all_features
from .metrics import compute_metrics
from .strategies import STRATEGY_REGISTRY, get_default_params
from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


@dataclass
class WFWindow:
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WFResult:
    window_id: int
    strategy: str
    symbol: str
    interval: str
    best_params: dict
    val_metric: float
    test_equity: pd.Series
    test_returns: pd.Series
    test_positions: pd.Series
    test_trades: pd.DataFrame
    test_metrics: dict


def build_windows(
    df: pd.DataFrame,
    train_days: int,
    val_days: int,
    test_days: int,
    step_days: int,
) -> list[WFWindow]:
    """Gera lista de janelas walk-forward.

    Cada janela tem período de treino, validação e teste não-sobrepostos.
    """
    if "timestamp" not in df.columns:
        raise KeyError("Coluna 'timestamp' necessária para walk-forward.")

    timestamps = df["timestamp"].sort_values().reset_index(drop=True)
    min_date = timestamps.iloc[0]
    max_date = timestamps.iloc[-1]

    windows: list[WFWindow] = []
    wid = 0
    train_start = min_date

    while True:
        train_end = train_start + pd.Timedelta(days=train_days)
        val_start = train_end
        val_end = val_start + pd.Timedelta(days=val_days)
        test_start = val_end
        test_end = test_start + pd.Timedelta(days=test_days)

        if test_end > max_date:
            break

        windows.append(WFWindow(
            window_id=wid,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        ))
        train_start += pd.Timedelta(days=step_days)
        wid += 1

    logger.info("Walk-forward: %d janelas geradas.", len(windows))
    return windows


def _slice_df(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
    return df[mask].reset_index(drop=True)


def _run_strategy_on_slice(
    df_slice: pd.DataFrame,
    strategy_name: str,
    params: dict,
    cfg: dict,
    bt_cfg: BacktestConfig,
) -> tuple[BacktestResult, dict]:
    """Recalcula features e roda backtest em um slice de dados."""
    from .strategies import STRATEGY_REGISTRY

    df_feat = compute_all_features(df_slice, cfg)
    func = STRATEGY_REGISTRY[strategy_name]

    if strategy_name == "random":
        signal = func(df_feat, **params)
    else:
        signal = func(df_feat, **params)

    result = run_backtest(df_feat, signal, bt_cfg)
    return result, df_feat


def optimize_params_on_validation(
    df_train_val: pd.DataFrame,
    strategy_name: str,
    cfg: dict,
    bt_cfg_template: BacktestConfig,
    selection_metric: str = "net_sharpe",
) -> tuple[dict, float]:
    """Busca os melhores parâmetros de uma estratégia no período treino+validação.

    Retorna (best_params, best_metric_value).
    Usa grid search simples sobre os parâmetros da config.
    """
    feat_cfg = cfg.get("features", {})

    # Grade de parâmetros por estratégia
    param_grid = _build_param_grid(strategy_name, feat_cfg)

    best_params = param_grid[0] if param_grid else {}
    best_score = -np.inf

    for params in param_grid:
        try:
            result, _ = _run_strategy_on_slice(
                df_train_val, strategy_name, params, cfg, bt_cfg_template
            )
            from .metrics import compute_metrics, annualized_return, annualized_volatility, sharpe_ratio
            score = sharpe_ratio(result.returns, bt_cfg_template.interval)
            if score > best_score:
                best_score = score
                best_params = params
        except Exception as exc:
            logger.debug("Params %s falhou: %s", params, exc)

    return best_params, best_score


def _build_param_grid(strategy_name: str, feat_cfg: dict) -> list[dict]:
    """Constrói grade de parâmetros para uma estratégia."""
    bb_windows = feat_cfg.get("bollinger_windows", [20])
    bb_ks = feat_cfg.get("bollinger_k", [2.0])
    reg_windows = feat_cfg.get("regression_windows", [100])
    reg_ks = feat_cfg.get("regression_k", [2.0])
    rsi_windows = feat_cfg.get("rsi_windows", [14])
    stoch_windows = feat_cfg.get("stochastic_windows", [14])

    grids = {
        "bollinger": [
            {"window": w, "k": k}
            for w in bb_windows
            for k in bb_ks
        ],
        "regression_bands": [
            {"window": w, "k": k}
            for w in reg_windows
            for k in reg_ks
        ],
        "return_residuals": [
            {"window": w, "k": k}
            for w in reg_windows
            for k in reg_ks
        ],
        "rsi": [{"window": w} for w in rsi_windows],
        "stochastic": [{"window": w} for w in stoch_windows],
        "technical_confluence": [{}],
        "random": [{}],
    }
    return grids.get(strategy_name, [{}])


def run_walk_forward(
    df: pd.DataFrame,
    strategy_name: str,
    cfg: dict,
    symbol: str,
    interval: str,
    fee_bps: float = 2.0,
    slippage_bps: float = 2.0,
) -> list[WFResult]:
    """Executa o processo walk-forward completo para uma estratégia."""
    wf_cfg = cfg["walk_forward"]
    windows = build_windows(
        df,
        train_days=wf_cfg["train_days"],
        val_days=wf_cfg["val_days"],
        test_days=wf_cfg["test_days"],
        step_days=wf_cfg["step_days"],
    )

    if not windows:
        logger.warning("Sem janelas walk-forward para %s %s", symbol, interval)
        return []

    bt_cfg_template = BacktestConfig(
        strategy_name=strategy_name,
        symbol=symbol,
        interval=interval,
        initial_capital=cfg["backtest"]["initial_capital"],
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        allow_short=cfg["backtest"]["allow_short"],
        max_position=cfg["backtest"]["max_position"],
    )

    results: list[WFResult] = []

    for win in windows:
        logger.info(
            "[WF %d] %s %s | treino %s–%s | val %s–%s | teste %s–%s",
            win.window_id, strategy_name, symbol,
            win.train_start.date(), win.train_end.date(),
            win.val_start.date(), win.val_end.date(),
            win.test_start.date(), win.test_end.date(),
        )

        df_train_val = _slice_df(df, win.train_start, win.val_end)
        df_test = _slice_df(df, win.test_start, win.test_end)

        if len(df_train_val) < 50 or len(df_test) < 10:
            logger.warning("[WF %d] Dados insuficientes, janela ignorada.", win.window_id)
            continue

        # Otimizar parâmetros em treino+validação
        best_params, val_score = optimize_params_on_validation(
            df_train_val, strategy_name, cfg, bt_cfg_template
        )

        # Aplicar melhores parâmetros no teste (out-of-sample)
        try:
            test_result, _ = _run_strategy_on_slice(
                df_test, strategy_name, best_params, cfg, bt_cfg_template
            )
        except Exception as exc:
            logger.warning("[WF %d] Erro no teste: %s", win.window_id, exc)
            continue

        test_metrics = compute_metrics(
            test_result.equity,
            test_result.returns,
            test_result.positions,
            test_result.trades,
            interval=interval,
            strategy_name=strategy_name,
            symbol=symbol,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

        results.append(WFResult(
            window_id=win.window_id,
            strategy=strategy_name,
            symbol=symbol,
            interval=interval,
            best_params=best_params,
            val_metric=val_score,
            test_equity=test_result.equity,
            test_returns=test_result.returns,
            test_positions=test_result.positions,
            test_trades=test_result.trades,
            test_metrics=test_metrics,
        ))

    return results


def save_walk_forward_results(
    all_results: list[WFResult],
    tables_dir: str | Path,
    backtests_dir: str | Path,
) -> None:
    """Salva resultados walk-forward em CSV, Parquet e tabela de parâmetros."""
    tables_dir = Path(tables_dir)
    backtests_dir = Path(backtests_dir)
    ensure_dir(tables_dir)
    ensure_dir(backtests_dir)

    if not all_results:
        logger.warning("Nenhum resultado walk-forward para salvar.")
        return

    # Tabela de métricas por janela
    metrics_rows = []
    params_rows = []
    equity_frames = []

    for r in all_results:
        row = {"window_id": r.window_id, **r.test_metrics}
        metrics_rows.append(row)

        params_rows.append({
            "window_id": r.window_id,
            "strategy": r.strategy,
            "symbol": r.symbol,
            "interval": r.interval,
            "val_metric": r.val_metric,
            **{f"param_{k}": v for k, v in r.best_params.items()},
        })

        eq_df = pd.DataFrame({
            "timestamp": r.test_equity.index,
            "equity": r.test_equity.values,
            "strategy": r.strategy,
            "symbol": r.symbol,
            "interval": r.interval,
            "window_id": r.window_id,
        })
        equity_frames.append(eq_df)

    pd.DataFrame(metrics_rows).to_csv(tables_dir / "walk_forward_results.csv", index=False)
    pd.DataFrame(params_rows).to_csv(tables_dir / "best_parameters_by_window.csv", index=False)

    if equity_frames:
        pd.concat(equity_frames, ignore_index=True).to_parquet(
            backtests_dir / "walk_forward_equity.parquet", index=False
        )

    logger.info("Resultados walk-forward salvos em %s", tables_dir)
