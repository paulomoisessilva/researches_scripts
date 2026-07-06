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

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .backtest import run_backtest, BacktestConfig, BacktestResult
from .indicators import compute_features_for_strategy
from .metrics import (
    break_even_cost_bps,
    break_even_total_return_bps,
    compute_metrics,
    max_drawdown,
    selection_metric_score,
    sharpe_ratio,
)
from .strategies import (
    BAND_STRATEGIES,
    STRATEGY_REGISTRY,
    compute_band_z,
    resolve_random_seeds,
    signal_from_z,
    strategy_random,
)
from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Estado compartilhado por workers do grid (ProcessPoolExecutor + initializer)
# ---------------------------------------------------------------------------

_worker_df: pd.DataFrame | None = None
_worker_bt_cfg: BacktestConfig | None = None
_worker_eval_start: pd.Timestamp | None = None
_worker_eval_end: pd.Timestamp | None = None
_worker_selection_metric: str = "net_sharpe"


@dataclass(frozen=True)
class BandGridTask:
    struct: dict
    thresh: dict
    z_values: np.ndarray
    strategy_name: str


@dataclass(frozen=True)
class StrategyGridTask:
    strategy_name: str
    params: dict


@dataclass(frozen=True)
class GridEvalResult:
    params: dict
    score: float


def _init_grid_worker(
    df_feat: pd.DataFrame,
    bt_cfg: BacktestConfig,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    selection_metric: str,
) -> None:
    global _worker_df, _worker_bt_cfg, _worker_eval_start, _worker_eval_end
    global _worker_selection_metric
    _worker_df = df_feat
    _worker_bt_cfg = bt_cfg
    _worker_eval_start = eval_start
    _worker_eval_end = eval_end
    _worker_selection_metric = selection_metric


def _score_backtest_on_period(
    result: BacktestResult,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    interval: str,
    selection_metric: str = "net_sharpe",
) -> float:
    """Calcula métrica de seleção apenas no subperíodo [eval_start, eval_end)."""
    mask = _period_mask(result.returns.index, eval_start, eval_end)
    return selection_metric_score(
        result.returns,
        result.trades,
        interval,
        selection_metric,
        index_mask=mask,
    )


def _period_mask(
    index: pd.Index,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Máscara booleana para [start, end) alinhada ao índice (timestamps)."""
    idx = pd.to_datetime(index)
    return (idx >= start) & (idx < end)


def _filter_trades_by_period(
    trades: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Mantém trades cujo exit_timestamp cai em [start, end)."""
    if trades.empty:
        return trades
    if "exit_timestamp" not in trades.columns:
        raise KeyError(
            "Coluna 'exit_timestamp' ausente em trades; "
            "não é possível filtrar por janela de avaliação."
        )
    exit_ts = pd.to_datetime(trades["exit_timestamp"], utc=True)
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    mask = (exit_ts >= start_ts) & (exit_ts < end_ts)
    return trades.loc[mask].reset_index(drop=True)


def _slice_backtest_result(
    result: BacktestResult,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> BacktestResult:
    """Extrai métricas de backtest apenas no subperíodo [start, end)."""
    mask = _period_mask(result.equity.index, start, end)
    return BacktestResult(
        config=result.config,
        equity=result.equity.loc[mask],
        returns=result.returns.loc[mask],
        positions=result.positions.loc[mask],
        trades=_filter_trades_by_period(result.trades, start, end),
        gross_returns=result.gross_returns.loc[mask],
    )


def _result_seed(result: WFResult) -> int | None:
    seed = result.best_params.get("seed")
    return int(seed) if seed is not None else None


def _get_wf_int(wf_cfg: dict, primary: str, *aliases: str) -> int:
    """Lê parâmetro inteiro do bloco walk_forward (com aliases)."""
    for key in (primary, *aliases):
        if key in wf_cfg:
            return int(wf_cfg[key])
    raise KeyError(
        f"Chave walk_forward ausente: '{primary}'"
        + (f" (aliases: {aliases})" if aliases else "")
    )


def _eval_band_combo(task: BandGridTask) -> GridEvalResult | None:
    global _worker_df, _worker_bt_cfg, _worker_eval_start, _worker_eval_end
    global _worker_selection_metric
    if (
        _worker_df is None
        or _worker_bt_cfg is None
        or _worker_eval_start is None
        or _worker_eval_end is None
    ):
        return None
    try:
        z = pd.Series(task.z_values, index=_worker_df.index)
        signal = signal_from_z(z, name=task.strategy_name, **task.thresh)
        result = run_backtest(_worker_df, signal, _worker_bt_cfg)
        score = _score_backtest_on_period(
            result,
            _worker_eval_start,
            _worker_eval_end,
            _worker_bt_cfg.interval,
            _worker_selection_metric,
        )
        return GridEvalResult(params={**task.struct, **task.thresh}, score=score)
    except Exception:
        return None


def _eval_strategy_combo(task: StrategyGridTask) -> GridEvalResult | None:
    global _worker_df, _worker_bt_cfg, _worker_eval_start, _worker_eval_end
    global _worker_selection_metric
    if (
        _worker_df is None
        or _worker_bt_cfg is None
        or _worker_eval_start is None
        or _worker_eval_end is None
    ):
        return None
    try:
        func = STRATEGY_REGISTRY[task.strategy_name]
        signal = func(_worker_df, **task.params)
        result = run_backtest(_worker_df, signal, _worker_bt_cfg)
        score = _score_backtest_on_period(
            result,
            _worker_eval_start,
            _worker_eval_end,
            _worker_bt_cfg.interval,
            _worker_selection_metric,
        )
        return GridEvalResult(params=task.params, score=score)
    except Exception:
        return None


def _resolve_n_workers(cfg: dict, n_workers: int | None = None) -> int:
    """Retorna número de workers; 0 ou 1 = modo sequencial."""
    if n_workers is not None:
        return max(0, int(n_workers))
    return max(0, int(cfg.get("walk_forward", {}).get("parallel_workers", 1)))


def _select_best_from_grid_results(
    results: list[GridEvalResult | None],
    default_params: dict,
) -> tuple[dict, float]:
    best_params = default_params
    best_score = -np.inf
    for result in results:
        if result is None:
            continue
        if result.score > best_score:
            best_score = result.score
            best_params = result.params
    return best_params, best_score


def _run_tasks_parallel(
    tasks: list[Any],
    df_feat: pd.DataFrame,
    bt_cfg: BacktestConfig,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    selection_metric: str,
    n_workers: int,
    eval_fn: Callable[[Any], GridEvalResult | None],
) -> list[GridEvalResult | None]:
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_grid_worker,
        initargs=(df_feat, bt_cfg, eval_start, eval_end, selection_metric),
    ) as executor:
        return list(executor.map(eval_fn, tasks))


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
    test_gross_returns: pd.Series
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


def _run_backtest_on_features(
    df_feat: pd.DataFrame,
    strategy_name: str,
    params: dict,
    bt_cfg: BacktestConfig,
) -> BacktestResult:
    """Roda estratégia + backtest em DataFrame que já possui features."""
    func = STRATEGY_REGISTRY[strategy_name]
    signal = func(df_feat, **params)
    return run_backtest(df_feat, signal, bt_cfg)


def _run_strategy_on_slice(
    df_slice: pd.DataFrame,
    strategy_name: str,
    params: dict,
    cfg: dict,
    bt_cfg: BacktestConfig,
    df_context: pd.DataFrame | None = None,
    context_end: pd.Timestamp | None = None,
) -> tuple[BacktestResult, pd.DataFrame]:
    """Calcula features com histórico e roda backtest.

    Se ``df_context`` for informado, features são calculadas sobre o contexto
    histórico até ``context_end`` (ou fim do slice), evitando janelas isoladas.
    """
    if df_context is not None:
        end = context_end if context_end is not None else df_slice["timestamp"].iloc[-1]
        df_hist = _slice_df(df_context, df_context["timestamp"].min(), end)
        df_feat_full = compute_features_for_strategy(strategy_name, df_hist, cfg)
        start = df_slice["timestamp"].iloc[0]
        df_feat = _slice_df(df_feat_full, start, end)
    else:
        df_feat = compute_features_for_strategy(strategy_name, df_slice, cfg)
    result = _run_backtest_on_features(df_feat, strategy_name, params, bt_cfg)
    return result, df_feat


def optimize_params_on_validation(
    df_train_val: pd.DataFrame,
    strategy_name: str,
    cfg: dict,
    bt_cfg_template: BacktestConfig,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    selection_metric: str = "net_sharpe",
    df_feat_train_val: pd.DataFrame | None = None,
    n_workers: int = 1,
) -> tuple[dict, float]:
    """Busca os melhores parâmetros usando desempenho APENAS na validação.

    O backtest roda sobre treino+validação (contexto histórico e estado da
    estratégia), mas o score de seleção é calculado só em [val_start, val_end).

    Parameters
    ----------
    df_feat_train_val:
        DataFrame com features já calculadas para o slice treino+validação.
        Se None, calcula features internamente (compatibilidade).
    n_workers:
        Workers para o grid search. <= 1 executa sequencialmente.
    """
    if df_feat_train_val is None:
        df_feat_train_val = compute_features_for_strategy(strategy_name, df_train_val, cfg)

    structural_grid = _build_structural_param_grid(strategy_name, cfg)
    threshold_grid = _build_threshold_param_grid(cfg)

    if strategy_name in BAND_STRATEGIES:
        return _optimize_band_strategy(
            df_feat_train_val,
            strategy_name,
            structural_grid,
            threshold_grid,
            bt_cfg_template,
            val_start=val_start,
            val_end=val_end,
            selection_metric=selection_metric,
            n_workers=n_workers,
        )

    param_grid = _build_param_grid(strategy_name, cfg)
    default_params = param_grid[0] if param_grid else {}

    if n_workers <= 1:
        best_params = default_params
        best_score = -np.inf
        for params in param_grid:
            try:
                result = _run_backtest_on_features(
                    df_feat_train_val, strategy_name, params, bt_cfg_template
                )
                score = _score_backtest_on_period(
                    result,
                    val_start,
                    val_end,
                    bt_cfg_template.interval,
                    selection_metric,
                )
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as exc:
                logger.debug("Params %s falhou: %s", params, exc)
        return best_params, best_score

    tasks = [StrategyGridTask(strategy_name=strategy_name, params=p) for p in param_grid]
    logger.debug(
        "Grid search paralelo (%s): %d workers, %d combinações.",
        strategy_name, n_workers, len(tasks),
    )
    results = _run_tasks_parallel(
        tasks,
        df_feat_train_val,
        bt_cfg_template,
        val_start,
        val_end,
        selection_metric,
        n_workers,
        _eval_strategy_combo,
    )
    return _select_best_from_grid_results(results, default_params)


def _build_band_grid_tasks(
    df_feat: pd.DataFrame,
    strategy_name: str,
    structural_grid: list[dict],
    threshold_grid: list[dict],
) -> list[BandGridTask]:
    tasks: list[BandGridTask] = []
    for struct in structural_grid:
        try:
            z = compute_band_z(df_feat, strategy_name, struct["window"], struct["k"])
        except Exception as exc:
            logger.debug("Structural params %s falhou ao calcular z: %s", struct, exc)
            continue
        z_values = z.values.copy()
        for thresh in threshold_grid:
            tasks.append(BandGridTask(struct, thresh, z_values, strategy_name))
    return tasks


def _optimize_band_strategy(
    df_feat: pd.DataFrame,
    strategy_name: str,
    structural_grid: list[dict],
    threshold_grid: list[dict],
    bt_cfg: BacktestConfig,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    selection_metric: str = "net_sharpe",
    n_workers: int = 1,
) -> tuple[dict, float]:
    """Grid search em dois níveis: z por (window, k), sinal por limiares."""
    default_params = (
        {**structural_grid[0], **threshold_grid[0]}
        if structural_grid and threshold_grid
        else {}
    )

    if n_workers <= 1:
        best_params = default_params
        best_score = -np.inf
        for struct in structural_grid:
            try:
                z = compute_band_z(df_feat, strategy_name, struct["window"], struct["k"])
            except Exception as exc:
                logger.debug("Structural params %s falhou ao calcular z: %s", struct, exc)
                continue

            for thresh in threshold_grid:
                try:
                    signal = signal_from_z(z, name=strategy_name, **thresh)
                    result = run_backtest(df_feat, signal, bt_cfg)
                    score = _score_backtest_on_period(
                        result,
                        val_start,
                        val_end,
                        bt_cfg.interval,
                        selection_metric,
                    )
                    params = {**struct, **thresh}
                    if score > best_score:
                        best_score = score
                        best_params = params
                except Exception as exc:
                    logger.debug("Params %s falhou: %s", {**struct, **thresh}, exc)
        return best_params, best_score

    tasks = _build_band_grid_tasks(df_feat, strategy_name, structural_grid, threshold_grid)
    if not tasks:
        return default_params, -np.inf

    logger.debug(
        "Grid search paralelo (%s): %d workers, %d combinações.",
        strategy_name, n_workers, len(tasks),
    )
    results = _run_tasks_parallel(
        tasks,
        df_feat,
        bt_cfg,
        val_start,
        val_end,
        selection_metric,
        n_workers,
        _eval_band_combo,
    )
    return _select_best_from_grid_results(results, default_params)


def _band_z_param_combos(cfg: dict) -> list[dict]:
    """Combinações de limiares z para estratégias de bandas."""
    strat_cfg = cfg.get("strategy", {})
    entry_long_zs = strat_cfg.get("entry_long_z", [0.8])
    entry_short_zs = strat_cfg.get("entry_short_z", [0.8])
    exit_long_zs = strat_cfg.get("exit_long_z", [0.0])
    exit_short_zs = strat_cfg.get("exit_short_z", [0.0])

    return [
        {
            "entry_long_z": elz,
            "entry_short_z": esz,
            "exit_long_z": exlz,
            "exit_short_z": exsz,
        }
        for elz in entry_long_zs
        for esz in entry_short_zs
        for exlz in exit_long_zs
        for exsz in exit_short_zs
    ]


def _build_threshold_param_grid(cfg: dict) -> list[dict]:
    """Grade de limiares z para estratégias de bandas."""
    return _band_z_param_combos(cfg)


def _build_structural_param_grid(strategy_name: str, cfg: dict) -> list[dict]:
    """Grade de parâmetros estruturais (window, k) por estratégia de bandas."""
    feat_cfg = cfg.get("features", {})

    if strategy_name == "bollinger":
        return [
            {"window": w, "k": k}
            for w in feat_cfg.get("bollinger_windows", [20])
            for k in feat_cfg.get("bollinger_k", [2.0])
        ]

    if strategy_name in ("regression_bands", "return_residuals"):
        return [
            {"window": w, "k": k}
            for w in feat_cfg.get("regression_windows", [100])
            for k in feat_cfg.get("regression_k", [2.0])
        ]

    return [{}]


def _build_param_grid(strategy_name: str, cfg: dict) -> list[dict]:
    """Constrói grade de parâmetros para uma estratégia."""
    feat_cfg = cfg.get("features", {})
    bb_windows = feat_cfg.get("bollinger_windows", [20])
    bb_ks = feat_cfg.get("bollinger_k", [2.0])
    reg_windows = feat_cfg.get("regression_windows", [100])
    reg_ks = feat_cfg.get("regression_k", [2.0])
    rsi_windows = feat_cfg.get("rsi_windows", [14])
    stoch_windows = feat_cfg.get("stochastic_windows", [14])
    z_combos = _band_z_param_combos(cfg)

    band_grids = {
        "bollinger": [
            {**{"window": w, "k": k}, **z}
            for w in bb_windows
            for k in bb_ks
            for z in z_combos
        ],
        "regression_bands": [
            {**{"window": w, "k": k}, **z}
            for w in reg_windows
            for k in reg_ks
            for z in z_combos
        ],
        "return_residuals": [
            {**{"window": w, "k": k}, **z}
            for w in reg_windows
            for k in reg_ks
            for z in z_combos
        ],
    }

    grids = {
        **band_grids,
        "rsi": [{"window": w} for w in rsi_windows],
        "stochastic": [{"window": w} for w in stoch_windows],
        "technical_confluence": [{}],
        "random": [{}],
    }
    return grids.get(strategy_name, [{}])


def _random_baseline_params(cfg: dict, seed: int) -> dict:
    rb = cfg.get("random_baseline", {})
    params: dict[str, Any] = {"seed": seed}
    if "trade_prob" in rb:
        params["trade_prob"] = float(rb["trade_prob"])
    return params


def _run_walk_forward_random(
    df: pd.DataFrame,
    cfg: dict,
    symbol: str,
    interval: str,
    windows: list[WFWindow],
    df_feat_full: pd.DataFrame,
    bt_cfg_template: BacktestConfig,
    wf_cfg: dict,
) -> list[WFResult]:
    """Walk-forward para baseline aleatório — sem otimização na validação."""
    seeds = resolve_random_seeds(cfg)
    results: list[WFResult] = []
    n_windows = len(windows)
    log_every = max(1, n_windows // 10)

    for seed_idx, seed in enumerate(seeds):
        params = _random_baseline_params(cfg, seed)
        # Um sinal por seed no dataset completo; cada janela backtesta só o teste.
        full_signal = strategy_random(df_feat_full, **params)

        for win_idx, win in enumerate(windows):
            ts = df_feat_full["timestamp"]
            test_mask = (ts >= win.test_start) & (ts < win.test_end)
            df_feat_test = df_feat_full.loc[test_mask].reset_index(drop=True)
            min_test_bars = 1 if wf_cfg.get("test_days", 1) == 1 else 10
            if len(df_feat_test) < min_test_bars:
                continue

            try:
                signal_test = full_signal.loc[test_mask].reset_index(drop=True)
                test_result = run_backtest(df_feat_test, signal_test, bt_cfg_template)
            except Exception as exc:
                logger.warning("[WF %d] random seed=%d erro: %s", win.window_id, seed, exc)
                continue

            test_metrics = compute_metrics(
                test_result.equity,
                test_result.returns,
                test_result.positions,
                test_result.trades,
                interval=interval,
                strategy_name="random",
                symbol=symbol,
                fee_bps=bt_cfg_template.fee_bps,
                slippage_bps=bt_cfg_template.slippage_bps,
                gross_returns=test_result.gross_returns,
            )

            results.append(WFResult(
                window_id=win.window_id,
                strategy="random",
                symbol=symbol,
                interval=interval,
                best_params=params,
                val_metric=float("nan"),
                test_equity=test_result.equity,
                test_returns=test_result.returns,
                test_gross_returns=test_result.gross_returns,
                test_positions=test_result.positions,
                test_trades=test_result.trades,
                test_metrics=test_metrics,
            ))

            if (win_idx + 1) % log_every == 0 or win_idx == n_windows - 1:
                logger.info(
                    "Random seed %d/%d | janela %d/%d concluída.",
                    seed_idx + 1, len(seeds), win_idx + 1, n_windows,
                )

    logger.info(
        "Random baseline: %d seeds x %d janelas -> %d resultados.",
        len(seeds), n_windows, len(results),
    )
    return results


def run_walk_forward(
    df: pd.DataFrame,
    strategy_name: str,
    cfg: dict,
    symbol: str,
    interval: str,
    fee_bps: float = 2.0,
    slippage_bps: float = 2.0,
    df_feat: pd.DataFrame | None = None,
    n_workers: int | None = None,
) -> list[WFResult]:
    """Executa o processo walk-forward completo para uma estratégia."""
    wf_cfg = cfg["walk_forward"]
    workers = _resolve_n_workers(cfg, n_workers)
    windows = build_windows(
        df,
        train_days=_get_wf_int(wf_cfg, "train_days"),
        val_days=_get_wf_int(wf_cfg, "validation_days", "val_days"),
        test_days=_get_wf_int(wf_cfg, "test_days"),
        step_days=_get_wf_int(wf_cfg, "step_days"),
    )

    if not windows:
        logger.warning("Sem janelas walk-forward para %s %s", symbol, interval)
        return []

    close_eod = wf_cfg.get("close_position_end_of_day", True)
    selection_metric = wf_cfg.get("selection_metric", "net_sharpe")

    bt_cfg_template = BacktestConfig(
        strategy_name=strategy_name,
        symbol=symbol,
        interval=interval,
        initial_capital=cfg["backtest"]["initial_capital"],
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        allow_short=cfg["backtest"]["allow_short"],
        max_position=cfg["backtest"]["max_position"],
        close_eod=close_eod,
    )

    if df_feat is None:
        logger.info("Pré-calculando features para %s (%d candles)...", strategy_name, len(df))
        df_feat_full = compute_features_for_strategy(strategy_name, df, cfg)
    else:
        logger.info("Usando features pré-carregadas para %s (%d candles).", strategy_name, len(df_feat))
        df_feat_full = df_feat

    if strategy_name == "random":
        return _run_walk_forward_random(
            df, cfg, symbol, interval, windows, df_feat_full, bt_cfg_template, wf_cfg
        )

    if workers > 1:
        logger.info("Grid search paralelo com %d workers.", workers)

    results: list[WFResult] = []

    for win in windows:
        logger.info(
            "[WF %d] %s %s | treino %s–%s | val %s–%s | teste %s–%s",
            win.window_id, strategy_name, symbol,
            win.train_start.date(), win.train_end.date(),
            win.val_start.date(), win.val_end.date(),
            win.test_start.date(), win.test_end.date(),
        )

        df_feat_train_val = _slice_df(df_feat_full, win.train_start, win.val_end)
        df_feat_test_context = _slice_df(df_feat_full, win.train_start, win.test_end)

        min_test_bars = 1 if wf_cfg.get("test_days", 1) == 1 else 10
        if len(df_feat_train_val) < 50 or len(df_feat_test_context) < min_test_bars:
            logger.warning("[WF %d] Dados insuficientes, janela ignorada.", win.window_id)
            continue

        best_params, val_score = optimize_params_on_validation(
            df_train_val=df,
            strategy_name=strategy_name,
            cfg=cfg,
            bt_cfg_template=bt_cfg_template,
            val_start=win.val_start,
            val_end=win.val_end,
            selection_metric=selection_metric,
            df_feat_train_val=df_feat_train_val,
            n_workers=workers,
        )

        try:
            test_result_full = _run_backtest_on_features(
                df_feat_test_context, strategy_name, best_params, bt_cfg_template
            )
            test_result = _slice_backtest_result(
                test_result_full, win.test_start, win.test_end
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
            gross_returns=test_result.gross_returns,
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
            test_gross_returns=test_result.gross_returns,
            test_positions=test_result.positions,
            test_trades=test_result.trades,
            test_metrics=test_metrics,
        ))

    return results


def aggregate_walk_forward_oos(all_results: list[WFResult]) -> pd.DataFrame:
    """Concatena períodos de teste OOS e calcula métricas + CBE por estratégia/seed."""
    groups: dict[tuple[str, str, str, int | None], list[WFResult]] = {}
    for r in all_results:
        key = (r.strategy, r.symbol, r.interval, _result_seed(r))
        groups.setdefault(key, []).append(r)

    rows: list[dict[str, Any]] = []
    for (strategy, symbol, interval, seed), results in sorted(groups.items()):
        results = sorted(results, key=lambda x: x.window_id)
        returns = pd.concat([r.test_returns for r in results]).sort_index()
        gross_returns = pd.concat([r.test_gross_returns for r in results]).sort_index()
        positions = pd.concat([r.test_positions for r in results]).sort_index()
        positions = positions.reindex(returns.index).fillna(0.0)
        gross_returns = gross_returns.reindex(returns.index).fillna(0.0)

        log_returns = returns.fillna(0)
        equity = pd.Series(np.exp(log_returns.cumsum()), index=returns.index)

        row: dict[str, Any] = {
            "strategy": strategy,
            "symbol": symbol,
            "interval": interval,
            "n_windows": len(results),
            "oos_total_return": float(np.exp(log_returns.sum()) - 1.0),
            "oos_sharpe": sharpe_ratio(returns, interval),
            "oos_max_drawdown": max_drawdown(equity),
            "number_of_trades": int(sum(len(r.test_trades) for r in results)),
            "break_even_total_return_bps": break_even_total_return_bps(gross_returns, positions),
            "break_even_cost_bps": break_even_cost_bps(gross_returns, positions, interval),
        }
        if seed is not None:
            row["seed"] = seed
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_random_seed_distribution(oos_summary: pd.DataFrame) -> pd.DataFrame:
    """Resume distribuição OOS do baseline aleatório multi-seed (estratégia ``random``)."""
    if oos_summary.empty or "seed" not in oos_summary.columns:
        return pd.DataFrame()

    df = oos_summary.loc[oos_summary["strategy"] == "random"].copy()
    if df.empty:
        return pd.DataFrame()

    metric_cols = ["oos_sharpe", "oos_total_return", "number_of_trades", "oos_max_drawdown"]
    rows: list[dict[str, Any]] = []
    for (strategy, symbol, interval), grp in df.groupby(["strategy", "symbol", "interval"]):
        row: dict[str, Any] = {
            "strategy": strategy,
            "symbol": symbol,
            "interval": interval,
            "n_seeds": len(grp),
        }
        for col in metric_cols:
            if col not in grp.columns:
                continue
            vals = grp[col].astype(float)
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=0))
            row[f"{col}_p05"] = float(vals.quantile(0.05))
            row[f"{col}_p50"] = float(vals.quantile(0.50))
            row[f"{col}_p95"] = float(vals.quantile(0.95))
        rows.append(row)

    return pd.DataFrame(rows)


def _merge_walk_forward_csv(
    new_df: pd.DataFrame,
    path: Path,
    key_cols: list[str],
) -> pd.DataFrame:
    """Substitui linhas com mesmas chaves e mantém estratégias de execuções anteriores."""
    if not path.exists() or new_df.empty:
        return new_df
    existing = pd.read_csv(path)
    if existing.empty:
        return new_df
    present_keys = new_df[key_cols].drop_duplicates()
    mask = pd.Series(True, index=existing.index)
    for col in key_cols:
        if col not in existing.columns:
            return pd.concat([existing, new_df], ignore_index=True)
    for _, key_row in present_keys.iterrows():
        row_mask = pd.Series(True, index=existing.index)
        for col in key_cols:
            row_mask &= existing[col] == key_row[col]
        mask &= ~row_mask
    kept = existing.loc[mask]
    return pd.concat([kept, new_df], ignore_index=True)


def _merge_walk_forward_equity(
    new_df: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    """Mescla equity OOS: substitui (strategy, symbol, interval, window_id) presentes no batch."""
    if not path.exists() or new_df.empty:
        return new_df
    existing = pd.read_parquet(path)
    if existing.empty:
        return new_df
    key_cols = ["strategy", "symbol", "interval", "window_id"]
    for col in key_cols:
        if col not in existing.columns:
            return pd.concat([existing, new_df], ignore_index=True)
    if "seed" in new_df.columns and "seed" in existing.columns:
        key_cols.append("seed")
    present = new_df[key_cols].drop_duplicates()
    mask = pd.Series(True, index=existing.index)
    for _, key_row in present.iterrows():
        row_mask = pd.Series(True, index=existing.index)
        for col in key_cols:
            row_mask &= existing[col] == key_row[col]
        mask &= ~row_mask
    kept = existing.loc[mask]
    return pd.concat([kept, new_df], ignore_index=True)


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
        if "seed" in r.best_params:
            row["seed"] = int(r.best_params["seed"])
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
        if "seed" in r.best_params:
            eq_df["seed"] = int(r.best_params["seed"])
        equity_frames.append(eq_df)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_keys = ["window_id", "strategy"]
    if "seed" in metrics_df.columns:
        metrics_keys.append("seed")
    metrics_merged = _merge_walk_forward_csv(
        metrics_df,
        tables_dir / "walk_forward_results.csv",
        metrics_keys,
    )
    metrics_merged.to_csv(tables_dir / "walk_forward_results.csv", index=False)

    params_df = pd.DataFrame(params_rows)
    params_keys = ["window_id", "strategy"]
    if "param_seed" in params_df.columns:
        params_keys.append("param_seed")
    params_merged = _merge_walk_forward_csv(
        params_df,
        tables_dir / "best_parameters_by_window.csv",
        params_keys,
    )
    params_merged.to_csv(tables_dir / "best_parameters_by_window.csv", index=False)

    oos_summary = aggregate_walk_forward_oos(all_results)
    oos_key_cols = ["strategy", "symbol", "interval"]
    if "seed" in oos_summary.columns:
        oos_key_cols.append("seed")
    oos_merged = _merge_walk_forward_csv(
        oos_summary,
        tables_dir / "walk_forward_oos_summary.csv",
        oos_key_cols,
    )
    oos_merged.to_csv(tables_dir / "walk_forward_oos_summary.csv", index=False)
    logger.info("Resumo OOS + CBE salvo: %d linhas.", len(oos_merged))

    seed_dist = aggregate_random_seed_distribution(oos_merged)
    if not seed_dist.empty:
        seed_dist.to_csv(tables_dir / "walk_forward_random_seed_distribution.csv", index=False)
        logger.info("Distribuição de seeds aleatórios salva: %d estratégias.", len(seed_dist))

    if equity_frames:
        equity_new = pd.concat(equity_frames, ignore_index=True)
        equity_merged = _merge_walk_forward_equity(
            equity_new,
            backtests_dir / "walk_forward_equity.parquet",
        )
        equity_merged.to_parquet(
            backtests_dir / "walk_forward_equity.parquet", index=False
        )

    logger.info("Resultados walk-forward salvos em %s", tables_dir)
