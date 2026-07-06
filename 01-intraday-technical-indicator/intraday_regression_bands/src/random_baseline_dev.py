"""Utilitários dev-only para baseline aleatório matched (fora do pipeline principal).

Não usado pelo walk-forward padrão. Ative apenas com
``random_baseline.enable_matched_random: true`` em configs experimentais.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .metrics import compute_metrics
from .strategies import STRATEGY_REGISTRY, resolve_random_seeds, strategy_random_matched
from .utils import setup_logger
from .walk_forward import (
    WFResult,
    WFWindow,
    _slice_backtest_result,
    _slice_df,
    optimize_params_on_validation,
)

logger = setup_logger(__name__)


def random_matched_strategy_name(reference: str) -> str:
    return f"random_matched_{reference}"


def parse_random_matched_strategy(strategy_name: str) -> str | None:
    prefix = "random_matched_"
    if strategy_name.startswith(prefix):
        return strategy_name[len(prefix):]
    return None


def expand_random_strategies_dev(strategies: list[str], cfg: dict) -> list[str]:
    """Expande random em variantes matched — somente se explicitamente habilitado."""
    rb = cfg.get("random_baseline", {})
    if not rb.get("enable_matched_random", False):
        return list(strategies)

    matched = rb.get("matched_strategies", [])
    out: list[str] = []
    for name in strategies:
        if name == "random":
            out.append("random")
            for ref in matched:
                if ref not in STRATEGY_REGISTRY:
                    logger.warning("matched_strategies: '%s' ignorada.", ref)
                    continue
                out.append(random_matched_strategy_name(ref))
        else:
            out.append(name)
    return out


def run_walk_forward_random_matched(
    df: pd.DataFrame,
    reference_strategy: str,
    cfg: dict,
    symbol: str,
    interval: str,
    windows: list[WFWindow],
    df_feat_full: pd.DataFrame,
    bt_cfg_template: BacktestConfig,
    wf_cfg: dict,
    n_workers: int,
) -> list[WFResult]:
    """Walk-forward aleatório matched — dev-only, não parte da metodologia principal."""
    seeds = resolve_random_seeds(cfg)
    strategy_name = random_matched_strategy_name(reference_strategy)
    ref_func = STRATEGY_REGISTRY[reference_strategy]
    selection_metric = wf_cfg.get("selection_metric", "net_sharpe")
    results: list[WFResult] = []

    for win in windows:
        df_feat_train_val = _slice_df(df_feat_full, win.train_start, win.val_end)
        df_feat_test_context = _slice_df(df_feat_full, win.train_start, win.test_end)

        min_test_bars = 1 if wf_cfg.get("test_days", 1) == 1 else 10
        if len(df_feat_train_val) < 50 or len(df_feat_test_context) < min_test_bars:
            continue

        ref_params, _ = optimize_params_on_validation(
            df_train_val=df,
            strategy_name=reference_strategy,
            cfg=cfg,
            bt_cfg_template=BacktestConfig(
                strategy_name=reference_strategy,
                symbol=symbol,
                interval=interval,
                initial_capital=bt_cfg_template.initial_capital,
                fee_bps=bt_cfg_template.fee_bps,
                slippage_bps=bt_cfg_template.slippage_bps,
                allow_short=bt_cfg_template.allow_short,
                max_position=bt_cfg_template.max_position,
                close_eod=bt_cfg_template.close_eod,
            ),
            val_start=win.val_start,
            val_end=win.val_end,
            selection_metric=selection_metric,
            df_feat_train_val=df_feat_train_val,
            n_workers=n_workers,
        )

        try:
            ref_signal = ref_func(df_feat_test_context, **ref_params)
        except Exception as exc:
            logger.warning("[WF %d] %s ref signal falhou: %s", win.window_id, strategy_name, exc)
            continue

        for seed in seeds:
            params: dict[str, Any] = {
                "seed": seed,
                "reference_strategy": reference_strategy,
                **ref_params,
            }
            try:
                signal = strategy_random_matched(
                    df_feat_test_context, ref_signal, seed=seed
                )
                test_result_full = run_backtest(df_feat_test_context, signal, bt_cfg_template)
                test_result = _slice_backtest_result(
                    test_result_full, win.test_start, win.test_end
                )
            except Exception as exc:
                logger.warning(
                    "[WF %d] %s seed=%d erro: %s", win.window_id, strategy_name, seed, exc
                )
                continue

            test_metrics = compute_metrics(
                test_result.equity,
                test_result.returns,
                test_result.positions,
                test_result.trades,
                interval=interval,
                strategy_name=strategy_name,
                symbol=symbol,
                fee_bps=bt_cfg_template.fee_bps,
                slippage_bps=bt_cfg_template.slippage_bps,
                gross_returns=test_result.gross_returns,
            )

            results.append(WFResult(
                window_id=win.window_id,
                strategy=strategy_name,
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

    logger.info(
        "%s (dev-only): %d seeds, %d janelas → %d resultados.",
        strategy_name, len(seeds), len(windows), len(results),
    )
    return results
