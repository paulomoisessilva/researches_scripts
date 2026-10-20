#!/usr/bin/env python3
"""Script 08 — Equivalência entre o motor de seleção fatiado e o de passagem única.

Para janelas amostradas, calcula a pontuação de validação de TODOS os combos do grid
pelos dois motores e compara: parâmetros escolhidos, pontuações combo a combo e o efeito
do estado inicial da estratégia (posição na 1ª barra da validação).

Uso:
    python scripts/08_check_engine_equivalence.py --config config.smoke.yaml --n-windows 19
    python scripts/08_check_engine_equivalence.py --config config.paper.yaml --n-windows 40
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.backtest import BacktestConfig
from src.config import load_config, resolve_date_range, resolve_strategies
from src.fast_select import compare_engines, summarize_comparison
from src.strategies import BAND_STRATEGIES
from src.utils import ensure_dir, setup_logger
from src.walk_forward import _get_wf_int, _prune_features_for_strategy, build_windows
from src.wf_runner import prepare_inputs, resolve_cost_scenarios, resolve_runs_root

logger = setup_logger("08_check_engine_equivalence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara motor fatiado x passagem única.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--symbol", type=str, default=None, help="Default: primeiro de data.symbols.")
    parser.add_argument("--interval", type=str, default=None, help="Default: primeiro de data.intervals.")
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--n-windows", type=int, default=40,
                        help="Janelas amostradas, espaçadas uniformemente no período OOS.")
    parser.add_argument("--costs", type=str, default=None, metavar="FEE:SLIP")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--train-days", type=int, default=None,
                        help="Sobrescreve walk_forward.train_days. Valores pequenos (ex.: 1) estressam o "
                             "estado inicial: o motor fatiado parte zerado poucas barras antes da validação.")
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.train_days is not None:
        cfg["walk_forward"]["train_days"] = args.train_days
    wf_cfg = cfg["walk_forward"]
    symbol = args.symbol or cfg["data"]["symbols"][0]
    interval = args.interval or cfg["data"]["intervals"][0]
    strategies = [s for s in resolve_strategies(cfg, args.strategies) if s in BAND_STRATEGIES]
    start_date, end_date = resolve_date_range(cfg)
    fee_bps, slip_bps = resolve_cost_scenarios(cfg, [args.costs] if args.costs else None)[0]
    workers = args.workers if args.workers is not None else int(wf_cfg.get("parallel_workers", 1))
    metric = wf_cfg.get("selection_metric", "net_sharpe")

    inputs = prepare_inputs(cfg, [symbol], [interval], strategies, start_date, end_date)
    df, df_feat = inputs[(symbol, interval)]
    windows = build_windows(
        df,
        train_days=_get_wf_int(wf_cfg, "train_days"),
        val_days=_get_wf_int(wf_cfg, "validation_days", "val_days"),
        test_days=_get_wf_int(wf_cfg, "test_days"),
        step_days=_get_wf_int(wf_cfg, "step_days"),
        oos_start=wf_cfg.get("oos_start"),
    )
    pick = np.unique(np.linspace(0, len(windows) - 1, min(args.n_windows, len(windows))).round().astype(int))
    sample = [windows[i] for i in pick]
    logger.info("%s %s | %d janelas (de %d) | %d workers | custo %g+%g bps | métrica %s",
                symbol, interval, len(sample), len(windows), workers, fee_bps, slip_bps, metric)

    bt_cfg = BacktestConfig(
        strategy_name="check", symbol=symbol, interval=interval,
        initial_capital=cfg["backtest"]["initial_capital"], fee_bps=fee_bps, slippage_bps=slip_bps,
        allow_short=cfg["backtest"]["allow_short"], max_position=cfg["backtest"]["max_position"],
        close_eod=wf_cfg.get("close_position_end_of_day", True),
    )

    out_dir = Path(args.out_dir) if args.out_dir else resolve_runs_root(cfg).parent / "engine_check"
    tag = f"_train{args.train_days}" if args.train_days is not None else ""
    ensure_dir(out_dir)

    summaries = []
    for strategy in strategies:
        t0 = time.time()
        cmp = compare_engines(
            _prune_features_for_strategy(df_feat, strategy), strategy, cfg, bt_cfg, sample, metric, workers
        )
        summary = {"strategy": strategy, **summarize_comparison(cmp), "seconds": round(time.time() - t0, 1)}
        summaries.append(summary)
        logger.info("%s: %s", strategy, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in summary.items()})

        detail = pd.DataFrame({
            "window_id": [w.window_id for w in sample],
            "val_start": [w.val_start for w in sample],
            "fast_best": [{**cmp.combos[i][0], **cmp.combos[i][1]} for i in cmp.fast_best],
            "sliced_best": [{**cmp.combos[i][0], **cmp.combos[i][1]} for i in cmp.sliced_best],
            "same_best": cmp.fast_best == cmp.sliced_best,
            "fast_best_score": cmp.fast_scores[cmp.fast_best, np.arange(len(sample))],
            "sliced_best_score": cmp.sliced_scores[cmp.sliced_best, np.arange(len(sample))],
        })
        detail.to_csv(out_dir / f"engine_check_{strategy}_{symbol}_{interval}{tag}.csv", index=False)

    table = pd.DataFrame(summaries)
    table.to_csv(out_dir / f"engine_check_summary_{symbol}_{interval}{tag}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(table.T.to_string())
    logger.info("Resultados em %s", out_dir)


if __name__ == "__main__":
    main()
