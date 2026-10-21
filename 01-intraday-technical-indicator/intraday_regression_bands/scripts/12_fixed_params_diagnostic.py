#!/usr/bin/env python3
"""Script 12 — Diagnóstico descritivo de parâmetros FIXOS (janela x k), sem seleção diária.

Avalia cada par (estratégia, janela, k) do grid do config como uma estratégia de parâmetro fixo
na série inteira do período, para separar "o sinal não tem edge" de "a seleção diária
sobreajusta". É **descritivo**: olha todas as combinações no mesmo período, então a melhor
delas é otimista por construção (data snooping). Como correção, roda Reality Check/SPA sobre o
conjunto inteiro de candidatas (retornos diários líquidos, bootstrap estacionário).

Uso:
    python scripts/12_fixed_params_diagnostic.py --config config.pilot_1m.yaml
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.backtest import BacktestConfig, run_backtest
from src.config import load_config, resolve_date_range, resolve_strategies
from src.fast_select import band_grid
from src.inference import bootstrap_means, net_returns_at_cost, reality_check_spa, stationary_bootstrap_indices
from src.metrics import annualization_factor
from src.strategies import BAND_STRATEGIES, compute_band_z, signal_from_z
from src.utils import ensure_dir, setup_logger
from src.wf_analysis import to_markdown
from src.wf_runner import prepare_inputs, resolve_cost_scenarios, resolve_runs_root

logger = setup_logger("12_fixed_params_diagnostic")

_state: dict = {}


def _init(df_feat: pd.DataFrame, bt_cfg: BacktestConfig, eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> None:
    _state.update(df=df_feat, bt=bt_cfg, start=eval_start, end=eval_end)


def _eval(task: tuple[str, dict, dict]) -> dict:
    """Backtest contínuo de uma combinação; devolve séries diárias bruta/unidades e contagem de trades."""
    strategy, struct, thresh = task
    st = _state
    z = compute_band_z(st["df"], strategy, struct["window"], struct["k"])
    signal = signal_from_z(z, name=strategy, **thresh)
    res = run_backtest(st["df"], signal, st["bt"])

    ts = pd.DatetimeIndex(st["df"]["timestamp"])
    mask = (ts >= st["start"]) & (ts < st["end"])
    day = ts[mask].normalize()
    gross = pd.Series(res.gross_returns.to_numpy()[mask], index=day).groupby(level=0).sum()
    units = pd.Series(res.trade_units.to_numpy()[mask], index=day).groupby(level=0).sum()
    n_trades = 0
    if not res.trades.empty:
        exit_ts = pd.to_datetime(res.trades["exit_timestamp"], utc=True)
        n_trades = int(((exit_ts >= st["start"]) & (exit_ts < st["end"])).sum())
    return {
        "strategy": strategy, "window": struct["window"], "k": struct["k"],
        "gross": gross, "units": units, "n_trades": n_trades,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnóstico de parâmetros fixos (janela x k).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--interval", type=str, default=None)
    parser.add_argument("--costs", nargs="+", type=float, default=[0, 1, 2, 3, 4, 6, 8])
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--block", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    wf = cfg["walk_forward"]
    symbol = args.symbol or cfg["data"]["symbols"][0]
    interval = args.interval or cfg["data"]["intervals"][0]
    strategies = [s for s in resolve_strategies(cfg) if s in BAND_STRATEGIES]
    start_date, end_date = resolve_date_range(cfg)
    fee, slip = resolve_cost_scenarios(cfg)[0]
    run_cost = fee + slip

    df, df_feat = prepare_inputs(cfg, [symbol], [interval], strategies, start_date, end_date)[(symbol, interval)]
    ts = pd.DatetimeIndex(df_feat["timestamp"])
    eval_start = pd.Timestamp(wf["oos_start"]).tz_localize(ts.tz)
    eval_end = ts[-1].normalize()  # só dias completos
    logger.info("%s %s | dias avaliados %s -> %s | custo %g+%g bps", symbol, interval,
                eval_start.date(), (eval_end - pd.Timedelta(days=1)).date(), fee, slip)

    bt_cfg = BacktestConfig(
        strategy_name="fixed", symbol=symbol, interval=interval,
        initial_capital=cfg["backtest"]["initial_capital"], fee_bps=fee, slippage_bps=slip,
        allow_short=cfg["backtest"]["allow_short"], max_position=cfg["backtest"]["max_position"],
        close_eod=wf.get("close_position_end_of_day", True),
    )
    tasks = [(s, struct, thresh) for s in strategies for struct, thresh in band_grid(s, cfg)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(df_feat, bt_cfg, eval_start, eval_end)) as ex:
        results = list(ex.map(_eval, tasks, chunksize=2))

    days = results[0]["gross"].index
    gross = np.column_stack([r["gross"].reindex(days).fillna(0.0).to_numpy() for r in results])
    units = np.column_stack([r["units"].reindex(days).fillna(0.0).to_numpy() for r in results])
    n_days = len(days)
    months = days.strftime("%Y-%m")

    rows = []
    for j, r in enumerate(results):
        net = net_returns_at_cost(gross[:, j], units[:, j], run_cost)
        g_sum, u_sum = gross[:, j].sum(), units[:, j].sum()
        cbe = 0.0 if g_sum <= 0 else (g_sum * 1e4 / u_sum if u_sum > 0 else np.nan)
        monthly_gross = pd.Series(gross[:, j]).groupby(months).sum()
        sd = net.std(ddof=1)
        rows.append({
            "strategy": r["strategy"], "window": r["window"], "k": r["k"],
            "gross_pct": (np.exp(g_sum) - 1) * 100,
            "net_pct": (np.exp(net.sum()) - 1) * 100,
            "cbe_bps": cbe,
            "trades_per_day": r["n_trades"] / n_days,
            "daily_sharpe_net": net.mean() / sd * np.sqrt(365) if sd > 0 else 0.0,
            "months_gross_positive": int((monthly_gross > 0).sum()),
            "n_months": int(len(monthly_gross)),
            "gross_first_half_pct": (np.exp(gross[: n_days // 2, j].sum()) - 1) * 100,
            "gross_second_half_pct": (np.exp(gross[n_days // 2:, j].sum()) - 1) * 100,
        })
    table = pd.DataFrame(rows)

    # RC/SPA sobre TODAS as candidatas (correção para o fato de olhar o grid inteiro)
    idx = stationary_bootstrap_indices(n_days, args.reps, args.block, np.random.default_rng(args.seed))
    spa_rows = []
    for cost in args.costs:
        d = net_returns_at_cost(gross, units, cost)
        r = reality_check_spa(d, bootstrap_means(d, idx))
        b = r.best_candidate
        spa_rows.append({
            "cost_bps_per_side": cost, "n_candidates": r.n_candidates,
            "best": f"{results[b]['strategy']}|w{results[b]['window']}|k{results[b]['k']}",
            "best_mean_bps_per_day": r.best_mean * 1e4,
            "rc_pvalue": r.rc_pvalue, "spa_pvalue_consistent": r.spa_pvalue_consistent,
        })
    spa = pd.DataFrame(spa_rows)

    out_dir = Path(args.out_dir) if args.out_dir else resolve_runs_root(cfg).parent / "diagnostic"
    ensure_dir(out_dir)
    table.to_csv(out_dir / "fixed_params.csv", index=False)
    spa.to_csv(out_dir / "fixed_params_spa.csv", index=False)

    md = [
        "# Diagnóstico de parâmetros fixos (descritivo)\n",
        f"{symbol} {interval}, {days[0].date()} a {days[-1].date()} ({n_days} dias), custo {fee:g}+{slip:g} bps, "
        f"fechamento diário: {'sim' if bt_cfg.close_eod else 'não'}. {len(results)} candidatas.\n",
        "> **Ressalva de data snooping:** todas as combinações foram avaliadas no MESMO período. A melhor delas é "
        "otimista por construção. O Reality Check/SPA abaixo corrige isso para a família inteira; nada aqui foi "
        "confirmado fora da amostra.\n",
    ]
    for strat in strategies:
        t = table[table.strategy == strat]
        for col, label, fmt in (("cbe_bps", "CBE total (bps por lado)", "{:.2f}"),
                                ("gross_pct", "Retorno bruto (%)", "{:.1f}"),
                                ("net_pct", f"Retorno líquido a {fee + slip:g} bps (%)", "{:.1f}")):
            pv = t.pivot(index="window", columns="k", values=col)
            md += [f"## {strat} — {label}: janela (linhas) x k (colunas)\n", to_markdown(pv, floatfmt=fmt), ""]
    top = table.sort_values("gross_pct", ascending=False).head(10)
    md += ["## 10 combinações com maior retorno bruto\n", to_markdown(top.set_index(["strategy", "window", "k"]), floatfmt="{:.2f}"), ""]
    md += ["## Estatísticas do grid\n", to_markdown(pd.DataFrame({
        "fração com bruto > 0": [(table.gross_pct > 0).mean()],
        "fração com CBE >= custo": [(table.cbe_bps >= run_cost).mean()],
        "maior CBE (bps)": [table.cbe_bps.max()],
        "mediana trades/dia": [table.trades_per_day.median()],
    }).set_index(pd.Index(["grid"])), floatfmt="{:.3f}"), ""]
    md += [f"## Reality Check/SPA sobre as {len(results)} candidatas (retorno diário líquido por custo)\n",
           to_markdown(spa.set_index("cost_bps_per_side"), floatfmt="{:.3f}"), ""]
    (out_dir / "fixed_params.md").write_text("\n".join(md), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(strategies), 2, figsize=(12, 3.6 * len(strategies)), squeeze=False)
        for i, strat in enumerate(strategies):
            t = table[table.strategy == strat]
            for j, (col, label) in enumerate((("gross_pct", "Retorno bruto (%)"), ("cbe_bps", "CBE (bps por lado)"))):
                pv = t.pivot(index="window", columns="k", values=col)
                ax = axes[i][j]
                im = ax.imshow(pv.values, aspect="auto", cmap="RdYlGn", vmin=-abs(pv.values).max(), vmax=abs(pv.values).max()) \
                    if col == "gross_pct" else ax.imshow(pv.values, aspect="auto", cmap="viridis")
                ax.set_xticks(range(len(pv.columns)), [f"{c:g}" for c in pv.columns])
                ax.set_yticks(range(len(pv.index)), pv.index)
                ax.set_title(f"{strat} — {label}")
                ax.set_xlabel("k"); ax.set_ylabel("janela (barras)")
                fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "fixed_params_heatmaps.png", dpi=130)
    except Exception as exc:  # gráfico é opcional
        logger.warning("Heatmap não gerado: %s", exc)

    logger.info("Resultados em %s", out_dir)
    pd.set_option("display.width", 200)
    print(spa.round(3).to_string(index=False))
    print(top.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
