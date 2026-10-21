#!/usr/bin/env python3
"""Script 15 — Diagnóstico DESCRITIVO de parâmetros fixos (janela x k) em 5 anos e 3 ativos, em 1 minuto.

Versão em blocos semestrais do script 12 (o DataFrame de features de um ano de 1 minuto não cabe
em vários processos). Cada combinação (estratégia, janela, k) do grid do config é avaliada como
estratégia de parâmetro fixo; as séries diárias de retorno bruto, unidades negociadas e operações
de todos os blocos são concatenadas por ativo.

É exploratório: olha todas as combinações no mesmo período, então a melhor é otimista por
construção. Como correção, o Reality Check/SPA roda sobre o conjunto inteiro de candidatas
(por ativo e para os 3 ativos juntos), em P1 (2021-01-06 a 2024-12-31) e em P2 (período completo).

Saídas em ``results/full_1m/fixed_params/``:
  ``daily_<SYMBOL>.npz``      séries diárias (dias x combos);
  ``fixed_<SYMBOL>.csv``      por combo e período (ALL e um por ano);
  ``fixed_spa.csv``           RC/SPA por família, período e custo;
  ``fixed_summary.md``.

Uso:
    python scripts/15_fixed_params_full_1m.py --config config.full_1m.yaml --workers 8
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.backtest import BacktestConfig
from src.config import get_project_root, load_config, resolve_date_range, resolve_strategies
from src.fixed_params import evaluate_chunk, summarize
from src.full_study import make_chunks
from src.inference import bootstrap_means, net_returns_at_cost, reality_check_spa, stationary_bootstrap_indices
from src.strategies import BAND_STRATEGIES
from src.utils import ensure_dir, setup_logger
from src.wf_analysis import to_markdown
from src.wf_runner import prepare_inputs, resolve_cost_scenarios, resolve_runs_root

logger = setup_logger("15_fixed_params_full_1m")

P1_END = "2024-12-31"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnóstico de parâmetros fixos, 5 anos, em blocos.")
    parser.add_argument("--config", type=str, default="config.full_1m.yaml")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--history-days", type=int, default=7)
    parser.add_argument("--costs", nargs="+", type=float, default=[0, 1, 2, 3, 4, 6, 8])
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--block", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--chunks", nargs="+", default=None, help="Rótulos (para teste rápido).")
    return parser.parse_args()


def run_symbol(base: dict, symbol: str, interval: str, strategies: list[str], chunks: list[dict],
               bt_cfg: BacktestConfig, workers: int, out_file: Path) -> dict:
    """Roda todos os blocos de um ativo e devolve as matrizes diárias concatenadas."""
    if out_file.exists():
        z = np.load(out_file, allow_pickle=True)
        logger.info("%s: reaproveitando %s", symbol, out_file.name)
        return {"days": pd.DatetimeIndex(z["days"]).tz_localize("UTC"), "combos": [tuple(c) for c in z["combos"].tolist()],
                "gross": z["gross"], "units": z["units"], "trades": z["trades"]}

    parts = []
    for chunk in chunks:
        t0 = time.time()
        cfg = copy.deepcopy(base)
        cfg["data"]["start_date"], cfg["data"]["end_date"] = chunk["start_date"], chunk["end_date"]
        cfg["walk_forward"]["oos_start"] = chunk["oos_start"]
        start_date, end_date = resolve_date_range(cfg)
        _, df_feat = prepare_inputs(cfg, [symbol], [interval], strategies, start_date, end_date)[(symbol, interval)]
        ts = pd.DatetimeIndex(df_feat["timestamp"])
        start = pd.Timestamp(chunk["oos_start"]).tz_localize(ts.tz)
        end = ts[-1].normalize()  # só dias completos
        res = evaluate_chunk(df_feat, strategies, cfg, bt_cfg, start, end, workers=workers)
        parts.append(res)
        logger.info("%s %s: %d dias, %d combos em %.1f min", symbol, chunk["label"], len(res["days"]),
                    len(res["combos"]), (time.time() - t0) / 60)
        del df_feat

    combos = parts[0]["combos"]
    assert all(p["combos"] == combos for p in parts), "grid diferente entre blocos"
    out = {
        "days": parts[0]["days"].append([p["days"] for p in parts[1:]]),
        "combos": combos,
        "gross": np.vstack([p["gross"] for p in parts]),
        "units": np.vstack([p["units"] for p in parts]),
        "trades": np.vstack([p["trades"] for p in parts]),
    }
    np.savez_compressed(out_file, days=out["days"].strftime("%Y-%m-%d").to_numpy(), combos=np.array(combos, dtype=object),
                        gross=out["gross"], units=out["units"], trades=out["trades"])
    return out


def spa_rows(days: pd.DatetimeIndex, d_by_cost: dict[float, np.ndarray], family: str, period: str,
             reps: int, block: float, seed: int, labels: list[str]) -> list[dict]:
    idx = stationary_bootstrap_indices(len(days), reps, block, np.random.default_rng(seed))
    rows = []
    for cost, d in d_by_cost.items():
        r = reality_check_spa(d, bootstrap_means(d, idx))
        rows.append({
            "family": family, "period": period, "cost_bps_per_side": cost, "n_candidates": r.n_candidates,
            "n_days": r.n_obs, "best_candidate": labels[r.best_candidate],
            "best_mean_bps_per_day": r.best_mean * 1e4, "rc_pvalue": r.rc_pvalue,
            "spa_pvalue_lower": r.spa_pvalue_lower, "spa_pvalue_consistent": r.spa_pvalue_consistent,
            "spa_pvalue_upper": r.spa_pvalue_upper,
        })
    return rows


def main() -> None:
    args = parse_args()
    root = get_project_root()
    base = load_config(args.config)
    symbols = args.symbols or base["data"]["symbols"]
    interval = base["data"]["intervals"][0]
    strategies = [s for s in resolve_strategies(base) if s in BAND_STRATEGIES]
    fee, slip = resolve_cost_scenarios(base)[0]
    run_cost = fee + slip
    wf = base["walk_forward"]
    chunks = make_chunks(date.fromisoformat(base["data"]["start_date"]), date.fromisoformat(base["data"]["end_date"]),
                         date.fromisoformat(wf["oos_start"]), args.history_days)
    if args.chunks:
        chunks = [c for c in chunks if c["label"] in args.chunks]
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else resolve_runs_root(base).parent / "fixed_params")

    all_data: dict[str, dict] = {}
    for symbol in symbols:
        bt_cfg = BacktestConfig(
            strategy_name="fixed", symbol=symbol, interval=interval,
            initial_capital=base["backtest"]["initial_capital"], fee_bps=fee, slippage_bps=slip,
            allow_short=base["backtest"]["allow_short"], max_position=base["backtest"]["max_position"],
            close_eod=wf.get("close_position_end_of_day", True),
        )
        data = run_symbol(base, symbol, interval, strategies, chunks, bt_cfg, args.workers, out_dir / f"daily_{symbol}.npz")
        all_data[symbol] = data
        for name, end in (("P1", P1_END), ("P2", None)):
            t = summarize(data["days"], data["combos"], data["gross"], data["units"], data["trades"], run_cost, end=end)
            t.insert(0, "symbol", symbol)
            t.insert(1, "window_set", name)
            t.to_csv(out_dir / f"fixed_{symbol}_{name}.csv", index=False)

    # --- Reality Check / SPA sobre as candidatas de cada ativo e de todos juntos ---
    spa: list[dict] = []
    days_common = None
    for symbol, data in all_data.items():
        days_common = data["days"] if days_common is None else days_common.intersection(data["days"])
    for period, end in (("P1", P1_END), ("P2", None)):
        days = days_common if end is None else days_common[days_common <= pd.Timestamp(end, tz=days_common.tz)]
        blocks, labels = [], []
        for symbol, data in all_data.items():
            pos = data["days"].get_indexer(days)
            g, u = data["gross"][pos], data["units"][pos]
            lab = [f"{symbol}|{s}|w{w}|k{k:g}" for s, w, k in data["combos"]]
            by_cost = {c: net_returns_at_cost(g, u, c) for c in args.costs}
            spa += spa_rows(days, by_cost, symbol, period, args.reps, args.block, args.seed, lab)
            blocks.append(by_cost)
            labels += lab
        pooled = {c: np.hstack([b[c] for b in blocks]) for c in args.costs}
        spa += spa_rows(days, pooled, "todas", period, args.reps, args.block, args.seed, labels)
    spa_df = pd.DataFrame(spa)
    spa_df.to_csv(out_dir / "fixed_spa.csv", index=False)

    # --- resumo ---
    md = [
        "# Diagnóstico de parâmetros fixos, 1 minuto, 5 anos (descritivo)\n",
        f"{len(symbols)} ativos, {len(all_data[symbols[0]]['combos'])} combinações por ativo, custo {fee:g}+{slip:g} bps por lado, "
        f"fechamento diário UTC. P1 = até {P1_END}; P2 = período completo.\n",
        "> **Ressalva de data snooping:** todas as combinações foram avaliadas no MESMO período. A melhor delas é "
        "otimista por construção. O RC/SPA abaixo corrige isso para a família inteira; nada aqui foi confirmado "
        "fora da amostra.\n",
    ]
    for symbol in symbols:
        t = pd.read_csv(out_dir / f"fixed_{symbol}_P2.csv")
        allp = t[t.period == "ALL"]
        md += [f"## {symbol}: 10 combinações com maior retorno bruto (P2)\n",
               to_markdown(allp.sort_values("gross_pct", ascending=False).head(10)
                           .set_index(["strategy", "window", "k"])[["gross_pct", "net_pct", "cbe_bps", "trades_per_day"]],
                           floatfmt="{:.2f}"), ""]
        yearly = t[t.period != "ALL"].groupby("period").agg(
            share_gross_positive=("gross_pct", lambda s: float((s > 0).mean())),
            max_cbe_bps=("cbe_bps", "max"), median_cbe_bps=("cbe_bps", "median"))
        md += [f"## {symbol}: fração de combinações com retorno bruto > 0, por ano\n", to_markdown(yearly, floatfmt="{:.2f}"), ""]
    for period in ("P1", "P2"):
        s = spa_df[spa_df.period == period]
        md += [f"## SPA consistente por família, {period}\n",
               to_markdown(s.pivot(index="family", columns="cost_bps_per_side", values="spa_pvalue_consistent"), floatfmt="{:.3f}"), ""]
        md += [f"## RC por família, {period}\n",
               to_markdown(s.pivot(index="family", columns="cost_bps_per_side", values="rc_pvalue"), floatfmt="{:.3f}"), ""]
    (out_dir / "fixed_summary.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("Resultados em %s", out_dir)


if __name__ == "__main__":
    main()
