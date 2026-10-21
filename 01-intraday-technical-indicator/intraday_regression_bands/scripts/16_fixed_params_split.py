#!/usr/bin/env python3
"""Script 16 — Persistência fora da amostra dos parâmetros fixos (exploratório, pós-hoc).

Usa as séries diárias do script 15 (``results/full_1m/fixed_params/daily_<SYMBOL>.npz``). Divide
o período em S (antes de ``--split``) e T (a partir dele) e pergunta: as combinações (janela x k)
que ganhariam líquido em S continuam ganhando em T? É uma checagem de persistência, não um teste
corrigido: a escolha da divisão foi feita depois de ver os resultados completos.

Uso:
    python scripts/16_fixed_params_split.py --config config.full_1m.yaml --split 2024-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.config import load_config
from src.fixed_params import split_persistence
from src.inference import bootstrap_means, net_returns_at_cost, reality_check_spa, stationary_bootstrap_indices
from src.utils import setup_logger
from src.wf_analysis import to_markdown
from src.wf_runner import resolve_cost_scenarios, resolve_runs_root

logger = setup_logger("16_fixed_params_split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistência de parâmetros fixos (exploratório).")
    parser.add_argument("--config", type=str, default="config.full_1m.yaml")
    parser.add_argument("--split", type=str, default="2024-01-01")
    parser.add_argument("--min-trades", type=int, default=30, help="Operações mínimas em S para o combo ser elegível.")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--block", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    fee, slip = resolve_cost_scenarios(cfg)[0]
    out_dir = resolve_runs_root(cfg).parent / "fixed_params"
    pct = lambda x: (np.exp(x) - 1.0) * 100.0  # noqa: E731

    summary, t_series, spa_rows = [], {}, []
    md = [
        "# Persistência fora da amostra de parâmetros fixos (exploratório)\n",
        f"S = antes de {args.split}; T = a partir de {args.split}. Custo {fee:g}+{slip:g} bps por lado. "
        f"Elegível = pelo menos {args.min_trades} operações em S. **Pós-hoc:** a divisão foi escolhida depois de "
        "ver os resultados do período completo; não corrige a busca múltipla.\n",
    ]
    for symbol in cfg["data"]["symbols"]:
        z = np.load(out_dir / f"daily_{symbol}.npz", allow_pickle=True)
        days = pd.DatetimeIndex(z["days"]).tz_localize("UTC")
        combos = [tuple(c) for c in z["combos"].tolist()]
        t = split_persistence(days, combos, z["gross"], z["units"], z["trades"], fee + slip, args.split, args.min_trades)
        t["net_pct_S"], t["net_pct_T"] = pct(t["net_S"]), pct(t["net_T"])
        t.insert(0, "symbol", symbol)
        t.to_csv(out_dir / f"split_{symbol}.csv", index=False)

        el = t[t["elegivel"]]
        pos = el[el["net_S"] > 0]
        # família escolhida só com S (líquido > 0 e elegível), avaliada em T
        mask_t = days >= pd.Timestamp(args.split, tz=days.tz)
        idx = [combos.index((r.strategy, r.window, r.k)) for r in pos.itertuples()]
        if idx:
            t_series[symbol] = net_returns_at_cost(z["gross"][mask_t][:, idx], z["units"][mask_t][:, idx], fee + slip)
        row = {
            "symbol": symbol, "combos": len(t), "eligible": len(el), "net_pos_S": len(pos),
            "of_those_net_pos_T": int((pos["net_T"] > 0).sum()),
            "mean_net_pct_T_of_those": float(pos["net_pct_T"].mean()) if len(pos) else float("nan"),
            "median_net_pct_T_of_those": float(pos["net_pct_T"].median()) if len(pos) else float("nan"),
            "spearman_net_S_vs_T": float(el["net_S"].rank().corr(el["net_T"].rank())) if len(el) > 2 else float("nan"),
        }
        summary.append(row)
        top = el.sort_values("net_S", ascending=False).head(args.top)
        md += [f"## {symbol}: {args.top} melhores elegíveis em S e o que aconteceu em T\n",
               to_markdown(top.set_index(["strategy", "window", "k"])[
                   ["net_pct_S", "cbe_S", "trades_S", "net_pct_T", "cbe_T", "trades_T"]], floatfmt="{:.2f}"), ""]
    fams = {s_: m for s_, m in t_series.items()}
    if fams:
        fams["todas"] = np.hstack(list(t_series.values()))
    for name, d in fams.items():
        if d.shape[0] < 30:
            continue
        boot = stationary_bootstrap_indices(d.shape[0], args.reps, args.block, np.random.default_rng(args.seed))
        r = reality_check_spa(d, bootstrap_means(d, boot))
        spa_rows.append({"family": name, "n_candidates": r.n_candidates, "n_days_T": r.n_obs,
                         "best_mean_bps_per_day": r.best_mean * 1e4, "rc_pvalue": r.rc_pvalue,
                         "spa_pvalue_consistent": r.spa_pvalue_consistent})
    spa = pd.DataFrame(spa_rows)
    spa.to_csv(out_dir / "split_spa_T.csv", index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(out_dir / "split_summary.csv", index=False)
    md = md[:2] + ["## Resumo\n", to_markdown(sm.set_index("symbol"), floatfmt="{:.2f}"), ""] + md[2:]
    (out_dir / "split_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
