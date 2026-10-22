#!/usr/bin/env python3
"""Script 22 — Ganhos, perdas e regras de saída sobre as operações do estudo v2 (descritivo).

Lê ``results/v2/analysis/trades.parquet`` (script 20) e as barras de ``data/processed_v2`` e grava em ``results/v2/analysis/exits/``:

* ``win_loss_by_group.csv``     acerto, ganho e perda médios e medianos, razão ganho/perda, cauda e duração, por ativo x frequência x estratégia;
* ``win_loss_deciles.csv``      cortes de decis do tamanho dos ganhos e das perdas, por frequência x estratégia (ativos somados);
* ``sorted_decile_share.csv``   quanto cada decil de operações (do pior ao melhor) pesa no resultado;
* ``excursions_summary.csv``    MAE dos ganhos e MFE das perdas (para onde o preço andou antes da saída);
* ``rules_by_group.csv``        média por operação e fração de saídas antecipadas de cada regra e combinação de regras, por grupo,
                                inclusive por tercil de volume da última hora;
* ``rules_summary.md``          resumo por frequência x estratégia.

Regras: stop loss, stop gain, trailing, break-even, parada por tempo, stop e alvo por ATR (da entrada e da barra) e combinações. Hipóteses em ``src/v2_exits.py``:
saída no nível (ou na abertura após salto), sem derrapagem, sem taxa e sem reentrada; é uma descrição do que a amostra teria feito.

Uso: python scripts/22_exits_v2.py
"""

import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.utils import setup_logger
from src.v2_behavior import sorted_decile_contribution, tail_deciles, win_loss_summary
from src.v2_exits import Rule, bar_indices, bars_from_frame, combine, excursions, rule_outcomes, trade_return
from src.v2_study import markdown_table
from src.v2_trades import BAR_MINUTES, OOS_START, market_features

logger = setup_logger("22_exits_v2")
ROOT = Path(__file__).parent.parent
SL = [5, 10, 20, 30, 50, 100, 200, 300]
TP = SL
TRAIL = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300]
TRAIL_COMBO = [5, 10, 20, 30, 50]
BE = [5, 10, 20, 30, 50, 75, 100, 150, 200]
TIME_MIN = [10, 20, 30, 60, 120, 240, 480, 960, 1440]
TIME_COMBO_MIN = [30, 60, 120, 240]
ATR_K = [1, 2, 3, 5, 8, 12, 20, 30]
ATR_K_COMBO = [1, 2, 3, 5]


def time_bars_for(bar_minutes: int) -> dict[int, int]:
    """Minutos efetivos -> número de barras (dois pedidos que arredondam para o mesmo número de barras viram uma regra só)."""
    return {nb * bar_minutes: nb for nb in sorted({max(1, int(np.ceil(m / bar_minutes))) for m in TIME_MIN})}


def combos_for(time_bars: dict[int, int]) -> dict[str, list[Rule]]:
    c: dict[str, list[Rule]] = {"base": []}
    for s in SL:
        c[f"SL{s}"] = [Rule("SL", s)]
    for t in TP:
        c[f"TP{t}"] = [Rule("TP", t)]
    for s, t in itertools.product(SL, TP):
        c[f"SL{s}+TP{t}"] = [Rule("SL", s), Rule("TP", t)]
    for d in TRAIL:
        c[f"TR{d}"] = [Rule("TR", d)]
    for x in BE:
        c[f"BE{x}"] = [Rule("BE", x)]
    for k in ATR_K:
        c[f"ASL{k}"] = [Rule("ASL", k)]
        c[f"DSL{k}"] = [Rule("DSL", k)]
        c[f"DTR{k}"] = [Rule("DTR", k)]
    for k, m in itertools.product(ATR_K_COMBO, ATR_K_COMBO):
        c[f"ASL{k}+ATP{m}"] = [Rule("ASL", k), Rule("ATP", m)]
    for minutes, nb in time_bars.items():
        tm = Rule("TIME", nb)
        c[f"TIME{minutes}"] = [tm]
        if minutes not in TIME_COMBO_MIN:
            continue
        for d in TRAIL_COMBO:
            c[f"TIME{minutes}+TR{d}"] = [tm, Rule("TR", d)]
        for s, t in itertools.product([20, 50, 100], [20, 50, 100]):
            c[f"TIME{minutes}+SL{s}+TP{t}"] = [tm, Rule("SL", s), Rule("TP", t)]
    for d, t in itertools.product([10, 20, 50], [20, 50, 100]):
        c[f"TR{d}+TP{t}"] = [Rule("TR", d), Rule("TP", t)]
    return c


def main() -> None:
    t0 = time.time()
    out = ROOT / "results" / "v2" / "analysis" / "exits"
    out.mkdir(parents=True, exist_ok=True)
    trades = pd.read_parquet(ROOT / "results" / "v2" / "analysis" / "trades.parquet")
    trades["strategy"] = trades["strategy"].astype(str); trades["side"] = trades["side"].astype(str)

    win_rows, dec_w, dec_l, sorted_rows = [], [], [], []
    for (freq, strategy), g in trades.groupby(["freq", "strategy"]):
        for asset, ga in g.groupby("asset"):
            win_rows.append({"freq": freq, "asset": asset, "strategy": strategy, **win_loss_summary(ga["gross_bps"].to_numpy(), ga["holding_min"].to_numpy())})
        w, l = tail_deciles(g["gross_bps"].to_numpy())
        dec_w.append({"freq": freq, "strategy": strategy, "side": "wins", **w.to_dict()}); dec_l.append({"freq": freq, "strategy": strategy, "side": "losses", **l.to_dict()})
        sorted_rows.append({"freq": freq, "strategy": strategy, **{f"D{i + 1}": v for i, v in enumerate(sorted_decile_contribution(g["gross_bps"].to_numpy()))}})
    pd.DataFrame(win_rows).to_csv(out / "win_loss_by_group.csv", index=False)
    pd.DataFrame(dec_w + dec_l).to_csv(out / "win_loss_deciles.csv", index=False)
    pd.DataFrame(sorted_rows).to_csv(out / "sorted_decile_share.csv", index=False)

    exc_rows, rule_rows = [], []
    for (freq, asset), tg in trades.groupby(["freq", "asset"]):
        bm = BAR_MINUTES[freq]
        df = pd.read_parquet(ROOT / "data" / "processed_v2" / f"{asset}_{freq}.parquet", columns=["timestamp", "open", "high", "low", "close"])
        ts, bars = bars_from_frame(df, atr_window=max(4, 60 // bm))
        m1 = pd.read_parquet(ROOT / "data" / "processed_v2" / f"{asset}_1m.parquet", columns=["timestamp", "close", "quote_volume"])
        qv = market_features(m1, windows=(60,))["qv"]
        qv_test = qv[qv.index >= OOS_START].dropna().to_numpy()
        edges = np.percentile(qv_test, [100 / 3, 200 / 3])
        combos = combos_for(time_bars_for(bm))
        needed = sorted({r for rules in combos.values() for r in rules}, key=lambda r: (r.kind, r.param))
        for strategy, g in tg.groupby("strategy"):
            e, x = bar_indices(ts, g["entry_ts"], g["exit_ts"])
            sign = np.where(g["side"].to_numpy() == "long", 1.0, -1.0)
            gross = g["gross_bps"].to_numpy()
            if np.abs(trade_return(bars, e, x, sign) - gross).max() > 1e-6:
                raise SystemExit("O retorno refeito das barras não bate com o gravado.")
            regime = np.digitize(qv.reindex(pd.DatetimeIndex(g["entry_ts"])).to_numpy(), edges)     # 0 = volume baixo, 1 = médio, 2 = alto
            mae, mfe = excursions(bars, e, x, sign)
            exc_rows.append(pd.DataFrame({"freq": freq, "asset": asset, "strategy": strategy, "gross": gross, "mae": mae, "mfe": mfe}))
            outcomes = rule_outcomes(bars, e, x, sign, needed)
            n_reg = np.bincount(regime, minlength=3)
            for name, rules in combos.items():
                ret, hit = combine(outcomes, rules, gross)
                rule_rows.append({"freq": freq, "asset": asset, "strategy": strategy, "combo": name, "n": len(g), "sum_bps": float(ret.sum()), "n_early_exit": int(hit.sum()),
                                  **{f"sum_vol{i}": float(ret[regime == i].sum()) for i in range(3)}, **{f"n_vol{i}": int(n_reg[i]) for i in range(3)}})
            logger.info("%s %s %s (%.0f s)", freq, asset, strategy, time.time() - t0)
    exc = pd.concat(exc_rows, ignore_index=True)
    ex_summary = []
    for (freq, strategy), g in exc.groupby(["freq", "strategy"]):
        w, l = g[g.gross > 0], g[g.gross < 0]
        ex_summary.append({"freq": freq, "strategy": strategy, "wins_MAE_median": w.mae.median(), "wins_MAE_p90": w.mae.quantile(.9),
                           **{f"wins_MAE_ge{k}_pct": 100 * (w.mae >= k).mean() for k in (10, 20, 50, 100)},
                           "losses_MFE_median": l.mfe.median(), "losses_MFE_p90": l.mfe.quantile(.9),
                           **{f"losses_MFE_ge{k}_pct": 100 * (l.mfe >= k).mean() for k in (10, 20, 50, 100)},
                           "loss_median": -l.gross.median(), "losses_MAE_median": l.mae.median()})
    pd.DataFrame(ex_summary).to_csv(out / "excursions_summary.csv", index=False)
    rules = pd.DataFrame(rule_rows)
    rules.to_csv(out / "rules_by_group.csv", index=False)

    # resumo por frequência x estratégia (ativos somados)
    pooled = rules.groupby(["freq", "strategy", "combo"]).agg(n=("n", "sum"), total=("sum_bps", "sum"), early=("n_early_exit", "sum")).reset_index()
    pooled["mean_bps"] = pooled["total"] / pooled["n"]; pooled["early_pct"] = 100 * pooled["early"] / pooled["n"]
    lines = ["# Regras de saída (descritivo)", "", "Média por operação em bps, brutas, ativos somados. Hipóteses: saída no nível (ou na abertura após salto), sem derrapagem, "
             "sem taxa e sem reentrada.", ""]
    base = pooled[pooled.combo == "base"].set_index(["freq", "strategy"])["mean_bps"]
    rows = []
    for (freq, strategy), g in pooled.groupby(["freq", "strategy"]):
        g = g[g.combo != "base"]
        best = g.loc[g["mean_bps"].idxmax()]
        rows.append({"freq": freq, "strategy": strategy, "baseline": base[(freq, strategy)], "best_rule": best["combo"], "best_mean": best["mean_bps"], "early_exit_pct": best["early_pct"]})
    best_rows = rows
    lines += [markdown_table(pd.DataFrame(rows), ".2f"), ""]

    # uma regra por vez: média por operação (e % de operações que saíram antes), por parâmetro
    single = pooled[pooled.combo.str.fullmatch(r"(SL|TP|TR|BE|ASL|DSL|DTR|TIME)\d+")].copy()
    single["family"] = single.combo.str.extract(r"^([A-Z]+)")[0]
    single["param"] = single.combo.str.extract(r"(\d+)$")[0].astype(int)
    single["cell"] = single.apply(lambda r: f"{r.mean_bps:.2f} ({r.early_pct:.0f}%)", axis=1)
    single["col"] = single.freq + " " + single.strategy
    names = {"SL": "Stop loss (bps)", "TP": "Stop gain (bps)", "TR": "Trailing (bps do pico)", "BE": "Break-even (ativa a X bps a favor)",
             "ASL": "Stop de k x ATR da entrada", "DSL": "Stop de k x ATR da barra", "DTR": "Trailing de k x ATR da barra", "TIME": "Parada por tempo (min)"}
    for fam, title in names.items():
        tab = single[single.family == fam].pivot(index="param", columns="col", values="cell")
        lines += [f"## {title}: média por operação (% com saída antecipada)", "", markdown_table(tab.reset_index(), ".2f"), ""]

    # volume da última hora (tercis dos minutos de teste): 'como foi' e a melhor regra de cada grupo, por tercil
    terc = rules.groupby(["freq", "strategy", "combo"]).agg(**{f"sum{i}": (f"sum_vol{i}", "sum") for i in range(3)}, **{f"n{i}": (f"n_vol{i}", "sum") for i in range(3)}).reset_index()
    tercile_rows = []
    for best in best_rows:
        for combo in ("base", best["best_rule"]):
            t = terc[(terc.freq == best["freq"]) & (terc.strategy == best["strategy"]) & (terc.combo == combo)].iloc[0]
            tercile_rows.append({"freq": best["freq"], "strategy": best["strategy"], "combo": combo, **{f"vol{i}_mean_bps": t[f"sum{i}"] / t[f"n{i}"] for i in range(3)}})
    lines += ["## Média por operação por tercil de volume da última hora (0 = baixo, 1 = médio, 2 = alto)", "", markdown_table(pd.DataFrame(tercile_rows), ".2f"), ""]
    (out / "rules_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Pronto em %.0f s: %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
