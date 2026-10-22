#!/usr/bin/env python3
"""Script 19 — Perfil das entradas por hora e dia da semana (análise descritiva do estudo v2).

Refaz a trajetória contínua de cada candidata (ativo x estratégia x lookback x frequência) com os
parâmetros que o walk-forward escolheu dia a dia (``best_parameters_by_window.csv``), extrai as
operações e agrupa as ENTRADAS pela hora UTC e pelo dia da semana da barra de entrada. Para cada faixa
(hora, dia da semana, hora x dia da semana) conta: entradas, % que deu lucro bruto (custo 0), % que
dá lucro a 2,0, 4,5 e 5,0 bps por lado (líquido = bruto − 2c por operação), retorno bruto médio por
operação e CBE da faixa. Os grupos são (frequência, estratégia) com os ativos e os lookbacks
somados, e também por ativo. Intervalos de confiança por bootstrap estacionário sobre os dias e
concordância do perfil entre a 1ª e a 2ª metade dos dias de teste.

Saídas em ``results/v2/analysis/``: ``entry_profile_{hour,weekday,cell}.csv``,
``entry_profile_half_split.csv``, ``entry_profile_arrays.npz`` e figuras em ``figs/``.

Uso: python scripts/19_entry_profile_v2.py [--reps 2000]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import BacktestConfig
from src.carried_walk_forward import carried_backtest
from src.config import filter_dataframe_by_date_range, load_config
from src.indicators import compute_all_features
from src.utils import setup_logger
from src.v2_study import FREQS, accumulate_profile, bootstrap_counts, half_split_agreement, profile_tables
from src.walk_forward import build_windows

logger = setup_logger("19_entry_profile_v2")
ROOT = Path(__file__).parent.parent
LOOKBACKS = [1, 2, 3, 7, 14, 21, 28]
WEEKDAYS = ["seg", "ter", "qua", "qui", "sex", "sab", "dom"]


def run_dir(root: Path, freq: str, lookback: int) -> Path:
    for d in sorted((root / freq / "walk_forward").iterdir()):
        if f"_va{lookback}_" in d.name:
            return d
    raise FileNotFoundError(f"run de {freq} com lookback {lookback} não encontrada")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results/v2")
    parser.add_argument("--out", default="results/v2/analysis")
    parser.add_argument("--reps", type=int, default=2000)
    args = parser.parse_args()
    results = ROOT / args.results
    out = ROOT / args.out
    (out / "figs").mkdir(parents=True, exist_ok=True)

    acc: dict[tuple, dict[str, np.ndarray]] = {}
    n_days = None
    t0 = time.time()
    checks = []
    for freq in FREQS:
        cfg = load_config(ROOT / f"config.v2_{freq}.yaml")
        wf = cfg["walk_forward"]
        oos = pd.Timestamp(wf["oos_start"], tz="UTC")
        n_days = int((pd.Timestamp(cfg["data"]["end_date"], tz="UTC") - pd.Timedelta(days=1) - oos).days) + 1
        for symbol in cfg["data"]["symbols"]:
            df = pd.read_parquet(ROOT / cfg["data"]["processed_dir"] / f"{symbol}_{freq}.parquet")
            df, _ = filter_dataframe_by_date_range(df, cfg["data"]["start_date"], cfg["data"]["end_date"])
            df_feat = compute_all_features(df, cfg, strategy_names=cfg["strategies"])
            bt = BacktestConfig(strategy_name="", symbol=symbol, interval=freq, fee_bps=0.0, slippage_bps=0.0, close_eod=False)
            for lookback in LOOKBACKS:
                rd = run_dir(results, freq, lookback)
                bp = pd.read_csv(rd / "best_parameters_by_window.csv")
                res = pd.read_csv(rd / "walk_forward_results.csv", usecols=["strategy", "symbol", "sum_gross_ret"])
                windows = build_windows(df[["timestamp"]], wf["train_days"], lookback, wf["test_days"], wf["step_days"], oos_start=wf["oos_start"])
                for strategy in cfg["strategies"]:
                    sel = bp[(bp["symbol"] == symbol) & (bp["strategy"] == strategy)].sort_values("window_id")
                    params = [
                        {"window": int(r.param_window), "k": float(r.param_k), "entry_long_z": float(r.param_entry_long_z),
                         "entry_short_z": float(r.param_entry_short_z), "exit_long_z": float(r.param_exit_long_z),
                         "exit_short_z": float(r.param_exit_short_z)}
                        for r in sel.itertuples()
                    ]
                    bt.strategy_name = strategy
                    result, _ = carried_backtest(df_feat, strategy, windows, params, bt)
                    # a trajetória refeita tem de bater com a que o walk-forward gravou
                    saved = float(res[(res["symbol"] == symbol) & (res["strategy"] == strategy)]["sum_gross_ret"].sum())
                    redone = float(result.gross_returns.sum())
                    checks.append(abs(saved - redone))
                    tr = result.trades
                    arrays = accumulate_profile(pd.DatetimeIndex(tr["entry_timestamp"]), tr["trade_return"].to_numpy(), oos, n_days)
                    for key in ((freq, strategy, symbol), (freq, strategy, "ALL")):
                        if key not in acc:
                            acc[key] = {k: np.zeros_like(v) for k, v in arrays.items()}
                        for k, v in arrays.items():
                            acc[key][k] += v
            logger.info("%s %s pronto (%.0f s).", freq, symbol, time.time() - t0)
            del df, df_feat
    logger.info("Maior diferença entre o bruto refeito e o gravado: %.2e", max(checks))
    assert max(checks) < 1e-8, "a trajetória refeita não bate com a gravada"

    np.savez_compressed(out / "entry_profile_arrays.npz", **{"|".join(k) + "|" + n: a for k, d in acc.items() for n, a in d.items()})
    counts = bootstrap_counts(n_days, args.reps, 10.0, 0)
    tables = {"hour": [], "weekday": [], "cell": []}
    split = []
    for key, arrays in acc.items():
        for level, tab in profile_tables(arrays, counts=counts).items():
            tab.insert(0, "asset", key[2]); tab.insert(0, "strategy", key[1]); tab.insert(0, "freq", key[0])
            tables[level].append(tab)
        hs = half_split_agreement(arrays, n_days // 2)
        hs.insert(0, "asset", key[2]); hs.insert(0, "strategy", key[1]); hs.insert(0, "freq", key[0])
        split.append(hs)
    for level, frames in tables.items():
        pd.concat(frames, ignore_index=True).to_csv(out / f"entry_profile_{level}.csv", index=False)
    pd.concat(split, ignore_index=True).to_csv(out / "entry_profile_half_split.csv", index=False)

    # ----- figuras (ativos somados)
    hour = pd.concat(tables["hour"]); weekday = pd.concat(tables["weekday"]); cell = pd.concat(tables["cell"])
    pooled = lambda t: t[t["asset"] == "ALL"]  # noqa: E731
    strategies = ["bollinger", "regression_bands"]

    def bars(table, x_labels, value, lo, hi, title, fname, ylabel, ref=None):
        fig, axes = plt.subplots(len(strategies), len(FREQS), figsize=(15, 7), sharey="row")
        for i, s in enumerate(strategies):
            for j, f in enumerate(FREQS):
                t = pooled(table[(table["freq"] == f) & (table["strategy"] == s)])
                ax = axes[i, j]
                v = t[value].to_numpy()
                ax.bar(range(len(t)), v, color="#4c78a8", yerr=[v - t[lo].to_numpy(), t[hi].to_numpy() - v], capsize=2, error_kw={"lw": 0.8})
                if ref == "half":
                    ax.axhline(0.5, color="#888", lw=0.8, ls="--")
                if ref == "zero":
                    ax.axhline(0.0, color="#888", lw=0.8)
                ax.set_xticks(range(len(t))); ax.set_xticklabels(x_labels, fontsize=7)
                ax.set_title(f"{s} · {f}", fontsize=10)
                if j == 0:
                    ax.set_ylabel(ylabel)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(out / "figs" / fname, dpi=130)
        plt.close(fig)

    hours = [f"{h:02d}" for h in range(24)]
    bars(hour, hours, "share_pos_gross", "share_pos_gross_ci_low", "share_pos_gross_ci_high",
         "Entradas com lucro bruto, por hora UTC da entrada (ativos e lookbacks somados; IC 95% por dias)",
         "entry_hour_share_positive.png", "% das entradas com lucro bruto", ref="half")
    bars(hour, hours, "mean_gross_bps", "mean_gross_bps_ci_low", "mean_gross_bps_ci_high",
         "Retorno bruto médio por operação (bps), por hora UTC da entrada", "entry_hour_mean_gross.png",
         "bps por operação", ref="zero")
    bars(weekday, WEEKDAYS, "share_pos_gross", "share_pos_gross_ci_low", "share_pos_gross_ci_high",
         "Entradas com lucro bruto, por dia da semana (UTC)", "entry_weekday_share_positive.png",
         "% das entradas com lucro bruto", ref="half")

    for value, fname, title in (("share_pos_gross", "entry_heatmap_share_positive.png", "% das entradas com lucro bruto"),
                                ("mean_gross_bps", "entry_heatmap_mean_gross.png", "Retorno bruto médio por operação (bps)"),
                                ("n_entries", "entry_heatmap_count.png", "Número de entradas")):
        fig, axes = plt.subplots(len(strategies), len(FREQS), figsize=(16, 6.5))
        for i, s in enumerate(strategies):
            for j, f in enumerate(FREQS):
                t = pooled(cell[(cell["freq"] == f) & (cell["strategy"] == s)])
                mat = t[value].to_numpy().reshape(7, 24)
                ax = axes[i, j]
                kw = {}
                if value == "share_pos_gross":
                    kw = {"cmap": "RdBu_r", "vmin": 0.4, "vmax": 0.6}
                elif value == "mean_gross_bps":
                    lim = np.nanpercentile(np.abs(mat), 95)
                    kw = {"cmap": "RdBu_r", "vmin": -lim, "vmax": lim}
                else:
                    kw = {"cmap": "Blues"}
                im = ax.imshow(mat, aspect="auto", **kw)
                ax.set_yticks(range(7)); ax.set_yticklabels(WEEKDAYS, fontsize=7)
                ax.set_xticks(range(0, 24, 3)); ax.set_xticklabels(hours[::3], fontsize=7)
                ax.set_title(f"{s} · {f}", fontsize=10)
                fig.colorbar(im, ax=ax, fraction=0.04)
        fig.suptitle(f"{title}: dia da semana x hora UTC (ativos e lookbacks somados)", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / "figs" / fname, dpi=130)
        plt.close(fig)
    logger.info("Pronto em %.0f s. Saídas em %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
