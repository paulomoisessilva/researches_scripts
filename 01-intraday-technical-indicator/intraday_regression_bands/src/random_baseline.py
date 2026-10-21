"""Baseline aleatório (placebo) em escala: paralelo por seed, gravado em blocos e com cache.

O placebo roda ``random_baseline.n_seeds`` seeds x ~1800 janelas de 1 dia. A versão
original fazia tudo num único processo, recalculava a máscara de datas (175 mil barras)
a cada janela e guardava os ~177 mil resultados na memória antes de salvar (~23 min e
~5,6 GB). Aqui:

* os recortes de teste de cada janela são calculados uma vez (``searchsorted``);
* cada seed é uma tarefa de um pool de processos e devolve só tabelas compactas;
* as seeds são gravadas em blocos (memória limitada);
* o resultado pode ser reaproveitado entre runs que compartilham o mesmo período OOS
  e custo (o placebo não depende do lookback de seleção), via ``random_cache_key``.

A lógica por janela é idêntica à original (``tests/test_random_baseline.py`` compara
com uma cópia do laço antigo).
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .metrics import compute_metrics
from .strategies import resolve_random_seeds, strategy_random
from .utils import ensure_dir, setup_logger
from .walk_forward import (
    WFResult,
    WFWindow,
    _random_baseline_params,
    aggregate_walk_forward_oos,
    build_result_tables,
    write_result_tables,
)

logger = setup_logger(__name__)

# Incrementar quando mudar a lógica do placebo (invalida caches antigos).
RANDOM_CACHE_VERSION = 1
CACHE_TABLES = (
    "walk_forward_results.csv",
    "best_parameters_by_window.csv",
    "walk_forward_oos_summary.csv",
)


# ---------------------------------------------------------------------------
# Contexto e execução por seed
# ---------------------------------------------------------------------------


@dataclass
class RandomContext:
    light: pd.DataFrame            # colunas timestamp/open (suficientes para o backtest)
    bounds: list[tuple[int, int]]  # (início, fim exclusivo) do teste de cada janela
    windows: list[WFWindow]
    test_frames: list[pd.DataFrame]
    cfg: dict
    bt_cfg: BacktestConfig
    symbol: str
    interval: str
    min_test_bars: int
    save_random_equity: bool


def build_context(
    df_feat: pd.DataFrame,
    cfg: dict,
    symbol: str,
    interval: str,
    windows: list[WFWindow],
    bt_cfg: BacktestConfig,
    wf_cfg: dict,
    save_random_equity: bool = True,
) -> RandomContext:
    light = df_feat[["timestamp", "open"]]
    ts = pd.DatetimeIndex(light["timestamp"])
    if not ts.is_monotonic_increasing:
        raise ValueError("Timestamps fora de ordem: o baseline aleatório exige série cronológica.")
    starts = ts.searchsorted([w.test_start for w in windows], side="left")
    ends = ts.searchsorted([w.test_end for w in windows], side="left")
    bounds = [(int(a), int(b)) for a, b in zip(starts, ends)]
    test_frames = [light.iloc[a:b].reset_index(drop=True) for a, b in bounds]
    return RandomContext(
        light=light,
        bounds=bounds,
        windows=windows,
        test_frames=test_frames,
        cfg=cfg,
        bt_cfg=bt_cfg,
        symbol=symbol,
        interval=interval,
        min_test_bars=1 if wf_cfg.get("test_days", 1) == 1 else 10,
        save_random_equity=save_random_equity,
    )


def seed_results(seed: int, ctx: RandomContext) -> list[WFResult]:
    """Resultados de todas as janelas para uma seed (mesma lógica do laço original)."""
    params = _random_baseline_params(ctx.cfg, seed)
    # Um sinal por seed no dataset completo; cada janela backtesta só o teste.
    full_signal = strategy_random(ctx.light, **params)

    results: list[WFResult] = []
    for win, (a, b), df_test in zip(ctx.windows, ctx.bounds, ctx.test_frames):
        if len(df_test) < ctx.min_test_bars:
            continue
        try:
            signal_test = full_signal.iloc[a:b].reset_index(drop=True)
            test_result = run_backtest(df_test, signal_test, ctx.bt_cfg)
        except Exception as exc:
            logger.warning("[WF %d] random seed=%d erro: %s", win.window_id, seed, exc)
            continue

        test_metrics = compute_metrics(
            test_result.equity,
            test_result.returns,
            test_result.positions,
            test_result.trades,
            interval=ctx.interval,
            strategy_name="random",
            symbol=ctx.symbol,
            fee_bps=ctx.bt_cfg.fee_bps,
            slippage_bps=ctx.bt_cfg.slippage_bps,
            gross_returns=test_result.gross_returns,
        )
        results.append(WFResult(
            window_id=win.window_id,
            strategy="random",
            symbol=ctx.symbol,
            interval=ctx.interval,
            best_params=params,
            val_metric=float("nan"),
            test_equity=test_result.equity,
            test_returns=test_result.returns,
            test_gross_returns=test_result.gross_returns,
            test_positions=test_result.positions,
            test_trades=test_result.trades,
            test_metrics=test_metrics,
        ))
    return results


_worker_ctx: RandomContext | None = None


def _init_worker(ctx: RandomContext) -> None:
    global _worker_ctx
    _worker_ctx = ctx


def _seed_tables(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[pd.DataFrame]] | None:
    """Tabelas compactas de uma seed (é o que volta do worker, não os WFResult)."""
    assert _worker_ctx is not None
    results = seed_results(seed, _worker_ctx)
    if not results:
        return None
    metrics_df, params_df, equity_frames = build_result_tables(
        results, _worker_ctx.save_random_equity
    )
    return metrics_df, params_df, aggregate_walk_forward_oos(results), equity_frames


# ---------------------------------------------------------------------------
# Execução em blocos
# ---------------------------------------------------------------------------


def run_random_baseline_to_disk(
    df_feat: pd.DataFrame,
    cfg: dict,
    symbol: str,
    interval: str,
    windows: list[WFWindow],
    bt_cfg: BacktestConfig,
    wf_cfg: dict,
    out_dir: str | Path,
    *,
    n_workers: int = 1,
    save_random_equity: bool = False,
    block_seeds: int | None = None,
) -> int:
    """Roda o placebo e grava as tabelas em ``out_dir`` por blocos de seeds. Retorna nº de seeds."""
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    seeds = resolve_random_seeds(cfg)
    ctx = build_context(df_feat, cfg, symbol, interval, windows, bt_cfg, wf_cfg, save_random_equity)
    workers = max(1, min(int(n_workers), len(seeds)))
    block = block_seeds or max(10, workers * 2)

    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(ctx,))
    else:
        _init_worker(ctx)

    try:
        done = 0
        for i in range(0, len(seeds), block):
            chunk = seeds[i:i + block]
            outputs = list(pool.map(_seed_tables, chunk, chunksize=1)) if pool else [
                _seed_tables(s) for s in chunk
            ]
            outputs = [o for o in outputs if o is not None]
            if outputs:
                write_result_tables(
                    pd.concat([o[0] for o in outputs], ignore_index=True),
                    pd.concat([o[1] for o in outputs], ignore_index=True),
                    pd.concat([o[2] for o in outputs], ignore_index=True),
                    [f for o in outputs for f in o[3]],
                    out_dir,
                    out_dir,
                )
            done += len(chunk)
            logger.info("Random baseline: %d/%d seeds concluídas.", done, len(seeds))
    finally:
        if pool is not None:
            pool.shutdown()
    return len(seeds)


# ---------------------------------------------------------------------------
# Cache compartilhado entre runs
# ---------------------------------------------------------------------------


def random_cache_key(
    cfg: dict,
    symbol: str,
    interval: str,
    fee_bps: float,
    slippage_bps: float,
    df_feat: pd.DataFrame,
) -> str:
    """Impressão digital de tudo que determina o resultado do placebo.

    Não inclui ``validation_days``/``train_days`` quando ``oos_start`` está definido: o
    placebo só depende dos dias de teste, que ficam idênticos entre os lookbacks.
    """
    wf = cfg.get("walk_forward", {})
    bt = cfg.get("backtest", {})
    windows_spec: dict[str, Any] = {
        "test_days": wf.get("test_days"),
        "step_days": wf.get("step_days"),
        "oos_start": str(wf["oos_start"]) if wf.get("oos_start") else None,
    }
    if not wf.get("oos_start"):  # sem oos_start o início dos testes depende de treino+validação
        windows_spec["train_days"] = wf.get("train_days")
        windows_spec["validation_days"] = wf.get("validation_days", wf.get("val_days"))

    ts = pd.DatetimeIndex(df_feat["timestamp"])
    data_hash = hashlib.sha1(df_feat["open"].to_numpy(dtype="float64").tobytes()).hexdigest()
    payload = {
        "version": RANDOM_CACHE_VERSION,
        "random_baseline": cfg.get("random_baseline"),
        "windows": windows_spec,
        "eod": bool(wf.get("close_position_end_of_day", True)),
        "backtest": {k: bt.get(k) for k in ("initial_capital", "allow_short", "max_position")},
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "symbol": symbol,
        "interval": interval,
        "n_bars": int(len(df_feat)),
        "first_ts": str(ts[0]) if len(ts) else None,
        "last_ts": str(ts[-1]) if len(ts) else None,
        "data_sha1": data_hash,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def cache_dir_for(cache_root: Path, symbol: str, interval: str, key: str) -> Path:
    return Path(cache_root) / f"{symbol}_{interval}_{key}"


def cache_is_complete(cache_dir: Path) -> bool:
    return (cache_dir / ".done").exists() and all((cache_dir / f).exists() for f in CACHE_TABLES)


def load_from_cache(cache_dir: Path, run_dir: Path) -> None:
    """Grava na run as tabelas do placebo guardadas no cache."""
    metrics = pd.read_csv(cache_dir / "walk_forward_results.csv", float_precision="round_trip")
    params = pd.read_csv(cache_dir / "best_parameters_by_window.csv", float_precision="round_trip")
    oos = pd.read_csv(cache_dir / "walk_forward_oos_summary.csv", float_precision="round_trip")
    write_result_tables(metrics, params, oos, [], run_dir, run_dir)


def export_to_cache(run_dir: Path, cache_dir: Path, symbol: str, interval: str) -> None:
    """Copia as linhas do placebo (deste ativo/intervalo) da run para o cache."""
    ensure_dir(cache_dir)
    for name in CACHE_TABLES:
        df = pd.read_csv(Path(run_dir) / name, float_precision="round_trip")
        mask = (df["strategy"] == "random") & (df["symbol"] == symbol) & (df["interval"] == interval)
        df.loc[mask].to_csv(cache_dir / name, index=False)
    (cache_dir / ".done").write_text("ok", encoding="utf-8")
