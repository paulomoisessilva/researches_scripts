"""Orquestração de execuções walk-forward identificadas por ``run_id``.

Cada execução (janela de treino x cenário de custo x período) grava em uma pasta
própria, ``results/walk_forward/<run_id>/``, com tabelas, equity e ``run_meta.json``.
Isso evita que rodadas com configurações diferentes se sobrescrevam ou se
misturem (a chave de merge dos CSVs é só ``window_id``/``strategy``/``seed``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    filter_dataframe_by_date_range,
    get_project_root,
    resolve_cost_bps,
    resolve_date_range,
)
from .indicators import features_parquet_path, load_or_compute_features
from .metrics import annualization_factor
from .utils import ensure_dir, setup_logger
from .backtest import BacktestConfig
from .random_baseline import (
    cache_dir_for,
    cache_is_complete,
    export_to_cache,
    load_from_cache,
    random_cache_key,
    run_random_baseline_to_disk,
)
from .walk_forward import (
    _get_wf_int,
    build_windows,
    run_walk_forward,
    save_walk_forward_results,
    validate_grid_features,
)

logger = setup_logger(__name__)

RUNS_SUBDIR = Path("results") / "walk_forward"
META_FILENAME = "run_meta.json"

def resolve_runs_root(cfg: dict[str, Any]) -> Path:
    """Pasta das runs: ``walk_forward.runs_dir`` (relativa à raiz do projeto) ou o padrão."""
    runs_dir = cfg.get("walk_forward", {}).get("runs_dir")
    return get_project_root() / (runs_dir or RUNS_SUBDIR)


# Seções do config que definem o experimento. Custos ficam de fora (estão no
# run_id) e ``parallel_workers`` não altera resultados.
_FINGERPRINT_BACKTEST_KEYS = (
    "initial_capital", "allow_short", "execution", "max_position",
    "close_position_end_of_day", "max_holding_bars",
)


# ---------------------------------------------------------------------------
# Resolução de parâmetros da execução
# ---------------------------------------------------------------------------


def parse_cost_scenarios(items: list[str]) -> list[tuple[float, float]]:
    """Converte ``["2:2", "0:0"]`` em ``[(2.0, 2.0), (0.0, 0.0)]`` (fee:slippage por lado)."""
    scenarios: list[tuple[float, float]] = []
    for item in items:
        try:
            fee_s, slip_s = str(item).split(":")
            scenarios.append((float(fee_s), float(slip_s)))
        except ValueError as exc:
            raise ValueError(
                f"Cenário de custo inválido: '{item}'. Use FEE:SLIPPAGE em bps, ex.: 2:2."
            ) from exc
    return scenarios


def resolve_cost_scenarios(
    cfg: dict[str, Any],
    cli_costs: list[str] | None = None,
    cli_fee: float | None = None,
    cli_slippage: float | None = None,
    study: bool = False,
) -> list[tuple[float, float]]:
    """Cenários de custo: --costs > --fee/--slippage > study.cost_scenarios > backtest[0].

    Sem ``study``, usa apenas o cenário primário (primeiro de ``study.cost_scenarios``).
    """
    if cli_costs:
        return parse_cost_scenarios(cli_costs)
    if cli_fee is not None or cli_slippage is not None:
        return [resolve_cost_bps(cfg, cli_fee, cli_slippage)]

    configured = cfg.get("study", {}).get("cost_scenarios")
    if configured:
        scenarios = [(float(f), float(s)) for f, s in configured]
        return scenarios if study else scenarios[:1]

    fee, slip = resolve_cost_bps(cfg)
    if fee == 0.0 and slip == 0.0:
        logger.warning(
            "Nenhum cenário de custo configurado: usando fee=0 e slippage=0 "
            "(resultados BRUTOS de custos). Informe --costs FEE:SLIP ou "
            "study.cost_scenarios no config."
        )
    return [(fee, slip)]


def resolve_train_days(
    cfg: dict[str, Any],
    cli_train_days: list[int] | None = None,
    study: bool = False,
) -> list[int]:
    """Janelas de treino: --train-days > study.train_days (com --study) > walk_forward.train_days."""
    if cli_train_days:
        return [int(d) for d in cli_train_days]
    if study and cfg.get("study", {}).get("train_days"):
        return [int(d) for d in cfg["study"]["train_days"]]
    return [int(cfg["walk_forward"]["train_days"])]


def resolve_validation_days(
    cfg: dict[str, Any],
    cli_validation_days: list[int] | None = None,
    study: bool = False,
) -> list[int]:
    """Janelas de validação (lookback de seleção): --validation-days > study.validation_days > config."""
    if cli_validation_days:
        return [int(d) for d in cli_validation_days]
    if study and cfg.get("study", {}).get("validation_days"):
        return [int(d) for d in cfg["study"]["validation_days"]]
    wf = cfg["walk_forward"]
    return [int(wf.get("validation_days", wf.get("val_days")))]


def _date_tag(value: str | None) -> str:
    return str(value).replace("-", "") if value else "na"


def make_run_id(
    cfg: dict[str, Any],
    fee_bps: float,
    slippage_bps: float,
    start_date: str | None,
    end_date: str | None,
    tag: str | None = None,
) -> str:
    """Identificador determinístico da execução, derivado de janelas, custo e período."""
    wf = cfg["walk_forward"]
    val_days = wf.get("validation_days", wf.get("val_days"))
    parts = [
        f"tr{wf['train_days']}",
        f"va{val_days}",
        f"te{wf['test_days']}",
        f"st{wf['step_days']}",
        f"fee{fee_bps:g}",
        f"slip{slippage_bps:g}",
        f"{_date_tag(start_date)}-{_date_tag(end_date)}",
    ]
    if wf.get("oos_start"):
        parts.append(f"oos{_date_tag(str(wf['oos_start']))}")
    if tag:
        parts.append(tag)
    return "_".join(parts)


def config_fingerprint(cfg: dict[str, Any]) -> str:
    """Hash das seções que definem o experimento (exclui custos e paralelismo)."""
    wf = {
        k: v for k, v in cfg.get("walk_forward", {}).items()
        if k not in ("parallel_workers", "runs_dir", "random_cache")
    }
    bt = {k: cfg.get("backtest", {}).get(k) for k in _FINGERPRINT_BACKTEST_KEYS}
    payload = {
        "features": cfg.get("features"),
        "strategy": cfg.get("strategy"),
        "random_baseline": cfg.get("random_baseline"),
        "walk_forward": wf,
        "backtest": bt,
        "data": {
            "start_date": cfg.get("data", {}).get("start_date"),
            "end_date": cfg.get("data", {}).get("end_date"),
        },
        "annualization_factor_15m": annualization_factor("15m"),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Entradas (dados + features), carregadas uma vez por estudo
# ---------------------------------------------------------------------------


def prepare_inputs(
    cfg: dict[str, Any],
    symbols: list[str],
    intervals: list[str],
    strategies: list[str],
    start_date: str | None,
    end_date: str | None,
    force_features: bool = False,
) -> dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]]:
    """Carrega OHLCV e features por (símbolo, intervalo) e valida o grid do config."""
    root = get_project_root()
    processed_dir = root / cfg["data"]["processed_dir"]
    inputs: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}

    for symbol in symbols:
        for interval in intervals:
            proc_path = processed_dir / f"{symbol}_{interval}.parquet"
            if not proc_path.exists():
                logger.warning("Dados processados não encontrados: %s", proc_path.name)
                continue

            df = pd.read_parquet(proc_path)
            df, date_filtered = filter_dataframe_by_date_range(df, start_date, end_date)
            logger.info("%s %s: %d candles no período.", symbol, interval, len(df))

            feat_path = features_parquet_path(root, symbol, interval)
            df_feat = load_or_compute_features(
                df,
                cfg,
                features_path=feat_path,
                force_compute=force_features or date_filtered,
                strategy_names=strategies,
            )
            if len(df_feat) != len(df) or (
                len(df) and df_feat["timestamp"].iloc[-1] != df["timestamp"].iloc[-1]
            ):
                logger.warning(
                    "Cache de features desalinhado com os dados (%d vs %d candles); "
                    "recalculando em memória.", len(df_feat), len(df),
                )
                df_feat = load_or_compute_features(
                    df, cfg, features_path=None, force_compute=True,
                    strategy_names=strategies,
                )

            validate_grid_features(df_feat, cfg, strategies)
            inputs[(symbol, interval)] = (df, df_feat)

    if not inputs:
        raise FileNotFoundError("Nenhum dataset processado encontrado para os símbolos/intervalos.")
    return inputs


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------


def _check_meta_compatible(meta_path: Path, meta: dict[str, Any]) -> None:
    if not meta_path.exists():
        return
    old = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("fingerprint", "fee_bps", "slippage_bps"):
        if old.get(key) != meta.get(key):
            raise ValueError(
                f"A pasta {meta_path.parent.name} já contém uma execução com "
                f"{key}={old.get(key)!r} (agora {meta.get(key)!r}). Use --fresh para "
                "recomeçar ou outro --run-id/--tag; misturar execuções corromperia as tabelas."
            )


def _run_random_strategy(
    run_cfg: dict[str, Any],
    df: pd.DataFrame,
    df_feat: pd.DataFrame,
    symbol: str,
    interval: str,
    fee_bps: float,
    slippage_bps: float,
    run_dir: Path,
    n_workers: int,
    save_random_equity: bool,
) -> tuple[str, int]:
    """Placebo: reaproveita o cache compartilhado (se houver) ou roda em paralelo, em blocos."""
    wf_cfg = run_cfg["walk_forward"]
    windows = build_windows(
        df,
        train_days=_get_wf_int(wf_cfg, "train_days"),
        val_days=_get_wf_int(wf_cfg, "validation_days", "val_days"),
        test_days=_get_wf_int(wf_cfg, "test_days"),
        step_days=_get_wf_int(wf_cfg, "step_days"),
        oos_start=wf_cfg.get("oos_start"),
    )
    if not windows:
        logger.warning("Sem janelas walk-forward para %s %s", symbol, interval)
        return "none", 0

    # Sem equity do random, o resultado independe do lookback e pode ser compartilhado.
    use_cache = bool(wf_cfg.get("random_cache", True)) and not save_random_equity
    cache_dir = None
    if use_cache:
        key = random_cache_key(run_cfg, symbol, interval, fee_bps, slippage_bps, df_feat)
        cache_dir = cache_dir_for(resolve_runs_root(run_cfg).parent / "cache" / "random", symbol, interval, key)
        if cache_is_complete(cache_dir):
            logger.info("Random baseline: reaproveitando cache %s", cache_dir.name)
            load_from_cache(cache_dir, run_dir)
            return "hit", len(windows)

    bt_cfg = BacktestConfig(
        strategy_name="random",
        symbol=symbol,
        interval=interval,
        initial_capital=run_cfg["backtest"]["initial_capital"],
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        allow_short=run_cfg["backtest"]["allow_short"],
        max_position=run_cfg["backtest"]["max_position"],
        close_eod=wf_cfg.get("close_position_end_of_day", True),
    )
    run_random_baseline_to_disk(
        df_feat, run_cfg, symbol, interval, windows, bt_cfg, wf_cfg, run_dir,
        n_workers=n_workers, save_random_equity=save_random_equity,
    )
    if cache_dir is not None:
        export_to_cache(run_dir, cache_dir, symbol, interval)
        return "miss (salvo no cache)", len(windows)
    return "desligado", len(windows)


def _merge_previous_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    """Ao retomar/ampliar uma run (ex.: novos ativos), preserva o que já constava no meta."""
    if not meta_path.exists():
        return
    old = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("symbols", "intervals", "strategies"):
        meta[key] = sorted(set(old.get(key, [])) | set(meta.get(key, [])))
    for key in ("n_windows", "random_cache"):
        merged = dict(old.get(key, {}))
        merged.update(meta.get(key, {}))
        if merged:
            meta[key] = merged
    meta["created_at"] = old.get("created_at", meta["created_at"])


def run_experiment(
    cfg: dict[str, Any],
    inputs: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]],
    strategies: list[str],
    fee_bps: float,
    slippage_bps: float,
    *,
    train_days: int | None = None,
    validation_days: int | None = None,
    run_id: str | None = None,
    tag: str | None = None,
    workers: int | None = None,
    fresh: bool = False,
    config_path: str | None = None,
) -> Path:
    """Roda o walk-forward para uma combinação (janelas, custo) e salva em pasta própria."""
    run_cfg = copy.deepcopy(cfg)
    if train_days is not None:
        run_cfg["walk_forward"]["train_days"] = int(train_days)
    if validation_days is not None:
        run_cfg["walk_forward"]["validation_days"] = int(validation_days)
        run_cfg["walk_forward"].pop("val_days", None)  # alias antigo não pode sobrepor

    start_date, end_date = resolve_date_range(run_cfg)
    run_id = run_id or make_run_id(run_cfg, fee_bps, slippage_bps, start_date, end_date, tag)
    run_dir = resolve_runs_root(run_cfg) / run_id

    wf_cfg = run_cfg["walk_forward"]
    meta: dict[str, Any] = {
        "run_id": run_id,
        "fingerprint": config_fingerprint(run_cfg),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "train_days": int(wf_cfg["train_days"]),
        "validation_days": int(wf_cfg.get("validation_days", wf_cfg.get("val_days"))),
        "test_days": int(wf_cfg["test_days"]),
        "step_days": int(wf_cfg["step_days"]),
        "oos_start": str(wf_cfg["oos_start"]) if wf_cfg.get("oos_start") else None,
        "selection_metric": wf_cfg.get("selection_metric", "net_sharpe"),
        "close_position_end_of_day": bool(wf_cfg.get("close_position_end_of_day", True)),
        "carry_position_state": bool(wf_cfg.get("carry_position_state", False)),
        "start_date": start_date,
        "end_date": end_date,
        "symbols": sorted({s for s, _ in inputs}),
        "intervals": sorted({i for _, i in inputs}),
        "strategies": list(strategies),
        "random_baseline": run_cfg.get("random_baseline"),
        "annualization_days_per_year": 365,
        "config_path": config_path,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if fresh and run_dir.exists():
        logger.info("--fresh: removendo %s", run_dir)
        shutil.rmtree(run_dir)
    ensure_dir(run_dir)
    meta_path = run_dir / META_FILENAME
    _check_meta_compatible(meta_path, meta)
    _merge_previous_meta(meta_path, meta)
    # Grava já no início: se a execução cair no meio, a retomada ainda é validada.
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("RUN %s", run_id)
    logger.info(
        "train=%s val=%s test=%s step=%s oos_start=%s | fee=%g slip=%g bps | %s",
        meta["train_days"], meta["validation_days"], meta["test_days"], meta["step_days"],
        meta["oos_start"], fee_bps, slippage_bps, strategies,
    )
    logger.info("=" * 60)

    save_random_equity = bool(wf_cfg.get("save_random_equity", True))
    n_workers = workers if workers is not None else wf_cfg.get("parallel_workers", 1)

    for (symbol, interval), (df, df_feat) in inputs.items():
        for strategy in strategies:
            logger.info("Estratégia %s (%s %s)", strategy, symbol, interval)
            if strategy == "random":
                status, n_windows = _run_random_strategy(
                    run_cfg, df, df_feat, symbol, interval, fee_bps, slippage_bps,
                    run_dir, n_workers, save_random_equity,
                )
                meta.setdefault("random_cache", {})[f"{symbol}_{interval}"] = status
                meta.setdefault("n_windows", {})[f"{symbol}_{interval}_{strategy}"] = n_windows
                continue
            results = run_walk_forward(
                df=df,
                strategy_name=strategy,
                cfg=run_cfg,
                symbol=symbol,
                interval=interval,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                df_feat=df_feat,
                n_workers=n_workers,
            )
            # Salva por estratégia para limitar memória e permitir retomada.
            save_walk_forward_results(
                results, run_dir, run_dir, save_random_equity=save_random_equity
            )
            meta.setdefault("n_windows", {})[f"{symbol}_{interval}_{strategy}"] = len(
                {r.window_id for r in results}
            )
            del results

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Run concluída: %s", run_dir)
    return run_dir
