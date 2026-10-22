"""Seleção de parâmetros em passagem única (estratégias de bandas).

O motor "fatiado" (``walk_forward.optimize_params_on_validation``) refaz, para cada
janela e cada combinação do grid, o backtest de treino+validação (~28 dias). Como a
janela anda 1 dia por vez, cada dia é recalculado ~28 vezes.

Aqui cada combinação roda **uma vez** na série inteira. O retorno líquido por barra
fica acumulado em somas (``C1 = cumsum(r)``, ``C2 = cumsum(r^2)``, ...), e a pontuação
de qualquer janela de validação ``[a, b)`` sai por subtração, sem novo backtest::

    n = b - a;  S = C1[b] - C1[a];  Q = C2[b] - C2[a]
    sharpe = (S / n) * sqrt(F) / sqrt((Q - S^2 / n) / (n - 1))

A diferença para o motor fatiado está só no estado inicial da estratégia: ali cada
janela parte "zerada" em ``train_start``; aqui o estado vem do histórico contínuo.
``compare_engines`` mede essa diferença.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .metrics import SELECTION_METRICS, annualization_factor
from .strategies import BAND_STRATEGIES, compute_band_z, signal_from_z

_VAR_TOL = 1e-24  # variância abaixo disso conta como zero (Sharpe = 0, como std == 0)


# ---------------------------------------------------------------------------
# Pontuação por somas acumuladas
# ---------------------------------------------------------------------------


def window_scores_from_returns(
    returns: np.ndarray,
    start_idx: np.ndarray,
    end_idx: np.ndarray,
    metric: str,
    factor: float,
) -> np.ndarray:
    """Pontuação de seleção de cada janela ``[start_idx, end_idx)`` a partir dos retornos.

    Reproduz ``metrics.selection_metric_score`` (inclusive ``-inf`` para < 2 barras).
    """
    if metric not in SELECTION_METRICS:
        raise ValueError(
            f"Métrica de seleção desconhecida: '{metric}'. Suportadas: {sorted(SELECTION_METRICS)}"
        )
    r = np.nan_to_num(np.asarray(returns, dtype=float))
    a = np.asarray(start_idx, dtype=int)
    b = np.asarray(end_idx, dtype=int)
    n = (b - a).astype(float)
    too_short = n < 2

    def csum(x: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(x)))

    c1 = csum(r)
    total = c1[b] - c1[a]

    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "net_sharpe":
            c2 = csum(r * r)
            sq = c2[b] - c2[a]
            var = (sq - total * total / n) / (n - 1)
            score = np.where(
                var > _VAR_TOL, (total / n) * np.sqrt(factor) / np.sqrt(var), 0.0
            )
        elif metric == "net_total_return":
            score = np.exp(total) - 1.0
        else:  # net_profit_factor
            cg = csum(np.where(r > 0, r, 0.0))
            cl = csum(np.where(r < 0, -r, 0.0))
            gains = cg[b] - cg[a]
            losses = cl[b] - cl[a]
            score = np.where(
                losses == 0,
                np.where(gains > 0, np.inf, -np.inf),
                gains / np.where(losses == 0, 1.0, losses),
            )
    return np.where(too_short, -np.inf, score)


# ---------------------------------------------------------------------------
# Grid e janelas
# ---------------------------------------------------------------------------


def band_grid(strategy_name: str, cfg: dict) -> list[tuple[dict, dict]]:
    """Combinações (estrutural, limiares) na MESMA ordem do motor fatiado (desempate idêntico)."""
    from .walk_forward import _build_structural_param_grid, _build_threshold_param_grid

    structural = _build_structural_param_grid(strategy_name, cfg)
    thresholds = _build_threshold_param_grid(cfg)
    return [(s, t) for s in structural for t in thresholds]


def validation_bounds(df_feat: pd.DataFrame, windows: list) -> tuple[np.ndarray, np.ndarray]:
    """Índices (barra inicial, barra final exclusiva) da validação de cada janela."""
    idx = pd.DatetimeIndex(df_feat["timestamp"])
    starts = idx.searchsorted([w.val_start for w in windows], side="left")
    ends = idx.searchsorted([w.val_end for w in windows], side="left")
    return np.asarray(starts, dtype=int), np.asarray(ends, dtype=int)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


def _init_fast_worker(
    df_feat: pd.DataFrame,
    strategy_name: str,
    bt_cfg: BacktestConfig,
    start_idx: np.ndarray,
    end_idx: np.ndarray,
    metric: str,
    factor: float,
) -> None:
    _state.update(
        df=df_feat, name=strategy_name, bt=bt_cfg, a=start_idx, b=end_idx,
        metric=metric, factor=factor,
    )


def _eval_combo(task: tuple[dict, dict]) -> tuple[np.ndarray, np.ndarray] | None:
    """Backtest único na série completa; devolve (pontuação por janela, posição na 1ª barra da validação)."""
    st = _state
    struct, thresh = task
    try:
        z = compute_band_z(st["df"], st["name"], struct["window"], struct["k"])
        signal = signal_from_z(z, name=st["name"], **thresh)
        result = run_backtest(st["df"], signal, st["bt"], collect_trades=False)
    except Exception:
        return None
    returns = result.returns.to_numpy(dtype=float)
    scores = window_scores_from_returns(returns, st["a"], st["b"], st["metric"], st["factor"])
    positions = result.positions.to_numpy(dtype=float)
    first_bar = np.minimum(st["a"], len(positions) - 1)
    return scores, positions[first_bar]


# ---------------------------------------------------------------------------
# Seleção
# ---------------------------------------------------------------------------


def first_best_index(scores: np.ndarray) -> np.ndarray:
    """Por janela (coluna), o primeiro combo do grid dentro de ``SELECTION_TIE_TOL`` do máximo.

    Mesma regra do motor fatiado (``score > melhor + tol``), então empates matemáticos não
    dependem de ruído de ponto flutuante nem do motor.
    """
    from .walk_forward import SELECTION_TIE_TOL

    masked = np.where(np.isnan(scores), -np.inf, scores)
    cutoff = masked.max(axis=0) - SELECTION_TIE_TOL
    return np.argmax(masked >= cutoff[None, :], axis=0)


@dataclass
class FastSelection:
    combos: list[tuple[dict, dict]]
    scores: np.ndarray           # (n_combos, n_windows); -inf quando o combo falha
    start_pos: np.ndarray        # (n_combos, n_windows) posição na 1ª barra da validação
    best_index: np.ndarray       # (n_windows,)

    def best_params(self, window_idx: int) -> dict:
        struct, thresh = self.combos[int(self.best_index[window_idx])]
        return {**struct, **thresh}

    def best_score(self, window_idx: int) -> float:
        return float(self.scores[int(self.best_index[window_idx]), window_idx])


def fast_select_band_params(
    df_feat: pd.DataFrame,
    strategy_name: str,
    cfg: dict,
    bt_cfg: BacktestConfig,
    windows: list,
    selection_metric: str = "net_sharpe",
    n_workers: int = 1,
) -> FastSelection:
    """Escolhe os parâmetros de cada janela com um backtest por combinação."""
    if strategy_name not in BAND_STRATEGIES:
        raise ValueError(f"Motor de passagem única só cobre {sorted(BAND_STRATEGIES)}.")

    combos = band_grid(strategy_name, cfg)
    start_idx, end_idx = validation_bounds(df_feat, windows)
    factor = annualization_factor(bt_cfg.interval)
    initargs = (df_feat, strategy_name, bt_cfg, start_idx, end_idx, selection_metric, factor)

    if n_workers > 1 and len(combos) > 1:
        with ProcessPoolExecutor(
            max_workers=min(n_workers, len(combos)),
            initializer=_init_fast_worker,
            initargs=initargs,
        ) as executor:
            outputs = list(executor.map(_eval_combo, combos, chunksize=2))
    else:
        _init_fast_worker(*initargs)
        outputs = [_eval_combo(c) for c in combos]

    n_win = len(windows)
    scores = np.full((len(combos), n_win), -np.inf)
    start_pos = np.zeros((len(combos), n_win))
    for i, out in enumerate(outputs):
        if out is not None:
            scores[i], start_pos[i] = out

    best_index = first_best_index(scores)
    return FastSelection(combos=combos, scores=scores, start_pos=start_pos, best_index=best_index)


# ---------------------------------------------------------------------------
# Comparação com o motor fatiado
# ---------------------------------------------------------------------------

_cmp: dict[str, Any] = {}


def _init_sliced_worker(
    df_feat: pd.DataFrame,
    strategy_name: str,
    bt_cfg: BacktestConfig,
    combos: list[tuple[dict, dict]],
    metric: str,
) -> None:
    _cmp.update(df=df_feat, name=strategy_name, bt=bt_cfg, combos=combos, metric=metric)


def _sliced_window_scores(win: Any) -> tuple[np.ndarray, np.ndarray]:
    """Pontuação de todos os combos pelo método fatiado (backtest de treino+validação da janela)."""
    from .walk_forward import _score_backtest_on_period, _slice_df

    st = _cmp
    df_tv = _slice_df(st["df"], win.train_start, win.val_end)
    ts = pd.DatetimeIndex(df_tv["timestamp"])
    first_val_bar = int(ts.searchsorted(win.val_start, side="left"))
    scores = np.full(len(st["combos"]), -np.inf)
    start_pos = np.zeros(len(st["combos"]))
    for i, (struct, thresh) in enumerate(st["combos"]):
        try:
            z = compute_band_z(df_tv, st["name"], struct["window"], struct["k"])
            signal = signal_from_z(z, name=st["name"], **thresh)
            result = run_backtest(df_tv, signal, st["bt"])
            scores[i] = _score_backtest_on_period(
                result, win.val_start, win.val_end, st["bt"].interval, st["metric"]
            )
            start_pos[i] = result.positions.iloc[min(first_val_bar, len(result.positions) - 1)]
        except Exception:
            continue
    return scores, start_pos


@dataclass
class EngineComparison:
    combos: list[tuple[dict, dict]]
    windows: list
    fast_scores: np.ndarray
    sliced_scores: np.ndarray
    fast_start_pos: np.ndarray
    sliced_start_pos: np.ndarray

    @property
    def fast_best(self) -> np.ndarray:
        return first_best_index(self.fast_scores)

    @property
    def sliced_best(self) -> np.ndarray:
        return first_best_index(self.sliced_scores)


def compare_engines(
    df_feat: pd.DataFrame,
    strategy_name: str,
    cfg: dict,
    bt_cfg: BacktestConfig,
    windows: list,
    selection_metric: str = "net_sharpe",
    n_workers: int = 1,
) -> EngineComparison:
    """Calcula a matriz de pontuações (combo x janela) pelos dois motores."""
    fast = fast_select_band_params(
        df_feat, strategy_name, cfg, bt_cfg, windows, selection_metric, n_workers
    )
    initargs = (df_feat, strategy_name, bt_cfg, fast.combos, selection_metric)
    if n_workers > 1 and len(windows) > 1:
        with ProcessPoolExecutor(
            max_workers=min(n_workers, len(windows)),
            initializer=_init_sliced_worker,
            initargs=initargs,
        ) as executor:
            outputs = list(executor.map(_sliced_window_scores, windows, chunksize=1))
    else:
        _init_sliced_worker(*initargs)
        outputs = [_sliced_window_scores(w) for w in windows]

    sliced_scores = np.column_stack([o[0] for o in outputs])
    sliced_start = np.column_stack([o[1] for o in outputs])
    return EngineComparison(
        combos=fast.combos,
        windows=windows,
        fast_scores=fast.scores,
        sliced_scores=sliced_scores,
        fast_start_pos=fast.start_pos,
        sliced_start_pos=sliced_start,
    )


def _scores_equal(a: np.ndarray, b: np.ndarray, atol: float) -> np.ndarray:
    both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    finite = np.isfinite(a) & np.isfinite(b)
    close = np.zeros(a.shape, dtype=bool)
    close[finite] = np.abs(a[finite] - b[finite]) <= atol
    return both_inf | close


def summarize_comparison(cmp: EngineComparison, atol: float = 1e-9) -> dict[str, Any]:
    """Resumo da equivalência: argmax, pontuações combo a combo e efeito do estado inicial."""
    equal = _scores_equal(cmp.fast_scores, cmp.sliced_scores, atol)
    same_state = cmp.fast_start_pos == cmp.sliced_start_pos
    fast_best, sliced_best = cmp.fast_best, cmp.sliced_best
    n_win = len(cmp.windows)
    cols = np.arange(n_win)

    same_best = fast_best == sliced_best
    # Regret (na régua do motor fatiado) quando o argmax difere
    regret = cmp.sliced_scores[sliced_best, cols] - cmp.sliced_scores[fast_best, cols]
    regret = np.where(np.isfinite(regret), regret, np.nan)

    finite = np.isfinite(cmp.fast_scores) & np.isfinite(cmp.sliced_scores)
    abs_diff = np.abs(cmp.fast_scores - cmp.sliced_scores)

    def rate(mask: np.ndarray, sel: np.ndarray) -> float:
        return float(mask[sel].mean()) if sel.any() else float("nan")

    return {
        "n_windows": n_win,
        "n_combos": len(cmp.combos),
        "same_best_params_rate": float(same_best.mean()),
        "n_windows_best_differs": int((~same_best).sum()),
        "max_regret_when_differs": float(np.nanmax(regret[~same_best])) if (~same_best).any() else 0.0,
        "score_equal_rate_all": float(equal.mean()),
        "same_start_state_rate": float(same_state.mean()),
        "score_equal_rate_same_state": rate(equal, same_state),
        "score_equal_rate_diff_state": rate(equal, ~same_state),
        "max_abs_score_diff": float(abs_diff[finite].max()) if finite.any() else 0.0,
        "max_abs_score_diff_same_state": (
            float(abs_diff[finite & same_state].max()) if (finite & same_state).any() else 0.0
        ),
    }
