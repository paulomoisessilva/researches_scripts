"""Regras de saída aplicadas, de forma descritiva, às operações que a amostra teve (estudo v2).

Para cada operação (entrada na abertura da barra ``e``, saída na abertura da barra ``x``) as barras ``e .. x-1`` dão a excursão
contra (MAE) e a favor (MFE) em bps do preço de entrada, usando máxima e mínima. Sobre elas simulam-se regras de saída:

* ``SL``  stop loss de ``p`` bps;                         ``TP``  stop gain (alvo) de ``p`` bps;
* ``TR``  trailing stop a ``p`` bps do pico;              ``BE``  break-even: depois de ``p`` bps a favor, stop na entrada;
* ``ASL`` / ``ATP``  stop e alvo de ``p`` x ATR da barra de entrada;
* ``DSL`` / ``DTR``  stop e trailing stop de ``p`` x ATR da própria barra (o ATR muda a cada barra);
* ``TIME`` saída na abertura da barra ``e + p`` (``p`` em barras).

Hipóteses: a saída acontece no nível; se o preço já abre além de um stop, a saída é na abertura; o alvo sai no nível; se stop e alvo
são tocados na mesma barra vale o stop; a parada por tempo sai na abertura e vence um toque na mesma barra; sem derrapagem, sem
taxas e sem reentrada (as entradas seguintes ficam como foram). Nenhuma delas é verificável só com preços de barra: é uma descrição.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BIG = 10 ** 9
PRIORITY = {"TIME": 0, "SL": 1, "TR": 1, "ASL": 1, "DSL": 1, "DTR": 1, "BE": 1, "TP": 2, "ATP": 2}
ATR_KINDS = ("ASL", "ATP", "DSL", "DTR")
DEFAULT_MAX_ROWS = 8_000_000


@dataclass(frozen=True)
class Rule:
    kind: str
    param: float

    def __post_init__(self) -> None:
        if self.kind not in PRIORITY:
            raise ValueError(f"Regra desconhecida: '{self.kind}'. Suportadas: {sorted(PRIORITY)}")


@dataclass
class Bars:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray | None = None  # ATR em bps do preço, conhecido ANTES de cada barra


def atr_bps(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """ATR (média da amplitude verdadeira, em bps do fechamento anterior) das ``window`` barras anteriores a cada barra."""
    prev = np.concatenate([[np.nan], close[:-1]])
    true_range = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)]) / prev * 1e4
    return pd.Series(true_range).rolling(window, min_periods=window).mean().shift(1).to_numpy()


def bars_from_frame(df: pd.DataFrame, atr_window: int | None = None) -> tuple[pd.DatetimeIndex, Bars]:
    ts = pd.DatetimeIndex(df["timestamp"])
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    return ts, Bars(o, h, l, c, atr_bps(h, l, c, atr_window) if atr_window else None)


def bar_indices(ts: pd.DatetimeIndex, entry_ts, exit_ts) -> tuple[np.ndarray, np.ndarray]:
    """Posição das barras de entrada e de saída; exige correspondência exata e saída depois da entrada."""
    e = ts.get_indexer(pd.DatetimeIndex(entry_ts))
    x = ts.get_indexer(pd.DatetimeIndex(exit_ts))
    if (e < 0).any() or (x < 0).any():
        raise ValueError("Há operações cujo horário não existe nas barras.")
    if (x <= e).any():
        raise ValueError("Há operações com saída na barra de entrada ou antes dela.")
    return e, x


def level_return(level_bps: np.ndarray | float, sign: np.ndarray | float) -> np.ndarray:
    """Retorno log (bps) da operação saindo ao preço ``entrada x (1 + sinal x nível/1e4)``; nível = excursão favorável em bps."""
    return sign * np.log1p(sign * np.asarray(level_bps, dtype=float) / 1e4) * 1e4


def trade_return(bars: Bars, e: np.ndarray, x: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Retorno bruto (bps) das operações, de abertura a abertura (confere com ``gross_bps`` dos resultados)."""
    return sign * np.log(bars.open[x] / bars.open[e]) * 1e4


def _chunks(lengths: np.ndarray, max_rows: int) -> list[tuple[int, int]]:
    n = len(lengths)
    cuts = np.searchsorted(np.cumsum(lengths), np.arange(max_rows, lengths.sum() + max_rows, max_rows))
    edges = np.unique(np.concatenate([[0], np.minimum(cuts + 1, n), [n]]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if a < b]


def _expand(bars: Bars, e: np.ndarray, lengths: np.ndarray, sign: np.ndarray) -> dict:
    """Uma linha por (operação, barra dentro da operação): excursões contra e a favor, abertura e pico anterior a favor."""
    starts = np.cumsum(lengths) - lengths
    rows_total = int(lengths.sum())
    tid = np.repeat(np.arange(len(lengths)), lengths)
    off = np.arange(rows_total) - starts[tid]
    rows = e[tid] + off
    p0 = bars.open[e][tid]
    sg = sign[tid]
    adv = np.where(sg > 0, (p0 - bars.low[rows]) / p0, (bars.high[rows] - p0) / p0) * 1e4
    fav = np.where(sg > 0, (bars.high[rows] - p0) / p0, (p0 - bars.low[rows]) / p0) * 1e4
    fav_open = sg * (bars.open[rows] - p0) / p0 * 1e4
    cummax = pd.Series(fav).groupby(tid).cummax().to_numpy()
    peak = np.empty(rows_total)
    peak[1:] = cummax[:-1]
    peak[starts] = 0.0
    return {"starts": starts, "tid": tid, "off": off, "rows": rows, "adv": adv, "fav": fav, "fav_open": fav_open, "peak": peak}


def excursions(bars: Bars, e: np.ndarray, x: np.ndarray, sign: np.ndarray, max_rows: int = DEFAULT_MAX_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """(MAE, MFE) em bps do preço de entrada, das barras ``e .. x-1`` de cada operação."""
    lengths = x - e
    mae, mfe = np.empty(len(e)), np.empty(len(e))
    for a, b in _chunks(lengths, max_rows):
        ex = _expand(bars, e[a:b], lengths[a:b], sign[a:b])
        mae[a:b] = np.maximum.reduceat(ex["adv"], ex["starts"])
        mfe[a:b] = np.maximum.reduceat(ex["fav"], ex["starts"])
    return mae, mfe


def rule_outcomes(
    bars: Bars,
    e: np.ndarray,
    x: np.ndarray,
    sign: np.ndarray,
    rules: list[Rule],
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[Rule, tuple[np.ndarray, np.ndarray]]:
    """Para cada regra, por operação: a barra (offset desde a entrada) em que dispara (``BIG`` se não dispara) e o retorno da saída."""
    n = len(e)
    lengths = x - e
    out = {r: (np.full(n, BIG, dtype=np.int64), np.zeros(n)) for r in rules}
    if any(r.kind in ATR_KINDS for r in rules) and bars.atr is None:
        raise ValueError("Regras por ATR exigem Bars.atr.")

    for r in (r for r in rules if r.kind == "TIME"):
        nb = max(1, int(r.param))
        trig = lengths > nb
        idx = np.minimum(e + nb, len(bars.open) - 1)
        out[r][0][trig] = nb
        out[r][1][:] = np.where(trig, sign * np.log(bars.open[idx] / bars.open[e]) * 1e4, 0.0)

    price_rules = [r for r in rules if r.kind != "TIME"]
    if not price_rules:
        return out
    for a, b in _chunks(lengths, max_rows):
        ex = _expand(bars, e[a:b], lengths[a:b], sign[a:b])
        starts, tid, off = ex["starts"], ex["tid"], ex["off"]
        adv, fav, fav_open, peak = ex["adv"], ex["fav"], ex["fav_open"], ex["peak"]
        sgn = sign[a:b]
        atr_entry = bars.atr[e[a:b]][tid] if bars.atr is not None else None
        atr_row = bars.atr[ex["rows"]] if bars.atr is not None else None
        for r in price_rules:
            if r.kind == "SL":
                cond, level, gap = adv >= r.param, np.full(len(adv), -float(r.param)), True
            elif r.kind == "TP":
                cond, level, gap = fav >= r.param, np.full(len(adv), float(r.param)), False
            elif r.kind == "TR":
                cond, level, gap = (peak + adv) >= r.param, peak - r.param, True
            elif r.kind == "BE":
                cond, level, gap = (peak >= r.param) & (adv >= 0), np.zeros(len(adv)), True
            elif r.kind == "ASL":
                cond, level, gap = adv >= r.param * atr_entry, -r.param * atr_entry, True
            elif r.kind == "DSL":
                cond, level, gap = adv >= r.param * atr_row, -r.param * atr_row, True
            elif r.kind == "DTR":
                cond, level, gap = (peak + adv) >= r.param * atr_row, peak - r.param * atr_row, True
            else:  # ATP
                cond, level, gap = fav >= r.param * atr_entry, r.param * atr_entry, False
            first = np.minimum.reduceat(np.where(cond, off, BIG), starts)
            trig = first < BIG
            lv = np.zeros(b - a)
            ix = starts[trig] + first[trig]
            lv[trig] = np.minimum(level[ix], fav_open[ix]) if gap else level[ix]
            out[r][0][a:b] = first
            out[r][1][a:b] = np.where(trig, level_return(lv, sgn), 0.0)
    return out


def combine(
    outcomes: dict[Rule, tuple[np.ndarray, np.ndarray]],
    rules: list[Rule],
    gross: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorno por operação com várias regras juntas: vale a que dispara primeiro (empate: tempo, depois stops, depois alvos).

    Devolve ``(retorno, houve_saida_antecipada)``; sem nenhuma regra disparada vale o retorno original ``gross``.
    """
    if not rules:
        return gross.copy(), np.zeros(len(gross), dtype=bool)
    none = BIG * 10
    keys = np.stack([np.where(outcomes[r][0] < BIG, outcomes[r][0] * 10 + PRIORITY[r.kind], none) for r in rules])
    pick = keys.argmin(axis=0)
    hit = keys.min(axis=0) < none
    chosen = np.stack([outcomes[r][1] for r in rules])[pick, np.arange(len(gross))]
    return np.where(hit, chosen, gross), hit
