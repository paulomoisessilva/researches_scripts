"""Núcleo vetorizado (NumPy) do sinal de reversão confirmada e do backtest.

Mesmas regras dos laços de referência (``strategies._reversal_confirmed_position_loop`` e
``backtest.run_backtest_loop``), sem passar por todas as barras em Python.

Sinal. As operações são raras (da ordem de 0,3 a 3 por 100 barras), então o estado só muda em
"eventos". Os eventos de entrada e as barras que satisfazem a saída saem de comparações
vetorizadas; um laço curto percorre só as operações: acha a próxima entrada com a posição zerada,
acha a primeira barra de saída depois dela (``searchsorted``) e recomeça dali. A posição sai de uma
soma acumulada. A saída é idêntica à do laço original, inclusive com NaN, saltos de z que saem de
uma posição e entram na outra na mesma barra, e empates exatos nos limiares.

Backtest. Com sinal sem NaN e sem ``max_holding_bars`` a posição não depende de estado: é o sinal
já deslocado (zerado nas barras de fechamento do dia, quando houver), e retorno bruto, unidades
negociadas e custo saem de operações em arrays.
"""

from __future__ import annotations

import numpy as np


def open_log_returns(opens: np.ndarray) -> np.ndarray:
    """ln(open_t / open_{t-1}); índice 0 = 0; valores não finitos viram NaN."""
    out = np.zeros(len(opens))
    if len(opens) < 2:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = np.log(opens[1:] / opens[:-1])
    out[~np.isfinite(out)] = np.nan
    return out


def _threshold(value, n: int) -> float | np.ndarray:
    """Limiar escalar ou por barra (vetor de tamanho ``n``)."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.shape != (n,):
        raise ValueError(f"Limiar por barra deve ter {n} valores, recebeu {arr.shape}.")
    return arr


def positions_from_pairs(
    z_prev: np.ndarray,
    z_curr: np.ndarray,
    entry_long_z: float | np.ndarray = 0.8,
    entry_short_z: float | np.ndarray = 0.8,
    exit_long_z: float | np.ndarray = 0.0,
    exit_short_z: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Posição (-1, 0, +1) da reversão confirmada, a partir dos pares (z anterior, z atual) por barra.

    Regra (a mesma do laço de referência), avaliada a partir da barra 1; a barra 0 fica zerada:

    * long : ``z_prev < -entry_long_z`` e ``z_curr >= -entry_long_z``; sai quando ``z_curr >= exit_long_z``
    * short: ``z_prev > +entry_short_z`` e ``z_curr <= +entry_short_z``; sai quando ``z_curr <= exit_short_z``
    * barra com NaN em ``z_prev`` ou ``z_curr``: nada muda (a posição se mantém)
    * saída e nova entrada podem acontecer na mesma barra

    ``z_prev[i]`` não precisa ser ``z_curr[i-1]``: a trajetória contínua com troca de parâmetros usa o
    z da regra nova nos dois lados do par. Os limiares podem ser escalares ou vetores por barra.
    """
    zp = np.asarray(z_prev, dtype=float)
    zc = np.asarray(z_curr, dtype=float)
    n = len(zc)
    if len(zp) != n:
        raise ValueError(f"z_prev ({len(zp)}) e z_curr ({n}) têm tamanhos diferentes.")
    pos = np.zeros(n)
    if n < 2:
        return pos

    el = _threshold(entry_long_z, n)
    es = _threshold(entry_short_z, n)
    xl = _threshold(exit_long_z, n)
    xs = _threshold(exit_short_z, n)

    valid = ~(np.isnan(zp) | np.isnan(zc))
    valid[0] = False
    ev_long = np.flatnonzero(valid & (zp < -el) & (zc >= -el))
    ev_short = np.flatnonzero(valid & (zp > es) & (zc <= es))
    if len(ev_long) == 0 and len(ev_short) == 0:
        return pos
    x_long = np.flatnonzero(valid & (zc >= xl))
    x_short = np.flatnonzero(valid & (zc <= xs))

    # Eventos são poucos: listas Python deixam o avanço dos ponteiros barato.
    evl = ev_long.tolist()
    evs = ev_short.tolist()
    n_l, n_s = len(evl), len(evs)
    big = n  # "não há mais evento"
    delta = np.zeros(n + 1)
    a = b = 0
    i = 1
    while True:
        while a < n_l and evl[a] < i:
            a += 1
        while b < n_s and evs[b] < i:
            b += 1
        e_l = evl[a] if a < n_l else big
        e_s = evs[b] if b < n_s else big
        if e_l == big and e_s == big:
            break
        if e_l <= e_s:  # empate: long tem prioridade (if/elif do laço de referência)
            e, side, exits = e_l, 1.0, x_long
        else:
            e, side, exits = e_s, -1.0, x_short
        c = int(np.searchsorted(exits, e + 1))
        x = int(exits[c]) if c < len(exits) else big
        delta[e] += side
        delta[x] -= side
        if x >= n:
            break
        i = x  # zerado depois da saída; um evento na própria barra x ainda vale
    return np.cumsum(delta[:n])


def reversal_position_array(
    z: np.ndarray,
    entry_long_z: float = 0.8,
    entry_short_z: float = 0.8,
    exit_long_z: float = 0.0,
    exit_short_z: float = 0.0,
) -> np.ndarray:
    """Posição da reversão confirmada para uma série z única (``z_prev[i] = z[i-1]``)."""
    zv = np.asarray(z, dtype=float)
    zp = np.empty_like(zv)
    if len(zv):
        zp[0] = np.nan
        zp[1:] = zv[:-1]
    return positions_from_pairs(zp, zv, entry_long_z, entry_short_z, exit_long_z, exit_short_z)


def backtest_arrays(
    opens: np.ndarray,
    desired: np.ndarray,
    fee_bps: float,
    slippage_bps: float,
    max_position: float = 1.0,
    allow_short: bool = True,
    eod_last_bar: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Posição, retorno bruto, retorno líquido e unidades negociadas, sem laço por barra.

    ``desired`` é a posição desejada na barra t (o sinal já deslocado em uma barra). Vale quando o
    sinal não tem NaN e não há ``max_holding_bars``; fora disso use ``backtest.run_backtest_loop``.
    ``eod_last_bar`` marca as barras em que a posição é forçada a zero (fechamento do dia).
    """
    opens = np.asarray(opens, dtype=float)
    n = len(opens)
    pos_desired = np.clip(np.asarray(desired, dtype=float), -max_position, max_position)
    if not allow_short:
        pos_desired = np.clip(pos_desired, 0, max_position)

    position = pos_desired.copy()
    if n:
        position[0] = 0.0
    if eod_last_bar is not None:
        forced = np.asarray(eod_last_bar, dtype=bool).copy()
        if n:
            forced[0] = False  # o laço de referência começa na barra 1
        position[forced] = 0.0

    prev = np.zeros(n)
    prev[1:] = position[:-1]
    units = np.abs(position - prev)

    open_ret = open_log_returns(opens)
    gross = np.where(np.isnan(open_ret), 0.0, prev * open_ret)
    per_side_rate = fee_bps / 10_000.0 + slippage_bps / 10_000.0
    net = gross - units * per_side_rate
    return {"position": position, "prev": prev, "gross": gross, "net": net, "units": units}
