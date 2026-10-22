import sys, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src.strategies import _reversal_confirmed_position, _shift_signal
from src.backtest import run_backtest, BacktestConfig, _open_log_returns, _utc_dates
from src.indicators import rolling_ols_last


def tm(f, *a, rep=3, **k):
    best = 1e9
    for _ in range(rep):
        t0 = time.perf_counter(); out = f(*a, **k); best = min(best, time.perf_counter() - t0)
    return best, out


# ---------------------------------------------------------------- A. comprehension vs laco vs numpy
X = np.random.default_rng(0).random(1000).tolist()
Y = np.random.default_rng(1).random(300).tolist()
def loop_append():
    out = []
    for i in X:
        for k in Y:
            out.append(i * k)
    return out
def comp():
    return [i * k for i in X for k in Y]
Xa, Ya = np.array(X), np.array(Y)
def npo():
    return np.multiply.outer(Xa, Ya).ravel()
m = len(X) * len(Y)
print("A. 300 mil produtos")
for name, f in [("for + append", loop_append), ("list comprehension", comp), ("numpy outer", npo)]:
    t, _ = tm(f)
    print(f"   {name:20s} {t/m*1e9:8.1f} ns por elemento")

# ---------------------------------------------------------------- B. sinal por eventos
def position_events(z, el=1.0, es=1.0, xl=0.0, xs=0.0):
    z = np.asarray(z, float); n = len(z)
    if n < 2:
        return np.zeros(n)
    zp, zc = z[:-1], z[1:]
    valid = ~(np.isnan(zp) | np.isnan(zc))
    ev_long = np.flatnonzero(valid & (zp < -el) & (zc >= -el)) + 1
    ev_short = np.flatnonzero(valid & (zp > es) & (zc <= es)) + 1
    x_long = np.flatnonzero(valid & (zc >= xl)) + 1
    x_short = np.flatnonzero(valid & (zc <= xs)) + 1
    delta = np.zeros(n + 1)
    i = 1
    nl, ns = len(ev_long), len(ev_short)
    while True:
        a = np.searchsorted(ev_long, i); b = np.searchsorted(ev_short, i)
        e_l = ev_long[a] if a < nl else n
        e_s = ev_short[b] if b < ns else n
        if e_l == n and e_s == n:
            break
        if e_l <= e_s:
            e, side, X_ = e_l, 1.0, x_long
        else:
            e, side, X_ = e_s, -1.0, x_short
        c = np.searchsorted(X_, e + 1)
        x = X_[c] if c < len(X_) else n
        delta[e] += side
        delta[x] -= side
        if x >= n:
            break
        i = x
    return np.cumsum(delta[:n])


rng = np.random.default_rng(7)
bad = 0
for trial in range(300):
    n = int(rng.integers(5, 400))
    kind = trial % 3
    if kind == 0:
        z = np.cumsum(rng.normal(0, 0.4, n))
    elif kind == 1:
        z = rng.normal(0, 2.0, n)                   # saltos grandes (reentrada na mesma barra)
    else:
        z = np.round(rng.normal(0, 1.5, n) * 2) / 2  # empates exatos nos limiares
    if trial % 4 == 0:
        z[rng.random(n) < 0.1] = np.nan
    el, es = rng.choice([0.5, 1.0, 1.5], 2)
    ref = _reversal_confirmed_position(pd.Series(z), el, es, 0.0, 0.0).to_numpy()
    new = position_events(z, el, es, 0.0, 0.0)
    if not np.array_equal(ref, new):
        bad += 1
print(f"B. sinal por eventos: {300 - bad}/300 casos aleatorios identicos ao laco original (NaN, saltos, empates)")

n = 300_000
rng = np.random.default_rng(1)
ts = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
price = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
s = pd.Series(price)
mid = s.rolling(100).mean(); sd = s.rolling(100).std()
z = ((s - mid) / ((mid + 3.0 * sd) - mid)).to_numpy()
t_ref, ref = tm(lambda: _reversal_confirmed_position(pd.Series(z), 1.0, 1.0).to_numpy())
t_new, new = tm(lambda: position_events(z, 1.0, 1.0))
print(f"   z realista (Bollinger 100, k=3): original {t_ref/n*1e6:.3f} us/barra | eventos {t_new/n*1e6:.3f} us/barra | iguais: {np.array_equal(ref, new)} | trocas: {int(np.abs(np.diff(ref)).sum())}")
z2 = z * 1.0
zz = (s - mid) / sd                                  # k=1 -> muitas operacoes
t_ref2, ref2 = tm(lambda: _reversal_confirmed_position(pd.Series(zz.to_numpy()), 1.0, 1.0).to_numpy())
t_new2, new2 = tm(lambda: position_events(zz.to_numpy(), 1.0, 1.0))
print(f"   muitas operacoes (k=1): original {t_ref2/n*1e6:.3f} | eventos {t_new2/n*1e6:.3f} us/barra | iguais: {np.array_equal(ref2, new2)} | unidades: {int(np.abs(np.diff(ref2)).sum())}")


# ---------------------------------------------------------------- C. backtest vetorizado
def backtest_vec(opens, ts_, sig, fee_bps, slip_bps, close_eod, max_position=1.0, allow_short=True):
    n_ = len(opens)
    desired = np.clip(sig, -max_position, max_position)
    if not allow_short:
        desired = np.clip(desired, 0, max_position)
    pos = desired.copy(); pos[0] = 0.0
    if close_eod:
        d = _utc_dates(ts_)
        eod = np.zeros(n_, dtype=bool)
        eod[:-1] = d[:-1] != d[1:]
        eod[-1] = True
        eod[0] = False
        pos[eod] = 0.0
    prev = np.empty(n_); prev[0] = 0.0; prev[1:] = pos[:-1]
    units = np.abs(pos - prev); units[0] = 0.0
    olr = _open_log_returns(opens)
    gross = np.where(np.isnan(olr), 0.0, prev * olr); gross[0] = 0.0
    ret = gross - units * ((fee_bps + slip_bps) / 1e4)
    return pos, gross, units, ret


opens = price
for eod in (False, True):
    cfg = BacktestConfig(strategy_name="b", symbol="X", interval="1m", fee_bps=2.0, slippage_bps=2.0, close_eod=eod)
    df = pd.DataFrame({"timestamp": ts, "open": opens, "close": opens})
    sig = _shift_signal(pd.Series(ref2))
    t_ref, res = tm(lambda: run_backtest(df, sig, cfg), rep=2)
    t_new, out = tm(lambda: backtest_vec(opens, df["timestamp"], sig.to_numpy(), 2.0, 2.0, eod))
    pos, gross, units, ret = out
    ok = (np.array_equal(res.positions.to_numpy(), pos) and np.array_equal(res.trade_units.to_numpy(), units)
          and np.array_equal(res.gross_returns.to_numpy(), gross) and np.allclose(res.returns.to_numpy(), ret, atol=1e-15, rtol=0))
    print(f"C. backtest close_eod={eod}: original {t_ref/n*1e6:.3f} us/barra | vetorizado {t_new/n*1e6:.3f} us/barra | identico: {ok}")


# ---------------------------------------------------------------- D. OLS rolante por somas acumuladas
def rolling_ols_prefix(y, w):
    y = np.asarray(y, float); n_ = len(y)
    y0 = y - y.mean()
    j = np.arange(n_, dtype=float)
    P1 = np.concatenate(([0.0], np.cumsum(y0)))
    P2 = np.concatenate(([0.0], np.cumsum(y0 * y0)))
    PJ = np.concatenate(([0.0], np.cumsum(j * y0)))
    i = np.arange(w - 1, n_)
    s = i - w + 1
    Sy = P1[i + 1] - P1[s]
    Syy = P2[i + 1] - P2[s]
    Sty = (PJ[i + 1] - PJ[s]) - s * Sy
    tbar = (w - 1) / 2.0
    Stt = w * (w * w - 1) / 12.0
    slope = (Sty - tbar * Sy) / Stt
    ybar = Sy / w
    sse = (Syy - Sy * Sy / w) - slope * slope * Stt
    sse = np.maximum(sse, 0.0)
    fitted = np.full(n_, np.nan); std = np.full(n_, np.nan)
    fitted[w - 1:] = ybar + y.mean() + slope * (w - 1 - tbar)
    std[w - 1:] = np.sqrt(sse / (w - 2))
    return fitted, std

lp = np.log(60000.0) + np.cumsum(rng.normal(0, 1e-3, n))
print("D. regressao rolante (300 mil barras de 1 min, log-preco ~ 11):")
for w in (100, 960):
    t_ref, (f_ref, s_ref) = tm(rolling_ols_last, lp, w, rep=1)
    t_new, (f_new, s_new) = tm(rolling_ols_prefix, lp, w)
    ok = ~np.isnan(f_ref)
    ef = np.max(np.abs(f_ref[ok] - f_new[ok]))
    es_rel = np.max(np.abs(s_ref[ok] - s_new[ok]) / s_ref[ok])
    print(f"   janela {w:4d}: atual {t_ref/n*1e6:.3f} us/barra | somas acumuladas {t_new/n*1e6:.3f} us/barra | erro max fitted {ef:.1e} | erro relativo max do desvio {es_rel:.1e}")
