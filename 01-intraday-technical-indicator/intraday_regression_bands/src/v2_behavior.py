"""O que explica a chance de uma operação dar lucro bruto: calendário, volume, volatilidade e persistência (estudo v2, descritivo).

O resultado é binário (lucro bruto > 0) ou o bruto em bps. Os modelos são logits (``statsmodels`` GLM binomial) e mínimos quadrados, com
erros-padrão robustos agrupados por semana ISO, que descrevem o que a amostra mostrou; nada aqui infere além dela. ``prepare_trades``
junta a base de operações às features de mercado; as funções devolvem tabelas (``DataFrame``) prontas para gravar.

Blocos de variáveis nos modelos: ``W`` dias da semana (segunda = referência), ``H`` horas UTC (hora 0 = referência), ``V`` volume e
volatilidade da última hora (nível), ``V3`` os de 1 h, 3 h e 24 h, ``R`` taxa de lucro das operações fechadas há pouco.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from .v2_trades import LOOKBACKS, OOS_START

WEEKDAYS = ["seg", "ter", "qua", "qui", "sex", "sab", "dom"]
HALF_DAYS = 987            # a 2ª metade dos 1.974 dias de teste começa 987 dias depois do primeiro
LEVEL_COLS = ["z_rv", "z_qv"]
REGIME_COLS = ["z_rv", "z_qv", "z_rv3", "z_qv3", "z_rv24", "z_qv24"]


# ---------------------------------------------------------------------------
# Preparação
# ---------------------------------------------------------------------------


def prepare_trades(
    trades: pd.DataFrame,
    feats: pd.DataFrame | None = None,
    oos_start: pd.Timestamp = OOS_START,
    require_features: bool = False,
) -> pd.DataFrame:
    """Acrescenta calendário (dia da semana, hora, ano, semana ISO), ``win`` e, se houver, as features de mercado da barra de entrada."""
    g = trades.reset_index(drop=True).copy()
    ts = pd.DatetimeIndex(g["entry_ts"])
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    g["wd"], g["hr"], g["year"] = ts.dayofweek.to_numpy(), ts.hour.to_numpy(), ts.year.to_numpy()
    iso = ts.isocalendar()
    g["week"] = pd.factorize(iso["year"].astype(int).to_numpy() * 100 + iso["week"].astype(int).to_numpy())[0]
    g["weekend"] = (g["wd"] >= 5).astype(float)
    g["sess"] = g["hr"] // 8
    g["win"] = (g["gross_bps"] > 0).astype(float)
    g["strat"] = (g["strategy"].astype(str) == "regression_bands").astype(float)
    g["second_half"] = np.asarray(ts >= oos_start + pd.Timedelta(days=HALF_DAYS))
    if feats is not None:
        m = feats.reindex(ts)
        for col in feats.columns:
            g[col] = m[col].to_numpy()
        if require_features:
            g = g.dropna(subset=[c for c in ("z_rv", "z_qv", "s_rv", "s_qv") if c in g.columns]).reset_index(drop=True)
    return g


def lookback_effects(g: pd.DataFrame, strategy_effect: bool = True) -> np.ndarray:
    """Efeitos fixos de lookback (e de estratégia): ``(n, k)``; colunas constantes dentro de um subgrupo são descartadas."""
    cols = []
    if strategy_effect and g["strat"].nunique() > 1:
        cols.append(g["strat"].to_numpy())
    lbs = sorted(g["lookback"].unique())
    cols += [(g["lookback"].to_numpy() == lb).astype(float) for lb in lbs[1:]]
    return np.column_stack(cols) if cols else np.empty((len(g), 0))


def design(g: pd.DataFrame, blocks: str = "", cols: list[str] | None = None, fixed: bool = True) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Matriz de desenho e posição das colunas de cada bloco. ``blocks``: letras ``W`` e ``H``; ``cols``: colunas numéricas extras."""
    columns = [np.ones(len(g))]
    pos: dict[str, list[int]] = {}
    k = 1
    if "W" in blocks:
        for d in range(1, 7):
            columns.append((g["wd"].to_numpy() == d).astype(float))
        pos["W"] = list(range(k, k + 6)); k += 6
    if "H" in blocks:
        for h in range(1, 24):
            columns.append((g["hr"].to_numpy() == h).astype(float))
        pos["H"] = list(range(k, k + 23)); k += 23
    for c in cols or []:
        columns.append(g[c].to_numpy(dtype=float))
        pos[c] = [k]; k += 1
    X = np.column_stack(columns)
    if fixed:
        X = np.column_stack([X, lookback_effects(g)])
    return X, pos


def fit_logit(y: np.ndarray, X: np.ndarray, groups: np.ndarray | None = None):
    glm = sm.GLM(y, X, family=sm.families.Binomial())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return glm.fit(cov_type="cluster", cov_kwds={"groups": groups}) if groups is not None else glm.fit()


def fit_ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})


def wald_p(res, idx: list[int]) -> float:
    """p do teste conjunto (qui-quadrado) de que os coeficientes em ``idx`` são todos zero, com a covariância do ajuste."""
    R = np.zeros((len(idx), len(res.params)))
    for r, i in enumerate(idx):
        R[r, i] = 1.0
    return float(res.wald_test(R, scalar=True, use_f=False).pvalue)


def rank_auc(score: np.ndarray, y: np.ndarray) -> float:
    n1 = float(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def oos_scores(g: pd.DataFrame, blocks: str, cols: list[str] | None = None, fixed: bool = True) -> dict[str, float]:
    """Ajusta na 1ª metade dos dias e avalia na 2ª: ganho de log-verossimilhança por 10 mil operações sobre o modelo só com efeitos fixos."""
    y = g["win"].to_numpy()
    tr = ~g["second_half"].to_numpy(); te = ~tr
    X0, _ = design(g, "", None, fixed)
    base = fit_logit(y[tr], X0[tr]); p0 = np.clip(base.predict(X0[te]), 1e-9, 1 - 1e-9)
    ll0 = float((y[te] * np.log(p0) + (1 - y[te]) * np.log(1 - p0)).sum())
    X, _ = design(g, blocks, cols, fixed)
    res = fit_logit(y[tr], X[tr]); p = np.clip(res.predict(X[te]), 1e-9, 1 - 1e-9)
    ll = float((y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)).sum())
    return {"dLL_x10k": (ll - ll0) / te.sum() * 1e4, "AUC": rank_auc(p, y[te])}


# ---------------------------------------------------------------------------
# Calendário
# ---------------------------------------------------------------------------


def calendar_summary(g: pd.DataFrame, fixed: bool = False) -> dict[str, float]:
    """Um grupo (ativo x frequência x estratégia): dia da semana e hora contra lucro bruto > 0 e contra o bruto em bps.

    Logit M1 (dia), M2 (hora), M3 (os dois); p conjuntos de Wald agrupados por semana; sábado contra segunda (razão de chances e, no
    MQO com o bruto cortado nos percentis 1 e 99, diferença em bps); sessões de 8 h + fim de semana; AUC fora da amostra.
    """
    y, grp = g["win"].to_numpy(), g["week"].to_numpy()
    out: dict[str, float] = {"n": len(g), "win_pct": 100 * float(y.mean()), "mean_gross_bps": float(g["gross_bps"].mean())}
    X1, p1 = design(g, "W", fixed=fixed); r1 = fit_logit(y, X1, grp)
    X2, p2 = design(g, "H", fixed=fixed); r2 = fit_logit(y, X2, grp)
    X3, p3 = design(g, "WH", fixed=fixed); r3 = fit_logit(y, X3, grp)
    out.update({"p_weekday": wald_p(r1, p1["W"]), "p_hour": wald_p(r2, p2["H"]),
                "p_weekday_given_hour": wald_p(r3, p3["W"]), "p_hour_given_weekday": wald_p(r3, p3["H"])})
    sat = p3["W"][4]  # sábado = dia 5 (índice 4 entre os 6 dummies)
    out["OR_saturday"] = float(np.exp(r3.params[sat])); out["p_saturday"] = float(r3.pvalues[sat])
    sess = g.assign(s1=(g["sess"] == 1).astype(float), s2=(g["sess"] == 2).astype(float))
    Xs, ps = design(sess, "", ["s1", "s2", "weekend"], fixed=fixed); rs = fit_logit(y, Xs, grp)
    out["p_session"] = wald_p(rs, ps["s1"] + ps["s2"]); out["OR_weekend"] = float(np.exp(rs.params[ps["weekend"][0]]))
    gross = g["gross_bps"].to_numpy(); lo, hi = np.percentile(gross, [1, 99]); gw = np.clip(gross, lo, hi)
    o = fit_ols(gw, X3, grp)
    out["sat_minus_mon_bps"] = float(o.params[sat]); out["p_ols_weekday"] = wald_p(o, p3["W"]); out["p_ols_hour"] = wald_p(o, p3["H"])
    for key, blocks, cols in (("weekday", "W", None), ("hour", "H", None), ("session", "", ["s1", "s2", "weekend"])):
        out[f"AUC_oos_{key}"] = oos_scores(sess, blocks, cols, fixed)["AUC"]
    return out


def weekend_by_year(g: pd.DataFrame) -> pd.DataFrame:
    """Diferença na % de operações com lucro entre fim de semana e dias úteis, por ano (modelo de probabilidade linear, agrupado por semana)."""
    rows = []
    for year, gy in g.groupby("year"):
        if len(gy) < 200:
            continue
        X = np.column_stack([np.ones(len(gy)), gy["weekend"].to_numpy()])
        r = fit_ols(gy["win"].to_numpy(), X, gy["week"].to_numpy())
        rows.append({"year": int(year), "n": len(gy), "weekend_pp": 100 * float(r.params[1]), "p": float(r.pvalues[1])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Volume e volatilidade
# ---------------------------------------------------------------------------


def regime_effects(g: pd.DataFrame) -> dict[str, float]:
    """Razão de chances (por desvio padrão) do volume e da volatilidade da última hora, em nível e em "surpresa", com os dois no modelo."""
    y, grp = g["win"].to_numpy(), g["week"].to_numpy()
    out: dict[str, float] = {"n": len(g)}
    for tag, cols in (("level", ["z_rv", "z_qv"]), ("surprise", ["s_rv", "s_qv"])):
        X, pos = design(g, "", cols); r = fit_logit(y, X, grp)
        out[f"OR_vol_{tag}"], out[f"p_vol_{tag}"] = float(np.exp(r.params[pos[cols[0]][0]])), float(r.pvalues[pos[cols[0]][0]])
        out[f"OR_volume_{tag}"], out[f"p_volume_{tag}"] = float(np.exp(r.params[pos[cols[1]][0]])), float(r.pvalues[pos[cols[1]][0]])
    return out


def weekend_mediation(g: pd.DataFrame) -> dict[str, float]:
    """O efeito do fim de semana com e sem controle de volume e volatilidade da última hora."""
    y, grp = g["win"].to_numpy(), g["week"].to_numpy()
    out = {}
    for tag, cols in (("raw", ["weekend"]), ("controlled", ["weekend", "z_rv", "z_qv"])):
        X, pos = design(g, "", cols); r = fit_logit(y, X, grp); i = pos["weekend"][0]
        out[f"OR_weekend_{tag}"], out[f"p_weekend_{tag}"] = float(np.exp(r.params[i])), float(r.pvalues[i])
    return out


def decile_table(g: pd.DataFrame, col: str, bins: int = 10) -> pd.Series:
    """% de operações com lucro por faixa (decil por padrão) de ``col``."""
    q = pd.qcut(g[col], bins, labels=False, duplicates="drop")
    return pd.Series({f"D{int(i) + 1}": 100 * g["win"][q == i].mean() for i in sorted(q.dropna().unique())})


def threshold_shape(g: pd.DataFrame, col: str, other: str, percentiles: tuple[int, ...] = (10, 25, 40, 50, 60, 75, 90)) -> dict[str, float]:
    """Modelo linear contra modelo com um ponto de quebra (o melhor entre os percentis): razão de chances por desvio padrão abaixo e acima.

    O ponto é escolhido nos próprios dados (descritivo); ``p_change`` testa a mudança de inclinação com erros agrupados por semana.
    """
    y, grp = g["win"].to_numpy(), g["week"].to_numpy()
    X, pos = design(g, "", [col, other]); lin = fit_logit(y, X)
    best = None
    for p in percentiles:
        k = float(np.percentile(g[col], p)); h = g.assign(_hinge=np.maximum(0.0, g[col].to_numpy() - k))
        Xh, _ = design(h, "", [col, other, "_hinge"]); r = fit_logit(y, Xh)
        if best is None or r.llf > best[0]:
            best = (r.llf, p, k)
    _, p, k = best
    h = g.assign(_hinge=np.maximum(0.0, g[col].to_numpy() - k))
    Xh, ph = design(h, "", [col, other, "_hinge"]); r = fit_logit(y, Xh, grp)
    b1, b2 = r.params[ph[col][0]], r.params[ph["_hinge"][0]]
    return {"OR_linear": float(np.exp(lin.params[pos[col][0]])), "knot_percentile": p, "OR_below": float(np.exp(b1)),
            "OR_above": float(np.exp(b1 + b2)), "p_change": float(r.pvalues[ph["_hinge"][0]]), "gain_vs_linear_x10k": (best[0] - lin.llf) / len(g) * 1e4}


def block_contributions(g: pd.DataFrame) -> dict[str, float]:
    """Quanto explicam volume/volatilidade (V), dia da semana (W) e hora (H): ganho de log-verossimilhança por 10 mil operações, contribuição
    única de cada bloco no modelo completo (% do ganho total), p de Wald de cada bloco e ganho fora da amostra."""
    y, grp, n = g["win"].to_numpy(), g["week"].to_numpy(), len(g)
    ll = {}
    specs = {"": ("", None), "V": ("", LEVEL_COLS), "W": ("W", None), "H": ("H", None), "VW": ("W", LEVEL_COLS), "VH": ("H", LEVEL_COLS),
             "WH": ("WH", None), "VWH": ("WH", LEVEL_COLS)}
    for name, (blocks, cols) in specs.items():
        X, _ = design(g, blocks, cols); ll[name] = fit_logit(y, X).llf
    tot = ll["VWH"] - ll[""]
    out = {"n": n, "gain_V": (ll["V"] - ll[""]) / n * 1e4, "gain_W": (ll["W"] - ll[""]) / n * 1e4, "gain_H": (ll["H"] - ll[""]) / n * 1e4,
           "gain_all": tot / n * 1e4, "unique_V_pct": 100 * (ll["VWH"] - ll["WH"]) / tot, "unique_W_pct": 100 * (ll["VWH"] - ll["VH"]) / tot,
           "unique_H_pct": 100 * (ll["VWH"] - ll["VW"]) / tot}
    X, pos = design(g, "WH", LEVEL_COLS); r = fit_logit(y, X, grp)
    out.update({"p_V": wald_p(r, pos["z_rv"] + pos["z_qv"]), "p_W": wald_p(r, pos["W"]), "p_H": wald_p(r, pos["H"])})
    for name, (blocks, cols) in {"V": ("", LEVEL_COLS), "W": ("W", None), "H": ("H", None), "all": ("WH", LEVEL_COLS)}.items():
        out[f"oos_dLL_{name}"] = oos_scores(g, blocks, cols)["dLL_x10k"]
    return out


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------


def hourly_counts(entry_ts, win: np.ndarray, n_hours: int, start: pd.Timestamp = OOS_START) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entradas (n) e entradas com lucro (w) por hora de relógio desde ``start``; devolve ``(hora_de_cada_operação, n, w)``."""
    ts = pd.DatetimeIndex(entry_ts)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    h = ((ts - start) // pd.Timedelta(hours=1)).to_numpy()
    keep = (h >= 0) & (h < n_hours)
    n = np.bincount(h[keep], minlength=n_hours).astype(float)
    w = np.bincount(h[keep], weights=win[keep], minlength=n_hours)
    return h, n, w


def pearson_residuals(n: np.ndarray, w: np.ndarray, p: float | np.ndarray) -> np.ndarray:
    """Resíduo de Pearson binomial por hora ((w - n p) / raiz(n p (1 - p))); NaN nas horas sem entradas."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (w - n * p) / np.sqrt(n * p * (1 - p))
    z[n < 1] = np.nan
    return z


def autocorrelation(z: np.ndarray, lags: tuple[int, ...] = (1, 24, 168)) -> dict[int, float]:
    """Correlação da série consigo mesma nas defasagens dadas, só com pares sem NaN."""
    out = {}
    for k in lags:
        a, b = z[k:], z[:-k]
        ok = np.isfinite(a) & np.isfinite(b)
        out[k] = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 30 else float("nan")
    return out


def hourly_acf(trades: pd.DataFrame, n_hours: int, lags: tuple[int, ...] = (1, 24, 168)) -> dict[int, float]:
    """Autocorrelação do acerto por hora de entrada de um conjunto de operações (resíduo de Pearson sobre a taxa média do conjunto)."""
    win = (trades["gross_bps"].to_numpy() > 0).astype(float)
    _, n, w = hourly_counts(trades["entry_ts"], win, n_hours)
    p = w.sum() / n.sum()
    return autocorrelation(pearson_residuals(n, w, p), lags)


def acf_overlap_summary(group: pd.DataFrame, n_hours: int) -> dict[str, float]:
    """A autocorrelação por hora com as 14 trajetórias somadas, sem operações repetidas e em cada trajetória isolada.

    Mostra se a persistência entre horas vizinhas vem de variantes que ganham e perdem juntas (some numa trajetória só) ou de cada
    trajetória em si.
    """
    row: dict[str, float] = {"n": len(group)}
    for tag, sub in (("pooled", group), ("dedup", group.drop_duplicates(subset=["entry_ts", "exit_ts", "side"]))):
        a = hourly_acf(sub, n_hours)
        row.update({f"{tag}_1h": a[1], f"{tag}_24h": a[24], f"{tag}_168h": a[168]})
        if tag == "dedup":
            row["dedup_share_pct"] = 100 * len(sub) / len(group)
    singles = [hourly_acf(gp, n_hours) for _, gp in group.groupby([group["strategy"].astype(str), "lookback"])]
    one = np.array([s[1] for s in singles])
    row.update({"single_1h_mean": float(one.mean()), "single_1h_min": float(one.min()), "single_1h_max": float(one.max()),
                "single_24h_mean": float(np.mean([s[24] for s in singles])), "single_168h_mean": float(np.mean([s[168] for s in singles]))})
    return row


def shrunk_rate(w: np.ndarray, n: np.ndarray, pbar: float, prior: float = 10.0) -> np.ndarray:
    """Taxa de lucro encolhida para a média do grupo (``prior`` operações de peso)."""
    return (w + prior * pbar) / (n + prior)


def lagged_hour_rate(rate: np.ndarray, hour_idx: np.ndarray, lag: int) -> np.ndarray:
    """Taxa da hora ``lag`` horas antes de cada operação (NaN se antes do início)."""
    out = np.full(len(hour_idx), np.nan)
    ok = hour_idx - lag >= 0
    out[ok] = rate[hour_idx[ok] - lag]
    return out


def seasonal_lag_tests(g: pd.DataFrame, n_hours: int) -> dict[str, float]:
    """O acerto da mesma hora do dia anterior (24 h) e da mesma hora da semana anterior (168 h) prevê o da próxima entrada?

    Razões de chances por desvio padrão, só com as defasagens e com calendário e volume/volatilidade, e ganho fora da amostra.
    """
    h, n, w = hourly_counts(g["entry_ts"], g["win"].to_numpy(), n_hours)
    pbar = w.sum() / n.sum()
    rate = shrunk_rate(w, n, pbar)
    sd = rate[n > 0].std()
    g = g.assign(lag24=(lagged_hour_rate(rate, h, 24) - pbar) / sd, lag168=(lagged_hour_rate(rate, h, 168) - pbar) / sd)
    g = g.dropna(subset=["lag24", "lag168"]).reset_index(drop=True)
    y, grp = g["win"].to_numpy(), g["week"].to_numpy()
    out: dict[str, float] = {"n": len(g)}
    for tag, blocks, extra in (("lags_only", "", []), ("with_calendar_regime", "WH", LEVEL_COLS)):
        X, pos = design(g, blocks, ["lag24", "lag168"] + extra); r = fit_logit(y, X, grp)
        for k in ("lag24", "lag168"):
            out[f"OR_{k}_{tag}"], out[f"p_{k}_{tag}"] = float(np.exp(r.params[pos[k][0]])), float(r.pvalues[pos[k][0]])
    base = oos_scores(g, "WH", LEVEL_COLS)["dLL_x10k"]
    out["oos_dLL_lags"] = oos_scores(g, "WH", LEVEL_COLS + ["lag24", "lag168"])["dLL_x10k"] - base
    return out


def recent_closed_rate(entry: np.ndarray, exit_: np.ndarray, win: np.ndarray, window: np.timedelta64) -> tuple[np.ndarray, np.ndarray]:
    """Para cada entrada: vitórias e total das operações do conjunto que FECHARAM em ``(entrada - window, entrada]`` (só o que já se sabe)."""
    order = np.argsort(exit_)
    ex, w = exit_[order], win[order]
    cum = np.concatenate([[0.0], np.cumsum(w)])
    hi = np.searchsorted(ex, entry, side="right")
    lo = np.searchsorted(ex, entry - window, side="right")
    return cum[hi] - cum[lo], (hi - lo).astype(float)


def recent_performance_tests(g: pd.DataFrame, exit_ts, prior: float = 10.0) -> dict[str, float]:
    """A taxa de lucro das operações fechadas nas últimas 1 h e 3 h prevê o acerto da próxima entrada, depois de controlar o regime?

    ``g`` deve conter as features de 1 h, 3 h e 24 h (``REGIME_COLS``). Razão de chances por desvio padrão sozinha, com o regime e com
    regime e calendário; contribuição única do histórico recente e ganho fora da amostra.
    """
    entry = pd.DatetimeIndex(g["entry_ts"]).tz_convert("UTC").to_numpy("datetime64[ns]")
    exit_ = pd.DatetimeIndex(exit_ts).tz_convert("UTC").to_numpy("datetime64[ns]")
    win = g["win"].to_numpy(); pbar = win.mean()
    g = g.copy()
    for lab, hours in (("1", 1), ("3", 3)):
        w_, n_ = recent_closed_rate(entry, exit_, win, np.timedelta64(hours, "h"))
        r = shrunk_rate(w_, n_, pbar, prior)
        g["z_r" + lab] = (r - pbar) / r.std()
        g["n_r" + lab] = n_
    g = g.dropna(subset=REGIME_COLS).reset_index(drop=True)
    y, grp = g["win"].to_numpy(), g["week"].to_numpy()
    out: dict[str, float] = {"n": len(g)}
    for tag, blocks, cols in (("alone", "", ["z_r1", "z_r3"]), ("regime", "", ["z_r1", "z_r3"] + REGIME_COLS),
                              ("regime_calendar", "WH", ["z_r1", "z_r3"] + REGIME_COLS)):
        X, pos = design(g, blocks, cols); r = fit_logit(y, X, grp)
        for k in ("z_r1", "z_r3"):
            out[f"OR_{k}_{tag}"], out[f"p_{k}_{tag}"] = float(np.exp(r.params[pos[k][0]])), float(r.pvalues[pos[k][0]])
    full_cols = ["z_r1", "z_r3"] + REGIME_COLS
    X, pos = design(g, "WH", full_cols); r = fit_logit(y, X, grp); out["p_recent_joint"] = wald_p(r, pos["z_r1"] + pos["z_r3"])
    ll_full = fit_logit(y, X).llf
    X2, _ = design(g, "WH", REGIME_COLS); out["unique_recent_pct"] = 100 * (ll_full - fit_logit(y, X2).llf) / (ll_full - fit_logit(y, design(g, "", None)[0]).llf)
    base = oos_scores(g, "WH", REGIME_COLS)["dLL_x10k"]
    out["oos_dLL_recent"] = oos_scores(g, "WH", full_cols)["dLL_x10k"] - base
    # % de lucro da próxima operação por quintil da taxa recente de 3 h (só onde há pelo menos 5 operações fechadas na janela)
    m5 = g["n_r3"] >= 5
    q = pd.qcut(g.loc[m5, "z_r3"], 5, labels=False, duplicates="drop")
    out["n_with_history"] = int(m5.sum())
    for i in sorted(q.dropna().unique()):
        out[f"win_pct_Q{int(i) + 1}"] = 100 * float(g.loc[m5, "win"][q == i].mean())
    return out


def half_agreement(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman entre dois perfis (ex.: acerto por hora na 1ª e na 2ª metade dos dias)."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return float("nan"), float("nan")
    rho, p = spearmanr(a[ok], b[ok])
    return float(rho), float(p)


# ---------------------------------------------------------------------------
# Distribuição de ganhos e perdas
# ---------------------------------------------------------------------------


def win_loss_summary(gross_bps: np.ndarray, durations_min: np.ndarray | None = None) -> dict[str, float]:
    """Acerto, ganho e perda médios e medianos, razão ganho/perda, e quanto da perda total vem da cauda."""
    r = np.asarray(gross_bps, dtype=float)
    w, l = r[r > 0], r[r < 0]
    worst_losers = np.sort(l)[: max(1, int(0.1 * len(l)))]
    worst_trades = np.sort(r)[: max(1, int(0.01 * len(r)))]
    out = {"n": len(r), "hit_pct": 100 * len(w) / len(r), "mean_win": float(w.mean()), "mean_loss": float(l.mean()), "median_win": float(np.median(w)),
           "median_loss": float(np.median(l)), "payoff": float(w.mean() / abs(l.mean())), "mean_per_trade": float(r.mean()),
           "worst10pct_losers_share_of_loss": 100 * float(worst_losers.sum() / l.sum()),
           "worst1pct_trades_share_of_loss": 100 * float(worst_trades.sum() / l.sum())}
    if durations_min is not None:
        d = np.asarray(durations_min, dtype=float)
        out["median_hold_min_wins"], out["median_hold_min_losses"] = float(np.median(d[r > 0])), float(np.median(d[r < 0]))
    return out


def tail_deciles(gross_bps: np.ndarray, cuts: tuple[int, ...] = tuple(range(10, 100, 10))) -> tuple[pd.Series, pd.Series]:
    """Cortes de decis do tamanho dos ganhos e (em valor absoluto) das perdas."""
    r = np.asarray(gross_bps, dtype=float)
    return (pd.Series({f"p{c}": float(np.percentile(r[r > 0], c)) for c in cuts}),
            pd.Series({f"p{c}": float(np.percentile(-r[r < 0], c)) for c in cuts}))


def sorted_decile_contribution(gross_bps: np.ndarray) -> np.ndarray:
    """Soma do bruto de cada decil de operações (do pior ao melhor), em % da soma dos valores absolutos."""
    r = np.sort(np.asarray(gross_bps, dtype=float)); n = len(r); tot = np.abs(r).sum()
    return np.array([r[int(i * n / 10): int((i + 1) * n / 10)].sum() / tot * 100 for i in range(10)])
