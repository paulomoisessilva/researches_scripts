"""Cálculo de indicadores técnicos — implementações causais (sem look-ahead).

Todos os indicadores usam apenas dados disponíveis até o candle t,
nunca dados de t+1 em diante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from .utils import setup_logger

logger = setup_logger(__name__)

_REQUIRED_FEATURE_COLS = ("timestamp", "close", "log_return", "log_price")


# ---------------------------------------------------------------------------
# Retorno logarítmico
# ---------------------------------------------------------------------------


def log_return(close: pd.Series) -> pd.Series:
    """Retorno logarítmico: ln(close_t / close_{t-1})."""
    return np.log(close / close.shift(1))


# ---------------------------------------------------------------------------
# RSI (Relative Strength Index)
# ---------------------------------------------------------------------------


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI usando média móvel exponencial de Wilder.

    Resultado entre 0 e 100. Os primeiros `window` valores são NaN.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # EWM com span = window (equivale ao alpha = 1/window de Wilder)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# Estocástico
# ---------------------------------------------------------------------------


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Oscilador estocástico %K e %D.

    Returns
    -------
    (stoch_k, stoch_d) — ambos entre 0 e 100.
    """
    lowest_low = low.rolling(window=k_window, min_periods=k_window).min()
    highest_high = high.rolling(window=k_window, min_periods=k_window).max()

    denom = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = 100.0 * (close - lowest_low) / denom
    stoch_d = stoch_k.rolling(window=d_window, min_periods=d_window).mean()

    return stoch_k, stoch_d


# ---------------------------------------------------------------------------
# Bandas de Bollinger clássicas
# ---------------------------------------------------------------------------


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    k: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bandas de Bollinger clássicas.

    Returns
    -------
    (upper, middle, lower)
    """
    middle = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=1)
    upper = middle + k * std
    lower = middle - k * std
    return upper, middle, lower


# ---------------------------------------------------------------------------
# Bandas ajustadas por regressão (Regression-Adjusted Bands)
# ---------------------------------------------------------------------------


def rolling_ols_last(
    values: np.ndarray,
    window: int,
    chunk: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """OLS rolling de ``values`` contra o tempo (0..window-1) dentro de cada janela.

    Devolve, para cada posição ``i >= window-1``, o valor ajustado no último ponto da janela e o
    desvio padrão dos resíduos (ddof=2). Equivale a rodar ``np.linalg.lstsq`` por janela, mas
    vetorizado: em blocos, centra cada janela e usa a inclinação ``sum(tc*y)/sum(tc^2)``.
    Janelas com NaN resultam em NaN.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    fitted = np.full(n, np.nan)
    std_res = np.full(n, np.nan)
    if n < window or window < 2:
        return fitted, std_res

    t = np.arange(window, dtype=float)
    tc = t - t.mean()
    stt = float((tc ** 2).sum())
    view = np.lib.stride_tricks.sliding_window_view(values, window)  # (n - window + 1, window)
    n_windows = view.shape[0]
    for start in range(0, n_windows, chunk):
        seg = view[start:start + chunk]
        y_mean = seg.mean(axis=1)
        yc = seg - y_mean[:, None]
        slope = (yc @ tc) / stt
        resid = yc - slope[:, None] * tc[None, :]
        sse = (resid ** 2).sum(axis=1)
        end = start + len(seg)
        fitted[window - 1 + start: window - 1 + end] = y_mean + slope * tc[-1]
        std_res[window - 1 + start: window - 1 + end] = (
            np.sqrt(sse / (window - 2)) if window > 2 else 0.0
        )
    return fitted, std_res


def regression_bands_multi_k(
    log_price: pd.Series,
    window: int,
    ks: list[float],
) -> tuple[pd.Series, dict[float, tuple[pd.Series, pd.Series]]]:
    """Calcula a regressão rolling UMA vez e devolve ``(fitted, {k: (upper, lower)})``."""
    fitted, std_res = rolling_ols_last(log_price.values, window)
    idx = log_price.index
    fitted_s = pd.Series(fitted, index=idx, name="reg_fitted")
    bands = {
        k: (
            pd.Series(fitted + k * std_res, index=idx, name="reg_upper"),
            pd.Series(fitted - k * std_res, index=idx, name="reg_lower"),
        )
        for k in ks
    }
    return fitted_s, bands


def regression_bands(
    log_price: pd.Series,
    window: int = 100,
    k: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bandas baseadas em regressão OLS rolling sobre log-preço.

    Modelo: log_price_t = alpha + beta * t + error_t
    Bandas : fitted ± k * std(residuals)

    Returns
    -------
    (upper, fitted, lower)
    """
    fitted, bands = regression_bands_multi_k(log_price, window, [k])
    upper, lower = bands[k]
    return upper, fitted, lower


# ---------------------------------------------------------------------------
# Bandas de resíduos de retorno (Return Residual Bands)
# ---------------------------------------------------------------------------


def return_residual_bands(
    log_ret: pd.Series,
    window: int = 100,
    k: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bandas baseadas em regressão OLS rolling sobre retornos logarítmicos.

    Modelo: return_t = alpha + beta * t + error_t
    Sinal  : residual < -k*std → long; residual > k*std → short

    Returns
    -------
    (upper_threshold, residual, lower_threshold)
    """
    n = len(log_ret)
    residual = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    values = log_ret.values

    for i in range(window - 1, n):
        segment = values[i - window + 1 : i + 1]
        if np.isnan(segment).any():
            continue
        t = np.arange(window, dtype=float)
        A = np.column_stack([np.ones(window), t])
        try:
            coef, *_ = np.linalg.lstsq(A, segment, rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = A @ coef
        res = segment - pred
        std_res = res.std(ddof=2) if window > 2 else 0.0
        residual[i] = res[-1]
        upper[i] = k * std_res
        lower[i] = -k * std_res

    idx = log_ret.index
    return (
        pd.Series(upper, index=idx, name="resid_upper"),
        pd.Series(residual, index=idx, name="resid"),
        pd.Series(lower, index=idx, name="resid_lower"),
    )


# ---------------------------------------------------------------------------
# Indicador de confluência técnica composta
# ---------------------------------------------------------------------------


def technical_confluence_score(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    rsi_window: int = 14,
    stoch_window: int = 14,
    bb_window: int = 20,
    bb_k: float = 2.0,
) -> pd.Series:
    """Score de confluência técnica composto (range: -3 a +3).

    Componentes:
    - RSI: +1 se < 35, -1 se > 65
    - Estocástico %K: +1 se < 20, -1 se > 80
    - Bollinger: +1 se close < lower, -1 se close > upper

    Valores positivos indicam pressão compradora, negativos indicam pressão vendedora.
    Não deve ser interpretado como probabilidade.
    """
    rsi_vals = rsi(close, window=rsi_window)
    stoch_k, _ = stochastic(high, low, close, k_window=stoch_window)
    bb_upper, _, bb_lower = bollinger_bands(close, window=bb_window, k=bb_k)

    score = pd.Series(0.0, index=close.index)
    score += (rsi_vals < 35).astype(float) - (rsi_vals > 65).astype(float)
    score += (stoch_k < 20).astype(float) - (stoch_k > 80).astype(float)
    score += (close < bb_lower).astype(float) - (close > bb_upper).astype(float)

    return score.rename("technical_confluence_score")


# ---------------------------------------------------------------------------
# Função principal: gerar todas as features
# ---------------------------------------------------------------------------


# Colunas base aceitas como entrada (OHLCV + metadados do pipeline).
_BASE_INPUT_COLS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "symbol",
    "interval",
)

# Grupos de features necessários por estratégia (grade completa do config).
_STRATEGY_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "bollinger": ("base", "bollinger"),
    "regression_bands": ("base", "regression"),
    "return_residuals": ("base", "residual"),
    "rsi": ("base", "rsi"),
    "stochastic": ("base", "stochastic"),
    "technical_confluence": ("base", "confluence"),
    "random": ("base",),
}

_FEATURE_GROUP_ORDER = (
    "base",
    "rsi",
    "stochastic",
    "bollinger",
    "regression",
    "residual",
    "confluence",
)


def _order_feature_groups(groups: set[str]) -> tuple[str, ...]:
    return tuple(g for g in _FEATURE_GROUP_ORDER if g in groups)


def _resolve_feature_groups(
    feat_cfg: dict,
    strategy_names: list[str] | None = None,
) -> tuple[str, ...]:
    """Resolve quais grupos de indicadores calcular.

    Se ``strategy_names`` for informado, usa a união dos grupos exigidos pelas
    estratégias. Caso contrário, inclui cada seção presente em ``features``.
    """
    if strategy_names is not None:
        groups: set[str] = set()
        for name in strategy_names:
            strat_groups = _STRATEGY_FEATURE_GROUPS.get(name)
            if strat_groups is None:
                raise ValueError(f"Estratégia desconhecida: {name}")
            groups.update(strat_groups)
        return _order_feature_groups(groups)

    groups = {"base"}
    if "rsi_windows" in feat_cfg:
        groups.add("rsi")
    if "stochastic_windows" in feat_cfg:
        groups.add("stochastic")
    if "bollinger_windows" in feat_cfg and "bollinger_k" in feat_cfg:
        groups.add("bollinger")
    if "regression_windows" in feat_cfg and "regression_k" in feat_cfg:
        groups.update(("regression", "residual"))
    if (
        "rsi_windows" in feat_cfg
        and "stochastic_windows" in feat_cfg
        and "bollinger_windows" in feat_cfg
        and "bollinger_k" in feat_cfg
    ):
        groups.add("confluence")
    return _order_feature_groups(groups)


def _extract_base_df(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [c for c in _BASE_INPUT_COLS if c in df.columns]
    return df[base_cols].copy()


def _compute_base_features(
    close: pd.Series,
) -> dict[str, pd.Series]:
    log_ret = log_return(close)
    return {
        "log_return": log_ret,
        "log_price": np.log(close),
    }


def _compute_rsi_features(close: pd.Series, feat_cfg: dict) -> dict[str, pd.Series]:
    return {f"rsi_{w}": rsi(close, window=w) for w in feat_cfg["rsi_windows"]}


def _compute_stochastic_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    feat_cfg: dict,
) -> dict[str, pd.Series]:
    new_cols: dict[str, pd.Series] = {}
    for w in feat_cfg["stochastic_windows"]:
        k, d = stochastic(high, low, close, k_window=w)
        new_cols[f"stoch_k_{w}"] = k
        new_cols[f"stoch_d_{w}"] = d
    return new_cols


def _compute_bollinger_features(close: pd.Series, feat_cfg: dict) -> dict[str, pd.Series]:
    new_cols: dict[str, pd.Series] = {}
    if feat_cfg.get("compact_bands", False):
        # só média e desvio por janela; a banda (mid + k*std) é montada na hora em compute_band_z
        # com as mesmas operações de ``bollinger_bands``, e o resultado é idêntico
        for w in feat_cfg["bollinger_windows"]:
            new_cols[f"bb_mid_{w}"] = close.rolling(window=w, min_periods=w).mean()
            new_cols[f"bb_std_{w}"] = close.rolling(window=w, min_periods=w).std(ddof=1)
        return new_cols
    for w in feat_cfg["bollinger_windows"]:
        for k in feat_cfg["bollinger_k"]:
            k_str = str(k).replace(".", "")
            upper, mid, lower = bollinger_bands(close, window=w, k=k)
            new_cols[f"bb_upper_{w}_{k_str}"] = upper
            new_cols[f"bb_mid_{w}"] = mid
            new_cols[f"bb_lower_{w}_{k_str}"] = lower
            new_cols[f"bb_pct_{w}_{k_str}"] = (close - lower) / (upper - lower).replace(0, np.nan)
    return new_cols


def _compute_regression_features(log_price: pd.Series, feat_cfg: dict) -> dict[str, pd.Series]:
    new_cols: dict[str, pd.Series] = {}
    if feat_cfg.get("compact_bands", False):
        # só o valor ajustado e o desvio dos resíduos por janela (ver _compute_bollinger_features)
        for w in feat_cfg["regression_windows"]:
            fitted, std_res = rolling_ols_last(log_price.values, w)
            new_cols[f"reg_fitted_{w}"] = pd.Series(fitted, index=log_price.index, name="reg_fitted")
            new_cols[f"reg_std_{w}"] = pd.Series(std_res, index=log_price.index, name="reg_std")
        return new_cols
    for w in feat_cfg["regression_windows"]:
        fitted, bands = regression_bands_multi_k(log_price, w, list(feat_cfg["regression_k"]))
        new_cols[f"reg_fitted_{w}"] = fitted
        for k, (upper, lower) in bands.items():
            k_str = str(k).replace(".", "")
            new_cols[f"reg_upper_{w}_{k_str}"] = upper
            new_cols[f"reg_lower_{w}_{k_str}"] = lower
    return new_cols


def _compute_residual_features(log_ret: pd.Series, feat_cfg: dict) -> dict[str, pd.Series]:
    new_cols: dict[str, pd.Series] = {}
    for w in feat_cfg["regression_windows"]:
        for k in feat_cfg["regression_k"]:
            k_str = str(k).replace(".", "")
            upper, residual, lower = return_residual_bands(log_ret, window=w, k=k)
            new_cols[f"resid_upper_{w}_{k_str}"] = upper
            new_cols[f"resid_{w}"] = residual
            new_cols[f"resid_lower_{w}_{k_str}"] = lower
    return new_cols


def _compute_confluence_features(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    feat_cfg: dict,
) -> dict[str, pd.Series]:
    bb_k_list = feat_cfg["bollinger_k"]
    bb_k_default = bb_k_list[1] if len(bb_k_list) > 1 else bb_k_list[0]
    return {
        "technical_confluence_score": technical_confluence_score(
            close, high, low,
            rsi_window=feat_cfg["rsi_windows"][0],
            stoch_window=feat_cfg["stochastic_windows"][0],
            bb_window=feat_cfg["bollinger_windows"][0],
            bb_k=bb_k_default,
        ),
    }


def _build_feature_columns(
    base_df: pd.DataFrame,
    feat_cfg: dict,
    groups: tuple[str, ...],
) -> dict[str, pd.Series]:
    close = base_df["close"]
    high = base_df["high"]
    low = base_df["low"]
    new_cols: dict[str, pd.Series] = {}

    if "base" in groups:
        new_cols.update(_compute_base_features(close))
    if "rsi" in groups:
        new_cols.update(_compute_rsi_features(close, feat_cfg))
    if "stochastic" in groups:
        new_cols.update(_compute_stochastic_features(high, low, close, feat_cfg))
    if "bollinger" in groups:
        new_cols.update(_compute_bollinger_features(close, feat_cfg))
    if "regression" in groups:
        log_price = new_cols.get("log_price", np.log(close))
        new_cols.update(_compute_regression_features(log_price, feat_cfg))
    if "residual" in groups:
        log_ret = new_cols.get("log_return", log_return(close))
        new_cols.update(_compute_residual_features(log_ret, feat_cfg))
    if "confluence" in groups:
        new_cols.update(_compute_confluence_features(close, high, low, feat_cfg))

    return new_cols


def compute_features_for_strategy(
    strategy_name: str,
    df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Calcula apenas os indicadores necessários para uma estratégia."""
    groups = _STRATEGY_FEATURE_GROUPS.get(strategy_name)
    if groups is None:
        raise ValueError(f"Estratégia desconhecida: {strategy_name}")

    feat_cfg = cfg["features"]
    base_df = _extract_base_df(df)
    new_cols = _build_feature_columns(base_df, feat_cfg, groups)
    return pd.concat([base_df, pd.DataFrame(new_cols, index=base_df.index)], axis=1)


def compute_all_features(
    df: pd.DataFrame,
    cfg: dict,
    *,
    strategy_names: list[str] | None = None,
) -> pd.DataFrame:
    """Calcula indicadores e adiciona ao DataFrame.

    Parameters
    ----------
    df  : DataFrame processado com colunas OHLCV + timestamp.
    cfg : Dicionário de configuração (seção 'features').
    strategy_names :
        Se informado, calcula apenas os grupos exigidos pelas estratégias.
        Caso contrário, calcula cada seção presente em ``cfg['features']``.
    """
    feat_cfg = cfg["features"]
    base_df = _extract_base_df(df)
    groups = _resolve_feature_groups(feat_cfg, strategy_names)
    new_cols = _build_feature_columns(base_df, feat_cfg, groups)
    return pd.concat([base_df, pd.DataFrame(new_cols, index=base_df.index)], axis=1)


def features_parquet_path(root: Path, symbol: str, interval: str) -> Path:
    """Caminho padrão do parquet de features gerado pelo script 03."""
    return root / "data" / "features" / f"{symbol}_{interval}_features.parquet"


def _validate_feature_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Features incompletas — colunas ausentes: {missing}")


def load_or_compute_features(
    df: pd.DataFrame,
    cfg: dict,
    features_path: Path | None = None,
    *,
    force_compute: bool = False,
    strategy_names: list[str] | None = None,
) -> pd.DataFrame:
    """Carrega features do parquet ou calcula se ausente.

    Parameters
    ----------
    df:
        DataFrame processado (OHLCV). Usado apenas se o parquet não existir
        ou ``force_compute=True``.
    features_path:
        Caminho do parquet. Se None ou inexistente (sem force), calcula via
        ``compute_all_features``.
    force_compute:
        Se True, ignora parquet e recalcula features.
    """
    if not force_compute and features_path is not None and features_path.exists():
        logger.info("Carregando features de parquet: %s", features_path.name)
        df_feat = pd.read_parquet(features_path)
        _validate_feature_columns(df_feat)
        return df_feat

    if features_path is not None and not features_path.exists():
        logger.warning(
            "Parquet de features não encontrado (%s). Recalculando — "
            "rode 'python scripts/03_run_features.py' para cachear.",
            features_path.name,
        )

    logger.info("Calculando features (parquet ausente ou --force-features).")
    return compute_all_features(df, cfg, strategy_names=strategy_names)
