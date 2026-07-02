"""Cálculo de indicadores técnicos — implementações causais (sem look-ahead).

Todos os indicadores usam apenas dados disponíveis até o candle t,
nunca dados de t+1 em diante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
    n = len(log_price)
    fitted = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    values = log_price.values

    for i in range(window - 1, n):
        segment = values[i - window + 1 : i + 1]
        t = np.arange(window, dtype=float)
        # OLS: [1, t] @ [alpha, beta]
        A = np.column_stack([np.ones(window), t])
        try:
            coef, *_ = np.linalg.lstsq(A, segment, rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = A @ coef
        residuals = segment - pred
        std_res = residuals.std(ddof=2) if window > 2 else 0.0
        fitted[i] = pred[-1]
        upper[i] = fitted[i] + k * std_res
        lower[i] = fitted[i] - k * std_res

    idx = log_price.index
    return (
        pd.Series(upper, index=idx, name="reg_upper"),
        pd.Series(fitted, index=idx, name="reg_fitted"),
        pd.Series(lower, index=idx, name="reg_lower"),
    )


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


def compute_all_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Calcula todos os indicadores e adiciona ao DataFrame.

    Parameters
    ----------
    df  : DataFrame processado com colunas OHLCV + timestamp.
    cfg : Dicionário de configuração (seção 'features').
    """
    feat_cfg = cfg["features"]
    result = df.copy()

    close = result["close"]
    high = result["high"]
    low = result["low"]
    log_ret = log_return(close)
    log_price = np.log(close)

    result["log_return"] = log_ret
    result["log_price"] = log_price

    # RSI
    for w in feat_cfg["rsi_windows"]:
        result[f"rsi_{w}"] = rsi(close, window=w)

    # Estocástico
    for w in feat_cfg["stochastic_windows"]:
        k, d = stochastic(high, low, close, k_window=w)
        result[f"stoch_k_{w}"] = k
        result[f"stoch_d_{w}"] = d

    # Bandas de Bollinger
    for w in feat_cfg["bollinger_windows"]:
        for k in feat_cfg["bollinger_k"]:
            k_str = str(k).replace(".", "")
            upper, mid, lower = bollinger_bands(close, window=w, k=k)
            result[f"bb_upper_{w}_{k_str}"] = upper
            result[f"bb_mid_{w}"] = mid
            result[f"bb_lower_{w}_{k_str}"] = lower
            result[f"bb_pct_{w}_{k_str}"] = (close - lower) / (upper - lower).replace(0, np.nan)

    # Bandas de regressão
    for w in feat_cfg["regression_windows"]:
        for k in feat_cfg["regression_k"]:
            k_str = str(k).replace(".", "")
            upper, fitted, lower = regression_bands(log_price, window=w, k=k)
            result[f"reg_upper_{w}_{k_str}"] = upper
            result[f"reg_fitted_{w}"] = fitted
            result[f"reg_lower_{w}_{k_str}"] = lower

    # Bandas de resíduos de retorno
    for w in feat_cfg["regression_windows"]:
        for k in feat_cfg["regression_k"]:
            k_str = str(k).replace(".", "")
            upper, residual, lower = return_residual_bands(log_ret, window=w, k=k)
            result[f"resid_upper_{w}_{k_str}"] = upper
            result[f"resid_{w}"] = residual
            result[f"resid_lower_{w}_{k_str}"] = lower

    # Indicador de confluência
    bb_k_list = feat_cfg["bollinger_k"]
    bb_k_default = bb_k_list[1] if len(bb_k_list) > 1 else bb_k_list[0]
    result["technical_confluence_score"] = technical_confluence_score(
        close, high, low,
        rsi_window=feat_cfg["rsi_windows"][0],
        stoch_window=feat_cfg["stochastic_windows"][0],
        bb_window=feat_cfg["bollinger_windows"][0],
        bb_k=bb_k_default,
    )

    return result
