"""Implementação de estratégias de trading intradiário.

Regra fundamental: sinal calculado no candle t, execução no open de t+1.
A coluna 'signal' indica a posição DESEJADA no próximo candle:
  +1 = comprado, -1 = vendido, 0 = neutro
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Utilitário
# ---------------------------------------------------------------------------


def _shift_signal(raw_signal: pd.Series) -> pd.Series:
    """Converte sinal calculado em t para posição executada em t+1.

    O sinal do candle t torna-se a posição que será assumida no open de t+1.
    """
    return raw_signal.shift(1).fillna(0)


# ---------------------------------------------------------------------------
# 1. Bollinger clássica
# ---------------------------------------------------------------------------


def strategy_bollinger(
    df: pd.DataFrame,
    window: int = 20,
    k: float = 2.0,
) -> pd.Series:
    """Estratégia de reversão à média com Bandas de Bollinger.

    Entrada comprada : close < lower_band
    Entrada vendida  : close > upper_band
    Saída            : close cruza a média
    """
    col_upper = f"bb_upper_{window}_{str(k).replace('.', '')}"
    col_mid = f"bb_mid_{window}"
    col_lower = f"bb_lower_{window}_{str(k).replace('.', '')}"

    if col_upper not in df.columns:
        raise KeyError(f"Feature '{col_upper}' não encontrada. Rode compute_all_features primeiro.")

    close = df["close"]
    upper = df[col_upper]
    mid = df[col_mid]
    lower = df[col_lower]

    position = pd.Series(0.0, index=df.index)
    current_pos = 0.0

    for i in range(1, len(df)):
        c = close.iloc[i]
        u = upper.iloc[i]
        m = mid.iloc[i]
        lo = lower.iloc[i]

        if pd.isna(u) or pd.isna(lo):
            position.iloc[i] = current_pos
            continue

        if current_pos == 0:
            if c < lo:
                current_pos = 1.0
            elif c > u:
                current_pos = -1.0
        elif current_pos == 1.0:
            if c >= m:
                current_pos = 0.0
        elif current_pos == -1.0:
            if c <= m:
                current_pos = 0.0

        position.iloc[i] = current_pos

    return _shift_signal(position).rename("bollinger")


# ---------------------------------------------------------------------------
# 2. Regression-Adjusted Bands
# ---------------------------------------------------------------------------


def strategy_regression_bands(
    df: pd.DataFrame,
    window: int = 100,
    k: float = 2.0,
) -> pd.Series:
    """Estratégia de reversão com bandas ajustadas por regressão rolling.

    Entrada comprada : log_price < lower_regression_band
    Entrada vendida  : log_price > upper_regression_band
    Saída            : log_price retorna à linha ajustada
    """
    k_str = str(k).replace(".", "")
    col_upper = f"reg_upper_{window}_{k_str}"
    col_fitted = f"reg_fitted_{window}"
    col_lower = f"reg_lower_{window}_{k_str}"

    if col_upper not in df.columns:
        raise KeyError(f"Feature '{col_upper}' não encontrada.")

    log_price = df["log_price"]
    upper = df[col_upper]
    fitted = df[col_fitted]
    lower = df[col_lower]

    position = pd.Series(0.0, index=df.index)
    current_pos = 0.0

    for i in range(1, len(df)):
        lp = log_price.iloc[i]
        u = upper.iloc[i]
        f = fitted.iloc[i]
        lo = lower.iloc[i]

        if pd.isna(u) or pd.isna(lo):
            position.iloc[i] = current_pos
            continue

        if current_pos == 0:
            if lp < lo:
                current_pos = 1.0
            elif lp > u:
                current_pos = -1.0
        elif current_pos == 1.0:
            if lp >= f:
                current_pos = 0.0
        elif current_pos == -1.0:
            if lp <= f:
                current_pos = 0.0

        position.iloc[i] = current_pos

    return _shift_signal(position).rename("regression_bands")


# ---------------------------------------------------------------------------
# 3. Return Residual Bands
# ---------------------------------------------------------------------------


def strategy_return_residuals(
    df: pd.DataFrame,
    window: int = 100,
    k: float = 2.0,
) -> pd.Series:
    """Estratégia baseada em resíduos de retorno.

    long  : residual_t < -k * std_residuals
    short : residual_t >  k * std_residuals
    """
    k_str = str(k).replace(".", "")
    col_upper = f"resid_upper_{window}_{k_str}"
    col_resid = f"resid_{window}"
    col_lower = f"resid_lower_{window}_{k_str}"

    if col_upper not in df.columns:
        raise KeyError(f"Feature '{col_upper}' não encontrada.")

    resid = df[col_resid]
    upper = df[col_upper]
    lower = df[col_lower]

    signal = pd.Series(0.0, index=df.index)
    signal[resid < lower] = 1.0
    signal[resid > upper] = -1.0

    return _shift_signal(signal).rename("return_residuals")


# ---------------------------------------------------------------------------
# 4. RSI isolado
# ---------------------------------------------------------------------------


def strategy_rsi(
    df: pd.DataFrame,
    window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.Series:
    """Estratégia simples de reversão com RSI.

    long  : RSI < oversold
    short : RSI > overbought
    """
    col = f"rsi_{window}"
    if col not in df.columns:
        raise KeyError(f"Feature '{col}' não encontrada.")

    rsi_vals = df[col]
    signal = pd.Series(0.0, index=df.index)
    signal[rsi_vals < oversold] = 1.0
    signal[rsi_vals > overbought] = -1.0

    return _shift_signal(signal).rename(f"rsi_{window}")


# ---------------------------------------------------------------------------
# 5. Estocástico isolado
# ---------------------------------------------------------------------------


def strategy_stochastic(
    df: pd.DataFrame,
    window: int = 14,
    oversold: float = 20.0,
    overbought: float = 80.0,
) -> pd.Series:
    """Estratégia de reversão com estocástico %K.

    long  : stochastic_k < oversold
    short : stochastic_k > overbought
    """
    col = f"stoch_k_{window}"
    if col not in df.columns:
        raise KeyError(f"Feature '{col}' não encontrada.")

    stoch = df[col]
    signal = pd.Series(0.0, index=df.index)
    signal[stoch < oversold] = 1.0
    signal[stoch > overbought] = -1.0

    return _shift_signal(signal).rename(f"stochastic_{window}")


# ---------------------------------------------------------------------------
# 6. Indicador de confluência técnica
# ---------------------------------------------------------------------------


def strategy_technical_confluence(
    df: pd.DataFrame,
    long_threshold: float = 2.0,
    short_threshold: float = -2.0,
) -> pd.Series:
    """Estratégia baseada no technical_confluence_score.

    long  : score >= long_threshold
    short : score <= short_threshold
    """
    col = "technical_confluence_score"
    if col not in df.columns:
        raise KeyError(f"Feature '{col}' não encontrada.")

    score = df[col]
    signal = pd.Series(0.0, index=df.index)
    signal[score >= long_threshold] = 1.0
    signal[score <= short_threshold] = -1.0

    return _shift_signal(signal).rename("technical_confluence")


# ---------------------------------------------------------------------------
# 7. Estratégia aleatória (baseline)
# ---------------------------------------------------------------------------


def strategy_random(
    df: pd.DataFrame,
    target_signal: pd.Series | None = None,
    seed: int = 42,
) -> pd.Series:
    """Estratégia aleatória com mesma frequência de trades de outra estratégia.

    Se target_signal for fornecido, a frequência de trades é igualada.
    Caso contrário usa frequência de 5% por candle.
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    if target_signal is not None:
        n_trades = (target_signal.diff().abs() > 0).sum()
        trade_prob = n_trades / max(n, 1)
    else:
        trade_prob = 0.05

    raw = pd.Series(0.0, index=df.index)
    for i in range(n):
        if rng.random() < trade_prob:
            raw.iloc[i] = rng.choice([-1.0, 1.0])

    return _shift_signal(raw).rename("random")


# ---------------------------------------------------------------------------
# Dispatcher de estratégias
# ---------------------------------------------------------------------------


STRATEGY_REGISTRY = {
    "bollinger": strategy_bollinger,
    "regression_bands": strategy_regression_bands,
    "return_residuals": strategy_return_residuals,
    "rsi": strategy_rsi,
    "stochastic": strategy_stochastic,
    "technical_confluence": strategy_technical_confluence,
    "random": strategy_random,
}


def get_default_params(strategy_name: str, cfg: dict) -> dict:
    """Retorna os parâmetros padrão de uma estratégia baseados na config."""
    feat = cfg.get("features", {})
    defaults = {
        "bollinger": {
            "window": feat.get("bollinger_windows", [20])[0],
            "k": feat.get("bollinger_k", [2.0])[1],
        },
        "regression_bands": {
            "window": feat.get("regression_windows", [100])[0],
            "k": feat.get("regression_k", [2.0])[1],
        },
        "return_residuals": {
            "window": feat.get("regression_windows", [100])[0],
            "k": feat.get("regression_k", [2.0])[1],
        },
        "rsi": {"window": feat.get("rsi_windows", [14])[0]},
        "stochastic": {"window": feat.get("stochastic_windows", [14])[0]},
        "technical_confluence": {},
        "random": {},
    }
    return defaults.get(strategy_name, {})


def compute_all_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Calcula os sinais de todas as estratégias e retorna DataFrame com colunas de posição."""
    signals = df[["timestamp"]].copy() if "timestamp" in df.columns else pd.DataFrame(index=df.index)

    for name, func in STRATEGY_REGISTRY.items():
        try:
            params = get_default_params(name, cfg)
            if name == "random":
                ref = signals.get("bollinger")
                sig = func(df, target_signal=ref, **params)
            else:
                sig = func(df, **params)
            signals[name] = sig
        except KeyError as exc:
            logger.warning("Estratégia '%s' ignorada: %s", name, exc)

    return signals
