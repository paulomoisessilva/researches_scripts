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


def _price_band_z(price: pd.Series, center: pd.Series, upper: pd.Series) -> pd.Series:
    """Distância normalizada do preço em relação às bandas simétricas."""
    half_width = (upper - center).replace(0, np.nan)
    return (price - center) / half_width


def _residual_z(residual: pd.Series, upper: pd.Series, k: float) -> pd.Series:
    """z = residual / residual_std, com residual_std = upper / k."""
    residual_std = (upper / k).replace(0, np.nan)
    return residual / residual_std


def _reversal_confirmed_position(
    z: pd.Series,
    entry_long_z: float = 0.8,
    entry_short_z: float = 0.8,
    exit_long_z: float = 0.0,
    exit_short_z: float = 0.0,
) -> pd.Series:
    """Gera posição com entrada por reversão confirmada no indicador z.

    Long  : z_{t-1} < -entry_long_z  e z_t >= -entry_long_z
    Short : z_{t-1} > +entry_short_z e z_t <= +entry_short_z
    Saída long  : z >= exit_long_z
    Saída short : z <= exit_short_z
    """
    position = pd.Series(0.0, index=z.index)
    current_pos = 0.0
    z_vals = z.values

    for i in range(1, len(z)):
        z_prev = z_vals[i - 1]
        z_curr = z_vals[i]

        if np.isnan(z_prev) or np.isnan(z_curr):
            position.iloc[i] = current_pos
            continue

        if current_pos == 1.0:
            if z_curr >= exit_long_z:
                current_pos = 0.0
        elif current_pos == -1.0:
            if z_curr <= exit_short_z:
                current_pos = 0.0

        if current_pos == 0.0:
            if z_prev < -entry_long_z and z_curr >= -entry_long_z:
                current_pos = 1.0
            elif z_prev > entry_short_z and z_curr <= entry_short_z:
                current_pos = -1.0

        position.iloc[i] = current_pos

    return position


def signal_from_z(
    z: pd.Series,
    entry_long_z: float = 0.8,
    entry_short_z: float = 0.8,
    exit_long_z: float = 0.0,
    exit_short_z: float = 0.0,
    name: str = "signal",
) -> pd.Series:
    """Converte série z em sinal de posição (com shift para execução t+1)."""
    position = _reversal_confirmed_position(
        z,
        entry_long_z=entry_long_z,
        entry_short_z=entry_short_z,
        exit_long_z=exit_long_z,
        exit_short_z=exit_short_z,
    )
    return _shift_signal(position).rename(name)


def compute_band_z(
    df: pd.DataFrame,
    strategy_name: str,
    window: int,
    k: float,
) -> pd.Series:
    """Calcula z normalizado para estratégias baseadas em bandas."""
    k_str = str(k).replace(".", "")

    if strategy_name == "bollinger":
        col_upper = f"bb_upper_{window}_{k_str}"
        col_mid = f"bb_mid_{window}"
        if col_upper not in df.columns:
            raise KeyError(f"Feature '{col_upper}' não encontrada.")
        return _price_band_z(df["close"], df[col_mid], df[col_upper])

    if strategy_name == "regression_bands":
        col_upper = f"reg_upper_{window}_{k_str}"
        col_fitted = f"reg_fitted_{window}"
        if col_upper not in df.columns:
            raise KeyError(f"Feature '{col_upper}' não encontrada.")
        return _price_band_z(df["log_price"], df[col_fitted], df[col_upper])

    if strategy_name == "return_residuals":
        col_upper = f"resid_upper_{window}_{k_str}"
        col_resid = f"resid_{window}"
        if col_upper not in df.columns:
            raise KeyError(f"Feature '{col_upper}' não encontrada.")
        return _residual_z(df[col_resid], df[col_upper], k)

    raise ValueError(f"Estratégia '{strategy_name}' não suporta compute_band_z.")


BAND_STRATEGIES = frozenset({"bollinger", "regression_bands", "return_residuals"})


# ---------------------------------------------------------------------------
# 1. Bollinger clássica
# ---------------------------------------------------------------------------


def strategy_bollinger(
    df: pd.DataFrame,
    window: int = 20,
    k: float = 2.0,
    entry_long_z: float = 0.8,
    entry_short_z: float = 0.8,
    exit_long_z: float = 0.0,
    exit_short_z: float = 0.0,
) -> pd.Series:
    """Reversão confirmada com Bandas de Bollinger via z normalizado.

    z = (close - mid) / (upper - mid)
    """
    z = compute_band_z(df, "bollinger", window, k)
    return signal_from_z(
        z,
        entry_long_z=entry_long_z,
        entry_short_z=entry_short_z,
        exit_long_z=exit_long_z,
        exit_short_z=exit_short_z,
        name="bollinger",
    )


# ---------------------------------------------------------------------------
# 2. Regression-Adjusted Bands
# ---------------------------------------------------------------------------


def strategy_regression_bands(
    df: pd.DataFrame,
    window: int = 100,
    k: float = 2.0,
    entry_long_z: float = 0.8,
    entry_short_z: float = 0.8,
    exit_long_z: float = 0.0,
    exit_short_z: float = 0.0,
) -> pd.Series:
    """Reversão confirmada com bandas ajustadas por regressão via z normalizado.

    z = (log_price - fitted) / (upper - fitted)
    """
    z = compute_band_z(df, "regression_bands", window, k)
    return signal_from_z(
        z,
        entry_long_z=entry_long_z,
        entry_short_z=entry_short_z,
        exit_long_z=exit_long_z,
        exit_short_z=exit_short_z,
        name="regression_bands",
    )


# ---------------------------------------------------------------------------
# 3. Return Residual Bands
# ---------------------------------------------------------------------------


def strategy_return_residuals(
    df: pd.DataFrame,
    window: int = 100,
    k: float = 2.0,
    entry_long_z: float = 0.8,
    entry_short_z: float = 0.8,
    exit_long_z: float = 0.0,
    exit_short_z: float = 0.0,
) -> pd.Series:
    """Reversão confirmada com bandas de resíduo de retorno via z normalizado.

    z = residual / residual_std
    """
    z = compute_band_z(df, "return_residuals", window, k)
    return signal_from_z(
        z,
        entry_long_z=entry_long_z,
        entry_short_z=entry_short_z,
        exit_long_z=exit_long_z,
        exit_short_z=exit_short_z,
        name="return_residuals",
    )


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


def resolve_random_seeds(cfg: dict) -> list[int]:
    """Lista de seeds para baseline aleatório (config.random_baseline)."""
    rb = cfg.get("random_baseline", {})
    if "seeds" in rb:
        return [int(s) for s in rb["seeds"]]
    n_seeds = int(rb.get("n_seeds", 1))
    seed_base = int(rb.get("seed_base", 42))
    if n_seeds <= 1:
        return [seed_base]
    return list(range(seed_base, seed_base + n_seeds))


def position_exposure_stats(positions: pd.Series) -> dict[str, float]:
    """Estatísticas de exposição long/short/flat e turnover."""
    pos = positions.fillna(0.0).astype(float)
    n = max(len(pos), 1)
    return {
        "long_pct": float((pos > 0).sum() / n),
        "short_pct": float((pos < 0).sum() / n),
        "flat_pct": float((pos == 0).sum() / n),
        "avg_position": float(pos.mean()),
        "position_changes": int((pos.diff().abs() > 0).sum()),
    }


def strategy_random(
    df: pd.DataFrame,
    target_signal: pd.Series | None = None,
    seed: int = 42,
    trade_prob: float | None = None,
) -> pd.Series:
    """Baseline aleatório multi-seed: trading direcional aleatório com taxa fixa.

    A cada candle, com probabilidade ``trade_prob``, abre posição long ou short
    (50/50) por um barra. Usado como placebo para verificar se estratégias reais
    superam trading direcional aleatório.
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    if trade_prob is None:
        if target_signal is not None:
            n_trades = (target_signal.diff().abs() > 0).sum()
            trade_prob = n_trades / max(n, 1)
        else:
            trade_prob = 0.05

    fire = rng.random(n) < trade_prob
    direction = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    raw = pd.Series(np.where(fire, direction, 0.0), index=df.index)

    return _shift_signal(raw).rename("random")


def strategy_random_matched(
    df: pd.DataFrame,
    target_signal: pd.Series,
    seed: int = 42,
) -> pd.Series:
    """Baseline matched dev-only — copia calendário de trades e randomiza direção.

    Não faz parte da metodologia principal. Ver ``src.random_baseline_dev``.
    """
    rng = np.random.default_rng(seed)
    target = target_signal.reindex(df.index).fillna(0.0).astype(float)
    raw = pd.Series(0.0, index=df.index)
    current = 0.0

    for i in range(len(target)):
        prev = target.iloc[i - 1] if i > 0 else 0.0
        curr = target.iloc[i]

        if i == 0:
            if curr != 0:
                current = float(rng.choice([-1.0, 1.0]))
        elif curr != prev:
            if curr == 0:
                current = 0.0
            else:
                current = float(rng.choice([-1.0, 1.0]))

        raw.iloc[i] = current

    return _shift_signal(raw).rename("random_matched")


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


def _band_strategy_z_params(cfg: dict) -> dict:
    """Parâmetros z compartilhados pelas estratégias de bandas."""
    strat = cfg.get("strategy", {})
    return {
        "entry_long_z": strat.get("entry_long_z", [0.8])[0],
        "entry_short_z": strat.get("entry_short_z", [0.8])[0],
        "exit_long_z": strat.get("exit_long_z", [0.0])[0],
        "exit_short_z": strat.get("exit_short_z", [0.0])[0],
    }


def get_default_params(strategy_name: str, cfg: dict) -> dict:
    """Retorna os parâmetros padrão de uma estratégia baseados na config."""
    feat = cfg.get("features", {})
    z_params = _band_strategy_z_params(cfg)
    defaults = {
        "bollinger": {
            "window": feat.get("bollinger_windows", [20])[0],
            "k": feat.get("bollinger_k", [2.0])[1],
            **z_params,
        },
        "regression_bands": {
            "window": feat.get("regression_windows", [100])[0],
            "k": feat.get("regression_k", [2.0])[1],
            **z_params,
        },
        "return_residuals": {
            "window": feat.get("regression_windows", [100])[0],
            "k": feat.get("regression_k", [2.0])[1],
            **z_params,
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
                rb = cfg.get("random_baseline", {})
                seed = int(rb.get("seed_base", 42))
                sig = func(
                    df,
                    target_signal=ref,
                    seed=seed,
                    trade_prob=rb.get("trade_prob"),
                )
            else:
                sig = func(df, **params)
            signals[name] = sig
        except KeyError as exc:
            logger.warning("Estratégia '%s' ignorada: %s", name, exc)

    return signals
