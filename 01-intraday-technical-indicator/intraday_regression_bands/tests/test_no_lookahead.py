"""Testes de ausência de look-ahead bias.

Verifica que:
  1. O sinal calculado no candle t não usa informação de t+1 em diante.
  2. A execução ocorre apenas no open do candle t+1.
  3. Indicadores rolling só usam dados até t.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest
from src.indicators import (
    bollinger_bands,
    compute_all_features,
    log_return,
    regression_bands,
    return_residual_bands,
    rsi,
    stochastic,
)
from src.strategies import (
    strategy_bollinger,
    strategy_rsi,
    strategy_stochastic,
    strategy_regression_bands,
)


class TestSignalNoLookahead:
    def test_rsi_signal_uses_only_past_data(self, sample_ohlcv, sample_config):
        """RSI em t não pode depender de close[t+1]."""
        df = compute_all_features(sample_ohlcv, sample_config)
        t = 100

        # Calcula RSI com série completa
        rsi_full = df["rsi_14"]

        # Calcula RSI truncando em t+1 (simulando que t+1 não existe)
        df_trunc = df.iloc[:t + 1].copy()
        rsi_trunc = rsi(df_trunc["close"], window=14)

        # O valor em t deve ser idêntico
        assert abs(rsi_full.iloc[t] - rsi_trunc.iloc[t]) < 1e-10, (
            f"RSI difere ao truncar: full={rsi_full.iloc[t]:.8f}, "
            f"trunc={rsi_trunc.iloc[t]:.8f} — look-ahead detectado!"
        )

    def test_bollinger_signal_uses_only_past_data(self, sample_ohlcv):
        """Bandas de Bollinger em t não podem depender de close[t+1]."""
        close = sample_ohlcv["close"]
        t = 150

        upper_full, mid_full, lower_full = bollinger_bands(close, window=20, k=2.0)
        upper_trunc, mid_trunc, lower_trunc = bollinger_bands(close.iloc[:t + 1], window=20, k=2.0)

        assert abs(upper_full.iloc[t] - upper_trunc.iloc[t]) < 1e-10, \
            "Bollinger upper difere ao truncar — look-ahead detectado!"
        assert abs(lower_full.iloc[t] - lower_trunc.iloc[t]) < 1e-10, \
            "Bollinger lower difere ao truncar — look-ahead detectado!"

    def test_regression_bands_no_lookahead(self, sample_ohlcv):
        """Bandas de regressão em t não podem depender de dados de t+1."""
        log_price = np.log(sample_ohlcv["close"])
        t = 200

        _, fitted_full, _ = regression_bands(log_price, window=50, k=2.0)
        _, fitted_trunc, _ = regression_bands(log_price.iloc[:t + 1], window=50, k=2.0)

        if pd.notna(fitted_full.iloc[t]) and pd.notna(fitted_trunc.iloc[t]):
            assert abs(fitted_full.iloc[t] - fitted_trunc.iloc[t]) < 1e-10, \
                "Regression bands difere ao truncar — look-ahead detectado!"

    def test_return_residuals_no_lookahead(self, sample_ohlcv):
        """Resíduos de retorno em t não podem depender de dados de t+1."""
        log_ret = log_return(sample_ohlcv["close"])
        t = 200

        _, resid_full, _ = return_residual_bands(log_ret, window=50, k=2.0)
        _, resid_trunc, _ = return_residual_bands(log_ret.iloc[:t + 1], window=50, k=2.0)

        if pd.notna(resid_full.iloc[t]) and pd.notna(resid_trunc.iloc[t]):
            assert abs(resid_full.iloc[t] - resid_trunc.iloc[t]) < 1e-10, \
                "Return residuals difere ao truncar — look-ahead detectado!"

    def test_stochastic_no_lookahead(self, sample_ohlcv):
        """Estocástico em t não pode depender de dados de t+1."""
        t = 100
        k_full, _ = stochastic(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            k_window=14,
        )
        k_trunc, _ = stochastic(
            sample_ohlcv["high"].iloc[:t + 1],
            sample_ohlcv["low"].iloc[:t + 1],
            sample_ohlcv["close"].iloc[:t + 1],
            k_window=14,
        )
        if pd.notna(k_full.iloc[t]) and pd.notna(k_trunc.iloc[t]):
            assert abs(k_full.iloc[t] - k_trunc.iloc[t]) < 1e-10, \
                "Stochastic difere ao truncar — look-ahead detectado!"


class TestExecutionNextCandle:
    def test_signal_shifted_to_next_candle(self, sample_ohlcv, sample_config):
        """O sinal calculado em t deve ser executado como posição em t+1.

        Isso significa que a posição em t+1 é baseada no sinal de t.
        A estratégia usa _shift_signal() internamente.
        """
        df = compute_all_features(sample_ohlcv, sample_config)

        # Pega sinal bruto de RSI (sem shift)
        rsi_vals = df["rsi_14"]
        raw_signal = pd.Series(0.0, index=df.index)
        raw_signal[rsi_vals < 30] = 1.0
        raw_signal[rsi_vals > 70] = -1.0

        # Sinal da estratégia (com shift)
        shifted_signal = strategy_rsi(df, window=14)

        # Em t=50, a posição da estratégia deve ser o sinal raw de t=49
        t = 50
        assert shifted_signal.iloc[t] == raw_signal.iloc[t - 1], (
            f"Sinal não deslocado corretamente: "
            f"shifted[{t}]={shifted_signal.iloc[t]}, "
            f"raw[{t-1}]={raw_signal.iloc[t-1]}"
        )

    def test_no_position_in_same_candle_as_signal(self, sample_ohlcv, sample_config):
        """Nunca deve haver execução no mesmo candle em que o sinal foi gerado."""
        df = compute_all_features(sample_ohlcv, sample_config)
        signal = strategy_rsi(df, window=14)

        # A posição[0] deve ser 0 (sinal do candle -1 não existe)
        assert signal.iloc[0] == 0.0, (
            f"Posição não-nula no candle 0: {signal.iloc[0]} — possível execução no mesmo candle!"
        )


class TestRollingWindowCausality:
    def test_rolling_mean_causal(self):
        """Rolling mean padrão do pandas já é causal (usa apenas dados até t)."""
        data = pd.Series(range(100), dtype=float)
        rolling_mean = data.rolling(window=10).mean()

        t = 50
        # Valor em t deve ser média de data[41:51]
        expected = data.iloc[41:51].mean()
        assert abs(rolling_mean.iloc[t] - expected) < 1e-10

    def test_no_future_values_in_rolling_std(self, sample_ohlcv):
        """std rolling não deve usar dados futuros."""
        close = sample_ohlcv["close"]
        t = 100

        std_full = close.rolling(window=20).std()
        std_trunc = close.iloc[:t + 1].rolling(window=20).std()

        if pd.notna(std_full.iloc[t]) and pd.notna(std_trunc.iloc[t]):
            assert abs(std_full.iloc[t] - std_trunc.iloc[t]) < 1e-10


class TestBacktestNoLookahead:
    def test_first_position_zero(self, sample_ohlcv, sample_config):
        """Primeira posição deve ser zero — não há sinal antes do candle 0."""
        df = compute_all_features(sample_ohlcv, sample_config)
        signal = strategy_bollinger(df, window=20, k=2.0)

        cfg_bt = BacktestConfig(
            strategy_name="bollinger",
            symbol="BTCUSDT",
            interval="1m",
            initial_capital=10_000,
        )
        result = run_backtest(df, signal, cfg_bt)

        assert result.positions.iloc[0] == 0.0, (
            f"Posição inicial não-nula: {result.positions.iloc[0]}"
        )

    def test_position_at_t_matches_signal_at_t(self, sample_ohlcv, sample_config):
        """A posição no candle t deve corresponder ao sinal do candle t-1."""
        df = compute_all_features(sample_ohlcv, sample_config)
        signal = strategy_rsi(df, window=14)

        cfg_bt = BacktestConfig(
            strategy_name="rsi",
            symbol="BTCUSDT",
            interval="1m",
            initial_capital=10_000,
        )
        result = run_backtest(df, signal, cfg_bt)

        # A posição executada em t é o sinal que entrou em t
        # (signal já vem deslocado 1 período)
        for t in range(10, 20):
            assert result.positions.iloc[t] == signal.iloc[t], (
                f"Posição[{t}]={result.positions.iloc[t]} != signal[{t}]={signal.iloc[t]}"
            )
