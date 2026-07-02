"""Testes de integridade do motor de backtest."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest
from src.indicators import compute_all_features, log_return
from src.strategies import strategy_bollinger, strategy_rsi


def _make_bt_cfg(
    strategy_name: str = "test",
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    allow_short: bool = True,
) -> BacktestConfig:
    return BacktestConfig(
        strategy_name=strategy_name,
        symbol="BTCUSDT",
        interval="1m",
        initial_capital=10_000.0,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        allow_short=allow_short,
    )


@pytest.fixture
def df_with_features(sample_ohlcv, sample_config):
    return compute_all_features(sample_ohlcv, sample_config)


class TestBacktestBasics:
    def test_equity_has_no_nulls(self, df_with_features, sample_config):
        signal = strategy_rsi(df_with_features, window=14)
        cfg_bt = _make_bt_cfg()
        result = run_backtest(df_with_features, signal, cfg_bt)
        assert result.equity.notna().all(), "Equity contém valores nulos."

    def test_equity_starts_at_initial_capital(self, df_with_features, sample_config):
        signal = strategy_rsi(df_with_features, window=14)
        cfg_bt = _make_bt_cfg()
        result = run_backtest(df_with_features, signal, cfg_bt)
        assert result.equity.iloc[0] == pytest.approx(10_000.0, rel=1e-6)

    def test_positions_only_valid_values(self, df_with_features):
        signal = strategy_rsi(df_with_features, window=14)
        cfg_bt = _make_bt_cfg()
        result = run_backtest(df_with_features, signal, cfg_bt)
        valid_positions = {-1.0, 0.0, 1.0}
        unique_pos = set(result.positions.unique())
        assert unique_pos.issubset(valid_positions), f"Posições inválidas: {unique_pos - valid_positions}"

    def test_returns_shape_matches_input(self, df_with_features):
        signal = strategy_rsi(df_with_features, window=14)
        cfg_bt = _make_bt_cfg()
        result = run_backtest(df_with_features, signal, cfg_bt)
        assert len(result.returns) == len(df_with_features)
        assert len(result.equity) == len(df_with_features)
        assert len(result.positions) == len(df_with_features)


class TestCostsReduceReturns:
    def test_higher_fee_reduces_total_return(self, df_with_features):
        """Custo mais alto deve reduzir (ou manter igual) o retorno total."""
        signal = strategy_bollinger(df_with_features, window=20, k=2.0)

        cfg_no_cost = _make_bt_cfg(fee_bps=0.0, slippage_bps=0.0)
        cfg_with_cost = _make_bt_cfg(fee_bps=10.0, slippage_bps=10.0)

        result_no_cost = run_backtest(df_with_features, signal, cfg_no_cost)
        result_with_cost = run_backtest(df_with_features, signal, cfg_with_cost)

        total_no_cost = result_no_cost.equity.iloc[-1]
        total_with_cost = result_with_cost.equity.iloc[-1]

        assert total_no_cost >= total_with_cost, (
            f"Custo alto deveria reduzir retorno: "
            f"sem_custo={total_no_cost:.2f}, com_custo={total_with_cost:.2f}"
        )

    def test_costs_strictly_reduce_with_trades(self, df_with_features):
        """Quando há trades, custo > 0 deve reduzir retorno versus custo = 0."""
        signal = strategy_rsi(df_with_features, window=14)

        cfg_zero = _make_bt_cfg(fee_bps=0.0, slippage_bps=0.0)
        cfg_cost = _make_bt_cfg(fee_bps=5.0, slippage_bps=5.0)

        r0 = run_backtest(df_with_features, signal, cfg_zero)
        rc = run_backtest(df_with_features, signal, cfg_cost)

        n_trades_0 = len(r0.trades)
        if n_trades_0 > 0:
            assert r0.equity.iloc[-1] >= rc.equity.iloc[-1]


class TestTradeCount:
    def test_trade_count_matches_position_changes(self, df_with_features):
        """Número de trades deve ser compatível com mudanças de posição."""
        signal = strategy_rsi(df_with_features, window=14)
        cfg_bt = _make_bt_cfg()
        result = run_backtest(df_with_features, signal, cfg_bt)

        pos_changes = (result.positions.diff().abs() > 0).sum()
        n_exits = len(result.trades)

        # Número de saídas deve ser <= número de mudanças de posição
        assert n_exits <= pos_changes, (
            f"Mais trades ({n_exits}) do que mudanças de posição ({pos_changes})."
        )

    def test_no_short_when_disabled(self, df_with_features):
        signal = strategy_rsi(df_with_features, window=14)
        cfg_bt = _make_bt_cfg(allow_short=False)
        result = run_backtest(df_with_features, signal, cfg_bt)
        assert (result.positions >= 0).all(), "Posições short encontradas com allow_short=False."


class TestEquityIntegrity:
    def test_equity_is_positive(self, df_with_features):
        signal = strategy_rsi(df_with_features, window=14)
        cfg_bt = _make_bt_cfg(fee_bps=0.0)
        result = run_backtest(df_with_features, signal, cfg_bt)
        assert (result.equity > 0).all(), "Equity negativou."

    def test_equity_consistent_with_returns(self, df_with_features):
        """Equity deve ser consistente com os retornos acumulados."""
        signal = strategy_rsi(df_with_features, window=14)
        cfg_bt = _make_bt_cfg(fee_bps=0.0, slippage_bps=0.0)
        result = run_backtest(df_with_features, signal, cfg_bt)

        # Recalcula equity a partir dos retornos
        cum_log = result.returns.fillna(0).cumsum()
        expected_equity = 10_000 * np.exp(cum_log)

        np.testing.assert_allclose(
            result.equity.values,
            expected_equity.values,
            rtol=1e-8,
            err_msg="Equity inconsistente com retornos acumulados.",
        )
