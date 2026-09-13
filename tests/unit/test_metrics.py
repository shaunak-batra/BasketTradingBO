"""Performance statistics with hand-computed expectations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.backtesting.metrics import (
    block_bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    max_drawdown,
    probabilistic_sharpe_ratio,
    psr_from_moments,
    sharpe_ratio,
    sortino_ratio,
    summarize_performance,
    trade_statistics,
)
from tests.fixtures.synthetic import business_days

RETURNS = np.array([0.01, -0.02, 0.03, 0.0])
LEDGER_COLUMNS = ["net_pnl", "return_on_equity", "holding_bars", "exit_reason", "is_open"]


def test_sharpe_uses_the_sample_standard_deviation():
    # Deviations from the 0.005 mean square to 0.0013 in total; ddof = 1.
    expected = 0.005 / math.sqrt(0.0013 / 3) * math.sqrt(252)
    assert sharpe_ratio(RETURNS) == pytest.approx(expected, rel=1e-12)


def test_sortino_uses_downside_deviation_over_all_periods():
    # mean(min(r, 0)^2) = 0.0004 / 4, so the downside deviation is 0.01.
    assert sortino_ratio(RETURNS) == pytest.approx(0.005 / 0.01 * math.sqrt(252), rel=1e-12)


def test_ratios_are_undefined_without_variance_or_downside():
    assert math.isnan(sharpe_ratio(np.zeros(10)))
    assert math.isnan(sortino_ratio(np.full(10, 0.001)))


def test_max_drawdown_depth_and_longest_underwater_spell():
    depth, days = max_drawdown(pd.Series([100.0, 120.0, 90.0, 95.0, 130.0, 100.0]))
    assert depth == pytest.approx(-0.25)
    assert days == 2


def test_psr_matches_the_closed_form():
    expected = stats.norm.cdf(0.1 * math.sqrt(100) / math.sqrt(1 + 0.5 * 0.01))
    assert psr_from_moments(0.1, 101, 0.0, 3.0) == pytest.approx(expected, rel=1e-12)


def test_psr_is_lower_for_negative_skew_and_fat_tails():
    base = psr_from_moments(0.1, 250, 0.0, 3.0)
    assert psr_from_moments(0.1, 250, -1.0, 3.0) < base
    assert psr_from_moments(0.1, 250, 0.0, 10.0) < base


def test_expected_max_sharpe_grows_with_the_number_of_trials():
    trials = np.random.default_rng(0).normal(0.0, 0.05, 200)
    values = [expected_max_sharpe(trials, n) for n in (2, 10, 100, 1000)]
    assert values == sorted(values)
    assert values[0] > 0
    assert expected_max_sharpe(trials, 1) == 0.0


def test_deflated_sharpe_is_below_probabilistic_sharpe_after_selection():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.001, 0.01, 500)
    trials = rng.normal(0.0, 0.03, 50)
    assert deflated_sharpe_ratio(returns, trials, 50) < probabilistic_sharpe_ratio(returns)
    assert deflated_sharpe_ratio(returns, trials, 1) == pytest.approx(probabilistic_sharpe_ratio(returns))


def test_block_bootstrap_interval_brackets_the_point_estimate():
    returns = np.random.default_rng(2).normal(0.0005, 0.01, 2520)
    low, high = block_bootstrap_sharpe_ci(returns, n_samples=500, block=20, seed=3)
    assert low < sharpe_ratio(returns) < high


def test_block_bootstrap_needs_enough_data():
    assert all(math.isnan(value) for value in block_bootstrap_sharpe_ci(np.ones(10), block=20))


def test_trade_statistics_use_closed_round_trips_only():
    trades = pd.DataFrame(
        {
            "net_pnl": [100.0, -50.0, 25.0, 999.0],
            "return_on_equity": [0.01, -0.005, 0.0025, 0.1],
            "holding_bars": [5, 3, 4, 1],
            "exit_reason": ["time_stop", "stop", "stop_loss", "open"],
            "is_open": [False, False, False, True],
        }
    )
    result = trade_statistics(trades)
    assert result["n_trades"] == 3
    assert result["win_rate"] == pytest.approx(2 / 3)
    assert result["profit_factor"] == pytest.approx(125 / 50)
    assert result["avg_holding_days"] == pytest.approx(4.0)
    # Each kind of stop is counted separately; the open position's reason is ignored.
    assert (result["n_zscore_stops"], result["n_loss_stops"], result["n_time_stops"]) == (1, 1, 1)


def test_trade_statistics_without_trades_are_undefined_not_zero():
    result = trade_statistics(pd.DataFrame(columns=LEDGER_COLUMNS))
    assert result["n_trades"] == 0
    assert math.isnan(result["win_rate"])
    assert math.isnan(result["profit_factor"])


def test_summary_total_return_and_cagr():
    index = business_days(253)
    equity = pd.Series(100.0 * 2 ** (np.arange(253) / 252), index=index)
    zeros = pd.Series(0.0, index=index)
    summary = summarize_performance(
        equity, pd.DataFrame(columns=LEDGER_COLUMNS), zeros, zeros, zeros, zeros, bootstrap_samples=200
    )
    assert summary["total_return"] == pytest.approx(1.0)
    assert summary["cagr"] == pytest.approx(1.0)
    assert summary["max_drawdown"] == 0.0
    assert summary["n_trades"] == 0
    assert summary["time_in_market"] == 0.0
