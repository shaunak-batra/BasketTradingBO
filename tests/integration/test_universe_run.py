"""End-to-end check of the universe screen on synthetic pairs (no network).

The pure helpers (pair enumeration, Benjamini-Hochberg, overlap handling) are unit-tested in
tests/unit/test_universe.py. This covers the orchestration that produces the published v3 numbers:
per-pair evaluation with cost replays, the distribution summary, the portfolio, the markdown report
and the charts.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd
import pytest

from src.pipeline import load_protocol
from src.strategy.signals import SignalParams
from src.universe import (
    benjamini_hochberg,
    evaluate_pair,
    plot_universe_results,
    portfolio_summary,
    render_markdown,
    summarise,
)
from src.utils.config import PROJECT_ROOT
from tests.fixtures.synthetic import cointegrated_prices, random_walk_prices


@pytest.fixture(scope="module")
def protocol():
    base = load_protocol(PROJECT_ROOT / "config" / "config_v3.yaml")
    return replace(
        base,
        walk_forward=replace(base.walk_forward, formation_days=250, trading_days=60, max_half_life_days=60.0),
        strategy=SignalParams(entry_z=1.5, exit_z=0.25, stop_z=4.0, lookback=40),
        reporting=replace(base.reporting, bootstrap_samples=200),
    )


@pytest.fixture(scope="module")
def evaluated(protocol):
    linked, _ = cointegrated_prices(n=1000, weights=(1.0, -1.0), half_life=5.0, spread_sigma=0.01, seed=21)
    linked.columns = ["AAA", "BBB"]
    unlinked = random_walk_prices(n=1000, k=2, seed=5)
    unlinked.columns = ["CCC", "DDD"]

    rows, returns, frictionless = [], {}, {}
    for family, prices in (("linked", linked), ("unlinked", unlinked)):
        row, pair_returns, pair_frictionless = evaluate_pair(prices, protocol)
        row.update(pair=pair_returns.name, family=family)
        rows.append(row)
        returns[pair_returns.name] = pair_returns
        frictionless[pair_frictionless.name] = pair_frictionless

    pairs = pd.DataFrame(rows)
    pairs["discovery"] = benjamini_hochberg(pairs["p_value"].to_numpy(), 0.10)
    return pairs, pd.DataFrame(returns).sort_index(), pd.DataFrame(frictionless).sort_index()


def test_a_mean_reverting_pair_trades_and_every_statistic_is_populated(evaluated):
    linked = evaluated[0].set_index("pair").loc["AAA/BBB"]
    assert linked["n_traded"] > 0
    assert linked["n_trades"] > 0
    for column in ("sharpe", "psr", "sharpe_frictionless", "sharpe_at_0bps", "sharpe_at_20bps", "mean_cost_bps"):
        assert math.isfinite(linked[column]), column
    assert linked["mean_cost_bps"] > 0
    assert linked["median_net_exposure"] <= 0.2  # a hedged pair passes the v3 hedging filter
    stops = int(linked["n_zscore_stops"] + linked["n_loss_stops"] + linked["n_time_stops"])
    assert 0 <= stops <= linked["n_trades"]


def test_the_same_trades_have_a_lower_sharpe_at_higher_cost(evaluated):
    linked = evaluated[0].set_index("pair").loc["AAA/BBB"]
    assert linked["sharpe_at_20bps"] < linked["sharpe_at_0bps"]


def test_summary_portfolio_report_and_charts(evaluated, protocol, tmp_path):
    pairs, returns, frictionless = evaluated
    tested = summarise(pairs)
    assert tested["n_evaluated"] == 2
    assert tested["n_with_trades"] >= 1
    assert tested["n_folds_traded"] <= tested["n_folds_total"]

    summary = {
        "run": {
            "created_utc": "2026-01-01T00:00:00+00:00",
            "protocol": "config_v3.yaml",
            "protocol_sha256": "a" * 64,
            "universe": "synthetic",
            "data_sha256": "b" * 64,
        },
        "fdr_level": 0.10,
        "n_pairs_defined": 2,
        "n_pairs_skipped": 0,
        "tested": tested,
        "portfolio": portfolio_summary(returns, protocol.reporting),
        "portfolio_frictionless": portfolio_summary(frictionless, protocol.reporting),
    }
    assert summary["portfolio"]["n_pairs"] == 2

    markdown = render_markdown(summary, pairs)
    assert "AAA/BBB" in markdown
    assert "Benjamini-Hochberg" in markdown

    paths = plot_universe_results(pairs, returns, frictionless, tmp_path / "plots")
    assert set(paths) == {"sharpe_distribution", "per_trade_economics", "portfolio_equity"}
    for path in paths.values():
        assert path.stat().st_size > 1000
