"""Explanatory figures for the README.

Each figure illustrates one idea using synthetic data or closed-form mathematics, built with the
same library functions the research pipeline uses. None of these figures is a research result:
empirical results live in ``results/``. The GLD/IAU constants in the leverage figure are quoted
from ``results/v3_universe/pairs.csv`` and labelled as such.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.backtesting.backtester import BacktestConfig, run_backtest
from src.backtesting.metrics import psr_from_moments
from src.cointegration.engine import johansen_test
from src.cointegration.spread import compute_spread, rolling_zscore
from src.strategy.signals import SignalParams, generate_signals

BLUE, LIGHT, RED, GREEN, ORANGE, GREY = "#1f5f99", "#8fb3d9", "#b23a3a", "#1a7f37", "#d08a1e", "0.35"
DPI = 110
TRADING_DAYS = 252

# Quoted from results/v3_universe/pairs.csv (positive control GLD/IAU).
CONTROL_GROSS_BPS = 0.53
CONTROL_COST_BPS = 10.32
CONTROL_WANTED_GROSS = 17.4


def _save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    return path


def synthetic_pair(n: int = 750, half_life_days: float = 10.0, seed: int = 3) -> pd.DataFrame:
    """Two prices whose log difference is an AR(1) process with the given half-life."""
    rng = np.random.default_rng(seed)
    phi = 0.5 ** (1.0 / half_life_days)
    shocks = rng.standard_normal(n)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + 0.012 * shocks[t]
    log_b = np.log(50.0) + np.cumsum(rng.standard_normal(n) * 0.015)
    log_a = np.log(2.0) + log_b + spread
    index = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame({"A": np.exp(log_a), "B": np.exp(log_b)}, index=index)


def cointegration_figure(path: Path) -> dict:
    """Prices, the stationary spread Johansen finds, and the z-score signal on it."""
    prices = synthetic_pair()
    johansen = johansen_test(np.log(prices))
    spread = compute_spread(prices, johansen.weights)
    params = SignalParams(entry_z=2.0, exit_z=0.5, stop_z=4.0, lookback=60)
    zscore = rolling_zscore(spread, params.lookback)
    state = generate_signals(zscore, params)["state"].to_numpy()

    figure = Figure(figsize=(11, 10))
    axes = figure.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0, 1.25]})

    ax = axes[0]
    ax.plot(prices.index, prices["A"], color=BLUE, lw=1.2, label="asset A")
    ax.plot(prices.index, prices["B"] * 2.0, color=ORANGE, lw=1.2, label="asset B, scaled by 2")
    ax.set_ylabel("price")
    ax.set_title("1. Each price wanders like a random walk, but the two never drift apart for long")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    w_a, w_b = johansen.weights
    ax.plot(spread.index, spread.to_numpy(), color=GREEN, lw=1.0)
    ax.axhline(float(spread.mean()), color=GREY, ls="--", lw=0.9, label="long-run mean")
    ax.set_ylabel("spread")
    ax.set_title(
        f"2. Johansen weights give a stationary spread S = {w_a:.2f} log A {w_b:+.2f} log B "
        f"(trace {johansen.trace_stats[0]:.1f} > critical {johansen.trace_critical[0]:.1f})"
    )
    ax.legend(frameon=False, loc="upper left")

    ax = axes[2]
    ax.fill_between(zscore.index, -4.6, 4.6, where=state > 0, color=GREEN, alpha=0.13, lw=0, label="long the spread")
    ax.fill_between(zscore.index, -4.6, 4.6, where=state < 0, color=RED, alpha=0.13, lw=0, label="short the spread")
    ax.plot(zscore.index, zscore.to_numpy(), color=BLUE, lw=0.9, label="z-score, 60-day window")
    for level, colour, style, label in ((2.0, RED, "--", "entry at 2"), (0.5, ORANGE, ":", "exit at 0.5"), (4.0, GREY, "-.", "stop at 4")):
        ax.axhline(level, color=colour, ls=style, lw=1.0, label=label)
        ax.axhline(-level, color=colour, ls=style, lw=1.0)
    ax.set_ylim(-4.6, 4.6)
    ax.set_ylabel("z-score")
    ax.set_title("3. Standardise the spread and trade the extremes: enter far from the mean, exit near it")
    ax.legend(ncol=3, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    for axis in axes:
        axis.grid(alpha=0.3)
    _save(figure, path)
    return {
        "cointegration_detected": johansen.is_cointegrated,
        "johansen_weights": [float(w) for w in johansen.weights],
    }


def walk_forward_figure(path: Path, formation_days: int = 504, trading_days: int = 126, n_folds: int = 6) -> dict:
    """Formation windows estimate everything; the following trading windows are traded blind."""
    figure = Figure(figsize=(11, 4.2))
    ax = figure.subplots()
    for fold in range(n_folds):
        start = fold * trading_days
        row = n_folds - fold
        ax.barh(row, formation_days, left=start, height=0.62, color=LIGHT, edgecolor="white")
        ax.barh(row, trading_days, left=start + formation_days, height=0.62, color=BLUE, edgecolor="white")
        ax.text(start + formation_days / 2, row, "estimate", ha="center", va="center", fontsize=8.5)
        ax.text(start + formation_days + trading_days / 2, row, "trade", ha="center", va="center", fontsize=8.5, color="white")
    ax.set_yticks([n_folds - fold for fold in range(n_folds)])
    ax.set_yticklabels([f"fold {fold}" for fold in range(n_folds)])
    ax.set_xlabel("trading days from the start of the data")
    ax.set_title(
        f"Walk-forward protocol: {formation_days} days to estimate, then {trading_days} days traded with everything frozen"
    )
    ax.legend(
        handles=[
            Patch(color=LIGHT, label="formation window: Johansen test, weights, half-life, volatility, thresholds"),
            Patch(color=BLUE, label="trading window: out of sample, stitched into one track record"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),  # below the axis, clear of every bar
        ncol=1,
        fontsize=8.5,
    )
    ax.grid(axis="x", alpha=0.3)
    _save(figure, path)
    return {}


def accounting_figure(path: Path) -> dict:
    """The v1 equity formula against cash accounting, on prices that never move."""
    n = 20
    index = pd.bdate_range("2022-01-03", periods=n)
    prices = pd.DataFrame({"A": np.full(n, 100.0), "B": np.full(n, 50.0)}, index=index)
    weights = np.array([1.0, -0.5])
    capital, position_size, cost_rate = 100_000.0, 0.20, 0.0015

    # v1: shares from initial capital, rounded down, and equity = capital + position value - cumulative costs.
    target_dollars = capital * position_size * weights / np.abs(weights).sum()
    shares = np.floor(target_dollars / prices.iloc[0].to_numpy())
    positions = np.zeros((n, 2))
    positions[5:15] = shares
    changes = np.abs(np.diff(np.vstack([np.zeros(2), positions]), axis=0))
    costs = (changes @ prices.iloc[0].to_numpy()) * cost_rate
    equity_v1 = capital + positions @ prices.iloc[0].to_numpy() - np.cumsum(costs)

    # v2: the same round trip through the real backtester (signal at bar 4 fills at bar 5).
    state = np.zeros(n, dtype=np.int64)
    state[4:14] = 1
    config = BacktestConfig(gross_exposure=position_size, cost_bps=cost_rate * 1e4, borrow_bps_annual=0.0)
    equity_v2 = run_backtest(prices, pd.Series(state, index=index), weights, config).equity.to_numpy()

    figure = Figure(figsize=(10, 4.6))
    ax = figure.subplots()
    bars = np.arange(n)
    ax.step(bars, equity_v1, where="post", color=RED, lw=2.0, label="v1: capital + position value - costs")
    ax.step(bars, equity_v2, where="post", color=BLUE, lw=2.0, label="v2: cash accounting")
    ax.axvspan(5, 15, color="0.93", lw=0, zorder=0)
    ax.annotate(
        f"+{equity_v1.max() / capital - 1:.1%} of 'profit' appears the moment shares are bought",
        (5, equity_v1.max()), textcoords="offset points", xytext=(40, -18), fontsize=9, color=RED,
    )
    ax.annotate(
        f"both finish at {equity_v2[-1]:,.0f}: the costs were the only real P&L",
        (15, equity_v2[-1]), textcoords="offset points", xytext=(-250, 34), fontsize=9, color=BLUE,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8),
    )
    ax.set_xlabel("bar (prices are identical on every bar; the grey band is the holding period)")
    ax.set_ylabel("equity ($)")
    ax.set_title("The v1 bug: on prices that never move, the old formula still reports a gain")
    ax.legend(frameon=False, loc="center right")
    ax.grid(alpha=0.3)
    _save(figure, path)
    return {"v1_peak": float(equity_v1.max()), "v1_final": float(equity_v1[-1]), "v2_final": float(equity_v2[-1])}


def half_life_figure(path: Path) -> dict:
    """Expected decay of a deviation under AR(1) dynamics for several half-lives."""
    horizon = np.arange(0, 127)
    figure = Figure(figsize=(10, 4.4))
    ax = figure.subplots()
    for half_life_days, colour in ((5, GREEN), (12, BLUE), (60, ORANGE)):
        phi = 0.5 ** (1.0 / half_life_days)
        ax.plot(horizon, 100.0 * phi**horizon, color=colour, lw=1.8, label=f"half-life {half_life_days} days (phi = {phi:.3f})")
    ax.axhline(50.0, color=GREY, ls=":", lw=1.0)
    ax.axvline(126, color=GREY, ls="--", lw=1.0, label="end of a 126-day trading window")
    ax.text(2, 52, "half of the deviation left", fontsize=8.5, color=GREY)
    ax.set_xlabel("days after the deviation")
    ax.set_ylabel("expected deviation left (%)")
    ax.set_title("Mean reversion speed: an AR(1) spread forgets a shock at the rate phi per day")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    _save(figure, path)
    return {}


def _annual_sharpe_for_psr(years: float, target: float = 0.95) -> float:
    grid = np.linspace(0.0, 3.0, 30001)
    n = int(round(years * TRADING_DAYS))
    values = np.array([psr_from_moments(s / math.sqrt(TRADING_DAYS), n, 0.0, 3.0) for s in grid])
    return float(grid[np.argmax(values >= target)])


def psr_figure(path: Path) -> dict:
    """Probability that the true Sharpe is above zero, by observed Sharpe and sample length."""
    sharpes = np.linspace(0.0, 1.6, 161)
    figure = Figure(figsize=(10, 4.8))
    ax = figure.subplots()
    needed = {}
    for years, colour in ((2, ORANGE), (6.5, BLUE), (13, GREEN)):
        n = int(round(years * TRADING_DAYS))
        curve = [psr_from_moments(s / math.sqrt(TRADING_DAYS), n, 0.0, 3.0) for s in sharpes]
        needed[years] = _annual_sharpe_for_psr(years)
        ax.plot(sharpes, curve, color=colour, lw=1.8, label=f"{years:g} years of daily returns: need Sharpe {needed[years]:.2f}")
    ax.axhline(0.95, color=RED, ls="--", lw=1.0, label="PSR = 0.95")
    psr_example = psr_from_moments(0.12 / math.sqrt(TRADING_DAYS), 13 * TRADING_DAYS, 0.0, 3.0)
    ax.scatter([0.12], [psr_example], color=GREEN, s=45, zorder=3)
    ax.annotate(
        f"Sharpe 0.12 over 13 years gives PSR {psr_example:.2f}",
        (0.12, psr_example), textcoords="offset points", xytext=(18, -20), fontsize=9, color=GREEN,
    )
    ax.set_xlabel("observed annualised Sharpe ratio")
    ax.set_ylabel("probabilistic Sharpe ratio")
    ax.set_title("How much Sharpe a sample needs before luck stops being a good explanation (normal returns)")
    ax.set_ylim(0.45, 1.01)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    _save(figure, path)
    return {"sharpe_needed_2y": needed[2], "sharpe_needed_13y": needed[13], "psr_at_0_12_over_13y": float(psr_example)}


def leverage_figure(path: Path) -> dict:
    """Net result per trade against gross exposure: leverage scales the answer, not its sign."""
    gross = np.linspace(0.0, 20.0, 201)
    figure = Figure(figsize=(10, 4.6))
    ax = figure.subplots()
    ax.plot(
        gross, (CONTROL_GROSS_BPS - CONTROL_COST_BPS) * gross, color=RED, lw=1.9,
        label=f"GLD/IAU as measured: {CONTROL_GROSS_BPS} bps gross, {CONTROL_COST_BPS} bps cost per unit of notional",
    )
    ax.plot(gross, (15.0 - 10.0) * gross, color=GREEN, lw=1.9, ls="--", label="hypothetical: 15 bps gross, 10 bps cost")
    ax.axhline(0.0, color=GREY, lw=1.0)
    ax.axvline(1.0, color=GREY, ls=":", lw=1.2, label="the 1x cap in the protocol")
    ax.axvline(CONTROL_WANTED_GROSS, color=RED, ls="-.", lw=1.0, label=f"what volatility targeting wanted for GLD/IAU ({CONTROL_WANTED_GROSS:g}x)")
    ax.set_xlabel("gross exposure (multiple of equity)")
    ax.set_ylabel("net result per trade (bps of equity)")
    ax.set_title("Net per trade = (gross bps - cost bps) x exposure: leverage cannot turn a negative edge positive")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax.grid(alpha=0.3)
    _save(figure, path)
    return {"control_net_bps_at_wanted_gross": (CONTROL_GROSS_BPS - CONTROL_COST_BPS) * CONTROL_WANTED_GROSS}


FIGURES = {
    "cointegration_intuition": cointegration_figure,
    "walk_forward_timeline": walk_forward_figure,
    "accounting_bug": accounting_figure,
    "half_life_decay": half_life_figure,
    "psr_intuition": psr_figure,
    "leverage_and_costs": leverage_figure,
}


def make_all(output_dir: str | Path) -> tuple[dict, dict[str, Path]]:
    """Write every explanatory figure to ``output_dir``; return the facts they illustrate and the paths."""
    output = Path(output_dir)
    facts: dict = {}
    paths: dict[str, Path] = {}
    for name, builder in FIGURES.items():
        path = output / f"{name}.png"
        facts.update(builder(path))
        paths[name] = path
    return facts, paths
