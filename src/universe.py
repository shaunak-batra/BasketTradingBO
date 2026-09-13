"""Run a protocol across a universe of pairs and report the whole distribution.

This exists because five hand-picked baskets cannot tell you whether a strategy works: you
chose them. Here the pairs are fixed in advance by a rule over a defined universe, every one
is reported, and significance is judged after Benjamini-Hochberg control, so a few
good-looking pairs among sixty are not mistaken for an edge.

Each pair is also replayed with its fold decisions frozen at every pre-registered cost level,
and once with no frictions at all. That separates "the signal is worthless" from "the signal
is real but smaller than the spread it has to cross".
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.metrics import (
    block_bootstrap_sharpe_ci,
    max_drawdown,
    probabilistic_sharpe_ratio,
    returns_from_equity,
    sharpe_ratio,
    trade_statistics,
)
from src.backtesting.walk_forward import replay_with_costs, run_walk_forward
from src.data.market_data import align_prices, download_adjusted_close, validate_prices
from src.utils.exceptions import BasketTradingError
from src.utils.io import sha256_file

logger = logging.getLogger(__name__)
TRADING_DAYS_PER_YEAR = 252


def universe_pairs(families: dict[str, list[str]]) -> list[tuple[str, str, str]]:
    """Every unordered pair inside each family, as ``(family, ticker_a, ticker_b)``.

    Pairs never cross families: the reason to expect a stationary spread is a shared
    economic driver, and that is what a family encodes.
    """
    pairs = []
    for family, tickers in families.items():
        if len(set(tickers)) != len(tickers):
            raise BasketTradingError(f"family {family} repeats a ticker")
        if len(tickers) < 2:
            raise BasketTradingError(f"family {family} needs at least two tickers")
        for a, b in combinations(sorted(tickers), 2):
            pairs.append((family, a, b))
    return pairs


def load_universe_prices(
    tickers: list[str], start: str, end: str, snapshot: Path, refresh: bool = False, downloader=download_adjusted_close
) -> tuple[pd.DataFrame, str]:
    """Download every ticker once into a single snapshot (outer join, gaps kept as NaN)."""
    snapshot = Path(snapshot)
    if snapshot.exists() and not refresh:
        frame = pd.read_csv(snapshot, index_col=0, parse_dates=True, float_precision="round_trip")
        missing = sorted(set(tickers) - set(frame.columns))
        if not missing:
            return frame, sha256_file(snapshot)
        logger.info("Snapshot is missing %s; downloading the universe again", missing)

    series = []
    for number, ticker in enumerate(sorted(tickers), start=1):
        logger.info("Downloading %s (%d/%d)", ticker, number, len(tickers))
        series.append(downloader(ticker, start, end))
    frame = pd.concat(series, axis=1).sort_index()
    frame.index.name = "date"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(snapshot, lineterminator="\n")
    frame = pd.read_csv(snapshot, index_col=0, parse_dates=True, float_precision="round_trip")
    return frame, sha256_file(snapshot)


def pair_prices(frame: pd.DataFrame, a: str, b: str, min_bars: int, max_missing_fraction: float) -> pd.DataFrame:
    """Prices for one pair over the overlap of the two histories.

    Truncating to the overlap (rather than demanding both cover the whole window) lets a
    younger fund into the universe without silently dropping dates from an older one.
    """
    both = frame[[a, b]]
    first_a, first_b = both[a].first_valid_index(), both[b].first_valid_index()
    last_a, last_b = both[a].last_valid_index(), both[b].last_valid_index()
    if None in (first_a, first_b, last_a, last_b):
        raise BasketTradingError(f"{a}/{b}: one of the series is empty")
    first, last = max(first_a, first_b), min(last_a, last_b)
    if first >= last:
        raise BasketTradingError(f"{a}/{b}: histories do not overlap")
    window = both.loc[first:last]
    prices, _ = align_prices([window[a].dropna(), window[b].dropna()], max_missing_fraction)
    if len(prices) < min_bars:
        raise BasketTradingError(f"{a}/{b}: only {len(prices)} overlapping bars, need {min_bars}")
    return validate_prices(prices)


def benjamini_hochberg(p_values: np.ndarray, level: float) -> np.ndarray:
    """Boolean mask of discoveries controlling the false discovery rate at ``level``.

    Step-up procedure: order the p-values, find the largest rank i with p_(i) <= i*q/m, and
    reject everything at or below that p-value. NaN p-values (nothing to test) never count.
    """
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    discoveries = np.zeros(len(p), dtype=bool)
    if not finite.any():
        return discoveries
    ranked = np.sort(p[finite])
    m = len(ranked)
    below = ranked <= level * np.arange(1, m + 1) / m
    if not below.any():
        return discoveries
    cutoff = ranked[np.flatnonzero(below).max()]
    discoveries[finite] = p[finite] <= cutoff
    return discoveries


def evaluate_pair(prices: pd.DataFrame, protocol) -> tuple[dict, pd.Series, pd.Series]:
    """Walk-forward one pair; returns its row, its daily returns and its frictionless returns."""
    result = run_walk_forward(prices, protocol.walk_forward, protocol.backtest, params=protocol.strategy)
    equity = result.equity
    returns = returns_from_equity(equity)
    reasons = [fold.skip_reason or "" for fold in result.folds]
    trades = trade_statistics(result.trades)
    psr = probabilistic_sharpe_ratio(returns)
    traded_folds = [fold for fold in result.folds if fold.traded]
    row = {
        "n_bars": len(prices),
        "oos_start": equity.index[0],
        "oos_end": equity.index[-1],
        "n_folds": len(result.folds),
        "n_traded": len(traded_folds),
        "n_not_cointegrated": sum(reason.startswith("not cointegrated") for reason in reasons),
        "n_not_hedged": sum(reason.startswith("net exposure") for reason in reasons),
        "n_half_life": sum(reason.startswith("half-life") for reason in reasons),
        "n_trades": trades["n_trades"],
        "win_rate": trades["win_rate"],
        "n_zscore_stops": trades["n_zscore_stops"],
        "n_loss_stops": trades["n_loss_stops"],
        "n_time_stops": trades["n_time_stops"],
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "sharpe": sharpe_ratio(returns),
        "psr": psr,
        "p_value": 1.0 - psr if math.isfinite(psr) else math.nan,
        "max_drawdown": max_drawdown(equity)[0],
        "time_in_market": float((result.gross_exposure > 0).mean()),
        "median_net_exposure": float(np.median([fold.net_exposure for fold in result.folds])),
        "median_spread_volatility": (
            float(np.median([fold.sizing_volatility for fold in traded_folds])) if traded_folds else math.nan
        ),
    }

    # Fold decisions stay frozen; only the cost model changes.
    for cost_bps in protocol.reporting.cost_sensitivity_bps:
        replayed = replay_with_costs(prices, result, replace(protocol.backtest, cost_bps=cost_bps))
        row[f"sharpe_at_{cost_bps:g}bps"] = sharpe_ratio(returns_from_equity(replayed.equity))

    frictionless = replay_with_costs(prices, result, replace(protocol.backtest, cost_bps=0.0, borrow_bps_annual=0.0))
    frictionless_returns = returns_from_equity(frictionless.equity)
    row["sharpe_frictionless"] = sharpe_ratio(frictionless_returns)
    row["total_return_frictionless"] = float(frictionless.equity.iloc[-1] / frictionless.equity.iloc[0] - 1.0)

    closed = result.trades[~result.trades["is_open"].astype(bool)] if len(result.trades) else result.trades
    gross_closed = (
        frictionless.trades[~frictionless.trades["is_open"].astype(bool)]
        if len(frictionless.trades)
        else frictionless.trades
    )
    if len(closed):
        notional = closed["entry_gross_notional"].mean()
        row["mean_gross_pnl_bps"] = float(gross_closed["gross_pnl"].mean() / notional * 1e4)
        row["mean_cost_bps"] = float((closed["trading_cost"] + closed["borrow_cost"]).mean() / notional * 1e4)
    else:
        row["mean_gross_pnl_bps"] = math.nan
        row["mean_cost_bps"] = math.nan

    label = f"{prices.columns[0]}/{prices.columns[1]}"
    return row, returns.rename(label), frictionless_returns.rename(label)


def portfolio_summary(returns_by_pair: pd.DataFrame, reporting) -> dict:
    """Equal-weight, daily rebalanced portfolio across every pair."""
    portfolio = returns_by_pair.fillna(0.0).mean(axis=1)
    equity = (1.0 + portfolio).cumprod()
    years = len(portfolio) / TRADING_DAYS_PER_YEAR
    low, high = block_bootstrap_sharpe_ci(
        portfolio, n_samples=reporting.bootstrap_samples, block=reporting.bootstrap_block, seed=reporting.seed
    )
    growth = float(equity.iloc[-1])
    return {
        "n_pairs": int(returns_by_pair.shape[1]),
        "start": portfolio.index[0],
        "end": portfolio.index[-1],
        "total_return": growth - 1.0,
        "cagr": growth ** (1 / years) - 1.0 if years > 0 and growth > 0 else math.nan,
        "ann_volatility": float(portfolio.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)),
        "sharpe": sharpe_ratio(portfolio),
        "sharpe_ci_low": low,
        "sharpe_ci_high": high,
        "psr": probabilistic_sharpe_ratio(portfolio),
        "max_drawdown": max_drawdown(equity)[0],
    }


def summarise(pairs_frame: pd.DataFrame) -> dict:
    """Distribution statistics across the evaluated pairs."""
    traded = pairs_frame[pairs_frame["n_trades"] > 0]
    return {
        "n_evaluated": int(len(pairs_frame)),
        "n_with_trades": int(len(traded)),
        "n_folds_total": int(pairs_frame["n_folds"].sum()),
        "n_folds_traded": int(pairs_frame["n_traded"].sum()),
        "n_folds_not_cointegrated": int(pairs_frame["n_not_cointegrated"].sum()),
        "n_folds_not_hedged": int(pairs_frame["n_not_hedged"].sum()),
        "n_folds_half_life": int(pairs_frame["n_half_life"].sum()),
        "n_round_trips": int(pairs_frame["n_trades"].sum()),
        "n_zscore_stops": int(pairs_frame["n_zscore_stops"].sum()),
        "n_loss_stops": int(pairs_frame["n_loss_stops"].sum()),
        "n_time_stops": int(pairs_frame["n_time_stops"].sum()),
        "median_sharpe": float(traded["sharpe"].median()) if len(traded) else math.nan,
        "median_sharpe_frictionless": float(traded["sharpe_frictionless"].median()) if len(traded) else math.nan,
        "n_positive_sharpe": int((traded["sharpe"] > 0).sum()),
        "n_positive_sharpe_frictionless": int((traded["sharpe_frictionless"] > 0).sum()),
        "median_gross_pnl_bps": float(traded["mean_gross_pnl_bps"].median()) if len(traded) else math.nan,
        "median_cost_bps": float(traded["mean_cost_bps"].median()) if len(traded) else math.nan,
        "median_folds_traded": float(pairs_frame["n_traded"].median()),
        "median_folds": float(pairs_frame["n_folds"].median()),
        "n_discoveries": int(pairs_frame["discovery"].sum()),
    }


def plot_universe_results(
    pairs: pd.DataFrame, returns_frame: pd.DataFrame, frictionless_frame: pd.DataFrame, output_dir: str | Path
) -> dict[str, Path]:
    """Three charts: the Sharpe distribution, the per-trade economics, and the portfolio."""
    from matplotlib.figure import Figure  # imported here so importing this module stays cheap

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    traded = pairs[pairs["n_trades"] > 0]
    blue, light, red, green, grey = "#1f5f99", "#8fb3d9", "#b23a3a", "#1a7f37", "0.35"
    paths: dict[str, Path] = {}

    def save(fig: "Figure", name: str) -> Path:
        path = output / name
        fig.savefig(path, dpi=110, bbox_inches="tight")
        return path

    # 1. Where the pairs landed, with and without frictions.
    figure = Figure(figsize=(9, 4.5))
    ax = figure.subplots()
    edges = np.histogram_bin_edges(
        np.concatenate([traded["sharpe"].to_numpy(), traded["sharpe_frictionless"].to_numpy()]), bins=18
    )
    ax.hist(traded["sharpe_frictionless"], bins=edges, color=light, label="no frictions")
    ax.hist(traded["sharpe"], bins=edges, color=blue, alpha=0.85, label="after costs")
    ax.axvline(0.0, color=grey, lw=1.0)
    ax.axvline(
        traded["sharpe"].median(), color=red, ls="--", lw=1.3,
        label=f"median after costs {traded['sharpe'].median():.2f}",
    )
    ax.set_xlabel("out-of-sample Sharpe ratio")
    ax.set_ylabel("number of pairs")
    ax.set_title(f"Out-of-sample Sharpe across {len(traded)} traded pairs")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    paths["sharpe_distribution"] = save(figure, "sharpe_distribution.png")

    # 2. Every trade has to clear its own costs, and most do not.
    figure = Figure(figsize=(10, 4.8))
    ax = figure.subplots()
    net = (traded["mean_gross_pnl_bps"] - traded["mean_cost_bps"]).sort_values()
    colours = [green if value > 0 else red for value in net]
    positions = np.arange(len(net))
    ax.bar(positions, net.to_numpy(), color=colours, width=0.8)
    ax.axhline(0.0, color=grey, lw=1.0)
    labels = traded.set_index(traded.index)["pair"]
    for number, control in enumerate(("GLD/IAU", "IVV/SPY")):
        matches = [i for i, index in enumerate(net.index) if labels.loc[index] == control]
        if matches:
            position = matches[0]
            ax.annotate(  # stagger the two labels so they do not sit on top of each other
                f"{control} (control)",
                (position, net.iloc[position]),
                textcoords="offset points",
                xytext=(-70 if number == 0 else 70, -34 - 26 * number),
                ha="center",
                fontsize=9,
                color=grey,
                arrowprops=dict(arrowstyle="->", color=grey, lw=0.8),
            )
    ax.set_xticks([])
    ax.set_xlabel("pairs, sorted")
    ax.set_ylabel("gross P&L minus cost, per trade (bps of notional)")
    ax.set_title(
        f"Per-trade economics: {int((net > 0).sum())} of {len(net)} pairs earn more than they pay to trade"
    )
    ax.grid(axis="y", alpha=0.3)
    paths["per_trade_economics"] = save(figure, "per_trade_economics.png")

    # 3. The portfolio, with and without frictions.
    figure = Figure(figsize=(10, 4.5))
    ax = figure.subplots()
    for frame, colour, label in (
        (frictionless_frame, light, "no frictions"),
        (returns_frame, blue, "after costs"),
    ):
        equity = (1.0 + frame.fillna(0.0).mean(axis=1)).cumprod() * 100.0
        ax.plot(equity.index, equity.to_numpy(), color=colour, lw=1.4, label=label)
    ax.axhline(100.0, color=grey, lw=0.8, ls="--")
    ax.set_ylabel("equal-weight portfolio (start = 100)")
    ax.set_title("Equal-weight portfolio of every pair in the universe")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    paths["portfolio_equity"] = save(figure, "portfolio_equity.png")

    return paths


def render_markdown(summary: dict, pairs: pd.DataFrame) -> str:
    portfolio = summary["portfolio"]
    frictionless = summary["portfolio_frictionless"]
    tested = summary["tested"]
    lines = [
        f"# v3.0 universe run ({summary['run']['created_utc']})",
        "",
        f"Protocol `{summary['run']['protocol']}` (sha256 {summary['run']['protocol_sha256'][:12]}), "
        f"universe `{summary['run']['universe']}`, price snapshot sha256 {summary['run']['data_sha256'][:12]}.",
        "",
        "## Distribution across pairs",
        "",
        f"- pairs defined: {summary['n_pairs_defined']}, evaluated: {tested['n_evaluated']}, "
        f"skipped for data: {summary['n_pairs_skipped']}",
        f"- pairs that traded at least once: {tested['n_with_trades']}; round trips: {tested['n_round_trips']}",
        f"- stop exits: z-score {tested['n_zscore_stops']}, loss {tested['n_loss_stops']}, "
        f"time {tested['n_time_stops']}",
        f"- folds traded: {tested['n_folds_traded']} of {tested['n_folds_total']} "
        f"({tested['n_folds_traded'] / tested['n_folds_total']:.1%}); skipped as not cointegrated "
        f"{tested['n_folds_not_cointegrated']}, not hedged {tested['n_folds_not_hedged']}, "
        f"half-life {tested['n_folds_half_life']}",
        f"- median out-of-sample Sharpe: {tested['median_sharpe']:.3f} with costs, "
        f"{tested['median_sharpe_frictionless']:.3f} with no frictions",
        f"- positive Sharpe: {tested['n_positive_sharpe']} of {tested['n_with_trades']} with costs, "
        f"{tested['n_positive_sharpe_frictionless']} with no frictions",
        f"- median gross P&L per trade {tested['median_gross_pnl_bps']:.1f} bps of notional against "
        f"{tested['median_cost_bps']:.1f} bps of cost",
        f"- discoveries after Benjamini-Hochberg at {summary['fdr_level']:.0%}: {tested['n_discoveries']}",
        "",
        "## Equal-weight portfolio of all pairs",
        "",
        f"- {portfolio['start']:%Y-%m-%d} to {portfolio['end']:%Y-%m-%d}, {portfolio['n_pairs']} pairs",
        f"- with costs: total return {portfolio['total_return']:.2%}, Sharpe {portfolio['sharpe']:.3f} "
        f"[{portfolio['sharpe_ci_low']:.2f}, {portfolio['sharpe_ci_high']:.2f}], PSR {portfolio['psr']:.3f}, "
        f"max drawdown {portfolio['max_drawdown']:.2%}",
        f"- with no frictions: total return {frictionless['total_return']:.2%}, "
        f"Sharpe {frictionless['sharpe']:.3f}, PSR {frictionless['psr']:.3f}",
        "",
        "## Best and worst pairs by Sharpe",
        "",
        "| Pair | Family | Folds traded | Trades | Total return | Sharpe | Sharpe (no frictions) | PSR | Discovery |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    ranked = pairs.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False)
    for row in pd.concat([ranked.head(5), ranked.tail(5)]).itertuples():
        lines.append(
            f"| {row.pair} | {row.family} | {row.n_traded}/{row.n_folds} | {row.n_trades} | "
            f"{row.total_return:.2%} | {row.sharpe:.2f} | {row.sharpe_frictionless:.2f} | {row.psr:.2f} | "
            f"{'yes' if row.discovery else 'no'} |"
        )
    return "\n".join(lines) + "\n"
