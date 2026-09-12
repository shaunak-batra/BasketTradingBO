"""Static charts for walk-forward results.

Uses matplotlib's object API (``Figure``) instead of ``pyplot``, so no global
plotting state is touched and no display is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.backtesting.walk_forward import WalkForwardResult

DPI = 110
BLUE = "#1f5f99"
LIGHT_BLUE = "#8fb3d9"
GREEN = "#1a7f37"
RED = "#b23a3a"
ORANGE = "#d08a1e"
GREY = "0.35"


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    return path


def _shade_untraded_folds(ax, result: WalkForwardResult, label: bool) -> None:
    labelled = not label
    for fold in result.folds:
        if not fold.traded:
            ax.axvspan(
                fold.trading_start,
                fold.trading_end,
                color="0.88",
                lw=0,
                zorder=0,
                label=None if labelled else "fold not traded",
            )
            labelled = True


def plot_equity_and_drawdown(result: WalkForwardResult, path: str | Path, title: str) -> Path:
    fig = Figure(figsize=(11, 6.5))
    ax_equity, ax_drawdown = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    equity = result.equity
    values = equity.to_numpy(dtype=float)

    ax_equity.plot(equity.index, values, color=BLUE, lw=1.3, label="out-of-sample equity")
    ax_equity.axhline(result.backtest_config.initial_capital, color=GREY, lw=0.8, ls="--", label="initial capital")
    _shade_untraded_folds(ax_equity, result, label=True)
    ax_equity.set_ylabel("equity ($)")
    ax_equity.set_title(f"{title}: stitched out-of-sample equity")
    ax_equity.legend(loc="upper left", frameon=False)

    drawdown = (values / np.maximum.accumulate(values) - 1.0) * 100.0
    ax_drawdown.fill_between(equity.index, drawdown, 0.0, color=RED, alpha=0.45, lw=0)
    _shade_untraded_folds(ax_drawdown, result, label=False)
    ax_drawdown.set_ylabel("drawdown (%)")

    for ax in (ax_equity, ax_drawdown):
        ax.grid(alpha=0.3)
    fig.align_ylabels()
    return _save(fig, path)


def plot_zscore(result: WalkForwardResult, path: str | Path, title: str) -> Path:
    fig = Figure(figsize=(11, 4.8))
    ax = fig.subplots()
    zscore = result.zscore
    ax.plot(zscore.index, zscore.to_numpy(dtype=float), color=GREEN, lw=0.9, label="spread z-score (fold weights)")

    traded = [fold for fold in result.folds if fold.traded]
    for i, fold in enumerate(traded):
        params = fold.params
        span = {"xmin": fold.trading_start, "xmax": fold.trading_end}
        ax.hlines([params.entry_z, -params.entry_z], colors=RED, linestyles="--", lw=1.0, label="entry" if i == 0 else None, **span)
        ax.hlines([params.exit_z, -params.exit_z], colors=ORANGE, linestyles=":", lw=1.0, label="exit" if i == 0 else None, **span)
        ax.hlines([params.stop_z, -params.stop_z], colors=GREY, linestyles="-.", lw=1.0, label="stop" if i == 0 else None, **span)

    trades = result.trades
    if len(trades):
        signal_dates = pd.DatetimeIndex(pd.to_datetime(trades["signal_date"]))
        z_at_signal = zscore.reindex(signal_dates).to_numpy(dtype=float)
        is_long = trades["direction"].to_numpy(dtype=float) > 0
        ax.scatter(signal_dates[is_long], z_at_signal[is_long], marker="^", color=GREEN, s=34, zorder=3, label="long entry signal")
        ax.scatter(signal_dates[~is_long], z_at_signal[~is_long], marker="v", color=RED, s=34, zorder=3, label="short entry signal")

    _shade_untraded_folds(ax, result, label=True)
    ax.axhline(0.0, color=GREY, lw=0.8)
    finite = np.abs(zscore.to_numpy(dtype=float))
    finite = finite[np.isfinite(finite)]
    band = max((fold.params.stop_z for fold in traded), default=0.0) + 0.3
    extent = finite.max() * 1.05 if finite.size else 0.0
    limit = float(np.clip(max(extent, band), 4.5, 8.0))
    ax.set_ylim(-limit, limit)
    ax.set_ylabel("z-score")
    ax.set_title(f"{title}: out-of-sample z-score and thresholds by fold")
    ax.legend(loc="upper right", ncol=3, fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_fold_sharpes(result: WalkForwardResult, path: str | Path, title: str) -> Path:
    fig = Figure(figsize=(11, 4.2))
    ax = fig.subplots()
    table = result.fold_table()
    traded = table[table["traded"]]
    if traded.empty:
        ax.text(0.5, 0.5, "no fold was traded", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        positions = np.arange(len(traded))
        width = 0.4
        ax.bar(positions - width / 2, traded["is_sharpe"].to_numpy(dtype=float), width, color=LIGHT_BLUE, label="in-sample Sharpe (formation window)")
        ax.bar(positions + width / 2, traded["oos_sharpe"].to_numpy(dtype=float), width, color=BLUE, label="out-of-sample Sharpe (next window)")
        ax.set_xticks(positions)
        ax.set_xticklabels([pd.Timestamp(date).strftime("%Y-%m") for date in traded["trading_start"]], rotation=45, ha="right")
        ax.axhline(0.0, color=GREY, lw=0.8)
        ax.set_ylabel("annualised Sharpe")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"{title}: in-sample vs out-of-sample Sharpe per traded fold")
    return _save(fig, path)
