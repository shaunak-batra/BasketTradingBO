"""End-to-end research run: data, walk-forward evaluation, statistics and artefacts.

``run_research`` is the single code path behind the command-line interface, the
case-study runner and the end-to-end tests, so every single-basket number
(results/case_studies) comes from the same place. The universe run
(results/v3_universe) does not go through ``run_research``: ``src.universe.evaluate_pair``
calls the same walk-forward engine, ``run_walk_forward``, directly for each pair, and
``src/universe.py`` computes the universe statistics itself.
"""

from __future__ import annotations

import argparse
import logging
import math
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.backtesting.backtester import BacktestConfig
from src.backtesting.metrics import max_drawdown, returns_from_equity, sharpe_ratio, summarize_performance
from src.backtesting.walk_forward import (
    OptimizationConfig,
    WalkForwardConfig,
    WalkForwardResult,
    replay_with_costs,
    run_walk_forward,
)
from src.data.market_data import Downloader, download_adjusted_close, load_prices
from src.risk.var import MIN_OBSERVATIONS, risk_table
from src.strategy.signals import SignalParams
from src.utils.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, build_dataclass, load_config
from src.utils.exceptions import ConfigError
from src.utils.io import save_json, sha256_file
from src.visualization.plots import plot_equity_and_drawdown, plot_fold_sharpes, plot_zscore
from src.visualization.reports import fmt_date, fmt_num, fmt_pct, write_report

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MODES = ("fixed", "optimized")
SECTIONS = ("data", "walk_forward", "strategy", "backtest", "optimization", "reporting")
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "scikit-optimize",
    "scikit-learn",
    "yfinance",
    "matplotlib",
)


@dataclass(frozen=True)
class DataConfig:
    snapshot_dir: str = "data/snapshots"
    max_missing_fraction: float = 0.02

    def __post_init__(self) -> None:
        if not 0 <= self.max_missing_fraction < 1:
            raise ConfigError(f"max_missing_fraction must be in [0, 1), got {self.max_missing_fraction}")


@dataclass(frozen=True)
class ReportingConfig:
    cost_sensitivity_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
    bootstrap_samples: int = 2000
    bootstrap_block: int = 20
    seed: int = 7

    def __post_init__(self) -> None:
        costs = tuple(float(cost) for cost in self.cost_sensitivity_bps)
        if not costs or any(not math.isfinite(cost) or cost < 0 for cost in costs):
            raise ConfigError("cost_sensitivity_bps must be a non-empty list of non-negative numbers")
        object.__setattr__(self, "cost_sensitivity_bps", costs)
        if self.bootstrap_samples < 100 or self.bootstrap_block < 1:
            raise ConfigError("bootstrap_samples must be >= 100 and bootstrap_block >= 1")


@dataclass(frozen=True)
class Protocol:
    """The complete, validated research protocol loaded from YAML."""

    data: DataConfig
    walk_forward: WalkForwardConfig
    strategy: SignalParams
    backtest: BacktestConfig
    optimization: OptimizationConfig
    reporting: ReportingConfig
    path: Path
    sha256: str

    def as_dict(self) -> dict:
        return {
            "data": asdict(self.data),
            "walk_forward": asdict(self.walk_forward),
            "strategy": self.strategy.as_dict(),
            "backtest": asdict(self.backtest),
            "optimization": asdict(self.optimization),
            "reporting": asdict(self.reporting),
        }


def load_protocol(path: str | Path | None = None) -> Protocol:
    """Load and validate the protocol file (default ``config/config.yaml``)."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = load_config(config_path)
    missing = [section for section in SECTIONS if section not in raw]
    unknown = sorted(set(raw) - set(SECTIONS))
    if missing or unknown:
        raise ConfigError(f"config sections missing: {missing}; unknown: {unknown}")
    return Protocol(
        data=build_dataclass(DataConfig, raw["data"], "data"),
        walk_forward=build_dataclass(WalkForwardConfig, raw["walk_forward"], "walk_forward"),
        strategy=build_dataclass(SignalParams, raw["strategy"], "strategy"),
        backtest=build_dataclass(BacktestConfig, raw["backtest"], "backtest"),
        optimization=build_dataclass(OptimizationConfig, raw["optimization"], "optimization"),
        reporting=build_dataclass(ReportingConfig, raw["reporting"], "reporting"),
        path=config_path.resolve(),
        sha256=sha256_file(config_path),
    )


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _git_state() -> dict:
    def git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    # results/ is excluded: runs write their outputs there, so a batch of runs would otherwise mark
    # every run after the first as dirty. ``dirty`` means uncommitted changes to code, configs or docs.
    status = git("status", "--porcelain", "--", ".", ":(exclude)results")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def _package_versions() -> dict:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _fold_counts(result: WalkForwardResult) -> dict:
    reasons = [fold.skip_reason or "" for fold in result.folds]
    return {
        "n_folds": len(result.folds),
        "n_traded": sum(fold.traded for fold in result.folds),
        "n_not_cointegrated": sum(reason.startswith("not cointegrated") for reason in reasons),
        "n_not_hedged": sum(reason.startswith("net exposure") for reason in reasons),
        "n_half_life_too_long": sum(reason.startswith("half-life") for reason in reasons),
        "n_no_in_sample_edge": sum(reason.startswith("no in-sample") for reason in reasons),
    }


def _cost_sensitivity(prices: pd.DataFrame, result: WalkForwardResult, protocol: Protocol) -> list[dict]:
    rows = []
    for cost_bps in protocol.reporting.cost_sensitivity_bps:
        replayed = replay_with_costs(prices, result, replace(protocol.backtest, cost_bps=cost_bps))
        rows.append(
            {
                "cost_bps": cost_bps,
                "total_return": float(replayed.equity.iloc[-1] / replayed.equity.iloc[0] - 1.0),
                "sharpe": sharpe_ratio(returns_from_equity(replayed.equity)),
                "max_drawdown": max_drawdown(replayed.equity)[0],
            }
        )
    return rows


def run_research(
    tickers: Sequence[str],
    start: str,
    end: str,
    mode: str,
    protocol: Protocol,
    output_dir: str | Path,
    refresh_data: bool = False,
    downloader: Downloader = download_adjusted_close,
    command: str | None = None,
) -> dict:
    """Run one basket through the protocol and write all artefacts to ``output_dir``.

    Returns the summary dict that is saved as ``results.json``.
    """
    if mode not in MODES:
        raise ConfigError(f"mode must be one of {MODES}, got {mode!r}")
    tickers = list(tickers)
    output_dir = Path(output_dir)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    snapshot_dir = Path(protocol.data.snapshot_dir)
    if not snapshot_dir.is_absolute():
        snapshot_dir = PROJECT_ROOT / snapshot_dir
    price_data = load_prices(
        tickers, start, end, snapshot_dir, protocol.data.max_missing_fraction, refresh_data, downloader
    )
    prices = price_data.prices
    logger.info("Loaded %d rows for %s (%s, sha256 %s)", len(prices), tickers, price_data.source, price_data.sha256[:12])

    result = run_walk_forward(
        prices,
        protocol.walk_forward,
        protocol.backtest,
        params=protocol.strategy if mode == "fixed" else None,
        optimization=protocol.optimization if mode == "optimized" else None,
    )
    reporting = protocol.reporting
    performance = summarize_performance(
        result.equity,
        result.trades,
        result.gross_exposure,
        result.traded_notional,
        result.trading_costs,
        result.borrow_costs,
        bootstrap_samples=reporting.bootstrap_samples,
        bootstrap_block=reporting.bootstrap_block,
        seed=reporting.seed,
    )
    oos_returns = returns_from_equity(result.equity)
    folds = result.fold_table()

    summary = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "tickers": tickers,
            "start": start,
            "end_exclusive": end,
            "mode": mode,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": _git_state(),
            "python": platform.python_version(),
            "packages": _package_versions(),
            "command": command,
        },
        "protocol": {"path": _display_path(protocol.path), "sha256": protocol.sha256, **protocol.as_dict()},
        "data": {
            "source": price_data.source,
            "path": _display_path(price_data.path),
            "sha256": price_data.sha256,
            "rows": len(prices),
            "first_date": prices.index[0],
            "last_date": prices.index[-1],
            "provenance": price_data.metadata,
        },
        "folds_summary": _fold_counts(result),
        "out_of_sample": performance,
        "cost_sensitivity": _cost_sensitivity(prices, result, protocol),
        "risk": [asdict(estimate) for estimate in risk_table(oos_returns)] if len(oos_returns) >= MIN_OBSERVATIONS else [],
        "folds": folds.to_dict(orient="records"),
    }

    save_json(summary, output_dir / "results.json")
    folds.to_csv(output_dir / "folds.csv", index=False, lineterminator="\n")
    result.trades.to_csv(output_dir / "trades.csv", index=False, lineterminator="\n")
    daily = pd.DataFrame(
        {
            "equity": result.equity,
            "return": result.equity.pct_change(),
            "gross_exposure": result.gross_exposure,
            "net_exposure": result.net_exposure,
            "zscore": result.zscore,
        }
    )
    daily.to_csv(output_dir / "equity.csv", index_label="date", lineterminator="\n")

    title = f"{' / '.join(tickers)} ({mode})"
    plots = {
        "equity": plot_equity_and_drawdown(result, output_dir / "plots" / "equity_drawdown.png", title),
        "zscore": plot_zscore(result, output_dir / "plots" / "zscore.png", title),
        "folds": plot_fold_sharpes(result, output_dir / "plots" / "fold_sharpes.png", title),
    }
    write_report(
        output_dir / "report.html",
        summary,
        folds,
        result.trades,
        {name: path.relative_to(output_dir).as_posix() for name, path in plots.items()},
    )
    logger.info("Wrote results to %s", output_dir)
    return summary


def format_console_summary(summary: dict, output_dir: Path) -> str:
    performance = summary["out_of_sample"]
    folds = summary["folds_summary"]
    return "\n".join(
        [
            f"Out-of-sample {fmt_date(performance['start'])} to {fmt_date(performance['end'])} | "
            f"mode {summary['run']['mode']} | folds traded {folds['n_traded']}/{folds['n_folds']} | "
            f"closed trades {performance['n_trades']}",
            f"Total return {fmt_pct(performance['total_return'])} | CAGR {fmt_pct(performance['cagr'])} | "
            f"Sharpe {fmt_num(performance['sharpe'])} [{fmt_num(performance['sharpe_ci_low'])}, "
            f"{fmt_num(performance['sharpe_ci_high'])}] | PSR {fmt_num(performance['psr'], 3)} | "
            f"max drawdown {fmt_pct(performance['max_drawdown'])}",
            f"Artefacts: {output_dir}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Walk-forward cointegration basket backtest. The research protocol lives in config/config.yaml."
    )
    parser.add_argument("--tickers", nargs="+", required=True, help="two or more tickers, e.g. KO PEP")
    parser.add_argument("--start", required=True, help="first date to download, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="end date (exclusive), YYYY-MM-DD")
    parser.add_argument("--mode", choices=MODES, default="fixed", help="fixed thresholds or Bayesian-optimised per fold")
    parser.add_argument("--config", default=None, help="protocol file (default: config/config.yaml)")
    parser.add_argument("--output-dir", default=None, help="default: results/scratch/<tickers>_<start>_<end>_<mode>")
    parser.add_argument("--refresh-data", action="store_true", help="re-download instead of using the local snapshot")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _parse_date(parser: argparse.ArgumentParser, value: str, flag: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError):
        parser.error(f"{flag} must be a date in YYYY-MM-DD format, got {value!r}")


def main(argv: Sequence[str] | None = None, downloader: Downloader = download_adjusted_close) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tickers = [ticker.strip().upper() for ticker in args.tickers]
    if len(tickers) < 2 or len(set(tickers)) != len(tickers):
        parser.error("provide at least two distinct tickers")
    start = _parse_date(parser, args.start, "--start")
    end = _parse_date(parser, args.end, "--end")
    if start >= end:
        parser.error("--start must be before --end")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    protocol = load_protocol(args.config)
    start_text, end_text = f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "results" / "scratch" / f"{'_'.join(tickers)}_{start_text}_{end_text}_{args.mode}"
    )
    argv_text = list(argv) if argv is not None else sys.argv[1:]
    summary = run_research(
        tickers,
        start_text,
        end_text,
        args.mode,
        protocol,
        output_dir,
        refresh_data=args.refresh_data,
        downloader=downloader,
        command="python scripts/run_pipeline.py " + " ".join(argv_text),
    )
    print(format_console_summary(summary, output_dir))
    return 0
