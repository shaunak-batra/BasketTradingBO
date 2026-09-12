"""End-to-end runs of the research pipeline on synthetic data (no network access)."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from src.pipeline import load_protocol, main, run_research
from src.strategy.signals import SignalParams
from src.utils.io import sha256_file
from tests.fixtures.synthetic import cointegrated_prices

TICKERS = ["AAA", "BBB", "CCC"]
START, END = "2015-01-01", "2018-01-01"


def synthetic_downloader():
    prices, _ = cointegrated_prices(n=800, half_life=5.0, spread_sigma=0.01, seed=21)
    prices.columns = TICKERS
    return lambda ticker, start, end: prices[ticker].rename(ticker)


def offline_downloader(*_):
    raise AssertionError("the network downloader must not be called")


def strict_json(path):
    def reject(constant):
        raise ValueError(f"non-standard JSON constant {constant}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


@pytest.fixture
def protocol(tmp_path):
    base = load_protocol()
    return replace(
        base,
        data=replace(base.data, snapshot_dir=str(tmp_path / "snapshots")),
        walk_forward=replace(base.walk_forward, formation_days=250, trading_days=60, max_half_life_days=60.0),
        strategy=SignalParams(entry_z=1.5, exit_z=0.25, stop_z=4.0, lookback=40),
        optimization=replace(
            base.optimization, n_calls=6, n_initial_points=3, space={**base.optimization.space, "lookback": (20, 120)}
        ),
        reporting=replace(base.reporting, bootstrap_samples=200),
    )


def test_repository_protocol_loads_and_is_internally_consistent():
    protocol = load_protocol()
    assert protocol.optimization.space["lookback"][1] <= protocol.walk_forward.formation_days - 20
    assert protocol.strategy.lookback <= protocol.walk_forward.formation_days - 20
    assert protocol.backtest.cost_bps in protocol.reporting.cost_sensitivity_bps
    assert 0.0 in protocol.reporting.cost_sensitivity_bps
    assert len(protocol.sha256) == 64


def test_fixed_run_writes_mutually_consistent_artefacts(tmp_path, protocol):
    out = tmp_path / "run"
    summary = run_research(TICKERS, START, END, "fixed", protocol, out, downloader=synthetic_downloader())
    saved = strict_json(out / "results.json")
    performance = saved["out_of_sample"]

    equity = pd.read_csv(out / "equity.csv", index_col=0, parse_dates=True)
    assert performance["total_return"] == pytest.approx(equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1, rel=1e-12)

    trades = pd.read_csv(out / "trades.csv")
    assert performance["n_trades"] > 0
    assert performance["n_trades"] == int((~trades["is_open"].astype(bool)).sum())

    snapshot = tmp_path / "snapshots" / f"{'_'.join(TICKERS)}_{START}_{END}.csv"
    assert saved["data"]["sha256"] == sha256_file(snapshot)

    base_cost = next(row for row in saved["cost_sensitivity"] if row["cost_bps"] == protocol.backtest.cost_bps)
    assert base_cost["total_return"] == pytest.approx(performance["total_return"], rel=1e-12)

    report = (out / "report.html").read_text(encoding="utf-8")
    assert saved["data"]["sha256"] in report
    assert "statistically distinguishable" in report or "PSR exceeds" in report
    for name in ("equity_drawdown.png", "zscore.png", "fold_sharpes.png"):
        assert (out / "plots" / name).stat().st_size > 1000
    assert summary["folds_summary"]["n_folds"] == len(saved["folds"])


def test_optimized_run_records_the_trials_behind_every_traded_fold(tmp_path, protocol):
    out = tmp_path / "optimized"
    run_research(TICKERS, START, END, "optimized", protocol, out, downloader=synthetic_downloader())
    saved = strict_json(out / "results.json")
    traded = [fold for fold in saved["folds"] if fold["traded"]]
    assert saved["run"]["mode"] == "optimized"
    assert traded
    assert all(fold["n_trials"] == protocol.optimization.n_calls for fold in traded)


def test_second_run_uses_the_snapshot_and_reproduces_the_results(tmp_path, protocol):
    first = run_research(TICKERS, START, END, "fixed", protocol, tmp_path / "a", downloader=synthetic_downloader())
    second = run_research(TICKERS, START, END, "fixed", protocol, tmp_path / "b", downloader=offline_downloader)
    assert second["data"]["source"] == "snapshot"
    assert second["data"]["sha256"] == first["data"]["sha256"]
    assert second["out_of_sample"]["total_return"] == first["out_of_sample"]["total_return"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--tickers", "AAA", "--start", "2020-01-01", "--end", "2021-01-01"],
        ["--tickers", "AAA", "aaa", "--start", "2020-01-01", "--end", "2021-01-01"],
        ["--tickers", "AAA", "BBB", "--start", "2021-01-01", "--end", "2020-01-01"],
        ["--tickers", "AAA", "BBB", "--start", "not-a-date", "--end", "2020-01-01"],
        ["--tickers", "AAA", "BBB", "--start", "2020-01-01", "--end", "2021-01-01", "--mode", "grid"],
    ],
)
def test_cli_rejects_invalid_arguments(argv):
    with pytest.raises(SystemExit):
        main(argv, downloader=offline_downloader)


def test_cli_runs_end_to_end(tmp_path, protocol, monkeypatch, capsys):
    monkeypatch.setattr("src.pipeline.load_protocol", lambda path=None: protocol)
    out = tmp_path / "cli"
    code = main(
        ["--tickers", *TICKERS, "--start", START, "--end", END, "--output-dir", str(out)],
        downloader=synthetic_downloader(),
    )
    assert code == 0
    assert (out / "results.json").exists()
    assert "Total return" in capsys.readouterr().out
