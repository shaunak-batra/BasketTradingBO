"""Self-contained HTML report built from the run summary (the dict also saved as results.json)."""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

LIMITATIONS = (
    "Baskets were chosen by the researcher from companies that exist today, so selection and survivorship "
    "bias are not controlled.",
    "Prices are Yahoo Finance adjusted closes. Dividend adjustment rewrites history and Yahoo revises it over "
    "time; results are pinned to the SHA-256 of the price snapshot used.",
    "Fills happen at the next daily close with a flat cost per side. There are no bid-ask dynamics, market "
    "impact, short-sale restrictions or borrow recalls.",
    "Fractional shares, no margin interest, no rebate on short proceeds, and uninvested cash earns nothing.",
    "Weights are frozen within each trading window, so the dollar hedge drifts as prices move.",
    "With few round trips the point Sharpe ratio is noisy; the bootstrap interval and PSR are the relevant "
    "evidence, and neither accounts for how many baskets or protocols a researcher tried.",
)

STYLE = """
body{font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:#1d2330;background:#f6f7f9;margin:0}
main{max-width:1100px;margin:0 auto;padding:32px 24px 64px;background:#fff}
h1{font-size:1.55rem;margin:0 0 4px}
h2{font-size:1.1rem;margin:32px 0 10px;border-bottom:1px solid #dde1e7;padding-bottom:4px}
.meta{color:#5b6475;font-size:.9rem}
.verdict{background:#f1f5fb;border-left:4px solid #1f5f99;padding:12px 16px;margin:18px 0;line-height:1.5}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{padding:5px 8px;border-bottom:1px solid #eceff3;text-align:left;vertical-align:top}
thead th{background:#f3f5f8}
table.kv th{width:48%;font-weight:600;color:#39414f}
figure{margin:18px 0}
img{max-width:100%;height:auto;border:1px solid #eceff3}
figcaption{color:#5b6475;font-size:.85rem}
li{margin:4px 0;line-height:1.45}
"""


def _missing(value: object) -> bool:
    if value is None or value is pd.NaT:
        return True
    return isinstance(value, (float, np.floating)) and math.isnan(value)


def fmt_pct(value: object, digits: int = 2) -> str:
    if _missing(value):
        return "n/a"
    number = float(value)
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number * 100:.{digits}f}%"


def fmt_num(value: object, digits: int = 2) -> str:
    if _missing(value):
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,d}"
    number = float(value)
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:,.{digits}f}"


def fmt_date(value: object) -> str:
    if _missing(value):
        return "n/a"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _key_values(rows: list[tuple[str, object]]) -> str:
    body = "".join(f"<tr><th>{_escape(key)}</th><td>{_escape(value)}</td></tr>" for key, value in rows)
    return f'<table class="kv"><tbody>{body}</tbody></table>'


def _flatten(prefix: str, value: object) -> list[tuple[str, object]]:
    if isinstance(value, dict):
        items: list[tuple[str, object]] = []
        for key, inner in value.items():
            items.extend(_flatten(f"{prefix}.{key}" if prefix else str(key), inner))
        return items
    if isinstance(value, (list, tuple)):
        return [(prefix, ", ".join(str(item) for item in value))]
    return [(prefix, value)]


def verdict(performance: dict) -> str:
    """Plain-language reading of the headline statistics."""
    sentences = [
        f"Out-of-sample Sharpe {fmt_num(performance['sharpe'])} (95% block-bootstrap interval "
        f"{fmt_num(performance['sharpe_ci_low'])} to {fmt_num(performance['sharpe_ci_high'])}), "
        f"probabilistic Sharpe ratio {fmt_num(performance['psr'], 3)}."
    ]
    psr = performance["psr"]
    if _missing(psr):
        sentences.append("Significance could not be assessed because the returns have no variance.")
    elif psr < 0.95:
        sentences.append("The result is not statistically distinguishable from zero skill at the 95% level.")
    else:
        sentences.append(
            "PSR exceeds 0.95, so a true Sharpe ratio above zero is likely under the PSR assumptions; that "
            "still does not correct for how many baskets or protocols were tried."
        )
    n_trades = performance["n_trades"]
    if n_trades < 30:
        sentences.append(f"Only {n_trades} closed round trips, so per-trade statistics are imprecise.")
    return " ".join(sentences)


def write_report(
    path: str | Path,
    summary: dict,
    folds: pd.DataFrame,
    trades: pd.DataFrame,
    plots: dict[str, str],
) -> Path:
    """Write ``report.html``. ``plots`` maps "equity", "zscore" and "folds" to paths relative to the report."""
    run = summary["run"]
    performance = summary["out_of_sample"]
    data = summary["data"]
    fold_counts = summary["folds_summary"]
    git = run.get("git") or {}
    commit = (git.get("commit") or "unknown")[:10] + (" (uncommitted changes)" if git.get("dirty") else "")
    title = f"{' / '.join(run['tickers'])}: walk-forward {run['mode']} run"

    returns_table = _key_values(
        [
            ("Out-of-sample period", f"{fmt_date(performance['start'])} to {fmt_date(performance['end'])} ({performance['n_days']} days)"),
            ("Total return", fmt_pct(performance["total_return"])),
            ("CAGR", fmt_pct(performance["cagr"])),
            ("Annualised volatility", fmt_pct(performance["ann_volatility"])),
            ("Sharpe (95% bootstrap interval)", f"{fmt_num(performance['sharpe'])} ({fmt_num(performance['sharpe_ci_low'])} to {fmt_num(performance['sharpe_ci_high'])})"),
            ("Probabilistic Sharpe ratio", fmt_num(performance["psr"], 3)),
            ("Sortino", fmt_num(performance["sortino"])),
            ("Max drawdown (longest spell, days)", f"{fmt_pct(performance['max_drawdown'])} ({performance['max_drawdown_days']})"),
            ("Calmar", fmt_num(performance["calmar"])),
            ("Skewness / kurtosis of daily returns", f"{fmt_num(performance['skewness'])} / {fmt_num(performance['kurtosis'])}"),
        ]
    )
    trading_table = _key_values(
        [
            ("Folds traded", f"{fold_counts['n_traded']} of {fold_counts['n_folds']}"),
            (
                "Skipped: not cointegrated / not hedged / half-life / no in-sample edge",
                f"{fold_counts['n_not_cointegrated']} / {fold_counts.get('n_not_hedged', 0)} / "
                f"{fold_counts['n_half_life_too_long']} / {fold_counts['n_no_in_sample_edge']}",
            ),
            ("Closed round trips", fmt_num(performance["n_trades"])),
            ("Win rate (per round trip)", fmt_pct(performance["win_rate"], 1)),
            ("Profit factor", fmt_num(performance["profit_factor"])),
            ("Average trade return on equity", fmt_pct(performance["avg_trade_return"], 3)),
            ("Average holding period", f"{fmt_num(performance['avg_holding_days'], 1)} days"),
            (
                "Stop exits: z-score / loss / time",
                f"{fmt_num(performance['n_zscore_stops'])} / {fmt_num(performance['n_loss_stops'])} / "
                f"{fmt_num(performance['n_time_stops'])}",
            ),
            ("Time in market", fmt_pct(performance["time_in_market"], 1)),
            ("Annual turnover (x equity)", fmt_num(performance["turnover_annual"], 1)),
            ("Trading costs / borrow costs ($)", f"{fmt_num(performance['total_trading_cost'])} / {fmt_num(performance['total_borrow_cost'])}"),
        ]
    )

    fold_rows = []
    for row in folds.to_dict(orient="records"):
        params = (
            f"{fmt_num(row['entry_z'])} / {fmt_num(row['exit_z'])} / {fmt_num(row['stop_z'])} / {fmt_num(row['lookback'], 0)}"
            if row["traded"]
            else ""
        )
        fold_rows.append(
            [
                int(row["fold"]),
                f"{fmt_date(row['trading_start'])} to {fmt_date(row['trading_end'])}",
                f"{fmt_num(row['trace_stat'])} / {fmt_num(row['trace_critical'])}",
                fmt_num(row["half_life_days"], 1),
                "traded" if row["traded"] else row["skip_reason"],
                params,
                fmt_num(row["is_sharpe"]),
                fmt_num(row["is_deflated_sharpe"], 3),
                fmt_pct(row["oos_return"]),
                fmt_num(row["oos_sharpe"]),
                fmt_num(row["oos_trades"]),
            ]
        )

    trade_rows = [
        [
            fmt_date(trade["signal_date"]),
            fmt_date(trade["entry_date"]),
            fmt_date(trade["exit_date"]),
            "long spread" if trade["direction"] > 0 else "short spread",
            trade["exit_reason"],
            fmt_num(trade["holding_bars"]),
            fmt_num(trade["net_pnl"]),
            fmt_pct(trade["return_on_equity"], 3),
            int(trade["fold"]),
        ]
        for trade in trades.to_dict(orient="records")
    ]

    cost_rows = [
        [fmt_num(row["cost_bps"], 1), fmt_pct(row["total_return"]), fmt_num(row["sharpe"]), fmt_pct(row["max_drawdown"])]
        for row in summary["cost_sensitivity"]
    ]
    risk_rows = [
        [row["method"], fmt_pct(row["confidence"], 0), fmt_pct(row["var"], 3), fmt_pct(row["expected_shortfall"], 3)]
        for row in summary["risk"]
    ]
    provenance = data.get("provenance") or {}
    data_table = _key_values(
        [
            ("Source this run", data["source"]),
            ("Snapshot file", data["path"]),
            ("Snapshot SHA-256", data["sha256"]),
            ("Rows / first date / last date", f"{data['rows']} / {fmt_date(data['first_date'])} / {fmt_date(data['last_date'])}"),
            ("Dates dropped while aligning tickers", provenance.get("dropped_dates", "n/a")),
            ("Downloaded (UTC)", provenance.get("downloaded_at_utc", "n/a")),
            ("Daily moves above 40% (log) flagged", len(provenance.get("suspicious_moves", []))),
        ]
    )
    figures = "".join(
        f'<figure><img src="{_escape(plots[key])}" alt="{_escape(caption)}"><figcaption>{_escape(caption)}</figcaption></figure>'
        for key, caption in (
            ("equity", "Stitched out-of-sample equity and drawdown. Grey spans are folds that were not traded."),
            ("zscore", "Out-of-sample z-score with each fold's thresholds and entry signals."),
            ("folds", "In-sample (formation) vs out-of-sample Sharpe for each traded fold."),
        )
    )
    limitations = "".join(f"<li>{_escape(item)}</li>" for item in LIMITATIONS)

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>{STYLE}</style>
</head>
<body>
<main>
<h1>{_escape(title)}</h1>
<p class="meta">Generated {_escape(run['created_utc'])} UTC from commit {_escape(commit)}. Protocol SHA-256 {_escape(summary['protocol']['sha256'][:16])}.</p>
<div class="verdict">{_escape(verdict(performance))}</div>

<h2>Headline statistics (out-of-sample only)</h2>
<div class="grid">{returns_table}{trading_table}</div>

<h2>Charts</h2>
{figures}

<h2>Folds</h2>
{_table(["Fold", "Trading window", "Trace stat / 5% crit", "Half-life (days)", "Status", "Entry / exit / stop / lookback", "In-sample Sharpe", "In-sample deflated Sharpe", "OOS return", "OOS Sharpe", "OOS trades"], fold_rows)}

<h2>Transaction-cost sensitivity</h2>
<p>Fold decisions (weights, parameters, signals) are frozen; only the per-side cost changes.</p>
{_table(["Cost (bps per side)", "Total return", "Sharpe", "Max drawdown"], cost_rows)}

<h2>Daily VaR and Expected Shortfall</h2>
{_table(["Method", "Confidence", "VaR", "Expected shortfall"], risk_rows) if risk_rows else "<p>Not enough out-of-sample returns.</p>"}

<h2>Data provenance</h2>
{data_table}

<h2>Protocol</h2>
{_key_values(_flatten("", summary["protocol"]))}

<h2>Trades</h2>
{_table(["Signal", "Entry", "Exit", "Direction", "Exit reason", "Bars held", "Net P&L ($)", "Return on equity", "Fold"], trade_rows) if trade_rows else "<p>No trades.</p>"}

<h2>Limitations</h2>
<ul>{limitations}</ul>
<p class="meta">Command: {_escape(run.get('command') or 'n/a')}</p>
</main>
</body>
</html>
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8", newline="\n")
    return path
