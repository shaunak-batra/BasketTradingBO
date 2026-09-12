"""Case-study headline table, built from results JSON and written into README.md between markers."""

from __future__ import annotations

START_MARKER = "<!-- CASE_STUDY_RESULTS:START -->"
END_MARKER = "<!-- CASE_STUDY_RESULTS:END -->"


def headline_row(basket: str, summary: dict) -> dict:
    """Headline numbers from a JSON-serialised run summary (dates as strings, NaN as None)."""
    performance = summary["out_of_sample"]
    folds = summary["folds_summary"]
    by_cost = {float(row["cost_bps"]): row for row in summary["cost_sensitivity"]}
    mode = summary["run"]["mode"]
    return {
        "basket": basket,
        "tickers": list(summary["run"]["tickers"]),
        "mode": mode,
        "oos_start": str(performance["start"])[:10],
        "oos_end": str(performance["end"])[:10],
        "folds_traded": folds["n_traded"],
        "n_folds": folds["n_folds"],
        "n_trades": performance["n_trades"],
        "total_return": performance["total_return"],
        "cagr": performance["cagr"],
        "sharpe": performance["sharpe"],
        "sharpe_ci_low": performance["sharpe_ci_low"],
        "sharpe_ci_high": performance["sharpe_ci_high"],
        "psr": performance["psr"],
        "max_drawdown": performance["max_drawdown"],
        "sharpe_at_0bps": by_cost.get(0.0, {}).get("sharpe"),
        "sharpe_at_20bps": by_cost.get(20.0, {}).get("sharpe"),
        "data_sha256": summary["data"]["sha256"],
        "results_dir": f"results/case_studies/{basket}/{mode}",
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_results_markdown(rows: list[dict]) -> str:
    lines = [
        "| Basket | Mode | Out-of-sample | Folds traded | Closed trades | Total return | CAGR | Sharpe [95% CI] | PSR | Max drawdown | Sharpe at 0 / 20 bps |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        cells = [
            f"[{' / '.join(row['tickers'])}]({row['results_dir']})",
            row["mode"],
            f"{row['oos_start'][:7]} to {row['oos_end'][:7]}",
            f"{row['folds_traded']}/{row['n_folds']}",
            str(row["n_trades"]),
            _pct(row["total_return"]),
            _pct(row["cagr"]),
            f"{_num(row['sharpe'])} [{_num(row['sharpe_ci_low'])}, {_num(row['sharpe_ci_high'])}]",
            _num(row["psr"]),
            _pct(row["max_drawdown"]),
            f"{_num(row['sharpe_at_0bps'])} / {_num(row['sharpe_at_20bps'])}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _marker_positions(text: str) -> tuple[int, int]:
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise ValueError("README is missing the case-study result markers")
    return start, end


def extract_block(text: str) -> str:
    start, end = _marker_positions(text)
    return text[start + len(START_MARKER) : end]


def replace_block(text: str, markdown: str) -> str:
    start, end = _marker_positions(text)
    return text[: start + len(START_MARKER)] + "\n" + markdown.strip() + "\n" + text[end:]
