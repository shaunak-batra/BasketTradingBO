"""README results-table rendering tests."""

from __future__ import annotations

import pytest

from src.visualization.summary import END_MARKER, START_MARKER, extract_block, headline_row, render_results_markdown, replace_block


def json_summary(mode: str = "fixed") -> dict:
    """A minimal JSON-serialised run summary, as saved in results.json."""
    return {
        "run": {"tickers": ["KO", "PEP"], "mode": mode},
        "data": {"sha256": "ab" * 32},
        "folds_summary": {"n_traded": 7, "n_folds": 26},
        "out_of_sample": {
            "start": "2012-01-04T00:00:00",
            "end": "2024-12-31T00:00:00",
            "n_trades": 12,
            "total_return": -0.0312,
            "cagr": -0.0024,
            "sharpe": -0.11,
            "sharpe_ci_low": -0.62,
            "sharpe_ci_high": 0.41,
            "psr": None,
            "max_drawdown": -0.087,
        },
        "cost_sensitivity": [
            {"cost_bps": 0.0, "sharpe": 0.05},
            {"cost_bps": 5.0, "sharpe": -0.11},
            {"cost_bps": 20.0, "sharpe": -0.4},
        ],
    }


def test_headline_row_extracts_the_published_numbers():
    row = headline_row("consumer_staples_ko_pep", json_summary())
    assert row["oos_start"] == "2012-01-04"
    assert row["folds_traded"] == 7 and row["n_folds"] == 26
    assert row["sharpe_at_0bps"] == 0.05 and row["sharpe_at_20bps"] == -0.4
    assert row["results_dir"] == "results/case_studies/consumer_staples_ko_pep/fixed"


def test_markdown_table_formats_missing_values_as_na():
    markdown = render_results_markdown([headline_row("consumer_staples_ko_pep", json_summary())])
    lines = markdown.splitlines()
    assert len(lines) == 3
    assert "[KO / PEP](results/case_studies/consumer_staples_ko_pep/fixed)" in lines[2]
    assert "-3.1%" in lines[2]
    assert "-0.11 [-0.62, 0.41]" in lines[2]
    assert "| n/a |" in lines[2]
    assert "0.05 / -0.40" in lines[2]


def test_replace_and_extract_block_round_trip():
    text = f"# Title\n\n{START_MARKER}\nold table\n{END_MARKER}\n\nAfter.\n"
    updated = replace_block(text, "new table")
    assert extract_block(updated).strip() == "new table"
    assert updated.startswith("# Title") and updated.endswith("After.\n")


def test_missing_markers_are_an_error():
    with pytest.raises(ValueError):
        replace_block("no markers here", "table")
