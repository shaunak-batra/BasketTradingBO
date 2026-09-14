#!/usr/bin/env python3
"""Run the pre-registered case studies and regenerate the README results table.

Examples:
    python scripts/run_case_studies.py                                   # every basket, both modes
    python scripts/run_case_studies.py --only consumer_staples_ko_pep    # selected baskets
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import MODES, load_protocol, run_research  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.exceptions import ConfigError  # noqa: E402
from src.utils.io import load_json, save_json, to_jsonable  # noqa: E402
from src.visualization.summary import headline_row, render_results_markdown, replace_block  # noqa: E402

CASE_STUDY_DIR = PROJECT_ROOT / "results" / "case_studies"
STUDIES_PATH = PROJECT_ROOT / "config" / "case_studies.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pre-registered case studies.")
    parser.add_argument("--studies", default=str(STUDIES_PATH), help="case study definitions")
    parser.add_argument("--config", default=None, help="protocol file (default: config/config.yaml)")
    parser.add_argument("--only", nargs="+", default=None, help="basket names to run")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--refresh-data", action="store_true", help="re-download instead of using snapshots")
    parser.add_argument("--no-readme", action="store_true", help="do not rewrite the README results table")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    studies = load_config(args.studies).get("baskets") or []
    names = [study["name"] for study in studies]
    if not names or len(set(names)) != len(names):
        raise ConfigError("case studies must define at least one basket with a unique name")
    if args.only:
        unknown = sorted(set(args.only) - set(names))
        if unknown:
            parser.error(f"unknown baskets: {unknown}")
    protocol = load_protocol(args.config)

    # Keep earlier rows only if they were produced under the same protocol file.
    summary_path = CASE_STUDY_DIR / "summary.json"
    previous = load_json(summary_path) if summary_path.exists() else {}
    rows = {}
    if previous.get("protocol_sha256") == protocol.sha256:
        rows = {(row["basket"], row["mode"]): row for row in previous.get("rows", []) if row["basket"] in names}

    for study in studies:
        if args.only and study["name"] not in args.only:
            continue
        for mode in args.modes:
            summary = run_research(
                study["tickers"],
                study["start"],
                study["end"],
                mode,
                protocol,
                CASE_STUDY_DIR / study["name"] / mode,
                refresh_data=args.refresh_data,
                command=f"python scripts/run_case_studies.py --only {study['name']} --modes {mode}",
            )
            rows[(study["name"], mode)] = headline_row(study["name"], to_jsonable(summary))

    order = {name: position for position, name in enumerate(names)}
    ordered = sorted(rows.values(), key=lambda row: (order[row["basket"]], MODES.index(row["mode"])))
    save_json({"protocol_sha256": protocol.sha256, "rows": ordered}, summary_path)
    markdown = render_results_markdown(ordered)
    # summary.md lives in results/case_studies/, so its links must be relative to that folder.
    summary_markdown = render_results_markdown(ordered, link_root="results/case_studies")
    # newline="\n" everywhere: re-running must change only the numbers, not every line ending.
    (CASE_STUDY_DIR / "summary.md").write_text(summary_markdown + "\n", encoding="utf-8", newline="\n")
    if not args.no_readme:
        readme = PROJECT_ROOT / "README.md"
        updated = replace_block(readme.read_text(encoding="utf-8"), markdown)
        readme.write_text(updated, encoding="utf-8", newline="\n")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
