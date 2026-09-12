#!/usr/bin/env python3
"""Run a protocol across a pre-registered universe of pairs.

    python scripts/run_universe.py --config config/config_v3.yaml --universe config/universe_v3.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import load_protocol  # noqa: E402
from src.universe import (  # noqa: E402
    benjamini_hochberg,
    evaluate_pair,
    load_universe_prices,
    pair_prices,
    plot_universe_results,
    portfolio_summary,
    render_markdown,
    summarise,
    universe_pairs,
)
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.exceptions import BasketTradingError  # noqa: E402
from src.utils.io import save_json  # noqa: E402

logger = logging.getLogger("run_universe")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config_v3.yaml"))
    parser.add_argument("--universe", default=str(PROJECT_ROOT / "config" / "universe_v3.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "v3_universe"))
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    protocol = load_protocol(args.config)
    universe = load_config(args.universe)
    settings, families = universe["evaluation"], universe["families"]
    pairs = universe_pairs(families)
    tickers = sorted({ticker for members in families.values() for ticker in members})
    logger.info("%d tickers, %d pairs", len(tickers), len(pairs))

    snapshot_dir = Path(protocol.data.snapshot_dir)
    if not snapshot_dir.is_absolute():
        snapshot_dir = PROJECT_ROOT / snapshot_dir
    snapshot = snapshot_dir / f"{settings['snapshot_name']}_{settings['start']}_{settings['end']}.csv"
    frame, data_sha256 = load_universe_prices(
        tickers, settings["start"], settings["end"], snapshot, refresh=args.refresh_data
    )

    rows, returns_by_pair, frictionless_by_pair, skipped = [], {}, {}, []
    for family, a, b in pairs:
        label = f"{a}/{b}"
        try:
            prices = pair_prices(frame, a, b, settings["min_bars"], protocol.data.max_missing_fraction)
            row, returns, frictionless_returns = evaluate_pair(prices, protocol)
        except BasketTradingError as error:
            logger.warning("%s skipped: %s", label, error)
            skipped.append({"pair": label, "family": family, "reason": str(error)})
            continue
        row.update(pair=label, family=family)
        rows.append(row)
        returns_by_pair[label] = returns
        frictionless_by_pair[label] = frictionless_returns
        logger.info(
            "%-12s folds %2d/%2d  trades %3d  return %7.2f%%  Sharpe %6.2f (no frictions %6.2f)",
            label, row["n_traded"], row["n_folds"], row["n_trades"], row["total_return"] * 100,
            row["sharpe"], row["sharpe_frictionless"],
        )

    if not rows:
        raise BasketTradingError("no pair could be evaluated")

    pairs_frame = pd.DataFrame(rows)
    pairs_frame["discovery"] = benjamini_hochberg(pairs_frame["p_value"].to_numpy(), settings["fdr_level"])
    returns_frame = pd.DataFrame(returns_by_pair).sort_index()
    frictionless_frame = pd.DataFrame(frictionless_by_pair).sort_index()

    summary = {
        "run": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "protocol": Path(args.config).name,
            "protocol_sha256": protocol.sha256,
            "universe": Path(args.universe).name,
            "data_snapshot": snapshot.name,
            "data_sha256": data_sha256,
            "command": "python scripts/run_universe.py " + " ".join(argv or sys.argv[1:]),
        },
        "protocol": protocol.as_dict(),
        "fdr_level": settings["fdr_level"],
        "n_pairs_defined": len(pairs),
        "n_pairs_skipped": len(skipped),
        "skipped": skipped,
        "tested": summarise(pairs_frame),
        "portfolio": portfolio_summary(returns_frame, protocol.reporting),
        "portfolio_frictionless": portfolio_summary(frictionless_frame, protocol.reporting),
        "pairs": pairs_frame.to_dict(orient="records"),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_dir / "summary.json")
    pairs_frame.to_csv(output_dir / "pairs.csv", index=False, lineterminator="\n")
    returns_frame.to_csv(output_dir / "pair_returns.csv", index_label="date", lineterminator="\n")
    plot_universe_results(pairs_frame, returns_frame, frictionless_frame, output_dir / "plots")
    markdown = render_markdown(summary, pairs_frame)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8", newline="\n")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
