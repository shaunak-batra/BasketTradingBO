#!/usr/bin/env python3
"""Measure how often the Johansen trace filter passes baskets that are not cointegrated.

The walk-forward trades a basket only when the trace test rejects rank 0 at 5%. On
independent random walks that should happen 5% of the time. This script measures the
actual rate under the protocol's settings (log prices, ``det_order=0``, ``k_ar_diff=1``,
a 504-bar formation window), with and without drift, and saves the result so the rates
quoted in the README and docs are reproducible.

Example:
    python scripts/johansen_size_simulation.py        # writes results/johansen_size/summary.json
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cointegration.engine import johansen_test  # noqa: E402
from src.utils.config import PROJECT_ROOT  # noqa: E402
from src.utils.io import save_json  # noqa: E402

SCENARIOS = (
    {"name": "2 assets, no drift", "k": 2, "drift": 0.0},
    {"name": "3 assets, no drift", "k": 3, "drift": 0.0},
    {"name": "2 assets, 0.05% daily drift", "k": 2, "drift": 0.0005},
    {"name": "2 assets, 0.2% daily drift", "k": 2, "drift": 0.002},
)


def rejection_rate(k: int, drift: float, n_bars: int, n_sims: int, sigma: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(n_sims):
        log_prices = np.log(100.0) + np.cumsum(drift + rng.standard_normal((n_bars, k)) * sigma, axis=0)
        rejections += johansen_test(log_prices, significance=0.05, det_order=0, k_ar_diff=1).is_cointegrated
    rate = rejections / n_sims
    half_width = 1.96 * math.sqrt(rate * (1 - rate) / n_sims)
    return {"rejections": int(rejections), "rate": rate, "ci_low": rate - half_width, "ci_high": rate + half_width}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Johansen trace-filter false-positive rate on random walks.")
    parser.add_argument("--sims", type=int, default=2000, help="simulations per scenario")
    parser.add_argument("--bars", type=int, default=504, help="formation window length")
    parser.add_argument("--sigma", type=float, default=0.015, help="daily log-return volatility")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "johansen_size" / "summary.json"))
    args = parser.parse_args(argv)

    rows = []
    for position, scenario in enumerate(SCENARIOS):
        result = rejection_rate(scenario["k"], scenario["drift"], args.bars, args.sims, args.sigma, seed=position)
        rows.append({**scenario, **result})
        print(
            f"{scenario['name']:<30} {result['rate']:6.1%}  "
            f"(95% interval {result['ci_low']:.1%} to {result['ci_high']:.1%}, {result['rejections']}/{args.sims})"
        )
    save_json(
        {
            "nominal_level": 0.05,
            "det_order": 0,
            "k_ar_diff": 1,
            "n_bars": args.bars,
            "n_sims": args.sims,
            "sigma": args.sigma,
            "seeds": "scenario index",
            "scenarios": rows,
        },
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
