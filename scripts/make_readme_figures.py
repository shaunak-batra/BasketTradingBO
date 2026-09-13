#!/usr/bin/env python3
"""Regenerate the explanatory figures used in the README.

These use synthetic data and closed-form mathematics only, so they need no price data or network.

    python scripts/make_readme_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import PROJECT_ROOT  # noqa: E402
from src.visualization.explainers import make_all  # noqa: E402


def main() -> int:
    facts, paths = make_all(PROJECT_ROOT / "docs" / "figures")
    for name, path in paths.items():
        print(f"{name}: {path.relative_to(PROJECT_ROOT).as_posix()}")
    for key, value in facts.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
