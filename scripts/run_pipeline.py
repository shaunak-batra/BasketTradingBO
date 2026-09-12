#!/usr/bin/env python3
"""Run one basket through the walk-forward protocol.

Examples:
    python scripts/run_pipeline.py --tickers KO PEP --start 2010-01-01 --end 2025-01-01
    python scripts/run_pipeline.py --tickers KO PEP --start 2010-01-01 --end 2025-01-01 --mode optimized
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
