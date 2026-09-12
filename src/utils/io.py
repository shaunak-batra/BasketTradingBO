"""Serialisation helpers: strict JSON output and file hashing."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas values to plain JSON types.

    Non-finite floats (NaN, +/-inf) become ``None`` so the output is valid JSON.
    Unknown types raise ``TypeError`` instead of being stringified silently.
    """
    if obj is None or isinstance(obj, str):
        return obj
    if obj is pd.NaT:
        return None
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(value) for value in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def save_json(data: Any, path: str | Path) -> Path:
    """Write ``data`` as indented UTF-8 JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(data), indent=2, allow_nan=False)
    # newline="\n" keeps the bytes identical on every OS, so re-running does not
    # rewrite every line of a committed file.
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path


def load_json(path: str | Path) -> Any:
    """Read a JSON file written by :func:`save_json`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
