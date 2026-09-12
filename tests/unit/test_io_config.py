"""Serialisation and configuration-loading tests."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtester import BacktestConfig
from src.strategy.signals import SignalParams
from src.utils.config import build_dataclass, load_config
from src.utils.exceptions import ConfigError
from src.utils.io import load_json, save_json, sha256_file, to_jsonable


def test_to_jsonable_converts_numpy_pandas_and_non_finite_values():
    data = {
        "int": np.int64(3),
        "float": np.float32(1.5),
        "nan": float("nan"),
        "inf": np.inf,
        "timestamp": pd.Timestamp("2020-01-02"),
        "array": np.array([1.0, np.nan]),
        "bool": np.bool_(True),
        "nat": pd.NaT,
        "tuple": (1, 2),
    }
    assert to_jsonable(data) == {
        "int": 3,
        "float": 1.5,
        "nan": None,
        "inf": None,
        "timestamp": "2020-01-02T00:00:00",
        "array": [1.0, None],
        "bool": True,
        "nat": None,
        "tuple": [1, 2],
    }


def test_unknown_objects_are_not_silently_stringified():
    with pytest.raises(TypeError):
        to_jsonable({"value": object()})


def test_save_json_writes_strict_json(tmp_path):
    path = save_json({"sharpe": float("nan"), "n": np.int64(2)}, tmp_path / "out" / "result.json")

    def reject(constant):
        raise ValueError(f"non-standard JSON constant {constant}")

    assert json.loads(path.read_text(encoding="utf-8"), parse_constant=reject) == {"sharpe": None, "n": 2}
    assert load_json(path) == {"sharpe": None, "n": 2}


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc" * 1000)
    assert sha256_file(path) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_missing_config_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_config_must_be_a_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_unknown_config_keys_are_rejected():
    with pytest.raises(ConfigError, match="costbps"):
        build_dataclass(BacktestConfig, {"cost_bps": 5.0, "costbps": 1.0}, "backtest")


def test_invalid_config_values_are_reported_as_config_errors():
    with pytest.raises(ConfigError):
        build_dataclass(SignalParams, {"entry_z": "two"}, "strategy")
