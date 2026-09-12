"""Load the YAML research protocol and build validated configuration objects."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from src.utils.exceptions import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

T = TypeVar("T")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read a YAML config file. A missing file is an error, not a silent default."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"Config file {config_path} must contain a mapping at the top level")
    return config


def build_dataclass(cls: type[T], values: dict[str, Any] | None, section: str) -> T:
    """Instantiate ``cls`` from a config section, rejecting unknown keys (typos)."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    values = dict(values or {})
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"Unknown keys in '{section}': {unknown}. Allowed keys: {sorted(allowed)}")
    try:
        return cls(**values)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid '{section}' configuration: {exc}") from exc
