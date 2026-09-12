"""Z-score state machine that turns a spread z-score into target positions.

States are +1 (long the spread), -1 (short the spread) and 0 (flat). The state at
bar t uses only information available at the close of bar t. The backtester fills
it ``execution_lag`` bars later, so a signal never trades on the price that
produced it.

Rules (all thresholds inclusive, in z-score units):

    flat  -> long    when -stop_z < z <= -entry_z
    flat  -> short   when  entry_z <= z < stop_z
    long  -> flat    when z >= -exit_z   (spread reverted)   or  z <= -stop_z (stop-loss)
    short -> flat    when z <=  exit_z   (spread reverted)   or  z >=  stop_z (stop-loss)

After a stop-loss, or when |z| is already beyond ``stop_z`` while flat, the machine
is disarmed until |z| < entry_z. That prevents re-entering a spread that is still
diverging on the very next bar. A NaN z-score (warm-up or zero variance) never
opens a position and leaves an open position unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.exceptions import ConfigError

ENTRY = "entry"
EXIT = "exit"
STOP = "stop"


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


@dataclass(frozen=True)
class SignalParams:
    """Signal thresholds (z units) and the rolling z-score window (bars)."""

    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    lookback: int = 60

    def __post_init__(self) -> None:
        for name in ("entry_z", "exit_z", "stop_z"):
            value = getattr(self, name)
            if not _is_real_number(value) or not math.isfinite(value):
                raise ConfigError(f"{name} must be a finite number, got {value!r}")
        if not isinstance(self.lookback, (int, np.integer)) or isinstance(self.lookback, (bool, np.bool_)):
            raise ConfigError(f"lookback must be an integer, got {self.lookback!r}")
        if self.lookback < 2:
            raise ConfigError(f"lookback must be >= 2, got {self.lookback}")
        if self.entry_z <= 0:
            raise ConfigError(f"entry_z must be > 0, got {self.entry_z}")
        if not 0 <= self.exit_z < self.entry_z:
            raise ConfigError(f"exit_z must satisfy 0 <= exit_z < entry_z, got exit_z={self.exit_z}, entry_z={self.entry_z}")
        if self.stop_z <= self.entry_z:
            raise ConfigError(f"stop_z must be > entry_z, got stop_z={self.stop_z}, entry_z={self.entry_z}")

    @classmethod
    def from_search_space(
        cls, entry_z: float, exit_fraction: float, stop_offset: float, lookback: int
    ) -> "SignalParams":
        """Map the optimizer's unconstrained box to valid thresholds.

        ``exit_z = exit_fraction * entry_z`` and ``stop_z = entry_z + stop_offset``, so
        every point of the search box satisfies exit < entry < stop by construction
        and no evaluation needs a penalty value.
        """
        entry = float(entry_z)
        return cls(
            entry_z=entry,
            exit_z=float(exit_fraction) * entry,
            stop_z=entry + float(stop_offset),
            lookback=int(lookback),
        )

    def as_dict(self) -> dict[str, float]:
        return {"entry_z": self.entry_z, "exit_z": self.exit_z, "stop_z": self.stop_z, "lookback": int(self.lookback)}


def generate_signals(zscore: pd.Series, params: SignalParams) -> pd.DataFrame:
    """Run the state machine over ``zscore``.

    Returns a DataFrame indexed like ``zscore`` with columns ``state`` (int8 target
    position decided at each bar's close) and ``event`` ("", "entry", "exit", "stop").
    """
    values = zscore.to_numpy(dtype=float)
    n = len(values)
    state = np.zeros(n, dtype=np.int8)
    event = np.full(n, "", dtype=object)

    entry, exit_, stop = params.entry_z, params.exit_z, params.stop_z
    position = 0
    armed = True

    for t in range(n):
        z = values[t]
        if math.isnan(z):
            state[t] = position
            continue

        if position == 0:
            if not armed:
                if abs(z) < entry:
                    armed = True
            elif -stop < z <= -entry:
                position = 1
                event[t] = ENTRY
            elif entry <= z < stop:
                position = -1
                event[t] = ENTRY
            elif abs(z) >= stop:
                armed = False
        elif position == 1:
            if z >= -exit_:
                position = 0
                event[t] = EXIT
            elif z <= -stop:
                position = 0
                event[t] = STOP
                armed = False
        else:
            if z <= exit_:
                position = 0
                event[t] = EXIT
            elif z >= stop:
                position = 0
                event[t] = STOP
                armed = False

        state[t] = position

    return pd.DataFrame({"state": state, "event": event}, index=zscore.index)
