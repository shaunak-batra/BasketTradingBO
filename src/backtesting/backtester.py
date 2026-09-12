"""Daily close-to-close backtest of a spread strategy with cash accounting.

Accounting for bar t (close prices ``P_t``, share vector ``q``):

    mark-to-market   pnl_t    = q_{t-1} . (P_t - P_{t-1})
    borrow fee       borrow_t = (b / 252) * sum_i max(-q_{t-1,i}, 0) * P_{t-1,i}
    trading cost     cost_t   = c * sum_i |shares traded_i| * P_{t,i}
    equity           E_t      = E_{t-1} + pnl_t - borrow_t - cost_t

Equity changes only through P&L and costs; buying or shorting shares is a cash
exchange and does not change equity. A position is sized once at entry and held in
constant shares until exit, so there are no hidden daily rebalancing costs. A target
state decided at bar t is filled at the close of bar t + ``execution_lag``.

Two execution-level risk rules are optional and off by default (v2 behaviour):

* ``stop_loss_fraction`` closes a position once its loss against the equity it was
  sized on reaches that fraction. Unlike a z-score stop it cannot be outrun by rising
  volatility.
* ``max_holding_bars`` closes a position that has been held too long.

After either fires, the same direction is not re-entered until the target state
returns to flat (or flips), so a stopped-out position cannot reopen on the next bar.

Sizing is either a fixed ``gross_exposure`` multiple of equity, or, when
``target_volatility`` is set, ``target_volatility / sizing_volatility`` capped at
``max_gross``, where ``sizing_volatility`` is an estimate made before the window
being traded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.strategy.signals import EXIT, STOP
from src.utils.exceptions import BacktestError, ConfigError

TRADING_DAYS_PER_YEAR = 252
STOP_LOSS = "stop_loss"
TIME_STOP = "time_stop"

TRADE_COLUMNS = [
    "direction",
    "signal_date",
    "entry_date",
    "exit_date",
    "exit_reason",
    "holding_bars",
    "entry_equity",
    "entry_gross_notional",
    "gross_pnl",
    "borrow_cost",
    "trading_cost",
    "net_pnl",
    "return_on_equity",
    "is_open",
]


@dataclass(frozen=True)
class BacktestConfig:
    """Execution, sizing and cost assumptions."""

    initial_capital: float = 100_000.0
    gross_exposure: float = 1.0
    target_volatility: float | None = None
    max_gross: float = 1.0
    stop_loss_fraction: float | None = None
    max_holding_bars: int | None = None
    cost_bps: float = 5.0
    borrow_bps_annual: float = 50.0
    execution_lag: int = 1

    def __post_init__(self) -> None:
        _require_finite(self.initial_capital, "initial_capital", minimum=0.0, strict=True)
        _require_finite(self.gross_exposure, "gross_exposure", minimum=0.0, strict=True)
        _require_finite(self.max_gross, "max_gross", minimum=0.0, strict=True)
        _require_finite(self.cost_bps, "cost_bps", minimum=0.0)
        _require_finite(self.borrow_bps_annual, "borrow_bps_annual", minimum=0.0)
        if self.target_volatility is not None:
            _require_finite(self.target_volatility, "target_volatility", minimum=0.0, strict=True)
        if self.stop_loss_fraction is not None:
            _require_finite(self.stop_loss_fraction, "stop_loss_fraction", minimum=0.0, strict=True)
            if self.stop_loss_fraction >= 1:
                raise ConfigError(f"stop_loss_fraction must be < 1, got {self.stop_loss_fraction}")
        if self.max_holding_bars is not None:
            if not _is_int(self.max_holding_bars) or self.max_holding_bars < 1:
                raise ConfigError(f"max_holding_bars must be an integer >= 1, got {self.max_holding_bars!r}")
        if not _is_int(self.execution_lag) or self.execution_lag < 0:
            raise ConfigError(f"execution_lag must be an integer >= 0, got {self.execution_lag!r}")


@dataclass
class BacktestResult:
    """Everything produced by one backtest. All series share the price index."""

    equity: pd.Series
    returns: pd.Series  # simple returns of equity; the first bar is 0 by definition
    pnl: pd.Series  # mark-to-market P&L on shares carried from the previous close
    borrow_costs: pd.Series
    trading_costs: pd.Series
    traded_notional: pd.Series
    positions: pd.DataFrame  # shares held after trading at each close
    gross_exposure: pd.Series  # sum |shares * price| / equity
    net_exposure: pd.Series  # sum shares * price / equity
    trades: pd.DataFrame  # one row per round trip, columns TRADE_COLUMNS
    config: BacktestConfig
    target_gross: float  # gross multiple of equity used at entry


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _require_finite(value: object, name: str, minimum: float, strict: bool = False) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value) or value < minimum or (strict and value == minimum):
        comparison = ">" if strict else ">="
        raise ConfigError(f"{name} must be finite and {comparison} {minimum}, got {value!r}")


def resolve_gross(config: BacktestConfig, sizing_volatility: float | None) -> float:
    """Gross exposure at entry, as a multiple of equity.

    Without ``target_volatility`` this is the fixed ``gross_exposure``. With it, gross is
    ``target_volatility / sizing_volatility`` capped at ``max_gross``, so a quiet spread
    gets a larger position and a violent one a smaller position, for comparable risk.
    """
    if config.target_volatility is None:
        return config.gross_exposure
    if sizing_volatility is None or not math.isfinite(sizing_volatility):
        raise BacktestError("target_volatility is set, so sizing_volatility must be a finite estimate")
    if sizing_volatility <= 0:
        return config.max_gross
    return min(config.max_gross, config.target_volatility / sizing_volatility)


def _validate_inputs(
    prices: pd.DataFrame,
    target_state: pd.Series,
    weights: npt.NDArray[np.float64],
    events: pd.Series | None,
) -> None:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise BacktestError("prices must be a non-empty DataFrame")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise BacktestError("prices index must be strictly increasing")
    values = prices.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise BacktestError("prices must be finite and strictly positive")
    if not target_state.index.equals(prices.index):
        raise BacktestError("target_state must share the prices index")
    states = target_state.to_numpy()
    if pd.isna(states).any() or not np.isin(states, (-1, 0, 1)).all():
        raise BacktestError("target_state values must be -1, 0 or 1")
    if weights.shape != (prices.shape[1],) or not np.all(np.isfinite(weights)) or np.abs(weights).sum() == 0:
        raise BacktestError(f"weights must be a finite, non-zero vector of length {prices.shape[1]}")
    if events is not None and not events.index.equals(prices.index):
        raise BacktestError("events must share the prices index")


def run_backtest(
    prices: pd.DataFrame,
    target_state: pd.Series,
    weights: npt.ArrayLike,
    config: BacktestConfig,
    events: pd.Series | None = None,
    force_flat_at_end: bool = True,
    initial_equity: float | None = None,
    sizing_volatility: float | None = None,
) -> BacktestResult:
    """Simulate trading ``target_state`` on the basket defined by ``weights``.

    Parameters
    ----------
    prices
        Close prices, one column per asset, strictly positive.
    target_state
        Target position (+1 / 0 / -1) decided at each bar's close.
    weights
        Cointegrating vector. Dollar allocation per asset is ``weights / sum(|weights|)``.
    config
        Costs, sizing, risk stops and execution lag.
    events
        Optional signal events ("entry" / "exit" / "stop") used to label exits.
    force_flat_at_end
        Close any open position at the last bar (used at walk-forward fold ends).
    initial_equity
        Starting equity; defaults to ``config.initial_capital``.
    sizing_volatility
        Annualised volatility estimate for the spread, required when
        ``config.target_volatility`` is set. It must come from data before this window.
    """
    w = np.asarray(weights, dtype=float)
    _validate_inputs(prices, target_state, w, events)

    equity = float(config.initial_capital if initial_equity is None else initial_equity)
    if not math.isfinite(equity) or equity <= 0:
        raise BacktestError(f"initial equity must be positive, got {equity}")

    price_matrix = prices.to_numpy(dtype=float)
    states = target_state.to_numpy(dtype=np.int64)
    event_values = events.to_numpy(dtype=object) if events is not None else None
    index = prices.index
    n_bars, n_assets = price_matrix.shape

    allocation = w / np.abs(w).sum()
    target_gross = resolve_gross(config, sizing_volatility)
    cost_rate = config.cost_bps / 1e4
    borrow_rate = config.borrow_bps_annual / 1e4 / TRADING_DAYS_PER_YEAR
    lag = int(config.execution_lag)

    equity_path = np.empty(n_bars)
    pnl_path = np.zeros(n_bars)
    borrow_path = np.zeros(n_bars)
    cost_path = np.zeros(n_bars)
    traded_path = np.zeros(n_bars)
    position_path = np.zeros((n_bars, n_assets))
    gross_path = np.zeros(n_bars)
    net_path = np.zeros(n_bars)

    shares = np.zeros(n_assets)
    current = 0
    blocked_direction = 0  # set after a risk stop, cleared when the signal resets
    open_trade: dict | None = None
    trades: list[dict] = []

    for t in range(n_bars):
        if t > 0 and current != 0:
            pnl = float(shares @ (price_matrix[t] - price_matrix[t - 1]))
            borrow = borrow_rate * float(-(np.minimum(shares, 0.0) @ price_matrix[t - 1]))
            equity += pnl - borrow
            pnl_path[t] = pnl
            borrow_path[t] = borrow
            open_trade["gross_pnl"] += pnl
            open_trade["borrow_cost"] += borrow

        # Execution-level risk rules, checked on this close before any new signal.
        stop_reason = None
        if current != 0 and open_trade is not None:
            if config.stop_loss_fraction is not None:
                loss = equity / open_trade["entry_equity"] - 1.0
                if loss <= -config.stop_loss_fraction:
                    stop_reason = STOP_LOSS
            if stop_reason is None and config.max_holding_bars is not None:
                if t - open_trade["entry_bar"] >= config.max_holding_bars:
                    stop_reason = TIME_STOP

        signal_bar = t - lag
        lagged_state = int(states[signal_bar]) if signal_bar >= 0 else 0
        desired = lagged_state
        forced_close = force_flat_at_end and t == n_bars - 1
        if forced_close or stop_reason is not None:
            desired = 0
        if blocked_direction != 0:
            if desired == blocked_direction:
                desired = 0  # do not re-enter the direction that was stopped out
            else:
                blocked_direction = 0

        if desired != current:
            if equity <= 0:
                raise BacktestError(f"Equity is non-positive ({equity:.2f}) at {index[t]}; simulation is invalid")

            if current != 0:
                notional = float(np.abs(shares) @ price_matrix[t])
                cost = cost_rate * notional
                equity -= cost
                cost_path[t] += cost
                traded_path[t] += notional
                open_trade["trading_cost"] += cost
                if stop_reason is not None:
                    reason = stop_reason
                    blocked_direction = current
                elif forced_close and lagged_state == current:
                    reason = "end_of_window"
                elif event_values is not None and signal_bar >= 0 and event_values[signal_bar] in (EXIT, STOP):
                    reason = str(event_values[signal_bar])
                else:
                    reason = "signal"
                open_trade.update(exit_date=index[t], exit_bar=t, exit_reason=reason, is_open=False)
                trades.append(_finalise_trade(open_trade))
                open_trade = None
                shares = np.zeros(n_assets)
                current = 0

            if desired != 0:
                equity_before_entry = equity
                new_shares = desired * target_gross * equity * allocation / price_matrix[t]
                notional = float(np.abs(new_shares) @ price_matrix[t])
                cost = cost_rate * notional
                equity -= cost
                cost_path[t] += cost
                traded_path[t] += notional
                shares = new_shares
                current = desired
                open_trade = {
                    "direction": desired,
                    "signal_date": index[signal_bar] if signal_bar >= 0 else pd.NaT,
                    "entry_date": index[t],
                    "entry_bar": t,
                    "entry_equity": equity_before_entry,
                    "entry_gross_notional": notional,
                    "gross_pnl": 0.0,
                    "borrow_cost": 0.0,
                    "trading_cost": cost,
                }

        equity_path[t] = equity
        position_path[t] = shares
        gross_path[t] = float(np.abs(shares) @ price_matrix[t]) / equity
        net_path[t] = float(shares @ price_matrix[t]) / equity

    if open_trade is not None:
        open_trade.update(exit_date=pd.NaT, exit_bar=n_bars - 1, exit_reason="open", is_open=True)
        trades.append(_finalise_trade(open_trade))

    equity_series = pd.Series(equity_path, index=index, name="equity")
    returns = equity_series.pct_change().fillna(0.0).rename("returns")
    trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)

    return BacktestResult(
        equity=equity_series,
        returns=returns,
        pnl=pd.Series(pnl_path, index=index, name="pnl"),
        borrow_costs=pd.Series(borrow_path, index=index, name="borrow_costs"),
        trading_costs=pd.Series(cost_path, index=index, name="trading_costs"),
        traded_notional=pd.Series(traded_path, index=index, name="traded_notional"),
        positions=pd.DataFrame(position_path, index=index, columns=prices.columns),
        gross_exposure=pd.Series(gross_path, index=index, name="gross_exposure"),
        net_exposure=pd.Series(net_path, index=index, name="net_exposure"),
        trades=trade_frame,
        config=config,
        target_gross=target_gross,
    )


def _finalise_trade(trade: dict) -> dict:
    entry_bar = trade.pop("entry_bar")
    exit_bar = trade.pop("exit_bar")
    trade["holding_bars"] = exit_bar - entry_bar
    trade["net_pnl"] = trade["gross_pnl"] - trade["borrow_cost"] - trade["trading_cost"]
    trade["return_on_equity"] = trade["net_pnl"] / trade["entry_equity"]
    return trade
