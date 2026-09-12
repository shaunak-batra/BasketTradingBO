"""Walk-forward evaluation: estimate on a formation window, then trade the next window blind.

For every fold:

1. Run the Johansen test on formation-window log prices; the leading eigenvector
   gives the basket weights.
2. Estimate the half-life and the volatility of the formation-window spread.
3. Apply the pre-registered filters, in order: cointegration, net exposure (is the
   basket actually hedged), half-life. A fold that fails any of them is not traded.
4. Choose signal parameters: either fixed, or tuned by Bayesian optimisation on
   the formation window only.
5. Trade the following ``trading_days`` bars with weights, parameters and position
   size frozen. The z-score warm-up uses the last ``lookback`` bars of the formation
   window, which are already in the past.
6. Close any open position at the end of the trading window.

Trading windows do not overlap, and equity carries from one fold to the next, so
the stitched equity curve is a single out-of-sample track record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.backtesting.backtester import TRADE_COLUMNS, BacktestConfig, BacktestResult, run_backtest
from src.backtesting.metrics import (
    TRADING_DAYS_PER_YEAR,
    deflated_sharpe_ratio,
    returns_from_equity,
    sharpe_ratio,
    trade_statistics,
)
from src.cointegration.engine import CRITICAL_VALUE_COLUMN, JohansenResult, johansen_test, net_exposure
from src.cointegration.spread import compute_spread, half_life, rolling_zscore
from src.data.market_data import validate_prices
from src.optimization.optimizer import maximize
from src.strategy.signals import SignalParams, generate_signals
from src.utils.exceptions import ConfigError, DataError

SEARCH_KEYS = ("entry_z", "exit_fraction", "stop_offset", "lookback")
MIN_IN_SAMPLE_TRADING_BARS = 20


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


@dataclass(frozen=True)
class WalkForwardConfig:
    formation_days: int = 504
    trading_days: int = 126
    significance: float = 0.05
    det_order: int = 0
    k_ar_diff: int = 1
    require_cointegration: bool = True
    max_half_life_days: float | None = 126.0
    max_net_exposure: float | None = None  # |sum(w)| / sum(|w|); None disables the check

    def __post_init__(self) -> None:
        if not _is_int(self.formation_days) or self.formation_days < 60:
            raise ConfigError(f"formation_days must be an integer >= 60, got {self.formation_days!r}")
        if not _is_int(self.trading_days) or self.trading_days < 2:
            raise ConfigError(f"trading_days must be an integer >= 2, got {self.trading_days!r}")
        if self.significance not in CRITICAL_VALUE_COLUMN:
            raise ConfigError(f"significance must be one of {sorted(CRITICAL_VALUE_COLUMN)}")
        if not isinstance(self.require_cointegration, bool):
            raise ConfigError("require_cointegration must be true or false")
        if self.max_half_life_days is not None and not self.max_half_life_days > 0:
            raise ConfigError("max_half_life_days must be positive or null")
        if self.max_net_exposure is not None and not 0 < self.max_net_exposure <= 1:
            raise ConfigError("max_net_exposure must be in (0, 1] or null")


@dataclass(frozen=True)
class OptimizationConfig:
    space: dict
    n_calls: int = 40
    n_initial_points: int = 12
    min_trades: int = 3
    random_state: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.space, dict) or set(self.space) != set(SEARCH_KEYS):
            raise ConfigError(f"optimization.space must define exactly {list(SEARCH_KEYS)}")
        space = {}
        for key in SEARCH_KEYS:
            bounds = tuple(self.space[key])
            if len(bounds) != 2 or not bounds[0] < bounds[1]:
                raise ConfigError(f"optimization.space.{key} must be [low, high] with low < high")
            space[key] = bounds
        if not (_is_int(space["lookback"][0]) and _is_int(space["lookback"][1]) and space["lookback"][0] >= 2):
            raise ConfigError("optimization.space.lookback bounds must be integers >= 2")
        if not (0 <= space["exit_fraction"][0] and space["exit_fraction"][1] < 1):
            raise ConfigError("optimization.space.exit_fraction must lie in [0, 1)")
        if space["entry_z"][0] <= 0 or space["stop_offset"][0] <= 0:
            raise ConfigError("entry_z and stop_offset lower bounds must be positive")
        object.__setattr__(self, "space", space)
        if not (_is_int(self.n_calls) and _is_int(self.n_initial_points)):
            raise ConfigError("n_calls and n_initial_points must be integers")
        if not 1 <= self.n_initial_points <= self.n_calls:
            raise ConfigError("need 1 <= n_initial_points <= n_calls")
        if not _is_int(self.min_trades) or self.min_trades < 0:
            raise ConfigError("min_trades must be an integer >= 0")
        if not _is_int(self.random_state):
            raise ConfigError("random_state must be an integer")


@dataclass
class FoldResult:
    index: int
    bars: tuple[int, int, int, int]  # formation [start, end), trading [start, end) as positions
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp
    johansen: JohansenResult
    half_life: float
    net_exposure: float
    sizing_volatility: float
    skip_reason: str | None
    params: SignalParams | None
    in_sample: dict | None
    zscore: pd.Series
    signals: pd.DataFrame | None
    backtest: BacktestResult

    @property
    def traded(self) -> bool:
        return self.skip_reason is None

    def summary(self) -> dict:
        equity = self.backtest.equity
        tickers = list(self.backtest.positions.columns)
        in_sample = self.in_sample or {}
        params = self.params.as_dict() if self.params else {}
        return {
            "fold": self.index,
            "formation_start": self.formation_start,
            "formation_end": self.formation_end,
            "trading_start": self.trading_start,
            "trading_end": self.trading_end,
            "trace_stat": float(self.johansen.trace_stats[0]),
            "trace_critical": float(self.johansen.trace_critical[0]),
            "cointegrated": self.johansen.is_cointegrated,
            "half_life_days": self.half_life,
            "net_exposure": self.net_exposure,
            "sizing_volatility": self.sizing_volatility,
            "target_gross": self.backtest.target_gross,
            "traded": self.traded,
            "skip_reason": self.skip_reason or "",
            **{f"weight_{ticker}": float(w) for ticker, w in zip(tickers, self.johansen.weights)},
            "entry_z": params.get("entry_z", math.nan),
            "exit_z": params.get("exit_z", math.nan),
            "stop_z": params.get("stop_z", math.nan),
            "lookback": params.get("lookback", math.nan),
            "n_trials": in_sample.get("n_trials", 0),
            "is_sharpe": in_sample.get("sharpe", math.nan),
            "is_trades": in_sample.get("n_trades", 0),
            "is_deflated_sharpe": in_sample.get("deflated_sharpe", math.nan),
            "oos_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
            "oos_sharpe": sharpe_ratio(returns_from_equity(equity)),
            "oos_trades": trade_statistics(self.backtest.trades)["n_trades"],
        }


@dataclass
class WalkForwardResult:
    mode: str  # "fixed" or "optimized"
    folds: list[FoldResult]
    equity: pd.Series
    zscore: pd.Series
    trades: pd.DataFrame
    gross_exposure: pd.Series
    net_exposure: pd.Series
    traded_notional: pd.Series
    trading_costs: pd.Series
    borrow_costs: pd.Series
    config: WalkForwardConfig
    backtest_config: BacktestConfig

    def fold_table(self) -> pd.DataFrame:
        return pd.DataFrame([fold.summary() for fold in self.folds])


def fold_windows(n_bars: int, formation_days: int, trading_days: int) -> list[tuple[int, int, int, int]]:
    """Positions ``(formation_start, formation_end, trading_start, trading_end)``, end-exclusive.

    Each trading window starts right after its formation window, consecutive trading
    windows abut without overlap, and the last one may be shorter.
    """
    windows = []
    start = 0
    while start + formation_days < n_bars:
        trading_start = start + formation_days
        trading_end = min(trading_start + trading_days, n_bars)
        if trading_end - trading_start < 2:
            break
        windows.append((start, trading_start, trading_start, trading_end))
        start += trading_days
    if not windows:
        raise DataError(
            f"{n_bars} bars is not enough for one fold (formation {formation_days} + at least 2 trading bars)"
        )
    return windows


def spread_volatility(spread: pd.Series) -> float:
    """Annualised volatility of daily spread changes, used to size positions."""
    changes = spread.diff().dropna()
    if len(changes) < 2:
        return math.nan
    return float(changes.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _window_signals(
    prices: pd.DataFrame, weights: np.ndarray, params: SignalParams, start: int, end: int, floor: int
) -> tuple[pd.Series, pd.DataFrame]:
    """Z-score and signals for bars [start, end), warming up on bars no earlier than ``floor``."""
    warm_start = max(floor, start - params.lookback + 1)
    spread = compute_spread(prices.iloc[warm_start:end], weights)
    zscore = rolling_zscore(spread, params.lookback).iloc[start - warm_start :]
    return zscore, generate_signals(zscore, params)


def _in_sample_backtest(
    prices: pd.DataFrame,
    weights: np.ndarray,
    params: SignalParams,
    start: int,
    end: int,
    config: BacktestConfig,
    sizing_volatility: float,
) -> BacktestResult:
    _, signals = _window_signals(prices, weights, params, start, end, floor=start)
    return run_backtest(
        prices.iloc[start:end],
        signals["state"],
        weights,
        config,
        events=signals["event"],
        force_flat_at_end=True,
        sizing_volatility=sizing_volatility,
    )


def _in_sample_summary(result: BacktestResult, valid_trial_sharpes: list[float], n_trials: int) -> dict:
    returns = returns_from_equity(result.equity)
    return {
        "sharpe": sharpe_ratio(returns),
        "total_return": float(result.equity.iloc[-1] / result.equity.iloc[0] - 1.0),
        "n_trades": trade_statistics(result.trades)["n_trades"],
        "n_trials": n_trials,
        "n_valid_trials": len(valid_trial_sharpes),
        "deflated_sharpe": deflated_sharpe_ratio(returns, valid_trial_sharpes, n_trials),
    }


def _tune_parameters(
    prices: pd.DataFrame,
    weights: np.ndarray,
    start: int,
    end: int,
    backtest_config: BacktestConfig,
    optimization: OptimizationConfig,
    sizing_volatility: float,
) -> tuple[SignalParams | None, dict]:
    """Bayesian optimisation of signal parameters on the formation window only.

    A parameter set with fewer than ``min_trades`` in-sample round trips scores 0, the
    Sharpe ratio of not trading. If no trial beats that baseline, ``None`` is returned
    and the fold is not traded.
    """
    valid_flags: list[bool] = []
    sharpes: list[float] = []

    def objective(point: dict) -> float:
        params = SignalParams.from_search_space(**point)
        result = _in_sample_backtest(prices, weights, params, start, end, backtest_config, sizing_volatility)
        n_trades = trade_statistics(result.trades)["n_trades"]
        sharpe = sharpe_ratio(returns_from_equity(result.equity))
        valid = n_trades >= optimization.min_trades and math.isfinite(sharpe)
        valid_flags.append(valid)
        sharpes.append(sharpe)
        return sharpe if valid else 0.0

    outcome = maximize(
        objective,
        optimization.space,
        n_calls=optimization.n_calls,
        n_initial_points=optimization.n_initial_points,
        random_state=optimization.random_state,
    )
    valid_trial_sharpes = [s / math.sqrt(TRADING_DAYS_PER_YEAR) for s, ok in zip(sharpes, valid_flags) if ok]
    best_index = max(range(outcome.n_trials), key=lambda i: outcome.trials[i].score)
    if not valid_flags[best_index] or outcome.trials[best_index].score <= 0:
        return None, {"n_trials": outcome.n_trials, "n_valid_trials": len(valid_trial_sharpes)}

    params = SignalParams.from_search_space(**outcome.trials[best_index].params)
    best = _in_sample_backtest(prices, weights, params, start, end, backtest_config, sizing_volatility)
    return params, _in_sample_summary(best, valid_trial_sharpes, outcome.n_trials)


def _flat_backtest(
    prices: pd.DataFrame, weights: np.ndarray, config: BacktestConfig, equity: float, sizing_volatility: float
) -> BacktestResult:
    flat = pd.Series(0, index=prices.index, dtype=np.int64)
    return run_backtest(
        prices, flat, weights, config, force_flat_at_end=True, initial_equity=equity, sizing_volatility=sizing_volatility
    )


def run_walk_forward(
    prices: pd.DataFrame,
    config: WalkForwardConfig,
    backtest_config: BacktestConfig,
    params: SignalParams | None = None,
    optimization: OptimizationConfig | None = None,
) -> WalkForwardResult:
    """Run the walk-forward protocol with fixed ``params`` or with ``optimization``."""
    if (params is None) == (optimization is None):
        raise ConfigError("provide exactly one of params (fixed mode) or optimization (optimized mode)")
    validate_prices(prices)
    max_lookback = params.lookback if params is not None else int(optimization.space["lookback"][1])
    if max_lookback > config.formation_days - MIN_IN_SAMPLE_TRADING_BARS:
        raise ConfigError(
            f"lookback up to {max_lookback} leaves fewer than {MIN_IN_SAMPLE_TRADING_BARS} tradable bars "
            f"in a {config.formation_days}-bar formation window"
        )

    log_prices = np.log(prices)
    index = prices.index
    equity = float(backtest_config.initial_capital)
    folds: list[FoldResult] = []

    for fold_number, bars in enumerate(fold_windows(len(prices), config.formation_days, config.trading_days)):
        formation_start, formation_end, trading_start, trading_end = bars
        johansen = johansen_test(
            log_prices.iloc[formation_start:formation_end],
            significance=config.significance,
            det_order=config.det_order,
            k_ar_diff=config.k_ar_diff,
        )
        weights = johansen.weights
        formation_spread = compute_spread(prices.iloc[formation_start:formation_end], weights)
        fold_half_life = half_life(formation_spread)
        fold_net_exposure = net_exposure(weights)
        sizing_vol = spread_volatility(formation_spread)

        skip_reason = None
        if config.require_cointegration and not johansen.is_cointegrated:
            skip_reason = (
                f"not cointegrated (trace {johansen.trace_stats[0]:.2f} <= "
                f"critical {johansen.trace_critical[0]:.2f})"
            )
        elif config.max_net_exposure is not None and fold_net_exposure > config.max_net_exposure:
            skip_reason = (
                f"net exposure {fold_net_exposure:.2f} exceeds {config.max_net_exposure:.2f} (basket is not hedged)"
            )
        elif config.max_half_life_days is not None and not (
            math.isfinite(fold_half_life) and fold_half_life <= config.max_half_life_days
        ):
            skip_reason = f"half-life {fold_half_life:.1f} bars exceeds {config.max_half_life_days:g}"

        fold_params, in_sample = None, None
        if skip_reason is None:
            if optimization is not None:
                fold_params, in_sample = _tune_parameters(
                    prices, weights, formation_start, formation_end, backtest_config, optimization, sizing_vol
                )
                if fold_params is None:
                    skip_reason = "no in-sample parameter set beat the no-trade baseline"
            else:
                fold_params = params
                result = _in_sample_backtest(
                    prices, weights, params, formation_start, formation_end, backtest_config, sizing_vol
                )
                sharpe_pp = sharpe_ratio(returns_from_equity(result.equity)) / math.sqrt(TRADING_DAYS_PER_YEAR)
                in_sample = _in_sample_summary(result, [sharpe_pp], n_trials=1)

        trade_prices = prices.iloc[trading_start:trading_end]
        if skip_reason is None:
            zscore, signals = _window_signals(
                prices, weights, fold_params, trading_start, trading_end, floor=formation_start
            )
            backtest = run_backtest(
                trade_prices,
                signals["state"],
                weights,
                backtest_config,
                events=signals["event"],
                force_flat_at_end=True,
                initial_equity=equity,
                sizing_volatility=sizing_vol,
            )
        else:
            zscore = pd.Series(np.nan, index=trade_prices.index, name="zscore")
            signals = None
            backtest = _flat_backtest(trade_prices, weights, backtest_config, equity, sizing_vol)

        equity = float(backtest.equity.iloc[-1])
        folds.append(
            FoldResult(
                index=fold_number,
                bars=bars,
                formation_start=index[formation_start],
                formation_end=index[formation_end - 1],
                trading_start=index[trading_start],
                trading_end=index[trading_end - 1],
                johansen=johansen,
                half_life=fold_half_life,
                net_exposure=fold_net_exposure,
                sizing_volatility=sizing_vol,
                skip_reason=skip_reason,
                params=fold_params if skip_reason is None else None,
                in_sample=in_sample,
                zscore=zscore,
                signals=signals,
                backtest=backtest,
            )
        )

    mode = "optimized" if optimization is not None else "fixed"
    return _assemble(mode, folds, config, backtest_config)


def replay_with_costs(
    prices: pd.DataFrame, result: WalkForwardResult, backtest_config: BacktestConfig
) -> WalkForwardResult:
    """Re-run every fold's out-of-sample trading with different execution assumptions.

    Fold decisions (weights, parameters, signals, skipped folds) are frozen, so only
    the effect of the cost model is measured.
    """
    equity = float(backtest_config.initial_capital)
    folds = []
    for fold in result.folds:
        _, _, trading_start, trading_end = fold.bars
        trade_prices = prices.iloc[trading_start:trading_end]
        if not trade_prices.index.equals(fold.backtest.equity.index):
            raise DataError("prices do not match the walk-forward result being replayed")
        if fold.traded:
            backtest = run_backtest(
                trade_prices,
                fold.signals["state"],
                fold.johansen.weights,
                backtest_config,
                events=fold.signals["event"],
                force_flat_at_end=True,
                initial_equity=equity,
                sizing_volatility=fold.sizing_volatility,
            )
        else:
            backtest = _flat_backtest(
                trade_prices, fold.johansen.weights, backtest_config, equity, fold.sizing_volatility
            )
        equity = float(backtest.equity.iloc[-1])
        folds.append(replace(fold, backtest=backtest))
    return _assemble(result.mode, folds, result.config, backtest_config)


def _assemble(
    mode: str, folds: list[FoldResult], config: WalkForwardConfig, backtest_config: BacktestConfig
) -> WalkForwardResult:
    def stitch(attribute: str) -> pd.Series:
        return pd.concat([getattr(fold.backtest, attribute) for fold in folds])

    trade_frames = [fold.backtest.trades.assign(fold=fold.index) for fold in folds if len(fold.backtest.trades)]
    trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(columns=[*TRADE_COLUMNS, "fold"])
    )
    return WalkForwardResult(
        mode=mode,
        folds=folds,
        equity=stitch("equity"),
        zscore=pd.concat([fold.zscore for fold in folds]).rename("zscore"),
        trades=trades,
        gross_exposure=stitch("gross_exposure"),
        net_exposure=stitch("net_exposure"),
        traded_notional=stitch("traded_notional"),
        trading_costs=stitch("trading_costs"),
        borrow_costs=stitch("borrow_costs"),
        config=config,
        backtest_config=backtest_config,
    )
