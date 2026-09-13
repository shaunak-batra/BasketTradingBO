"""Performance statistics from an equity curve and a trade ledger.

Conventions
-----------
* Daily simple returns of equity; the first bar (no prior equity) is excluded.
* 252 trading days per year. Sharpe and Sortino use a zero risk-free rate, and
  uninvested cash is assumed to earn nothing (conservative for a long/short book).
* Every calendar trading day is included, flat days too, so statistics describe
  the strategy as run, not only the days it held a position.
* Undefined statistics (e.g. Sharpe with zero variance) are NaN, never 0.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252
EULER_MASCHERONI = 0.5772156649015329
_ZERO_VARIANCE = 1e-15


def returns_from_equity(equity: pd.Series) -> pd.Series:
    """Simple returns of an equity curve, excluding the first bar."""
    if len(equity) < 2:
        raise ValueError("equity must contain at least two observations")
    return equity.pct_change().iloc[1:]


def sharpe_ratio(returns: pd.Series | np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sharpe ratio (zero risk-free rate, sample standard deviation)."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return math.nan
    sd = r.std(ddof=1)
    if sd <= _ZERO_VARIANCE:
        return math.nan
    return float(r.mean() / sd * math.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series | np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sortino ratio with a 0% target.

    Downside deviation is ``sqrt(mean(min(r, 0)^2))`` over all periods, not the
    standard deviation of the negative returns alone.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return math.nan
    downside = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2)))
    if downside <= _ZERO_VARIANCE:
        return math.nan
    return float(r.mean() / downside * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Maximum peak-to-trough decline (negative fraction) and the longest underwater spell in bars."""
    values = equity.to_numpy(dtype=float)
    peaks = np.maximum.accumulate(values)
    drawdown = values / peaks - 1.0
    longest = run = 0
    for below in drawdown < 0:
        run = run + 1 if below else 0
        longest = max(longest, run)
    return float(drawdown.min()), int(longest)


def psr_from_moments(
    sharpe_per_period: float, n_obs: int, skewness: float, kurtosis: float, benchmark_per_period: float = 0.0
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012).

    Probability that the true per-period Sharpe exceeds ``benchmark_per_period``
    given ``n_obs`` observations with sample ``skewness`` and raw ``kurtosis``
    (normal = 3). Non-normal returns widen the Sharpe ratio's standard error.
    """
    inputs = (sharpe_per_period, skewness, kurtosis, benchmark_per_period)
    if n_obs < 2 or not all(math.isfinite(x) for x in inputs):
        return math.nan
    variance = 1.0 - skewness * sharpe_per_period + (kurtosis - 1.0) / 4.0 * sharpe_per_period**2
    if variance <= 0:
        return math.nan
    z = (sharpe_per_period - benchmark_per_period) * math.sqrt(n_obs - 1) / math.sqrt(variance)
    return float(stats.norm.cdf(z))


def _sharpe_moments(returns: pd.Series | np.ndarray) -> tuple[float, int, float, float] | None:
    r = np.asarray(returns, dtype=float)
    if len(r) < 4:
        return None
    sd = r.std(ddof=1)
    if sd <= _ZERO_VARIANCE:
        return None
    skewness = float(stats.skew(r, bias=False))
    kurtosis = float(stats.kurtosis(r, fisher=False, bias=False))
    return float(r.mean() / sd), len(r), skewness, kurtosis


def probabilistic_sharpe_ratio(returns: pd.Series | np.ndarray, benchmark_per_period: float = 0.0) -> float:
    """PSR of a return series against a per-period Sharpe benchmark (default 0)."""
    moments = _sharpe_moments(returns)
    if moments is None:
        return math.nan
    sharpe, n_obs, skewness, kurtosis = moments
    return psr_from_moments(sharpe, n_obs, skewness, kurtosis, benchmark_per_period)


def expected_max_sharpe(trial_sharpes_per_period: np.ndarray | list[float], n_trials: int) -> float:
    """Expected maximum Sharpe among ``n_trials`` skill-less trials (Bailey & Lopez de Prado, 2014).

    Uses the cross-sectional standard deviation of the observed trial Sharpe ratios.
    """
    if n_trials < 2:
        return 0.0
    trials = np.asarray(trial_sharpes_per_period, dtype=float)
    trials = trials[np.isfinite(trials)]
    if len(trials) < 2:
        return math.nan
    sd = float(trials.std(ddof=1))
    gamma = EULER_MASCHERONI
    return sd * (
        (1 - gamma) * stats.norm.ppf(1 - 1 / n_trials) + gamma * stats.norm.ppf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray, trial_sharpes_per_period: np.ndarray | list[float], n_trials: int
) -> float:
    """Deflated Sharpe Ratio: PSR against the Sharpe expected from selecting the best of ``n_trials``."""
    benchmark = expected_max_sharpe(trial_sharpes_per_period, n_trials)
    if not math.isfinite(benchmark):
        return math.nan
    return probabilistic_sharpe_ratio(returns, benchmark_per_period=benchmark)


def block_bootstrap_sharpe_ci(
    returns: pd.Series | np.ndarray,
    n_samples: int = 2000,
    block: int = 20,
    confidence: float = 0.95,
    seed: int = 0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> tuple[float, float]:
    """Percentile confidence interval for the annualised Sharpe ratio.

    Uses a moving-block bootstrap so that autocorrelation within ``block`` bars
    (e.g. multi-day holding periods) is preserved in the resamples.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 2 * block or not 0 < confidence < 1:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_samples, n_blocks))
    indices = (starts[:, :, None] + np.arange(block)).reshape(n_samples, -1)[:, :n]
    samples = r[indices]
    means = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=1)
    valid = sds > _ZERO_VARIANCE
    if not valid.any():
        return math.nan, math.nan
    sharpes = means[valid] / sds[valid] * math.sqrt(periods_per_year)
    lower, upper = np.percentile(sharpes, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100])
    return float(lower), float(upper)


def trade_statistics(trades: pd.DataFrame) -> dict[str, float]:
    """Round-trip statistics from closed trades (open positions are excluded).

    Stop exits are counted by kind, because they answer different questions: a z-score stop
    fires when the spread diverges past ``stop_z``, a loss stop when the position loses a set
    fraction of the equity it was sized on, and a time stop when it has been held too long.
    """
    closed = trades.loc[~trades["is_open"].astype(bool)] if len(trades) else trades
    n_trades = int(len(closed))
    if n_trades == 0:
        return {
            "n_trades": 0,
            "win_rate": math.nan,
            "profit_factor": math.nan,
            "avg_trade_return": math.nan,
            "avg_holding_days": math.nan,
            "n_zscore_stops": 0,
            "n_loss_stops": 0,
            "n_time_stops": 0,
        }
    pnl = closed["net_pnl"].to_numpy(dtype=float)
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses > 0:
        profit_factor = float(gains / losses)
    else:
        profit_factor = math.inf if gains > 0 else math.nan
    reasons = closed["exit_reason"]
    return {
        "n_trades": n_trades,
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": profit_factor,
        "avg_trade_return": float(closed["return_on_equity"].mean()),
        "avg_holding_days": float(closed["holding_bars"].mean()),
        "n_zscore_stops": int((reasons == "stop").sum()),
        "n_loss_stops": int((reasons == "stop_loss").sum()),
        "n_time_stops": int((reasons == "time_stop").sum()),
    }


def summarize_performance(
    equity: pd.Series,
    trades: pd.DataFrame,
    gross_exposure: pd.Series,
    traded_notional: pd.Series,
    trading_costs: pd.Series,
    borrow_costs: pd.Series,
    bootstrap_samples: int = 2000,
    bootstrap_block: int = 20,
    seed: int = 0,
) -> dict[str, float]:
    """All headline statistics for one equity curve, in a stable key order."""
    returns = returns_from_equity(equity)
    n_returns = len(returns)
    years = n_returns / TRADING_DAYS_PER_YEAR
    growth = float(equity.iloc[-1] / equity.iloc[0])
    total_return = growth - 1.0
    cagr = growth ** (1 / years) - 1.0 if years > 0 and growth > 0 else math.nan
    sd = float(returns.std(ddof=1)) if n_returns > 1 else math.nan
    drawdown, drawdown_days = max_drawdown(equity)
    moments = _sharpe_moments(returns)
    ci_low, ci_high = block_bootstrap_sharpe_ci(
        returns, n_samples=bootstrap_samples, block=bootstrap_block, seed=seed
    )
    invested = gross_exposure > 0
    mean_equity = float(equity.mean())

    summary: dict[str, float] = {
        "start": equity.index[0],
        "end": equity.index[-1],
        "n_days": n_returns,
        "total_return": total_return,
        "cagr": cagr,
        "ann_volatility": sd * math.sqrt(TRADING_DAYS_PER_YEAR) if math.isfinite(sd) else math.nan,
        "sharpe": sharpe_ratio(returns),
        "sharpe_ci_low": ci_low,
        "sharpe_ci_high": ci_high,
        "psr": probabilistic_sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": drawdown,
        "max_drawdown_days": drawdown_days,
        "calmar": cagr / abs(drawdown) if drawdown < 0 and math.isfinite(cagr) else math.nan,
        "skewness": moments[2] if moments else math.nan,
        "kurtosis": moments[3] if moments else math.nan,
        "time_in_market": float(invested.mean()),
        "avg_gross_exposure_when_invested": float(gross_exposure[invested].mean()) if invested.any() else math.nan,
        "turnover_annual": float(traded_notional.sum() / mean_equity / years) if years > 0 else math.nan,
        "total_trading_cost": float(trading_costs.sum()),
        "total_borrow_cost": float(borrow_costs.sum()),
    }
    summary.update(trade_statistics(trades))
    return summary
