# Statistical Arbitrage Basket Trading with Bayesian Optimization

A production-grade quantitative trading system that exploits mean-reverting relationships in cointegrated asset baskets. This framework combines rigorous econometric testing (Johansen cointegration), Ornstein-Uhlenbeck process modeling, and hyperparameter optimization via Gaussian Process-based Bayesian optimization.

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
  - [Cointegration Theory](#cointegration-theory)
  - [Spread Construction and Ornstein-Uhlenbeck Process](#spread-construction-and-ornstein-uhlenbeck-process)
  - [Signal Generation via Z-Score](#signal-generation-via-z-score)
  - [Bayesian Optimization](#bayesian-optimization)
  - [Risk Management](#risk-management)
- [Implementation Details](#implementation-details)
- [Case Study: TSLA-NFLX-PLTR Basket](#case-study-tsla-nflx-pltr-basket)
- [Installation](#installation)
- [Usage](#usage)
- [Further Implementation Ideas](#further-implementation-ideas)
- [References and Resources](#references-and-resources)

---

## Overview

This system implements a mean-reversion statistical arbitrage strategy on cointegrated baskets of equities. The core workflow involves:

1. **Cointegration Testing**: Johansen trace test to identify long-run equilibrium relationships
2. **Spread Construction**: Weighted linear combination of log prices using eigenvectors
3. **Mean Reversion Modeling**: Ornstein-Uhlenbeck process characterization (half-life, Hurst exponent)
4. **Signal Generation**: Threshold-based state machine with z-score normalization
5. **Hyperparameter Optimization**: Bayesian optimization (Gaussian Process + Expected Improvement)
6. **Risk Management**: VaR/CVaR analysis using historical, parametric, and Cornish-Fisher methods
7. **Backtesting**: Transaction cost modeling with slippage and commission

The system is built with production considerations: caching (HDF5/Parquet), structured logging, comprehensive unit tests, and modular architecture.

---

## Mathematical Foundation

### Cointegration Theory

**Definition**: Two or more non-stationary time series are cointegrated if a linear combination of them is stationary.

For price series **P** = [P₁, P₂, ..., Pₙ], we seek a cointegrating vector **β** such that:

```
S_t = β₁·log(P₁,t) + β₂·log(P₂,t) + ... + βₙ·log(Pₙ,t)
```

where S_t is stationary (mean-reverting).

**Johansen Test**:

The Johansen procedure tests the rank of the cointegrating matrix using a Vector Error Correction Model (VECM):

```
ΔX_t = Π·X_{t-1} + Γ₁·ΔX_{t-1} + ... + Γ_{k-1}·ΔX_{t-k+1} + ε_t
```

where Π = α·β' (α = adjustment coefficients, β = cointegrating vectors).

The trace statistic tests H₀: rank(Π) ≤ r vs H₁: rank(Π) > r:

```
λ_trace(r) = -T · Σ_{i=r+1}^n ln(1 - λ_i)
```

where λ_i are eigenvalues of Π, sorted in descending order.

**Implementation**:

```python
# From src/cointegration/engine.py
def test_cointegration(self, prices: pd.DataFrame) -> CointegrationResult:
    # Convert deterministic term to integer
    det_order_map = {"nc": -1, "c": 0, "ct": 1, "ctt": 2}
    det_order = det_order_map.get(self.deterministic_term, 0)

    # Run Johansen test
    result = coint_johansen(prices, det_order=det_order, k_ar_diff=1)

    # Extract trace statistics and critical values
    test_stats = result.lr1  # Trace statistic
    critical_vals = result.cvt  # Critical values for trace

    # Determine cointegrating rank
    sig_level_idx = {0.10: 0, 0.05: 1, 0.01: 2}.get(self.significance_level, 1)

    cointegrating_rank = 0
    for i in range(len(test_stats)):
        if test_stats[i] > critical_vals[i, sig_level_idx]:
            cointegrating_rank = i + 1
        else:
            break

    # Eigenvectors represent cointegrating relationships
    eigenvectors = result.evec
    eigenvalues = result.eig
```

The first eigenvector (corresponding to the largest eigenvalue) represents the strongest cointegrating relationship and is used as the basket weights.

---

### Spread Construction and Ornstein-Uhlenbeck Process

Once we have the cointegrating vector **β**, the spread is:

```
S_t = β' · log(P_t)
```

We model the spread as an Ornstein-Uhlenbeck (OU) process:

```
dS_t = θ(μ - S_t)dt + σ·dW_t
```

where:
- θ > 0: mean reversion speed
- μ: long-term mean
- σ: volatility
- W_t: Wiener process

**Half-Life Estimation**:

The half-life τ represents the expected time for the spread to revert halfway to its mean:

```
τ = -ln(2) / θ
```

To estimate θ, we run OLS regression:

```
ΔS_t = α + θ·S_{t-1} + ε_t
```

**Implementation**:

```python
# From src/cointegration/spread.py
def calculate_half_life(self, spread: pd.Series) -> float:
    # Calculate lagged spread and differences
    spread_lag = spread.shift(1)
    spread_diff = spread.diff()

    # Drop NaN values
    df = pd.DataFrame({
        'spread_lag': spread_lag,
        'spread_diff': spread_diff
    }).dropna()

    # OLS regression: Δspread = α + θ*spread_lag
    model = LinearRegression()
    model.fit(df[['spread_lag']], df['spread_diff'])

    theta = model.coef_[0]

    if theta >= 0:
        raise ValueError(f"No mean reversion detected: θ = {theta:.4f} ≥ 0")

    half_life = -np.log(2) / theta
    return half_life
```

**Hurst Exponent**:

The Hurst exponent H characterizes the time series behavior:
- H < 0.5: Mean-reverting (anti-persistent)
- H = 0.5: Random walk (Brownian motion)
- H > 0.5: Trending (persistent)

```python
# From src/cointegration/spread.py
def calculate_hurst_exponent(self, spread: pd.Series, max_lag: int = 100) -> float:
    lags = range(2, min(max_lag, len(spread) // 2))
    tau = []

    for lag in lags:
        # Calculate standard deviation of differences
        std = np.std([spread[i] - spread[i - lag] for i in range(lag, len(spread))])
        tau.append(std)

    # Linear regression of log(tau) on log(lags)
    # tau ∝ lag^H  =>  log(tau) = H·log(lag) + c
    model = LinearRegression()
    model.fit(np.log(list(lags)).reshape(-1, 1), np.log(tau))

    hurst = model.coef_[0]
    return hurst
```

---

### Signal Generation via Z-Score

The spread is normalized into a z-score using a rolling window:

```
Z_t = (S_t - μ_{window}) / σ_{window}
```

where μ_{window} and σ_{window} are the rolling mean and standard deviation.

**Trading Logic** (State Machine):

```
States: IDLE (0), LONG (1), SHORT (-1)

Transitions:
  IDLE → LONG:   Z_t < -θ_entry    (spread oversold, expect reversion up)
  IDLE → SHORT:  Z_t > +θ_entry    (spread overbought, expect reversion down)

  LONG → IDLE:   Z_t > -θ_exit  OR  Z_t < -θ_stop
  SHORT → IDLE:  Z_t < +θ_exit  OR  Z_t > +θ_stop
```

**Implementation**:

```python
# From src/strategy/signals.py
def generate_signals(self, zscore: pd.Series) -> pd.Series:
    signals = pd.Series(0, index=zscore.index, name='signal')
    position = 0  # Current position state

    for i in range(len(zscore)):
        z = zscore.iloc[i]

        if pd.isna(z):
            signals.iloc[i] = position
            continue

        if position == 0:  # No position (IDLE)
            if z < -self.entry_threshold:
                position = 1  # Enter long
            elif z > self.entry_threshold:
                position = -1  # Enter short

        elif position == 1:  # Long position
            # Exit if crosses exit threshold or hits stop loss
            if z > -self.exit_threshold or z < -self.stop_loss:
                position = 0  # Exit long

        elif position == -1:  # Short position
            # Exit if crosses exit threshold or hits stop loss
            if z < self.exit_threshold or z > self.stop_loss:
                position = 0  # Exit short

        signals.iloc[i] = position

    return signals
```

This state machine ensures we only trade when the spread deviates significantly from its mean, and we exit when it reverts.

---

### Bayesian Optimization

Traditional grid search or random search is inefficient for hyperparameter tuning. Bayesian optimization uses a probabilistic model (Gaussian Process) to intelligently explore the parameter space.

**Gaussian Process Surrogate**:

We model the objective function f(x) (e.g., negative Sharpe ratio) as a Gaussian Process:

```
f(x) ~ GP(μ(x), k(x, x'))
```

where k is the kernel function (commonly Matérn 5/2):

```
k(x, x') = σ²·(1 + √5·r + 5r²/3)·exp(-√5·r)
where r = ||x - x'|| / ℓ
```

**Expected Improvement Acquisition Function**:

At iteration t, we have observations D_t = {(x_i, y_i)}. The Expected Improvement (EI) quantifies the expected gain over the current best:

```
EI(x) = E[max(f_best - f(x), 0)]
      = (f_best - μ(x))·Φ(Z) + σ(x)·φ(Z)

where:
  Z = (f_best - μ(x)) / σ(x)
  Φ(·) = CDF of standard normal
  φ(·) = PDF of standard normal
```

We select the next point by maximizing EI:

```
x_{next} = argmax EI(x)
```

**Implementation**:

```python
# From src/optimization/optimizer.py
def optimize(self) -> OptimizationResult:
    @use_named_args(self.dimensions)
    def objective_wrapper(**params):
        score = self.objective_function(params)

        # Track evaluation
        self.all_params.append(params.copy())
        self.all_scores.append(float(score))

        # Update convergence history
        if len(self.convergence_history) == 0:
            self.convergence_history.append(float(score))
        else:
            best_so_far = min(self.convergence_history[-1], float(score))
            self.convergence_history.append(best_so_far)

        return score

    # Run optimization with Gaussian Process + Expected Improvement
    result = gp_minimize(
        func=objective_wrapper,
        dimensions=self.dimensions,
        n_calls=self.n_iterations,
        n_initial_points=self.n_initial_points,
        acq_func="EI",  # Expected Improvement
        random_state=self.random_state,
        verbose=False
    )

    # Extract best parameters
    best_params = {
        name: float(value)
        for name, value in zip(self.param_names, result.x)
    }

    return OptimizationResult(...)
```

The objective function evaluates a full backtest for each parameter combination, returning the negative Sharpe ratio to minimize.

---

### Risk Management

**Value at Risk (VaR)**:

VaR at confidence level α is the maximum expected loss over a given time horizon:

```
P(Loss > VaR_α) = 1 - α
```

**Methods**:

1. **Historical VaR**: Empirical quantile
   ```
   VaR_α = -Quantile_{1-α}(returns)
   ```

2. **Parametric VaR**: Assumes normal distribution
   ```
   VaR_α = -(μ + σ·z_α)
   where z_α = Φ^{-1}(1-α)
   ```

3. **Cornish-Fisher VaR**: Adjusts for skewness and kurtosis
   ```
   z_CF = z_α + (z_α² - 1)·S/6 + (z_α³ - 3z_α)·K/24 - (2z_α³ - 5z_α)·S²/36

   VaR_α = -(μ + σ·z_CF)

   where:
     S = skewness
     K = excess kurtosis
   ```

**Expected Shortfall (CVaR)**:

Conditional VaR measures the expected loss given that VaR is exceeded:

```
CVaR_α = E[Loss | Loss > VaR_α]
```

**Implementation**:

```python
# From src/risk/manager.py
def calculate_var_cornish_fisher(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> VaRResult:
    alpha = 1 - confidence_level

    # Calculate moments
    mu = returns.mean()
    sigma = returns.std()
    skew = returns.skew()
    kurt = returns.kurtosis()  # Excess kurtosis

    # Get base z-score
    z_alpha = stats.norm.ppf(alpha)

    # Cornish-Fisher adjustment
    z_cf = (
        z_alpha
        + (z_alpha**2 - 1) * skew / 6
        + (z_alpha**3 - 3 * z_alpha) * kurt / 24
        - (2 * z_alpha**3 - 5 * z_alpha) * skew**2 / 36
    )

    # VaR with adjusted z-score
    var = -(mu + sigma * z_cf)

    return VaRResult(
        var=float(var),
        confidence_level=confidence_level,
        method="cornish_fisher"
    )
```

---

## Implementation Details

### Architecture

```
BasketTradingBO/
├── src/
│   ├── cointegration/          # Johansen test, VECM, spread calculation
│   │   ├── engine.py
│   │   └── spread.py
│   ├── optimization/           # Bayesian optimization
│   │   └── optimizer.py
│   ├── strategy/               # Signal generation, filters
│   │   ├── signals.py
│   │   ├── filters.py
│   │   └── portfolio.py
│   ├── backtesting/            # Backtest engine, metrics
│   │   ├── backtester.py
│   │   └── metrics.py
│   ├── risk/                   # VaR, position sizing
│   │   └── manager.py
│   ├── data/                   # Market data, caching
│   │   ├── market_data.py
│   │   ├── cache.py
│   │   └── features.py
│   ├── visualization/          # Plots, reports
│   │   ├── plots.py
│   │   └── reports.py
│   └── utils/                  # Config, logging, exceptions
│       ├── config.py
│       ├── logger.py
│       ├── exceptions.py
│       └── io.py
├── scripts/
│   ├── run_pipeline.py         # Main orchestration script
│   ├── run_optimization.py
│   └── run_backtest.py
├── tests/
│   ├── unit/
│   └── integration/
├── config/
│   └── config.yaml
└── results/                    # Generated outputs
    └── TICKER1_TICKER2_TICKER3/
        └── YYYYMMDD_HHMMSS/
            ├── backtest_report.html
            ├── pipeline_results.json
            ├── correlation_matrix.png
            └── plots/
```

### Key Features

- **Caching**: HDF5 and Parquet caching for market data with TTL
- **Logging**: Structured JSON logging with timestamps and context
- **Error Handling**: Custom exception hierarchy with rich context
- **Testing**: Comprehensive unit tests with property-based testing (Hypothesis)
- **Type Safety**: Full type hints with numpy typing
- **Performance**: Vectorized operations, efficient pandas usage

---

## Case Study: TSLA-NFLX-PLTR Basket

Let's analyze the results from a 3-year backtest (2022-01-01 to 2025-01-01) on the basket [TSLA, NFLX, PLTR].

### Performance Metrics

![Performance Metrics](screenshots/PERFROMANCE%20METRICS.png)

**Results Overview**:
- **Total Return**: -4.62%
- **Annualized Return**: -1.31%
- **Sharpe Ratio**: -0.18
- **Sortino Ratio**: -0.13
- **Max Drawdown**: -8.19%
- **Calmar Ratio**: -0.16
- **Win Rate**: 28.02%
- **Profit Factor**: 0.88
- **Number of Trades**: 416

**Analysis**:

The negative returns indicate that the TSLA-NFLX-PLTR basket did not exhibit stable cointegration over this period. This is a critical insight: not all asset combinations are suitable for mean-reversion strategies.

The low win rate (28%) coupled with a profit factor below 1 suggests that losses outweighed gains. The Sharpe ratio of -0.18 indicates risk-adjusted underperformance relative to cash.

**What went wrong?**

1. **Trending Spreads**: These growth stocks experienced significant directional trends (especially TSLA and PLTR) that violated the mean-reversion assumption.

2. **Weak Cointegration**: The eigenvalues from the Johansen test likely showed weak statistical significance, but the pipeline continued for demonstration.

3. **Parameter Mismatch**: Default parameters (entry threshold = 2.0, exit = 0.5) may not have been optimal for this volatile basket.

### Portfolio Performance

![Portfolio Performance](screenshots/PORTFOLIO%20PERFORMANCE.png)

The portfolio equity curve shows several distinct phases:

1. **Early 2022**: Brief profitable period with rapid gains (early mean reversion signals)
2. **Mid 2022**: Sharp drawdown as spread broke away from historical mean
3. **2023**: Extended flat period with minimal opportunities
4. **2024**: Further deterioration as spread continued trending

The **Trading Signals** panel (bottom) shows the strategy was predominantly in positions, suggesting over-trading. The long periods in SHORT positions (pink) during 2022-2023 coincided with losses, as the spread continued to widen rather than revert.

### Spread and Z-Score Analysis

![Spread and Z-Score](screenshots/Spread%20and%20Z-Score.png)

**Basket Spread** (top panel):

The spread shows a clear **downward trend** from 2022 onwards, violating the stationarity assumption. A stationary spread should fluctuate around a constant mean. This trending behavior is a red flag that cointegration has broken down.

**Z-Score with Trading Thresholds** (bottom panel):

- **Red dashed lines**: Entry thresholds (±2.0σ)
- **Orange dashed lines**: Exit thresholds (±0.5σ)
- **Green triangles**: Long entry signals
- **Red triangles**: Short entry signals

Notable observations:

1. **Early 2022**: Z-score oscillated around zero with successful mean reversion (green triangles at -2σ, followed by reversion)

2. **Late 2024-2025**: Z-score dropped below -2σ and continued declining to -4σ, triggering multiple long signals that resulted in losses (spread kept trending down)

3. **Lack of Mean Reversion**: Post-2022, the spread rarely reverted to the mean after breaching entry thresholds

**Diagnostic**: The spread's half-life likely exceeded 60 days or showed θ ≥ 0 in the OU regression, indicating no mean reversion.

### Transaction Costs

From the HTML report:
- **Total Transaction Costs**: $629.08
- **Average Cost per Trade**: $1.51
- **Cost as % of Returns**: 13.62%

With 416 trades over 3 years, the strategy turned over frequently (averaging 2-3 trades per week). Transaction costs consumed 13.62% of gross returns, which is significant for a strategy with negative returns.

**Lesson**: Mean-reversion strategies are sensitive to transaction costs. High-frequency rebalancing erodes profitability, especially when spread movements are small.

### Key Takeaways

1. **Cointegration is Data-Dependent**: Not all asset groups cointegrate. Tech growth stocks (TSLA, NFLX, PLTR) have idiosyncratic drivers that can override statistical relationships.

2. **Regime Changes**: Relationships stable in one market regime may break down in others. The 2022-2024 period included Fed rate hikes, tech selloff, and AI hype (PLTR), creating divergence.

3. **Pre-Screening is Critical**: Before deploying capital, verify:
   - Johansen test p-value < 0.05
   - Half-life in range 10-60 days
   - Hurst exponent < 0.5
   - Rolling cointegration tests to detect regime changes

4. **Optimization Opportunity**: Running Bayesian optimization (`--optimize`) might have found better parameters, but no amount of optimization can fix a fundamentally non-cointegrated basket.

**Better Candidates**: Sector ETFs (XLF, KBE, KRE for financials), commodities (GLD, SLV), or stocks within the same industry (banks: JPM, BAC, GS) typically show stronger cointegration.

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/BasketTradingBO.git
   cd BasketTradingBO
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

   Or for development with additional tools:
   ```bash
   pip install -e ".[dev]"
   ```

### Dependencies

Core dependencies (see `setup.py`):
- **Data**: `yfinance`, `pandas-datareader`, `pandas`, `numpy`
- **Statistics**: `statsmodels`, `scipy`, `scikit-learn`
- **Optimization**: `scikit-optimize`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Storage**: `tables` (HDF5), `pyarrow` (Parquet)
- **Testing**: `pytest`, `hypothesis`

---

## Usage

### Basic Workflow

1. **Run Complete Pipeline**:

   ```bash
   python scripts/run_pipeline.py \
       --tickers JPM BAC GS \
       --start 2020-01-01 \
       --end 2023-12-31 \
       --optimize \
       --n-iterations 50 \
       --output-dir results
   ```

   This will:
   - Fetch data from Yahoo Finance
   - Test for cointegration (Johansen)
   - Calculate spread and z-score
   - Run Bayesian optimization (50 iterations)
   - Backtest with optimized parameters
   - Generate HTML report with plots

2. **View Results**:

   Results are saved in `results/JPM_BAC_GS/YYYYMMDD_HHMMSS/`:
   - `backtest_report.html`: Interactive HTML report
   - `pipeline_results.json`: All metrics in JSON format
   - `plots/`: Individual plot files
   - `correlation_matrix.png`: Asset correlation heatmap
   - `var_distribution.png`: VaR distribution plot

3. **Example Output**:

   ```
   ============================================================
   BASKET TRADING PIPELINE - COMPLETED
   ============================================================
   Results saved to: results/JPM_BAC_GS/20251119_034828

   Quick Access:
     - View Report:  start results/JPM_BAC_GS/20251119_034828/backtest_report.html
     - View Metrics: cat results/JPM_BAC_GS/20251119_034828/pipeline_results.json
     - View Plots:   explorer results/JPM_BAC_GS/20251119_034828/plots
   ============================================================
   ```

### Advanced Usage

#### Custom Configuration

Create `config/config.yaml`:

```yaml
cointegration:
  method: "johansen"
  significance_level: 0.05
  johansen:
    deterministic_term: "c"  # constant
    test_statistic: "trace"
  adf:
    regression: "c"
    autolag: "BIC"
    maxlag: 10
  spread:
    lookback_window: 252
    half_life_min: 5
    half_life_max: 60

strategy:
  signal:
    entry_threshold: 2.0
    exit_threshold: 0.5
    stop_loss: 4.0
  filters:
    min_holding_period: 5

backtesting:
  initial_capital: 100000.0
  commission: 0.001  # 0.1%
  slippage: 0.0005   # 0.05%
  position_size: 0.20

risk:
  max_position_size: 0.30
  max_portfolio_var: 0.02
  var_confidence_level: 0.95
  lookback_window: 252

data:
  cache:
    enabled: true
    backend: "hdf5"  # or "parquet"
    ttl_days: 7
    directory: ".cache"
```

#### Programmatic API

```python
from src.data.market_data import MarketDataAdapter
from src.cointegration.engine import CointegrationEngine
from src.cointegration.spread import SpreadCalculator
from src.strategy.signals import SignalGenerator
from src.backtesting.backtester import Backtester
from src.optimization.optimizer import BayesianOptimizer

# 1. Fetch data
adapter = MarketDataAdapter()
prices = adapter.fetch_data(["JPM", "BAC", "GS"], "2020-01-01", "2023-12-31")

# 2. Test cointegration
engine = CointegrationEngine()
coint_result = engine.test_cointegration(prices)

if coint_result.is_cointegrated:
    # 3. Calculate spread
    weights = coint_result.eigenvectors[:, 0]
    calc = SpreadCalculator()
    spread = calc.calculate_spread(prices, weights)
    zscore = calc.calculate_zscore(spread, lookback=252)

    # 4. Generate signals
    signal_gen = SignalGenerator(entry_threshold=2.0, exit_threshold=0.5, stop_loss=4.0)
    signals = signal_gen.generate_signals(zscore)

    # 5. Backtest
    backtester = Backtester(initial_capital=100000.0, commission=0.001, slippage=0.0005)
    result = backtester.run_backtest(prices, signals, weights)

    print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"Total Return: {result.metrics.total_return:.2%}")
    print(f"Max Drawdown: {result.metrics.max_drawdown:.2%}")
```

#### Bayesian Optimization Example

```python
def objective(params):
    """Objective function to minimize (negative Sharpe ratio)."""
    entry_threshold = params["entry_threshold"]
    exit_threshold = params["exit_threshold"]
    stop_loss = params["stop_loss"]
    lookback = int(params["lookback_window"])

    # Recalculate z-score with new lookback
    zscore = calc.calculate_zscore(spread, lookback=lookback)

    # Generate signals
    signal_gen = SignalGenerator(entry_threshold, exit_threshold, stop_loss)
    signals = signal_gen.generate_signals(zscore)

    # Backtest
    result = backtester.run_backtest(prices, signals, weights)

    # Return negative Sharpe (we minimize)
    return -result.metrics.sharpe_ratio

# Define parameter space
parameter_space = {
    "entry_threshold": (1.5, 3.0),
    "exit_threshold": (0.2, 1.0),
    "stop_loss": (3.0, 5.0),
    "lookback_window": (60.0, 504.0),
}

# Run optimization
optimizer = BayesianOptimizer(
    objective_function=objective,
    parameter_space=parameter_space,
    n_initial_points=10,
    n_iterations=50,
    random_state=42
)

result = optimizer.optimize()
print(f"Best Sharpe: {-result.best_score:.3f}")
print(f"Best Params: {result.best_params}")
```

---

## Further Implementation Ideas

### 1. Dynamic Cointegration Monitoring

**Problem**: Cointegration relationships can break down over time due to regime changes.

**Solution**: Implement rolling window cointegration tests:

```python
def rolling_cointegration_test(prices: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Test cointegration on rolling windows.

    Returns
    -------
    pd.Series
        Time series of p-values. Values < 0.05 indicate cointegration.
    """
    p_values = []
    dates = []

    for i in range(window, len(prices)):
        window_prices = prices.iloc[i-window:i]

        try:
            result = coint_engine.test_cointegration(window_prices)
            p_values.append(result.p_value)
            dates.append(prices.index[i])
        except:
            p_values.append(1.0)  # No cointegration
            dates.append(prices.index[i])

    return pd.Series(p_values, index=dates)
```

**Trading Rule**: Only trade when rolling p-value < 0.05 for the past N days.

### 2. Multi-Basket Portfolio

**Concept**: Diversify across multiple uncorrelated cointegrated baskets.

```python
baskets = [
    {"tickers": ["JPM", "BAC", "GS"], "name": "banks"},
    {"tickers": ["XOM", "CVX", "COP"], "name": "energy"},
    {"tickers": ["JNJ", "PFE", "MRK"], "name": "pharma"},
]

portfolio_returns = pd.DataFrame()

for basket in baskets:
    # Run pipeline for each basket
    result = run_pipeline(basket["tickers"], ...)
    portfolio_returns[basket["name"]] = result.returns

# Combine with equal weights or optimize weights via mean-variance
total_returns = portfolio_returns.mean(axis=1)
sharpe_ratio = total_returns.mean() / total_returns.std() * np.sqrt(252)
```

### 3. Kalman Filter for Dynamic Hedge Ratios

Instead of static eigenvectors, use a Kalman filter to estimate time-varying hedge ratios:

```python
from pykalman import KalmanFilter

def kalman_hedge_ratio(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    """
    Estimate dynamic hedge ratio β_t in y_t = β_t·x_t + ε_t.
    """
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[x.values],
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01
    )

    state_means, _ = kf.filter(y.values)
    return pd.Series(state_means.flatten(), index=y.index)
```

### 4. Machine Learning for Signal Enhancement

Augment z-score signals with ML features:

```python
from sklearn.ensemble import RandomForestClassifier

# Features
features = pd.DataFrame({
    'zscore': zscore,
    'zscore_ma': zscore.rolling(20).mean(),
    'spread_volatility': spread.rolling(20).std(),
    'half_life': spread.rolling(60).apply(lambda x: calculate_half_life(x)),
    'hurst': spread.rolling(60).apply(lambda x: calculate_hurst_exponent(x)),
    'rsi': calculate_rsi(spread, period=14),
    'volume_ratio': volume.rolling(20).mean() / volume.rolling(60).mean(),
})

# Labels: 1 if profitable in next N days, 0 otherwise
labels = (result.returns.shift(-5).rolling(5).sum() > 0).astype(int)

# Train classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(features.dropna(), labels.dropna())

# Use predictions to filter signals
signal_strength = clf.predict_proba(features)[:, 1]
enhanced_signals = signals * (signal_strength > 0.6)  # Only trade high-confidence
```

### 5. Transaction Cost Optimization

Minimize rebalancing frequency using tolerance bands:

```python
def should_rebalance(current_position, target_position, tolerance=0.05):
    """Only rebalance if deviation exceeds tolerance."""
    deviation = abs(current_position - target_position) / abs(target_position)
    return deviation > tolerance
```

### 6. Alternative Data Integration

Incorporate sentiment, news, or options data:

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Scrape news for basket constituents
news = fetch_news(tickers, start_date, end_date)

# Calculate sentiment
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = news['headline'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Use sentiment as a regime filter
# Only trade when sentiment is neutral (avoid strong directional trends)
trade_when = (sentiment_scores.abs() < 0.5)
filtered_signals = signals * trade_when
```

### 7. Regime-Switching Models

Use Hidden Markov Models to identify market regimes:

```python
from hmmlearn.hmm import GaussianHMM

# Fit HMM to returns
model = GaussianHMM(n_components=2, covariance_type="full")
model.fit(returns.values.reshape(-1, 1))

# Predict regimes
regimes = model.predict(returns.values.reshape(-1, 1))

# Only trade in mean-reverting regime
mean_reverting_regime = regimes == 0  # Assuming regime 0 is low volatility
filtered_signals = signals * mean_reverting_regime
```

### 8. Options-Based Hedging

Use options to hedge tail risk:

```python
# Calculate portfolio delta
portfolio_delta = (positions * prices).sum()

# Purchase out-of-the-money puts to hedge downside
hedge_strike = current_price * 0.95
hedge_quantity = portfolio_delta * 0.5  # Hedge 50% of exposure
```

---

## References and Resources

### Academic Papers

1. **Engle, R. F., & Granger, C. W. J. (1987)**. "Co-integration and error correction: Representation, estimation, and testing." *Econometrica*, 55(2), 251-276.
   - Foundational paper on cointegration and error correction models.

2. **Johansen, S. (1991)**. "Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models." *Econometrica*, 59(6), 1551-1580.
   - The Johansen test for multivariate cointegration.

3. **Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006)**. "Pairs trading: Performance of a relative-value arbitrage rule." *Review of Financial Studies*, 19(3), 797-827.
   - Empirical analysis of pairs trading profitability.

4. **Avellaneda, M., & Lee, J. H. (2010)**. "Statistical arbitrage in the US equities market." *Quantitative Finance*, 10(7), 761-782.
   - Statistical arbitrage using PCA on baskets of stocks.

5. **Pole, A. (2007)**. *Statistical Arbitrage: Algorithmic Trading Insights and Techniques*. Wiley.
   - Comprehensive book on statistical arbitrage strategies.

6. **Vidyamurthy, G. (2004)**. *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
   - Practical guide to pairs trading with cointegration.

7. **Brochu, E., Cora, V. M., & de Freitas, N. (2010)**. "A tutorial on Bayesian optimization of expensive cost functions." *arXiv preprint arXiv:1012.2599*.
   - Tutorial on Bayesian optimization with Gaussian Processes.

8. **Mockus, J. (1974)**. "On Bayesian methods for seeking the extremum." *Optimization Techniques IFIP Technical Conference*, 400-404.
   - Original work on Bayesian global optimization.

### Books

1. **Chan, E. P. (2013)**. *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.
   - Practical strategies including mean reversion and statistical arbitrage.

2. **Chan, E. P. (2009)**. *Quantitative Trading: How to Build Your Own Algorithmic Trading Business*. Wiley.
   - Foundation for building quantitative trading systems.

3. **Tsay, R. S. (2010)**. *Analysis of Financial Time Series* (3rd ed.). Wiley.
   - Comprehensive coverage of time series econometrics, ARIMA, GARCH, cointegration.

4. **Hamilton, J. D. (1994)**. *Time Series Analysis*. Princeton University Press.
   - Advanced treatment of time series methods including cointegration and VECM.

5. **Enders, W. (2014)**. *Applied Econometric Time Series* (4th ed.). Wiley.
   - Applied econometrics with focus on unit roots and cointegration.

6. **de Prado, M. L. (2018)**. *Advances in Financial Machine Learning*. Wiley.
   - Modern ML techniques for finance, including labeling, feature engineering, and backtesting.

7. **Jansen, S. (2020)**. *Machine Learning for Algorithmic Trading* (2nd ed.). Packt.
   - Practical ML applications in trading with Python code examples.

### Online Resources

1. **QuantStart**: https://www.quantstart.com/
   - Tutorials on pairs trading, cointegration, and backtesting.

2. **Hudson & Thames (Quantitative Research)**: https://hudsonthames.org/
   - Open-source implementations of research (PortfolioLab, MlFinLab).

3. **Quantopian Lectures** (Archive): https://www.quantopian.com/lectures
   - Video lectures on statistical arbitrage, risk management, and portfolio optimization.

4. **Statsmodels Documentation**: https://www.statsmodels.org/
   - Python library for cointegration tests, VECM, and time series analysis.

5. **Scikit-Optimize Documentation**: https://scikit-optimize.github.io/
   - Bayesian optimization library used in this project.

### Research Repositories

1. **SSRN (Social Science Research Network)**: https://www.ssrn.com/
   - Preprints of finance and economics research.

2. **arXiv Quantitative Finance**: https://arxiv.org/archive/q-fin
   - Open-access preprints of quantitative finance papers.

3. **Journal of Portfolio Management**: https://jpm.pm-research.com/
   - Peer-reviewed research on portfolio management and trading strategies.

### Software Libraries

1. **statsmodels**: Time series analysis, cointegration tests
2. **scikit-optimize**: Bayesian optimization
3. **scikit-learn**: Machine learning, regression
4. **yfinance**: Market data retrieval
5. **pandas**: Data manipulation
6. **numpy**: Numerical computing
7. **scipy**: Scientific computing, statistics
8. **matplotlib/seaborn**: Visualization
9. **pytest**: Testing framework
10. **hypothesis**: Property-based testing

---

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading involves risk, and you can lose money. The authors are not responsible for any financial losses incurred through use of this software. Always perform thorough backtesting and paper trading before deploying real capital.



---

**Last Updated**: November 19, 2025
