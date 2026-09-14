# Audit of v1 and the v2 rebuild

September 2026

Version 1 of this project published backtest results, a TSLA/NFLX/PLTR case study run with default
parameters, and performance claims. An audit found that the backtester could not record profit or loss
correctly, so **every v1 number was an artefact of the accounting, not of the strategy**.
Several other defects compounded the problem. v2 is a rebuild. This document records what
was wrong, the evidence, and the test that now guards against each defect.

## Defects found in v1

| # | v1 defect | Evidence | v2 fix | Guarded by |
|---|---|---|---|---|
| 1 | Equity was computed as `capital + sum(shares * price) - costs`, with no cash leg. Opening a position moved equity by the basket's net market value, and closing it reset equity to capital minus costs, so no trade could ever realise a profit or loss. | With flat prices, opening a position moved equity from 100,000 to 106,570; closing it returned 99,940 (only the costs were lost). The v1 TSLA/NFLX/PLTR "-4.62%" equals -$629 of costs plus -$3,991 of market value in the position still open at the end. | Cash accounting: equity changes only by mark-to-market P&L, trading costs and borrow fees. | `test_flat_prices_lose_exactly_the_round_trip_cost`, `test_equity_change_is_fully_explained_by_pnl_and_costs` |
| 2 | When the Johansen test found no cointegration it raised an exception; the pipeline caught it and silently traded fallback weights (1 on the first asset and -1/(n-1) on each of the others, so `[1, -1]` for a pair and `[1, -0.5, -0.5]` for three assets), still logged as a "cointegrating vector". | The committed v1 JPM/BAC run traded `[1, -1]` and the committed JPM/BAC/GS run traded `[1, -0.5, -0.5]` (commit 2f9cda8). | The test returns a result. A fold that fails the pre-registered filters is not traded, and the reason is recorded. | `test_independent_random_walks_are_rarely_traded` |
| 3 | Johansen ran on raw prices, but its eigenvector was applied to log prices. | TSLA/NFLX/PLTR passed Johansen on raw prices, yet the traded log spread showed no mean reversion. | Johansen, the spread and the half-life all use log prices. | `test_johansen_is_estimated_on_log_formation_prices`, `test_spread_is_the_weighted_sum_of_log_prices` |
| 4 | Weights were estimated on the full sample and parameters were optimised on the same data that was then reported. | Look-ahead by construction. | Walk-forward: weights and parameters come from a formation window; only the following, unseen window is reported. | `test_future_prices_cannot_change_past_results`, `test_future_prices_cannot_change_past_results_with_optimisation` |
| 5 | A signal computed from the close of bar t was filled at that same close. | Same-bar execution. | Fills happen at the next close (`execution_lag: 1`). | `test_signal_is_filled_after_the_close_that_produced_it`, `test_future_prices_cannot_change_past_equity` |
| 6 | The z-score filled its warm-up with an expanding window, so it traded on a handful of observations. | v1 TSLA/NFLX/PLTR opened shorts in January 2022 despite a 252-day lookback. | No partial windows: the z-score is undefined until `lookback` bars exist. | `test_zscore_has_no_partial_windows_and_matches_a_manual_calculation`, `test_zscore_is_causal` |
| 7 | Positions were re-sized every day to a constant dollar amount of the *initial* capital, charging costs daily. | "416 trades" for 6 round trips. | Positions are sized once at entry on current equity and held in constant shares. | `test_one_ledger_row_per_round_trip`, `test_next_trade_is_sized_on_current_equity` |
| 8 | Metric definitions were wrong: `num_trades` counted days with a non-zero return, win rate was computed over all days, average trade duration was hard-coded to 1.0, Sortino used the standard deviation of negative returns only, and annualised return compounded the arithmetic mean. | Documented "win rate 28%" was the share of trading days with a positive return (4 of 6 trades actually won). | Trade statistics come from a round-trip ledger; CAGR is geometric; Sortino uses downside deviation over all periods. | `tests/unit/test_metrics.py` |
| 9 | The Johansen "p-value" was hard-coded (0.05 or 0.10) and "is_stationary" was really Hurst < 0.5. | Code inspection. | Trace and max-eigenvalue statistics are reported with their critical values; no p-value is invented. | `test_summary_reports_statistics_with_critical_values_and_no_p_value` |
| 10 | The optimiser's stop-loss range overlapped the entry range, so some trials raised; every exception scored 1e6. Position size was also searched even though it barely affects Sharpe. | Code inspection of the v1 optimiser (overlapping entry and stop ranges, exceptions scored 1e6, position size searched). | The search box is re-parameterised so every point is valid; errors propagate; position size is not searched. | `test_every_point_in_the_search_box_is_a_valid_parameter_set`, `test_objective_errors_propagate_instead_of_becoming_penalties` |
| 11 | Parametric Expected Shortfall had a sign error (it came out negative). | Code inspection. | Fixed. | `test_parametric_var_and_es_match_the_gaussian_formulas` |
| 12 | Broken code and tests: the cache module failed to import, `run_backtest.py` and `run_optimization.py` called methods that did not exist, and the suite had 15 failures, 6 errors and 59% coverage. Two fixtures were wrong: the "mean-reverting" series was explosive and the "cointegrated" prices were not cointegrated. | `pytest` output. | Unused v1 modules removed; tests rewritten around hand-computed cases and invariants. | Whole suite, run in CI. |
| 13 | Documentation did not match the code or results: an example output (Sharpe 1.23, +15.4%) matched no run; the README claimed interactive charts, VaR in the report, caching and property-based tests; the case-study narrative contradicted its own data. | Comparison of README against code and committed results. | The README results table is generated from `results/case_studies/summary.json`. | `test_readme_results_table_is_generated_from_the_committed_results`, `test_readme_local_links_point_to_existing_files` |

## What did not change

The research idea is the same: Johansen cointegration on a basket, a z-score state machine
with entry, exit and stop thresholds, and Gaussian-process Bayesian optimisation of those
thresholds. v2 changes how that idea is evaluated, not what it is.

## Process safeguards added in v2

* The protocol (`config/config.yaml`) and the baskets (`config/case_studies.yaml`) were fixed
  before v2 produced any result. Each `results.json` records the protocol file's SHA-256.
* Each result also records the SHA-256 of the exact price snapshot, the package versions, the git
  commit (HEAD) and a `dirty` flag that is true when the working tree had uncommitted changes. The
  commit identifies the code only when `dirty` is false.
* Nothing converts an error into a default value or a penalty score.
