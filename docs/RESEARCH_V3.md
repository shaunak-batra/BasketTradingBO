# v3 research memo

**Status: written before v3.0 was run.** The v3.0 scope in section 4 was then frozen as
`config/config_v3.yaml`, with the universe in `config/universe_v3.yaml`, and the success criteria in section 5
were fixed before the run. Results are in [RESULTS_V3.md](RESULTS_V3.md). The other candidates (C4 to C8) are
still proposals. v2 stays exactly as it is (`config/config.yaml`, results in `results/case_studies/`). This memo
collects the evidence for what to change, and the rules under which a v3 could be tested honestly.

## 1. What v2 actually showed

Measured from the committed v2 results, not from impressions.

| Observation | Number | Where it came from |
|---|---|---|
| Baskets are usually **not hedged** | net exposure above 0.2 of gross in 53% of all folds and 50% of traded folds; above 0.5 in 22% of traded folds | Johansen weights per fold in `results.json` |
| In-sample Sharpe barely predicts out-of-sample | correlation +0.21 (fixed), +0.20 (optimized) across traded folds | fold tables |
| Tuning lowered the hit rate | optimised folds positive out of sample 35% of the time vs 56% for fixed thresholds | fold tables |
| Trades rarely go far underwater, and the ones that do never come back | of 84 closed trades, median worst drawdown inside a trade was -1.1%; 11 breached -5% and only 3 recovered; **none that breached -10% recovered** | trade ledgers + daily equity |
| A hard stop would have helped | replacing every trade that breached the level with a loss at that level: mean trade -0.65% actual, -0.18% with a -10% stop, -0.12% with -5% | same |
| The half-life filter never binds | traded folds had half-lives of 6.6 to 29.7 days (median 12.2) against a 126-day limit | fold tables |
| The book is flat most of the time | 23.5% of folds traded | fold tables |
| The cointegration filter is oversized | 12-13% rejection at a nominal 5% on driftless random walks; 5-6% when prices drift | simulation, see README limitations |

Read together: the losses came from **directional exposure and a stop that could not fire**, not from the
signal being backwards. The tuning layer added variance, not edge.

## 2. The rule that governs everything below

Every change in §3 was suggested by the v2 out-of-sample results. That makes those results **design data**.
Testing v3 on them and reporting the outcome as out-of-sample would repeat, in slower motion, the mistake
this project exists to avoid.

So a v3 needs a fresh evaluation set. Options, in order of preference:

1. **New universe (preferred).** Screen a defined universe mechanically (for example every pair within each
   GICS sub-industry of the S&P 500, or a fixed list of liquid sector and country ETFs). Report the whole
   distribution of outcomes. The five v2 baskets are reported separately and labelled as the design set.
2. **Temporal hold-out.** Design on 2010-2018, evaluate only on 2019 onward. Simple, but one regime.
3. **Both.** Preferred if the data supports it.

Plus the same discipline as v2: freeze the protocol file, record its hash, run once, report everything
including the failures, and **change one thing per version** so that any difference is attributable.

## 3. Candidate changes

### C1. Require hedged weights

* **Why.** Half of all traded folds carried more than 0.2 of gross as net exposure; the two worst trades
  were essentially long-only positions in a crash.
* **Rule.** Reject a fold when `|sum(w)| / sum(|w|)` exceeds a pre-registered cap (0.2 is the natural
  starting point), or construct the position beta-neutral using formation-window betas. Market-neutral
  practice normally keeps portfolio beta inside ±0.1.
* **Code.** `net_exposure(weights)` helper in [src/cointegration/engine.py](../src/cointegration/engine.py);
  new filter beside the cointegration and half-life checks in
  [src/backtesting/walk_forward.py](../src/backtesting/walk_forward.py).
* **Test.** A synthetic same-sign vector must be skipped; realised net exposure in the backtest must stay
  inside the cap.
* **Effort.** Small. **Expected effect:** removes the largest loss mechanism; also reduces the number of
  tradable folds, which are already scarce.

### C2. Replace the z-score stop with a loss-based stop and a time stop

* **Why.** The XOM/CVX trade lost 52% of entry equity without triggering a -4σ stop, because the rolling
  standard deviation grew as fast as the loss. A z-score stop is not a loss limit.
* **Rule.** Close the position when its mark-to-market loss reaches a pre-registered fraction of the equity
  it was sized on (-10% cut zero winners in v2, so it is the conservative choice), and when holding time
  exceeds a multiple of the formation half-life (3x is the common choice). Keep the z-score stop only as a
  signal-level exit. Stop-losses at a residual band (e.g. 4σ) and minimum-profit constructions are both
  standard in the pairs literature.
* **Code.** Execution-level rule inside [src/backtesting/backtester.py](../src/backtesting/backtester.py)
  (the backtester gains a `stop_loss_fraction`, checked against equity at entry), plus `max_holding_bars`, also
  checked in the backtester.
* **Test.** A constructed losing path must exit on the bar the threshold is breached, with the ledger
  reason recorded; no effect when the threshold is not breached.
* **Effort.** Small. **Expected effect:** truncates the left tail. It cannot create edge.

### C3. Target volatility instead of fixed 1x gross

* **Why.** Gross exposure is currently constant, so risk per trade varies with whatever the spread's
  volatility happens to be. That inflates the variance of results without adding return.
* **Rule.** Size each entry so that the formation-window estimate of spread volatility times gross equals a
  pre-registered annualised risk target (8-10%), with a cap on gross. Volatility-managed portfolios in the
  factor literature raise Sharpe ratios substantially (Moreira and Muir), though the mechanism there is
  timing, not sizing.
* **Code.** Replace `gross_exposure` with `target_volatility` plus `max_gross` in `BacktestConfig`, and pass
  the fold's volatility estimate into sizing.
* **Test.** Realised volatility of fold returns should cluster near the target on synthetic data.
* **Effort.** Small to medium. **Expected effect:** comparable risk across folds and baskets; the Sharpe
  ratio becomes a fairer comparison. Not an alpha source.

### C4. Fix the cointegration test's size

* **Why.** Measured 12-13% false positives at a nominal 5%. Applying critical values from the wrong
  deterministic-term case is a known way to roughly double the rejection rate, which matches what we saw.
* **Rule.** Either bootstrap the rank test (Cavaliere, Rahbek and Taylor, 2012, with the heteroskedastic
  variant from 2014) or use critical values for the deterministic case actually assumed. Bootstrapping is
  more work but removes the ambiguity and handles heteroskedasticity, which equity spreads have.
* **Code.** New `src/cointegration/bootstrap.py`: estimate the VECM under the null rank, resample residuals
  (wild bootstrap), recompute the trace statistic, take the empirical quantile. 199-399 replications per
  fold.
* **Test.** Measured size on driftless random walks must land near 5%.
* **Effort.** Medium, and it multiplies compute per fold by the number of replications, so fold-level
  parallelism becomes necessary.

### C5. Screen a universe and control multiple testing

* **Why.** Five hand-picked baskets is the biggest remaining hole. Testing many pairs at 5% produces false
  positives in proportion to the number of tests; with 500 names there are ~125,000 possible pairs.
* **Rule.** Define the universe and the screen in advance; apply Benjamini-Hochberg false-discovery control
  to the cointegration tests, and judge the strategy on the distribution of out-of-sample results, not the
  best basket. The asset-pricing literature argues for much higher hurdles than t = 2 once the number of
  trials is accounted for (Harvey, Liu and Zhu suggest t > 3.0).
* **Data.** This needs survivorship-bias-free prices: Yahoo has no delisted names. CRSP (academic), Norgate
  or Sharadar all cover delistings; survivor-only backtests have been estimated to overstate annual returns
  by roughly 5 percentage points.
* **Code.** `scripts/run_universe.py` plus a universe definition file; fold loop parallelised with joblib.
* **Effort.** Medium to large, mostly data plumbing and compute. **This is the change that most affects
  whether the conclusion is trustworthy.**

### C6. Dynamic hedge ratios (Kalman filter)

* **Why.** Weights are frozen for six months, so the hedge drifts as prices move.
* **Rule.** Model the hedge ratio as a random walk in a state-space model and update it daily
  (Elliott, van der Hoek and Malcolm, 2005).
* **Code.** New `src/cointegration/kalman.py`; the backtester already supports changing target positions.
* **Test.** Recovers a known time-varying beta on synthetic data.
* **Effort.** Medium. **Caution:** it adds process-noise parameters, which is new tuning surface. Test it as
  its own protocol version.

### C7. Model-implied thresholds instead of tuned ones

* **Why.** Bayesian optimisation over four parameters produced in-sample Sharpe ratios of 1.6-2.2 that did
  not survive. Fewer free parameters is the direct answer to that.
* **Rule.** With the spread modelled as an Ornstein-Uhlenbeck process, Bertram (2010) gives analytic entry
  and exit levels that maximise return per unit time, or the Sharpe ratio, net of transaction costs. The
  thresholds then follow from two estimated quantities (mean-reversion speed and volatility) instead of four
  searched ones.
* **Code.** New module; replaces `_tune_parameters` in the optimised mode.
* **Effort.** Medium. **Expected effect:** less overfitting surface, and a much better story: parameters
  estimated rather than searched.

### C8. Factor-residual statistical arbitrage

* **Why.** Pairwise cointegration finds few tradable relationships (23.5% of folds). Trading the residual
  after removing market and sector factors, with an OU model on the residual and an s-score signal, covers a
  far larger opportunity set (Avellaneda and Lee, 2010).
* **Expectations.** Their reported Sharpe was 1.44 for 1997-2007 but only ~0.9 for 2003-2007, and the pairs
  literature documents declining profitability with more non-converging pairs (Do and Faff; Krauss's survey).
  Treat any positive result from this decade with suspicion.
* **Effort.** Large; effectively a second project sharing the backtester and statistics.

## 4. Recommended v3 scope

Keep it minimal and mechanical, so that the result is interpretable:

* **v3.0 = C1 + C2 + C3.** Three rules, no new tuned parameters, all motivated by measured v2 failures.
  Evaluate on a new universe (C5 at whatever scale the data allows).
* **v3.1 = v3.0 + C4.** Once the bootstrap test is in, the filter means what it says.
* **v4 candidates, one per version:** C7 (model-implied thresholds), then C6 (Kalman), then C8 as a separate
  project.

## 5. Pre-registered success criteria

Decide before running, or the run cannot fail honestly:

* **Promising** if, across the screened universe, the median out-of-sample Sharpe ratio is above zero **and**
  the number of baskets with PSR > 0.95 after Benjamini-Hochberg control exceeds what the null predicts.
* **Not promising** if the median out-of-sample Sharpe is at or below zero across at least 50 baskets. In
  that case the honest conclusion is that this strategy family does not work in this period at daily
  frequency with these costs, and the write-up says so.
* Either way, report the full distribution, the skipped folds, and the cost sensitivity.

## 6. Costs and prerequisites

| Item | Cost |
|---|---|
| Survivorship-bias-free data | CRSP via a university, or a commercial feed (Norgate, Sharadar). Needed for C5 with single names; ETF-only universes are a cheaper but still biased compromise |
| Compute | C4's bootstrap multiplies per-fold cost by ~200-400; C5 multiplies by the number of baskets. Parallel folds and a faster inner loop become prerequisites |
| Engineering | C1-C3 are roughly a day including tests; C4 a day; C5 depends mostly on data access |

## 7. Open questions

* Which universe, exactly, and screened how (sub-industry, ETF list, liquidity floor)?
* Trade more than one cointegrating vector when the rank exceeds 1?
* Is daily close-to-close the right frequency, given that the edge, if any, may be intraday?
* Should the half-life filter be tightened (it never binds now) or dropped?

## 8. References

* Cavaliere, G., Rahbek, A., and Taylor, A. M. R. (2012). Bootstrap determination of the co-integration rank
  in vector autoregressive models. *Econometrica*, 80(4). https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA9099
* Osterwald-Lenum critical values and deterministic-term confusion:
  https://ideas.repec.org/p/lbo/lbowps/2007_12.html and statsmodels' own notes
  https://github.com/statsmodels/statsmodels/issues/3129
* Avellaneda, M., and Lee, J.-H. (2010). Statistical arbitrage in the US equities market.
  *Quantitative Finance*, 10(7). https://www.tandfonline.com/doi/abs/10.1080/14697680903124632
* Bertram, W. K. (2010). Analytic solutions for optimal statistical arbitrage trading. *Physica A*, 389(11).
  https://www.sciencedirect.com/science/article/abs/pii/S0378437110001019
* Elliott, R. J., van der Hoek, J., and Malcolm, W. P. (2005). Pairs trading. *Quantitative Finance*, 5(3).
  https://qiniu-images.datayes.com/uqer/4PAIRS%20TRADING.pdf
* Harvey, C. R., Liu, Y., and Zhu, H. (2016). ... and the cross-section of expected returns.
  *Review of Financial Studies*. https://www.nber.org/system/files/working_papers/w20592/w20592.pdf
* Moreira, A., and Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*.
  https://www.nber.org/system/files/working_papers/w22208/w22208.pdf
* Krauss, C. (2017). Statistical arbitrage pairs trading strategies: review and outlook.
  *Journal of Economic Surveys*. https://onlinelibrary.wiley.com/doi/abs/10.1111/joes.12153
* Survivorship-bias-free data: https://www.quantrocket.com/blog/survivorship-bias/ and
  https://concretumgroup.com/how-to-construct-a-survivorship-bias-free-database-in-norgate-using-python/
