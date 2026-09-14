# v3.0 results: hedged, stopped, risk-sized, and tested on a fresh universe

Protocol [config/config_v3.yaml](../config/config_v3.yaml), universe
[config/universe_v3.yaml](../config/universe_v3.yaml), full output in
[results/v3_universe/](../results/v3_universe/). Both files were frozen before this run existed.

Reproduce with:

```bash
python scripts/run_universe.py --config config/config_v3.yaml --universe config/universe_v3.yaml
```

## What v3.0 changed

Three rules, each answering a failure measured in v2 (see [RESEARCH_V3.md](RESEARCH_V3.md)). Signal
thresholds, windows and costs are identical to v2's fixed-threshold run, and optimisation is switched off. The
evaluation universe also changed (see below), so v2 and v3.0 are not a controlled comparison: no difference
between them can be attributed to the three rules alone, or to any one of them.

| Rule | Setting | The v2 failure it answers |
|---|---|---|
| Hedged-weight cap | `max_net_exposure: 0.20` | Half of v2's traded folds were directional, not spreads |
| Loss stop and time stop | `stop_loss_fraction: 0.10`, `max_holding_bars: 36` | A z-score stop never fired while a trade lost 52% |
| Volatility targeting | `target_volatility: 0.10`, `max_gross: 1.0` | Fixed 1x gross made risk per trade arbitrary |

## How it was evaluated

Every v3 rule was chosen after seeing v2's out-of-sample results, so those five baskets are design
data and cannot also be the test. v3 runs on a different universe: **60 ETF pairs inside 17 families**,
every pair within a family, no pair across families, and none of v2's tickers. Two pairs are
**positive controls** (GLD/IAU and IVV/SPY track the same underlying), so the machinery has something
it must find. The other pairs in their families are ordinary test pairs.

Significance is judged with Benjamini-Hochberg control at 10%, using `1 - PSR` as each pair's p-value,
across the 42 pairs that traded (a pair that never trades has a flat equity curve and no PSR).

## Headline result

| | With costs (5 bps/side + 50 bps borrow) | Frictionless |
|---|---|---|
| Median out-of-sample Sharpe across traded pairs | **-0.140** | -0.019 |
| Pairs with positive Sharpe | 14 of 42 | 20 of 42 |
| Equal-weight portfolio, 2012-2024 | **-0.69%** total, Sharpe **-0.389** [-0.90, 0.08], PSR 0.081 | +0.31% total, Sharpe 0.177 [-0.35, 0.66], PSR 0.738 |
| Discoveries after Benjamini-Hochberg at 10% | **0 of 42** tested (60 evaluated) | - |

Activity: 176 of 1,560 folds traded (11.3%), 542 round trips across 42 pairs. Folds were skipped as
not cointegrated 1,276 times, as unhedged 102 times, and for a long half-life 6 times.

Even with every friction removed, the median pair is flat and the portfolio earns 0.31% over thirteen
years. There is no meaningful gross edge to recover, let alone one that survives costs.

## The decisive evidence: the positive controls

| Control | Folds traded | Round trips | Spread volatility | Gross P&L per trade | Cost per trade | Sharpe | Sharpe frictionless |
|---|---|---|---|---|---|---|---|
| GLD/IAU | 24 of 26 | 119 | 0.6%/yr | **0.53 bps** | **10.3 bps** | **-2.80** | +0.22 |
| IVV/SPY | 16 of 26 | 71 | 0.3%/yr | **0.51 bps** | **10.2 bps** | **-2.43** | +0.21 |

GLD/IAU traded more folds than any other pair in the universe, and IVV/SPY was third, behind only
HYG/JNK. That is exactly right: they are genuinely cointegrated, and the pipeline found it. **The
machinery works.** They also had the two lowest Sharpe ratios in the universe, because the spread they
capture is roughly twenty times smaller than the cost of capturing it.

That is the whole result in miniature: where cointegration is unambiguous, the tradable spread is
tiny; where the spread is large enough to pay for trading, the cointegration is not stable.

## Why leverage cannot rescue it

Volatility targeting wanted large positions in these quiet spreads (a 10% target against their
volatility implies about 17x gross for GLD/IAU and 33x for IVV/SPY), and `max_gross: 1.0` capped it.
Lifting that cap would not help.
P&L per trade and cost per trade are both proportional to notional:

```
net per trade = (gross_bps - cost_bps) x notional
```

With gross at 0.5 bps and cost at 10 bps, more notional multiplies a negative number. Across the
whole universe, only 14 of 42 traded pairs earned more gross per trade than they paid in costs, and
the median pair earned **-1.2 bps against 12.1 bps of cost**. This is a cost problem in the sense
that costs are larger than the edge, not in the sense that a cheaper broker would fix it.

## Cost sensitivity

Fold decisions frozen, only the cost model changed (borrow still charged):

| Cost per side | Median Sharpe | Pairs with positive Sharpe |
|---|---|---|
| 0 bps | -0.033 | 19 of 42 |
| 5 bps (the protocol) | -0.140 | 14 of 42 |
| 10 bps | -0.244 | 12 of 42 |
| 20 bps | -0.341 | 8 of 42 |

## Verdict against the pre-registered criteria

[RESEARCH_V3.md](RESEARCH_V3.md) set these before the run:

* **Promising** if the median out-of-sample Sharpe is above zero *and* the count of PSR > 0.95 pairs
  after Benjamini-Hochberg exceeds the null expectation. As implemented, that count is the number of
  Benjamini-Hochberg discoveries at 10% on `1 - PSR`, which is zero; no pair reached PSR > 0.95 either (highest
  0.910). With a median of -0.140: **not met**.
* **Not promising** if the median is at or below zero across at least 50 baskets. All 60 pairs were
  evaluated and 42 traded. The median across those 42 is -0.140; the 18 that never traded have no Sharpe
  ratio, and counting them as zero gives a median of 0.00 across all 60, still at or below zero: **met**.
  The criterion did not say how to treat baskets that never trade, so this is an interpretation; counting them
  as zero puts the median exactly on the boundary, and either reading gives the same verdict.

So the conclusion under the pre-registered criteria stands: **daily cointegration pairs trading on this
universe, with these costs, does not work**, and the three v3 rules did not change that. Two of them did what
they were designed to do, which was narrower: the hedging filter removed 102 directional folds, and the 36-day
time stop closed 69 of the 542 trades. Volatility targeting mostly could not act: 39 of the 42 traded pairs had
a median formation-window spread volatility below the 10% target, so `max_gross: 1.0` capped most of their
positions, and risk per trade was not made comparable across pairs. The 10% loss stop never fired: once
positions were hedged and held to at most 1x gross on these quiet spreads, no trade lost 10% of the equity it
was sized on. (The z-score stop fired 10 times.) Cleaner risk, same absent edge.

## What would change the conclusion

Not another parameter. The result says the gross edge in stable spreads at daily frequency is about one
twentieth of the cost of trading it, so the next versions have to change the economics, not the settings:

* a shorter holding frequency, where the spread is larger relative to costs (intraday, or at least
  execution at the open with limit orders);
* residual-based statistical arbitrage on a factor model rather than pairwise cointegration, which
  gives many more, larger residuals to trade (Avellaneda and Lee, 2010);
* a universe where relationships are structurally tighter (ETF creation/redemption arbitrage,
  dual-listed shares, futures calendar spreads), accepting that those are crowded for good reason.

## Limitations of this test

* ETFs sidestep most survivorship bias but not all: funds close, and this universe is made of
  survivors chosen today.
* The 5% trace test over-rejects: about 10% of simulated two-asset random walks pass it, with no drift or a
  realistic one (`scripts/johansen_size_simulation.py`), so the cointegration filter is looser than its label.
  A bootstrap rank test (C4 in [RESEARCH_V3.md](RESEARCH_V3.md), planned for v3.1) would address this; it is not
  implemented yet.
* `1 - PSR` is an asymptotic p-value, and pair returns are correlated (overlapping ETFs), so
  Benjamini-Hochberg control here is approximate rather than exact.
* Signal fills are next-close at a flat cost per side, while loss and time stops (69 of the 542 exits
  here) fill at the close on which they are detected, which is slightly optimistic. Real execution could be
  cheaper with patient limit orders, which is precisely the direction the conclusion points.
* Six-month trading windows with a two-year formation window is one design among many; testing others
  would require a new protocol version and a fresh evaluation set.
