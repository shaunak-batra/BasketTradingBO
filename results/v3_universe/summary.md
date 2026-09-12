# v3.0 universe run (2026-09-12T06:19:08+00:00)

Protocol `config_v3.yaml` (sha256 b864faaec874), universe `universe_v3.yaml`, price snapshot sha256 81b0afe1ec01.

## Distribution across pairs

- pairs defined: 60, evaluated: 60, skipped for data: 0
- pairs that traded at least once: 42; round trips: 542
- folds traded: 176 of 1560 (11.3%); skipped as not cointegrated 1276, not hedged 102, half-life 6
- median out-of-sample Sharpe: -0.140 with costs, -0.019 with no frictions
- positive Sharpe: 14 of 42 with costs, 20 with no frictions
- median gross P&L per trade -1.2 bps of notional against 12.1 bps of cost
- discoveries after Benjamini-Hochberg at 10%: 0

## Equal-weight portfolio of all pairs

- 2012-01-04 to 2024-12-31, 60 pairs
- with costs: total return -0.69%, Sharpe -0.389 [-0.90, 0.08], PSR 0.081, max drawdown -1.02%
- with no frictions: total return 0.31%, Sharpe 0.177, PSR 0.738

## Best and worst pairs by Sharpe

| Pair | Family | Folds traded | Trades | Total return | Sharpe | Sharpe (no frictions) | PSR | Discovery |
|---|---|---|---|---|---|---|---|---|
| EWJ/EWT | developed_asia | 2/26 | 4 | 4.91% | 0.35 | 0.38 | 0.91 | no |
| IYE/XOP | energy | 6/26 | 15 | 8.29% | 0.33 | 0.40 | 0.88 | no |
| EWL/EWQ | developed_europe | 3/26 | 8 | 3.46% | 0.29 | 0.36 | 0.85 | no |
| EWT/EWY | developed_asia | 4/26 | 9 | 6.46% | 0.24 | 0.28 | 0.81 | no |
| EWQ/EWU | developed_europe | 3/26 | 7 | 2.26% | 0.18 | 0.24 | 0.74 | no |
| IYR/RWR | real_estate | 1/26 | 2 | -1.11% | -0.39 | -0.31 | 0.08 | no |
| EWH/EWJ | developed_asia | 3/26 | 7 | -9.30% | -0.48 | -0.44 | 0.04 | no |
| HYG/JNK | credit | 20/26 | 70 | -3.45% | -0.62 | 0.82 | 0.01 | no |
| IVV/SPY | broad_us_equity | 16/26 | 71 | -6.68% | -2.43 | 0.21 | 0.00 | no |
| GLD/IAU | precious_metals | 24/26 | 119 | -11.04% | -2.80 | 0.22 | 0.00 | no |
