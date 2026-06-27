# Results Summary

This document summarizes the fixed-seed outputs produced by the pricing, implied-volatility, surface, and delta-hedging experiments.

## Implied-volatility inversion

Bisection, safeguarded Newton-Raphson, and Brent inversion recovered the known synthetic volatilities. The largest absolute error in the exported solver comparison is 1.75e-11. The synthetic surface was also inverted from call and put prices; the largest recovered-IV error across the surface is 8.05e-07.

Interpretation: the numerical solvers are accurate in the controlled cases. The important practical caveat is that real option chains introduce low-vega contracts, wide spreads, stale quotes, and missing strikes, so solver accuracy alone is not enough.

## Static-arbitrage checks

The synthetic surface passed strike-slice checks in the exported tables: price bounds, monotonicity, slope bounds, convexity, and calendar total-variance checks. Strike checks passed for all saved maturities: True. Calendar checks passed across saved log-moneyness slices: True.

Interpretation: a smooth-looking surface is not automatically economically valid. The project therefore checks basic static-arbitrage conditions rather than only plotting implied volatility.

## Delta-hedging frequency

With implied and realized volatility both set to 20% and no transaction costs, terminal P&L RMSE decreased from 1.50 with monthly hedging to 1.07 with weekly hedging and 0.45 with daily hedging.

Interpretation: more frequent rebalancing reduced replication error in the Black-Scholes-style setting, as expected.

## Transaction-cost trade-off

With a 5-basis-point proportional trading cost, average transaction cost increased from 0.11 for monthly hedging to 0.30 for daily hedging. Daily hedging still had lower volatility than monthly hedging, but the average terminal P&L moved lower because rebalancing became more expensive.

Interpretation: rebalancing frequency is not a one-way improvement. Lower hedge error must be weighed against turnover and cost.

## Realized versus implied volatility

With implied volatility fixed at 20%, the average daily-hedged short-call P&L was 3.95 when realized volatility was simulated at 10%, approximately -0.01 at 20%, and -4.00 at 30%.

Interpretation: the short option benefited when realized volatility was below the volatility embedded in the premium and lost when realized volatility exceeded it.

## Main limitation

Most saved results use synthetic surfaces and simulated price paths. The new option-chain notebook provides a market-data workflow, but a real snapshot must be supplied separately because live market data availability and redistribution rights depend on the chosen data provider.
