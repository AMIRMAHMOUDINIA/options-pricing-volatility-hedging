# Future Work

The current project is a controlled study of option-pricing and hedging mechanics. The natural extensions should make the modelling assumptions more market-facing without turning the repository into an unfocused trading system.

## 1. Real option-chain snapshot

The next empirical step is to load a real option-chain snapshot, clean bid-ask quotes, remove unstable low-vega contracts, recover implied volatility, and plot smile/skew by maturity. The added notebook `09_real_option_chain_iv_snapshot.ipynb` provides this workflow. It expects a user-supplied CSV because option-chain redistribution rights vary by data source.

## 2. Dividend and forward adjustment

The current examples use a simple risk-free-rate setup. For equity options, the implied forward should account for dividends or borrow effects. A useful extension would estimate the forward from put-call parity and compute log-forward moneyness from market prices.

## 3. Stochastic volatility and jumps

Discrete delta hedging could be stress-tested under stochastic volatility, volatility jumps, and price jumps. This would show where continuous-diffusion assumptions understate hedge losses.

## 4. Surface interpolation

The synthetic surface is directly defined on a grid. A market-data version would need careful interpolation across strike and maturity while avoiding static-arbitrage violations.

## 5. Model comparison

Black-Scholes could be compared with a local-volatility or stochastic-volatility model. The comparison should focus on pricing residuals, hedge P&L, and robustness, not only in-sample fit.

## 6. Conservative reporting

Future reports should continue to separate option premium, hedge cashflows, financing, transaction costs, settlement payoff, and residual attribution. This keeps the hedge ledger auditable.
