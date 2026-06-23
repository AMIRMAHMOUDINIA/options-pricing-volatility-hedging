# Notes from working through the model

These are the points I kept returning to while connecting the formulas to the hedge ledger.

## Why the stock's expected return disappears from Black–Scholes

Black–Scholes prices the option through replication and no arbitrage. Once the option and stock are combined into a locally delta-neutral position, the instantaneous stock-direction exposure disappears. Under the model assumptions, that locally riskless position must earn the risk-free rate, so the real-world stock drift drops out of the pricing equation.

## The different roles of \(N(d_1)\) and \(N(d_2)\)

For a non-dividend-paying European call, \(N(d_1)\) is the call delta. Under the classical assumptions, \(N(d_2)\) is the risk-neutral probability that the option finishes in the money. The two terms look similar but answer different questions.

## Why implied volatility is usually unique

For an interior European option price, the Black–Scholes value rises with volatility whenever vega is positive. A valid market price inside the no-arbitrage bounds therefore maps to one positive implied volatility.

## Why Newton–Raphson needs protection

Newton's method can become unstable when vega is small, the starting value is poor, or the proposed step leaves the valid volatility interval. I kept a bracket around the solution and use its midpoint whenever the Newton step is unsafe.

## Why convexity in strike matters

European call prices should be decreasing and convex in strike. A convexity violation can imply a butterfly-arbitrage opportunity. In the continuous-strike limit, the second strike derivative of the discounted call-price curve is linked to the risk-neutral terminal density, which cannot be negative.

## Why a short-call hedge holds stock

A short call has negative delta. Holding a positive amount of stock offsets that local directional exposure. The required stock position changes as spot, time, and volatility change.

## Hedge frequency is not a one-way improvement

More frequent rebalancing usually reduces discrete hedging error in the frictionless model. It also increases turnover. Once transaction costs are included, the smallest replication error and the best net result need not occur at the same frequency.

## Delta-hedged P&L and the volatility difference

A useful approximation links delta-hedged P&L to the gamma-weighted difference between realized and implied variance. The exact discrete result also contains rebalancing error, financing, transaction costs, jumps, and model error.

## Delta neutrality is local

A delta-neutral position can still carry gamma, vega, theta, jump, and liquidity exposure. A large move can change delta quickly, leaving the hedge exposed between rebalancing times.

## Questions I would examine next

- How do discrete dividends change parity, pricing, and hedging?
- How large is the error from replacing bid/ask execution with a proportional-cost approximation?
- How does a jump-diffusion or stochastic-volatility path change the residual attribution?
- Can an SVI-type surface retain the same transparent arbitrage checks?
- How does hedging change when several options share the same underlying?
