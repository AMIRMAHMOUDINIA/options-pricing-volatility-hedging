# Model choices and calculations

This note records the assumptions I used and the accounting choices that mattered most while building the pricing and hedging experiments.

## 1. Payoff and profit are not the same quantity

European call and put terminal payoffs are

\[
C_T=\max(S_T-K,0), \qquad P_T=\max(K-S_T,0).
\]

Profit also includes the premium and the position direction. I kept these concepts separate throughout the code because an option can finish in the money and still lose money after the premium is included.

## 2. No-arbitrage conditions

For a non-dividend-paying underlying,

\[
\max(S_0-Ke^{-rT},0)\le C_0\le S_0,
\]

\[
\max(Ke^{-rT}-S_0,0)\le P_0\le Ke^{-rT}.
\]

Put–call parity is

\[
C_0-P_0=S_0-Ke^{-rT}.
\]

I use these relationships as checks that do not depend on the later numerical implementation.

## 3. Black–Scholes pricing

The implementation assumes a non-dividend-paying European option:

\[
C_0=S_0N(d_1)-Ke^{-rT}N(d_2),
\]

\[
P_0=Ke^{-rT}N(-d_2)-S_0N(-d_1),
\]

with

\[
d_1=\frac{\ln(S_0/K)+(r+\tfrac12\sigma^2)T}{\sigma\sqrt{T}},
\qquad d_2=d_1-\sigma\sqrt{T}.
\]

Expiration and zero-volatility limits are handled analytically rather than hidden behind arbitrary epsilon values.

## 4. Greeks

The code implements analytical delta, gamma, vega, calendar-time theta, and rho. I compare them with central finite differences so that the analytical formulas are not validating themselves.

Raw vega and rho use decimal volatility and rate changes. Display values divide them by 100 to represent one percentage-point moves. Annual theta is divided by 365 for a calendar-day value.

## 5. Implied volatility

Implied volatility solves

\[
V_{BS}(\sigma)-V_{market}=0.
\]

I kept three solvers because they reveal different numerical behavior:

- bisection gives transparent convergence once a bracket is valid;
- safeguarded Newton–Raphson is faster when vega is informative but falls back when the step is unsafe;
- Brent's method serves as a robust reference implementation.

Prices are checked against no-arbitrage bounds before inversion. A price at the lower bound maps to zero volatility; a price at the upper bound has no finite Black–Scholes implied volatility.

## 6. A deliberately simple volatility surface

I used

\[
\sigma(k,T)=\sigma_0+\alpha\ln(1+T)+\beta k+\gamma k^2,
\]

where

\[
k=\ln(K/F_{0,T}), \qquad F_{0,T}=S_0e^{rT}.
\]

The point was not to propose a production surface model. I wanted a controllable shape that could be converted into option prices, inverted again, and checked for:

- price bounds;
- decreasing call price with strike;
- vertical-spread slope limits;
- convexity in strike;
- non-decreasing total variance \(w=\sigma^2T\) across maturity.

The quadratic form is only used on the tested grid and is not assumed to be globally arbitrage-free.

## 7. Price paths

Underlying paths follow geometric Brownian motion:

\[
S_{t+\Delta t}=S_t\exp\left[(\mu-\tfrac12\sigma_{real}^2)\Delta t+\sigma_{real}\sqrt{\Delta t}Z\right].
\]

Random seeds are fixed so that changes in the code can be compared against the same paths.

## 8. Dynamic delta hedging

For option-position sign \(s\in\{+1,-1\}\), unit count \(N\), and option delta \(\Delta_t\), the target stock position is

\[
q_t=-s\Delta_tN.
\]

The cash account records:

- option premium paid or received;
- stock purchases and sales;
- interest on positive or negative cash;
- proportional transaction costs;
- terminal stock liquidation;
- option settlement.

No external cash is injected after inception. This is the accounting condition I use to call the hedge self-financing.

## 9. P&L attribution

### Exact cash ledger

\[
\Pi_T=C_{option}+C_{stock}+C_{settlement}+I-TC.
\]

### Exact counterfactual attribution

I rerun the same path under:

1. model premium and zero cost;
2. actual premium and zero cost;
3. actual premium and actual cost.

This separates model hedging P&L, initial pricing edge, and transaction-cost drag without changing the underlying path.

### Local Greek attribution

\[
\Delta V\approx\Delta\Delta S+\tfrac12\Gamma(\Delta S)^2+\Theta\Delta t+\text{Vega}\Delta\sigma+\rho\Delta r.
\]

### Gamma-weighted variance attribution

For a delta-hedged position,

\[
d\Pi\approx\tfrac12sN\Gamma_tS_t^2(\sigma_{real}^2-\sigma_{imp}^2)dt.
\]

The last two are approximations. Their residuals contain discrete rebalancing, higher-order terms, and path effects.
