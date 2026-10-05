# Vanna and volga/vomma

This extension adds two higher-order Black-Scholes volatility sensitivities to the European option-pricing layer.

## Definitions

Vanna measures the interaction between spot and volatility:

`Vanna = d²V / (dS dσ)`

It can also be interpreted as the sensitivity of delta to volatility, or equivalently the sensitivity of vega to spot.

Volga, also called vomma, measures the curvature of option value with respect to volatility:

`Volga = d²V / dσ²`

Under the non-dividend Black-Scholes assumptions used by the project's European pricing layer:

`Vanna = -phi(d1) d2 / sigma`

`Volga = Vega d1 d2 / sigma`

where `phi` is the standard-normal density.

Calls and puts with the same inputs have identical vanna and volga in this model. Put-call parity differs only by terms without volatility dependence, so taking these volatility derivatives removes the parity adjustment.

## Independent numerical validation

The analytical formulas are not checked by differentiating the analytical Greek formulas themselves. The numerical benchmark works directly from Black-Scholes option prices.

Vanna is estimated using a mixed central finite difference:

`[V(S+h,σ+k)-V(S+h,σ-k)-V(S-h,σ+k)+V(S-h,σ-k)] / (4hk)`

Volga is estimated using a second central difference:

`[V(σ+k)-2V(σ)+V(σ-k)] / k²`

The saved validation grid uses spot 100, rate 3%, maturity 0.75 years, volatility 25%, calls and puts, and strikes from 70 through 130.

## Validation results

- Maximum absolute vanna error: **0.0000001068**
- Maximum absolute volga error: **0.0000087596**

The call and put calculations are both included in the saved CSV. Because their analytical values are identical under Black-Scholes, the two option types also provide a parity-based structural check.

![Vanna validation](../outputs/figures/vanna_validation.png)

![Volga validation](../outputs/figures/volga_validation.png)

## Units

The raw derivatives use decimal volatility. Therefore a one-volatility-point move corresponds to `0.01` in sigma.

The package also provides market-unit conversions:

- vanna per volatility point = raw vanna × 0.01;
- volga per volatility-point squared = raw volga × 0.01².

For a finite volatility change, volga enters a second-order Taylor approximation with the usual one-half coefficient.

## Interpretation boundary

These Greeks belong to the same constant-volatility, non-dividend-paying Black-Scholes framework as the project's existing analytical Greeks. They are local sensitivities, not a stochastic-volatility model and not a claim that implied volatility actually remains constant when spot moves.
