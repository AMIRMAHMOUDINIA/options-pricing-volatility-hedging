"""Options Lab: pricing, implied volatility, surfaces, and delta hedging."""

from .arbitrage import PriceBounds, european_option_bounds, put_call_parity_gap
from .black_scholes import black_scholes_d1_d2, black_scholes_price
from .greeks import Greeks, MarketGreeks, black_scholes_greeks, to_market_greeks
from .implied_volatility import (
    ImpliedVolatilityResult,
    implied_volatility_bisection,
    implied_volatility_brent,
    implied_volatility_newton,
)
from .payoffs import call_payoff, option_profit, put_payoff

__all__ = [
    "PriceBounds",
    "Greeks",
    "MarketGreeks",
    "ImpliedVolatilityResult",
    "black_scholes_d1_d2",
    "black_scholes_price",
    "black_scholes_greeks",
    "to_market_greeks",
    "european_option_bounds",
    "put_call_parity_gap",
    "implied_volatility_bisection",
    "implied_volatility_brent",
    "implied_volatility_newton",
    "call_payoff",
    "put_payoff",
    "option_profit",
]
