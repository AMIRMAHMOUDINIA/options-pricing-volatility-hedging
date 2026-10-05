"""Options Lab: pricing, volatility surfaces, Greeks, American exercise, and hedging."""

from .arbitrage import (
    PriceBounds,
    european_option_bounds,
    put_call_parity_gap,
)
from .binomial import (
    CRRResult,
    EarlyExercisePoint,
    crr_binomial_price,
    crr_binomial_tree,
)
from .black_scholes import (
    black_scholes_d1_d2,
    black_scholes_price,
)
from .greeks import (
    Greeks,
    MarketGreeks,
    black_scholes_greeks,
    to_market_greeks,
)
from .higher_order_greeks import (
    HigherOrderGreeks,
    MarketHigherOrderGreeks,
    black_scholes_higher_order_greeks,
    to_market_higher_order_greeks,
)
from .implied_volatility import (
    ImpliedVolatilityResult,
    implied_volatility_bisection,
    implied_volatility_brent,
    implied_volatility_newton,
)
from .numerical_higher_order_greeks import (
    numerical_higher_order_greeks,
)
from .payoffs import (
    call_payoff,
    option_profit,
    put_payoff,
)
from .svi import (
    SVIFitResult,
    SVIParameters,
    SVISurfaceFit,
    fit_svi_smile,
    fit_svi_surface,
    svi_butterfly_g,
    svi_implied_volatility,
    svi_total_variance,
)

__all__ = [
    "CRRResult",
    "EarlyExercisePoint",
    "Greeks",
    "HigherOrderGreeks",
    "ImpliedVolatilityResult",
    "MarketGreeks",
    "MarketHigherOrderGreeks",
    "PriceBounds",
    "SVIFitResult",
    "SVIParameters",
    "SVISurfaceFit",
    "black_scholes_d1_d2",
    "black_scholes_greeks",
    "black_scholes_higher_order_greeks",
    "black_scholes_price",
    "call_payoff",
    "crr_binomial_price",
    "crr_binomial_tree",
    "european_option_bounds",
    "fit_svi_smile",
    "fit_svi_surface",
    "implied_volatility_bisection",
    "implied_volatility_brent",
    "implied_volatility_newton",
    "numerical_higher_order_greeks",
    "option_profit",
    "put_call_parity_gap",
    "put_payoff",
    "svi_butterfly_g",
    "svi_implied_volatility",
    "svi_total_variance",
    "to_market_greeks",
    "to_market_higher_order_greeks",
]
