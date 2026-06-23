"""Implied-volatility inversion with bisection, safeguarded Newton, and Brent."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from scipy.optimize import brentq

from .arbitrage import european_option_bounds
from .black_scholes import black_scholes_price
from .greeks import black_scholes_greeks

OptionType = Literal["call", "put"]
SolverMethod = Literal["bisection", "newton", "brent"]


@dataclass(frozen=True)
class ImpliedVolatilityResult:
    volatility: float
    converged: bool
    iterations: int
    residual: float
    method: SolverMethod


@dataclass(frozen=True)
class ImpliedVolatilityQuote:
    bid: float
    mid: float
    ask: float


def _validate_inputs(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    option_type: OptionType,
    price_tolerance: float,
) -> tuple[float, float]:
    for name, value in {
        "market_price": market_price,
        "spot": spot,
        "strike": strike,
        "rate": rate,
        "time_to_expiry": time_to_expiry,
        "price_tolerance": price_tolerance,
    }.items():
        if not isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if market_price < 0:
        raise ValueError("Market price cannot be negative.")
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or price_tolerance <= 0:
        raise ValueError("Spot, strike, time, and price tolerance must be strictly positive.")
    if option_type not in {"call", "put"}:
        raise ValueError("Option type must be either 'call' or 'put'.")

    bounds = european_option_bounds(spot, strike, rate, time_to_expiry, option_type)
    boundary_tolerance = min(
        price_tolerance,
        1e-14 * max(1.0, abs(bounds.lower), abs(bounds.upper)),
    )
    if market_price < bounds.lower - boundary_tolerance:
        raise ValueError(f"Market price is below the no-arbitrage lower bound of {bounds.lower:.12g}.")
    if market_price > bounds.upper + boundary_tolerance:
        raise ValueError(f"Market price is above the no-arbitrage upper bound of {bounds.upper:.12g}.")
    if market_price >= bounds.upper - boundary_tolerance:
        raise ValueError("Market price is at or numerically too close to the upper bound. No finite implied volatility exists.")
    return bounds.lower, bounds.upper


def _residual(
    volatility: float,
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    option_type: OptionType,
) -> float:
    return float(black_scholes_price(spot, strike, rate, time_to_expiry, volatility, option_type)) - market_price


def _boundary_result(
    market_price: float,
    lower_bound: float,
    price_tolerance: float,
    method: SolverMethod,
) -> ImpliedVolatilityResult | None:
    residual = lower_bound - market_price
    boundary_tolerance = min(
        price_tolerance,
        1e-14 * max(1.0, abs(lower_bound), abs(market_price)),
    )
    if abs(residual) <= boundary_tolerance:
        return ImpliedVolatilityResult(0.0, True, 0, residual, method)
    return None


def _upper_bracket(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    option_type: OptionType,
    initial_upper: float = 0.50,
    maximum_upper: float = 100.0,
) -> float:
    if initial_upper <= 0 or maximum_upper <= initial_upper:
        raise ValueError("Volatility bracket limits are invalid.")
    upper = initial_upper
    while upper <= maximum_upper:
        if _residual(upper, market_price, spot, strike, rate, time_to_expiry, option_type) >= 0:
            return upper
        upper *= 2.0
    raise RuntimeError("Could not bracket the implied-volatility root.")


def implied_volatility_bisection(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    option_type: OptionType,
    price_tolerance: float = 1e-10,
    volatility_tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> ImpliedVolatilityResult:
    if volatility_tolerance <= 0 or max_iterations <= 0:
        raise ValueError("Tolerances and iteration count must be positive.")
    lower_price, _ = _validate_inputs(
        market_price, spot, strike, rate, time_to_expiry, option_type, price_tolerance
    )
    boundary = _boundary_result(market_price, lower_price, price_tolerance, "bisection")
    if boundary is not None:
        return boundary

    low, high = 0.0, _upper_bracket(market_price, spot, strike, rate, time_to_expiry, option_type)
    midpoint = 0.5 * (low + high)
    residual = _residual(midpoint, market_price, spot, strike, rate, time_to_expiry, option_type)
    for iteration in range(1, max_iterations + 1):
        midpoint = 0.5 * (low + high)
        residual = _residual(midpoint, market_price, spot, strike, rate, time_to_expiry, option_type)
        if abs(residual) <= price_tolerance or high - low <= volatility_tolerance:
            return ImpliedVolatilityResult(midpoint, True, iteration, residual, "bisection")
        if residual < 0:
            low = midpoint
        else:
            high = midpoint
    return ImpliedVolatilityResult(midpoint, False, max_iterations, residual, "bisection")


def implied_volatility_newton(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    option_type: OptionType,
    initial_guess: float = 0.20,
    price_tolerance: float = 1e-10,
    volatility_tolerance: float = 1e-10,
    vega_floor: float = 1e-10,
    max_iterations: int = 100,
) -> ImpliedVolatilityResult:
    if initial_guess <= 0 or volatility_tolerance <= 0 or vega_floor <= 0 or max_iterations <= 0:
        raise ValueError("Newton controls must be strictly positive.")
    lower_price, _ = _validate_inputs(
        market_price, spot, strike, rate, time_to_expiry, option_type, price_tolerance
    )
    boundary = _boundary_result(market_price, lower_price, price_tolerance, "newton")
    if boundary is not None:
        return boundary

    low, high = 0.0, _upper_bracket(market_price, spot, strike, rate, time_to_expiry, option_type)
    volatility = min(max(initial_guess, 1e-12), high)

    for iteration in range(1, max_iterations + 1):
        residual = _residual(volatility, market_price, spot, strike, rate, time_to_expiry, option_type)
        if abs(residual) <= price_tolerance:
            return ImpliedVolatilityResult(volatility, True, iteration, residual, "newton")
        if residual < 0:
            low = volatility
        else:
            high = volatility
        if high - low <= volatility_tolerance:
            root = 0.5 * (low + high)
            final_residual = _residual(root, market_price, spot, strike, rate, time_to_expiry, option_type)
            return ImpliedVolatilityResult(root, abs(final_residual) <= price_tolerance, iteration, final_residual, "newton")

        vega = float(black_scholes_greeks(spot, strike, rate, time_to_expiry, volatility, option_type).vega)
        candidate = volatility - residual / vega if abs(vega) >= vega_floor else float("nan")
        if not isfinite(candidate) or not (low < candidate < high):
            candidate = 0.5 * (low + high)
        if abs(candidate - volatility) <= volatility_tolerance:
            final_residual = _residual(candidate, market_price, spot, strike, rate, time_to_expiry, option_type)
            effective_price_tolerance = max(
                price_tolerance,
                abs(vega) * volatility_tolerance,
            )
            return ImpliedVolatilityResult(
                candidate,
                abs(final_residual) <= effective_price_tolerance,
                iteration,
                final_residual,
                "newton",
            )
        volatility = candidate

    final_residual = _residual(volatility, market_price, spot, strike, rate, time_to_expiry, option_type)
    return ImpliedVolatilityResult(volatility, False, max_iterations, final_residual, "newton")


def implied_volatility_brent(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    option_type: OptionType,
    price_tolerance: float = 1e-10,
    volatility_tolerance: float = 1e-12,
    max_iterations: int = 100,
) -> ImpliedVolatilityResult:
    if volatility_tolerance <= 0 or max_iterations <= 0:
        raise ValueError("Tolerance and iteration count must be positive.")
    lower_price, _ = _validate_inputs(
        market_price, spot, strike, rate, time_to_expiry, option_type, price_tolerance
    )
    boundary = _boundary_result(market_price, lower_price, price_tolerance, "brent")
    if boundary is not None:
        return boundary

    upper = _upper_bracket(market_price, spot, strike, rate, time_to_expiry, option_type)
    root, details = brentq(
        lambda vol: _residual(vol, market_price, spot, strike, rate, time_to_expiry, option_type),
        0.0,
        upper,
        xtol=volatility_tolerance,
        rtol=1e-12,
        maxiter=max_iterations,
        full_output=True,
        disp=False,
    )
    residual = _residual(root, market_price, spot, strike, rate, time_to_expiry, option_type)
    return ImpliedVolatilityResult(float(root), bool(details.converged), int(details.iterations), residual, "brent")


def implied_volatility_quote(
    bid_price: float,
    ask_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    option_type: OptionType,
) -> ImpliedVolatilityQuote:
    if bid_price < 0 or ask_price < bid_price:
        raise ValueError("Require 0 <= bid <= ask.")
    mid_price = 0.5 * (bid_price + ask_price)
    bid = implied_volatility_brent(bid_price, spot, strike, rate, time_to_expiry, option_type).volatility
    mid = implied_volatility_brent(mid_price, spot, strike, rate, time_to_expiry, option_type).volatility
    ask = implied_volatility_brent(ask_price, spot, strike, rate, time_to_expiry, option_type).volatility
    return ImpliedVolatilityQuote(bid, mid, ask)
