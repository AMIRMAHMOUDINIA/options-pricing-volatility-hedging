"""No-arbitrage bounds and put-call parity for European options."""

from dataclasses import dataclass
from math import exp, isfinite
from typing import Literal

OptionType = Literal["call", "put"]
ParitySignal = Literal[
    "parity_holds",
    "fiduciary_call_overpriced",
    "fiduciary_call_underpriced",
]


@dataclass(frozen=True)
class PriceBounds:
    lower: float
    upper: float


def _validate_market_inputs(spot: float, strike: float, time_to_expiry: float) -> None:
    for name, value in {
        "spot": spot,
        "strike": strike,
        "time_to_expiry": time_to_expiry,
    }.items():
        if not isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if spot < 0:
        raise ValueError("Spot cannot be negative.")
    if strike <= 0:
        raise ValueError("Strike must be strictly positive.")
    if time_to_expiry < 0:
        raise ValueError("Time to expiry cannot be negative.")


def _validate_option_price(option_price: float, name: str) -> None:
    if not isfinite(option_price):
        raise ValueError(f"{name} must be finite.")
    if option_price < 0:
        raise ValueError(f"{name} cannot be negative.")


def discount_factor(rate: float, time_to_expiry: float) -> float:
    if not isfinite(rate):
        raise ValueError("Rate must be finite.")
    if not isfinite(time_to_expiry) or time_to_expiry < 0:
        raise ValueError("Time to expiry must be finite and non-negative.")
    return exp(-rate * time_to_expiry)


def present_value(future_amount: float, rate: float, time_to_expiry: float) -> float:
    if not isfinite(future_amount) or future_amount < 0:
        raise ValueError("Future amount must be finite and non-negative.")
    return future_amount * discount_factor(rate, time_to_expiry)


def european_option_bounds(
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    option_type: OptionType,
) -> PriceBounds:
    _validate_market_inputs(spot, strike, time_to_expiry)
    if option_type not in {"call", "put"}:
        raise ValueError("Option type must be either 'call' or 'put'.")

    if time_to_expiry == 0:
        payoff = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
        return PriceBounds(payoff, payoff)

    pv_strike = present_value(strike, rate, time_to_expiry)
    if option_type == "call":
        return PriceBounds(max(spot - pv_strike, 0.0), spot)
    return PriceBounds(max(pv_strike - spot, 0.0), pv_strike)


def price_within_bounds(
    option_price: float,
    bounds: PriceBounds,
    tolerance: float = 1e-10,
) -> bool:
    if not isfinite(option_price):
        raise ValueError("Option price must be finite.")
    if tolerance < 0:
        raise ValueError("Tolerance cannot be negative.")
    return bounds.lower - tolerance <= option_price <= bounds.upper + tolerance


def parity_implied_put_price(
    call_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
) -> float:
    _validate_option_price(call_price, "call_price")
    _validate_market_inputs(spot, strike, time_to_expiry)
    return call_price - spot + present_value(strike, rate, time_to_expiry)


def parity_implied_call_price(
    put_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
) -> float:
    _validate_option_price(put_price, "put_price")
    _validate_market_inputs(spot, strike, time_to_expiry)
    return put_price + spot - present_value(strike, rate, time_to_expiry)


def put_call_parity_gap(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
) -> float:
    _validate_option_price(call_price, "call_price")
    _validate_option_price(put_price, "put_price")
    _validate_market_inputs(spot, strike, time_to_expiry)
    return call_price + present_value(strike, rate, time_to_expiry) - put_price - spot


def parity_signal(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    tolerance: float = 1e-8,
) -> ParitySignal:
    if tolerance < 0:
        raise ValueError("Tolerance cannot be negative.")
    gap = put_call_parity_gap(call_price, put_price, spot, strike, rate, time_to_expiry)
    if gap > tolerance:
        return "fiduciary_call_overpriced"
    if gap < -tolerance:
        return "fiduciary_call_underpriced"
    return "parity_holds"
