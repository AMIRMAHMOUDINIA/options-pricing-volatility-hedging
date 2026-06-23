"""Finite-difference validation of Black-Scholes Greeks."""

from math import isfinite
from typing import Literal

from .black_scholes import black_scholes_price
from .greeks import Greeks

OptionType = Literal["call", "put"]


def numerical_greeks(
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    volatility: float,
    option_type: OptionType,
    spot_step: float | None = None,
    volatility_step: float = 1e-4,
    time_step: float = 1e-5,
    rate_step: float = 1e-5,
) -> Greeks:
    for name, value in {
        "spot": spot,
        "strike": strike,
        "rate": rate,
        "time_to_expiry": time_to_expiry,
        "volatility": volatility,
    }.items():
        if not isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
        raise ValueError("Spot, strike, time, and volatility must be strictly positive.")
    if option_type not in {"call", "put"}:
        raise ValueError("Option type must be either 'call' or 'put'.")

    h_s = max(1e-4 * spot, 1e-6) if spot_step is None else spot_step
    for name, value in {
        "spot_step": h_s,
        "volatility_step": volatility_step,
        "time_step": time_step,
        "rate_step": rate_step,
    }.items():
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive.")
    if spot - h_s <= 0 or volatility - volatility_step <= 0 or time_to_expiry - time_step <= 0:
        raise ValueError("Finite-difference step is too large for the supplied point.")

    def price(s: float = spot, r: float = rate, t: float = time_to_expiry, v: float = volatility) -> float:
        return float(black_scholes_price(s, strike, r, t, v, option_type))

    base = price()
    up_s, down_s = price(s=spot + h_s), price(s=spot - h_s)
    delta = (up_s - down_s) / (2.0 * h_s)
    gamma = (up_s - 2.0 * base + down_s) / h_s**2

    up_v, down_v = price(v=volatility + volatility_step), price(v=volatility - volatility_step)
    vega = (up_v - down_v) / (2.0 * volatility_step)

    more_time, less_time = price(t=time_to_expiry + time_step), price(t=time_to_expiry - time_step)
    theta = (less_time - more_time) / (2.0 * time_step)

    up_r, down_r = price(r=rate + rate_step), price(r=rate - rate_step)
    rho = (up_r - down_r) / (2.0 * rate_step)
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
