"""Analytical Black-Scholes Greeks and market-unit conversions."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

OptionType = Literal["call", "put"]
Position = Literal["long", "short"]
NumericResult = float | NDArray[np.float64]


@dataclass(frozen=True)
class Greeks:
    delta: NumericResult
    gamma: NumericResult
    vega: NumericResult
    theta: NumericResult
    rho: NumericResult


@dataclass(frozen=True)
class MarketGreeks:
    delta: NumericResult
    gamma: NumericResult
    vega_per_vol_point: NumericResult
    theta_per_day: NumericResult
    rho_per_rate_point: NumericResult


def _as_result(value: NDArray[np.float64]) -> NumericResult:
    return float(value) if value.ndim == 0 else value


def _prepare_inputs(
    spot: ArrayLike,
    strike: ArrayLike,
    rate: ArrayLike,
    time_to_expiry: ArrayLike,
    volatility: ArrayLike,
) -> tuple[NDArray[np.float64], ...]:
    try:
        arrays = np.broadcast_arrays(
            np.asarray(spot, dtype=float),
            np.asarray(strike, dtype=float),
            np.asarray(rate, dtype=float),
            np.asarray(time_to_expiry, dtype=float),
            np.asarray(volatility, dtype=float),
        )
    except ValueError as exc:
        raise ValueError("Greek inputs could not be broadcast to a common shape.") from exc

    spot_a, strike_a, rate_a, time_a, vol_a = arrays
    for name, values in {
        "spot": spot_a,
        "strike": strike_a,
        "rate": rate_a,
        "time_to_expiry": time_a,
        "volatility": vol_a,
    }.items():
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
    if np.any(spot_a <= 0):
        raise ValueError("Spot must be strictly positive.")
    if np.any(strike_a <= 0):
        raise ValueError("Strike must be strictly positive.")
    if np.any(time_a <= 0):
        raise ValueError("Analytical Greeks require strictly positive time to expiry.")
    if np.any(vol_a <= 0):
        raise ValueError("Analytical Greeks require strictly positive volatility.")
    return spot_a, strike_a, rate_a, time_a, vol_a


def _normal_pdf(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(-0.5 * values**2) / np.sqrt(2.0 * np.pi)


def black_scholes_greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    rate: ArrayLike,
    time_to_expiry: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType,
) -> Greeks:
    if option_type not in {"call", "put"}:
        raise ValueError("Option type must be either 'call' or 'put'.")

    s, k, r, t, v = _prepare_inputs(spot, strike, rate, time_to_expiry, volatility)
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * v**2) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t
    pdf_d1 = _normal_pdf(d1)
    disc = np.exp(-r * t)

    gamma = pdf_d1 / (s * v * sqrt_t)
    vega = s * pdf_d1 * sqrt_t
    common_theta = -(s * pdf_d1 * v) / (2.0 * sqrt_t)

    if option_type == "call":
        delta = ndtr(d1)
        theta = common_theta - r * k * disc * ndtr(d2)
        rho = k * t * disc * ndtr(d2)
    else:
        delta = ndtr(d1) - 1.0
        theta = common_theta + r * k * disc * ndtr(-d2)
        rho = -k * t * disc * ndtr(-d2)

    return Greeks(
        delta=_as_result(delta),
        gamma=_as_result(gamma),
        vega=_as_result(vega),
        theta=_as_result(theta),
        rho=_as_result(rho),
    )


def _scale(value: NumericResult, factor: float) -> NumericResult:
    return _as_result(np.asarray(value, dtype=float) * factor)


def to_market_greeks(greeks: Greeks, days_per_year: float = 365.0) -> MarketGreeks:
    if not np.isfinite(days_per_year) or days_per_year <= 0:
        raise ValueError("Days per year must be finite and strictly positive.")
    return MarketGreeks(
        delta=greeks.delta,
        gamma=greeks.gamma,
        vega_per_vol_point=_scale(greeks.vega, 0.01),
        theta_per_day=_scale(greeks.theta, 1.0 / days_per_year),
        rho_per_rate_point=_scale(greeks.rho, 0.01),
    )


def scale_position_greek(
    greek_value: NumericResult,
    position: Position,
    quantity: float = 1.0,
    contract_multiplier: float = 1.0,
) -> NumericResult:
    if position not in {"long", "short"}:
        raise ValueError("Position must be either 'long' or 'short'.")
    if not np.isfinite(quantity) or quantity <= 0:
        raise ValueError("Quantity must be finite and strictly positive.")
    if not np.isfinite(contract_multiplier) or contract_multiplier <= 0:
        raise ValueError("Contract multiplier must be finite and strictly positive.")
    sign = 1.0 if position == "long" else -1.0
    return _scale(greek_value, sign * quantity * contract_multiplier)
