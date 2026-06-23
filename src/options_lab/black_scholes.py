"""Black-Scholes pricing for non-dividend-paying European options."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

OptionType = Literal["call", "put"]
NumericResult = float | NDArray[np.float64]


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
        raise ValueError("Inputs could not be broadcast to a common shape.") from exc

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
    if np.any(time_a < 0):
        raise ValueError("Time to expiry cannot be negative.")
    if np.any(vol_a < 0):
        raise ValueError("Volatility cannot be negative.")
    return spot_a, strike_a, rate_a, time_a, vol_a


def black_scholes_d1_d2(
    spot: ArrayLike,
    strike: ArrayLike,
    rate: ArrayLike,
    time_to_expiry: ArrayLike,
    volatility: ArrayLike,
) -> tuple[NumericResult, NumericResult]:
    spot_a, strike_a, rate_a, time_a, vol_a = _prepare_inputs(
        spot, strike, rate, time_to_expiry, volatility
    )
    if np.any(time_a <= 0):
        raise ValueError("d1 and d2 require strictly positive time to expiry.")
    if np.any(vol_a <= 0):
        raise ValueError("d1 and d2 require strictly positive volatility.")

    sqrt_time = np.sqrt(time_a)
    d1 = (
        np.log(spot_a / strike_a)
        + (rate_a + 0.5 * vol_a**2) * time_a
    ) / (vol_a * sqrt_time)
    d2 = d1 - vol_a * sqrt_time
    return _as_result(d1), _as_result(d2)


def black_scholes_price(
    spot: ArrayLike,
    strike: ArrayLike,
    rate: ArrayLike,
    time_to_expiry: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType,
) -> NumericResult:
    if option_type not in {"call", "put"}:
        raise ValueError("Option type must be either 'call' or 'put'.")

    spot_a, strike_a, rate_a, time_a, vol_a = _prepare_inputs(
        spot, strike, rate, time_to_expiry, volatility
    )
    result = np.empty_like(spot_a, dtype=float)

    expired = time_a == 0
    deterministic = (time_a > 0) & (vol_a == 0)
    regular = (time_a > 0) & (vol_a > 0)

    if np.any(expired):
        if option_type == "call":
            result[expired] = np.maximum(spot_a[expired] - strike_a[expired], 0.0)
        else:
            result[expired] = np.maximum(strike_a[expired] - spot_a[expired], 0.0)

    if np.any(deterministic):
        pv_strike = strike_a[deterministic] * np.exp(-rate_a[deterministic] * time_a[deterministic])
        if option_type == "call":
            result[deterministic] = np.maximum(spot_a[deterministic] - pv_strike, 0.0)
        else:
            result[deterministic] = np.maximum(pv_strike - spot_a[deterministic], 0.0)

    if np.any(regular):
        s = spot_a[regular]
        k = strike_a[regular]
        r = rate_a[regular]
        t = time_a[regular]
        v = vol_a[regular]
        sqrt_t = np.sqrt(t)
        d1 = (np.log(s / k) + (r + 0.5 * v**2) * t) / (v * sqrt_t)
        d2 = d1 - v * sqrt_t
        pv_k = k * np.exp(-r * t)
        if option_type == "call":
            result[regular] = s * ndtr(d1) - pv_k * ndtr(d2)
        else:
            result[regular] = pv_k * ndtr(-d2) - s * ndtr(-d1)

    return _as_result(result)
