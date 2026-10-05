"""Finite-difference validation for Black-Scholes vanna and volga."""

from __future__ import annotations

from math import isfinite
from typing import Literal

from .black_scholes import (
    black_scholes_price,
)
from .higher_order_greeks import (
    HigherOrderGreeks,
)


OptionType = Literal["call", "put"]


def numerical_higher_order_greeks(
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    volatility: float,
    option_type: OptionType,
    spot_bump: float | None = None,
    volatility_bump: float = 1e-4,
) -> HigherOrderGreeks:
    """Estimate vanna and volga directly from option prices.

    Vanna is obtained from a mixed central finite difference in spot and
    volatility. Volga is obtained from a second central difference in
    volatility.

    This function deliberately differentiates option prices rather than
    analytical first-order Greeks so that it serves as an independent
    check of the closed-form formulas.
    """

    for name, value in {
        "spot": spot,
        "strike": strike,
        "rate": rate,
        "time_to_expiry": time_to_expiry,
        "volatility": volatility,
        "volatility_bump": volatility_bump,
    }.items():
        if not isfinite(
            value
        ):
            raise ValueError(
                f"{name} must be finite."
            )

    if spot <= 0:
        raise ValueError(
            "Spot must be strictly positive."
        )

    if strike <= 0:
        raise ValueError(
            "Strike must be strictly positive."
        )

    if time_to_expiry <= 0:
        raise ValueError(
            "Numerical higher-order Greeks require "
            "strictly positive time to expiry."
        )

    if volatility <= 0:
        raise ValueError(
            "Numerical higher-order Greeks require "
            "strictly positive volatility."
        )

    if option_type not in {
        "call",
        "put",
    }:
        raise ValueError(
            "Option type must be either 'call' or 'put'."
        )

    if volatility_bump <= 0:
        raise ValueError(
            "Volatility bump must be strictly positive."
        )

    if (
        volatility
        - volatility_bump
        <= 0
    ):
        raise ValueError(
            "Volatility bump is too large relative to volatility."
        )

    if spot_bump is None:
        spot_bump = max(
            1e-4 * spot,
            1e-4,
        )

    if (
        not isfinite(
            spot_bump
        )
        or spot_bump <= 0
    ):
        raise ValueError(
            "Spot bump must be finite and strictly positive."
        )

    if (
        spot
        - spot_bump
        <= 0
    ):
        raise ValueError(
            "Spot bump is too large relative to spot."
        )

    def price(
        s: float,
        sigma: float,
    ) -> float:
        return float(
            black_scholes_price(
                spot=s,
                strike=strike,
                rate=rate,
                time_to_expiry=time_to_expiry,
                volatility=sigma,
                option_type=option_type,
            )
        )

    s_up = (
        spot
        + spot_bump
    )

    s_down = (
        spot
        - spot_bump
    )

    vol_up = (
        volatility
        + volatility_bump
    )

    vol_down = (
        volatility
        - volatility_bump
    )

    price_up_up = price(
        s_up,
        vol_up,
    )

    price_up_down = price(
        s_up,
        vol_down,
    )

    price_down_up = price(
        s_down,
        vol_up,
    )

    price_down_down = price(
        s_down,
        vol_down,
    )

    vanna = (
        price_up_up
        - price_up_down
        - price_down_up
        + price_down_down
    ) / (
        4.0
        * spot_bump
        * volatility_bump
    )

    center = price(
        spot,
        volatility,
    )

    volga = (
        price(
            spot,
            vol_up,
        )
        - 2.0 * center
        + price(
            spot,
            vol_down,
        )
    ) / (
        volatility_bump**2
    )

    return HigherOrderGreeks(
        vanna=float(
            vanna
        ),
        volga=float(
            volga
        ),
    )
