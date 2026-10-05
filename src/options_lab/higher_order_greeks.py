"""Analytical Black-Scholes vanna and volga for European options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


OptionType = Literal["call", "put"]
NumericResult = float | NDArray[np.float64]


@dataclass(frozen=True)
class HigherOrderGreeks:
    """Raw higher-order Black-Scholes sensitivities."""

    vanna: NumericResult
    volga: NumericResult


@dataclass(frozen=True)
class MarketHigherOrderGreeks:
    """Higher-order Greeks expressed in volatility-point units."""

    vanna_per_vol_point: NumericResult
    volga_per_vol_point_squared: NumericResult


def _as_result(
    value: NDArray[np.float64],
) -> NumericResult:
    return (
        float(value)
        if value.ndim == 0
        else value
    )


def _prepare_inputs(
    spot: ArrayLike,
    strike: ArrayLike,
    rate: ArrayLike,
    time_to_expiry: ArrayLike,
    volatility: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    ...,
]:
    try:
        arrays = np.broadcast_arrays(
            np.asarray(
                spot,
                dtype=float,
            ),
            np.asarray(
                strike,
                dtype=float,
            ),
            np.asarray(
                rate,
                dtype=float,
            ),
            np.asarray(
                time_to_expiry,
                dtype=float,
            ),
            np.asarray(
                volatility,
                dtype=float,
            ),
        )

    except ValueError as exc:
        raise ValueError(
            "Higher-order Greek inputs could not be "
            "broadcast to a common shape."
        ) from exc

    (
        spot_a,
        strike_a,
        rate_a,
        time_a,
        vol_a,
    ) = arrays

    for name, values in {
        "spot": spot_a,
        "strike": strike_a,
        "rate": rate_a,
        "time_to_expiry": time_a,
        "volatility": vol_a,
    }.items():
        if np.any(
            ~np.isfinite(values)
        ):
            raise ValueError(
                f"{name} must contain only finite values."
            )

    if np.any(
        spot_a <= 0
    ):
        raise ValueError(
            "Spot must be strictly positive."
        )

    if np.any(
        strike_a <= 0
    ):
        raise ValueError(
            "Strike must be strictly positive."
        )

    if np.any(
        time_a <= 0
    ):
        raise ValueError(
            "Vanna and volga require strictly positive "
            "time to expiry."
        )

    if np.any(
        vol_a <= 0
    ):
        raise ValueError(
            "Vanna and volga require strictly positive "
            "volatility."
        )

    return (
        spot_a,
        strike_a,
        rate_a,
        time_a,
        vol_a,
    )


def _normal_pdf(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    return (
        np.exp(
            -0.5 * values**2
        )
        / np.sqrt(
            2.0 * np.pi
        )
    )


def black_scholes_higher_order_greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    rate: ArrayLike,
    time_to_expiry: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType,
) -> HigherOrderGreeks:
    """Return analytical vanna and volga/vomma.

    The Black-Scholes baseline in this project assumes no dividends.

    Vanna is

        d²V / (dS dσ)

    and volga/vomma is

        d²V / dσ².

    For European calls and puts under the same Black-Scholes assumptions,
    these two sensitivities are identical because the non-option terms in
    put-call parity contain no volatility dependence.
    """

    if option_type not in {
        "call",
        "put",
    }:
        raise ValueError(
            "Option type must be either 'call' or 'put'."
        )

    (
        s,
        k,
        r,
        t,
        v,
    ) = _prepare_inputs(
        spot,
        strike,
        rate,
        time_to_expiry,
        volatility,
    )

    sqrt_t = np.sqrt(t)

    d1 = (
        np.log(
            s / k
        )
        + (
            r
            + 0.5 * v**2
        )
        * t
    ) / (
        v * sqrt_t
    )

    d2 = (
        d1
        - v * sqrt_t
    )

    pdf_d1 = _normal_pdf(
        d1
    )

    vega = (
        s
        * pdf_d1
        * sqrt_t
    )

    vanna = (
        -pdf_d1
        * d2
        / v
    )

    volga = (
        vega
        * d1
        * d2
        / v
    )

    return HigherOrderGreeks(
        vanna=_as_result(
            vanna
        ),
        volga=_as_result(
            volga
        ),
    )


def to_market_higher_order_greeks(
    greeks: HigherOrderGreeks,
) -> MarketHigherOrderGreeks:
    """Convert decimal-volatility derivatives to volatility-point units.

    A one-volatility-point move is 0.01 in decimal volatility.

    Therefore:
        vanna per vol point = raw vanna * 0.01
        volga per vol-point squared = raw volga * 0.01²
    """

    vanna = (
        np.asarray(
            greeks.vanna,
            dtype=float,
        )
        * 0.01
    )

    volga = (
        np.asarray(
            greeks.volga,
            dtype=float,
        )
        * 0.01**2
    )

    return MarketHigherOrderGreeks(
        vanna_per_vol_point=_as_result(
            vanna
        ),
        volga_per_vol_point_squared=_as_result(
            volga
        ),
    )
