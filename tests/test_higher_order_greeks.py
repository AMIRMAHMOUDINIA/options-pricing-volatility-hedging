from math import isclose

import pytest

from options_lab.higher_order_greeks import (
    black_scholes_higher_order_greeks,
    to_market_higher_order_greeks,
)
from options_lab.numerical_higher_order_greeks import (
    numerical_higher_order_greeks,
)


def test_known_vanna_and_volga_values() -> None:
    greeks = (
        black_scholes_higher_order_greeks(
            100.0,
            100.0,
            0.05,
            1.0,
            0.20,
            "call",
        )
    )

    assert isclose(
        greeks.vanna,
        -0.28143026018770345,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    assert isclose(
        greeks.volga,
        9.850059106569622,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    "option_type",
    [
        "call",
        "put",
    ],
)
@pytest.mark.parametrize(
    "strike",
    [
        80.0,
        100.0,
        120.0,
    ],
)
def test_analytical_and_numerical_higher_order_greeks_agree(
    option_type: str,
    strike: float,
) -> None:
    analytical = (
        black_scholes_higher_order_greeks(
            100.0,
            strike,
            0.03,
            0.75,
            0.25,
            option_type,
        )
    )

    numerical = (
        numerical_higher_order_greeks(
            100.0,
            strike,
            0.03,
            0.75,
            0.25,
            option_type,
        )
    )

    assert isclose(
        analytical.vanna,
        numerical.vanna,
        rel_tol=2e-5,
        abs_tol=2e-5,
    )

    assert isclose(
        analytical.volga,
        numerical.volga,
        rel_tol=2e-5,
        abs_tol=2e-4,
    )


def test_call_and_put_vanna_volga_match() -> None:
    call = (
        black_scholes_higher_order_greeks(
            100.0,
            110.0,
            0.04,
            0.6,
            0.30,
            "call",
        )
    )

    put = (
        black_scholes_higher_order_greeks(
            100.0,
            110.0,
            0.04,
            0.6,
            0.30,
            "put",
        )
    )

    assert isclose(
        call.vanna,
        put.vanna,
        rel_tol=0.0,
        abs_tol=1e-14,
    )

    assert isclose(
        call.volga,
        put.volga,
        rel_tol=0.0,
        abs_tol=1e-14,
    )


def test_market_unit_conversion() -> None:
    raw = (
        black_scholes_higher_order_greeks(
            100.0,
            100.0,
            0.05,
            1.0,
            0.20,
            "call",
        )
    )

    market = (
        to_market_higher_order_greeks(
            raw
        )
    )

    assert isclose(
        market.vanna_per_vol_point,
        raw.vanna * 0.01,
        rel_tol=1e-12,
    )

    assert isclose(
        market.volga_per_vol_point_squared,
        raw.volga * 0.0001,
        rel_tol=1e-12,
    )


def test_invalid_analytical_boundaries_raise() -> None:
    with pytest.raises(
        ValueError
    ):
        black_scholes_higher_order_greeks(
            100.0,
            100.0,
            0.05,
            0.0,
            0.20,
            "call",
        )

    with pytest.raises(
        ValueError
    ):
        black_scholes_higher_order_greeks(
            100.0,
            100.0,
            0.05,
            1.0,
            0.0,
            "call",
        )


def test_invalid_numerical_bump_raises() -> None:
    with pytest.raises(
        ValueError
    ):
        numerical_higher_order_greeks(
            100.0,
            100.0,
            0.05,
            1.0,
            0.20,
            "call",
            volatility_bump=0.25,
        )
