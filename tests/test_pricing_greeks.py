from math import isclose

import numpy as np
import pytest

from options_lab.arbitrage import european_option_bounds, price_within_bounds, put_call_parity_gap
from options_lab.black_scholes import black_scholes_d1_d2, black_scholes_price
from options_lab.greeks import black_scholes_greeks, scale_position_greek, to_market_greeks
from options_lab.numerical_greeks import numerical_greeks


def test_known_black_scholes_values() -> None:
    d1, d2 = black_scholes_d1_d2(100.0, 100.0, 0.05, 1.0, 0.20)
    call = black_scholes_price(100.0, 100.0, 0.05, 1.0, 0.20, "call")
    put = black_scholes_price(100.0, 100.0, 0.05, 1.0, 0.20, "put")
    assert isclose(d1, 0.35, rel_tol=1e-12)
    assert isclose(d2, 0.15, rel_tol=1e-12)
    assert isclose(call, 10.450583572185565, rel_tol=1e-12)
    assert isclose(put, 5.573526022256971, rel_tol=1e-12)
    assert isclose(put_call_parity_gap(call, put, 100.0, 100.0, 0.05, 1.0), 0.0, abs_tol=1e-12)
    assert price_within_bounds(call, european_option_bounds(100.0, 100.0, 0.05, 1.0, "call"))


def test_boundaries_and_vectorization() -> None:
    assert black_scholes_price(120.0, 100.0, 0.05, 0.0, 0.2, "call") == 20.0
    expected = max(100.0 - 100.0 * np.exp(-0.05), 0.0)
    assert isclose(black_scholes_price(100.0, 100.0, 0.05, 1.0, 0.0, "call"), expected)
    spots = np.array([80.0, 100.0, 120.0])
    calls = np.asarray(black_scholes_price(spots, 100.0, 0.05, 1.0, 0.2, "call"))
    puts = np.asarray(black_scholes_price(spots, 100.0, 0.05, 1.0, 0.2, "put"))
    assert np.all(np.diff(calls) > 0)
    assert np.all(np.diff(puts) < 0)


def test_known_greeks_and_units() -> None:
    call = black_scholes_greeks(100.0, 100.0, 0.05, 1.0, 0.20, "call")
    put = black_scholes_greeks(100.0, 100.0, 0.05, 1.0, 0.20, "put")
    assert isclose(call.delta, 0.6368306511756191, rel_tol=1e-12)
    assert isclose(call.gamma, 0.018762017345846895, rel_tol=1e-12)
    assert isclose(call.vega, 37.52403469169379, rel_tol=1e-12)
    assert isclose(call.theta, -6.414027546438197, rel_tol=1e-12)
    assert isclose(call.rho, 53.232481545376345, rel_tol=1e-12)
    assert isclose(call.delta - put.delta, 1.0, abs_tol=1e-12)
    assert isclose(call.gamma, put.gamma, abs_tol=1e-12)
    assert isclose(call.vega, put.vega, abs_tol=1e-12)
    market = to_market_greeks(call)
    assert isclose(market.vega_per_vol_point, call.vega / 100.0)
    assert isclose(market.theta_per_day, call.theta / 365.0)
    assert isclose(scale_position_greek(call.delta, "long", 1, 100), 63.68306511756191)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_analytical_and_numerical_greeks_agree(option_type: str) -> None:
    analytical = black_scholes_greeks(100.0, 100.0, 0.05, 1.0, 0.20, option_type)
    numerical = numerical_greeks(100.0, 100.0, 0.05, 1.0, 0.20, option_type)
    assert isclose(analytical.delta, numerical.delta, rel_tol=1e-6, abs_tol=1e-8)
    assert isclose(analytical.gamma, numerical.gamma, rel_tol=1e-5, abs_tol=1e-7)
    assert isclose(analytical.vega, numerical.vega, rel_tol=1e-6, abs_tol=1e-6)
    assert isclose(analytical.theta, numerical.theta, rel_tol=1e-6, abs_tol=1e-6)
    assert isclose(analytical.rho, numerical.rho, rel_tol=1e-6, abs_tol=1e-6)


def test_invalid_greek_boundary_raises() -> None:
    with pytest.raises(ValueError):
        black_scholes_greeks(100.0, 100.0, 0.05, 0.0, 0.2, "call")
