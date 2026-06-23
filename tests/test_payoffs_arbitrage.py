from math import isclose

import numpy as np
import pytest

from options_lab.arbitrage import (
    european_option_bounds,
    parity_implied_put_price,
    parity_signal,
    present_value,
    price_within_bounds,
    put_call_parity_gap,
)
from options_lab.payoffs import call_payoff, option_profit, put_payoff


def test_payoffs_and_profit() -> None:
    spots = np.array([80.0, 100.0, 105.0, 120.0])
    np.testing.assert_allclose(call_payoff(spots, 100.0), [0.0, 0.0, 5.0, 20.0])
    np.testing.assert_allclose(put_payoff(spots, 100.0), [20.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(option_profit(call_payoff(spots, 100.0), 6.0, "long"), [-6, -6, -1, 14])


def test_long_and_short_profit_are_opposites() -> None:
    payoff = call_payoff(np.array([80.0, 100.0, 120.0]), 100.0)
    np.testing.assert_allclose(
        option_profit(payoff, 6.0, "short"),
        -np.asarray(option_profit(payoff, 6.0, "long")),
    )


def test_bounds_and_parity() -> None:
    assert isclose(present_value(100.0, 0.05, 1.0), 95.1229424500714, rel_tol=1e-12)
    call_bounds = european_option_bounds(100.0, 100.0, 0.05, 1.0, "call")
    put_bounds = european_option_bounds(100.0, 100.0, 0.05, 1.0, "put")
    assert isclose(call_bounds.lower, 4.8770575499286, rel_tol=1e-12)
    assert call_bounds.upper == 100.0
    assert put_bounds.lower == 0.0
    assert isclose(put_bounds.upper, 95.1229424500714, rel_tol=1e-12)
    implied_put = parity_implied_put_price(10.0, 100.0, 100.0, 0.05, 1.0)
    assert isclose(implied_put, 5.1229424500714, rel_tol=1e-12)
    assert isclose(put_call_parity_gap(10.0, implied_put, 100.0, 100.0, 0.05, 1.0), 0.0, abs_tol=1e-12)
    assert parity_signal(10.0, 3.0, 100.0, 100.0, 0.05, 1.0) == "fiduciary_call_overpriced"
    assert price_within_bounds(10.0, call_bounds)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        call_payoff(100.0, -1.0)
    with pytest.raises(ValueError):
        option_profit(1.0, -1.0)
    with pytest.raises(ValueError):
        european_option_bounds(100.0, 100.0, 0.05, -1.0, "call")
