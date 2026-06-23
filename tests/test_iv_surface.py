from math import exp, isclose

import numpy as np
import pytest

from options_lab.black_scholes import black_scholes_price
from options_lab.implied_volatility import (
    implied_volatility_bisection,
    implied_volatility_brent,
    implied_volatility_newton,
)
from options_lab.volatility_surface import (
    check_calendar_total_variance,
    check_call_slice_arbitrage,
    forward_price,
    generate_synthetic_surface,
    log_forward_moneyness,
    recover_surface_implied_volatility,
    synthetic_implied_volatility,
)


@pytest.mark.parametrize("solver", [implied_volatility_bisection, implied_volatility_newton, implied_volatility_brent])
@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("volatility", [0.1, 0.2, 0.5, 1.0])
def test_iv_solver_recovers_known_volatility(solver, option_type: str, volatility: float) -> None:
    price = float(black_scholes_price(100.0, 100.0, 0.05, 1.0, volatility, option_type))
    kwargs = dict(
        market_price=price, spot=100.0, strike=100.0, rate=0.05,
        time_to_expiry=1.0, option_type=option_type,
    )
    if solver is implied_volatility_newton:
        kwargs["initial_guess"] = 0.3
    result = solver(**kwargs)
    assert result.converged
    assert isclose(result.volatility, volatility, rel_tol=1e-7, abs_tol=1e-8)
    assert abs(result.residual) <= 1e-8


def test_iv_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="below"):
        implied_volatility_brent(3.0, 100.0, 100.0, 0.05, 1.0, "call")
    with pytest.raises(ValueError, match="No finite"):
        implied_volatility_brent(100.0, 100.0, 100.0, 0.05, 1.0, "call")
    lower = 100.0 - 100.0 * exp(-0.05)
    assert implied_volatility_brent(lower, 100.0, 100.0, 0.05, 1.0, "call").volatility == 0.0


def test_surface_construction_recovery_and_checks() -> None:
    fwd = forward_price(100.0, 0.03, 1.0)
    assert isclose(log_forward_moneyness(fwd, fwd), 0.0, abs_tol=1e-12)
    assert synthetic_implied_volatility(-0.2, 1.0) > synthetic_implied_volatility(0.2, 1.0)
    surface = generate_synthetic_surface(
        100.0, 0.03, np.linspace(-0.4, 0.4, 17), [30 / 365, 90 / 365, 180 / 365, 1.0, 2.0]
    )
    assert len(surface) == 85
    recovered = recover_surface_implied_volatility(surface, "call_price", "call")
    assert recovered["iv_converged"].all()
    assert recovered["iv_absolute_error"].max() < 1e-8
    assert check_call_slice_arbitrage(surface)["all_checks_passed"].all()
    assert check_calendar_total_variance(surface)["calendar_check_passed"].all()
