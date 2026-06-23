from math import exp, isclose

import numpy as np

from options_lab.black_scholes import black_scholes_price
from options_lab.delta_hedging import delta_hedge_path
from options_lab.greeks import black_scholes_greeks
from options_lab.pnl_attribution import (
    attribute_cash_ledger,
    counterfactual_pnl_attribution,
    gamma_variance_attribution,
    greek_path_attribution,
)
from options_lab.price_paths import simulate_gbm_paths


def test_gbm_shape_and_reproducibility() -> None:
    t1, p1 = simulate_gbm_paths(100, 0.05, 0.2, 1.0, 10, 3, seed=42)
    t2, p2 = simulate_gbm_paths(100, 0.05, 0.2, 1.0, 10, 3, seed=42)
    assert t1.shape == (11,)
    assert p1.shape == (3, 11)
    np.testing.assert_allclose(t1, t2)
    np.testing.assert_allclose(p1, p2)


def test_single_period_accounting() -> None:
    s0, st, k, r, t, vol = 100.0, 120.0, 100.0, 0.05, 1.0, 0.2
    premium = float(black_scholes_price(s0, k, r, t, vol, "call"))
    delta = float(black_scholes_greeks(s0, k, r, t, vol, "call").delta)
    expected = (premium - delta * s0) * exp(r * t) + delta * st - max(st - k, 0.0)
    result = delta_hedge_path(np.array([0.0, 1.0]), np.array([s0, st]), k, r, vol, "call", "short")
    assert isclose(result.terminal_pnl, expected, rel_tol=1e-12, abs_tol=1e-12)
    assert isclose(result.history["portfolio_value"].iloc[0], 0.0, abs_tol=1e-12)


def test_long_short_symmetry_and_cost_drag() -> None:
    times = np.linspace(0.0, 1.0, 13)
    path = np.array([100, 102, 101, 104, 108, 106, 110, 107, 112, 115, 113, 118, 120], float)
    long = delta_hedge_path(times, path, 100, 0.05, 0.2, "call", "long")
    short = delta_hedge_path(times, path, 100, 0.05, 0.2, "call", "short")
    assert isclose(long.terminal_pnl, -short.terminal_pnl, abs_tol=1e-10)
    costly = delta_hedge_path(times, path, 100, 0.05, 0.2, "call", "short", transaction_cost_rate=0.001)
    assert costly.total_transaction_cost > 0
    assert costly.terminal_pnl < short.terminal_pnl


def test_exact_attributions_reconcile() -> None:
    times = np.linspace(0.0, 1.0, 53)
    path = np.linspace(100.0, 110.0, 53)
    result = delta_hedge_path(times, path, 100, 0.05, 0.2, "call", "short", transaction_cost_rate=0.001)
    ledger = attribute_cash_ledger(result)
    assert isclose(ledger.reconciliation_error, 0.0, abs_tol=1e-10)

    cf = counterfactual_pnl_attribution(
        times, path, 100, 0.05, 0.2, "call", "short",
        transaction_cost_rate=0.001, actual_option_premium=11.0,
    )
    assert isclose(cf.reconstruction_error, 0.0, abs_tol=1e-10)
    assert isclose(cf.premium_edge_at_expiry, cf.premium_edge_at_inception * exp(0.05), rel_tol=1e-10)
    assert isclose(cf.transaction_cost_drag_at_expiry, -cf.transaction_cost_future_value, rel_tol=1e-10)


def test_greek_and_variance_attribution_run() -> None:
    times, paths = simulate_gbm_paths(100, 0.05, 0.2, 1.0, 252, 1, seed=3)
    result = delta_hedge_path(times, paths[0], 100, 0.05, 0.2, "call", "short")
    greek = greek_path_attribution(result, 100, 0.05, 0.2, "call", "short")
    assert isclose(greek.reconciliation_error, 0.0, abs_tol=1e-8)
    variance = gamma_variance_attribution(result, 100, 0.05, 0.2, "call", "short")
    assert len(variance.intervals) == 252
    assert np.isfinite(variance.total_variance_pnl)
