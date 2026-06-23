"""Exact and approximate P&L attribution for dynamically hedged options."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .black_scholes import black_scholes_price
from .delta_hedging import HedgePathResult, delta_hedge_path
from .greeks import black_scholes_greeks
from .payoffs import call_payoff, put_payoff

OptionType = Literal["call", "put"]
Position = Literal["long", "short"]


@dataclass(frozen=True)
class AccountingAttribution:
    option_entry_cashflow: float
    stock_trading_cashflow: float
    option_settlement_cashflow: float
    financing_cashflow: float
    transaction_cost_pnl: float
    reconstructed_terminal_pnl: float
    reported_terminal_pnl: float
    reconciliation_error: float


@dataclass(frozen=True)
class CounterfactualAttribution:
    model_premium_per_unit: float
    actual_premium_per_unit: float
    model_hedging_pnl: float
    premium_edge_at_inception: float
    premium_edge_at_expiry: float
    transaction_cost_drag_at_expiry: float
    nominal_transaction_cost: float
    transaction_cost_future_value: float
    attributed_terminal_pnl: float
    actual_terminal_pnl: float
    reconstruction_error: float


@dataclass(frozen=True)
class GreekPathAttribution:
    intervals: pd.DataFrame
    initial_model_mark_to_market: float
    exact_interval_pnl: float
    greek_explained_interval_pnl: float
    taylor_residual: float
    reconstructed_terminal_pnl: float
    reported_terminal_pnl: float
    reconciliation_error: float


@dataclass(frozen=True)
class VarianceAttribution:
    intervals: pd.DataFrame
    total_variance_pnl: float
    model_hedging_pnl: float
    residual_pnl: float


def attribute_cash_ledger(result: HedgePathResult) -> AccountingAttribution:
    history = result.history
    required = {
        "option_cashflow", "stock_trade_cashflow", "option_settlement",
        "interest_accrual", "transaction_cost",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError("Hedge history is missing columns: " + ", ".join(sorted(missing)))

    option_entry = float(history["option_cashflow"].sum())
    stock_cash = float(history["stock_trade_cashflow"].sum())
    settlement = float(history["option_settlement"].sum())
    financing = float(history["interest_accrual"].sum())
    cost_pnl = -float(history["transaction_cost"].sum())
    reconstructed = option_entry + stock_cash + settlement + financing + cost_pnl
    reported = float(result.terminal_pnl)
    return AccountingAttribution(
        option_entry, stock_cash, settlement, financing, cost_pnl,
        reconstructed, reported, reported - reconstructed,
    )


def counterfactual_pnl_attribution(
    time_grid: ArrayLike,
    spot_path: ArrayLike,
    strike: float,
    rate: float,
    implied_volatility: float,
    option_type: OptionType,
    position: Position,
    quantity: float = 1.0,
    contract_multiplier: float = 1.0,
    transaction_cost_rate: float = 0.0,
    actual_option_premium: float | None = None,
    rebalance_indices: tuple[int, ...] | None = None,
) -> CounterfactualAttribution:
    times = np.asarray(time_grid, dtype=float)
    spots = np.asarray(spot_path, dtype=float)
    if times.ndim != 1 or spots.ndim != 1 or len(times) != len(spots):
        raise ValueError("Time grid and spot path must be equal-length one-dimensional arrays.")
    maturity = float(times[-1])
    model_premium = float(black_scholes_price(
        spots[0], strike, rate, maturity, implied_volatility, option_type
    ))
    if actual_option_premium is None:
        actual_premium = model_premium
    else:
        if not isfinite(actual_option_premium) or actual_option_premium < 0:
            raise ValueError("Actual option premium must be finite and non-negative.")
        actual_premium = float(actual_option_premium)

    common = dict(
        time_grid=times,
        spot_path=spots,
        strike=strike,
        rate=rate,
        implied_volatility=implied_volatility,
        option_type=option_type,
        position=position,
        quantity=quantity,
        contract_multiplier=contract_multiplier,
        rebalance_indices=rebalance_indices,
    )
    model_no_cost = delta_hedge_path(
        **common, transaction_cost_rate=0.0, option_premium=model_premium
    )
    actual_no_cost = delta_hedge_path(
        **common, transaction_cost_rate=0.0, option_premium=actual_premium
    )
    actual_with_cost = delta_hedge_path(
        **common, transaction_cost_rate=transaction_cost_rate, option_premium=actual_premium
    )

    sign = 1.0 if position == "long" else -1.0
    units = quantity * contract_multiplier
    inception_edge = -sign * (actual_premium - model_premium) * units
    terminal_edge = actual_no_cost.terminal_pnl - model_no_cost.terminal_pnl
    cost_drag = actual_with_cost.terminal_pnl - actual_no_cost.terminal_pnl
    history = actual_with_cost.history
    nominal_cost = float(history["transaction_cost"].sum())
    future_value_cost = float(np.sum(
        history["transaction_cost"].to_numpy(float)
        * np.exp(rate * (maturity - history["time"].to_numpy(float)))
    ))
    attributed = model_no_cost.terminal_pnl + terminal_edge + cost_drag
    actual = actual_with_cost.terminal_pnl
    return CounterfactualAttribution(
        model_premium_per_unit=model_premium,
        actual_premium_per_unit=actual_premium,
        model_hedging_pnl=model_no_cost.terminal_pnl,
        premium_edge_at_inception=inception_edge,
        premium_edge_at_expiry=terminal_edge,
        transaction_cost_drag_at_expiry=cost_drag,
        nominal_transaction_cost=nominal_cost,
        transaction_cost_future_value=future_value_cost,
        attributed_terminal_pnl=attributed,
        actual_terminal_pnl=actual,
        reconstruction_error=actual - attributed,
    )


def _payoff(spot: float, strike: float, option_type: OptionType) -> float:
    return float(call_payoff(spot, strike) if option_type == "call" else put_payoff(spot, strike))


def greek_path_attribution(
    result: HedgePathResult,
    strike: float,
    rate: float,
    implied_volatility: float,
    option_type: OptionType,
    position: Position,
    quantity: float = 1.0,
    contract_multiplier: float = 1.0,
) -> GreekPathAttribution:
    history = result.history.reset_index(drop=True)
    sign = 1.0 if position == "long" else -1.0
    units = quantity * contract_multiplier
    maturity = float(history["time"].iloc[-1])
    model_initial = float(black_scholes_price(
        float(history["spot"].iloc[0]), strike, rate, maturity, implied_volatility, option_type
    ))
    initial_model_mtm = (
        float(history["cash_account"].iloc[0])
        + float(history["stock_market_value"].iloc[0])
        + sign * model_initial * units
    )

    rows = []
    for i in range(len(history) - 1):
        current, following = history.iloc[i], history.iloc[i + 1]
        t0, t1 = float(current["time"]), float(following["time"])
        dt = t1 - t0
        remaining = maturity - t0
        s0, s1 = float(current["spot"]), float(following["spot"])
        ds = s1 - s0
        v0 = float(black_scholes_price(s0, strike, rate, remaining, implied_volatility, option_type))
        v1 = _payoff(s1, strike, option_type) if i + 1 == len(history) - 1 else float(
            black_scholes_price(s1, strike, rate, maturity - t1, implied_volatility, option_type)
        )
        greeks = black_scholes_greeks(s0, strike, rate, remaining, implied_volatility, option_type)
        delta, gamma, theta = float(greeks.delta), float(greeks.gamma), float(greeks.theta)
        exact_option = sign * units * (v1 - v0)
        delta_pnl = sign * units * delta * ds
        gamma_pnl = sign * units * 0.5 * gamma * ds**2
        theta_pnl = sign * units * theta * dt
        stock_position = float(current["actual_stock_position"])
        stock_pnl = stock_position * ds
        financing = float(following["interest_accrual"])
        cost_pnl = -float(following["transaction_cost"])
        exact_portfolio = exact_option + stock_pnl + financing + cost_pnl
        explained = delta_pnl + gamma_pnl + theta_pnl + stock_pnl + financing + cost_pnl
        rows.append({
            "interval_start": t0,
            "interval_end": t1,
            "dt": dt,
            "spot_start": s0,
            "spot_end": s1,
            "spot_change": ds,
            "model_option_value_start": v0,
            "model_option_value_end": v1,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "actual_stock_position": stock_position,
            "exact_option_pnl": exact_option,
            "option_delta_pnl": delta_pnl,
            "option_gamma_pnl": gamma_pnl,
            "option_theta_pnl": theta_pnl,
            "stock_pnl": stock_pnl,
            "delta_hedge_mismatch_pnl": delta_pnl + stock_pnl,
            "financing_pnl": financing,
            "transaction_cost_pnl": cost_pnl,
            "exact_portfolio_pnl": exact_portfolio,
            "greek_explained_pnl": explained,
            "taylor_residual": exact_portfolio - explained,
        })
    intervals = pd.DataFrame(rows)
    exact_total = float(intervals["exact_portfolio_pnl"].sum())
    explained_total = float(intervals["greek_explained_pnl"].sum())
    residual = float(intervals["taylor_residual"].sum())
    reconstructed = initial_model_mtm + exact_total
    reported = float(result.terminal_pnl)
    return GreekPathAttribution(
        intervals, initial_model_mtm, exact_total, explained_total, residual,
        reconstructed, reported, reported - reconstructed,
    )


def gamma_variance_attribution(
    result: HedgePathResult,
    strike: float,
    rate: float,
    implied_volatility: float,
    option_type: OptionType,
    position: Position,
    quantity: float = 1.0,
    contract_multiplier: float = 1.0,
    tolerance: float = 1e-10,
) -> VarianceAttribution:
    history = result.history.reset_index(drop=True)
    if not history.iloc[:-1]["is_rebalance"].all():
        raise ValueError("Variance attribution requires rebalancing every interval.")
    if float(history["transaction_cost"].sum()) > tolerance:
        raise ValueError("Variance attribution requires zero transaction costs.")

    sign = 1.0 if position == "long" else -1.0
    units = quantity * contract_multiplier
    maturity = float(history["time"].iloc[-1])
    model_premium = float(black_scholes_price(
        float(history["spot"].iloc[0]), strike, rate, maturity, implied_volatility, option_type
    ))
    initial_model_value = (
        float(history["cash_account"].iloc[0])
        + float(history["stock_market_value"].iloc[0])
        + sign * model_premium * units
    )
    if abs(initial_model_value) > tolerance:
        raise ValueError("Variance attribution requires entry at model premium.")

    rows = []
    for i in range(len(history) - 1):
        current, following = history.iloc[i], history.iloc[i + 1]
        t0, t1 = float(current["time"]), float(following["time"])
        dt = t1 - t0
        s0, s1 = float(current["spot"]), float(following["spot"])
        gamma = float(black_scholes_greeks(
            s0, strike, rate, maturity - t0, implied_volatility, option_type
        ).gamma)
        log_return = float(np.log(s1 / s0))
        realized_increment = log_return**2
        implied_increment = implied_volatility**2 * dt
        spread = realized_increment - implied_increment
        pnl = 0.5 * sign * units * gamma * s0**2 * spread
        rows.append({
            "interval_start": t0,
            "interval_end": t1,
            "dt": dt,
            "spot_start": s0,
            "spot_end": s1,
            "gamma": gamma,
            "gamma_dollar_exposure": gamma * s0**2,
            "realized_variance_increment": realized_increment,
            "implied_variance_increment": implied_increment,
            "variance_spread_increment": spread,
            "variance_pnl": pnl,
        })
    intervals = pd.DataFrame(rows)
    total = float(intervals["variance_pnl"].sum())
    model_pnl = float(result.terminal_pnl)
    return VarianceAttribution(intervals, total, model_pnl, model_pnl - total)
