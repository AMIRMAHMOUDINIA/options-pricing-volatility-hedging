"""Self-financing dynamic delta hedging for one European option position."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .black_scholes import black_scholes_price
from .greeks import black_scholes_greeks
from .metrics import realized_volatility_from_path
from .payoffs import call_payoff, put_payoff

OptionType = Literal["call", "put"]
Position = Literal["long", "short"]


@dataclass(frozen=True)
class HedgePathResult:
    history: pd.DataFrame
    terminal_pnl: float
    option_premium_per_unit: float
    option_payoff_per_unit: float
    total_transaction_cost: float
    number_of_rebalances: int


def _terminal_payoff(terminal_spot: float, strike: float, option_type: OptionType) -> float:
    if option_type == "call":
        return float(call_payoff(terminal_spot, strike))
    if option_type == "put":
        return float(put_payoff(terminal_spot, strike))
    raise ValueError("Option type must be either 'call' or 'put'.")


def _validate_inputs(
    time_grid: ArrayLike,
    spot_path: ArrayLike,
    strike: float,
    rate: float,
    implied_volatility: float,
    option_type: OptionType,
    position: Position,
    quantity: float,
    contract_multiplier: float,
    transaction_cost_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    times, spots = np.asarray(time_grid, dtype=float), np.asarray(spot_path, dtype=float)
    if times.ndim != 1 or spots.ndim != 1 or len(times) != len(spots) or len(times) < 2:
        raise ValueError("Time grid and spot path must be equal-length one-dimensional arrays.")
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(spots)):
        raise ValueError("Time and spot values must be finite.")
    if not np.isclose(times[0], 0.0) or np.any(np.diff(times) <= 0):
        raise ValueError("Time grid must start at zero and be strictly increasing.")
    if np.any(spots <= 0):
        raise ValueError("Spot prices must be strictly positive.")
    for name, value in {
        "strike": strike,
        "rate": rate,
        "implied_volatility": implied_volatility,
        "quantity": quantity,
        "contract_multiplier": contract_multiplier,
        "transaction_cost_rate": transaction_cost_rate,
    }.items():
        if not isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if strike <= 0 or implied_volatility <= 0 or quantity <= 0 or contract_multiplier <= 0:
        raise ValueError("Strike, volatility, quantity, and multiplier must be strictly positive.")
    if transaction_cost_rate < 0:
        raise ValueError("Transaction cost rate cannot be negative.")
    if option_type not in {"call", "put"} or position not in {"long", "short"}:
        raise ValueError("Invalid option type or position.")
    return times, spots


def delta_hedge_path(
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
    option_premium: float | None = None,
    rebalance_indices: tuple[int, ...] | None = None,
) -> HedgePathResult:
    times, spots = _validate_inputs(
        time_grid, spot_path, strike, rate, implied_volatility, option_type,
        position, quantity, contract_multiplier, transaction_cost_rate
    )
    terminal_index = len(times) - 1
    maturity = float(times[-1])
    if rebalance_indices is None:
        rebalance_set = set(range(terminal_index))
    else:
        rebalance_set = {int(i) for i in rebalance_indices}
        if 0 not in rebalance_set:
            raise ValueError("Rebalance indices must include inception index 0.")
        if any(i < 0 or i >= terminal_index for i in rebalance_set):
            raise ValueError("Rebalance indices must precede expiration.")

    sign = 1.0 if position == "long" else -1.0
    units = quantity * contract_multiplier
    model_premium = float(black_scholes_price(spots[0], strike, rate, maturity, implied_volatility, option_type))
    if option_premium is None:
        premium = model_premium
    else:
        if not isfinite(option_premium) or option_premium < 0:
            raise ValueError("Option premium must be finite and non-negative.")
        premium = float(option_premium)

    initial_delta = float(black_scholes_greeks(spots[0], strike, rate, maturity, implied_volatility, option_type).delta)
    stock_position = -sign * initial_delta * units
    stock_trade = stock_position
    transaction_cost = transaction_cost_rate * abs(stock_trade) * spots[0]
    option_cashflow = -sign * premium * units
    stock_trade_cashflow = -stock_trade * spots[0]
    cash = option_cashflow + stock_trade_cashflow - transaction_cost
    cumulative_cost = transaction_cost

    rows: list[dict[str, float | bool]] = [{
        "time": float(times[0]),
        "time_to_expiry": maturity,
        "spot": float(spots[0]),
        "is_rebalance": True,
        "option_price_per_unit": premium,
        "model_option_price_per_unit": model_premium,
        "option_delta_per_unit": initial_delta,
        "target_stock_position": stock_position,
        "actual_stock_position": stock_position,
        "stock_trade": stock_trade,
        "stock_trade_cashflow": stock_trade_cashflow,
        "option_cashflow": option_cashflow,
        "option_settlement": 0.0,
        "interest_accrual": 0.0,
        "transaction_cost": transaction_cost,
        "cumulative_transaction_cost": cumulative_cost,
        "cash_account": cash,
        "stock_market_value": stock_position * spots[0],
        "option_position_value": sign * model_premium * units,
        "net_delta": stock_position + sign * initial_delta * units,
        "portfolio_value": cash + stock_position * spots[0] + sign * model_premium * units,
    }]

    for index in range(1, terminal_index):
        dt = float(times[index] - times[index - 1])
        interest = cash * (np.exp(rate * dt) - 1.0)
        cash += interest
        remaining = maturity - float(times[index])
        model_value = float(black_scholes_price(spots[index], strike, rate, remaining, implied_volatility, option_type))
        delta = float(black_scholes_greeks(spots[index], strike, rate, remaining, implied_volatility, option_type).delta)
        target = -sign * delta * units
        is_rebalance = index in rebalance_set
        trade = target - stock_position if is_rebalance else 0.0
        trade_cashflow = -trade * spots[index]
        cost = transaction_cost_rate * abs(trade) * spots[index]
        cash += trade_cashflow - cost
        stock_position += trade
        cumulative_cost += cost
        option_value = sign * model_value * units
        net_delta = stock_position + sign * delta * units
        rows.append({
            "time": float(times[index]),
            "time_to_expiry": remaining,
            "spot": float(spots[index]),
            "is_rebalance": is_rebalance,
            "option_price_per_unit": model_value,
            "model_option_price_per_unit": model_value,
            "option_delta_per_unit": delta,
            "target_stock_position": target,
            "actual_stock_position": stock_position,
            "stock_trade": trade,
            "stock_trade_cashflow": trade_cashflow,
            "option_cashflow": 0.0,
            "option_settlement": 0.0,
            "interest_accrual": interest,
            "transaction_cost": cost,
            "cumulative_transaction_cost": cumulative_cost,
            "cash_account": cash,
            "stock_market_value": stock_position * spots[index],
            "option_position_value": option_value,
            "net_delta": net_delta,
            "portfolio_value": cash + stock_position * spots[index] + option_value,
        })

    final_dt = float(times[-1] - times[-2])
    final_interest = cash * (np.exp(rate * final_dt) - 1.0)
    cash += final_interest
    terminal_spot = float(spots[-1])
    payoff = _terminal_payoff(terminal_spot, strike, option_type)
    option_settlement = sign * payoff * units
    liquidation_trade = -stock_position
    liquidation_cashflow = -liquidation_trade * terminal_spot
    liquidation_cost = transaction_cost_rate * abs(liquidation_trade) * terminal_spot
    cash += liquidation_cashflow - liquidation_cost + option_settlement
    cumulative_cost += liquidation_cost

    rows.append({
        "time": float(times[-1]),
        "time_to_expiry": 0.0,
        "spot": terminal_spot,
        "is_rebalance": False,
        "option_price_per_unit": payoff,
        "model_option_price_per_unit": payoff,
        "option_delta_per_unit": np.nan,
        "target_stock_position": 0.0,
        "actual_stock_position": 0.0,
        "stock_trade": liquidation_trade,
        "stock_trade_cashflow": liquidation_cashflow,
        "option_cashflow": 0.0,
        "option_settlement": option_settlement,
        "interest_accrual": final_interest,
        "transaction_cost": liquidation_cost,
        "cumulative_transaction_cost": cumulative_cost,
        "cash_account": cash,
        "stock_market_value": 0.0,
        "option_position_value": 0.0,
        "net_delta": 0.0,
        "portfolio_value": cash,
    })

    return HedgePathResult(
        history=pd.DataFrame(rows),
        terminal_pnl=float(cash),
        option_premium_per_unit=premium,
        option_payoff_per_unit=payoff,
        total_transaction_cost=float(cumulative_cost),
        number_of_rebalances=max(len(rebalance_set) - 1, 0),
    )


def run_delta_hedging_experiment(
    time_grid: np.ndarray,
    spot_paths: np.ndarray,
    strike: float,
    rate: float,
    implied_volatility: float,
    option_type: OptionType,
    position: Position,
    transaction_cost_rate: float = 0.0,
    quantity: float = 1.0,
    contract_multiplier: float = 1.0,
    rebalance_indices: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    if spot_paths.ndim != 2 or spot_paths.shape[1] != len(time_grid):
        raise ValueError("Spot paths must be a 2D array matching the time grid.")
    rows = []
    total_time = float(time_grid[-1])
    for path_id, path in enumerate(spot_paths):
        result = delta_hedge_path(
            time_grid, path, strike, rate, implied_volatility, option_type,
            position, quantity, contract_multiplier, transaction_cost_rate,
            rebalance_indices=rebalance_indices,
        )
        rows.append({
            "path_id": path_id,
            "terminal_spot": float(path[-1]),
            "realized_volatility": realized_volatility_from_path(path, total_time),
            "terminal_pnl": result.terminal_pnl,
            "total_transaction_cost": result.total_transaction_cost,
            "option_premium_per_unit": result.option_premium_per_unit,
            "option_payoff_per_unit": result.option_payoff_per_unit,
            "number_of_rebalances": result.number_of_rebalances,
        })
    return pd.DataFrame(rows)
