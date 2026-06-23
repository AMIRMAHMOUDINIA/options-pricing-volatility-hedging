"""Terminal option payoffs and position profit utilities."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Position = Literal["long", "short"]
NumericResult = float | NDArray[np.float64]


def _as_result(value: NDArray[np.float64]) -> NumericResult:
    return float(value) if value.ndim == 0 else value


def _validate_spot_and_strike(
    spot_at_expiry: ArrayLike,
    strike: float,
) -> NDArray[np.float64]:
    if not np.isfinite(strike) or strike <= 0:
        raise ValueError("Strike must be finite and strictly positive.")

    spot = np.asarray(spot_at_expiry, dtype=float)
    if np.any(~np.isfinite(spot)):
        raise ValueError("Expiration prices must be finite.")
    if np.any(spot < 0):
        raise ValueError("Expiration prices cannot be negative.")
    return spot


def call_payoff(spot_at_expiry: ArrayLike, strike: float) -> NumericResult:
    """Return ``max(S_T - K, 0)`` for a European call."""
    spot = _validate_spot_and_strike(spot_at_expiry, strike)
    return _as_result(np.maximum(spot - strike, 0.0))


def put_payoff(spot_at_expiry: ArrayLike, strike: float) -> NumericResult:
    """Return ``max(K - S_T, 0)`` for a European put."""
    spot = _validate_spot_and_strike(spot_at_expiry, strike)
    return _as_result(np.maximum(strike - spot, 0.0))


def option_profit(
    payoff: ArrayLike,
    premium: float,
    position: Position = "long",
    quantity: float = 1.0,
    contract_multiplier: float = 1.0,
) -> NumericResult:
    """Convert terminal payoff into terminal position profit.

    Financing, fees, spreads, margin costs, and taxes are excluded.
    """
    if not np.isfinite(premium) or premium < 0:
        raise ValueError("Premium must be finite and non-negative.")
    if not np.isfinite(quantity) or quantity <= 0:
        raise ValueError("Quantity must be finite and strictly positive.")
    if not np.isfinite(contract_multiplier) or contract_multiplier <= 0:
        raise ValueError("Contract multiplier must be finite and strictly positive.")
    if position not in {"long", "short"}:
        raise ValueError("Position must be either 'long' or 'short'.")

    payoff_array = np.asarray(payoff, dtype=float)
    if np.any(~np.isfinite(payoff_array)):
        raise ValueError("Payoff must contain only finite values.")
    if np.any(payoff_array < 0):
        raise ValueError("Option payoff cannot be negative.")

    sign = 1.0 if position == "long" else -1.0
    profit = sign * (payoff_array - premium) * quantity * contract_multiplier
    return _as_result(profit)
