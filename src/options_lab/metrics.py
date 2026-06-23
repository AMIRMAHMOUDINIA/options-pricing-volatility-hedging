"""Risk and simulation metrics."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def realized_volatility_from_path(spot_path: ArrayLike, time_to_expiry: float) -> float:
    spots = np.asarray(spot_path, dtype=float)
    if spots.ndim != 1 or len(spots) < 2:
        raise ValueError("Spot path must be one-dimensional with at least two observations.")
    if np.any(~np.isfinite(spots)) or np.any(spots <= 0):
        raise ValueError("Spot prices must be finite and strictly positive.")
    if not np.isfinite(time_to_expiry) or time_to_expiry <= 0:
        raise ValueError("Time to expiry must be finite and positive.")
    log_returns = np.diff(np.log(spots))
    return float(np.sqrt(np.sum(log_returns**2) / time_to_expiry))


def summarize_hedging_results(results: pd.DataFrame) -> pd.Series:
    required = {"terminal_pnl", "total_transaction_cost"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError("Results are missing columns: " + ", ".join(sorted(missing)))
    pnl = results["terminal_pnl"].to_numpy(float)
    costs = results["total_transaction_cost"].to_numpy(float)
    if len(pnl) == 0:
        raise ValueError("Results cannot be empty.")
    return pd.Series({
        "number_of_paths": len(results),
        "mean_terminal_pnl": float(np.mean(pnl)),
        "median_terminal_pnl": float(np.median(pnl)),
        "pnl_standard_deviation": float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0,
        "root_mean_squared_pnl": float(np.sqrt(np.mean(pnl**2))),
        "mean_absolute_pnl": float(np.mean(np.abs(pnl))),
        "pnl_5_percentile": float(np.quantile(pnl, 0.05)),
        "pnl_95_percentile": float(np.quantile(pnl, 0.95)),
        "probability_of_loss": float(np.mean(pnl < 0)),
        "mean_transaction_cost": float(np.mean(costs)),
    })
