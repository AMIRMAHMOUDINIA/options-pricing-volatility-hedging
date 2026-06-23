"""Underlying-price path simulation and hedge schedules."""

from math import isfinite

import numpy as np
from numpy.typing import NDArray


def simulate_gbm_paths(
    spot: float,
    drift: float,
    realized_volatility: float,
    time_to_expiry: float,
    number_of_steps: int,
    number_of_paths: int,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    for name, value in {
        "spot": spot,
        "drift": drift,
        "realized_volatility": realized_volatility,
        "time_to_expiry": time_to_expiry,
    }.items():
        if not isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if spot <= 0 or time_to_expiry <= 0:
        raise ValueError("Spot and time to expiry must be strictly positive.")
    if realized_volatility < 0:
        raise ValueError("Realized volatility cannot be negative.")
    if not isinstance(number_of_steps, int) or number_of_steps <= 0:
        raise ValueError("Number of steps must be a positive integer.")
    if not isinstance(number_of_paths, int) or number_of_paths <= 0:
        raise ValueError("Number of paths must be a positive integer.")

    time_grid = np.linspace(0.0, time_to_expiry, number_of_steps + 1)
    dt = time_to_expiry / number_of_steps
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((number_of_paths, number_of_steps))
    log_increments = (
        (drift - 0.5 * realized_volatility**2) * dt
        + realized_volatility * np.sqrt(dt) * shocks
    )
    cumulative = np.cumsum(log_increments, axis=1)
    paths = np.empty((number_of_paths, number_of_steps + 1), dtype=float)
    paths[:, 0] = spot
    paths[:, 1:] = spot * np.exp(cumulative)
    return time_grid, paths


def regular_rebalance_indices(number_of_steps: int, every_n_steps: int) -> tuple[int, ...]:
    if not isinstance(number_of_steps, int) or number_of_steps <= 0:
        raise ValueError("Number of steps must be a positive integer.")
    if not isinstance(every_n_steps, int) or every_n_steps <= 0:
        raise ValueError("every_n_steps must be a positive integer.")
    return tuple(int(index) for index in np.arange(0, number_of_steps, every_n_steps, dtype=int))
