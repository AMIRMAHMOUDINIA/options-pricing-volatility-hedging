"""Synthetic implied-volatility surfaces and basic static-arbitrage diagnostics."""

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .arbitrage import european_option_bounds
from .black_scholes import black_scholes_price
from .implied_volatility import implied_volatility_brent

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class SmileParameters:
    base_volatility: float = 0.22
    term_slope: float = -0.02
    skew: float = -0.08
    curvature: float = 0.18


def forward_price(spot: float, rate: float, time_to_expiry: float) -> float:
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("Spot must be finite and strictly positive.")
    if not np.isfinite(rate):
        raise ValueError("Rate must be finite.")
    if not np.isfinite(time_to_expiry) or time_to_expiry <= 0:
        raise ValueError("Time to expiry must be finite and positive.")
    return float(spot * np.exp(rate * time_to_expiry))


def log_forward_moneyness(strike: float | np.ndarray, forward: float) -> float | np.ndarray:
    strike_a = np.asarray(strike, dtype=float)
    if np.any(~np.isfinite(strike_a)) or np.any(strike_a <= 0):
        raise ValueError("Strike must contain finite, strictly positive values.")
    if not np.isfinite(forward) or forward <= 0:
        raise ValueError("Forward must be finite and positive.")
    result = np.log(strike_a / forward)
    return float(result) if result.ndim == 0 else result


def synthetic_implied_volatility(
    log_moneyness: float | np.ndarray,
    time_to_expiry: float,
    parameters: SmileParameters = SmileParameters(),
) -> float | np.ndarray:
    k = np.asarray(log_moneyness, dtype=float)
    if np.any(~np.isfinite(k)):
        raise ValueError("Log moneyness must be finite.")
    if not np.isfinite(time_to_expiry) or time_to_expiry <= 0:
        raise ValueError("Time to expiry must be finite and positive.")
    for name, value in parameters.__dict__.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    atm = parameters.base_volatility + parameters.term_slope * np.log1p(time_to_expiry)
    volatility = atm + parameters.skew * k + parameters.curvature * k**2
    if np.any(volatility <= 0):
        raise ValueError("Synthetic parameters produced non-positive volatility.")
    return float(volatility) if volatility.ndim == 0 else volatility


def generate_synthetic_surface(
    spot: float,
    rate: float,
    log_moneyness_grid: Iterable[float],
    maturities: Iterable[float],
    parameters: SmileParameters = SmileParameters(),
) -> pd.DataFrame:
    k_values = np.unique(np.sort(np.asarray(list(log_moneyness_grid), dtype=float)))
    maturity_values = np.unique(np.sort(np.asarray(list(maturities), dtype=float)))
    if k_values.size == 0 or maturity_values.size == 0:
        raise ValueError("Moneyness and maturity grids cannot be empty.")
    if np.any(~np.isfinite(k_values)) or np.any(~np.isfinite(maturity_values)) or np.any(maturity_values <= 0):
        raise ValueError("Grid values must be finite and maturities positive.")

    rows: list[dict[str, float]] = []
    for maturity in maturity_values:
        fwd = forward_price(spot, rate, float(maturity))
        vol = np.asarray(synthetic_implied_volatility(k_values, float(maturity), parameters), dtype=float)
        strikes = fwd * np.exp(k_values)
        calls = np.asarray(black_scholes_price(spot, strikes, rate, maturity, vol, "call"), dtype=float)
        puts = np.asarray(black_scholes_price(spot, strikes, rate, maturity, vol, "put"), dtype=float)
        for k, strike, sigma, call, put in zip(k_values, strikes, vol, calls, puts, strict=True):
            rows.append({
                "spot": float(spot),
                "rate": float(rate),
                "time_to_expiry": float(maturity),
                "forward": float(fwd),
                "strike": float(strike),
                "log_moneyness": float(k),
                "implied_volatility": float(sigma),
                "total_variance": float(sigma**2 * maturity),
                "call_price": float(call),
                "put_price": float(put),
            })
    return pd.DataFrame(rows).sort_values(["time_to_expiry", "strike"], ignore_index=True)


def recover_surface_implied_volatility(
    surface: pd.DataFrame,
    price_column: str,
    option_type: OptionType,
) -> pd.DataFrame:
    required = {"spot", "strike", "rate", "time_to_expiry", price_column}
    missing = required - set(surface.columns)
    if missing:
        raise ValueError("Surface is missing columns: " + ", ".join(sorted(missing)))
    recovered = surface.copy()
    results = []
    for row in recovered.itertuples(index=False):
        result = implied_volatility_brent(
            float(getattr(row, price_column)), float(row.spot), float(row.strike),
            float(row.rate), float(row.time_to_expiry), option_type
        )
        results.append(result)
    recovered["recovered_implied_volatility"] = [r.volatility for r in results]
    recovered["iv_converged"] = [r.converged for r in results]
    recovered["iv_iterations"] = [r.iterations for r in results]
    recovered["iv_price_residual"] = [r.residual for r in results]
    if "implied_volatility" in recovered:
        recovered["iv_absolute_error"] = np.abs(
            recovered["recovered_implied_volatility"] - recovered["implied_volatility"]
        )
    return recovered


def check_call_slice_arbitrage(surface: pd.DataFrame, tolerance: float = 1e-10) -> pd.DataFrame:
    if tolerance < 0:
        raise ValueError("Tolerance cannot be negative.")
    required = {"spot", "rate", "time_to_expiry", "strike", "call_price"}
    missing = required - set(surface.columns)
    if missing:
        raise ValueError("Surface is missing columns: " + ", ".join(sorted(missing)))

    reports = []
    for maturity, group in surface.groupby("time_to_expiry", sort=True):
        group = group.sort_values("strike")
        strikes = group["strike"].to_numpy(float)
        prices = group["call_price"].to_numpy(float)
        if len(strikes) < 3 or np.any(np.diff(strikes) <= 0):
            raise ValueError("Each maturity requires at least three strictly increasing strikes.")
        spot, rate = float(group["spot"].iloc[0]), float(group["rate"].iloc[0])
        bound_pass = []
        for strike, price in zip(strikes, prices, strict=True):
            bounds = european_option_bounds(spot, float(strike), rate, float(maturity), "call")
            bound_pass.append(bounds.lower - tolerance <= price <= bounds.upper + tolerance)
        slopes = np.diff(prices) / np.diff(strikes)
        slope_changes = np.diff(slopes)
        discounted_unit = np.exp(-rate * float(maturity))
        reports.append({
            "time_to_expiry": float(maturity),
            "price_bounds_passed": bool(all(bound_pass)),
            "monotonicity_passed": bool(np.all(slopes <= tolerance)),
            "slope_bounds_passed": bool(np.all(slopes >= -discounted_unit - tolerance) and np.all(slopes <= tolerance)),
            "convexity_passed": bool(np.all(slope_changes >= -tolerance)),
            "minimum_slope": float(np.min(slopes)),
            "maximum_slope": float(np.max(slopes)),
            "minimum_slope_change": float(np.min(slope_changes)),
        })
    report = pd.DataFrame(reports)
    report["all_checks_passed"] = report[
        ["price_bounds_passed", "monotonicity_passed", "slope_bounds_passed", "convexity_passed"]
    ].all(axis=1)
    return report


def check_calendar_total_variance(surface: pd.DataFrame, tolerance: float = 1e-12) -> pd.DataFrame:
    required = {"time_to_expiry", "log_moneyness", "total_variance"}
    missing = required - set(surface.columns)
    if missing:
        raise ValueError("Surface is missing columns: " + ", ".join(sorted(missing)))
    pivot = surface.pivot(index="time_to_expiry", columns="log_moneyness", values="total_variance").sort_index()
    if pivot.isna().any().any():
        raise ValueError("Calendar check requires a complete common moneyness grid.")
    rows = []
    for k in pivot.columns:
        changes = np.diff(pivot[k].to_numpy(float))
        rows.append({
            "log_moneyness": float(k),
            "minimum_total_variance_change": float(np.min(changes)),
            "calendar_check_passed": bool(np.all(changes >= -tolerance)),
        })
    return pd.DataFrame(rows)
