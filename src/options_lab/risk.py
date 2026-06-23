"""Local Greek-based P&L approximation."""


def approximate_option_pnl(
    delta: float,
    gamma: float,
    vega: float,
    theta: float,
    rho: float,
    spot_change: float = 0.0,
    volatility_change: float = 0.0,
    elapsed_time_years: float = 0.0,
    rate_change: float = 0.0,
) -> float:
    """Second-order spot and first-order volatility/time/rate approximation."""
    return float(
        delta * spot_change
        + 0.5 * gamma * spot_change**2
        + vega * volatility_change
        + theta * elapsed_time_years
        + rho * rate_change
    )
