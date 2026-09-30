"""Raw-SVI smile calibration and transparent arbitrage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit


@dataclass(frozen=True)
class SVIParameters:
    """Raw-SVI parameters.

    Total variance is

        w(k) = a + b * [rho * (k - m) + sqrt((k - m)^2 + sigma^2)]

    where k is log-forward moneyness.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        values = np.array(
            [self.a, self.b, self.rho, self.m, self.sigma],
            dtype=float,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError("All SVI parameters must be finite.")

        if self.b <= 0:
            raise ValueError("SVI b must be strictly positive.")

        if not -1.0 < self.rho < 1.0:
            raise ValueError("SVI rho must lie strictly between -1 and 1.")

        if self.sigma <= 0:
            raise ValueError("SVI sigma must be strictly positive.")

        if self.minimum_total_variance <= 0:
            raise ValueError(
                "SVI parameters imply non-positive minimum total variance."
            )

    @property
    def minimum_total_variance(self) -> float:
        return float(
            self.a
            + self.b
            * self.sigma
            * np.sqrt(1.0 - self.rho**2)
        )

    @property
    def left_wing_slope(self) -> float:
        return float(self.b * (1.0 - self.rho))

    @property
    def right_wing_slope(self) -> float:
        return float(self.b * (1.0 + self.rho))


@dataclass(frozen=True)
class SVIFitResult:
    parameters: SVIParameters
    success: bool
    n_observations: int
    rmse_total_variance: float
    rmse_implied_volatility: float
    max_abs_iv_error: float
    minimum_butterfly_g: float
    grid_butterfly_check_passed: bool
    optimizer_message: str


@dataclass(frozen=True)
class SVISurfaceFit:
    parameters: pd.DataFrame
    points: pd.DataFrame


def svi_total_variance(
    log_moneyness: float | np.ndarray,
    parameters: SVIParameters,
) -> float | np.ndarray:
    """Evaluate raw-SVI total variance."""

    k = np.asarray(log_moneyness, dtype=float)

    if np.any(~np.isfinite(k)):
        raise ValueError("Log moneyness must be finite.")

    x = k - parameters.m

    total_variance = (
        parameters.a
        + parameters.b
        * (
            parameters.rho * x
            + np.sqrt(x**2 + parameters.sigma**2)
        )
    )

    if np.any(total_variance <= 0):
        raise ValueError(
            "SVI produced non-positive total variance."
        )

    if total_variance.ndim == 0:
        return float(total_variance)

    return total_variance


def svi_implied_volatility(
    log_moneyness: float | np.ndarray,
    time_to_expiry: float,
    parameters: SVIParameters,
) -> float | np.ndarray:
    """Convert SVI total variance into implied volatility."""

    if (
        not np.isfinite(time_to_expiry)
        or time_to_expiry <= 0
    ):
        raise ValueError(
            "Time to expiry must be finite and positive."
        )

    total_variance = np.asarray(
        svi_total_variance(log_moneyness, parameters),
        dtype=float,
    )

    volatility = np.sqrt(
        total_variance / time_to_expiry
    )

    if volatility.ndim == 0:
        return float(volatility)

    return volatility


def _svi_derivatives(
    log_moneyness: np.ndarray,
    parameters: SVIParameters,
) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(log_moneyness, dtype=float)
    x = k - parameters.m

    root = np.sqrt(
        x**2 + parameters.sigma**2
    )

    first = parameters.b * (
        parameters.rho + x / root
    )

    second = (
        parameters.b
        * parameters.sigma**2
        / root**3
    )

    return first, second


def svi_butterfly_g(
    log_moneyness: float | np.ndarray,
    parameters: SVIParameters,
) -> float | np.ndarray:
    """Evaluate the standard SVI density/butterfly diagnostic g(k).

    A negative value indicates a local butterfly-arbitrage violation.
    Checking a finite grid is a numerical diagnostic, not a formal proof
    of global arbitrage freedom.
    """

    k = np.asarray(log_moneyness, dtype=float)

    total_variance = np.asarray(
        svi_total_variance(k, parameters),
        dtype=float,
    )

    first, second = _svi_derivatives(
        k,
        parameters,
    )

    g_value = (
        (
            1.0
            - k
            * first
            / (2.0 * total_variance)
        )
        ** 2
        - (first**2 / 4.0)
        * (
            1.0 / total_variance
            + 0.25
        )
        + second / 2.0
    )

    if g_value.ndim == 0:
        return float(g_value)

    return g_value


def _decode_parameters(
    theta: np.ndarray,
) -> SVIParameters:
    """Map unconstrained optimization variables into valid SVI parameters."""

    variance_floor = float(
        np.exp(theta[0])
    )

    rho = float(
        np.tanh(theta[2])
    )

    sigma = float(
        np.exp(theta[4])
    )

    # Keep both Lee-style asymptotic total-variance slopes below 2.
    b_cap = 1.999 / (
        1.0 + abs(rho)
    )

    b = float(
        b_cap * expit(theta[1])
    )

    a = float(
        variance_floor
        - b
        * sigma
        * np.sqrt(
            max(
                1e-16,
                1.0 - rho**2,
            )
        )
    )

    return SVIParameters(
        a=a,
        b=b,
        rho=rho,
        m=float(theta[3]),
        sigma=sigma,
    )


def _initial_parameters(
    log_moneyness: np.ndarray,
    total_variance: np.ndarray,
) -> np.ndarray:
    minimum_variance = max(
        float(np.min(total_variance)) * 0.8,
        1e-8,
    )

    minimum_index = int(
        np.argmin(total_variance)
    )

    m = float(
        log_moneyness[minimum_index]
    )

    sigma = max(
        float(np.std(log_moneyness)),
        0.10,
    )

    rho = 0.0

    span = max(
        float(np.ptp(log_moneyness)),
        0.10,
    )

    b_guess = max(
        float(
            (
                np.max(total_variance)
                - np.min(total_variance)
            )
            / span
        ),
        1e-3,
    )

    b_cap = 1.999

    ratio = min(
        max(
            b_guess / b_cap,
            1e-6,
        ),
        1.0 - 1e-6,
    )

    beta = float(
        np.log(
            ratio / (1.0 - ratio)
        )
    )

    return np.array(
        [
            np.log(minimum_variance),
            beta,
            np.arctanh(rho),
            m,
            np.log(sigma),
        ],
        dtype=float,
    )


def fit_svi_smile(
    log_moneyness: np.ndarray,
    implied_volatility: np.ndarray,
    time_to_expiry: float,
    weights: np.ndarray | None = None,
    minimum_points: int = 5,
    max_function_evaluations: int = 20_000,
) -> SVIFitResult:
    """Fit one raw-SVI smile in total-variance space."""

    if (
        not np.isfinite(time_to_expiry)
        or time_to_expiry <= 0
    ):
        raise ValueError(
            "Time to expiry must be finite and positive."
        )

    if minimum_points < 5:
        raise ValueError(
            "At least five points are required for a five-parameter SVI fit."
        )

    k = np.asarray(
        log_moneyness,
        dtype=float,
    )

    volatility = np.asarray(
        implied_volatility,
        dtype=float,
    )

    if k.shape != volatility.shape:
        raise ValueError(
            "Moneyness and volatility arrays must have the same shape."
        )

    if weights is None:
        fit_weights = np.ones_like(
            k,
            dtype=float,
        )
    else:
        fit_weights = np.asarray(
            weights,
            dtype=float,
        )

        if fit_weights.shape != k.shape:
            raise ValueError(
                "Weights must have the same shape as the observations."
            )

    valid = (
        np.isfinite(k)
        & np.isfinite(volatility)
        & (volatility > 0)
        & np.isfinite(fit_weights)
        & (fit_weights > 0)
    )

    k = k[valid]
    volatility = volatility[valid]
    fit_weights = fit_weights[valid]

    if (
        len(k) < minimum_points
        or np.unique(k).size < minimum_points
    ):
        raise ValueError(
            "Insufficient distinct smile points for SVI calibration."
        )

    order = np.argsort(k)

    k = k[order]
    volatility = volatility[order]
    fit_weights = fit_weights[order]

    fit_weights = (
        fit_weights
        / np.mean(fit_weights)
    )

    total_variance = (
        volatility**2
        * time_to_expiry
    )

    theta0 = _initial_parameters(
        k,
        total_variance,
    )

    lower_bounds = np.array(
        [
            -30.0,
            -20.0,
            -4.0,
            float(np.min(k)) - 2.0,
            -10.0,
        ],
        dtype=float,
    )

    upper_bounds = np.array(
        [
            5.0,
            20.0,
            4.0,
            float(np.max(k)) + 2.0,
            3.0,
        ],
        dtype=float,
    )

    def residuals(
        theta: np.ndarray,
    ) -> np.ndarray:
        parameters = _decode_parameters(
            theta
        )

        fitted = np.asarray(
            svi_total_variance(
                k,
                parameters,
            ),
            dtype=float,
        )

        return (
            np.sqrt(fit_weights)
            * (
                fitted
                - total_variance
            )
        )

    optimizer = least_squares(
        residuals,
        theta0,
        bounds=(
            lower_bounds,
            upper_bounds,
        ),
        max_nfev=max_function_evaluations,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    parameters = _decode_parameters(
        optimizer.x
    )

    fitted_total_variance = np.asarray(
        svi_total_variance(
            k,
            parameters,
        ),
        dtype=float,
    )

    fitted_volatility = np.sqrt(
        fitted_total_variance
        / time_to_expiry
    )

    grid = np.linspace(
        float(np.min(k)) - 0.5,
        float(np.max(k)) + 0.5,
        2_001,
    )

    butterfly_g = np.asarray(
        svi_butterfly_g(
            grid,
            parameters,
        ),
        dtype=float,
    )

    return SVIFitResult(
        parameters=parameters,
        success=bool(optimizer.success),
        n_observations=int(len(k)),
        rmse_total_variance=float(
            np.sqrt(
                np.mean(
                    (
                        fitted_total_variance
                        - total_variance
                    )
                    ** 2
                )
            )
        ),
        rmse_implied_volatility=float(
            np.sqrt(
                np.mean(
                    (
                        fitted_volatility
                        - volatility
                    )
                    ** 2
                )
            )
        ),
        max_abs_iv_error=float(
            np.max(
                np.abs(
                    fitted_volatility
                    - volatility
                )
            )
        ),
        minimum_butterfly_g=float(
            np.min(butterfly_g)
        ),
        grid_butterfly_check_passed=bool(
            np.all(
                butterfly_g >= -1e-10
            )
        ),
        optimizer_message=str(
            optimizer.message
        ),
    )


def fit_svi_surface(
    frame: pd.DataFrame,
    minimum_points: int = 7,
    weight_column: str | None = None,
) -> SVISurfaceFit:
    """Fit one SVI smile to each expiry in an option-chain DataFrame.

    Required columns:
        expiry
        time_to_expiry
        log_forward_moneyness
        implied_volatility
    """

    required = {
        "expiry",
        "time_to_expiry",
        "log_forward_moneyness",
        "implied_volatility",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            "SVI surface input is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    if (
        weight_column is not None
        and weight_column not in frame.columns
    ):
        raise ValueError(
            f"Weight column {weight_column!r} is not present."
        )

    parameter_rows: list[
        dict[str, object]
    ] = []

    fitted_point_frames: list[
        pd.DataFrame
    ] = []

    for expiry, group in frame.groupby(
        "expiry",
        sort=True,
    ):
        group = group.dropna(
            subset=[
                "time_to_expiry",
                "log_forward_moneyness",
                "implied_volatility",
            ]
        ).copy()

        group = group[
            group[
                "implied_volatility"
            ]
            > 0
        ]

        if (
            group[
                "log_forward_moneyness"
            ].nunique()
            < minimum_points
        ):
            continue

        maturity = float(
            group[
                "time_to_expiry"
            ].median()
        )

        weights = None

        if weight_column is not None:
            weights = group[
                weight_column
            ].to_numpy(
                dtype=float
            )

        result = fit_svi_smile(
            log_moneyness=group[
                "log_forward_moneyness"
            ].to_numpy(
                dtype=float
            ),
            implied_volatility=group[
                "implied_volatility"
            ].to_numpy(
                dtype=float
            ),
            time_to_expiry=maturity,
            weights=weights,
            minimum_points=minimum_points,
        )

        parameters = (
            result.parameters
        )

        k = group[
            "log_forward_moneyness"
        ].to_numpy(
            dtype=float
        )

        fitted_total_variance = np.asarray(
            svi_total_variance(
                k,
                parameters,
            ),
            dtype=float,
        )

        fitted_volatility = np.sqrt(
            fitted_total_variance
            / maturity
        )

        group[
            "market_total_variance"
        ] = (
            group[
                "implied_volatility"
            ].to_numpy(
                dtype=float
            )
            ** 2
            * maturity
        )

        group[
            "svi_total_variance"
        ] = fitted_total_variance

        group[
            "svi_implied_volatility"
        ] = fitted_volatility

        group[
            "svi_iv_residual"
        ] = (
            fitted_volatility
            - group[
                "implied_volatility"
            ].to_numpy(
                dtype=float
            )
        )

        fitted_point_frames.append(
            group
        )

        parameter_rows.append(
            {
                "expiry": expiry,
                "time_to_expiry": maturity,
                "n_observations": result.n_observations,
                "a": parameters.a,
                "b": parameters.b,
                "rho": parameters.rho,
                "m": parameters.m,
                "sigma": parameters.sigma,
                "minimum_total_variance": parameters.minimum_total_variance,
                "left_wing_slope": parameters.left_wing_slope,
                "right_wing_slope": parameters.right_wing_slope,
                "rmse_total_variance": result.rmse_total_variance,
                "rmse_implied_volatility": result.rmse_implied_volatility,
                "max_abs_iv_error": result.max_abs_iv_error,
                "minimum_butterfly_g": result.minimum_butterfly_g,
                "grid_butterfly_check_passed": result.grid_butterfly_check_passed,
                "optimizer_success": result.success,
                "optimizer_message": result.optimizer_message,
            }
        )

    if not parameter_rows:
        raise ValueError(
            "No expiry contained enough valid observations for SVI fitting."
        )

    parameters_frame = pd.DataFrame(
        parameter_rows
    ).sort_values(
        "time_to_expiry",
        ignore_index=True,
    )

    points_frame = pd.concat(
        fitted_point_frames,
        ignore_index=True,
    ).sort_values(
        [
            "time_to_expiry",
            "log_forward_moneyness",
        ],
        ignore_index=True,
    )

    return SVISurfaceFit(
        parameters=parameters_frame,
        points=points_frame,
    )
