"""Cox-Ross-Rubinstein binomial pricing for European and American options."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, sqrt
from typing import Literal

import numpy as np


OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


@dataclass(frozen=True)
class EarlyExercisePoint:
    """One point on a discrete early-exercise boundary."""

    step: int
    time: float
    boundary_spot: float
    exercise_node_count: int


@dataclass(frozen=True)
class CRRResult:
    """Result and diagnostics from a CRR tree."""

    price: float
    steps: int
    up_factor: float
    down_factor: float
    risk_neutral_probability: float
    discount_factor: float
    early_exercise: tuple[EarlyExercisePoint, ...]


def _validate_inputs(
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    volatility: float,
    option_type: OptionType,
    steps: int,
    exercise_style: ExerciseStyle,
    dividend_yield: float,
    exercise_tolerance: float,
) -> None:
    values = {
        "spot": spot,
        "strike": strike,
        "rate": rate,
        "time_to_expiry": time_to_expiry,
        "volatility": volatility,
        "dividend_yield": dividend_yield,
        "exercise_tolerance": exercise_tolerance,
    }

    for name, value in values.items():
        if not isfinite(value):
            raise ValueError(
                f"{name} must be finite."
            )

    if spot <= 0:
        raise ValueError(
            "Spot must be strictly positive."
        )

    if strike <= 0:
        raise ValueError(
            "Strike must be strictly positive."
        )

    if time_to_expiry < 0:
        raise ValueError(
            "Time to expiry cannot be negative."
        )

    if volatility < 0:
        raise ValueError(
            "Volatility cannot be negative."
        )

    if exercise_tolerance < 0:
        raise ValueError(
            "Exercise tolerance cannot be negative."
        )

    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps <= 0
    ):
        raise ValueError(
            "Steps must be a strictly positive integer."
        )

    if option_type not in {
        "call",
        "put",
    }:
        raise ValueError(
            "Option type must be either 'call' or 'put'."
        )

    if exercise_style not in {
        "european",
        "american",
    }:
        raise ValueError(
            "Exercise style must be either 'european' or 'american'."
        )


def _intrinsic_value(
    spot: np.ndarray | float,
    strike: float,
    option_type: OptionType,
) -> np.ndarray:
    values = np.asarray(
        spot,
        dtype=float,
    )

    if option_type == "call":
        return np.maximum(
            values - strike,
            0.0,
        )

    return np.maximum(
        strike - values,
        0.0,
    )


def _zero_volatility_result(
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    option_type: OptionType,
    steps: int,
    exercise_style: ExerciseStyle,
    dividend_yield: float,
) -> CRRResult:
    """Handle the deterministic risk-neutral path when sigma is zero."""

    times = np.linspace(
        0.0,
        time_to_expiry,
        steps + 1,
    )

    spots = (
        spot
        * np.exp(
            (rate - dividend_yield)
            * times
        )
    )

    intrinsic = _intrinsic_value(
        spots,
        strike,
        option_type,
    )

    discounted = (
        np.exp(
            -rate * times
        )
        * intrinsic
    )

    if exercise_style == "european":
        price = float(
            discounted[-1]
        )
        exercise_points: tuple[
            EarlyExercisePoint,
            ...
        ] = ()

    else:
        best_index = int(
            np.argmax(discounted)
        )

        price = float(
            discounted[
                best_index
            ]
        )

        if (
            best_index < steps
            and intrinsic[
                best_index
            ]
            > 0
        ):
            exercise_points = (
                EarlyExercisePoint(
                    step=best_index,
                    time=float(
                        times[
                            best_index
                        ]
                    ),
                    boundary_spot=float(
                        spots[
                            best_index
                        ]
                    ),
                    exercise_node_count=1,
                ),
            )
        else:
            exercise_points = ()

    return CRRResult(
        price=price,
        steps=steps,
        up_factor=1.0,
        down_factor=1.0,
        risk_neutral_probability=float("nan"),
        discount_factor=float(
            exp(
                -rate
                * time_to_expiry
                / steps
            )
        ),
        early_exercise=exercise_points,
    )


def crr_binomial_tree(
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    volatility: float,
    option_type: OptionType,
    steps: int = 500,
    exercise_style: ExerciseStyle = "european",
    dividend_yield: float = 0.0,
    exercise_tolerance: float = 1e-12,
) -> CRRResult:
    """Price a vanilla option with a Cox-Ross-Rubinstein tree.

    Continuous dividend yield is represented by ``dividend_yield``.

    For an American option, intrinsic value and continuation value are
    compared at every node. The returned boundary contains only genuine
    pre-expiry exercise nodes.
    """

    _validate_inputs(
        spot=spot,
        strike=strike,
        rate=rate,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
        option_type=option_type,
        steps=steps,
        exercise_style=exercise_style,
        dividend_yield=dividend_yield,
        exercise_tolerance=exercise_tolerance,
    )

    if time_to_expiry == 0:
        price = float(
            _intrinsic_value(
                spot,
                strike,
                option_type,
            )
        )

        return CRRResult(
            price=price,
            steps=steps,
            up_factor=float("nan"),
            down_factor=float("nan"),
            risk_neutral_probability=float("nan"),
            discount_factor=1.0,
            early_exercise=(),
        )

    if volatility == 0:
        return _zero_volatility_result(
            spot=spot,
            strike=strike,
            rate=rate,
            time_to_expiry=time_to_expiry,
            option_type=option_type,
            steps=steps,
            exercise_style=exercise_style,
            dividend_yield=dividend_yield,
        )

    dt = (
        time_to_expiry
        / steps
    )

    up = exp(
        volatility
        * sqrt(dt)
    )

    down = 1.0 / up

    growth = exp(
        (rate - dividend_yield)
        * dt
    )

    denominator = (
        up - down
    )

    probability = (
        growth - down
    ) / denominator

    probability_tolerance = 1e-14

    if (
        probability
        < -probability_tolerance
        or probability
        > 1.0 + probability_tolerance
    ):
        raise ValueError(
            "CRR risk-neutral probability lies outside [0, 1]. "
            "Increase the number of steps or review rate, dividend yield, "
            "and volatility inputs."
        )

    probability = float(
        np.clip(
            probability,
            0.0,
            1.0,
        )
    )

    discount = exp(
        -rate * dt
    )

    terminal_indices = np.arange(
        steps + 1,
        dtype=float,
    )

    terminal_spots = (
        spot
        * down**steps
        * (
            up / down
        )
        ** terminal_indices
    )

    option_values = _intrinsic_value(
        terminal_spots,
        strike,
        option_type,
    )

    boundary_points: list[
        EarlyExercisePoint
    ] = []

    for step in range(
        steps - 1,
        -1,
        -1,
    ):
        continuation = discount * (
            probability
            * option_values[
                1 : step + 2
            ]
            + (
                1.0
                - probability
            )
            * option_values[
                : step + 1
            ]
        )

        if exercise_style == "european":
            option_values = continuation
            continue

        node_indices = np.arange(
            step + 1,
            dtype=float,
        )

        node_spots = (
            spot
            * down**step
            * (
                up / down
            )
            ** node_indices
        )

        intrinsic = _intrinsic_value(
            node_spots,
            strike,
            option_type,
        )

        exercise_mask = (
            intrinsic
            > continuation
            + exercise_tolerance
        )

        if np.any(
            exercise_mask
        ):
            exercise_spots = (
                node_spots[
                    exercise_mask
                ]
            )

            if option_type == "put":
                boundary_spot = float(
                    np.max(
                        exercise_spots
                    )
                )
            else:
                boundary_spot = float(
                    np.min(
                        exercise_spots
                    )
                )

            boundary_points.append(
                EarlyExercisePoint(
                    step=step,
                    time=float(
                        step * dt
                    ),
                    boundary_spot=boundary_spot,
                    exercise_node_count=int(
                        np.sum(
                            exercise_mask
                        )
                    ),
                )
            )

        option_values = np.maximum(
            intrinsic,
            continuation,
        )

    boundary_points.sort(
        key=lambda point: point.step
    )

    return CRRResult(
        price=float(
            option_values[0]
        ),
        steps=steps,
        up_factor=float(up),
        down_factor=float(down),
        risk_neutral_probability=probability,
        discount_factor=float(
            discount
        ),
        early_exercise=tuple(
            boundary_points
        ),
    )


def crr_binomial_price(
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    volatility: float,
    option_type: OptionType,
    steps: int = 500,
    exercise_style: ExerciseStyle = "european",
    dividend_yield: float = 0.0,
) -> float:
    """Return only the CRR option price."""

    return crr_binomial_tree(
        spot=spot,
        strike=strike,
        rate=rate,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
        option_type=option_type,
        steps=steps,
        exercise_style=exercise_style,
        dividend_yield=dividend_yield,
    ).price
