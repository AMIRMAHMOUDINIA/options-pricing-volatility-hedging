"""Preparation and diagnostics for empirical option-volatility surfaces."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .svi import (
    SVIParameters,
    svi_total_variance,
)


def prepare_deribit_svi_quotes(
    frame: pd.DataFrame,
    minimum_time_to_expiry: float = 7.0 / 365.0,
    maximum_abs_log_moneyness: float = 1.25,
) -> pd.DataFrame:
    """Prepare one OTM volatility observation per expiry/strike.

    A single median forward reference is used for every option within an
    expiry. This avoids assigning slightly different moneyness values to
    the call and put at the same strike when exchange summary timestamps
    differ slightly.

    The preferred quote is:
        put  when K < F
        call when K >= F

    If the preferred side is unavailable, the available side is retained.
    """

    if minimum_time_to_expiry <= 0:
        raise ValueError(
            "minimum_time_to_expiry must be positive."
        )

    if maximum_abs_log_moneyness <= 0:
        raise ValueError(
            "maximum_abs_log_moneyness must be positive."
        )

    required = {
        "expiry",
        "strike",
        "option_type",
        "underlying_price",
        "time_to_expiry",
        "implied_volatility",
        "valid_svi_quote",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            "Empirical surface input is missing columns: "
            + ", ".join(sorted(missing))
        )

    work = frame.copy()

    work["option_type"] = (
        work["option_type"]
        .astype(str)
        .str.lower()
    )

    work = work[
        work["valid_svi_quote"].astype(bool)
        & (
            work["time_to_expiry"]
            >= minimum_time_to_expiry
        )
        & np.isfinite(
            work["underlying_price"]
        )
        & (
            work["underlying_price"]
            > 0
        )
        & np.isfinite(
            work["strike"]
        )
        & (
            work["strike"]
            > 0
        )
        & np.isfinite(
            work["implied_volatility"]
        )
        & (
            work["implied_volatility"]
            > 0
        )
        & work["option_type"].isin(
            ["call", "put"]
        )
    ].copy()

    if work.empty:
        raise ValueError(
            "No valid Deribit quotes remain after basic filtering."
        )

    # Use one common forward reference for each expiry.
    work["calibration_forward"] = (
        work.groupby(
            "expiry"
        )["underlying_price"]
        .transform("median")
    )

    work["log_forward_moneyness"] = np.log(
        work["strike"]
        / work["calibration_forward"]
    )

    work = work[
        work[
            "log_forward_moneyness"
        ].abs()
        <= maximum_abs_log_moneyness
    ].copy()

    if work.empty:
        raise ValueError(
            "No quotes remain inside the requested moneyness range."
        )

    work["_otm_preferred"] = (
        (
            (
                work[
                    "log_forward_moneyness"
                ]
                < 0
            )
            & (
                work[
                    "option_type"
                ]
                == "put"
            )
        )
        |
        (
            (
                work[
                    "log_forward_moneyness"
                ]
                >= 0
            )
            & (
                work[
                    "option_type"
                ]
                == "call"
            )
        )
    )

    # Put preferred OTM quotes first. If one side is missing, retain the
    # available contract as a fallback.
    work = work.sort_values(
        [
            "expiry",
            "strike",
            "_otm_preferred",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    selected = (
        work.drop_duplicates(
            subset=[
                "expiry",
                "strike",
            ],
            keep="first",
        )
        .copy()
    )

    selected["quote_selection"] = np.where(
        selected["_otm_preferred"],
        "otm",
        "fallback",
    )

    selected = selected.drop(
        columns=[
            "_otm_preferred",
        ]
    )

    duplicates = selected.duplicated(
        subset=[
            "expiry",
            "strike",
        ]
    )

    if duplicates.any():
        raise RuntimeError(
            "Quote preparation did not produce one observation per strike."
        )

    return selected.sort_values(
        [
            "expiry",
            "strike",
        ],
        ignore_index=True,
    )


def check_svi_calendar_overlap(
    parameters: pd.DataFrame,
    points: pd.DataFrame,
    grid_size: int = 301,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Check adjacent SVI slices for calendar total-variance crossings.

    The test is performed only on the common observed log-moneyness range
    of each adjacent expiry pair. It therefore avoids interpreting SVI
    wing extrapolation as observed-market evidence.
    """

    if grid_size < 3:
        raise ValueError(
            "grid_size must be at least 3."
        )

    if tolerance < 0:
        raise ValueError(
            "tolerance cannot be negative."
        )

    required_parameters = {
        "expiry",
        "time_to_expiry",
        "a",
        "b",
        "rho",
        "m",
        "sigma",
    }

    missing_parameters = (
        required_parameters
        - set(parameters.columns)
    )

    if missing_parameters:
        raise ValueError(
            "Parameter table is missing columns: "
            + ", ".join(
                sorted(
                    missing_parameters
                )
            )
        )

    required_points = {
        "expiry",
        "log_forward_moneyness",
    }

    missing_points = (
        required_points
        - set(points.columns)
    )

    if missing_points:
        raise ValueError(
            "Fitted points are missing columns: "
            + ", ".join(
                sorted(
                    missing_points
                )
            )
        )

    ordered = parameters.sort_values(
        "time_to_expiry",
        ignore_index=True,
    )

    rows: list[
        dict[str, object]
    ] = []

    for index in range(
        len(ordered) - 1
    ):
        earlier = ordered.iloc[
            index
        ]

        later = ordered.iloc[
            index + 1
        ]

        earlier_expiry = earlier[
            "expiry"
        ]

        later_expiry = later[
            "expiry"
        ]

        earlier_points = points[
            points["expiry"]
            == earlier_expiry
        ]

        later_points = points[
            points["expiry"]
            == later_expiry
        ]

        if (
            earlier_points.empty
            or later_points.empty
        ):
            rows.append(
                {
                    "earlier_expiry": earlier_expiry,
                    "later_expiry": later_expiry,
                    "earlier_time_to_expiry": float(
                        earlier[
                            "time_to_expiry"
                        ]
                    ),
                    "later_time_to_expiry": float(
                        later[
                            "time_to_expiry"
                        ]
                    ),
                    "overlap_k_min": np.nan,
                    "overlap_k_max": np.nan,
                    "overlap_width": np.nan,
                    "minimum_total_variance_change": np.nan,
                    "k_at_minimum_change": np.nan,
                    "calendar_check_passed": False,
                    "has_observed_overlap": False,
                }
            )

            continue

        earlier_min = float(
            earlier_points[
                "log_forward_moneyness"
            ].min()
        )

        earlier_max = float(
            earlier_points[
                "log_forward_moneyness"
            ].max()
        )

        later_min = float(
            later_points[
                "log_forward_moneyness"
            ].min()
        )

        later_max = float(
            later_points[
                "log_forward_moneyness"
            ].max()
        )

        overlap_min = max(
            earlier_min,
            later_min,
        )

        overlap_max = min(
            earlier_max,
            later_max,
        )

        if overlap_max <= overlap_min:
            rows.append(
                {
                    "earlier_expiry": earlier_expiry,
                    "later_expiry": later_expiry,
                    "earlier_time_to_expiry": float(
                        earlier[
                            "time_to_expiry"
                        ]
                    ),
                    "later_time_to_expiry": float(
                        later[
                            "time_to_expiry"
                        ]
                    ),
                    "overlap_k_min": overlap_min,
                    "overlap_k_max": overlap_max,
                    "overlap_width": max(
                        0.0,
                        overlap_max
                        - overlap_min,
                    ),
                    "minimum_total_variance_change": np.nan,
                    "k_at_minimum_change": np.nan,
                    "calendar_check_passed": False,
                    "has_observed_overlap": False,
                }
            )

            continue

        grid = np.linspace(
            overlap_min,
            overlap_max,
            grid_size,
        )

        earlier_parameters = (
            SVIParameters(
                a=float(
                    earlier["a"]
                ),
                b=float(
                    earlier["b"]
                ),
                rho=float(
                    earlier["rho"]
                ),
                m=float(
                    earlier["m"]
                ),
                sigma=float(
                    earlier["sigma"]
                ),
            )
        )

        later_parameters = (
            SVIParameters(
                a=float(
                    later["a"]
                ),
                b=float(
                    later["b"]
                ),
                rho=float(
                    later["rho"]
                ),
                m=float(
                    later["m"]
                ),
                sigma=float(
                    later["sigma"]
                ),
            )
        )

        earlier_variance = np.asarray(
            svi_total_variance(
                grid,
                earlier_parameters,
            ),
            dtype=float,
        )

        later_variance = np.asarray(
            svi_total_variance(
                grid,
                later_parameters,
            ),
            dtype=float,
        )

        change = (
            later_variance
            - earlier_variance
        )

        minimum_index = int(
            np.argmin(change)
        )

        minimum_change = float(
            change[
                minimum_index
            ]
        )

        rows.append(
            {
                "earlier_expiry": earlier_expiry,
                "later_expiry": later_expiry,
                "earlier_time_to_expiry": float(
                    earlier[
                        "time_to_expiry"
                    ]
                ),
                "later_time_to_expiry": float(
                    later[
                        "time_to_expiry"
                    ]
                ),
                "overlap_k_min": overlap_min,
                "overlap_k_max": overlap_max,
                "overlap_width": (
                    overlap_max
                    - overlap_min
                ),
                "minimum_total_variance_change": minimum_change,
                "k_at_minimum_change": float(
                    grid[
                        minimum_index
                    ]
                ),
                "calendar_check_passed": bool(
                    minimum_change
                    >= -tolerance
                ),
                "has_observed_overlap": True,
            }
        )

    return pd.DataFrame(
        rows
    )
