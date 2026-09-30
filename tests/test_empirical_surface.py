import numpy as np
import pandas as pd

from options_lab.empirical_surface import (
    check_svi_calendar_overlap,
    prepare_deribit_svi_quotes,
)


def test_prepare_deribit_quotes_uses_one_otm_quote_per_strike() -> None:
    expiry = pd.Timestamp(
        "2027-03-26T08:00:00Z"
    )

    frame = pd.DataFrame(
        [
            {
                "expiry": expiry,
                "strike": 90.0,
                "option_type": "call",
                "underlying_price": 100.1,
                "time_to_expiry": 0.5,
                "implied_volatility": 0.31,
                "valid_svi_quote": True,
            },
            {
                "expiry": expiry,
                "strike": 90.0,
                "option_type": "put",
                "underlying_price": 99.9,
                "time_to_expiry": 0.5,
                "implied_volatility": 0.31,
                "valid_svi_quote": True,
            },
            {
                "expiry": expiry,
                "strike": 110.0,
                "option_type": "call",
                "underlying_price": 100.2,
                "time_to_expiry": 0.5,
                "implied_volatility": 0.32,
                "valid_svi_quote": True,
            },
            {
                "expiry": expiry,
                "strike": 110.0,
                "option_type": "put",
                "underlying_price": 99.8,
                "time_to_expiry": 0.5,
                "implied_volatility": 0.32,
                "valid_svi_quote": True,
            },
        ]
    )

    selected = (
        prepare_deribit_svi_quotes(
            frame,
            minimum_time_to_expiry=0.01,
            maximum_abs_log_moneyness=1.0,
        )
    )

    assert len(selected) == 2

    assert (
        selected[
            "calibration_forward"
        ].nunique()
        == 1
    )

    lower = selected[
        selected["strike"] == 90.0
    ].iloc[0]

    upper = selected[
        selected["strike"] == 110.0
    ].iloc[0]

    assert lower[
        "option_type"
    ] == "put"

    assert upper[
        "option_type"
    ] == "call"

    assert (
        selected[
            "quote_selection"
        ]
        == "otm"
    ).all()


def test_calendar_check_uses_common_observed_support() -> None:
    expiry_1 = pd.Timestamp(
        "2027-03-26T08:00:00Z"
    )

    expiry_2 = pd.Timestamp(
        "2027-09-24T08:00:00Z"
    )

    parameters = pd.DataFrame(
        [
            {
                "expiry": expiry_1,
                "time_to_expiry": 0.5,
                "a": 0.020,
                "b": 0.080,
                "rho": -0.20,
                "m": 0.0,
                "sigma": 0.20,
            },
            {
                "expiry": expiry_2,
                "time_to_expiry": 1.0,
                "a": 0.040,
                "b": 0.080,
                "rho": -0.20,
                "m": 0.0,
                "sigma": 0.20,
            },
        ]
    )

    points = pd.DataFrame(
        {
            "expiry": (
                [expiry_1] * 5
                + [expiry_2] * 5
            ),
            "log_forward_moneyness": (
                list(
                    np.linspace(
                        -0.30,
                        0.30,
                        5,
                    )
                )
                + list(
                    np.linspace(
                        -0.20,
                        0.40,
                        5,
                    )
                )
            ),
        }
    )

    report = (
        check_svi_calendar_overlap(
            parameters,
            points,
        )
    )

    assert len(report) == 1

    row = report.iloc[0]

    assert np.isclose(
        row[
            "overlap_k_min"
        ],
        -0.20,
    )

    assert np.isclose(
        row[
            "overlap_k_max"
        ],
        0.30,
    )

    assert bool(
        row[
            "has_observed_overlap"
        ]
    )

    assert bool(
        row[
            "calendar_check_passed"
        ]
    )

    assert (
        row[
            "minimum_total_variance_change"
        ]
        > 0
    )
