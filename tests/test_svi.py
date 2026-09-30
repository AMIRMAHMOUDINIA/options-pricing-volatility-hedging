import numpy as np
import pandas as pd

from options_lab.svi import (
    SVIParameters,
    fit_svi_smile,
    fit_svi_surface,
    svi_butterfly_g,
    svi_implied_volatility,
    svi_total_variance,
)


def test_svi_recovers_exact_synthetic_smile() -> None:
    parameters = SVIParameters(
        a=0.01,
        b=0.12,
        rho=-0.35,
        m=0.02,
        sigma=0.18,
    )

    maturity = 0.75

    log_moneyness = np.linspace(
        -0.50,
        0.50,
        41,
    )

    implied_volatility = (
        svi_implied_volatility(
            log_moneyness,
            maturity,
            parameters,
        )
    )

    result = fit_svi_smile(
        log_moneyness,
        implied_volatility,
        maturity,
    )

    assert result.success
    assert (
        result.rmse_implied_volatility
        < 1e-8
    )
    assert (
        result.max_abs_iv_error
        < 1e-7
    )
    assert (
        result.grid_butterfly_check_passed
    )


def test_svi_total_variance_and_butterfly_diagnostic() -> None:
    parameters = SVIParameters(
        a=0.015,
        b=0.10,
        rho=-0.25,
        m=0.0,
        sigma=0.20,
    )

    grid = np.linspace(
        -1.0,
        1.0,
        501,
    )

    total_variance = (
        svi_total_variance(
            grid,
            parameters,
        )
    )

    butterfly_g = (
        svi_butterfly_g(
            grid,
            parameters,
        )
    )

    assert np.all(
        total_variance > 0
    )

    assert np.min(
        butterfly_g
    ) > 0


def test_fit_svi_surface_across_two_expiries() -> None:
    rows = []

    specifications = [
        (
            pd.Timestamp(
                "2027-03-26",
                tz="UTC",
            ),
            0.50,
            SVIParameters(
                0.010,
                0.12,
                -0.35,
                0.02,
                0.18,
            ),
        ),
        (
            pd.Timestamp(
                "2027-09-24",
                tz="UTC",
            ),
            1.00,
            SVIParameters(
                0.020,
                0.10,
                -0.25,
                0.00,
                0.22,
            ),
        ),
    ]

    for (
        expiry,
        maturity,
        parameters,
    ) in specifications:
        log_moneyness = np.linspace(
            -0.40,
            0.40,
            17,
        )

        implied_volatility = (
            svi_implied_volatility(
                log_moneyness,
                maturity,
                parameters,
            )
        )

        for k, volatility in zip(
            log_moneyness,
            implied_volatility,
            strict=True,
        ):
            rows.append(
                {
                    "expiry": expiry,
                    "time_to_expiry": maturity,
                    "log_forward_moneyness": float(
                        k
                    ),
                    "implied_volatility": float(
                        volatility
                    ),
                }
            )

    frame = pd.DataFrame(
        rows
    )

    fit = fit_svi_surface(
        frame,
        minimum_points=7,
    )

    assert len(
        fit.parameters
    ) == 2

    assert fit.parameters[
        "optimizer_success"
    ].all()

    assert fit.parameters[
        "grid_butterfly_check_passed"
    ].all()

    assert (
        fit.parameters[
            "rmse_implied_volatility"
        ].max()
        < 1e-7
    )

    assert (
        fit.points[
            "svi_iv_residual"
        ].abs().max()
        < 1e-6
    )
