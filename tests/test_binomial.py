from math import exp, isclose, sqrt

import pytest

from options_lab.binomial import (
    crr_binomial_price,
    crr_binomial_tree,
)
from options_lab.black_scholes import (
    black_scholes_price,
)


def test_one_step_crr_matches_manual_replication() -> None:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    maturity = 1.0
    volatility = 0.20

    up = exp(
        volatility
        * sqrt(maturity)
    )
    down = 1.0 / up

    probability = (
        exp(rate * maturity)
        - down
    ) / (
        up - down
    )

    up_payoff = max(
        spot * up - strike,
        0.0,
    )

    down_payoff = max(
        spot * down - strike,
        0.0,
    )

    expected = exp(
        -rate * maturity
    ) * (
        probability * up_payoff
        + (
            1.0
            - probability
        )
        * down_payoff
    )

    actual = crr_binomial_price(
        spot,
        strike,
        rate,
        maturity,
        volatility,
        "call",
        steps=1,
    )

    assert isclose(
        actual,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    "option_type",
    [
        "call",
        "put",
    ],
)
def test_european_crr_converges_to_black_scholes(
    option_type: str,
) -> None:
    benchmark = black_scholes_price(
        100.0,
        100.0,
        0.05,
        1.0,
        0.20,
        option_type,
    )

    tree = crr_binomial_price(
        100.0,
        100.0,
        0.05,
        1.0,
        0.20,
        option_type,
        steps=800,
        exercise_style="european",
    )

    assert abs(
        tree - benchmark
    ) < 0.003


def test_non_dividend_american_call_matches_european_tree() -> None:
    european = crr_binomial_tree(
        100.0,
        100.0,
        0.05,
        1.0,
        0.20,
        "call",
        steps=500,
        exercise_style="european",
    )

    american = crr_binomial_tree(
        100.0,
        100.0,
        0.05,
        1.0,
        0.20,
        "call",
        steps=500,
        exercise_style="american",
    )

    assert isclose(
        american.price,
        european.price,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    assert (
        american.early_exercise
        == ()
    )


def test_american_put_has_positive_early_exercise_premium() -> None:
    european = crr_binomial_tree(
        80.0,
        100.0,
        0.05,
        1.0,
        0.25,
        "put",
        steps=500,
        exercise_style="european",
    )

    american = crr_binomial_tree(
        80.0,
        100.0,
        0.05,
        1.0,
        0.25,
        "put",
        steps=500,
        exercise_style="american",
    )

    assert (
        american.price
        > european.price
    )

    assert (
        american.price
        - european.price
        > 1.0
    )

    assert len(
        american.early_exercise
    ) > 0

    assert all(
        point.boundary_spot
        < 100.0
        for point
        in american.early_exercise
    )


def test_dividend_paying_american_call_can_exercise_early() -> None:
    european = crr_binomial_tree(
        120.0,
        100.0,
        0.02,
        1.0,
        0.20,
        "call",
        steps=500,
        exercise_style="european",
        dividend_yield=0.08,
    )

    american = crr_binomial_tree(
        120.0,
        100.0,
        0.02,
        1.0,
        0.20,
        "call",
        steps=500,
        exercise_style="american",
        dividend_yield=0.08,
    )

    assert (
        american.price
        > european.price
    )

    assert len(
        american.early_exercise
    ) > 0

    assert all(
        point.boundary_spot
        > 100.0
        for point
        in american.early_exercise
    )


def test_american_value_is_never_below_european_value() -> None:
    for option_type in [
        "call",
        "put",
    ]:
        european = crr_binomial_price(
            95.0,
            100.0,
            0.04,
            0.75,
            0.30,
            option_type,
            steps=300,
            exercise_style="european",
            dividend_yield=0.02,
        )

        american = crr_binomial_price(
            95.0,
            100.0,
            0.04,
            0.75,
            0.30,
            option_type,
            steps=300,
            exercise_style="american",
            dividend_yield=0.02,
        )

        assert (
            american
            >= european
            - 1e-12
        )


def test_expiry_and_zero_volatility_are_handled() -> None:
    assert (
        crr_binomial_price(
            120.0,
            100.0,
            0.05,
            0.0,
            0.20,
            "call",
        )
        == 20.0
    )

    deterministic = crr_binomial_price(
        100.0,
        100.0,
        0.05,
        1.0,
        0.0,
        "call",
        steps=100,
        exercise_style="european",
    )

    terminal_spot = (
        100.0
        * exp(0.05)
    )

    expected = exp(
        -0.05
    ) * max(
        terminal_spot
        - 100.0,
        0.0,
    )

    assert isclose(
        deterministic,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_invalid_crr_probability_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="probability",
    ):
        crr_binomial_price(
            100.0,
            100.0,
            2.0,
            1.0,
            0.01,
            "call",
            steps=1,
        )
