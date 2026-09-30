"""CRR convergence and American early-exercise experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from options_lab.binomial import (
    crr_binomial_tree,
)
from options_lab.black_scholes import (
    black_scholes_price,
)


ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def convergence_experiment() -> pd.DataFrame:
    steps_grid = [
        25,
        50,
        100,
        200,
        400,
        800,
    ]

    call_benchmark = float(
        black_scholes_price(
            100.0,
            100.0,
            0.05,
            1.0,
            0.20,
            "call",
        )
    )

    put_benchmark = float(
        black_scholes_price(
            100.0,
            100.0,
            0.05,
            1.0,
            0.20,
            "put",
        )
    )

    rows = []

    for steps in steps_grid:
        call = crr_binomial_tree(
            100.0,
            100.0,
            0.05,
            1.0,
            0.20,
            "call",
            steps=steps,
            exercise_style="european",
        )

        put = crr_binomial_tree(
            100.0,
            100.0,
            0.05,
            1.0,
            0.20,
            "put",
            steps=steps,
            exercise_style="european",
        )

        rows.append(
            {
                "steps": steps,
                "crr_call": call.price,
                "black_scholes_call": call_benchmark,
                "call_absolute_error": abs(
                    call.price
                    - call_benchmark
                ),
                "crr_put": put.price,
                "black_scholes_put": put_benchmark,
                "put_absolute_error": abs(
                    put.price
                    - put_benchmark
                ),
                "risk_neutral_probability": (
                    call.risk_neutral_probability
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def scenario_experiment() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    scenarios = [
        {
            "scenario": "non_dividend_call",
            "spot": 100.0,
            "strike": 100.0,
            "rate": 0.05,
            "time_to_expiry": 1.0,
            "volatility": 0.20,
            "option_type": "call",
            "dividend_yield": 0.0,
        },
        {
            "scenario": "deep_itm_put",
            "spot": 80.0,
            "strike": 100.0,
            "rate": 0.05,
            "time_to_expiry": 1.0,
            "volatility": 0.25,
            "option_type": "put",
            "dividend_yield": 0.0,
        },
        {
            "scenario": "dividend_call",
            "spot": 120.0,
            "strike": 100.0,
            "rate": 0.02,
            "time_to_expiry": 1.0,
            "volatility": 0.20,
            "option_type": "call",
            "dividend_yield": 0.08,
        },
    ]

    rows = []
    put_boundary_rows = []
    call_boundary_rows = []

    for scenario in scenarios:
        common = dict(
            spot=scenario["spot"],
            strike=scenario["strike"],
            rate=scenario["rate"],
            time_to_expiry=scenario[
                "time_to_expiry"
            ],
            volatility=scenario[
                "volatility"
            ],
            option_type=scenario[
                "option_type"
            ],
            steps=800,
            dividend_yield=scenario[
                "dividend_yield"
            ],
        )

        european = crr_binomial_tree(
            **common,
            exercise_style="european",
        )

        american = crr_binomial_tree(
            **common,
            exercise_style="american",
        )

        rows.append(
            {
                **scenario,
                "steps": 800,
                "european_price": european.price,
                "american_price": american.price,
                "early_exercise_premium": (
                    american.price
                    - european.price
                ),
                "boundary_points": len(
                    american.early_exercise
                ),
            }
        )

        boundary_rows = [
            {
                "scenario": scenario[
                    "scenario"
                ],
                "step": point.step,
                "time": point.time,
                "time_to_expiry": (
                    scenario[
                        "time_to_expiry"
                    ]
                    - point.time
                ),
                "boundary_spot": (
                    point.boundary_spot
                ),
                "exercise_node_count": (
                    point.exercise_node_count
                ),
                "strike": scenario[
                    "strike"
                ],
            }
            for point
            in american.early_exercise
        ]

        if (
            scenario["scenario"]
            == "deep_itm_put"
        ):
            put_boundary_rows.extend(
                boundary_rows
            )

        if (
            scenario["scenario"]
            == "dividend_call"
        ):
            call_boundary_rows.extend(
                boundary_rows
            )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(
            put_boundary_rows
        ),
        pd.DataFrame(
            call_boundary_rows
        ),
    )


def save_convergence_figure(
    convergence: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        convergence["steps"],
        convergence[
            "call_absolute_error"
        ],
        marker="o",
        label="European call",
    )

    ax.plot(
        convergence["steps"],
        convergence[
            "put_absolute_error"
        ],
        marker="o",
        label="European put",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        "CRR time steps"
    )

    ax.set_ylabel(
        "Absolute pricing error"
    )

    ax.set_title(
        "CRR convergence to Black-Scholes"
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "crr_convergence.png",
        dpi=180,
    )

    plt.close(fig)


def save_boundary_figure(
    put_boundary: pd.DataFrame,
    call_boundary: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    if not put_boundary.empty:
        ax.plot(
            put_boundary[
                "time_to_expiry"
            ],
            put_boundary[
                "boundary_spot"
            ],
            label="American put boundary",
        )

    if not call_boundary.empty:
        ax.plot(
            call_boundary[
                "time_to_expiry"
            ],
            call_boundary[
                "boundary_spot"
            ],
            label=(
                "Dividend-paying "
                "American call boundary"
            ),
        )

    ax.axhline(
        100.0,
        linewidth=1.0,
        linestyle="--",
        label="Strike",
    )

    ax.set_xlabel(
        "Time remaining to expiry"
    )

    ax.set_ylabel(
        "Boundary spot"
    )

    ax.set_title(
        "Discrete CRR early-exercise boundaries"
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "american_exercise_boundaries.png",
        dpi=180,
    )

    plt.close(fig)


def main() -> None:
    convergence = (
        convergence_experiment()
    )

    (
        scenarios,
        put_boundary,
        call_boundary,
    ) = scenario_experiment()

    convergence_path = (
        TABLE_DIR
        / "crr_convergence.csv"
    )

    scenario_path = (
        TABLE_DIR
        / "american_option_scenarios.csv"
    )

    put_boundary_path = (
        TABLE_DIR
        / "american_put_boundary.csv"
    )

    call_boundary_path = (
        TABLE_DIR
        / "american_dividend_call_boundary.csv"
    )

    convergence.to_csv(
        convergence_path,
        index=False,
    )

    scenarios.to_csv(
        scenario_path,
        index=False,
    )

    put_boundary.to_csv(
        put_boundary_path,
        index=False,
    )

    call_boundary.to_csv(
        call_boundary_path,
        index=False,
    )

    save_convergence_figure(
        convergence
    )

    save_boundary_figure(
        put_boundary,
        call_boundary,
    )

    print(
        "=== CRR CONVERGENCE ==="
    )

    print(
        convergence.to_string(
            index=False
        )
    )

    print()
    print(
        "=== AMERICAN OPTION SCENARIOS ==="
    )

    print(
        scenarios[
            [
                "scenario",
                "option_type",
                "dividend_yield",
                "european_price",
                "american_price",
                "early_exercise_premium",
                "boundary_points",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "American put boundary points:",
        len(put_boundary),
    )

    print(
        "Dividend call boundary points:",
        len(call_boundary),
    )


if __name__ == "__main__":
    main()
