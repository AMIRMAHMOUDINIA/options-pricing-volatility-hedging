"""Validate analytical vanna and volga against finite differences."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from options_lab.higher_order_greeks import (
    black_scholes_higher_order_greeks,
)
from options_lab.numerical_higher_order_greeks import (
    numerical_higher_order_greeks,
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


def build_validation_table() -> pd.DataFrame:
    rows = []

    for option_type in [
        "call",
        "put",
    ]:
        for strike in [
            70.0,
            80.0,
            90.0,
            100.0,
            110.0,
            120.0,
            130.0,
        ]:
            analytical = (
                black_scholes_higher_order_greeks(
                    spot=100.0,
                    strike=strike,
                    rate=0.03,
                    time_to_expiry=0.75,
                    volatility=0.25,
                    option_type=option_type,
                )
            )

            numerical = (
                numerical_higher_order_greeks(
                    spot=100.0,
                    strike=strike,
                    rate=0.03,
                    time_to_expiry=0.75,
                    volatility=0.25,
                    option_type=option_type,
                )
            )

            rows.append(
                {
                    "option_type": option_type,
                    "spot": 100.0,
                    "strike": strike,
                    "rate": 0.03,
                    "time_to_expiry": 0.75,
                    "volatility": 0.25,
                    "analytical_vanna": (
                        analytical.vanna
                    ),
                    "numerical_vanna": (
                        numerical.vanna
                    ),
                    "absolute_vanna_error": abs(
                        analytical.vanna
                        - numerical.vanna
                    ),
                    "analytical_volga": (
                        analytical.volga
                    ),
                    "numerical_volga": (
                        numerical.volga
                    ),
                    "absolute_volga_error": abs(
                        analytical.volga
                        - numerical.volga
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def save_vanna_figure(
    table: pd.DataFrame,
) -> None:
    call = table[
        table["option_type"]
        == "call"
    ].sort_values(
        "strike"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        call["strike"],
        call[
            "analytical_vanna"
        ],
        marker="o",
        label="Analytical",
    )

    ax.plot(
        call["strike"],
        call[
            "numerical_vanna"
        ],
        marker="x",
        linestyle="--",
        label="Finite difference",
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Strike"
    )

    ax.set_ylabel(
        "Vanna"
    )

    ax.set_title(
        "Black-Scholes vanna validation"
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "vanna_validation.png",
        dpi=180,
    )

    plt.close(fig)


def save_volga_figure(
    table: pd.DataFrame,
) -> None:
    call = table[
        table["option_type"]
        == "call"
    ].sort_values(
        "strike"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        call["strike"],
        call[
            "analytical_volga"
        ],
        marker="o",
        label="Analytical",
    )

    ax.plot(
        call["strike"],
        call[
            "numerical_volga"
        ],
        marker="x",
        linestyle="--",
        label="Finite difference",
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Strike"
    )

    ax.set_ylabel(
        "Volga / vomma"
    )

    ax.set_title(
        "Black-Scholes volga validation"
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "volga_validation.png",
        dpi=180,
    )

    plt.close(fig)


def main() -> None:
    table = (
        build_validation_table()
    )

    table_path = (
        TABLE_DIR
        / "higher_order_greeks_validation.csv"
    )

    table.to_csv(
        table_path,
        index=False,
    )

    save_vanna_figure(
        table
    )

    save_volga_figure(
        table
    )

    print(
        "=== HIGHER-ORDER GREEK VALIDATION ==="
    )

    print(
        table.to_string(
            index=False
        )
    )

    print()

    print(
        "Maximum absolute vanna error:",
        table[
            "absolute_vanna_error"
        ].max(),
    )

    print(
        "Maximum absolute volga error:",
        table[
            "absolute_volga_error"
        ].max(),
    )

    print()
    print(
        f"Wrote: {table_path}"
    )

    print(
        "Wrote:",
        FIGURE_DIR
        / "vanna_validation.png",
    )

    print(
        "Wrote:",
        FIGURE_DIR
        / "volga_validation.png",
    )


if __name__ == "__main__":
    main()
