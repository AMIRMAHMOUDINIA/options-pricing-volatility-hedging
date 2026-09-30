"""Fetch a Deribit BTC option chain and fit SVI smiles by expiry."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from options_lab.deribit import (
    fetch_deribit_option_snapshot,
)
from options_lab.empirical_surface import (
    check_svi_calendar_overlap,
    prepare_deribit_svi_quotes,
)
from options_lab.svi import (
    fit_svi_surface,
)


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main() -> None:
    chain = fetch_deribit_option_snapshot(
        currency="BTC"
    )

    raw_path = (
        RAW_DIR
        / "deribit_btc_option_snapshot.csv"
    )

    chain.to_csv(
        raw_path,
        index=False,
    )

    quotes = prepare_deribit_svi_quotes(
        chain,
        minimum_time_to_expiry=7.0 / 365.0,
        maximum_abs_log_moneyness=1.25,
    )

    fit = fit_svi_surface(
        quotes,
        minimum_points=7,
    )

    calendar = check_svi_calendar_overlap(
        fit.parameters,
        fit.points,
    )

    parameters_path = (
        TABLE_DIR
        / "deribit_btc_svi_parameters.csv"
    )

    points_path = (
        TABLE_DIR
        / "deribit_btc_svi_points.csv"
    )

    calendar_path = (
        TABLE_DIR
        / "deribit_btc_svi_calendar.csv"
    )

    fit.parameters.to_csv(
        parameters_path,
        index=False,
    )

    fit.points.to_csv(
        points_path,
        index=False,
    )

    calendar.to_csv(
        calendar_path,
        index=False,
    )

    selected_expiries = (
        fit.parameters
        .sort_values(
            "time_to_expiry"
        )
        .head(6)["expiry"]
        .tolist()
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for expiry in selected_expiries:
        group = (
            fit.points[
                fit.points["expiry"]
                == expiry
            ]
            .sort_values(
                "log_forward_moneyness"
            )
        )

        label = str(
            pd.Timestamp(
                expiry
            ).date()
        )

        ax.scatter(
            group[
                "log_forward_moneyness"
            ],
            100.0
            * group[
                "implied_volatility"
            ],
            s=18,
            alpha=0.65,
        )

        ax.plot(
            group[
                "log_forward_moneyness"
            ],
            100.0
            * group[
                "svi_implied_volatility"
            ],
            linewidth=1.5,
            label=label,
        )

    ax.axvline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Log-forward moneyness: ln(K/F)"
    )

    ax.set_ylabel(
        "Implied volatility (%)"
    )

    ax.set_title(
        "Deribit BTC OTM smiles with raw-SVI fits"
    )

    ax.legend(
        title="Expiry",
        fontsize=8,
    )

    fig.tight_layout()

    figure_path = (
        FIGURE_DIR
        / "deribit_btc_svi_smiles.png"
    )

    fig.savefig(
        figure_path,
        dpi=180,
    )

    plt.close(fig)

    print(
        f"Raw contracts: {len(chain):,}"
    )

    print(
        "OTM observations retained: "
        f"{len(quotes):,}"
    )

    print(
        f"Fitted expiries: {len(fit.parameters):,}"
    )

    print()

    print(
        "=== SVI FIT QUALITY ==="
    )

    print(
        fit.parameters[
            [
                "expiry",
                "time_to_expiry",
                "n_observations",
                "rmse_implied_volatility",
                "minimum_butterfly_g",
                "grid_butterfly_check_passed",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "=== CALENDAR CHECK ON COMMON OBSERVED SUPPORT ==="
    )

    print(
        calendar[
            [
                "earlier_expiry",
                "later_expiry",
                "overlap_k_min",
                "overlap_k_max",
                "minimum_total_variance_change",
                "k_at_minimum_change",
                "calendar_check_passed",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    valid_calendar_rows = calendar[
        calendar[
            "has_observed_overlap"
        ]
    ]

    all_calendar_passed = bool(
        valid_calendar_rows[
            "calendar_check_passed"
        ].all()
    )

    print(
        "All adjacent calendar checks passed "
        "on common observed support:",
        all_calendar_passed,
    )

    print()

    print(
        f"Wrote: {raw_path}"
    )

    print(
        f"Wrote: {parameters_path}"
    )

    print(
        f"Wrote: {points_path}"
    )

    print(
        f"Wrote: {calendar_path}"
    )

    print(
        f"Wrote: {figure_path}"
    )


if __name__ == "__main__":
    main()
