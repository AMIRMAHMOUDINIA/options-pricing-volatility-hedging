"""Generate the repository's reproducible tables, datasets, and figures."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from options_lab.black_scholes import black_scholes_price
from options_lab.delta_hedging import delta_hedge_path, run_delta_hedging_experiment
from options_lab.greeks import black_scholes_greeks
from options_lab.implied_volatility import (
    implied_volatility_bisection,
    implied_volatility_brent,
    implied_volatility_newton,
)
from options_lab.metrics import summarize_hedging_results
from options_lab.payoffs import call_payoff, option_profit
from options_lab.pnl_attribution import (
    attribute_cash_ledger,
    counterfactual_pnl_attribution,
    gamma_variance_attribution,
    greek_path_attribution,
)
from options_lab.price_paths import regular_rebalance_indices, simulate_gbm_paths
from options_lab.volatility_surface import (
    check_calendar_total_variance,
    check_call_slice_arbitrage,
    generate_synthetic_surface,
    recover_surface_implied_volatility,
)

FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
DATA = ROOT / "data" / "processed"
for directory in (FIGURES, TABLES, DATA):
    directory.mkdir(parents=True, exist_ok=True)


def save_current_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def payoff_and_pricing_outputs() -> None:
    spot_expiry = np.linspace(40.0, 160.0, 481)
    strike = 100.0
    premium = 8.0
    profit = option_profit(call_payoff(spot_expiry, strike), premium, "long")
    plt.figure(figsize=(8, 5))
    plt.plot(spot_expiry, profit)
    plt.axhline(0.0, linewidth=1)
    plt.axvline(strike, linestyle="--", linewidth=1)
    plt.xlabel("Underlying price at expiration")
    plt.ylabel("Profit per underlying unit")
    plt.title("Long Call Profit at Expiration")
    save_current_figure("01_long_call_profit.png")

    spot_grid = np.linspace(50.0, 150.0, 401)
    call_values = black_scholes_price(spot_grid, 100.0, 0.03, 1.0, 0.20, "call")
    plt.figure(figsize=(8, 5))
    plt.plot(spot_grid, call_values)
    plt.xlabel("Current underlying price")
    plt.ylabel("European call value")
    plt.title("Black–Scholes Call Value versus Spot")
    save_current_figure("02_call_value_vs_spot.png")

    call_greeks = black_scholes_greeks(spot_grid, 100.0, 0.03, 1.0, 0.20, "call")
    plt.figure(figsize=(8, 5))
    plt.plot(spot_grid, call_greeks.delta)
    plt.xlabel("Current underlying price")
    plt.ylabel("Call delta")
    plt.title("Call Delta versus Spot")
    save_current_figure("03_call_delta_vs_spot.png")

    plt.figure(figsize=(8, 5))
    plt.plot(spot_grid, call_greeks.gamma)
    plt.xlabel("Current underlying price")
    plt.ylabel("Gamma")
    plt.title("Option Gamma versus Spot")
    save_current_figure("04_gamma_vs_spot.png")


def implied_volatility_outputs() -> None:
    rows = []
    for true_vol in (0.10, 0.20, 0.40, 0.80):
        price = float(black_scholes_price(100.0, 100.0, 0.03, 1.0, true_vol, "call"))
        for method, solver in (
            ("bisection", implied_volatility_bisection),
            ("newton", implied_volatility_newton),
            ("brent", implied_volatility_brent),
        ):
            kwargs = dict(
                market_price=price,
                spot=100.0,
                strike=100.0,
                rate=0.03,
                time_to_expiry=1.0,
                option_type="call",
            )
            if method == "newton":
                kwargs["initial_guess"] = 0.30
            result = solver(**kwargs)
            rows.append({
                "true_volatility": true_vol,
                "method": method,
                "recovered_volatility": result.volatility,
                "absolute_error": abs(result.volatility - true_vol),
                "iterations": result.iterations,
                "price_residual": result.residual,
                "converged": result.converged,
            })
    pd.DataFrame(rows).to_csv(TABLES / "iv_solver_comparison.csv", index=False)


def surface_outputs() -> None:
    surface = generate_synthetic_surface(
        spot=100.0,
        rate=0.03,
        log_moneyness_grid=np.linspace(-0.40, 0.40, 17),
        maturities=[30 / 365, 90 / 365, 180 / 365, 1.0, 2.0],
    )
    surface.to_csv(DATA / "synthetic_option_surface.csv", index=False)
    recovered_call = recover_surface_implied_volatility(surface, "call_price", "call")
    recovered_put = recover_surface_implied_volatility(surface, "put_price", "put")
    recovery = surface[["time_to_expiry", "log_moneyness", "strike", "implied_volatility"]].copy()
    recovery["call_recovered_iv"] = recovered_call["recovered_implied_volatility"]
    recovery["put_recovered_iv"] = recovered_put["recovered_implied_volatility"]
    recovery["call_abs_error"] = recovered_call["iv_absolute_error"]
    recovery["put_abs_error"] = recovered_put["iv_absolute_error"]
    recovery["call_put_iv_difference"] = np.abs(recovery["call_recovered_iv"] - recovery["put_recovered_iv"])
    recovery.to_csv(TABLES / "surface_iv_recovery.csv", index=False)
    check_call_slice_arbitrage(surface).to_csv(TABLES / "surface_strike_arbitrage_checks.csv", index=False)
    check_calendar_total_variance(surface).to_csv(TABLES / "surface_calendar_checks.csv", index=False)

    plt.figure(figsize=(8, 5))
    for maturity in sorted(surface["time_to_expiry"].unique()):
        group = surface[surface["time_to_expiry"] == maturity]
        plt.plot(group["log_moneyness"], 100 * group["implied_volatility"], marker="o", label=f"T={maturity:.2f}")
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Log-forward moneyness: ln(K/F)")
    plt.ylabel("Implied volatility (%)")
    plt.title("Synthetic Implied-Volatility Smiles")
    plt.legend()
    save_current_figure("05_volatility_smiles.png")

    atm = surface[np.isclose(surface["log_moneyness"], 0.0)]
    plt.figure(figsize=(8, 5))
    plt.plot(atm["time_to_expiry"], 100 * atm["implied_volatility"], marker="o")
    plt.xlabel("Time to expiry (years)")
    plt.ylabel("ATM-forward implied volatility (%)")
    plt.title("ATM Implied-Volatility Term Structure")
    save_current_figure("06_atm_term_structure.png")

    pivot = surface.pivot(index="time_to_expiry", columns="log_moneyness", values="implied_volatility")
    k_mesh, t_mesh = np.meshgrid(pivot.columns.to_numpy(), pivot.index.to_numpy())
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(k_mesh, t_mesh, 100 * pivot.to_numpy())
    ax.set_xlabel("Log-forward moneyness")
    ax.set_ylabel("Time to expiry")
    ax.set_zlabel("Implied volatility (%)")
    ax.set_title("Synthetic Implied-Volatility Surface")
    plt.savefig(FIGURES / "07_volatility_surface_3d.png", dpi=180, bbox_inches="tight")
    plt.close()


def hedging_outputs() -> None:
    number_of_paths = 60
    time_grid, paths = simulate_gbm_paths(
        spot=100.0,
        drift=0.03,
        realized_volatility=0.20,
        time_to_expiry=1.0,
        number_of_steps=252,
        number_of_paths=number_of_paths,
        seed=2026,
    )
    schedules = {
        "monthly": regular_rebalance_indices(252, 21),
        "weekly": regular_rebalance_indices(252, 5),
        "daily": regular_rebalance_indices(252, 1),
    }
    summaries = []
    results_by_frequency: dict[str, pd.DataFrame] = {}
    for cost_rate, cost_label in ((0.0, "no_cost"), (0.0005, "five_bps")):
        for frequency, indices in schedules.items():
            results = run_delta_hedging_experiment(
                time_grid=time_grid,
                spot_paths=paths,
                strike=100.0,
                rate=0.03,
                implied_volatility=0.20,
                option_type="call",
                position="short",
                transaction_cost_rate=cost_rate,
                rebalance_indices=indices,
            )
            summary = summarize_hedging_results(results).to_dict()
            summary.update({"frequency": frequency, "transaction_cost_rate": cost_rate, "cost_label": cost_label})
            summaries.append(summary)
            if cost_rate == 0.0:
                results_by_frequency[frequency] = results
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(TABLES / "hedge_frequency_summary.csv", index=False)
    for frequency, results in results_by_frequency.items():
        results.to_csv(TABLES / f"hedging_paths_{frequency}_no_cost.csv", index=False)

    no_cost = summary_df[summary_df["transaction_cost_rate"] == 0.0]
    plt.figure(figsize=(8, 5))
    plt.plot(no_cost["frequency"], no_cost["root_mean_squared_pnl"], marker="o")
    plt.xlabel("Hedge frequency")
    plt.ylabel("Terminal P&L RMSE")
    plt.title("Replication Error versus Hedge Frequency")
    save_current_figure("08_hedge_frequency_rmse.png")

    daily_results = results_by_frequency["daily"]
    plt.figure(figsize=(8, 5))
    plt.hist(daily_results["terminal_pnl"], bins=35)
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("Terminal hedging P&L")
    plt.ylabel("Number of paths")
    plt.title("Daily-Hedged Short Call P&L Distribution")
    save_current_figure("09_daily_hedge_pnl_distribution.png")

    selected_path = paths[0]
    selected_result = delta_hedge_path(
        time_grid, selected_path, 100.0, 0.03, 0.20, "call", "short",
        transaction_cost_rate=0.0005,
    )
    selected_result.history.to_csv(TABLES / "selected_hedge_history.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(selected_result.history["time"], selected_result.history["spot"])
    plt.axhline(100.0, linestyle="--", linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Underlying price")
    plt.title("Selected Simulated Price Path")
    save_current_figure("10_selected_price_path.png")

    plt.figure(figsize=(8, 5))
    plt.plot(selected_result.history["time"], selected_result.history["actual_stock_position"], label="Actual hedge")
    plt.plot(selected_result.history["time"], selected_result.history["target_stock_position"], linestyle="--", label="Target hedge")
    plt.xlabel("Time")
    plt.ylabel("Underlying units")
    plt.title("Dynamic Delta-Hedge Position")
    plt.legend()
    save_current_figure("11_selected_hedge_position.png")

    plt.figure(figsize=(8, 5))
    plt.plot(selected_result.history["time"], selected_result.history["portfolio_value"])
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Marked hedged-portfolio value")
    plt.title("Hedged Portfolio Value through Time")
    save_current_figure("12_selected_portfolio_value.png")

    accounting = attribute_cash_ledger(selected_result)
    counterfactual = counterfactual_pnl_attribution(
        time_grid, selected_path, 100.0, 0.03, 0.20, "call", "short",
        transaction_cost_rate=0.0005,
        actual_option_premium=float(selected_result.option_premium_per_unit) + 0.25,
    )
    greek = greek_path_attribution(selected_result, 100.0, 0.03, 0.20, "call", "short")
    model_no_cost = delta_hedge_path(time_grid, selected_path, 100.0, 0.03, 0.20, "call", "short")
    variance = gamma_variance_attribution(model_no_cost, 100.0, 0.03, 0.20, "call", "short")

    pd.DataFrame([
        {"component": "option_entry_cashflow", "value": accounting.option_entry_cashflow},
        {"component": "stock_trading_cashflow", "value": accounting.stock_trading_cashflow},
        {"component": "option_settlement_cashflow", "value": accounting.option_settlement_cashflow},
        {"component": "financing_cashflow", "value": accounting.financing_cashflow},
        {"component": "transaction_cost_pnl", "value": accounting.transaction_cost_pnl},
        {"component": "terminal_pnl", "value": accounting.reported_terminal_pnl},
        {"component": "ledger_reconciliation_error", "value": accounting.reconciliation_error},
        {"component": "model_hedging_pnl", "value": counterfactual.model_hedging_pnl},
        {"component": "premium_edge_at_expiry", "value": counterfactual.premium_edge_at_expiry},
        {"component": "transaction_cost_drag_at_expiry", "value": counterfactual.transaction_cost_drag_at_expiry},
        {"component": "counterfactual_reconstruction_error", "value": counterfactual.reconstruction_error},
        {"component": "greek_taylor_residual", "value": greek.taylor_residual},
        {"component": "greek_reconciliation_error", "value": greek.reconciliation_error},
        {"component": "gamma_variance_pnl", "value": variance.total_variance_pnl},
        {"component": "gamma_variance_residual", "value": variance.residual_pnl},
    ]).to_csv(TABLES / "selected_path_pnl_attribution.csv", index=False)
    greek.intervals.to_csv(TABLES / "selected_path_greek_intervals.csv", index=False)


def realized_volatility_outputs() -> None:
    rows = []
    for realized_volatility in (0.10, 0.15, 0.20, 0.25, 0.30):
        time_grid, paths = simulate_gbm_paths(
            spot=100.0,
            drift=0.03,
            realized_volatility=realized_volatility,
            time_to_expiry=1.0,
            number_of_steps=252,
            number_of_paths=60,
            seed=1000 + int(100 * realized_volatility),
        )
        results = run_delta_hedging_experiment(
            time_grid, paths, 100.0, 0.03, 0.20, "call", "short",
            rebalance_indices=regular_rebalance_indices(252, 1),
        )
        summary = summarize_hedging_results(results).to_dict()
        summary["simulated_realized_volatility"] = realized_volatility
        rows.append(summary)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(TABLES / "realized_vs_implied_volatility_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(100 * summary_df["simulated_realized_volatility"], summary_df["mean_terminal_pnl"], marker="o")
    plt.axhline(0.0, linewidth=1)
    plt.axvline(20.0, linestyle="--", linewidth=1)
    plt.xlabel("Simulated realized volatility (%)")
    plt.ylabel("Mean terminal P&L")
    plt.title("Short-Call Hedging P&L versus Realized Volatility")
    save_current_figure("13_mean_pnl_vs_realized_volatility.png")


def main() -> None:
    payoff_and_pricing_outputs()
    implied_volatility_outputs()
    surface_outputs()
    hedging_outputs()
    realized_volatility_outputs()
    print("Generated reproducible datasets, tables, and figures.")


if __name__ == "__main__":
    main()
