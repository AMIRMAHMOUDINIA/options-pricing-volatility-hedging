"""Utilities for inspecting user-supplied option-chain snapshots.

The functions in this module are deliberately source-agnostic. A snapshot can be
exported from any permitted data provider as long as it contains the expected
columns documented in data/raw/README.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from options_lab.greeks import black_scholes_greeks
from options_lab.implied_volatility import implied_volatility_brent

REQUIRED_OPTION_CHAIN_COLUMNS = {
    "expiry",
    "option_type",
    "strike",
    "bid",
    "ask",
    "underlying_price",
    "rate",
    "snapshot_date",
}


@dataclass(frozen=True)
class OptionChainCleaningConfig:
    """Filtering controls for a market option-chain snapshot."""

    minimum_price: float = 0.01
    maximum_relative_spread: float = 0.50
    minimum_time_to_expiry: float = 7.0 / 365.0
    minimum_vega: float = 1e-4


def load_option_chain_snapshot(path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV option-chain snapshot."""

    frame = pd.read_csv(path)
    missing = REQUIRED_OPTION_CHAIN_COLUMNS.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required option-chain columns: {missing_text}")
    return frame.copy()


def prepare_option_chain(frame: pd.DataFrame) -> pd.DataFrame:
    """Create mid prices and time-to-expiry values from a raw snapshot."""

    out = frame.copy()
    out["option_type"] = out["option_type"].str.lower().str.strip()
    out["expiry"] = pd.to_datetime(out["expiry"])
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"])
    out["mid"] = 0.5 * (out["bid"].astype(float) + out["ask"].astype(float))
    out["relative_spread"] = (out["ask"].astype(float) - out["bid"].astype(float)) / out["mid"].replace(0.0, np.nan)
    out["time_to_expiry"] = (out["expiry"] - out["snapshot_date"]).dt.days / 365.0
    return out


def recover_market_implied_volatilities(
    frame: pd.DataFrame,
    config: OptionChainCleaningConfig | None = None,
) -> pd.DataFrame:
    """Clean a snapshot and recover implied volatilities from mid prices."""

    if config is None:
        config = OptionChainCleaningConfig()
    prepared = prepare_option_chain(frame)
    cleaned = prepared[
        (prepared["mid"] >= config.minimum_price)
        & (prepared["relative_spread"] <= config.maximum_relative_spread)
        & (prepared["time_to_expiry"] >= config.minimum_time_to_expiry)
        & (prepared["option_type"].isin(["call", "put"]))
    ].copy()

    implied_vols: list[float] = []
    vegas: list[float] = []
    converged: list[bool] = []
    messages: list[str] = []
    for row in cleaned.itertuples(index=False):
        try:
            result = implied_volatility_brent(
                market_price=float(row.mid),
                spot=float(row.underlying_price),
                strike=float(row.strike),
                rate=float(row.rate),
                time_to_expiry=float(row.time_to_expiry),
                option_type=str(row.option_type),
            )
            greeks = black_scholes_greeks(
                float(row.underlying_price),
                float(row.strike),
                float(row.rate),
                float(row.time_to_expiry),
                float(result.volatility),
                str(row.option_type),
            )
            implied_vols.append(float(result.volatility))
            vegas.append(float(greeks.vega))
            converged.append(bool(result.converged))
            messages.append("ok")
        except Exception as exc:  # noqa: BLE001 - keep row-level diagnostics
            implied_vols.append(np.nan)
            vegas.append(np.nan)
            converged.append(False)
            messages.append(str(exc))
    cleaned["recovered_implied_volatility"] = implied_vols
    cleaned["model_vega"] = vegas
    cleaned["iv_converged"] = converged
    cleaned["iv_message"] = messages
    cleaned = cleaned[(cleaned["iv_converged"]) & (cleaned["model_vega"] >= config.minimum_vega)].copy()
    cleaned["forward"] = cleaned["underlying_price"] * np.exp(cleaned["rate"] * cleaned["time_to_expiry"])
    cleaned["log_forward_moneyness"] = np.log(cleaned["strike"] / cleaned["forward"])
    return cleaned
