"""Public Deribit option-chain snapshots for empirical volatility work."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import requests


DERIBIT_PRODUCTION_API = "https://www.deribit.com/api/v2"


def _rpc_get(
    method: str,
    params: dict[str, Any],
    base_url: str = DERIBIT_PRODUCTION_API,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Call a public Deribit HTTP endpoint.

    The Deribit API exposes JSON-RPC methods through REST-like HTTP
    endpoints. Query parameters are used explicitly here rather than a
    request body on GET so that the request survives proxies and HTTP
    intermediaries consistently.
    """

    url = f"{base_url.rstrip('/')}/{method}"

    try:
        response = requests.get(
            url,
            params=params,
            headers={
                "Accept": "application/json",
                "User-Agent": (
                    "options-pricing-volatility-hedging/1.0"
                ),
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Deribit request failed for {method}: {exc}"
        ) from exc

    if not response.ok:
        body_preview = response.text[:1000]

        raise RuntimeError(
            "Deribit HTTP request failed.\n"
            f"Method: {method}\n"
            f"Status: {response.status_code}\n"
            f"URL: {response.url}\n"
            f"Response: {body_preview}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Deribit returned non-JSON data for {method}."
        ) from exc

    if "error" in body:
        raise RuntimeError(
            f"Deribit API error for {method}: {body['error']}"
        )

    result = body.get("result")

    if not isinstance(result, list):
        raise RuntimeError(
            f"Deribit API did not return a result list for {method}."
        )

    return result


def build_deribit_option_snapshot(
    instruments: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    snapshot_time: object | None = None,
    mark_iv_scale: float = 100.0,
) -> pd.DataFrame:
    """Merge Deribit instrument metadata and option summaries.

    Deribit mark IV is represented in percentage points by the public
    option-chain endpoint. The default scale converts it to decimal
    volatility for the rest of this project.

    Raw bid/ask option prices remain in Deribit's native quoted units.
    They are not passed through the generic Black-Scholes IV inverter.
    """

    if mark_iv_scale <= 0:
        raise ValueError(
            "mark_iv_scale must be positive."
        )

    instrument_frame = pd.DataFrame(
        instruments
    )

    summary_frame = pd.DataFrame(
        summaries
    )

    required_instrument_columns = {
        "instrument_name",
        "expiration_timestamp",
        "option_type",
        "strike",
        "is_active",
        "state",
    }

    missing = (
        required_instrument_columns
        - set(instrument_frame.columns)
    )

    if missing:
        raise ValueError(
            "Deribit instrument metadata is missing: "
            + ", ".join(sorted(missing))
        )

    if "instrument_name" not in summary_frame.columns:
        raise ValueError(
            "Deribit summaries require instrument_name."
        )

    metadata = instrument_frame[
        [
            "instrument_name",
            "expiration_timestamp",
            "option_type",
            "strike",
            "is_active",
            "state",
        ]
    ].copy()

    merged = summary_frame.merge(
        metadata,
        on="instrument_name",
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "Instrument metadata and summaries had no matching option rows."
        )

    if snapshot_time is None:
        snapshot = pd.Timestamp.now(
            tz="UTC"
        )
    else:
        snapshot = pd.Timestamp(
            snapshot_time
        )

        if snapshot.tzinfo is None:
            snapshot = snapshot.tz_localize(
                "UTC"
            )
        else:
            snapshot = snapshot.tz_convert(
                "UTC"
            )

    numeric_columns = [
        "expiration_timestamp",
        "strike",
        "mark_iv",
        "underlying_price",
        "interest_rate",
        "bid_price",
        "ask_price",
        "mid_price",
        "mark_price",
        "open_interest",
        "volume",
    ]

    for column in numeric_columns:
        if column in merged.columns:
            merged[column] = pd.to_numeric(
                merged[column],
                errors="coerce",
            )

    merged["snapshot_time"] = snapshot

    merged["expiry"] = pd.to_datetime(
        merged["expiration_timestamp"],
        unit="ms",
        utc=True,
    )

    merged["time_to_expiry"] = (
        merged["expiry"] - snapshot
    ).dt.total_seconds() / (
        365.0 * 24.0 * 60.0 * 60.0
    )

    merged["implied_volatility"] = (
        merged["mark_iv"]
        / mark_iv_scale
    )

    # Deribit provides the underlying/forward reference used for the
    # option-chain valuation.
    merged["forward"] = merged[
        "underlying_price"
    ]

    merged["log_forward_moneyness"] = np.log(
        merged["strike"]
        / merged["forward"]
    )

    if "base_currency" in merged.columns:
        merged["quoted_price_unit"] = merged[
            "base_currency"
        ]

    merged["valid_svi_quote"] = (
        merged["is_active"].astype(bool)
        & (merged["state"] == "open")
        & (merged["time_to_expiry"] > 0)
        & np.isfinite(merged["strike"])
        & (merged["strike"] > 0)
        & np.isfinite(merged["forward"])
        & (merged["forward"] > 0)
        & np.isfinite(
            merged["implied_volatility"]
        )
        & (
            merged["implied_volatility"]
            > 0
        )
    )

    return merged.sort_values(
        [
            "expiry",
            "strike",
            "option_type",
        ],
        ignore_index=True,
    )


def fetch_deribit_option_snapshot(
    currency: str = "BTC",
    snapshot_time: object | None = None,
    base_url: str = DERIBIT_PRODUCTION_API,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch one public point-in-time Deribit option-chain snapshot."""

    currency = currency.upper()

    instruments = _rpc_get(
        "public/get_instruments",
        {
            "currency": currency,
            "kind": "option",
            "expired": "false",
        },
        base_url=base_url,
        timeout=timeout,
    )

    summaries = _rpc_get(
        "public/get_book_summary_by_currency",
        {
            "currency": currency,
            "kind": "option",
        },
        base_url=base_url,
        timeout=timeout,
    )

    return build_deribit_option_snapshot(
        instruments=instruments,
        summaries=summaries,
        snapshot_time=snapshot_time,
    )
