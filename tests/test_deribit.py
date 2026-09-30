import numpy as np
import pandas as pd

from options_lab.deribit import (
    build_deribit_option_snapshot,
)


def test_build_deribit_option_snapshot() -> None:
    expiry = pd.Timestamp(
        "2026-12-31T08:00:00Z"
    )

    expiration_timestamp = int(
        expiry.timestamp() * 1000
    )

    instruments = [
        {
            "instrument_name": "BTC-31DEC26-100000-C",
            "expiration_timestamp": expiration_timestamp,
            "option_type": "call",
            "strike": 100000.0,
            "is_active": True,
            "state": "open",
        },
        {
            "instrument_name": "BTC-31DEC26-100000-P",
            "expiration_timestamp": expiration_timestamp,
            "option_type": "put",
            "strike": 100000.0,
            "is_active": True,
            "state": "open",
        },
    ]

    summaries = [
        {
            "instrument_name": "BTC-31DEC26-100000-C",
            "mark_iv": 60.0,
            "underlying_price": 110000.0,
            "interest_rate": 0.02,
            "bid_price": 0.10,
            "ask_price": 0.11,
            "mid_price": 0.105,
            "mark_price": 0.106,
            "open_interest": 20.0,
            "volume": 5.0,
            "base_currency": "BTC",
        },
        {
            "instrument_name": "BTC-31DEC26-100000-P",
            "mark_iv": 62.0,
            "underlying_price": 110000.0,
            "interest_rate": 0.02,
            "bid_price": 0.08,
            "ask_price": 0.09,
            "mid_price": 0.085,
            "mark_price": 0.086,
            "open_interest": 15.0,
            "volume": 4.0,
            "base_currency": "BTC",
        },
    ]

    snapshot = (
        build_deribit_option_snapshot(
            instruments,
            summaries,
            snapshot_time="2026-09-30T12:00:00Z",
        )
    )

    assert len(snapshot) == 2

    np.testing.assert_allclose(
        snapshot[
            "implied_volatility"
        ].to_numpy(),
        [0.60, 0.62],
    )

    expected_moneyness = np.log(
        100000.0 / 110000.0
    )

    np.testing.assert_allclose(
        snapshot[
            "log_forward_moneyness"
        ].to_numpy(),
        expected_moneyness,
    )

    assert snapshot[
        "valid_svi_quote"
    ].all()

    assert (
        snapshot[
            "quoted_price_unit"
        ]
        == "BTC"
    ).all()

    assert (
        snapshot[
            "time_to_expiry"
        ]
        > 0
    ).all()
