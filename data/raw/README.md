# Raw option-chain snapshot

This folder is reserved for a user-supplied option-chain snapshot.

The notebook `notebooks/09_real_option_chain_iv_snapshot.ipynb` expects a file named:

```text
option_chain_snapshot.csv
```

Minimum schema:

| column | description |
|---|---|
| `expiry` | option expiry date, e.g. `2026-09-18` |
| `option_type` | `call` or `put` |
| `strike` | strike price |
| `bid` | option bid price |
| `ask` | option ask price |
| `underlying_price` | spot or reference price of the underlying |
| `rate` | continuously compounded risk-free rate, decimal |
| `snapshot_date` | quote snapshot date, e.g. `2026-06-26` |

Optional columns such as volume, open interest, exchange, and quote timestamp can be added and will be preserved by the cleaning workflow.

A real snapshot is not committed here because redistribution rights vary by data vendor. The workflow is intentionally source-agnostic: export a chain from a permitted data source, save it as the expected CSV, and run the notebook.
