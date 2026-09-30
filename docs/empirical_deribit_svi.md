# Empirical Deribit BTC SVI calibration

This extension moves the volatility-surface part of the project from controlled synthetic smiles to a frozen public BTC option-chain snapshot from Deribit.

Snapshot time: **2026-09-30 18:00:39 UTC**

## Data preparation

The workflow combines public Deribit option instrument metadata with the option book-summary endpoint. Exchange mark implied volatility is converted from percentage points to decimal volatility.

Calls and puts at the same strike are not treated as two independent smile observations. A common median underlying/forward reference is computed for each expiry, then one out-of-the-money contract is retained per strike:

- put when `K < F`;
- call when `K >= F`;
- the available side is retained as a fallback if the preferred side is absent.

The empirical calibration excludes expiries with less than seven days remaining and restricts observations to `|ln(K/F)| <= 1.25`.

## Raw SVI calibration

Each expiry is fitted independently in total-variance space using:

`w(k) = a + b [rho (k - m) + sqrt((k - m)^2 + sigma^2)]`

where `k = ln(K/F)` and `w(k) = sigma_IV(k)^2 T`.

The numerical parameterization enforces positive minimum total variance, positive `b` and `sigma`, `|rho| < 1`, and bounded asymptotic wing slopes.

## Arbitrage diagnostics

### Butterfly diagnostic

Within each expiry, the fitted SVI curve is evaluated on a dense log-moneyness grid using the standard SVI `g(k)` density diagnostic. A negative value indicates a local butterfly-arbitrage violation.

### Calendar diagnostic

Adjacent expiries are compared in total-variance space only over the log-moneyness interval jointly represented by both observed smile slices.

This avoids interpreting unconstrained parametric wing extrapolation as direct evidence of an observed market arbitrage.

These are numerical diagnostics, not a formal proof that the complete continuous surface is globally arbitrage-free.

## Frozen snapshot results

- Raw option contracts downloaded: **948**
- Unique OTM expiry/strike observations used: **355**
- Duplicate expiry/strike calibration rows: **0**
- Expiries fitted: **8**
- Maximum per-expiry IV RMSE: **0.005296** (0.530 volatility percentage points)
- Minimum butterfly `g(k)`: **0.014997**
- All butterfly diagnostics passed: **True**
- Adjacent-expiry calendar comparisons: **7**
- Minimum total-variance increase on common observed support: **0.002013**
- All observed-support calendar checks passed: **True**

## Per-expiry fit quality

| Expiry | Observations | IV RMSE | Min butterfly g | Butterfly check |
|---|---:|---:|---:|:---:|
| 2026-10-09 | 26 | 0.004196 | 0.116419 | pass |
| 2026-10-16 | 20 | 0.001362 | 0.098834 | pass |
| 2026-10-30 | 56 | 0.005296 | 0.194147 | pass |
| 2026-11-27 | 48 | 0.001184 | 0.014997 | pass |
| 2026-12-25 | 58 | 0.002859 | 0.058152 | pass |
| 2027-03-26 | 52 | 0.002111 | 0.120834 | pass |
| 2027-06-25 | 54 | 0.000862 | 0.155366 | pass |
| 2027-09-24 | 41 | 0.000680 | 0.261971 | pass |

## Interpretation

The fitted smiles reproduce the exchange mark-IV cross-sections closely, but fitting error alone is not sufficient to judge a volatility surface. Butterfly and calendar diagnostics are therefore part of the calibration workflow.

During development, a fixed wide-moneyness calendar comparison produced a large apparent crossing between two short-dated SVI slices. The crossing disappeared when the comparison was restricted to strikes jointly represented by both expiries. This demonstrates why SVI extrapolation outside observed strike support should be distinguished from market data.

Individual raw-SVI parameters should also be interpreted cautiously. Different parameter combinations can represent similar observed smiles, especially when strike support is narrow. The emphasis here is therefore on fitted total variance, residuals, and arbitrage diagnostics.

## Reproduction

The repository stores the frozen raw snapshot, fitted SVI points, parameter table, calendar diagnostics, and generated figure.

Run `python scripts/run_deribit_svi_snapshot.py` to obtain a fresh market snapshot.

Because that command queries live public market data, a future run will not reproduce the numerical values above. The committed raw CSV is the reproducibility anchor for this empirical result.
