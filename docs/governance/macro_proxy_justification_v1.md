# Macro Proxy Justification v1
Date: 2026-02-23
Status: CP1 Contract Freeze (Week 3 Day 1)

## Scope
This document records the explicit proxy inclusion/exclusion decisions required by Section 5.2 for the Macro Monitor Agent.

## Decisions

| Proxy | Decision | Rationale |
| :--- | :--- | :--- |
| `BRENT_CRUDE` | Included | Brent materially affects India's energy import bill, inflation pass-through, and INR pressure, so it is retained as an accepted `MacroIndicatorType`. |
| `DXY` | Excluded (for Week 3 catalog) | DXY is not required for the Week 3 macro publish set. India FX/macro sensitivity is covered by `INR_USD` and `INDIA_US_10Y_SPREAD`, avoiding redundant proxy overlap in the initial contract freeze. |

## Contract Impact
- No new indicator is introduced for DXY in schema v1.1.
- `BRENT_CRUDE` remains in the accepted enum set for compatibility.
- Week 3 required publish set remains: `CPI`, `WPI`, `IIP`, `FII_FLOW`, `DII_FLOW`, `FX_RESERVES`, `RBI_BULLETIN`, `INDIA_US_10Y_SPREAD`.

## Review Trigger
Revisit DXY inclusion if:
- INR volatility regime materially changes, or
- Downstream model diagnostics show incremental signal value beyond `INR_USD` + `INDIA_US_10Y_SPREAD`.
