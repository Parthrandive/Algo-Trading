# Macro Backfill Notes (India Features)

## Scope
Backfill and join behavior was updated for these model-facing macro features:
- `FII_FLOW`
- `DII_FLOW`
- `WPI`
- `CPI`
- `IIP`
- `FX_RESERVES`
- `INDIA_US_10Y_SPREAD`
- `RBI_BULLETIN`

## Source Routing
- `CPI`
  - Primary probe: MOSPI site (`https://www.mospi.gov.in/`)
  - Historical fallback: FRED series `INDCPIALLMINMEI`
- `WPI`
  - Primary probe: OEA site (`https://eaindustry.nic.in/`)
  - Historical fallback: FRED series `WPIATT01INM661N`
- `IIP`
  - Primary probe: MOSPI site (`https://www.mospi.gov.in/`)
  - Historical fallback: FRED series `INDPRINTO01IXOBM`
- `FX_RESERVES`
  - Primary: RBI WSS scrape (latest official reading)
  - Historical fallback: FRED series `TRESEGINM052N` (converted from USD mn to USD bn)
- `INDIA_US_10Y_SPREAD`
  - Historical fallback legs:
    - India 10Y: FRED `INDIRLTLT01STM`
    - US 10Y: FRED `DGS10`
  - Spread formula: `(india_10y_percent - us_10y_percent) * 100` (bps)
- `RBI_BULLETIN`
  - Primary: RBI bulletin archive (`BS_ViewBulletin.aspx`) using year/month archive traversal
- `FII_FLOW` / `DII_FLOW`
  - Primary: NSE APIs
    - `fiidiiTradeReact`
    - `fiidiiTradeNse` fallback endpoint
  - Limitation: NSE endpoint is latest-day oriented; deep historical range is not guaranteed from this route.

## Join and Leakage Controls
- Macro joins on hourly bars now use **release-aware as-of logic**.
- Effective timestamp is:
  - `release_date` when available
  - otherwise `observation_timestamp + configured publication delay`
- Join uses backward as-of only, so:
  - no future leakage
  - no pre-release fill
  - no backfill before first real observation

## Anti-Fabrication Rules
- Simulated/fake macro values were removed from targeted ingestion paths.
- Missing macro history is not forced to `0.0` in raw macro columns.
- `vwap` is dropped from model input if all-null.

## Feature Gating Rule
- Configurable threshold: `MACRO_FEATURE_COVERAGE_THRESHOLD` (default `0.60` = 60%).
- Per-feature report includes:
  - first available date
  - last available date
  - row count
  - coverage % across model window
  - missing values after join
  - `train_ready`
- Features with `train_ready=false` are excluded from model input (set missing in join output) but remain in raw storage.

## Known Limitations
- `FII_FLOW` and `DII_FLOW` may remain under-covered for long backfills if upstream only serves latest records.
- `WPI` fallback history can be stale at the tail in some datasets; gating will automatically mark it not train-ready when coverage is insufficient.
