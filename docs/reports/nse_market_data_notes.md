# NSE Market Data Notes

## Historical sources

- `nsemine` is the primary historical backfill source for NSE equities.
- `BROKER_API_BASE_URL` is the first fallback when a broker historical endpoint is configured.
- `yfinance` is the broader fallback for symbols or windows where NSE-side history is shallow or unavailable.

Historical backfill now merges sources in priority order and keeps the earliest source on overlapping timestamps. A symbol is not treated as a clean success unless the stored dataset passes the historical quality gate.

## Live sources

- `nsepython` is the primary live NSE quote source for `.NS` equities.
- `BROKER_API_BASE_URL` is an optional live quote fallback.
- `yfinance` is the last live fallback.

Live observations are stored separately in `live_market_observations`. Quote-level live data is not promoted into `ohlcv_bars` unless an observation is explicitly marked as a completed bar.

## Historical vs live handoff

- Historical backfill writes finalized bars to `ohlcv_bars`.
- Live polling writes quote or partial-bar observations to `live_market_observations` and `ticks`.
- Live data only writes into `ohlcv_bars` when the observation is an explicit `final_bar`.
- Duplicate protection remains on `(timestamp, symbol, interval)` for finalized bars.

## Symbol gating rules

Historical symbol readiness is written to `market_data_quality` with `dataset_type='historical'`.

- `status=train_ready`: eligible for training
- `status=partial`: stored in raw market tables, excluded from training
- `status=failed`: no usable dataset

Current gate fields include:

- first timestamp
- last timestamp
- row count
- duplicate count
- expected rows
- missing intervals
- gap count
- largest gap
- zero-volume ratio
- coverage percent
- history days

Default gate thresholds are loaded from `configs/nse_sentinel_runtime_v1.json`.

## Asset-aware rules

- Equity zero-volume ratio is checked as a quality signal.
- Forex zero volume is not treated as a training-data error.
- `vwap` remains optional and is dropped downstream when all-null.

## Known limitations

- `nsepython` live observations are quote-level snapshots, not completed hourly bars.
- `yfinance` remains useful for deep fallback coverage, but its timestamps and delays should be treated as secondary to NSE-native sources when both exist.
- Existing historical bar rows do not store `asset_type` directly; asset type is inferred consistently for quality and live-status metadata.
