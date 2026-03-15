# Phase 2 Handoff: Canonical Feature Tables and Data Dictionary

## 1. Canonical Feature Tables
This document outlines the Gold-tier unified feature tables available for Phase 2 Sentiment and Model Serving consumption.

### A. Sentinel OHLCV (Entity: `Symbol`)
- `dataset_snapshot_id`: Unique identifier tying this row to Bronze/Silver.
- `symbol`: The NSE symbol indicator (e.g., `RELIANCE`).
- `timestamp`: The UTC event time down to millisecond precision.
- `open`, `high`, `low`, `close`, `volume`: Real-time market metrics.
- `provenance_id`: Silver record origin.
- `quality_status`: `pass`/`warn` based on data anomalies.

### B. Macro Indicator (Entity: `Macro_Entity`)
- `indicator_name`: The specific macroeconomic indicator (e.g., `INCOME_TAX_RATE`).
- `value`: The numeric latest value of the indicator.
- `release_date`: The official release date vs ingestion date.
- `is_stale`: Boolean flag if the data point missed its expected SLA window.
- `weight_adjustment`: Multiplier for downstream reliance during stale periods.

### C. Textual Data Agent (Partner Delivery)
*(Detailed in Textual Handoff Package)*
- `NewsArticle_v1.0`, `SocialPost_v1.0`, `EarningsTranscript_v1.0`
- Companion sidecar metadata for sentiment: `confidence`, `manipulation_risk_score`, `ttl_seconds`.

## 2. Dataset Tiering & Retention Policy

| Tier | Format / Storage | Retention Period | Description |
|---|---|---|---|
| **Bronze** | Raw JSON / Deep Storage (S3) | 7 years (Compliance) | Raw event payloads as they arrived from APIs/scrapers. |
| **Silver** | Parquet / DB | 1 year (Hot), 3 years (Cold) | Normalized canonical records, deduplicated, schema-validated. |
| **Gold** | Fast KV / Time-series DB | 6 months (Hot) | Aggregated, feature-engineered views ready for models. |

## 3. Universe Selection Filters & Rebalance Cadence
- **Filter**: Current universe is locked to Top 50 NSE symbols by liquidity.
- **Cadence**: Universe dynamically re-evaluates on a quarterly basis (matching NSE expiry schedules).
- **Hooks**: A `config.universe` dict allows model builders to override this list during backtesting.
