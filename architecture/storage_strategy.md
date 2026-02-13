# Data Storage Strategy & Provenance

## 1. Storage Tiers

We utilize a "Medallion" architecture adapted for high-frequency financial data.

### Tier 1: Bronze (Raw)
- **Content**: Immutable, raw payloads exactly as received from the source.
- **Format**: JSON lines (`.jsonl`) compressed with Zstd, or original binary format if applicable.
- **Partitioning**: `source_id/YYYY-MM-DD/HH/`
- **retention**: Indefinite (Cheap storage, e.g., S3/Blob).
- **Purpose**: Replay, debugging, and audit trails. **Never** queried by models directly.

### Tier 2: Silver (Canonical)
- **Content**: Cleaned, validated, and normalized data.
- **Format**: Parquet (Snappy compression).
- **Schema**: Strictly typed (Enforced by `SchemaRegistry`).
- **Partitioning**: `symbol/YYYY-MM/` (optimized for time-series queries per asset).
- **Cadence**: 
    - **Primary**: Hourly bars (Open, High, Low, Close, Volume, VWAP).
    - **Secondary**: Daily bars (aggregated from Hourly for confirmation).
- **Purpose**: Source of truth for all backtesting and feature engineering.

### Tier 3: Gold (Feature-Ready)
- **Content**: Model-ready features (e.g., RSI, Volatility, Macro-adjusted returns).
- **Format**: Parquet or specialized tensor formats (e.g., `.npy` for training).
- **Partitioning**: `model_version/YYYY-MM/`
- **Purpose**: Direct input to Inference and Training agents.

## 2. Provenance Protocol

Every record in **Silver** and **Gold** tiers MUST carry the following metadata tags:

| Field | Type | Description |
| :--- | :--- | :--- |
| `event_id` | UUID | Unique identifier for the data point. |
| `source_type` | Enum | `official_api`, `broker_api`, `fallback_scraper`, `manual_override`. |
| `ingest_ts` | ISO8601 | Server timestamp when data was received (UTC). |
| `origin_ts` | ISO8601 | Timestamp reported by the source (Exchange time). |
| `schema_ver` | String | Version of the schema used for validation (e.g., `v1.2`). |
| `quality_flag` | Enum | `pass` (all checks ok), `warn` (soft limit breach), `fail` (quarantined). |

## 3. Data Cadence

- **Primary Trading Horizon**: Hourly.
- **Rational**: Balances signal noise with execution frequency for the Indian market context.
- **Constraint**: All downstream models must be able to function with Hourly resolution. Daily data is *only* for regime context.
