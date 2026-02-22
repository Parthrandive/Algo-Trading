# Data Contracts v1
Date: 2026-02-11
Status: Frozen for Week 1 Day 3 handoff

## 1. Contracted Entities
- `Tick` (`Tick_v1.0`)
- `Bar` (`Bar_v1.0`)
- `CorporateAction` (`CorporateAction_v1.0`)
- `MacroIndicator` (`MacroIndicator_v1.0`)
- `NewsArticle` (`NewsArticle_v1.0`)
- `SocialPost` (`SocialPost_v1.0`)
- `EarningsTranscript` (`EarningsTranscript_v1.0`)

## 2. Mandatory Provenance Fields
All canonical records carry:
- `source_type`: provenance enum (`official_api`, `broker_api`, `fallback_scraper`, `manual_override`, and text-specific sources where applicable).
- `ingestion_timestamp_utc`: timezone-aware ingestion timestamp in UTC.
- `ingestion_timestamp_ist`: timezone-aware ingestion timestamp in Asia/Kolkata.
- `schema_version`: schema semantic version string.
- `quality_status`: quality enum (`pass`, `warn`, `fail`).

## 3. Backward-Compatible Input Aliases
To keep old payloads valid during migration:
- `source` -> `source_type`
- `ingestion_timestamp` -> `ingestion_timestamp_utc`
- `quality_flag` -> `quality_status`

Canonical serialization names remain the v1 contract names listed in Section 2.

## 4. Validation Baseline
- Strict contract mode: unknown fields are rejected (`extra=forbid`).
- `Bar` enforces cross-field OHLC validity.
- `CorporateAction` enforces timezone-aware `ex_date`/`record_date`, valid ratio format, action-specific payload requirements (`dividend -> value`, `split/bonus/rights -> ratio`), and `record_date >= ex_date`.
- Engagement counters in `SocialPost` are non-negative.
- Registry validation blocks unknown version keys.

## 5. Registry Freeze
Schema registry v1 is implemented in `/Users/juhi/Desktop/algo-trading/src/utils/schema_registry.py` and pre-registers all entities above.
