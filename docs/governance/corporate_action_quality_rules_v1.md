# Corporate Action Quality Rules v1
**Date:** February 22, 2026  
**Scope:** Week 2 Day 12 corporate-actions ingest controls

## Canonical Schema
- Schema ID: `market.corporate_action.v1`
- Canonical model: `src/schemas/market_data.py::CorporateAction`
- Mandatory provenance fields:
  - `source_type`
  - `ingestion_timestamp_utc`
  - `ingestion_timestamp_ist`
  - `schema_version`
  - `quality_status`

## Validation Rules
- `ex_date` must be timezone-aware and normalized to UTC.
- `record_date` is optional; when present it must be timezone-aware and `record_date >= ex_date`.
- `ratio` is required for `split`, `bonus`, and `rights` actions and must follow `<num>:<num>` format.
- `value` is required for `dividend` actions and must be `> 0`.
- `action_type` enum is restricted to: `dividend`, `split`, `bonus`, `rights`.

## Ingest Behavior
- Bronze capture:
  - Pipeline method: `SentinelIngestPipeline.ingest_corporate_actions`
  - Bronze schema tag: `market.corporate_action.v1`
- Silver persistence:
  - Parquet recorder path: `data/silver/corporate_actions/<symbol>/<year>/<month>/<date>.parquet`
  - DB recorder table: `corporate_actions`

## Verification Commands
- Corporate-action ingest and type coverage:
  - `python3 scripts/verify_corp_actions.py RELIANCE.NS TATASTEEL.NS`
- End-to-end completeness (Bronze + Silver):
  - `python3 scripts/verify_completeness.py`
