# Architecture Decisions v1
Date: 2026-02-10
Status: Locked (Week 1 Day 2)

## 1. Time Horizon
- Primary trading data horizon: `Hourly`.
- Daily bars are derived for confirmation/regime context only.

## 2. Storage Contract
- `Bronze`: immutable raw payloads.
- `Silver`: cleaned canonical records validated by schema registry.
- `Gold`: feature-ready datasets for downstream modeling.

## 3. Provenance Requirements
- Required metadata on canonical records:
- `source_type`
- `ingestion_timestamp_utc`
- `ingestion_timestamp_ist`
- `schema_version`
- `quality_status`

## 4. Degradation Policy
- Operating states: `normal`, `reduce-only`, `close-only advisory`.
- Feed integrity failures force degrade mode until health recovery and fresh validated data.

## 5. Interface Guardrails
- Cross-module payloads are schema-validated.
- Execution-critical loop consumes only published snapshots and never blocks on ingestion.
