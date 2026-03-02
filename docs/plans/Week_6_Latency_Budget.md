# Week 6 Latency Budget — Sync S1 (March 3, 2026)

> To be populated with actual measurements from `src/utils/latency.py` profiler.

## Your Pipelines (Numeric Data)

### NSE Sentinel Agent

| Stage | P50 (s) | P95 (s) | P99 (s) | Budget (s) | Status |
|-------|--------:|--------:|--------:|-----------:|--------|
| `ingest_quote` (tick → Silver) | — | — | — | TBD | ⏳ |
| `ingest_historical` (bars → Silver) | — | — | — | TBD | ⏳ |
| `ingest_corporate_actions` | — | — | — | TBD | ⏳ |

### Macro Monitor Agent

| Stage | P50 (s) | P95 (s) | P99 (s) | Budget (s) | Status |
|-------|--------:|--------:|--------:|-----------:|--------|
| `run_ingest` (fetch → Silver) | — | — | — | TBD | ⏳ |

### Preprocessing Agent

| Stage | P50 (s) | P95 (s) | P99 (s) | Budget (s) | Status |
|-------|--------:|--------:|--------:|-----------:|--------|
| `process_snapshot` (Silver → Gold) | — | — | — | TBD | ⏳ |
| `replay_snapshot` | — | — | — | TBD | ⏳ |

## Partner Pipeline (Textual Data) — To Be Filled at S1

### Textual Data Agent

| Stage | P50 (s) | P95 (s) | P99 (s) | Budget (s) | Status |
|-------|--------:|--------:|--------:|-----------:|--------|
| Adapter fetch (all sources) | — | — | — | TBD | ⏳ |
| Clean + validate | — | — | — | TBD | ⏳ |
| Export canonical + sidecar | — | — | — | TBD | ⏳ |

## Agreed Budget Rules
- [ ] P95 must be below agreed budget for all pipelines
- [ ] P99 spikes above 2× budget trigger an optimization ticket
- [ ] Latency profiles re-measured after each optimization pass

## Notes
_Space for discussion notes from S1 sync._
