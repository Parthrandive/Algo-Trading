# Week 6 Latency Budget — Sync S1 (March 3, 2026)

> Populated with actual measurements from `src/utils/latency.py` profiler.

## Your Pipelines (Numeric Data)

### NSE Sentinel Agent

| Stage | P50 (s) | P95 (s) | P99 (s) | Budget (s) | Status |
|-------|--------:|--------:|--------:|-----------:|--------|
| `ingest_quote` (tick → Silver) | 0.0067 | 0.0067 | 0.0067 | 1.0 | ✅ |
| `ingest_historical` (bars → Silver) | 0.0059 | 0.0059 | 0.0059 | 2.0 | ✅ |
| `ingest_corporate_actions` | 0.0085 | 0.0085 | 0.0085 | 2.0 | ✅ |

### Macro Monitor Agent

| Stage | P50 (s) | P95 (s) | P99 (s) | Budget (s) | Status |
|-------|--------:|--------:|--------:|-----------:|--------|
| `run_ingest` (fetch → Silver) | 0.0054 | 3.2675 | 3.2675 | 5.0 | ✅ |

### Preprocessing Agent

| Stage | P50 (s) | P95 (s) | P99 (s) | Budget (s) | Status |
|-------|--------:|--------:|--------:|-----------:|--------|
| `process_snapshot` (Silver → Gold) | 0.0213 | 1.0184 | 1.0184 | 2.0 | ✅ |
| `replay_snapshot` | 0.0113 | 0.0113 | 0.0113 | 2.0 | ✅ |

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
