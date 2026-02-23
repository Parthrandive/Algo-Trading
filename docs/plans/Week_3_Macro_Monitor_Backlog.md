# Week 3 — Partner Plan: Macro Monitor Agent
**Dates:** Mon, February 23 – Sun, March 1, 2026
**Owner:** Partner
**Aligned to:** Phase 1 Data Orchestration Execution Plan v2.0 — Section 5.2, 5.5
**Master Plan Reference:** Multi_Agent_AI_Trading_System_Plan_Updated v1.3.7 — Section 5.2

---

## Goal
Build the Macro Monitor Agent to ingest India macro and cross-asset indicators with scheduled jobs, freshness SLAs, and alerting. Publish canonical sample payloads for downstream Preprocessing Agent consumption.

## Phase 1 GO/NO-GO Contribution (Section 16.1)
This workstream contributes to:
- **Macro job schedule adherence ≥ 95%**
- **Provenance tagging coverage = 100%**
- **Data uptime ≥ 99.5%** (for macro feeds during scheduled windows)

---

## Frozen Indicator Catalog (Schema v1.1)

| Indicator | Source | Period | Freshness Window | Unit |
|-----------|--------|--------|------------------|------|
| CPI | MOSPI / RBI | Monthly | 48h after release | % |
| WPI | Office of Economic Adviser | Monthly | 48h after release | % |
| IIP | MOSPI | Monthly | 48h after release | % |
| FII Flow | NSDL / NSE | Daily | 4h (by EOD+4h) | INR_Cr |
| DII Flow | NSDL / NSE | Daily | 4h (by EOD+4h) | INR_Cr |
| FX Reserves | RBI Weekly Statistical Supplement | Weekly | 24h after Friday release | USD_Bn |
| RBI Bulletins | RBI website | Irregular | 24h after publication | count |
| India-US 10Y Spread | RBI + FRED/Treasury.gov | Daily | 6h | bps |

### RBI Bulletin Encoding Rule (Schema Compatibility)
- `MacroIndicator.value` is numeric in `src/schemas/macro_data.py`.
- For `RBI_BULLETIN`, encode publication as an event marker: `value=1.0`, `unit="count"`, `period="Irregular"`.
- Raw bulletin text extraction remains in the Textual Data Agent scope.

### Global Proxy Justification (Section 5.2 Requirement)
> Global proxies such as Brent and DXY must be **explicitly justified as India-relevant or removed**.

| Proxy | Included? | Justification |
|-------|-----------|---------------|
| Brent Crude (`BRENT_CRUDE`) | Yes — already in enum | Direct impact on India's import bill, INR pressure, and inflation expectations |
| DXY | Not in current catalog | Defer — India-US 10Y spread + INR/USD capture the relevant FX signal |

---

## Shared Schema Contract

All macro data **must** conform to the `MacroIndicator` Pydantic model in `src/schemas/macro_data.py`:
```python
# Required fields for every record:
indicator_name: MacroIndicatorType  # enum value from catalog above
value: float
unit: str          # e.g., "%", "INR_Cr", "bps", "USD_Bn", "count"
period: str        # "Daily", "Weekly", "Monthly"
timestamp: datetime  # observation date (timezone-aware)
source_type: SourceType  # "official_api" | "broker_api" | "fallback_scraper"
schema_version: str  # Must be "1.1"
quality_status: QualityFlag  # "pass" | "warn" | "fail"
# Auto-populated:
ingestion_timestamp_utc, ingestion_timestamp_ist, region
```

> **⚠️ CRITICAL:** The `MacroIndicatorType` enum must include these exact values:
> `CPI, WPI, IIP, GDP, INR_USD, BRENT_CRUDE, US_10Y, INDIA_10Y, FII_FLOW, DII_FLOW, REPO_RATE, FX_RESERVES, INDIA_US_10Y_SPREAD, RBI_BULLETIN`

### Indicator Scope (Week 3 vs Accepted Enum)
- **Accepted enum set:** 14 values above must be accepted by schema/contracts.
- **Week 3 required publish set (CP3):** `CPI, WPI, IIP, FII_FLOW, DII_FLOW, FX_RESERVES, RBI_BULLETIN, INDIA_US_10Y_SPREAD`.

### Data Usage Policy (Section 3 Requirement)
Only publicly released or contractually licensed feeds are permitted. Unpublished or embargoed information is **prohibited**.

---

## Daily Backlog

### Day 1 (Mon, Feb 23) — Contract Freeze 🔒 CP1
- [x] Confirm indicator catalog matches table above
- [x] Add `FX_RESERVES`, `INDIA_US_10Y_SPREAD`, `RBI_BULLETIN` to `MacroIndicatorType` enum
- [x] Bump `schema_version` default to `"1.1"`
- [x] Document global proxy justification (Brent included, DXY excluded with rationale)
- [x] Create `configs/macro_monitor_runtime_v1.json` with per-indicator source config (URL patterns, retry, rate limits, freshness windows)
- [x] **Deliverable:** Frozen schema + runtime config + proxy justification committed

### Day 2 (Tue, Feb 24) — Source Connector Skeletons
- [ ] Create `src/agents/macro/` package
- [ ] Define `MacroClientInterface` (Protocol) with `get_indicator(name, date_range)` method
- [ ] Scaffold concrete client stubs: `MOSPIClient`, `RBIClient`, `NSEDIIFIIClient`, `FXReservesClient`, `BondSpreadClient`
- [ ] Write `MacroSilverRecorder` (Parquet + DB persistence, following sentinel recorder patterns)
- [ ] Ensure provenance tagging on every record (source_type, ingestion timestamps, schema_version, quality_status)
- [ ] **Deliverable:** All skeletons committed, type-check passes

### Day 3 (Wed, Feb 25) — Parsers: CPI / WPI / IIP / RBI Bulletins
- [ ] Implement `CPIParser` — normalize MOSPI payload → `MacroIndicator`
- [ ] Implement `WPIParser` — normalize OEA payload → `MacroIndicator`
- [ ] Implement `IIPParser` — normalize MOSPI payload → `MacroIndicator`
- [ ] Implement `RBIBulletinParser` — extract structured fields from bulletin HTML/PDF
- [ ] Add data quality checks: missingness, outliers, and latency (Section 5.2 requirement)
- [ ] Unit tests for all 4 parsers (`tests/agents/macro/test_parsers.py`)
- [ ] **Deliverable:** All 4 parsers produce valid `MacroIndicator` records with quality checks

### Day 4 (Thu, Feb 26) — Parsers: FII/DII, FX Reserves, 10Y Spread
- [ ] Implement `FIIDIIParser` — normalize NSDL/NSE daily flow data
- [ ] Implement `FXReservesParser` — normalize RBI weekly supplement
- [ ] Implement `BondSpreadParser` — compute India-US 10Y spread from RBI + FRED
- [ ] `MacroIngestPipeline` — Bronze → Silver with provenance tags
- [ ] Feed integrity failure → trigger `reduce-only` advisory flag (Section 5.1 pattern)
- [ ] Unit tests for all 3 parsers + pipeline integration test
- [ ] **Deliverable:** All parsers + pipeline working end-to-end with provenance

### Day 5 (Fri, Feb 27) — Scheduler + Reliability 🔒 CP2
- [ ] Implement `MacroScheduler` — cron jobs per indicator mapped to release cadence from runtime config
- [ ] Add retry with exponential backoff
- [ ] Add idempotent ingest guard (dedup on `indicator_name + timestamp`)
- [ ] Add provenance logging to `IngestionLog` table (dataset snapshot IDs, code hashes — Section 13)
- [ ] Scheduler unit tests (`tests/agents/macro/test_scheduler.py`)
- [ ] **Deliverable:** CP2 evidence — all scheduled jobs configured, retry + dedup + provenance verified

### Day 6 (Sat, Feb 28) — Freshness + Sample Payloads 🔒 CP3 (Sync Gate A)
- [ ] Implement freshness window checks (per-indicator, from runtime config)
- [ ] Add stale marker logic — flag indicators past their freshness window
- [ ] Add outlier and stale-value detection (Section 5.2 requirement)
- [ ] Add alert rule definitions (log-based + configurable webhook stub)
- [ ] Generate daily indicator completeness report
- [ ] **Publish canonical sample payloads** to `data/macro_samples/`:
  - One JSON file per indicator, each containing 5–10 sample `MacroIndicator` records
  - Must pass list validation: `TypeAdapter(list[MacroIndicator]).validate_python(payload)`
  - Must cover all 8 indicators in the Week 3 required publish set
  - Must have realistic timestamps and values
- [ ] **Deliverable:** CP3 evidence — sample payloads committed, freshness + anomaly checks active

> **🔗 SYNC GATE A:** The Preprocessing Agent will consume files from `data/macro_samples/` on this day. Sample payloads must be valid `MacroIndicator` JSON (schema v1.1).

### Day 7 (Sun, Mar 1) — Test Window + Report 🔒 CP4
- [ ] Run scheduler in test mode for a simulated window
- [ ] Generate completeness report: target **≥ 95%** scheduled jobs completed
- [ ] Generate freshness report: all indicators within window or correctly flagged stale
- [ ] Verify anomaly detection dashboards are operational with active alerting (Section 5.2 acceptance)
- [ ] Publish `CP4_Week_3_Signoff.md` with evidence tables
- [ ] **Deliverable:** Week 3 acceptance gate passed

---

## Checkpoint Summary

| CP | Date | Gate | Pass Criteria |
|----|------|------|---------------|
| CP1 | Feb 23 | Contract Freeze | Schema v1.1 frozen, runtime config committed, enum has all 14 indicator types, proxy justification documented |
| CP2 | Feb 27 | Ingestion Readiness | Scheduler configured, retry/dedup verified, provenance logging active with dataset snapshot IDs |
| CP3 | Feb 28 | **Sync Gate A** | Sample payloads in `data/macro_samples/` pass `TypeAdapter(list[MacroIndicator]).validate_python(payload)` for all 8 Week 3 required indicators; freshness SLA + anomaly detection operational |
| CP4 | Mar 1 | Week 3 Acceptance | ≥ 95% job completion in test window, completeness + freshness reports published |

---

## Acceptance Criteria Cross-Reference (Section 5.2)

| Acceptance Criterion | Where Addressed |
|---------------------|-----------------|
| Macro indicators list is defined and versioned | CP1 — Frozen Indicator Catalog |
| Indicators include WPI, CPI, IIP, FII/DII flows, FX reserves, RBI bulletins, India-US 10Y spread | CP1 — Catalog table |
| Global proxies (Brent, DXY) explicitly justified as India-relevant or removed | CP1 — Global Proxy Justification table |
| Data quality checks cover missingness, outliers, and latency | Day 3-4 parsers + Day 6 freshness/outlier detection |
| ≥ 95% scheduled macro jobs complete in test window | CP4 — Test window report |
| Freshness SLAs and anomaly detection dashboards operational | CP3/CP4 — Freshness checks + alerting |

## Integration Points with Preprocessing Agent

| When | What you provide | What they consume |
|------|-----------------|-------------------|
| CP1 (Feb 23) | Frozen `MacroIndicatorType` enum + schema v1.1 | Preprocessing I/O contract references your schema |
| CP3 (Feb 28) | `data/macro_samples/*.json` | Preprocessing loader validates + transforms these |
| CP4 (Mar 1) | Completeness report | Integration-readiness note references your report |
