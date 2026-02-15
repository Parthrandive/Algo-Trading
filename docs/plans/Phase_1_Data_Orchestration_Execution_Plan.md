# Phase 1 Execution Plan: Data Orchestration Layer
Aligned to: `Multi_Agent_AI_Trading_System_Plan_Updated.md` (Sections 3, 4, 5, 13, 15, 16)
Proposed Duration: 6 weeks + 1 week hypercare
Version: 1.0 (February 2026)

## 1. Phase Goal
Build a production-ready data orchestration layer for Indian markets (NSE equities/indices, USD/INR, F&O) that reliably feeds downstream modeling with validated, reproducible, and low-latency data.

Exit condition: all Phase 1 acceptance criteria in Sections 5.1 to 5.4 are met, plus the cross-cutting controls in Sections 3, 4, 13, and 15 needed for safe promotion to Phase 2.

## 2. Scope
In scope:
- `NSE Sentinel Agent` for market and reference data ingestion.
- `Macro Monitor Agent` for India macro and cross-asset indicators.
- `Preprocessing Agent` for deterministic feature-ready outputs.
- `Textual Data Agent` for news/report/transcript ingestion for later sentiment modeling.
- Data quality SLAs, observability, provenance tagging, fallback logic, and audit logs.

Out of scope:
- Alpha model training and signal generation (Phase 2+).
- Live order routing and execution decisions (Phase 3+).
- Online learning and live deployment promotion decisions (Phase 5+).

## 3. Architecture Decisions (Locked in Week 1)
- Primary trading horizon for stored signal-ready data: `Hourly`.
- Daily bars are generated only for confirmation/regime context features.
- Storage tiers:
- Bronze: raw source payloads (immutable, source-tagged).
- Silver: cleaned and normalized canonical series.
- Gold: feature-ready tables for Phase 2 consumption.
- Every event carries provenance tags:
- source type (`official_api`, `broker_api`, `fallback_scraper`)
- ingestion timestamp (UTC + IST)
- schema version
- quality status (`pass`, `warn`, `fail`)
- Failure policy:
- Primary feed failure immediately triggers `reduce-only` or `close-only` advisory flag for downstream risk systems until integrity checks recover.

## 4. Week-by-Week Plan

| Week | Focus | Deliverables | Acceptance Gate |
| --- | --- | --- | --- |
| Week 1 | Foundation + governance | Source inventory, data contracts, schema registry v1, SLA definitions, control matrix, runbook skeleton, environment setup | Design sign-off by Platform, Risk, Compliance |
| Week 2 | NSE Sentinel Agent | Primary connectors (official/broker endpoints), fallback connector, clock sync checks, timestamp validation, corporate actions ingest v1 | End-to-end ingest for core OHLCV + corp actions, with provenance tags |
| Week 3 | Macro Monitor Agent | Macro indicator catalog (WPI, CPI, IIP, FII/DII, FX reserves, RBI bulletins, India-US 10Y spread), ingestion jobs, freshness checks | 95%+ scheduled macro jobs complete in test window with alerting |
| Week 4 | Preprocessing Agent | Deterministic transforms, normalization, directional change features, corporate action adjustment validator, leakage test harness | Reproducibility test passes on repeated runs; leakage tests green |
| Week 5 | Textual Data Agent | Source connectors (NSE news, Economic Times, RBI docs, transcripts), PDF parser, code-mixed text handling rules, dedup pipeline | Text ingestion quality spot-check passes; metadata completeness >= 98% |
| Week 6 | Hardening + integration | Latency optimization, SLA dashboard, failover drills, audit trail verification, backfill and replay tests, Phase 2 handoff package | Phase 1 exit checklist complete and approved |
| Hypercare (Week 7) | Stabilization | Incident fixes, tuning thresholds, final documentation, sign-off memo | No Sev-1 incidents for 5 consecutive trading days |

## 5. Workstream Detail by Agent

### 5.1 NSE Sentinel Agent (Section 5.1)
Tasks:
- Implement source hierarchy: primary official API/exchange or broker endpoint; scraper fallback only when primary fails.
- Add redundancy for price, volume, and corporate actions feeds.
- Enforce clock sync and timestamp monotonicity checks.
- Publish provenance flags and feed-health status into canonical tables.
- Trigger automated risk advisory flags on feed integrity failures.

Validation:
- Missing data ratio per symbol per hour < 0.5% during trading hours.
- Timestamp drift alarms trigger at configured threshold.
- Failover switch tested in controlled drill with automatic recovery.

### 5.2 Macro Monitor Agent (Section 5.2)
Tasks:
- Freeze and version indicator dictionary and release cadence.
- Build source-specific parsers and normalization rules.
- Define allowed latency windows for each macro source.
- Add outlier and stale-value detection.

Validation:
- Indicator completeness report generated daily.
- Freshness SLA and anomaly detection dashboards operational.

### 5.3 Preprocessing Agent (Section 5.3)
Tasks:
- Implement deterministic transform graph with strict input/output schema versions.
- Build rolling normalization and directional change threshold modules.
- Adjust historical series for corporate actions using reference verification.
- Add leakage test suite for lag correctness and alignment.
- Introduce feature approval workflow: proposal -> offline eval -> shadow -> promote.

Validation:
- Same input snapshot reproduces identical output hash.
- Leakage test suite must pass before any feature promotion.

### 5.4 Textual Data Agent (Section 5.4)
Tasks:
- Build ingestion for news, reports, and transcripts.
- Implement PDF extraction and text quality checks.
- Add code-mixed English/Hinglish tokenization strategy and language tags.
- Define search/index metadata for downstream sentiment training.

Validation:
- Spot-check precision target for extracted fields (headline, timestamp, source, ticker mapping).
- Duplicate and spam filtering performance reported daily.

## 6. Cross-Cutting Controls
Compliance and audit:
- Pre-trade relevant data controls logged with model/data version references.
- Retention and access policies documented and enforced.

MLOps and governance:
- Dataset snapshot IDs, code hashes, and schema versions registered for each pipeline run.
- Promotion gates from dev -> staging -> Phase 2 handoff documented in CI checks.

Ops and resilience:
- Centralized metrics/traces/logging for ingest latency, pipeline lag, and failure counts.
- Incident runbook with escalation path and MTTR targets.

## 7. KPIs and SLAs for Phase Exit
- Data uptime during NSE trading hours: `>= 99.5%`.
- Core symbol coverage completeness: `>= 99.0%`.
- Macro job schedule adherence: `>= 95%`.
- Canonical dataset reproducibility checks: `100% pass`.
- Leakage test status for promoted features: `100% pass`.
- Provenance tagging coverage: `100% of records`.
- Critical audit fields completeness: `100%`.

## 8. Risks and Mitigations
- Risk: primary feed instability.
- Mitigation: automatic fallback, degrade-mode advisory flags, replay/backfill jobs.
- Risk: schema drift from upstream providers.
- Mitigation: schema registry checks with fail-fast validation and compatibility tests.
- Risk: delayed macro releases.
- Mitigation: freshness thresholds, stale markers, and downstream weighting controls.
- Risk: noisy textual extraction from PDFs.
- Mitigation: document-type specific parsers, spot checks, and confidence scores.

## 9. Phase 1 Exit Checklist
- All Section 5.1 to 5.4 acceptance criteria are evidence-backed and signed off.
- SLA dashboard live with 10+ consecutive trading-day evidence.
- Incident and failover drills executed with documented outcomes.
- Data contracts and schema versions frozen for Phase 2 integration.
- Handoff pack delivered: architecture, runbooks, dataset catalog, known limitations.

## 10. Phase 2 Handoff Artifacts
- Canonical feature tables and dictionary.
- Provenance and quality scoring fields for model consumption.
- Replayable sample datasets for model benchmarking.
- Signed release note with open issues and mitigation owners.
