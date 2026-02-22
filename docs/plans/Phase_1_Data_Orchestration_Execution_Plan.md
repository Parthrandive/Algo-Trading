# Phase 1 Execution Plan: Data Orchestration Layer
Aligned to: `Multi_Agent_AI_Trading_System_Plan_Updated.md` v1.3.7 (Sections 3, 4, 5, 13, 15, 16)
Proposed Duration: 6 weeks + 1 week hypercare
Version: 2.0 (February 2026)

## 1. Phase Goal
Build a production-ready data orchestration layer for Indian markets (NSE equities/indices, USD/INR, F&O) that reliably feeds downstream modeling with validated, reproducible, and low-latency data.

Exit condition: all Phase 1 acceptance criteria in Sections 5.1 to 5.5 are met, plus the cross-cutting controls in Sections 3, 4, 13, and 15 needed for safe promotion to Phase 2.

### Phase 1 GO / NO-GO Gate (Section 16.1)
| Benchmark | Threshold |
|-----------|-----------|
| Data uptime during NSE hours | ≥ 99.5% for ≥ 10 consecutive trading days |
| Core symbol completeness | ≥ 99.0% |
| Provenance tagging coverage | 100% |
| Leakage tests | 100% pass |
| **NO-GO trigger** | Any critical data integrity failure unresolved, or any required benchmark not met |

## 2. Scope
In scope:
- `NSE Sentinel Agent` for market and reference data ingestion.
- `Macro Monitor Agent` for India macro and cross-asset indicators.
- `Preprocessing Agent` for deterministic feature-ready outputs.
- `Textual Data Agent` for news/report/transcript ingestion for later sentiment modeling.
- `Data Scale Roadmap & Replay` framework for reproducibility and growth.
- Data quality SLAs, observability, provenance tagging, fallback logic, and audit logs.

Out of scope:
- Alpha model training and signal generation (Phase 2+).
- Live order routing and execution decisions (Phase 3+).
- Online learning and live deployment promotion decisions (Phase 5+).

## 3. Architecture Decisions (Locked in Week 1)
- Primary trading horizon for stored signal-ready data: `Hourly`.
- Daily bars are generated only for confirmation/regime context.
- Data sampling frequency is locked to primary horizon to avoid model-data mismatch.
- Asset universe is explicit and versioned (NSE equities, indices, USD/INR, F&O).
- Universe selection policy numeric filters:
  - Avg Daily Turnover ≥ ₹10 Cr (6-month average)
  - Impact Cost ≤ 0.50%
  - Free float ≥ 20% (per NSE IWF methodology)
  - Price ≥ ₹50
  - Listing age ≥ 12 months
  - No stocks with > 5 circuit hits in the last 3 months
- Universe review and rebalance occurs quarterly via the Preprocessing Agent.
- Storage tiers:
  - Bronze: raw source payloads (immutable, source-tagged).
  - Silver: cleaned and normalized canonical series.
  - Gold: feature-ready tables for Phase 2 consumption.
- Dataset tiering: hot (in-memory/cache), warm (low-latency store), and cold (archive) with retention policies.
- Every event carries provenance tags:
  - source type (`official_api`, `broker_api`, `fallback_scraper`)
  - ingestion timestamp (UTC + IST)
  - schema version
  - quality status (`pass`, `warn`, `fail`)
- Data usage policy: only publicly released or contractually licensed feeds are permitted; unpublished or embargoed information is prohibited.
- Failure policy:
  - Primary feed failure immediately triggers `reduce-only` or `close-only` advisory flag for downstream risk systems until integrity checks recover.
  - Signal provenance is tagged per event (primary API vs fallback scraper) and exposed to downstream risk controls.

## 4. Week-by-Week Plan

| Week | Focus | Deliverables | Acceptance Gate |
| --- | --- | --- | --- |
| Week 1 | Foundation + governance | Source inventory, data contracts, schema registry v1, SLA definitions, control matrix, runbook skeleton, environment setup, universe selection policy, replay framework design | Design sign-off by Platform, Risk, Compliance |
| Week 2 | NSE Sentinel Agent | Primary connectors (official/broker endpoints), fallback connector, clock sync checks, timestamp validation, corporate actions ingest v1 | End-to-end ingest for core OHLCV + corp actions, with provenance tags |
| Week 3 | Macro Monitor + Preprocessing Agent | Macro indicator catalog v1.1 (WPI, CPI, IIP, FII/DII, FX reserves, RBI bulletins, India-US 10Y spread), ingestion jobs, freshness checks; Preprocessing I/O contracts, transform graph, lag alignment, leakage harness | Macro: 95%+ scheduled macro jobs complete in test window with alerting. Preprocessing: leakage tests green, reproducibility hash stable |
| Week 4 | Preprocessing Agent (hardening) + Textual Data Agent (start) | Deterministic transforms, normalization, directional change features, corporate action adjustment validator; Text source connectors skeleton | Reproducibility test passes on repeated runs; leakage tests green; text connector skeletons committed |
| Week 5 | Textual Data Agent | Source connectors (NSE news, Economic Times, RBI docs, transcripts), PDF parser, code-mixed text handling rules, dedup pipeline, spam/adversarial filtering | Text ingestion quality spot-check passes; metadata completeness >= 98%; pump-and-dump detection active |
| Week 6 | Hardening + integration | Latency optimization, SLA dashboard, failover drills, audit trail verification, backfill and replay tests, data scale roadmap finalization, Phase 2 handoff package | Phase 1 exit checklist complete and approved |
| Hypercare (Week 7) | Stabilization | Incident fixes, tuning thresholds, final documentation, sign-off memo | No Sev-1 incidents for 5 consecutive trading days |

## 5. Workstream Detail by Agent

### 5.1 NSE Sentinel Agent (Section 5.1)
Tasks:
- Implement source hierarchy: primary official API/exchange or broker endpoint; scraper fallback only when primary fails.
- Data sources include nsepython, jugad-data, and official NSE APIs or their replacements.
- Add redundancy for price, volume, and corporate actions feeds.
- Enforce clock sync and timestamp monotonicity checks across all streams.
- Publish provenance flags and feed-health status into canonical tables.
- Tag signal provenance per event (primary API vs fallback scraper) and expose to downstream risk controls.
- Trigger automated risk advisory flags (`reduce-only` or `close-only` mode) on feed integrity failures.

Validation:
- Missing data ratio per symbol per hour < 0.5% during trading hours.
- Timestamp drift alarms trigger at configured threshold.
- Failover switch tested in controlled drill with automatic recovery.
- Feed integrity failure correctly triggers reduce-only/close-only advisory.

### 5.2 Macro Monitor Agent (Section 5.2)
Tasks:
- Freeze and version indicator dictionary and release cadence.
- Indicators include: WPI, CPI, IIP, FII/DII flows, FX reserves, RBI bulletins, India-US 10Y spread.
- Global proxies (Brent, DXY) are explicitly justified as India-relevant or removed.
- Build source-specific parsers and normalization rules.
- Define allowed latency windows for each macro source.
- Add outlier and stale-value detection.
- Data quality checks cover missingness, outliers, and latency.

Validation:
- Indicator completeness report generated daily.
- Freshness SLA and anomaly detection dashboards operational.
- Global proxy inclusion is justified with documented rationale.

### 5.3 Preprocessing Agent (Section 5.3)
Tasks:
- Implement deterministic transform graph with strict input/output schema versions.
- Build rolling normalization and directional change threshold modules.
- Adjust historical series for corporate actions using reference verification.
- Add leakage test suite for lag correctness and alignment.
- Introduce feature approval workflow: proposal → offline eval → shadow → promote.
- Universe review and rebalance logic (quarterly cadence).

Validation:
- Same input snapshot reproduces identical output hash.
- Leakage test suite must pass before any feature promotion.
- Feature engineering is deterministic and reproducible.

### 5.4 Textual Data Agent (Section 5.4)
Tasks:
- Build ingestion for news, reports, and transcripts.
- Source list includes NSE news, Economic Times, RBI reports, and earnings transcripts.
- Social data collection from X with documented keyword and semantic search rules.
- Implement PDF extraction and text quality checks with spot checks.
- Add code-mixed English/Hinglish tokenization strategy and language tags.
- Define search/index metadata for downstream sentiment training.
- Implement spam and adversarial text filtering, source deduplication, and noise handling.
- Pump-and-dump and slang-scam detection for code-mixed text.

Validation:
- Spot-check precision target for extracted fields (headline, timestamp, source, ticker mapping).
- Duplicate and spam filtering performance reported daily.
- PDF extraction pipeline validated with spot checks.

### 5.5 Data Scale Roadmap and Replay (Section 5.5)
Tasks:
- Define current and target daily processing volumes (GB → TB) with quarterly checkpoints.
- Implement dataset tiering: hot (in-memory/cache), warm (low-latency store), cold (archive) with retention policies.
- Build replay framework that reconstructs any trading day from raw feed payloads through feature artifacts with deterministic identifiers.
- Support both event-time and wall-clock playback for strategy diagnostics and latency forensics.

Validation:
- Data processing roadmap documented and reviewed.
- Replay framework can reconstruct a full trading day from Bronze → Gold.
- Both event-time and wall-clock replay modes are functional.
- Deterministic identifiers are assigned to every dataset snapshot.

## 6. Cross-Cutting Controls

### Compliance and Audit (Sections 3, 13)
- Pre-trade relevant data controls logged with model/data version references.
- Retention and access policies documented and enforced.
- Data usage policy prohibits unpublished or embargoed information.
- Audit trail includes data version, schema version, and provenance tags.

### MLOps and Governance (Section 13)
- Dataset snapshot IDs, code hashes, and schema versions registered for each pipeline run.
- Promotion gates from dev → staging → Phase 2 handoff documented in CI checks.
- Model registry includes data snapshot and code hash.
- Training pipelines are reproducible from raw data.
- Fixed uplift claims are prohibited in approval artifacts unless backed by controlled benchmark or shadow A/B evidence.
- Human oversight gates defined, including review cadence and approval authority.

### Ops and Resilience (Section 15)
- Centralized metrics/traces/logging for ingest latency, pipeline lag, and failure counts.
- Incident runbook with escalation path and MTTR targets.
- Secrets management centralized and audited.
- DR plan defines RPO and RTO targets.
- System time sync enforced via NTP with drift alerts.
- Observability includes metrics, traces, and alerts for data latency, inference time, and execution failures.
- Operational dashboards track decision staleness, feature lag, mode-switch frequency, OOD trigger rate, kill-switch false positives, and MTTR.
- Interim operating model: single owner + reviewer accountability with explicit metric ownership.

## 7. KPIs and SLAs for Phase Exit

| KPI | Target |
| --- | --- |
| Data uptime during NSE trading hours | ≥ 99.5% for ≥ 10 consecutive trading days |
| Core symbol coverage completeness | ≥ 99.0% |
| Macro job schedule adherence | ≥ 95% |
| Canonical dataset reproducibility checks | 100% pass |
| Leakage test status for promoted features | 100% pass |
| Provenance tagging coverage | 100% of records |
| Critical audit fields completeness | 100% |
| Text metadata completeness | ≥ 98% |
| Replay framework operational | Full day reconstruction verified |

## 8. Risks and Mitigations
- Risk: primary feed instability.
  - Mitigation: automatic fallback, degrade-mode advisory flags, replay/backfill jobs.
- Risk: schema drift from upstream providers.
  - Mitigation: schema registry checks with fail-fast validation and compatibility tests.
- Risk: delayed macro releases.
  - Mitigation: freshness thresholds, stale markers, and downstream weighting controls.
- Risk: noisy textual extraction from PDFs.
  - Mitigation: document-type specific parsers, spot checks, and confidence scores.
- Risk: data poisoning or feed freeze.
  - Mitigation: feed-freeze simulations, correct behavior when no ticks arrive for defined windows, engine distinguishes zero volume from missing data.
- Risk: unpublished or embargoed data usage.
  - Mitigation: data usage policy enforcement, only publicly released or licensed feeds permitted.

## 9. Phase 1 Exit Checklist
- All Section 5.1 to 5.5 acceptance criteria are evidence-backed and signed off.
- SLA dashboard live with ≥ 10 consecutive trading-day evidence.
- Incident and failover drills executed with documented outcomes.
- Data contracts and schema versions frozen for Phase 2 integration.
- Replay framework verified (full day reconstruction from raw feeds → features).
- Data scale roadmap reviewed and quarterly checkpoints set.
- Go/No-Go benchmark gate (Section 16.1) fully satisfied.
- Handoff pack delivered: architecture, runbooks, dataset catalog, known limitations.

## 10. Phase 2 Handoff Artifacts
- Canonical feature tables and dictionary.
- Provenance and quality scoring fields for model consumption.
- Replayable sample datasets for model benchmarking.
- Universe selection filters and rebalance cadence documentation.
- Dataset tiering and retention policy documentation.
- Signed release note with open issues and mitigation owners.

## 11. Go-Live Checklist Reference (Section 16.1)
The following go-live minimums apply across all phases. Phase 1 is responsible for data uptime and compliance readiness:

| Metric | Minimum |
| --- | --- |
| Paper trading | ≥ 3 calendar months with ≥ 80% uptime |
| Annualized Sharpe | ≥ 1.8 |
| Sortino | ≥ 2.0 |
| Max Drawdown | ≤ 8% |
| Win rate | ≥ 52% |
| Profit factor | ≥ 1.5 |
| Avg realized slippage | ≤ model estimate + 20 bps |
| Data uptime (NSE hours) | ≥ 99.5% |
| Critical compliance violations | Zero |
| Broker/SEBI flags or rejections | Zero |
