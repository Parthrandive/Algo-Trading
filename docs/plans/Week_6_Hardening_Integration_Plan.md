# Week 6 Plan: Hardening + Integration (Phase 1 Final Build Week)

**Week window**: Monday, March 2, 2026 to Sunday, March 8, 2026
**Alignment**: Phase 1 Execution Plan §4 (Week 6) — "Hardening + integration"
**Your focus**: NSE Sentinel Agent, Macro Monitor Agent, Preprocessing Agent, Data Scale & Replay, cross-agent integration, Phase 2 handoff
**Partner focus**: Textual Data Agent hardening, PDF/Hinglish quality bar, text dedup & spam tuning, textual replay support

---

## Week 6 Goal

Harden every agent, run end-to-end integration across all four data streams, execute failover drills, lock the SLA dashboard, validate the replay framework on a full trading day, and produce the Phase 2 handoff package. Week 7 (Hypercare) should begin with zero known Sev-1 issues.

---

## Sync Points with Partner (Textual Data Agent)

| Sync | Day | Purpose | Your deliverable | Partner deliverable |
|------|-----|---------|-------------------|---------------------|
| **S1** | Day 2 (Tue) | Latency & contract alignment | Unified latency budget doc covering Sentinel + Macro + Preprocessing pipelines | Textual pipeline latency profile + any contract field updates since Week 5 |
| **S2** | Day 4 (Thu) | Cross-agent replay dry run | Replay harness configured for Sentinel → Preprocessing → Gold tier | Textual replay slice (Bronze text → Silver canonical → sidecar metadata) producing matching snapshot IDs |
| **S3** | Day 5 (Fri) | Failover drill (joint) | Sentinel feed-kill + Macro stale-source drill scripts | Text source outage + PDF fallback drill scripts |
| **S4** | Day 7 (Sun) | Phase 2 handoff review | Handoff package for numeric data streams | Textual handoff package; joint sign-off on combined exit checklist |

---

## Day-by-Day Execution

### Day 1 — Monday, March 2: Latency Profiling & SLA Dashboard Shell

**Theme**: Measure where we stand, build the dashboard frame.

- [ ] **Latency profiling**: Instrument and measure end-to-end latency for each pipeline:
  - NSE Sentinel: tick ingestion → Bronze → Silver canonical.
  - Macro Monitor: scheduled fetch → Bronze → indicator table.
  - Preprocessing: Silver input → Gold feature output.
  - Record P50/P95/P99 numbers for each leg.
- [ ] **SLA dashboard v1**: Build a dashboard (can be a simple Python/Streamlit or HTML page) that renders:
  - Data uptime percentage during NSE hours (target ≥ 99.5%).
  - Core symbol completeness (target ≥ 99.0%).
  - Macro job schedule adherence (target ≥ 95%).
  - Provenance tagging coverage (target 100%).
  - Leakage test status (target 100% pass).
- [ ] **Audit trail spot-check**: Verify every record in Silver/Gold carries `source_type`, `ingestion_timestamp_utc/ist`, `schema_version`, `quality_status`. Log gaps.
- [ ] Create a shared latency budget document skeleton for Sync S1 (tomorrow).

**Output**: Latency profile report, SLA dashboard shell, audit-gap log, S1 prep doc.

---

### Day 2 — Tuesday, March 3: Latency Optimization + Contract Alignment (**Sync S1**)

**Theme**: Squeeze latency, align contracts with partner.

- [ ] **Optimize hot paths** identified on Day 1:
  - Sentinel: batch DB writes, connection pooling for broker/NSE endpoints.
  - Preprocessing: cache intermediate transform results; pre-compute rolling windows.
  - Macro: parallelize independent indicator fetches.
- [ ] **Provenance gap remediation**: Fix any audit-trail gaps found on Day 1 (missing tags, incorrect timestamps).
- [ ] **Data contract freeze check**: Verify all schema versions (`OHLCV_v1.0`, `MacroIndicator_v1.0`, `ExogenousIndicator_v1.0`) are stable. Confirm no breaking fields changed since Week 5.

> **🔄 Sync S1 — with Partner**
> - Exchange latency profiles (your numeric pipelines vs. partner's text pipeline).
> - Align on unified latency budget: what's acceptable end-to-end for each tier.
> - Confirm text schema fields (`NewsArticle_v1.0`, `SocialPost_v1.0`, `EarningsTranscript_v1.0`) are unchanged or document deltas.
> - Agree on common `dataset_snapshot_id` format for cross-stream replay.

**Output**: Optimized pipeline with measurable latency improvement, latency budget doc (joint), contract delta log.

---

### Day 3 — Wednesday, March 4: Backfill & Replay Framework Validation

**Theme**: Prove the replay framework works end-to-end on numeric streams.

- [ ] **Full-day replay test** (pick a recent trading day):
  - Replay raw Bronze feeds through Sentinel → Preprocessing → Gold tier.
  - Verify deterministic output: same Bronze input → identical Gold hash.
  - Verify both **event-time** and **wall-clock** playback modes work.
- [ ] **Macro replay**: Replay a day's macro indicator fetches from Bronze payloads; verify freshness/staleness markers are correctly reconstructed.
- [ ] **Backfill gap test**: Simulate a scenario where 2 hours of data were missed → run backfill → verify gap is closed and Gold tier is complete.
- [ ] **Snapshot ID assignment**: Confirm every replay run produces a unique, deterministic `dataset_snapshot_id` that ties Bronze → Silver → Gold.
- [ ] Prepare replay harness for cross-agent run (Day 4 sync with partner).

**Output**: Replay test report (pass/fail per mode), backfill scenario evidence, snapshot ID verification log.

---

### Day 4 — Thursday, March 5: Cross-Agent Replay & Integration Test (**Sync S2**)

**Theme**: Prove all four data streams can replay together and produce consistent snapshot IDs.

- [x] **Cross-agent integration test**: Run a joint replay harness that covers:
  - Sentinel OHLCV + Corp Actions → Preprocessing → Gold.
  - Macro indicators → indicator tables.
  - Textual canonical records → Silver text + sidecar metadata *(partner's pipeline)*.
- [x] **Snapshot ID cross-reference**: Verify that all four streams' `dataset_snapshot_id` values for the same trading day are traceable and linkable.
- [x] **Schema compatibility matrix**: Validate that Gold-tier outputs from all streams can be loaded into a unified data catalog without field conflicts.
- [x] **Data scale roadmap draft**: Document current daily processing volumes (GB), target volumes for Phase 2, and quarterly checkpoints (per §5.5).

> **🔄 Sync S2 — with Partner**
> - Partner runs textual replay slice for the same trading day.
> - Compare snapshot IDs for cross-stream consistency.
> - Validate text sidecar metadata (`confidence`, `ttl_seconds`, `manipulation_risk_score`) is present and typed correctly.
> - Log any integration issues in a shared defect tracker.

**Output**: Cross-agent replay report, schema compatibility matrix, data scale roadmap v1, S2 decision log.
**Status**: Completed with conditions; see `docs/reports/day4_sync_s2/`.

---

### Day 5 — Friday, March 6: Failover Drills & Resilience Testing (**Sync S3**)

**Theme**: Break things on purpose, prove the system recovers.

- [ ] **Sentinel failover drill**:
  - Simulate primary NSE API outage → verify automatic fallback to scraper connector.
  - Verify `reduce-only` / `close-only` advisory flag is raised on feed integrity failure.
  - Verify automatic recovery when primary feed returns.
  - Measure MTTR (mean time to recovery).
- [ ] **Macro stale-source drill**:
  - Simulate delayed macro release → verify stale markers appear correctly.
  - Verify downstream weighting controls acknowledge stale data.
- [ ] **Preprocessing idempotency check**: Re-run the same Gold-generation pipeline twice → confirm identical output hash (reproducibility).
- [ ] **Feed freeze simulation**: Stop all ticks for a configurable window → confirm engine distinguishes zero volume from missing data.
- [ ] **Incident runbook update**: Log all drill outcomes into the runbook with MTTR and escalation path.

> **🔄 Sync S3 — with Partner (Joint Failover Drill)**
> - Partner simulates text source outage (e.g., Economic Times scraper down) → verify fallback route activation, sidecar `source_route_detail` updates.
> - Partner simulates PDF extraction failure → verify partial record handling.
> - Cross-check: when text feed goes down, does it affect Sentinel/Macro/Preprocessing at all? (Expected: no coupling.)
> - Log joint drill results and MTTR for text sources.

**Output**: Failover drill evidence doc (all agents), MTTR measurements, updated incident runbook, S3 joint drill log.

---

### Day 6 — Saturday, March 7: SLA Dashboard Finalization & Exit Checklist Evidence

**Theme**: Close every checkbox on the Phase 1 exit checklist.

- [ ] **SLA dashboard go-live**:
  - Populate with real data from the last 10+ consecutive trading days.
  - Verify all KPIs meet thresholds:
    - Data uptime ≥ 99.5%.
    - Core symbol completeness ≥ 99.0%.
    - Macro job adherence ≥ 95%.
    - Reproducibility checks 100% pass.
    - Leakage tests 100% pass.
    - Provenance coverage 100%.
    - Text metadata completeness ≥ 98% *(partner's metric, request evidence)*.
- [ ] **Audit trail verification**: Run a full audit report — every record in Gold must trace back to Bronze with provenance, schema version, and quality status.
- [ ] **Data contracts freeze**: Tag all schema versions as `FROZEN_FOR_PHASE_2` in the schema registry.
- [ ] **Exit checklist walkthrough**: Go through §9 (Phase 1 Exit Checklist) item by item, link evidence for each.
- [ ] **GO / NO-GO pre-check**: Evaluate §16.1 benchmarks and flag any at-risk items.

**Output**: SLA dashboard (live with 10-day evidence), audit trail report, frozen schemas, Phase 1 exit checklist (evidence-linked), GO/NO-GO pre-assessment.

---

### Day 7 — Sunday, March 8: Phase 2 Handoff Package & Final Sign-Off (**Sync S4**)

**Theme**: Package everything and sign off.

- [ ] **Phase 2 handoff artifacts** (per §10):
  - Canonical feature tables and data dictionary.
  - Provenance and quality scoring fields documentation.
  - Replayable sample datasets for model benchmarking.
  - Universe selection filters and rebalance cadence doc.
  - Dataset tiering and retention policy doc.
  - Signed release note with open issues and mitigation owners.
- [ ] **Architecture + runbook package**:
  - Final architecture diagram (all 4 agents + storage tiers + replay framework).
  - Updated runbooks with drill outcomes and MTTR targets.
  - Dataset catalog with Bronze/Silver/Gold paths and retention policies.
  - Known limitations list.
- [ ] **Data scale roadmap finalization**: Finalize quarterly volume checkpoints and tier migration triggers (§5.5).

> **🔄 Sync S4 — with Partner (Final Sign-off)**
> - Partner delivers textual handoff package (text schema docs, adapter runbook, Hinglish strategy, source allowlist, dedup/spam tuning parameters).
> - Joint review of combined exit checklist — all streams must be green or formally waived.
> - Sign combined GO / NO-GO gate document.
> - Publish open issues register with owners and deadlines for Hypercare (Week 7).
> - Archive all checkpoint artifacts (CP1–CP4 from prior weeks + CP5 final).

**Output**: Complete Phase 2 handoff package, signed GO/NO-GO memo, open issues register, Hypercare entry brief.

---

## Week 6 Exit Criteria

| Criterion | Evidence Required |
|-----------|-------------------|
| SLA dashboard live with ≥ 10 trading-day data | Dashboard screenshot + data dump |
| All KPIs at or above threshold | KPI summary table with actual vs target |
| Failover drills executed for all agents | Drill logs with MTTR measurements |
| Replay framework: full-day Bronze → Gold reconstruction | Replay test report with hash verification |
| Data contracts frozen for Phase 2 | Schema registry tags |
| Audit trail: 100% provenance coverage | Audit report |
| Cross-agent integration test passed | Joint replay report + schema compatibility matrix |
| Phase 2 handoff pack delivered | Artifact checklist signed |
| GO / NO-GO gate evaluated | Signed memo |

---

## Risk Watch for Week 6

| Risk | Mitigation |
|------|-----------|
| 10 consecutive trading-day data insufficient | Start counting from earliest available; if < 10 days, extend into Hypercare with daily monitoring |
| Text metadata completeness < 98% | Partner escalation at S2; prioritize dedup/spam fixes on Day 5–6 |
| Replay non-determinism in Preprocessing | Debug hash divergence on Day 3; root cause before Day 4 cross-agent run |
| Latency P99 exceeds budget | Profile hotspots on Day 1; batch/cache optimizations on Day 2; defer cosmetic optimizations to Hypercare |
| Schema drift discovered late | Contract freeze check on Day 2; any delta must be resolved before Day 4 S2 sync |
