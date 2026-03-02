# Week 4 Plan: GlobalMacroExogenous Agent

## Title and Metadata
- Week window: Monday, March 2, 2026 to Sunday, March 8, 2026
- Owner: Partner
- Dependencies: WorldMonitor feeds/panels, approved public proxy feeds, existing provenance and quality semantics
- Master-plan alignment: Phase 1 Data Orchestration enrichment; downstream use in Regime and Consensus layers as Slow-Loop context (no Fast-Loop dependency in Week 4)

Related plans:
- [Week 4 Textual Data Agent Plan](./week-4-textual-data-agent-plan.md)
- [Week 4 Parallel Integration Checkpoints Plan](./week-4-Parallel-Integration-Checkpoints%20plan.md)

## Goal
Build a separate exogenous-risk signal agent that computes point-in-time global spillover features for India markets (USD/INR and NSE universe) while keeping outputs deterministic, provenance-tagged, and replay-safe.

## Scope and Non-Goals
In scope:
- Source ingestion and normalization for exogenous macro/global risk signals
- Signal scoring into a fixed 5-indicator exogenous feature family
- Provenance, compliance, freshness, and reliability controls
- Nightly scheduled run plus on-demand refresh path
- Handoff artifacts for integration checkpoints

Out of scope:
- Direct Fast Loop dependency
- Live weight escalation in decision policy
- Production override logic or auto-trading controls

## Architecture Boundary
- This agent is independent from the Textual Data Agent.
- Textual Agent remains focused on unstructured text collection and text quality controls.
- GlobalMacroExogenous Agent produces structured numeric exogenous signals.
- Integration between streams is artifact-based via CP1/CP2/CP3 checkpoints; no shared mutable logic in Week 4.

## Data Sources
- WorldMonitor panels/endpoints for sanctions, trade friction, macro velocity, and policy-rate context.
- Public proxy sources for redundancy when needed.
- Existing local macro context as optional secondary reference.
- Compliance policy: only publicly released or contractually licensed feeds; rejected sources must be logged with reason codes.

## Public Interface: `ExogenousIndicator_v1.0`
Contract objective:
- Introduce a new schema family for Week 4 exogenous outputs.
- Do not mutate frozen `MacroIndicator` contract in this week.

Required fields:
- `indicator_name`: enum
  - `GLOBAL_TRADE_DISRUPTION_SCORE`
  - `CENTRAL_BANK_SURPRISE_INDEX`
  - `SANCTIONS_GEOPOLITICAL_INTENSITY`
  - `MACRO_VELOCITY_SCORE`
  - `EXOGENOUS_RISK_PROXY`
- `value`: float
- `unit`: string (`index_0_1` for normalized scores; explicit units when needed)
- `period`: string (`Daily` or `EventDriven`)
- `timestamp`: timezone-aware event timestamp
- `source_type`: existing provenance enum
- `ingestion_timestamp_utc`
- `ingestion_timestamp_ist`
- `schema_version`: `1.0`
- `quality_status`: `pass|warn|fail`
- `quality_flags`: list of rule flags
- `dataset_snapshot_id`
- `compliance_status`: `allow|reject`
- `compliance_reason`

## Scoring Definitions
- `global_trade_disruption_score`: normalized weighted change in trade restrictions and chokepoint stress indicators.
- `central_bank_surprise_index`: normalized divergence proxy between expected and observed global policy-rate path context.
- `sanctions_geopolitical_intensity`: rolling sanctions/unrest intensity trend score.
- `macro_velocity_score`: short-window acceleration score from high-velocity macro narratives/events.
- `exogenous_risk_proxy`: weighted composite of the four prior scores, with weights frozen at CP1.

## Daily Plan (Partner Track)
### Day 1 (Monday, March 2, 2026): Source and Contract Freeze Prep
- Finalize source inventory and compliance allowlist.
- Draft `ExogenousIndicator_v1.0` and scoring formula definitions.
- Define output paths and field dictionary candidates for CP1.
- Output: draft contract pack + source governance sheet.

### Day 2 (Tuesday, March 3, 2026): Connector and CP1 Contract Freeze
- Implement connector skeletons and parser normalization stubs.
- Add provenance and quality scaffolding.
- Complete CP1 contract freeze with schema IDs, output paths, scoring definitions, and compliance rules.
- Output: CP1-ready contract markdown + sample payload seed.

### Day 3 (Wednesday, March 4, 2026): Score Computation and Artifact Generation
- Compute all 5 exogenous scores on sample windows.
- Add persistence path and replay sample artifact generation.
- Output: first complete exogenous sample package.

### Day 4 (Thursday, March 5, 2026): Freshness and Reliability Controls
- Implement freshness rules and stale marker behavior.
- Implement outage fallback and reliability logging.
- Participate in CP2 mid-week integration checkpoint.
- Output: freshness and fallback report + CP2 integration evidence.

### Day 5 (Friday, March 6, 2026): Validation Suite
- Run schema validation tests.
- Run timestamp monotonicity and no-look-ahead checks.
- Validate compliance rejection flow and reason logging.
- Output: validation report and unresolved defect list.

### Day 6 (Saturday, March 7, 2026): Dry Run and Reporting
- Execute recent-window dry run.
- Publish completeness and freshness reports.
- Prepare CP3 signoff materials.
- Output: pre-signoff evidence package.

### Day 7 (Sunday, March 8, 2026): Final Signoff Handoff
- Finalize CP3 pass/fail artifact set.
- Submit unresolved issue register and next sprint recommendations.
- Complete cross-stream signoff with Textual owner.
- Output: CP3 signoff pack.

## Test Cases and Scenarios
- Schema validation with valid and invalid `ExogenousIndicator_v1.0` payloads.
- Point-in-time integrity test ensuring no look-ahead timestamp leakage.
- Freshness behavior test for fresh, stale, and expired inputs.
- Source outage fallback test for WorldMonitor unavailability.
- Cross-stream timestamp alignment check against textual artifacts in UTC.
- Compliance rejection test with blocked source and reason logging verification.

## Exit Criteria
- Five exogenous signals are computed and persisted with full provenance.
- 100 percent schema validation pass on accepted records.
- Freshness policy is active with deterministic stale markers.
- Compliance rejections are logged with reason codes.
- Replay sample package is generated and reviewed at CP3.
