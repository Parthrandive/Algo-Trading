## Implementation Plan for Week 4: Textual Data Agent (Starting March 2, 2026)

**Alignment Note**: This weekly plan is scoped as a Phase 1 delivery slice, but it includes mandatory hooks needed by Phase 2 Sentiment, Phase 4 Risk, and Phase 5 validation gates from the integrated multi-agent plan (v1.3.7).

**Parallel Coordination References**
- [Week 4 GlobalMacroExogenous Agent Plan](./week-4-GlobalMacroExogenous-Agent%20plan.md)
- [Week 4 Parallel Integration Checkpoints Plan](./week-4-Parallel-Integration-Checkpoints%20plan.md)

**Overall Goal for the Week**: Deliver a working Textual Data Agent that collects text from approved sources (NSE news, Economic Times, RBI reports, earnings transcripts, X posts), supports PDF extraction and code-mixed English/Hinglish handling, and produces auditable outputs for Sentiment ingestion with provenance, reliability scoring, and quality controls.

**Mandatory Outputs by End of Week**
- Source adapters for the required textual feeds with normalized schema.
- Provenance tags per record, including whether source came from primary API/feed or fallback scraper.
- Compliance filter that blocks unpublished/embargoed/unlicensed data and logs reject reasons.
- Quality and safety controls: missingness, deduplication, spam/manipulation flags, slang-scam patterns.
- Contract-safe canonical records that validate against existing text schemas (`NewsArticle_v1.0`, `SocialPost_v1.0`, `EarningsTranscript_v1.0`).
- Sidecar sentiment/risk metadata keyed to canonical records (`confidence`, `ttl_seconds`, manipulation and compliance diagnostics).
- Test evidence for leakage/time alignment and spot checks for PDF extraction quality.

## Cross-Plan Mapping (v1.3.7)
- Phase 1.4 Textual Data Agent requirements: source coverage, X keyword/semantic rules, PDF spot checks, Hinglish strategy.
- Compliance requirements: only publicly released or contractually licensed feeds.
- Phase 1 go/no-go dependency: provenance coverage and leakage test pass rates.
- Phase 2 dependency: timestamp alignment and robustness controls for spam/adversarial/manipulative text.
- Risk dependency: sentiment reliability signals and downgrade-ready flags for mismatch/manipulation scenarios.

## Parallel Workstream Dependency (GlobalMacroExogenous)
- Textual and GlobalMacroExogenous are independent build tracks for Week 4.
- Integration is checkpoint-driven through CP1, CP2, and CP3.
- No direct code coupling is required in this week; only artifact and contract exchange is required.

## Shared Integration Artifacts
Textual outputs for checkpoints:
- Canonical text schema validation report.
- Sidecar metadata report (`confidence`, `ttl_seconds`, manipulation and compliance diagnostics).
- Timestamp alignment evidence (UTC-based).

Inputs expected from partner (GlobalMacroExogenous):
- Exogenous schema report (`ExogenousIndicator_v1.0`).
- Five-signal sample payload package.
- Freshness and fallback behavior report.

## Data Contract (Textual Agent -> Sentiment Agent)
### 1. Canonical Contract (must be schema-valid and DB-safe)
Canonical payloads must validate against current models and registry keys:
- `NewsArticle_v1.0`
- `SocialPost_v1.0`
- `EarningsTranscript_v1.0`

Common mandatory fields for all text records:
- `source_id`
- `timestamp` (event time)
- `content`
- `source_type`
- `language` (`en`, `hi`, `code_mixed`)
- `ingestion_timestamp_utc`
- `ingestion_timestamp_ist`
- `schema_version` (`1.0` for this week)
- `quality_status` (`pass`, `warn`, `fail`)

Type-specific required fields:
- `NewsArticle`: `headline`, `publisher`
- `SocialPost`: `platform`, `likes`, `shares`
- `EarningsTranscript`: `symbol`, `quarter`, `year`

### 2. Sidecar Metadata (non-canonical; never injected into canonical schema payload)
Store operational fields in a sidecar record keyed by (`source_type`, `source_id`, `ingestion_timestamp_utc`):
- `source_route_detail` (`primary_api`, `official_feed`, `fallback_scraper`)
- `quality_flags` (list)
- `manipulation_risk_score` (0.0 to 1.0)
- `confidence` (0.0 to 1.0)
- `ttl_seconds`
- `compliance_status` (`allow`, `reject`)
- `compliance_reason` (nullable string)

Implementation note:
- Keep canonical payload strict (`extra=forbid` compatibility) and persist sidecar metadata separately so DB writes remain lossless for core fields.

## Daily Execution Plan

### Day 1: March 2, 2026 (Monday) - Setup, Policy, and Schema
- Finalize source allowlist and legal/compliance rules for each feed.
- Freeze canonical schema mapping to existing text models and define sidecar metadata schema.
- Create `textual_data_agent.py` skeleton with modules for adapters, cleaners, validators, and exporters.
- Define baseline keyword/semantic query templates for X (India market context only).
- Output: design note + schema file + adapter stubs.

Day 1 implementation snapshot (2026-03-02):
- Runtime policy and X query templates: `configs/textual_data_agent_runtime_v1.json`
- Sidecar schema: `src/schemas/text_sidecar.py`
- Textual skeleton package: `src/agents/textual/`
- Design note: `docs/plans/week-4-textual-data-agent-day1-design-note.md`
- Day 1 freeze tests: `tests/test_textual_day1.py`

### Day 2: March 3, 2026 (Tuesday) - Core Source Ingestion + Provenance
- Implement collectors for NSE news, Economic Times, RBI reports.
- Enforce canonical text fields (`source_id`, `timestamp`, `content`, `source_type`, ingestion timestamps, schema version, quality status).
- Add compliance gate before persistence; reject and log blocked content.
- Persist canonical payloads plus sidecar diagnostics for replay/audit.
- Join CP1 contract freeze with partner to lock schema IDs, output paths, and compliance rules.
- Output: working ingestion for 3 core sources with provenance and compliance logs.

### Day 3: March 4, 2026 (Wednesday) - X Collection + Reliability Controls
- Implement X ingestion with keyword and semantic rules, rate-limit handling, and retries.
- Add India relevance filters and source reliability weighting.
- Tag noisy/suspicious bursts (duplicate cascades, abnormal hashtag spam).
- Output: X pipeline integrated with reliability tags and noise flags.

### Day 4: March 5, 2026 (Thursday) - PDF Pipeline + Hinglish/Code-Mixed Processing
- Build PDF extraction for RBI bulletins and earnings transcripts.
- Run spot checks and record extraction quality metrics.
- Implement Hinglish/code-mixed processing strategy (detection, normalization, transliteration where needed).
- Add slang-scam lexicon hooks used by manipulation detection.
- Output: validated PDF + language pipeline with spot-check report.

Day 4 implementation snapshot (2026-03-05):
- Day 4 spot-check report: `docs/plans/week-4-textual-data-agent-day4-spot-check-report.md`
- PDF extraction service and metrics hooks: `src/agents/textual/services/pdf_service.py`
- Language detection/normalization/transliteration hooks: `src/agents/textual/services/language_service.py`
- Scam lexicon safety hooks: `src/agents/textual/services/safety_service.py`
- Day 4 validation tests: `tests/test_textual_day4.py`

### Day 5: March 6, 2026 (Friday) - Quality, Safety, and Sentiment Handoff
- Add missingness, deduplication, outlier-length, and malformed timestamp checks.
- Add adversarial/spam/manipulation filters and `manipulation_risk_score`.
- Add sidecar handoff fields: `confidence`, `ttl_seconds`, freshness checks.
- Export schema-valid canonical records to Silver and sidecar metadata to a separate artifact.
- Ensure no unknown fields are added to canonical payloads (strict-schema compatibility).
- Join CP2 integration review to validate compatibility with exogenous artifact pack and shared UTC alignment policy.
- Output: end-to-end pipeline producing sentiment-ready records.

### Day 6: March 7, 2026 (Saturday) - Test Pack and Gate Evidence
- Build tests for ingestion, provenance completeness, timestamp alignment, and leakage.
- Add tests for PDF extraction quality thresholds and Hinglish normalization sanity checks.
- Add failure-mode tests (rate limits, source outage, fallback route activation).
- Produce gate evidence summary (coverage, failure counts, unresolved defects).
- Output: test report + defect log + remediation list.

### Day 7: March 8, 2026 (Sunday) - Review, Hardening, and Hand-off
- Run full-day dry run on recent data slices.
- Review canonical validation errors, compliance rejects, stale TTL breaches, and fallback-source rates.
- Write README/runbook section (inputs, outputs, schema, failure handling).
- Prepare partner review packet with what is complete vs deferred.
- Join CP3 final signoff and publish cross-stream decision log and open issues list.
- Output: review-ready delivery package.

## Week-4 Exit Criteria (Must Pass)
- Source coverage complete for: NSE news, Economic Times, RBI reports, earnings transcripts, X.
- Provenance tagging coverage = 100 percent on emitted records.
- Compliance gate active with auditable reject logs.
- Leakage and timestamp alignment tests = 100 percent pass for current suite.
- PDF extraction spot checks documented and accepted.
- Hinglish/code-mixed handling strategy documented and exercised on sample data.
- Canonical text records pass schema validation for `NewsArticle_v1.0` / `SocialPost_v1.0` / `EarningsTranscript_v1.0`.
- Canonical persistence path is DB-safe (no required-field loss, no unknown-field schema rejections).
- Sidecar sentiment fields (`confidence`, `ttl_seconds`) present for all `quality_status=pass` canonical records.
- Manipulation/spam flags available for downstream risk and sentiment downgrade logic.
- Cross-stream checkpoint signoff complete (CP1, CP2, CP3) with documented pass/fail outcomes.

## Explicit Deferrals (Not in Week 4)
- Full sentiment model training/fine-tuning.
- Fast cache runtime tuning and sub-100 ms scoring optimization.
- Live trading integration, crisis override activation, and A/B production gates.
