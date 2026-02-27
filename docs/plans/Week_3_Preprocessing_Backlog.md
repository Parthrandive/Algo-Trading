# Week 3 — Your Plan: Preprocessing Agent
**Dates:** Mon, February 23 – Sun, March 1, 2026
**Owner:** You
**Aligned to:** Phase 1 Data Orchestration Execution Plan v2.0 — Sections 5.3, 5.5
**Master Plan Reference:** Multi_Agent_AI_Trading_System_Plan_Updated v1.3.7 — Sections 5.3, 5.5

---

## Goal
Build the Preprocessing Agent with deterministic transforms, schema-validated loaders, lag-alignment, leakage testing, and reproducibility hashing. Consume macro sample payloads at CP3 to validate end-to-end compatibility. Lay groundwork for the Data Scale & Replay framework (Section 5.5).

## Phase 1 GO/NO-GO Contribution (Section 16.1)
This workstream contributes to:
- **Leakage tests = 100% pass**
- **Canonical dataset reproducibility checks = 100% pass**
- **Provenance tagging coverage = 100%**
- **Core symbol completeness ≥ 99.0%** (via universe selection validation)

---

## Shared Schema Contract (Must Match Partner's)

Your loaders **must accept** `MacroIndicator` records with `schema_version: "1.1"` and these indicator types:
```
CPI, WPI, IIP, GDP, INR_USD, BRENT_CRUDE, US_10Y, INDIA_10Y,
FII_FLOW, DII_FLOW, REPO_RATE, FX_RESERVES, INDIA_US_10Y_SPREAD, RBI_BULLETIN
```

Scope rule:
- **Accepted enum set:** all 14 indicator values above.
- **Week 3 required integration set (CP3):** `CPI, WPI, IIP, FII_FLOW, DII_FLOW, FX_RESERVES, RBI_BULLETIN, INDIA_US_10Y_SPREAD`.
- For `RBI_BULLETIN`, expect numeric event encoding from Macro Monitor: `value=1.0`, `unit="count"`, `period="Irregular"`.

Your preprocessing I/O contract defines:
- **Input:** Silver-layer `MacroIndicator` (v1.1) + market records (`Bar` v1.0, `Tick` v1.0)
- **Output:** Gold-layer `TransformOutput` with output hash, config version, and processed records

### Key Acceptance Criteria (Section 5.3)
- Feature engineering is **deterministic and reproducible**
- Rolling window normalization and directional change thresholds implemented
- Corporate action adjustments validated against a reference source
- Leakage tests confirm **strict time alignment and lagging**
- Feature approval process defined: **proposal → offline eval → shadow → promote**

### Data Scale & Replay Contribution (Section 5.5)
- Replay framework can reconstruct any trading day from raw feeds → feature artifacts with **deterministic identifiers**
- Dataset tiering awareness: hot/warm/cold with retention policies
- Dataset snapshot IDs assigned to every pipeline run output

---

## Daily Backlog

### Day 1 (Mon, Feb 23) — I/O Contract Freeze 🔒 CP1
- [x] Create `src/schemas/preprocessing_data.py`:
  - `TransformConfig` — name, version, input/output schema versions, parameters
  - `TransformOutput` — output_hash (SHA-256), input_snapshot_id, transform_config_version, records
  - `PreprocessingContract` — input/output field specs with type + range constraints
- [x] Create `configs/preprocessing_contract_v1.json`:
  - Accepted input schemas: `MacroIndicator v1.1`, `Bar v1.0`, `Tick v1.0`
  - Output schema version: `TransformOutput v1.0`
  - Dataset snapshot ID format and deterministic identifier rules (Section 5.5)
- [x] Verify partner's `MacroIndicatorType` enum matches the shared contract list
- [x] Define feature approval workflow skeleton: proposal → offline eval → shadow → promote (Section 5.3)
- [x] Create `docs/plans/CP1_Contract_Freeze.md` with evidence
- [x] **Deliverable:** Contract frozen, schema guardrails committed, feature approval workflow documented

### Day 2 (Tue, Feb 24) — Transform Graph Scaffold + Loaders
- [x] Create `src/agents/preprocessing/` package
- [x] Build `loader.py`:
  - `MacroLoader` — reads Silver Parquet/JSON, validates against `MacroIndicator` schema v1.1
  - `MarketLoader` — reads Silver OHLCV + real-time tick Parquet, validates against `Bar`/`Tick` schemas
  - Schema version mismatch → raise `SchemaVersionError`
  - Assign dataset snapshot IDs to loaded data (Section 5.5)
- [x] Build `transform_graph.py`:
  - `TransformNode` base class: `input_schema`, `output_schema`, `version`, `transform(df) → df`
  - `TransformGraph` — DAG of nodes, topological execution order
  - Config-driven: reads `configs/transform_config_v1.json`
  - Provenance: every node output tagged with source, schema version, quality status
- [x] **Deliverable:** Scaffold committed, loads sample data successfully with snapshot IDs

### Day 3 (Wed, Feb 25) — Normalization Modules + Config Versioning
- [x] Build `normalizers.py`:
  - `ZScoreNormalizer(TransformNode)` — rolling z-score with configurable window
  - `MinMaxNormalizer(TransformNode)` — rolling min-max scaling
  - `LogReturnNormalizer(TransformNode)` — log returns for price series
  - `DirectionalChangeDetector(TransformNode)` — directional change thresholds (Section 5.3)
- [x] Create `configs/transform_config_v1.json`:
  - Each transform: name, version, parameters (window size, etc.), input/output schema
  - Config versioned and immutable per run — changes create new version
- [x] Tests: `tests/agents/preprocessing/test_transform_graph.py`
  - DAG execution order, version mismatch rejection, config loading, determinism
- [x] **Deliverable:** Normalizers + directional change detector working with versioned config

### Day 4 (Thu, Feb 26) — Lag-Alignment + Pipeline Wiring
- [x] Build `lag_alignment.py`:
  - `LagAligner` — aligns macro indicators to market timeseries with proper lag
  - Configurable lag rules per indicator (e.g., CPI lagged to publication date, not observation date)
  - **Key guarantee:** No future macro data leaks into aligned output (Section 5.3 leakage requirement)
  - Corporate action adjustments validated against reference source (Section 5.3)
- [x] Build `pipeline.py`:
  - `PreprocessingPipeline` — orchestrates: loader → lag alignment → transform graph → output
  - Produces `TransformOutput` with SHA-256 output hash and dataset snapshot ID
  - Deterministic: same input snapshot → same output hash (Section 5.3 reproducibility)
- [x] Build initial replay support (Section 5.5):
  - Pipeline can reconstruct outputs from stored Bronze/Silver snapshots
  - Event-time playback mode (wall-clock playback deferred to Week 6)
- [x] **Deliverable:** Full pipeline wired with replay support, end-to-end test with mock data passes

### Day 5 (Fri, Feb 27) — Leakage + Reproducibility 🔒 CP2
- [ ] Build `leakage_test.py`:
  - `LeakageTestHarness` — checks for future information leaking into transforms
  - Tests: timestamp ordering after alignment, no look-ahead bias, lag correctness
  - Strict time alignment and lagging verification (Section 5.3 acceptance)
- [ ] Build `reproducibility.py`:
  - `ReproducibilityHasher` — SHA-256 of full output given frozen input snapshot + config
  - Deterministic replay: same input snapshot → identical hash
  - Dataset snapshot IDs tracked through pipeline (Section 5.5)
- [ ] Tests:
  - `tests/agents/preprocessing/test_leakage.py` — with known-bad data (planted future leak)
  - `tests/agents/preprocessing/test_reproducibility.py` — hash stability across 3 runs
- [ ] Create `docs/plans/CP2_Pipeline_Readiness.md`
- [ ] **Deliverable:** CP2 evidence — leakage tests green (100% pass), reproducibility hash stable

### Day 6 (Sat, Feb 28) — Consume Macro Samples 🔒 CP3 (Sync Gate A)
- [ ] **Consume** partner's `data/macro_samples/*.json` files
- [ ] Run through `MacroLoader` → validate list payloads with `TypeAdapter(list[MacroIndicator]).validate_python(payload)`
- [ ] Run through full `PreprocessingPipeline`:
  - Lag alignment with macro sample timestamps
  - Normalization transforms (z-score, min-max, log returns)
  - Directional change detection
  - Output hash generation with dataset snapshot ID
- [ ] Compatibility test results:
  - Schema validation: PASS/FAIL per indicator
  - Transform execution: PASS/FAIL
  - Output hash: recorded for baseline
  - Provenance tags: verified on every output record
- [ ] Create `docs/plans/CP3_Sync_Gate_A.md` with results table
- [ ] **Deliverable:** CP3 evidence — all 8 Week 3 required indicators load + transform successfully

> **🔗 SYNC GATE A:** You depend on partner publishing `data/macro_samples/` by end of Day 6. If samples are delayed:
> 1. Use locally generated mock samples (matching schema v1.1) to unblock
> 2. Re-run compatibility when partner samples arrive
> 3. Note discrepancy in CP3 document

### Day 7 (Sun, Mar 1) — Deterministic Replay + Sign-off 🔒 CP4
- [ ] Freeze a baseline input snapshot (macro samples + OHLCV Silver data) with snapshot ID
- [ ] Run deterministic replay: full pipeline 3 times → verify identical output hash
- [ ] Run event-time replay from stored snapshots (Section 5.5 verification)
- [ ] Publish baseline output hash as reference for future regression testing
- [ ] Cross-reference partner's CP4 completeness report
- [ ] Create `docs/plans/CP4_Week_3_Signoff.md`:
  - Reproducibility: PASS (3 identical hashes)
  - Leakage: PASS (all tests green — 100% per GO/NO-GO gate)
  - Compatibility: PASS (partner samples consumed)
  - Replay: PASS (event-time reconstruction verified)
  - Integration readiness: READY for Week 4
- [ ] **Deliverable:** CP4 evidence — integration-readiness note published

---

## Checkpoint Summary

| CP | Date | Gate | Pass Criteria |
|----|------|------|---------------|
| CP1 | Feb 23 | Contract Freeze | `preprocessing_contract_v1.json` committed, accepted input = MacroIndicator v1.1 + Bar/Tick v1.0, feature approval workflow documented |
| CP2 | Feb 27 | Preprocessing Readiness | Leakage tests 100% green, reproducibility hash stable across 3 runs, dataset snapshot IDs assigned |
| CP3 | Feb 28 | **Sync Gate A** | All 8 macro sample payloads load + transform through pipeline with valid provenance |
| CP4 | Mar 1 | Week 3 Readiness | Deterministic replay passes, event-time replay verified, integration-readiness note published |

---

## Acceptance Criteria Cross-Reference (Section 5.3 + 5.5)

| Acceptance Criterion | Where Addressed |
|---------------------|-----------------|
| Feature engineering is deterministic and reproducible | CP2 — reproducibility hash; CP4 — deterministic replay |
| Rolling window normalization and directional change thresholds | Day 3 — normalizers + directional change detector |
| Corporate action adjustments validated against reference source | Day 4 — lag alignment with corp action handling |
| Leakage tests confirm strict time alignment and lagging | CP2 — leakage test harness (100% pass) |
| Feature approval process defined (proposal → offline eval → shadow → promote) | CP1 — workflow skeleton |
| Replay framework reconstructs from raw feeds with deterministic IDs | Day 4 + CP4 — event-time replay |
| Dataset tiering defined (hot/warm/cold) | CP1 — preprocessing contract |

---

## Dependencies on Partner (Macro Monitor Agent)

| When | What you need from partner | Fallback if delayed |
|------|---------------------------|-------------------|
| CP1 (Feb 23) | Frozen `MacroIndicatorType` enum with all 14 values | Use proposed enum list; reconcile at CP3 |
| CP3 (Feb 28) | `data/macro_samples/*.json` with valid records for all 8 indicators | Generate mock samples locally; re-test when partner delivers |
| CP4 (Mar 1) | Completeness report for cross-reference | Note as open item in sign-off |

---

## New Files Summary

| File | Purpose |
|------|---------|
| `src/schemas/preprocessing_data.py` | TransformConfig, TransformOutput, PreprocessingContract models |
| `src/agents/preprocessing/__init__.py` | Package init |
| `src/agents/preprocessing/loader.py` | Schema-validated Silver layer loaders with snapshot IDs |
| `src/agents/preprocessing/transform_graph.py` | TransformNode + TransformGraph DAG |
| `src/agents/preprocessing/normalizers.py` | ZScore, MinMax, LogReturn, DirectionalChange transforms |
| `src/agents/preprocessing/lag_alignment.py` | Macro-to-market lag alignment + corp action validation |
| `src/agents/preprocessing/pipeline.py` | PreprocessingPipeline orchestrator with replay support |
| `src/agents/preprocessing/leakage_test.py` | LeakageTestHarness |
| `src/agents/preprocessing/reproducibility.py` | ReproducibilityHasher |
| `configs/preprocessing_contract_v1.json` | I/O contract specification with snapshot ID format |
| `configs/transform_config_v1.json` | Transform definitions (versioned, immutable per run) |
