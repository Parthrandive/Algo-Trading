# Consensus Agent — Remaining Work Plan

**Status**: Week 4 ~90% COMPLETE | **Date**: March 21, 2026  
**Ref**: Phase_2_Analyst_Board_Plan.md → Week 4 (March 30 – April 5)

---

## What's Already Built ✅ (much more than initially assessed)

### Core Agent (`src/agents/consensus/consensus_agent.py`)
- LSTAR (Logistic Smooth Transition) model — `compute_lstar_transition()`
- ESTAR (Exponential Smooth Transition) model — `compute_estar_transition()`
- Auto-switching: ESTAR when volatility > threshold, else LSTAR
- Weighted consensus (tech=42%, regime=35%, sentiment=23%)
- Transition-adjusted dynamic weights
- Crisis routing (max 70% cap via `max_crisis_weight`)
- Safety bias boost for protective signals
- Divergence detection → NORMAL / REDUCED / PROTECTIVE risk modes
- Confidence computation with crisis penalty

### Schemas (`src/agents/consensus/schemas.py`)
- `ConsensusInput` — Pydantic model with all transition vars (volatility, macro_differential, rbi_signal, sentiment_quantile, crisis_probability)
- `ConsensusOutput` — full output model with transition score, risk mode, weights
- `AgentSignal` — normalized sub-agent signal with `is_protective` flag
- `ConsensusTransitionModel` — LSTAR/ESTAR enum
- `ConsensusRiskMode` — NORMAL/REDUCED/PROTECTIVE enum

### Adapters (`src/agents/consensus/adapters.py`)
- `build_consensus_input()` — coerces raw agent outputs into ConsensusInput
- `build_consensus_input_from_phase2_payload()` — Phase 2 payload adapter
- Full datetime, float, and signal coercion utilities

### Offline Pipeline (`src/agents/consensus/offline_pipeline.py` — 1,502 lines)
- Walk-forward expanding-window backtest
- Technical model training (LogisticRegression + Ridge)
- Regime model training (HMM + PEARL + OOD)
- Sentiment scoring via SentimentAgent integration
- Hourly sentiment aggregation with freshness/staleness tracking
- Weighted baseline consensus
- **Challenger model: BayesianRidge with LSTAR/ESTAR gate** (this IS the Bayesian estimation)
- Ablation table (no_sentiment, no_regime, simple_average)
- Leakage audit (macro asof, timestamp alignment, future label join)
- Full metrics: accuracy, balanced_accuracy, macro_F1, confusion matrix, per-class P/R/F1
- Slice accuracy (high-volatility, stale sentiment, strong disagreement)
- Proxy utility after transaction costs
- Model recommendation logic (challenger vs baseline)
- Markdown report generation
- Model serialization (pickle + JSON)

### Runtime Config (`configs/consensus_agent_runtime_v1.json`)
- All transition, routing, risk mode, and weight parameters externalized

### DB Wiring
- `Phase2Recorder.save_consensus_signal()` — exists ✅
- `consensus_signals` table — 15 rows from live agent run ✅

---

## What's Missing ❌

### 1. Consensus Model Card Registration
**Priority**: HIGH | **Effort**: 15 min

**What**: No model card registered in DB for the consensus agent.

**Tasks**:
- [ ] Register model cards: `consensus_weighted_v1` and `consensus_challenger_v1`
- [ ] Include hyperparameters from runtime config JSON

---

### 2. Stability Test Suite
**Priority**: HIGH | **Effort**: 0.5 day

**What**: The plan requires explicit stability tests that don't exist as standalone test files.

> Note: The offline pipeline *does* compute slice-level accuracy (high-vol, stale sentiment, disagreement) which partially covers this. But no standalone test file exists.

**Tasks**:
- [ ] Create `tests/test_consensus_stability.py`
  - Test: consensus score bounded [-1, 1] for all inputs
  - Test: high-volatility inputs switch to ESTAR
  - Test: crisis probability > 0.7 triggers regime weight increase
  - Test: text floods alone cannot flip direction (sentiment weight capped at 23%)
  - Test: max agent divergence → PROTECTIVE mode (score = 0)
  - Test: stale/missing sentiment → weight auto-reduced
  - Test: OOD alien state → protective routing
- [ ] Run and verify all tests pass

---

### 3. Dual-Loop Boundary Tests
**Priority**: MEDIUM | **Effort**: 0.5 day

**What**: Tests ensuring fast-loop (p99 ≤ 8ms) and slow-loop boundaries are respected.

**Tasks**:
- [ ] Create `tests/test_consensus_dual_loop.py`
  - Test: `ConsensusAgent.run()` completes in < 10ms (no DB or model loading)
  - Test: offline pipeline runs end-to-end without crossing loop boundaries
  - Test: sentinel mode triggers reduced confidence when data is stale

---

### 4. Live DB Pipeline Integration
**Priority**: MEDIUM | **Effort**: 0.5 day

**What**: The offline pipeline reads from parquet files. A live version that reads from DB and writes `consensus_signals` to DB doesn't exist yet.

**Tasks**:
- [ ] Create `scripts/run_live_consensus.py`
  - Read latest technical_predictions, regime_predictions, sentiment_scores from DB
  - Build ConsensusInput payload
  - Run ConsensusAgent.run()
  - Save to consensus_signals via Phase2Recorder
- [ ] Can be cron-scheduled or called manually

---

### 5. Offline Pipeline Execution (at least one complete run)
**Priority**: HIGH | **Effort**: 0.5 day

**What**: The 1,500-line offline pipeline exists but I see no report output indicating it has been successfully run end-to-end with current data.

**Tasks**:
- [ ] Run `offline_pipeline.run_pipeline()` with current symbols
- [ ] Verify it produces: metrics.json, report.md, artifacts CSVs, model pickles
- [ ] Check the recommendation output (weighted vs challenger)
- [ ] Save report to `data/reports/consensus_runs/`

---

## Summary Table

| # | Task | Priority | Effort | Status |
|---|---|---|---|---|
| 1 | Register consensus model cards | HIGH | 15 min | ❌ |
| 2 | Stability test suite | HIGH | 0.5 day | ❌ |
| 3 | Dual-loop boundary tests | MEDIUM | 0.5 day | ❌ |
| 4 | Live DB pipeline script | MEDIUM | 0.5 day | ❌ |
| 5 | Run offline pipeline end-to-end | HIGH | 0.5 day | ❌ |

**Total estimated effort**: ~2 days

---

## Recommended Build Order

```
Step 1: Register model cards (#1) — 15 min, quick win
Step 2: Run offline pipeline (#5) — validates the entire system works
Step 3: Stability tests (#2) — proves robustness
Step 4: Live DB script (#4) — connects offline to live
Step 5: Dual-loop tests (#3) — validates latency boundaries
```

---

## Key Insight: Earlier Audit Was Wrong

The initial audit flagged LSTAR, ESTAR, and Bayesian estimation as "NOT BUILT". In reality:

| Feature | Initially Assessed | Actually |
|---|---|---|
| LSTAR | ❌ Not built | ✅ `compute_lstar_transition()` in consensus_agent.py |
| ESTAR | ❌ Not built | ✅ `compute_estar_transition()` in consensus_agent.py |
| Bayesian estimation | ❌ Not built | ✅ `BayesianRidge` challenger in offline_pipeline.py |
| Transition function | ❌ Missing macro/RBI | ✅ `_transition_gate()` uses volatility, macro_differential, rbi_signal, sentiment_quantile |
| Stability slice tests | ❌ Not built | ⚠️ Computed in pipeline metrics but no standalone test file |

This is because there were two consensus agent implementations — the DB-wiring version created mid-session, and the original complete version that was already there.
