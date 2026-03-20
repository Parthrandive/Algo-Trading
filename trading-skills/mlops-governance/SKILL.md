# SKILL: MLOps & Model Governance
**Project:** Multi-Agent AI Trading System — Indian Market
**Applies To:** All model training, promotion, rollback, and live monitoring workflows
**Version:** aligned with Plan v1.3.7

---

## PURPOSE
This skill governs how every model artifact moves from research to production and is maintained in production. It enforces evidence-first standards, champion-challenger gating, rollback readiness, and research throughput measurement.

---

## 1. EVIDENCE-FIRST POLICY (Non-Negotiable)

### 1.1 Prohibited Language in Approval Artifacts
**Never use fixed uplift claims in any approval document.** The following are examples of prohibited statements:
- ❌ "This model will improve Sharpe by 0.3"
- ❌ "Rust rewrite will deliver 5× latency speedup"
- ❌ "Expected 15 % reduction in drawdown"

**Permitted language:**
- ✅ "Unknown until controlled benchmark or shadow A/B; target is non-regression plus measurable improvement"
- ✅ "Benchmark shows measured p99 improvement of X ms versus baseline on replay workload"

### 1.2 Technology-Change Approval Requirements
Language/runtime/hardware changes (Rust, C++, FPGA) require:
- Measured p99 and p99.9 deltas versus software baseline
- Correctness parity evidence (identical output on test suite)
- Observability parity evidence (metrics, traces, alerts)
- Recovery/failover drill parity evidence

---

## 2. MODEL REGISTRY

### 2.1 Required Fields Per Artifact
Every promoted model artifact must record:
- `model_id`: unique identifier
- `version`: semantic version string
- `training_data_snapshot_hash`: hash of training dataset
- `code_hash`: hash of training pipeline code
- `feature_schema_version`: version of input feature schema
- `hyperparameters`: full config (no defaults assumed)
- `validation_metrics`: all required metrics from Phase gate
- `baseline_comparison`: delta vs naive baseline and prior champion
- `plan_version`: which version of the project plan this artifact was produced under
- `created_by` and `reviewed_by`
- `promotion_gate_checklist`: completed checklist (see Section 4)

### 2.2 Reproducibility Requirement
Every training pipeline must be runnable from raw data to artifact using only the recorded `data_snapshot_hash` and `code_hash`. Reproducibility test is mandatory before promotion.

---

## 3. CHAMPION-CHALLENGER FRAMEWORK

### 3.1 Shadow Mode
- All challengers run in shadow mode before any production promotion
- Shadow window minimum: defined per phase (typically >= 2 weeks for model changes; >= 1 month for major strategy changes)
- Shadow A/B enforcement for USD/INR and Gold track changes: **Tier 2** (after 3 months paper trading); strongly recommended before Tier 2 activation

### 3.2 Promotion Requirements
Challenger promotes to champion only if:
- [ ] Non-regression: Sharpe, Sortino, Calmar, MDD, slippage all >= champion values (no statistically significant regression)
- [ ] Latency gates: p99 and p99.9 within approved budgets
- [ ] Risk budget adherence validated
- [ ] Mode-switching false-positive rate within acceptance limit
- [ ] Zero critical compliance violations in shadow window
- [ ] Completed governance templates (Release Record + CI Benchmark Evidence Checklist)

### 3.3 Demotion Triggers
Immediate demotion (champion reverts to prior version) if:
- Student policy drift exceeds agreement-rate threshold
- p99 regresses beyond degrade threshold
- Any risk limit breach caused by model change
- Critical data integrity failure linked to model artifact

---

## 4. RELEASE GOVERNANCE TEMPLATES

### 4.1 Change Request Template (fill before every promotion)
```
Change Request ID: 
Model / Component: 
Plan Version: 
Author:         Reviewer:
---
CHANGE DESCRIPTION:
[What is changing and why]

EXPECTED IMPACT:
[Unknown until controlled benchmark or shadow A/B; target: non-regression + measurable improvement]

SYSTEM-LEVEL IMPACT:
[Latency budget, risk controls, data dependencies, schema changes]

GO CRITERION:
[All applicable phase-gate checks passing]

NO-GO CRITERION:
[Any latency, risk, or correctness gate failure]

ROLLBACK TRIGGER:
[Specific condition that initiates rollback; owner responsible]

ROLLBACK PLAN:
[Step-by-step rollback procedure; tested date]
```

### 4.2 CI Benchmark Evidence Checklist (required for execution-path changes)
```
Benchmark Suite Run ID:
Workload: [ ] Replay  [ ] Peak-Load Synthetic
---
[ ] p50 latency: ___ ms  (within budget: ___ ms)
[ ] p95 latency: ___ ms  (within budget: ___ ms)
[ ] p99 latency: ___ ms  (stretch target <= 8 ms; degrade threshold > 10 ms)
[ ] p99.9 latency: ___ ms  (within jitter limit)
[ ] Correctness test: PASS / FAIL
[ ] Degrade-path test (synthetic latency stress): PASS / FAIL
[ ] No regression vs prior baseline: PASS / FAIL
Signed by: ___  Date: ___
```

### 4.3 48-Hour Post-Deploy Review Template
```
Deploy ID:       Deploy Date:
Component:       Deployed By:
---
METRICS (first 48 hours):
- Latency: p50 ___ / p99 ___ / p99.9 ___
- Slippage: realised vs model estimate ___
- Risk events (breach count): ___
- Drift alerts: ___
- Incident count: ___
- Mode-switch events: ___

VERDICT: [ ] STABLE  [ ] WATCH  [ ] ROLLBACK
Notes:
Signed by: ___
```

---

## 5. ROLLOUT STRATEGY

### 5.1 Sequential Override Rollout
Enable overrides **one at a time** in this order:
1. Enable override; validate in paper trading or shadow mode
2. Confirm 48-hour post-deploy review is STABLE
3. Only then enable the next override
**Never enable multiple new overrides simultaneously.**

### 5.2 Rollout Tiers (Current Cycle)
| Tier | Status | Allowed Actions |
|------|--------|----------------|
| Tier 1 | Implement now | Real-time impact monitor, dynamic vol-scaled budgets, order-book imbalance in Fast Loop, tightened latency discipline |
| Tier 2 | After 3 months paper trading | Shadow A/B enforcement, basic crisis-weighted voting (60–70 % cap) |
| Tier 3 | DEFERRED — current cycle | FPGA/hardware acceleration, market-making module, online RL micro-updates, full pod model |

**Tier 3 items are BLOCKED until Tier 1 and Tier 2 are complete and stable.**

---

## 6. TRAINING PIPELINES

### 6.1 Nightly Incremental Updates
- TTM or equivalent lightweight adapter
- Must complete within configured nightly window
- Failure alert to owner; fallback: retain prior day's model

### 6.2 PEARL / Meta-Learning
- Adapts to most recent 1–3 month regime
- Training stability tests via automated scripts (e.g., `test_retraining.py`)
- Distributed training: parallelised jobs with cost and reproducibility controls

### 6.3 Sentiment Re-Fine-Tuning
- Weekly cadence
- Validated against SEntFiN or equivalent holdout after every run

### 6.4 Synthetic Data Policy
- All synthetic data must be labelled as `synthetic` in every artifact
- Synthetic data **excluded from performance claims** unless validated on real data
- Scope: permissible for stress library augmentation and text augmentation; not for core backtest results

---

## 7. RESEARCH THROUGHPUT KPIs

Track and report monthly:
| KPI | Definition | Target |
|-----|-----------|--------|
| Research-to-production lead time | Idea → backtest → paper → shadow → live (end-to-end days) | Review and trend |
| Experiments per week | Count of completed offline experiments | Review and trend |
| Compute cost per approved update | $ or ₹ per promoted strategy change | Budget-guarded |
| Experiment failure taxonomy | Data-integrity failures vs model-logic failures | Corrective-action ownership per type |

---

## 8. VALIDATION & EVOLUTION LOOP (Phase 5)

### 8.1 Backtesting Standards
- Framework: VectorBT or equivalent with realistic simulation (slippage, commissions, partial fills)
- Period: 2018/2019 to present; all mandatory crisis windows included
- Walk-forward evaluation and time-based CV required; no random shuffling
- Survivorship bias controls: point-in-time universe data including delisted symbols

### 8.2 Dual-Loop Boundary Tests
- Validate: text floods alone cannot flip portfolio direction without confirming technical or risk conditions
- Test with: synthetic high-volume positive/negative news floods against flat market conditions

### 8.3 Research Loop
- Dedicated rapid-hypothesis loop for new z_t context variables and feature candidates
- Every hypothesis has explicit accept/reject criteria before being admitted to backtest pipeline

---

## MLOPS AGENT CHECKLIST
- [ ] Every model artifact has complete registry record (all required fields)
- [ ] Reproducibility test passed: pipeline reruns from raw data → same artifact hash
- [ ] No fixed uplift claims in any approval document
- [ ] Promotion checklist complete (non-regression on all metrics)
- [ ] Change request template filled and signed
- [ ] CI benchmark evidence checklist attached for execution-path changes
- [ ] 48-hour post-deploy review scheduled
- [ ] Rollout is sequential: no simultaneous override enablements
- [ ] Tier 3 items confirmed NOT enabled
- [ ] Research throughput KPIs reported this month
- [ ] Synthetic data labelled; excluded from core performance claims
