# Phase 3: Strategic Executive - One-Week Agent Build Plan

**Phase window**: Monday, March 30, 2026 to Sunday, April 5, 2026 (1 week)  
**Alignment**: Multi-Agent Plan v1.3.7 Section 7 (Phase 3), Section 16.1 (GO/NO-GO gates), and `docs/architecture/execution_loops.md`  
**Team**: Owner + Partner (2-person team)  
**Prerequisites**:  
- Phase 2 GO gate passed (Technical, Regime, Sentiment, Consensus outputs available).  
- Phase 3 observation input contract finalized and versioned.  
- Replay and backtest infrastructure ready for 2019-2025 evaluations.

---

## Phase 3 Goal

Build all Phase 3 Strategic Executive agents and runtime orchestration needed for production-style paper trading:
- Teacher policy agents (SAC, PPO, TD3)
- Ensemble decision agent (maximum-entropy action selection)
- Student execution agent (distilled low-latency policy for Fast Loop)
- Teacher monitoring and drift-control agent (Slow Loop governance)
- Strategic Executive orchestrator (dual-loop publish/consume runtime)

---

## Phase 3 GO Benchmarks (Section 16.1)

| Benchmark | Required Evidence |
|---|---|
| Fast Loop decision stack meets latency release gates (`p99 <= 8 ms` stretch; enforced degrade path above `10 ms`; `p99.9` jitter limits) | Latency benchmark report + degrade-path test logs |
| Teacher-student agreement threshold passes crisis slices | Crisis-slice agreement report |
| Rollback path is tested and deterministic | Rollback drill logs + release checklist |

---

## Agent Inventory (All Phase 3 Agents)

| Agent | Loop | Core Responsibility | Primary Deliverable |
|---|---|---|---|
| SAC Policy Agent (Teacher) | Slow/Offline | Policy learning under stochastic regimes | Trained SAC artifact + evaluation report |
| PPO Policy Agent (Teacher) | Slow/Offline | Stable policy optimization baseline | Trained PPO artifact + evaluation report |
| TD3 Policy Agent (Teacher) | Slow/Offline | Continuous-action robustness and variance control | Trained TD3 artifact + evaluation report |
| Ensemble Decision Agent | Slow/Offline + Runtime Read | Maximum-entropy action synthesis + offline threshold calibration | Ensemble decision module + calibration pack |
| Student Execution Agent | Fast Loop | Distilled deterministic low-latency inference | Compressed student artifact + latency profile |
| Teacher Monitoring Agent | Slow Loop | Agreement checks, drift detection, promotion/demotion logic | Drift monitor + governance triggers |
| Strategic Executive Orchestrator | Fast + Slow | Dual-loop handoff, snapshot routing, fail-safe action gating | Runtime orchestrator + integration tests |

---

## Week Plan (Day-by-Day)

### Day 1 (Mon): Contracts, Features, and Reward Library
- Freeze Phase 3 observation schema (versioned) from Phase 2 outputs:
  - Regime probabilities
  - VaR/ES
  - Macro differentials
  - Sentiment scores
  - Microstructure flags (order-book imbalance, queue pressure when healthy)
- Implement reward library:
  - RA-DRL composite
  - Sharpe, Sortino, Calmar, Omega, Kelly variants
  - Regime-aware dynamic weighting and downside penalties
- Create `src/agents/strategic_executive/` package skeleton and shared interfaces.
- Output: Observation schema + reward module + package skeleton.

### Day 2 (Tue): Build Teacher Policy Agents (SAC, PPO, TD3)
- Implement policy wrappers and training harnesses for SAC, PPO, TD3.
- Define each policy's training cadence, hyperparameter ranges, and evaluation metrics.
- Add reproducible training configs and artifact metadata tracking.
- Output: 3 runnable teacher policy agents with training entrypoints.

### Day 3 (Wed): Train and Validate Teacher Agents
- Train SAC, PPO, TD3 on historical windows (2019-2024) with walk-forward splits.
- Evaluate on holdout periods (2024-2025), including crisis slices.
- Produce ablations and per-policy strengths/weaknesses.
- Output: Trained teacher artifacts + baseline evaluation bundle.

### Day 4 (Thu): Build Ensemble Decision Agent
- Implement maximum-entropy ensemble action policy.
- Implement multi-threshold genetic search for offline calibration only.
- Freeze calibrated threshold set for runtime read-only use.
- Add deterministic decision traces for auditability.
- Output: Ensemble decision engine + calibration and traceability artifacts.

### Day 5 (Fri): Build Student Execution + Teacher Monitoring Agents
- Distill teacher ensemble into a student policy optimized for Fast Loop.
- Apply compression plan (distillation/quantization/pruning as applicable) and profile runtime determinism.
- Implement teacher-student agreement checks, crisis-slice promotion gates, and drift-triggered auto-demotion.
- Output: Student execution artifact + teacher-monitoring governance module.

### Day 6 (Sat): Build Strategic Executive Orchestrator (Dual Loop)
- Implement Fast Loop runtime path:
  - Student-only inference
  - O(1) snapshot reads
  - No heavy inference, no blocking calls
- Implement Slow Loop runtime path:
  - Teacher monitoring
  - Periodic policy snapshot refresh
  - Asynchronous analytics and checks
- Implement fail-safe action gates:
  - Stale/expired/quality-fail snapshot -> reduce-only or close-only
  - Latency cap breach safeguards
  - Risk-control-based trade rejection
- Output: End-to-end Strategic Executive orchestration pipeline.

### Day 7 (Sun): Hardening, Gate Review, and Handoff
- Run full Phase 2 -> Phase 3 integration replay and paper-mode dry run.
- Execute Phase 3 GO benchmark checks:
  - Fast Loop latency + jitter gates
  - Degrade-path behavior above latency cap
  - Crisis-slice teacher-student agreement
  - Rollback and demotion drills
- Finalize Phase 3 checkpoint package (model cards, metrics, known limitations, runbook deltas).
- Output: Phase 3 handoff package and GO/NO-GO assessment.

---

## Testing Matrix (Run Daily)

| Test Track | Frequency | Purpose |
|---|---|---|
| Unit tests (reward, schema, policy interfaces) | Daily | Catch logic and contract regressions early |
| Training smoke tests | Daily (before full runs) | Validate harness integrity |
| Walk-forward validation | Days 3-5 | Verify generalization and regime robustness |
| Crisis-slice agreement tests | Days 5-7 | Validate promotion safety for student policy |
| Latency benchmarks (`p50/p95/p99/p99.9`) | Days 5-7 | Enforce Fast Loop execution gates |
| Degrade and rollback drills | Days 6-7 | Validate deterministic fail-safe behavior |
| Integration replay (Phase 2 -> Phase 3) | Days 6-7 | Confirm full decision-path correctness |

---

## Exit Criteria for This Week

- All seven Phase 3 agents/components are implemented and integrated.
- Teacher artifacts (SAC, PPO, TD3) are trained, evaluated, and documented.
- Student execution artifact passes latency and agreement promotion gates.
- Dual-loop orchestrator passes non-blocking and fail-safe tests.
- Phase 3 GO/NO-GO evidence is complete for Section 16.1 evaluation.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| RL training runtime exceeds day budget | Use smoke-train loops daytime, full runs overnight |
| Student fails latency envelope | Tighten model compression and prune non-critical features |
| Teacher-student disagreement in crisis windows | Keep student unpromoted; expand crisis training slices and recalibrate thresholds |
| Integration mismatch with Phase 2 payloads | Freeze schema version and add strict validation at ingestion boundary |
| Fast Loop regressions from feature expansion | Gate new features behind quality flags and benchmark CI checks |
