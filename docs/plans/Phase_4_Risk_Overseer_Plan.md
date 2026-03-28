# Phase 4: Independent Risk Overseer — Standalone Hardening Plan

**Phase window**: Monday, April 27, 2026 to Sunday, May 24, 2026 (4 weeks)
**Alignment**: Multi-Agent Plan v1.3.7, `trading-skills/risk-overseer/SKILL.md`
**Prerequisite**: Phase 3 Strategic Executive complete; Fast Loop execution and paper-trading harness operational.

---

## Phase 4 Goal
Extract and harden the **Risk Overseer** as a fully independent agent with **absolute veto authority** over all execution decisions. Implement the comprehensive Kill Switch Hierarchy, formalise the Crisis Taxonomy state machine, automate Impossible Scenario stress testing, and deploy live drift detection. Ensure the Risk Overseer operates asynchronously to, and outside of, the Fast Loop to maintain the ≤ 8ms latency budget.

---

## Week 1: Core Architecture & Kill Switch Hierarchy
**Focus**: Decoupling risk from execution; implementing L1–L4 kill switches and operating modes.

### Week 1 Completion Audit (Gap-Closed)

| Requirement (Risk Skill) | Week 1 Coverage | Gap Closure Added |
|---|---|---|
| Independent overseer with veto authority | Covered at architecture level | Added explicit fail-closed heartbeat and timeout gates |
| 4 operating modes + one-way-down transitions | Covered conceptually | Added formal transition matrix + recovery prerequisites |
| L1–L4 kill-switch hierarchy | Covered conceptually | Added concrete trigger fields and mandatory audit payloads |
| Kill-switch exit control | Partially implied | Added explicit operator acknowledgement requirement for exit |
| Mode-change telemetry | Missing detail | Added event contract (`MODE_CHANGE`, `KILL_SWITCH_TRIGGER`, `KILL_SWITCH_EXIT`) |
| Drill evidence and sign-off | Missing detail | Added drill matrix with pass criteria and MTTR logging |

### Day 1: Standalone Overseer Service & Event Contract
- Define independent Risk Overseer service boundary (asynchronous RPC/Pub-Sub; not on Fast Loop critical path).
- Implement operating mode state machine:
  - `Normal` → `Reduce-Only` → `Close-Only` → `Kill-Switch` (one-way down only).
  - Recovery allowed only when all triggering conditions are cleared.
  - `Kill-Switch` exit requires explicit operator acknowledgement.
- Implement mandatory event payload schema for all mode transitions:
  - `event_id`, `event_type`, `timestamp_utc`, `from_mode`, `to_mode`, `trigger_layer`, `trigger_reason`, `authorizer`, `snapshot_id`, `plan_version`.
- Add mandatory events:
  - `MODE_CHANGE`
  - `KILL_SWITCH_TRIGGER`
  - `KILL_SWITCH_EXIT`

### Day 2: L1 & L2 Kill Switches (Model & Portfolio)
- **L1 Model Switch** (`Reduce-Only` minimum):
  - Trigger on student drift breach, teacher-student agreement breach, or model anomaly flag.
  - Log threshold values at trigger time in the audit payload.
- **L2 Portfolio Switch** (`Reduce-Only` minimum):
  - Trigger on max drawdown breach, daily loss limit breach, or concentration breach.
  - Enforce no-new-opens while allowing reductions/closes.
- Add Week 1 configuration keys (must be versioned):
  - `l1_student_drift_threshold`
  - `l1_teacher_student_divergence_threshold`
  - `l2_max_drawdown_limit`
  - `l2_daily_loss_limit`
  - `l2_concentration_limit`

### Day 3: L3 & L4 Kill Switches (Broker & Manual)
- **L3 Broker Switch** (`Close-Only` minimum):
  - Trigger on broker API failures, margin-call signals, or rejection-rate threshold breach.
  - Include rejection-rate rolling window + threshold in config.
- **L4 Manual Switch** (`Kill-Switch` immediate):
  - Secure endpoint requiring authenticated operator identity.
  - Mandatory audit fields: `operator_id`, `reason`, `ticket_id`, `timestamp_utc`.
  - Manual trigger must cancel open orders and reject all new orders.

### Day 4-5: Integration, Veto, and Drills
- Bind Execution Engine to Risk Overseer veto with fail-closed semantics:
  - If Risk Overseer heartbeat is stale/unreachable, execution degrades to `Close-Only` or stricter.
  - No order submission when risk decision is unavailable.
- Run L1–L4 drill suite in paper-trading harness:
  - Each layer tested independently and in escalation chain.
  - Capture MTTR, false-positive flag, mode path, and recovery evidence.
- Week 1 drill pass criteria:
  - 100% trigger fidelity for injected breaches.
  - No upward mode transitions without explicit recovery checks.
  - `Kill-Switch` exit always includes operator acknowledgement record.

### Week 1 Exit Gate (Must Pass)
- [ ] State machine enforces one-way-down transitions.
- [ ] Recovery logic implemented and tested for each mode.
- [ ] L1–L4 trigger handlers implemented with versioned thresholds.
- [ ] Execution veto path fail-closes when overseer is unreachable.
- [ ] Mode-change and kill-switch events emit full audit payload.
- [ ] Drill report includes MTTR, pass/fail, and sign-off owner/reviewer.
- [ ] All Week 1 controls validated in paper-trading harness before Week 2 start.

---

## Week 2: Crisis Taxonomy & Advanced Constraints
**Focus**: Managing complex multi-condition market crises and agent divergences.

### Week 2 Completion Audit (Gap-Closed)

| Requirement (Risk Skill) | Week 2 Coverage | Gap Closure Added |
|---|---|---|
| Full Crisis entry (3 conditions + hysteresis) | Partially specified | Added deterministic confirmation ticks before `FULL_CRISIS` transition |
| Full Crisis revalidation timer | Missing | Added max-duration revalidation window with controlled one-step auto-revert when crisis conditions are not revalidated |
| Crisis-weighted routing cap (`<= 70%`) | Mentioned, not wired | Added `crisis_weight_cap` to overseer decision metadata/audit payload for downstream routing enforcement |
| Agent divergence neutral-hold + staged re-risk | High-level only | Added staged fractions (`25% -> 50% -> 75% -> 100%`) emitted in risk metadata with explicit completion signal |
| Slow-crash protective de-risking | Missing concrete control surface | Added `SLOW_CRASH` protective budget fraction control (non-blanket kill-switch) |
| Negative sentiment + mismatch handling | Missing explicit outputs | Added `SENTIMENT` protective budget caps for both extreme negative z-score and mismatch regime |
| OOD staged de-risking without immediate neutralization | Incomplete | Added staged OOD caps (stage 1/2/3) in `NORMAL` mode and hard-limit bypass to immediate kill-switch |
| Execution integration of staged caps | Missing | Added buy-side quantity scaling by overseer `risk_budget_fraction` with full audit event |
| Test coverage for Week 2 controls | Missing | Added dedicated Week 2 unit tests for overseer transitions and execution wiring |

### Day 1: Crisis Entry Protocol
- Implement the 3-condition Full Crisis entry (vol break + liquidity deterioration + confidence drop).
- Configure hysteresis (time-bound confirmation to avoid whipsawing).
- Implement crisis-weighted routing (cap crisis agent weight to ≤ 70%).

### Day 2: Agent Divergence & Neutral-Hold
- Implement Divergence Protocol: freeze new opens if ≥ 2 major agents disagree fundamentally.
- Build the staged re-risking pipeline (25% → 50% → 75% → 100% budget scaling per alignment signal).

### Day 3: Slow-Crash / Freeze Protocol
- Differentiate between a flash shock and a slow drawdown accumulation.
- Implement staged de-risking before limits are blown.

### Day 4-5: Negative Sentiment & OOD Handlers
- Integrate FinBERT sentiment: trigger protective caps on extreme negative spikes (`z_t` < threshold).
- Implement sentiment-price mismatch downgrade logic (e.g., positive news but price dumping).
- Hard-wire OOD (Out of Distribution) staged de-risking without immediate neutralization.

### Week 2 Exit Gate (Must Pass)
- [ ] Full Crisis requires all 3 conditions and hysteresis ticks before mode escalation.
- [ ] Full Crisis max-duration timer auto-reverts one step if revalidation fails.
- [ ] `crisis_weight_cap` is emitted and audited on every `FULL_CRISIS` decision.
- [ ] Divergence protocol emits staged re-risk fractions (`25/50/75/100`) in order.
- [ ] Slow-crash and sentiment handlers apply protective budget caps without forced immediate kill-switching.
- [ ] OOD staged de-risking runs in tiers; OOD + hard-limit breach bypasses staging to kill-switch.
- [ ] Execution engine applies overseer `risk_budget_fraction` deterministically and records scaling in audit trail.
- [ ] Week 2 overseer + execution tests pass in CI/paper-trading harness.

---

## Week 3: Impossible Scenarios & Stress Test Framework
**Focus**: Formalising the mandatory stress scenario library into an automated testing suite.

### Week 3 Completion Audit (Gap-Closed)

| Requirement (Risk Skill) | Week 3 Coverage | Gap Closure Added |
|---|---|---|
| Mandatory stress scenario library versioned + typed | Not formalized | Added canonical library entries with `historical` / `synthetic` / `impossible` labels and explicit IDs |
| Historical + synthetic replay injectors | Partially implied | Added deterministic scenario replay expansion API for nightly CI wiring |
| Capacity stress at 1x/2x/3x | Missing automation primitive | Added automatic capacity matrix expansion helper for every scenario |
| 3x impact breach policy | Partially checked | Added explicit hard-cap forcing signal (`2x`) when `3x` impact exceeds `0.25%` |
| Impossible scenario guardrails | Incomplete | Added evaluator checks for correlation-variance validity, safe-mode on feed-integrity uncertainty, and zero-vs-missing distinction |
| Snapback stress measurement | Partially checked | Hardened snapback failure signaling as a first-class review action trigger |
| Quarterly stress review artifact | Missing | Added automated quarterly review template with owner/reviewer sign-off fields and required action list |
| Week 3 test coverage | Missing | Added dedicated Week 3 stress framework unit tests |

### Day 1: Historical & Synthetic Injectors
- Build replay nodes for RBI rate hikes, INR flash moves, COVID crashes, and the 2008 GFC.
- Integrate these into a nightly CI stress testing pipeline.

### Day 2: Capacity Stress Tests
- Implement automated replay at 1×, 2×, and 3× current notional capacity.
- Programmatic assertion: fail CI if impact exceeds 0.25% at 3× capacity, auto-forcing a 2× hard cap.

### Day 3: Impossible Scenario Drills
- Program synthetic correlation inversion data feeds.
- Program frozen constituent prices to test zero-volume vs. missing data distinction.
- Program simultaneous multi-asset liquidity vacuums.

### Day 4-5: Snapback testing & Review
- Build flash-shock snapback tests measuring ticks required to clip net exposure.
- Generate the automated Quarterly Stress Review template.

### Week 3 Exit Gate (Must Pass)
- [ ] Versioned scenario library includes all required historical/synthetic/impossible scenarios.
- [ ] Nightly CI stress set expands all scenarios over `1x/2x/3x` capacity multipliers.
- [ ] `3x` impact breaches over `0.25%` force a `2x` hard-cap recommendation and fail the run.
- [ ] Correlation inversion fails if variance coherence checks fail.
- [ ] Frozen-price drills fail when zero-volume vs missing-data classification is ambiguous.
- [ ] Feed-integrity uncertainty drills fail if safe mode is not engaged.
- [ ] Snapback breaches are surfaced as explicit review actions in quarterly stress reports.
- [ ] Week 3 stress framework tests pass in CI.

---

## Week 4: Drift Detection & Go-Live Gates
**Focus**: Predictive anomaly detection and preparation for production cash trading.

### Week 4 Completion Audit (Gap-Closed)

| Requirement (Risk Skill) | Week 4 Coverage | Gap Closure Added |
|---|---|---|
| Continuous ADDM-style drift monitoring | Missing | Added continuous drift engine over Phase 2 input drift + Phase 3 output drift |
| Drift-linked position sizing from provenance reliability | Missing | Added dynamic size multiplier based on reliability + drift pressure (not fixed haircut) |
| Sustained drift demotion escalation | Missing | Added sustained-window drift alerts with automatic demotion trigger signaling |
| 30-day false-trigger governance | Partially in Tier1-B engine | Added explicit operator-review escalation report when false-trigger breach pauses auto-adjustment |
| Full-stack SHAP coverage gate (`>=80%`) | Missing Week 4 gate binding | Added formal Week 4 risk-gate reviewer check on XAI coverage |
| PnL attribution + risk override visibility | Missing Week 4 gate binding | Added gate check requiring active attribution dashboard and visible live Risk Overseer override events |
| Final surprise L4 drill check | Missing | Added deterministic surprise L4 drill runner with pass/fail audit contract |
| Week 4 integration test coverage | Missing | Added dedicated Week 4 drift/go-live gate unit tests |

### Day 1-2: Automated Drift Detection (ADDM)
- Deploy ADDM pipeline to continuously monitor Phase 2 input distributions and Phase 3 policy outputs.
- Tie provenance reliability scores dynamically to position sizing (no fixed haircuts).
- Implement sustained drift alerts escalating to automatic model demotion.

### Day 3: False-Trigger Rate Tracking
- Harden the Tier 1-B Volatility Budgets with a rolling 30-day false-trigger tracker.
- If the trigger rate breaches limits, automatically escalate to operator review and pause down-scaling.

### Day 4-5: Full-Stack Risk Gate Review
- Run the final Go-Live phase audit.
- Confirm SHAP coverage ≥ 80% on all paper trades.
- Confirm PnL Attribution dashboards reflect live Risk Overseer overrides.
- Execute a final, surprise L4 Kill-Switch drill during live market hours in the paper trading environment.

### Week 4 Exit Gate (Must Pass)
- [ ] ADDM drift decisions are emitted continuously for Phase 2 and Phase 3 drift channels.
- [ ] Position sizing reacts dynamically to provenance reliability and current drift pressure.
- [ ] Sustained drift threshold breaches trigger demotion escalation signals.
- [ ] False-trigger rate breach triggers operator-review escalation and confirms down-scaling pause.
- [ ] SHAP/XAI coverage gate passes at `>= 80%`.
- [ ] Attribution dashboard is active and includes live Risk Overseer override visibility.
- [ ] Final surprise L4 kill-switch drill passes with recorded trigger event and cancel-orders behavior.
- [ ] Week 4 drift and go-live gate tests pass in CI.
