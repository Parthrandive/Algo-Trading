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

### Day 1: Standalone Overseer Service & Schema
- Define the independent Risk Overseer architecture (RPC/Pub-Sub to Execution Engine).
- Implement the 4 Core Operating Modes state machine: `Normal` → `Reduce-Only` → `Close-Only` → `Kill-Switch`.
- Enforce the "one-way down" state transition rule requiring explicit multi-condition recovery.

### Day 2: L1 & L2 Kill Switches (Model & Portfolio)
- **L1 Model Switch**: Implement automatic pause on student drift or teacher-student divergence breaches.
- **L2 Portfolio Switch**: Implement `Reduce-Only` triggers on max drawdown breaches or daily loss limits.

### Day 3: L3 & L4 Kill Switches (Broker & Manual)
- **L3 Broker Switch**: Implement `Close-Only` downgrade on API errors, margin calls, or high rejection rates.
- **L4 Manual Switch**: Build secure dashboard endpoint for immediate operator `Kill-Switch` triggering.

### Day 4-5: Integration & Execution Veto
- Bind the Execution Engine to the Risk Overseer veto (Execution must fail-closed if Risk Overseer is unreachable).
- Run full L1–L4 drills in the paper-trading harness with simulated breaches.

---

## Week 2: Crisis Taxonomy & Advanced Constraints
**Focus**: Managing complex multi-condition market crises and agent divergences.

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

---

## Week 3: Impossible Scenarios & Stress Test Framework
**Focus**: Formalising the mandatory stress scenario library into an automated testing suite.

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

---

## Week 4: Drift Detection & Go-Live Gates
**Focus**: Predictive anomaly detection and preparation for production cash trading.

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
