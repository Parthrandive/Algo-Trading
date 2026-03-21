# Phase 3: Strategic Executive — Weekly Execution Plan

**Phase window**: Monday, March 30, 2026 to Sunday, April 27, 2026 (4 weeks)
**Alignment**: Multi-Agent Plan v1.3.7 §7 (Phase 3: Strategic Executive), §8 (Portfolio Construction), §10 (Execution & Market Impact)
**Team**: Owner + Partner (2-person team)
**Prerequisite**: Phase 2 Analyst Board GO gate passed; all four Phase 2 agents producing validated signals.

---

## Phase 3 Goal

Build the RL-based Trader Ensemble (SAC, PPO, TD3) that consumes Phase 2 Analyst Board signals and produces actionable trade execution decisions. Implement the full observation-to-execution pipeline including portfolio construction, risk budgeting, teacher-student distillation, order execution, and Tier 1 enhancements. All components must meet latency, compliance, and paper-trading readiness targets by end of phase.

---

## Phase 3 GO Benchmarks (from §16.1)

| Benchmark | Required Evidence |
|---|---|
| Fast Loop decision stack meets latency release gates (stretch p99 ≤ 8ms, enforced degrade path above 10ms, p99.9 jitter limits) | Latency benchmark report with p50/p95/p99/p99.9 |
| Teacher-student agreement threshold passes on crisis slices | Crisis-slice agreement report (avg-day + crisis) |
| Rollback path is tested | Rollback drill log with MTTR measurement |
| Walk-forward evaluation on holdout data | Backtest report: Sharpe, Sortino, MDD, win rate |
| Observation space schema versioned and validated | Schema doc + validation test results |

---

## Dependencies from Phase 2

| Input | Source Table | Used By |
|---|---|---|
| Technical predictions (price forecast, direction, VaR/ES) | `technical_predictions` | Observation Assembler |
| Regime state & transition probabilities | `regime_predictions` | Observation Assembler |
| Sentiment scores & z_t threshold variable | `sentiment_scores` | Observation Assembler |
| Consensus signals (final direction, confidence, weights) | `consensus_signals` | Observation Assembler |
| Model cards (model IDs, versions) | `model_cards` | Provenance tracking |

---

## Week 1 (March 30 – April 5): RL Foundation & Observation Space

**Owner**: You | **Focus**: Observation pipeline, reward library, policy skeletons

### Day 1 (Mon): Observation Space & Package Skeleton

**Tasks:**
- Create `src/agents/strategic/` package skeleton with `__init__.py`, `config.py`, `schemas.py`
- Define the Phase 3 observation space schema (§7.3) consuming all Phase 2 outputs:
  - Technical: `price_forecast`, `direction`, `confidence`, `var_95`, `es_95`
  - Regime: `regime_state`, `transition_probability`, `risk_level`
  - Sentiment: `sentiment_score`, `z_t`, `sentiment_class`
  - Consensus: `final_direction`, `final_confidence`, `crisis_mode`, `agent_divergence`
  - Microstructure: `orderbook_imbalance`, `queue_pressure` (when feed quality is healthy)
  - Portfolio state: `current_position`, `unrealized_pnl`, `notional_exposure`
- Build Observation Assembler that reads from Phase 2 tables and materializes `observations` table (see Database Plan §3.3)
- Define observation schema versioning system
- Version and freeze the observation space schema for downstream consumption

**Tests & Validation:**
- Unit test: observation assembler produces valid observation vectors from mock Phase 2 data
- Schema validation: observation payload matches versioned schema
- Leakage test: no future data in observation timestamps
- Integration test: observations table writes correctly to PostgreSQL

**Output:** Package skeleton + Observation Assembler + versioned observation schema

---

### Day 2 (Tue): Reward Library & Training Environment

**Tasks:**
- Implement reward function library (§7.4) with all variants:
  - **RA-DRL composite**: risk-adjusted return with regime-adaptive weighting
  - **Sharpe ratio**: annualized risk-adjusted returns
  - **Sortino ratio**: downside-deviation adjusted returns
  - **Calmar ratio**: return/max-drawdown ratio
  - **Omega ratio**: probability-weighted gains vs losses
  - **Kelly criterion**: optimal bet sizing variant
- Implement dynamic reward weighting tied to regime state and sentiment signals
- Add increased downside penalties during RBI outer band or negative sentiment spikes
- Build RL training environment (Gymnasium-compatible) that wraps historical data with:
  - Observation space from Day 1
  - Action space: `buy`, `sell`, `hold`, `close`, `reduce` with continuous sizing
  - Reward computation using the reward library
  - Transaction cost modeling (brokerage + slippage + impact)
  - Episode reset and walk-forward data iteration
- Create `src/agents/strategic/reward.py` and `src/agents/strategic/environment.py`
- Write reward computation to `reward_logs` table (Database Plan §3.8)

**Tests & Validation:**
- Unit tests for each reward function against hand-computed examples
- Verify dynamic weighting: reward weights shift correctly when regime changes
- Environment test: step(), reset(), observation_space, action_space all valid
- Verify transaction costs are non-zero and realistic vs NSE brokerage schedules
- Edge case: reward under zero-volume and missing-data scenarios

**Output:** Reward library + Gymnasium-compatible training environment

---

### Day 3 (Wed): SAC Policy Implementation

**Tasks:**
- Implement SAC (Soft Actor-Critic) policy using Stable-Baselines3 (or custom):
  - Entropy-regularized actor-critic with automatic temperature tuning
  - Network architecture: observation → shared encoder → actor head + critic head
  - Hyperparameters: documented and configurable via config file
- Set up training harness with:
  - Walk-forward data splits (2019–2023 train, 2023–2025 holdout)
  - Checkpoint saving with model registry metadata
  - Training metrics logging (episode reward, loss curves, entropy)
- Train SAC on historical data for initial baseline
- Register trained policy in `rl_policies` table (Database Plan §3.1)
- Log training run to `rl_training_runs` table (Database Plan §3.2)

**Tests & Validation:**
- Verify SAC produces valid actions from observation space
- Training convergence: reward curve shows improvement over random baseline
- Walk-forward holdout: Sharpe and MDD on holdout set
- Model serialization: save → load → identical predictions
- Latency benchmark: raw SAC inference time on single observation

**Output:** Trained SAC baseline policy + model card

---

### Day 4 (Thu): PPO Policy Implementation

**Tasks:**
- Implement PPO (Proximal Policy Optimization) policy:
  - Clipped surrogate objective with value function baseline
  - GAE (Generalized Advantage Estimation) for advantage computation
  - Can share the same environment wrapper from Day 2
- Configure PPO-specific hyperparameters (clip ratio, GAE lambda, number of epochs)
- Train PPO on the same data splits as SAC for comparable evaluation
- Register in `rl_policies` table

**Tests & Validation:**
- PPO produces valid actions
- Training convergence comparison vs SAC baseline
- Walk-forward holdout metrics
- Ablation: PPO vs SAC on crisis windows specifically

**Output:** Trained PPO baseline policy + model card

---

### Day 5 (Fri): TD3 Policy & Ensemble Skeleton

**Tasks:**
- Implement TD3 (Twin Delayed DDPG) policy:
  - Twin Q-networks with delayed policy updates
  - Target policy smoothing (noise injection)
  - Configure delay update frequency and noise parameters
- Train TD3 on same data splits
- Build initial ensemble skeleton combining all three policies:
  - Equal-weight averaging as starting baseline
  - Placeholder for maximum entropy framework (Week 2)
  - Individual policy contribution tracking
- Create ensemble output writer to `trade_decisions` table (Database Plan §3.4)

**Tests & Validation:**
- TD3 produces valid actions consistently
- Training convergence comparison vs SAC and PPO
- Ensemble baseline: equal-weight vs individual policy performance on holdout
- All three policies registered in model registry

**Output:** Trained TD3 policy + ensemble skeleton + baseline comparison report

---

### Day 6 (Sat): Model Cards, Baselines & Ablation

**Tasks:**
- Complete model cards for all three policies (SAC, PPO, TD3):
  - Hyperparameters, data requirements, known limitations
  - Training metrics summary, holdout performance
- Run comprehensive ablation analysis:
  - Each observation feature group removed (technical, regime, sentiment, consensus, microstructure)
  - Measure impact on Sharpe, MDD, and win rate per policy
- Run comparison against naive baselines:
  - Buy-and-hold
  - Random action
  - Consensus-signal-only (no RL)
- Produce Week 1 training report

**Tests & Validation:**
- Ablation results documented per policy
- All three policies outperform random baseline on risk-adjusted metrics
- Model cards complete with all required fields

**Output:** Model cards + ablation report + baseline comparison

---

### Day 7 (Sun): Week 1 Review & Integration Gate

- Review all Week 1 deliverables
- Fix defects from test results
- Verify all three policies can run end-to-end: Phase 2 data → observation → policy → trade decision
- Freeze observation space schema for Week 2 consumption
- Prepare Week 2 handoff: ensemble integration, distillation, portfolio

---

## Week 2 (April 7 – April 13): Ensemble Decision & Distillation

**Owner**: Partner | **Focus**: Ensemble decision mechanism, teacher-student distillation, policy snapshots

### Day 1 (Mon): Maximum Entropy Ensemble Framework

**Tasks:**
- Implement maximum entropy decision framework (§7.2):
  - Entropy-weighted policy combination
  - Each policy's contribution weighted by confidence and diversity
  - Temperature parameter controlling exploration vs exploitation
- Implement multi-threshold genetic algorithm for action selection (§7.2):
  - Genetic search for optimal action thresholds across signals
  - Runs **offline only** — never in live execution path
  - Output: threshold table used by live decision engine
- Create `src/agents/strategic/ensemble.py`
- Log ensemble decision to `trade_decisions` table with per-policy weights (`sac_weight`, `ppo_weight`, `td3_weight`)

**Tests & Validation:**
- Entropy framework produces valid combined actions
- GA threshold search converges on synthetic data
- Ensemble outperforms best individual policy on risk-adjusted metrics
- Policy weights sum to 1.0 and are non-negative

**Output:** Max-entropy ensemble engine + GA threshold optimizer

---

### Day 2 (Tue): Teacher-Student Distillation

**Tasks:**
- Implement teacher-student distillation pipeline (§7.2):
  - **Teacher**: full ensemble (SAC + PPO + TD3 with max-entropy)
  - **Student**: lightweight single-network model for Fast Loop execution
  - KL-divergence loss for student to match teacher action distributions
  - Student architecture: small MLP or quantized version of teacher
- Train student on teacher's decision outputs across diverse market conditions
- Measure teacher-student agreement on:
  - **Average-day slices** (normal market conditions)
  - **Crisis slices** — critical gate (§7.2): 2008, 2013, COVID, RBI policy events
- Implement automatic student demotion when drift exceeds threshold
- Register student in `student_policies` table (Database Plan §3.10)
- Log distillation run to `distillation_runs` table (Database Plan §3.11)

**Tests & Validation:**
- Student produces valid actions within Fast Loop latency budget (p99 ≤ 8ms)
- Teacher-student agreement ≥ defined threshold on average-day slices
- **Crisis-slice agreement meets hard promotion gate** (must pass before promotion)
- Student drift monitoring: synthetic drift injection triggers auto-demotion
- Student inference latency benchmark: p99 and p99.9

**Output:** Distilled student policy + agreement report + latency benchmark

---

### Day 3 (Wed): Policy Snapshots & Cross-Loop Protocol

**Tasks:**
- Implement policy snapshot system aligned with `execution_loops.md` cross-loop protocol:
  - Atomic snapshot updates (pointer swap — old consistent state OR new consistent state)
  - Each snapshot carries: `snapshot_id`, `generated_at`, `expires_at`, `schema_version`, `quality_status`, `source_type`
  - Scheduled refresh cadence (every ~10 minutes) or earlier on async completion
- Write to `policy_snapshots` table (Database Plan §3.9)
- Implement Fast Loop policy consumer:
  - In-memory cache of active policy snapshot
  - O(1) lookup — no DB queries on critical path
  - Stale/expired snapshot → automatic degradation to reduce-only mode
- Implement snapshot refresh triggered by:
  - Scheduled timer (10-min cadence)
  - Slow Loop deliberation output
  - Emergency swap (model demotion, risk event)
- Create `src/agents/strategic/policy_manager.py`

**Tests & Validation:**
- Atomic swap test: Fast Loop never sees partial state
- Stale snapshot triggers reduce-only mode correctly
- Snapshot TTL expiry triggers expected behavior
- Emergency swap: student demotion → fallback to previous snapshot
- Policy refresh latency: measured end-to-end

**Output:** Policy snapshot system + Fast Loop consumer + cross-loop protocol

---

### Day 4 (Thu): Deliberation Engine (Slow Loop)

**Tasks:**
- Implement Slow Loop deliberation process (§7.5):
  - Runs asynchronously, never blocks Fast Loop
  - Heavy reasoning: policy search, simulation, reward reweighting
  - Outputs new policy snapshots when better policy is found
- Implement deliberation bypass under high-volatility triggers
- Trade rejection logic tied to:
  - Risk-control breaches
  - Stale-data TTL breaches
  - Latency-cap breaches (not pending simulation)
- Log all deliberation outputs to `deliberation_logs` table (Database Plan §3.12)
- Create `src/agents/strategic/deliberation.py`

**Tests & Validation:**
- Slow Loop deliberation runs independently of Fast Loop
- Deliberation bypass activates when volatility triggers fire
- Trade rejection: confirm all three rejection paths work
- Deliberation latency: typical 100–500ms, never blocks Fast Loop
- Output snapshot is valid and consumable by policy manager

**Output:** Deliberation engine + bypass logic + rejection handlers

---

### Day 5 (Fri): Portfolio Construction & Risk Budgets

**Tasks:**
- Implement portfolio construction engine (§8):
  - Risk budgets per asset class and sector (defined, enforced)
  - Correlation and concentration limits enforced before order generation
  - Position sizing tied to forecast uncertainty and liquidity
  - Inventory and exposure state model: gross, net, hedge-adjusted
- Implement hedging policy engine:
  - Cross-asset correlation checks
  - Explicit risk-unwind logic
  - Hedging as first-class action type (partial unwind, cross-asset hedge, emergency neutralization)
- Capacity ceiling enforcement:
  - Initial: ₹5–10 Cr notional
  - Scale to ₹25 Cr after 6 months live validation
  - Stress-tested at 1x, 2x, 3x capacity
  - Hard cap at 2x if impact exceeds 0.25% at 3x
- Max participation limits documented and enforced
- Write portfolio state to `portfolio_snapshots` table (Database Plan §3.7)
- Create `src/agents/strategic/portfolio.py`

**Tests & Validation:**
- Risk budget enforcement: positions exceeding budget are rejected
- Correlation limit: highly correlated additions are blocked
- Position sizing: verify sizing scales with uncertainty
- Capacity ceiling: orders exceeding capacity are rejected
- Hedge trigger: exposure breach → automatic hedge action
- Downside risk emphasis test: verify during panic regime

**Output:** Portfolio construction engine + risk budget system

---

### Day 6 (Sat): Order Execution Engine

**Tasks:**
- Implement order execution engine (§10):
  - Supports partial fills, pacing, participation limits
  - Order types: market, limit, stop-loss, SL-market
  - Circuit breaker and halt handling
- Implement slippage model:
  - Validated against intraday data
  - Linked to capacity assumptions
  - Market impact model defined
- Implement pre-trade compliance checks (§3):
  - Position limits, margin rules, circuit breaker rules
  - Audit trail: order intent, execution, cancellation, model version
- Implement real-time routing health checks:
  - Auto-degradation to reduce-only/close-only on infrastructure failure
- Write orders to `orders` and `order_fills` tables (Database Plan §3.5, §3.6)
- Create `src/agents/strategic/execution.py`

**Tests & Validation:**
- Partial fill handling: verify aggregation and avg fill price
- Circuit breaker test: orders rejected during halts
- Compliance check: non-compliant orders rejected with reason logged
- Health check: infrastructure failure → automatic mode downgrade
- Slippage measurement: compare model estimate vs realized
- Audit trail completeness check

**Output:** Order execution engine + compliance checks + slippage model

---

### Day 7 (Sun): Week 2 Review & Integration Gate

- Full pipeline test: Phase 2 signals → observation → ensemble → portfolio → order
- Review Week 2 test results
- Verify cross-loop protocol: Fast Loop + Slow Loop running independently
- Verify all database tables written correctly
- Prepare Week 3 handoff: Tier 1 enhancements, latency tuning

---

## Week 3 (April 14 – April 20): Tier 1 Enhancements & Latency Discipline

**Owner**: You | **Focus**: Tier 1 production enhancements per §18.1

### Day 1 (Mon): Real-Time Impact & Slippage Monitor (Tier 1-A)

**Tasks:**
- Define impact thresholds by instrument bucket:
  - Liquid index names (NIFTY 50 constituents)
  - Mid-liquidity names (broader NSE universe)
  - Forex (USDINR)
- Build live metrics pipeline:
  - Realized slippage (vs model estimate)
  - Participation rate (% of ADV)
  - ADV-linked impact score
- Implement automatic position-sizing reducer:
  - Triggers on threshold breach
  - Cooldown and hysteresis to prevent flapping
  - Logged to `trade_decisions.risk_override`
- Dashboard panels for threshold breaches and auto-reduction events

**Tests & Validation:**
- Replay drill: threshold breach triggers expected size reduction
- Cooldown test: no rapid re-triggering within cooldown window
- No regressions in order intent, fill tracking, risk logs
- Post-deploy review template captures incident counts

**Output:** Impact monitor + auto-reducer + dashboard

---

### Day 2 (Tue): Dynamic Volatility-Scaled Risk Budgets (Tier 1-B)

**Tasks:**
- Define volatility regime calculation and sigma thresholds per asset cluster
- Map each regime to exposure caps:
  - **Normal**: full risk budget
  - **Elevated** (1σ breach): reduced cap (e.g. 70% of normal)
  - **Protective** (2σ breach): minimum cap (e.g. 30% of normal)
- Implement automatic cap adjustment in risk overseer BEFORE hard kill-switch triggers
- Risk state telemetry and alerts for regime transitions and cap changes
- Integrate with portfolio construction engine from Week 2 Day 5

**Tests & Validation:**
- Stress drill: exposure caps scale down before kill-switch events
- Cap transitions are deterministic and auditable
- False-trigger rate within rollout acceptance limits
- Edge case: rapid volatility whipsaw doesn't cause flapping (hysteresis)

**Output:** Volatility-scaled risk budget engine

---

### Day 3 (Wed): Order-Book Imbalance in Fast Loop (Tier 1-C)

**Tasks:**
- Define canonical imbalance and queue-pressure features with schema versioning
- Publish features via non-blocking snapshot path for Fast Loop consumption
- Feature quality guardrails:
  - Staleness detection
  - Missing book levels handling
  - Source-quality flags
- Safe degradation: feature quality failures don't block execution
- Shadow comparison vs baseline (without order-book features)

**Tests & Validation:**
- Fast Loop p99 and p99.9 latency remain within release gates after feature enablement
- Feature quality failures degrade safely (reduced feature set, not execution block)
- Shadow comparison shows non-regression on slippage and risk controls
- Schema versioning: old and new feature schemas coexist during rollout

**Output:** Order-book imbalance feature pipeline for Fast Loop

---

### Day 4 (Thu): Fast Loop Latency Discipline (Tier 1-D)

**Tasks:**
- Instrument full decision path timing: p50, p95, p99, p99.9, and jitter
- Implement CI benchmark gate for execution-path pull requests:
  - Automated replay simulation under peak load
  - Benchmark artifacts generated per qualifying change
- Validate degrade safeguard behavior when latency breaches > 10ms:
  - Auto-trip to degrade mode
  - Skip non-essential features while maintaining risk controls
  - Alert and log the breach
- Weekly latency regression report template with owner and remediation

**Tests & Validation:**
- CI produces benchmark artifacts for replay and peak-load on every qualifying change
- Degrade-path tests pass consistently under synthetic latency stress
- Latency trend tracked and stable
- End-to-end decision path profiling: identify hotspots

**Output:** Latency benchmark CI gate + degrade safeguard + regression report template

---

### Day 5 (Fri): Safe RL Training & Deployment Gates

**Tasks:**
- Implement safe RL training protocol (§7.6):
  - Online learning gated behind offline training + paper-trading validation
  - Exploration constrained by risk limits and hard safety filters
  - No live trading without risk committee approval + rollback readiness
- Implement promotion pipeline:
  - `candidate` → `shadow` → `champion` status transitions
  - Champion-challenger promotion requires:
    - Walk-forward holdout outperformance
    - No regression in Sharpe, MDD, slippage
    - Crisis-slice agreement (for students)
    - Latency gate compliance
  - Rollback: tested revert to previous champion
- Create promotion gate automation in CI/CD

**Tests & Validation:**
- Promotion gate: candidate without required evidence cannot promote
- Rollback drill: revert to previous model, verify system stability
- Exploration constraint: RL actions clamped to risk limits during training
- Rollback MTTR measurement

**Output:** Safe RL deployment pipeline + promotion gates + rollback drill

---

### Day 6 (Sat): XAI Logging & PnL Attribution

**Tasks:**
- Implement SHAP-based XAI logging (§11):
  - Top-k feature explanations for ≥ 80% of trades
  - Per-agent and per-signal-family attribution
  - Logged alongside trade decisions
- Build live PnL attribution dashboard:
  - Attribution per agent (technical, regime, sentiment, consensus)
  - Attribution per signal family
  - Per-symbol and per-sector breakdown
  - Real-time during stress windows
- Implement operational dashboards:
  - Decision staleness, feature lag
  - Mode-switch frequency
  - OOD trigger rate
  - Kill-switch false positives
  - MTTR tracking

**Tests & Validation:**
- XAI coverage: ≥ 80% of test trades have explanations
- Attribution sums correctly across agents
- Dashboard loads and displays correct data
- Real-time attribution during simulated stress window

**Output:** XAI logging + PnL attribution + operational dashboards

---

### Day 7 (Sun): Week 3 Review & Tier 1 Gate

- Review all Tier 1 enhancement implementations
- Run combined pipeline with all Tier 1 features enabled
- Latency benchmark: full pipeline with all features
- Verify no regressions from Tier 1 additions
- Produce Week 3 checkpoint report

---

## Week 4 (April 21 – April 27): Integration, Backtesting & Paper-Trading Readiness

**Owner**: Both | **Focus**: Full-stack validation and paper-trading setup

### Day 1 (Mon): Full-Stack Backtesting

**Tasks:**
- Run full-stack walk-forward backtest:
  - Phase 1 data → Phase 2 signals → Phase 3 ensemble → portfolio → orders → PnL
  - Backtest span: 2019 to present
  - Covering: COVID crash, budget cycles, elections, Oct–Nov 2022 volatility
- Walk-forward with time-based cross-validation
- Leakage audit: timestamp checks + feature lag verification
- Survivorship bias controls: point-in-time universe (including delisted symbols)

**Tests & Validation:**
- Backtest report: Sharpe, Sortino, Calmar, MDD, win rate, profit factor
- Compare vs go-live targets (Sharpe ≥ 1.8, Sortino ≥ 2.0, MDD ≤ 8%)
- Leakage test suite passes 100%
- No survivorship bias leaks detected

**Output:** Full-stack backtest report

---

### Day 2 (Tue): Stress Testing & Impossible Scenarios

**Tasks:**
- Run stress scenario library (§9):
  - RBI surprise rate hike
  - INR flash move
  - Liquidity drought
  - Historical: 2008 GFC, 2013 taper tantrum, COVID
- Impossible-scenario tests:
  - Correlation inversion
  - Frozen constituent prices
  - Simultaneous multi-asset liquidity vacuum
- Data poisoning and feed-freeze simulations
- Snapback tests: measure recovery ticks after flash-shock
- Capacity stress tests at 1x, 2x, 3x notional

**Tests & Validation:**
- System enters correct protective mode during each stress scenario
- Impossible scenarios: system degrades gracefully (no crashes, no data corruption)
- Zero-volume vs missing-data distinction verified
- Snapback recovery within acceptable tick count
- 3x capacity: impact check vs 0.25% threshold

**Output:** Stress test report

---

### Day 3 (Wed): Paper-Trading Harness

**Tasks:**
- Build paper-trading simulation environment:
  - Realistic order fill simulation (delays, partial fills, rejections)
  - Slippage model from execution engine
  - Simulated broker responses
  - Position and PnL tracking
- Configure to consume live Phase 1 data feeds
- Deploy all Phase 2 + Phase 3 agents in paper-trading mode
- Set up continuous monitoring: PnL, drawdown, mode switches, alert rates
- Uptime tracking (target ≥ 80% during NSE trading hours)

**Tests & Validation:**
- Paper-trading session completes one full trading day without crashes
- Slippage model produces realistic fill estimates
- All agents produce signals and decisions correctly
- Monitoring dashboards display correct metrics

**Output:** Paper-trading environment operational

---

### Day 4 (Thu): MLOps & Model Governance

**Tasks:**
- Verify model registry completeness (§13):
  - All models registered with data snapshot + code hash
  - Training pipelines reproducible from raw data
- Champion-challenger promotion gate test:
  - Simulate challenger outperforming champion
  - Verify promotion workflow end-to-end
- Rollback test: revert model version, verify clean rollback
- Training stability automation (`test_retraining.py` pattern)
- Research-to-production lead time measurement setup
- Experiment tracking setup (per policy, per strategy family)

**Tests & Validation:**
- Model registry: all 6+ models (3 RL + students + ensemble) registered
- Promotion gate: automated gates enforce all required checks
- Rollback: tested with zero data loss
- Reproducibility: re-train from snapshot produces comparable results

**Output:** MLOps verification report

---

### Day 5 (Fri): Phase 3 Gate Evidence Collection

**Tasks:**
- Collect all Phase 3 GO benchmark evidence (§16.1):
  - Fast Loop latency benchmark: p50/p95/p99/p99.9
  - Teacher-student crisis-slice agreement report
  - Rollback drill log with MTTR
  - Walk-forward backtest results
  - Observation schema validation
- Compile per-policy model cards
- Compile Tier 1 enhancement evidence:
  - Slippage monitor functional
  - Volatility-scaled budgets functional
  - Order-book imbalance integrated
  - Latency discipline CI gate passing
- Compliance audit trail review

**Tests & Validation:**
- All Phase 3 GO benchmarks pass documented thresholds
- No blocking defects remain open
- Compliance audit trail is complete for all test trades

**Output:** Phase 3 gate evidence package

---

### Day 6 (Sat): GO/NO-GO Assessment

**Tasks:**
- Formal Phase 3 GO/NO-GO evaluation against §16.1 gates:
  - ✅ / ❌ Latency gates met?
  - ✅ / ❌ Teacher-student agreement on crisis slices?
  - ✅ / ❌ Rollback tested successfully?
  - ✅ / ❌ Walk-forward validation passes?
  - ✅ / ❌ All Tier 1 enhancements operational?
- Document any at-risk items with mitigation plans
- Risk assessment for paper-trading phase

**Output:** GO/NO-GO assessment document

---

### Day 7 (Sun): Handoff & Paper-Trading Launch Planning

- Package Phase 3 delivery
- Final documentation review
- Define paper-trading calendar:
  - Mandatory ≥ 3 calendar months
  - Target ≥ 80% uptime during NSE trading hours
  - Weekly review cadence
  - Escalation procedures
- Define Phase 4 (Risk Overseer standalone hardening) scope
- Final sign-off memo

---

## New Code Modules

| Module | Path | Purpose |
|---|---|---|
| **Strategic Agent Package** | `src/agents/strategic/__init__.py` | Package init |
| **Config** | `src/agents/strategic/config.py` | Hyperparameters, thresholds, feature flags |
| **Schemas** | `src/agents/strategic/schemas.py` | Observation, action, decision schemas |
| **Observation Assembler** | `src/agents/strategic/observation.py` | Phase 2 → observation vector |
| **Reward Library** | `src/agents/strategic/reward.py` | All reward functions |
| **Training Environment** | `src/agents/strategic/environment.py` | Gymnasium-compatible env |
| **SAC Policy** | `src/agents/strategic/policies/sac.py` | SAC implementation |
| **PPO Policy** | `src/agents/strategic/policies/ppo.py` | PPO implementation |
| **TD3 Policy** | `src/agents/strategic/policies/td3.py` | TD3 implementation |
| **Ensemble Engine** | `src/agents/strategic/ensemble.py` | Max-entropy ensemble |
| **Distillation** | `src/agents/strategic/distillation.py` | Teacher-student pipeline |
| **Policy Manager** | `src/agents/strategic/policy_manager.py` | Snapshot system, Fast Loop cache |
| **Deliberation** | `src/agents/strategic/deliberation.py` | Slow Loop reasoning |
| **Portfolio** | `src/agents/strategic/portfolio.py` | Portfolio construction, risk budgets |
| **Execution** | `src/agents/strategic/execution.py` | Order routing, compliance, fills |
| **Impact Monitor** | `src/agents/strategic/impact_monitor.py` | Tier 1-A: slippage monitoring |
| **Risk Budgets** | `src/agents/strategic/risk_budgets.py` | Tier 1-B: volatility-scaled |
| **Phase 3 Recorder** | `src/db/phase3_recorder.py` | DB insert/upsert helpers |
| **Phase 3 Models** | `src/db/models.py` (extend) | 12 new SQLAlchemy tables |
| **Training Scripts** | `scripts/train_rl_policies.py` | RL training orchestrator |
| **Distillation Script** | `scripts/train_student.py` | Student distillation |

---

## Database Tables (12 new — see [Phase_3_Database_Plan.md](file:///Users/juhi/Desktop/algo-trading/docs/plans/Phase_3_Database_Plan.md))

| Table | Written By | Read By |
|---|---|---|
| `rl_policies` | Training pipeline | Ensemble, promotion gates |
| `rl_training_runs` | Training pipeline | Audit, reporting |
| `observations` | Observation Assembler | Ensemble, backtesting |
| `trade_decisions` | Ensemble Engine | Execution, risk, audit |
| `orders` | Execution Engine | Fill tracking, compliance |
| `order_fills` | Execution Engine | PnL, slippage |
| `portfolio_snapshots` | Portfolio Engine | Risk, dashboards |
| `reward_logs` | Reward Computer | Training, analytics |
| `policy_snapshots` | Policy Manager | Fast Loop, audit |
| `student_policies` | Distillation pipeline | Fast Loop, promotion |
| `distillation_runs` | Distillation pipeline | Audit, reporting |
| `deliberation_logs` | Deliberation Engine | Analytics |

---

## Risk Watch

| Risk | Mitigation |
|---|---|
| RL training is compute-intensive and may not converge in 1 day per policy | Use Stable-Baselines3 proven implementations; run overnight full training; use quick smoke-train for debugging |
| Teacher-student distillation may not achieve high agreement on crisis slices | Pre-select diverse crisis data; iterate on student architecture; allow Day 2 overflow into Day 3 |
| Fast Loop latency target (p99 ≤ 8ms) may be hard to hit initially | Profile early (Day 3 Week 1); quantize aggressively; strip non-essential features until targets met |
| Portfolio construction + execution is substantial scope for 2 days | Focus on core position sizing + basic order routing first; defer advanced hedging to Week 3 |
| Integration between Phase 2 and Phase 3 observation assembly may have data quality issues | Freeze schema in Week 1; add extensive quality checks in observation assembler |
| GA threshold search may overfit on training data | Use walk-forward splits; constrain search space; validate on holdout |

---

## Phase 3 Exit Criteria

| Criterion | Evidence Required |
|---|---|
| SAC, PPO, TD3 trained with walk-forward evaluation | Training reports + backtest results per policy |
| Max-entropy ensemble outperforms best individual policy | Comparison report on holdout set |
| Teacher-student agreement on crisis slices | Crisis-slice agreement report |
| Student policy meets Fast Loop latency budget | Latency benchmark: p99, p99.9 |
| Portfolio construction enforces risk budgets | Risk budget test log |
| Order execution with compliance checks | Compliance audit trail |
| All Tier 1 enhancements operational | Tier 1 evidence package |
| Full-stack walk-forward backtest complete | Backtest report vs go-live targets |
| Stress test library passes | Stress test report |
| Paper-trading environment running | Deployment confirmation |
| Rollback tested and documented | Rollback drill log + MTTR |
| All 12 database tables operational | Schema + sample data verification |
| All model cards complete | Model card metadata files |
| Phase 3 GO gate passes (§16.1) | GO/NO-GO assessment |
