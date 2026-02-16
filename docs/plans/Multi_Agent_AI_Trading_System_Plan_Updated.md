# Multi-Agent AI Trading System
Updated Integrated Plan for Indian Market (USD/INR + NSE Stocks & Indices)
Version 1.3.7 | February 2026

## 0. Document Control
Purpose: maintain versioning, accountability, and auditability.
Acceptance Criteria:
- Single owner and reviewer list are defined.
- Single owner and reviewer are defined for each asset cluster and artifact; full cross-functional pod operating model is deferred in the current cycle.
- Versioning and changelog are maintained in this document.
- Each model and data artifact is traceable to a plan version.
- Version 1.2 change set captures dual-loop execution, sentiment cache decoupling, crisis override voting, and impossible-scenario stress tests.
- Version 1.2.1 applies minimum-safe scope: ship dual-loop, sentiment cache, basic crisis mode, and impossible stress testing; defer aggressive alien-state automation, snapback smoothing tuning, and full winner-takes-all automation.
- Version 1.3.0 adds hardware-acceleration track, market-making readiness module, research-to-production throughput KPIs, low-latency ML release gates, pod operating model, data-scale roadmap and replay, and change-outcome accountability reviews.
- Version 1.3.1 tightens execution latency targets (5 to 8 ms stretch with enforced degrade cap), adds real-time impact and attribution controls, introduces mandatory shadow A/B for major FX-gold changes, and formalizes optional online RL micro-updates under strict safety gates.
- Version 1.3.2 removes speculative uplift language, adds explicit no-assumed-speedup policy for Rust/C++/FPGA choices, and introduces phase-by-phase go/no-go benchmark gates.
- Version 1.3.3 added evidence-first rigor guidance and prioritization for IMC-inspired tech adoption.
- Version 1.3.4 folds rigor guidance directly into MLOps, rollout, and phase-gate sections and removes the standalone memo section.
- Version 1.3.5 operationalizes release governance with mandatory evidence-first change-request and CI benchmark checklist templates.
- Version 1.3.6 applies implementation freeze discipline with explicit Tier 1, Tier 2, and Tier 3 priorities; Tier 3 items are deferred for current-cycle delivery.
- Version 1.3.7 decomposes Tier 1 priorities into concrete execution tasks, artifacts, and completion checks for immediate implementation.

## 1. Executive Summary
This updated plan refines the existing multi-agent trading system for Indian markets by adding concrete governance, execution, and validation controls. It preserves the core multi-agent architecture while tightening scope, compliance, data integrity, and rollout safety. The plan incorporates a dedicated Sentiment Agent in Phase 2 using a fine-tuned FinBERT model to improve event-driven robustness. Deployment and promotion decisions are governed by explicit benchmark gates in Section 16 and Appendix A rather than assumed performance uplifts. Version 1.3.7 keeps strict tier prioritization and adds concrete Tier 1 execution tasks for immediate delivery.

## 2. Scope, Objectives, and Trading Constraints
Defines the trading horizon, asset universe, allowed actions, and performance targets.
Acceptance Criteria:
- Primary trading frequency is Hourly and explicitly documented.
- Daily timeframe is used only for confirmation and regime context filtering.
- Data sampling frequency is locked to the primary horizon to avoid model-data mismatch.
- Asset universe is explicit and versioned (NSE equities, indices, USD/INR, F&O).
- Universe selection policy includes numeric filters: Avg Daily Turnover >= ₹10 Cr (6-month average), Impact Cost <= 0.50 percent, Free float >= 20 percent (per NSE IWF methodology), Price >= ₹50, Listing age >= 12 months, and no stocks with > 5 circuit hits in the last 3 months.
- Universe review and rebalance occurs quarterly via the Preprocessing Agent.
- Allowed actions are specified per product (shorting, leverage, lot size, margin, order types).
- Performance objectives include return, risk, drawdown, capacity, and turnover targets.
- Target performance metrics are stated as goals, not guarantees, with confidence intervals.

## 3. Regulatory and Broker Compliance
Defines compliance requirements and audit trails for SEBI and broker rules.
Acceptance Criteria:
- Compliance checklist exists for SEBI and broker rules.
- Pre-trade checks include position limits, margin rules, and circuit breaker rules.
- Audit trail includes order intent, execution, cancellation, and model version used.
- Static Mumbai IP whitelisting and broker access controls are documented as production-only requirements.
- Regulatory reporting automation and retention policy are documented.
- Algorithm approval or notification requirements and periodic audit obligations are documented where applicable.
- Data usage policy explicitly prohibits unpublished or embargoed information; only publicly released or contractually licensed feeds are permitted.

## 4. System Architecture and Interfaces
Defines module boundaries, data flows, APIs, and latency budgets.
Acceptance Criteria:
- Each module has defined inputs, outputs, and failure modes.
- Data flow diagram exists with transport protocols and schemas.
- Latency budgets are documented per stage (ingest, features, inference, execution).
- Stage SLAs include p50, p95, and p99 targets with hard caps and automatic mode degradation on breach.
- Architecture enforces dual-loop separation: Fast Loop (execution critical) and Slow Loop (context and policy updates).
- Fast Loop target is p99 <= 8 ms as a stretch objective, with enforced degrade or bypass if p99 exceeds 10 ms, and processes only technical microstructure, order-imbalance features, and cache reads.
- Slow Loop supports heavier inference (100 to 500 ms typical) and is asynchronous, never blocking order placement.
- Volatility bypass thresholds use hysteresis and cooldown windows to prevent rapid mode flapping.
- Deployment topology (Mumbai-based low-latency VPS) is specified.
- Minimum-safe release profile is explicit: advanced overrides are disabled by default and only basic crisis-weighted controls are active.
- ML production rulebook is explicit: no heavy inference in Fast Loop, student-policy inference only in the execution path, and teacher policy execution stays offline or Slow Loop only.
- Model artifact promotion includes deterministic runtime and compression checks; p99 and p99.9 tail-latency budgets are mandatory release gates.
- Hardware acceleration policy is explicit: FPGA is permitted only for execution-path critical components with measured latency benefit; non-critical components remain CPU/GPU.
- Critical path implementation policy allows Rust/C++ microservices for latency-sensitive functions (for example order-book parsing and routing adapters) while orchestration remains language-agnostic.
- No language or hardware path is assumed to deliver fixed speedup multipliers; Rust/C++/FPGA promotion requires measured p99 and p99.9 improvement versus baseline plus operational parity on correctness, observability, and recovery drills.

## 5. Phase 1: Data Orchestration Layer
Specialized agents for acquisition and preprocessing.

### 5.1 NSE Sentinel Agent
Real-time and historical market data acquisition from NSE sources.
Acceptance Criteria:
- Data sources include nsepython, jugad-data, and official NSE APIs or their replacements.
- Redundancy exists for core price, volume, and corporate action feeds.
- Timestamp validation and clock sync are enforced across all streams.
- Official API or exchange/broker endpoints are the primary source, with web scraping restricted to fallback mode.
- Signal provenance is tagged per event (primary API vs fallback scraper) and exposed to downstream risk controls.
- Primary data-feed failure triggers automatic reduce-only or close-only mode until integrity checks pass.

### 5.2 Macro Monitor Agent
Indian macro indicators and cross-asset signals.
Acceptance Criteria:
- Macro indicators list is defined and versioned.
- Indicators include WPI, CPI, IIP, FII/DII flows, FX reserves, RBI bulletins, and India-US 10Y spread.
- Global proxies such as Brent and DXY are explicitly justified as India-relevant or removed.
- Data quality checks cover missingness, outliers, and latency.

### 5.3 Preprocessing Agent
Normalization, noise filtering, corporate actions, and feature readiness.
Acceptance Criteria:
- Feature engineering is deterministic and reproducible.
- Rolling window normalization and directional change thresholds are implemented.
- Corporate action adjustments are validated against a reference source.
- Leakage tests confirm strict time alignment and lagging.
- Feature approval process is defined (proposal, offline evaluation, shadow testing, promotion).

### 5.4 Textual Data Agent
Text collection for sentiment analysis and event detection.
Acceptance Criteria:
- Source list includes NSE news, Economic Times, RBI reports, and earnings transcripts.
- Social data collection from X is documented with keyword and semantic search rules.
- PDF extraction pipeline is validated with spot checks.
- Code-mixed English and Hinglish handling strategy is documented.

### 5.5 Data Scale Roadmap and Replay
Data platform growth and reproducibility posture.
Acceptance Criteria:
- Data processing roadmap defines current and target daily processing volumes (GB to TB) with quarterly checkpoints.
- Dataset tiering is explicit: hot (in-memory/cache), warm (low-latency store), and cold (archive) with retention policies.
- Replay framework can reconstruct any trading day from raw feed payloads through feature artifacts with deterministic identifiers.
- Replay supports both event-time and wall-clock playback for strategy diagnostics and latency forensics.

## 6. Phase 2: Analyst Board (Modeling)
Specialized models for forecasting, regime detection, sentiment, and consensus.

### 6.1 Technical Agent
Time-series modeling for price prediction and action thresholds.
Acceptance Criteria:
- Model families include ARIMA-LSTM hybrids, 2D CNNs, and GARCH quantiles for VaR and ES.
- Model targets and metrics are specified for each family.
- Baselines and ablation tests are implemented.

### 6.2 Regime Agent
Regime detection and transition modeling.
Acceptance Criteria:
- PEARL or an equivalent meta-learning model is specified.
- Regime definitions include RBI inner and outer band zones.
- Transition criteria cover event-driven shifts and volatility regime changes.
- Calibration tests and stability checks are documented.
- Regime training and stress validation include structural breaks (2008 crisis, 2013 taper tantrum, COVID window, and RBI policy discontinuities).
- Novelty or out-of-distribution detection uses explicit statistical distance thresholds with an alien-state flag.
- Alien-state detection triggers staged de-risking (full risk, reduced risk, neutral or cash) with documented escalation.

### 6.3 Sentiment Agent
FinBERT-based sentiment modeling for Indian markets.
Acceptance Criteria:
- Base model is ProsusAI/finbert or equivalent with documented justification.
- Fine-tuning datasets include Indian sources such as harixn/indian_news_sentiment and SEntFiN.
- Bayesian priors or regularization are defined for stability.
- Daily sentiment aggregates form a threshold variable (z_t) used in regime logic.
- Precision and recall thresholds are defined by class, with timestamp alignment tests.
- Robustness controls include spam and adversarial text filtering, source deduplication, and noise handling.
- Dual-speed sentiment design is defined: intraday fast lane (lightweight model plus keyword rules) and offline slow lane (deep model, nightly updates).
- Sentiment runs on an independent timeline and writes outputs to fast cache (e.g., Redis) with timestamp, confidence, and TTL.
- Fast Loop consumes the latest cached sentiment via O(1) memory reads and never waits for fresh NLP inference.
- Intraday sentiment path targets <= 100 ms score generation from headline arrival under normal load, outside the execution-critical path.
- Sentiment freshness TTL and event-priority routing are documented to prevent stale text overriding fresh market signals.
- Cache decision policy is deterministic: fresh -> use, stale -> downweight, expired -> ignore.
- Cache read failure triggers technical-only reduced-risk execution mode until cache health recovers.
- Pump-and-dump and slang-scam detection for code-mixed text is required; manipulative hype spikes can force sentiment downgrade to neutral or protective.

### 6.4 Consensus Agent
Aggregates signals using LSTAR or ESTAR with Bayesian estimation.
Acceptance Criteria:
- Logistic and exponential transition functions are specified.
- Bayesian estimation of smoothness and location parameters is documented.
- Transition function includes volatility, macro differentials, RBI signals, and sentiment quantiles.
- Consensus stability is tested during high-volatility windows.
- Default mode uses weighted consensus with explicit safety bias toward protective signals.
- Basic crisis mode uses crisis-weighted routing with capped dominance (max 60 to 70 percent for the crisis agent), not full winner-takes-all; activation is Tier 2 and scheduled after three months of stable paper trading.
- Agent divergence (fundamental disagreement between agents) is modeled as a dedicated regime signal that triggers staged risk reduction or temporary freeze.
- Full winner-takes-all crisis automation is deferred until false-positive and stability thresholds are met in shadow mode.
- Snapback tests are retained for measurement; automatic Bayesian smoothing retuning from snapback output is deferred.

## 7. Phase 3: Strategic Executive (Trader Ensemble)
RL ensemble for trade execution decisions.

### 7.1 Ensemble Composition
Acceptance Criteria:
- Ensemble includes SAC, PPO, and TD3 or justified alternatives.
- Each policy has defined training cadence and evaluation metrics.
- Optional market-making policy head is supported for low-volatility liquidity-provision windows, disabled by default and deferred to Tier 3 in the current cycle.

### 7.2 Decision Mechanism
Acceptance Criteria:
- Decision policy uses a maximum entropy framework.
- Multi-threshold genetic algorithm is specified for action selection.
- Genetic algorithm evolution and search run offline in training and calibration.
- Live execution uses teacher-student distillation, where the student policy mimics teacher decisions under strict latency budgets.
- Student promotion requires teacher-student agreement thresholds on crisis slices, not only average-day performance.
- Offline teacher monitoring remains active in production analytics, and student drift beyond threshold triggers automatic demotion.
- Fast Loop inference is restricted to distilled student policy only; teacher inference is blocked from execution-critical path.
- Student model artifacts require deterministic runtime profiles and approved compression plan (for example quantization/pruning/distillation) before promotion.

### 7.3 Observation Space
Acceptance Criteria:
- Observation space includes outputs from Technical, Regime, Sentiment, and Consensus agents.
- Features include regime probabilities, VaR and ES levels, macro differentials, and sentiment scores.
- Observation schema is versioned and validated.
- Fast Loop microstructure set explicitly includes order-book imbalance and queue-pressure features when feed quality is healthy.

### 7.4 Reward and Utility Functions
Acceptance Criteria:
- Reward library includes RA-DRL composite, Sharpe, Sortino, Calmar, Omega, and Kelly variants.
- Dynamic weighting is tied to regime state and sentiment signals.
- Downside penalties increase during RBI outer band or negative sentiment spikes.
- Formulas are documented in an appendix or code reference.
- Liquidity-provision reward variant is defined for optional market-making mode, including maker-rebate capture net of adverse-selection and inventory-risk penalties.

### 7.5 Deliberation Process
Acceptance Criteria:
- Critical-path thinker logic is lightweight and deterministic; heavy reasoning remains in the Slow Loop.
- Simulations and policy search are removed from the execution-critical path and run asynchronously.
- Execution consumes the latest validated policy snapshot, with scheduled refresh (e.g., every 10 minutes) or earlier on async completion.
- Deliberation bypass is enabled under high-volatility triggers and does not block order decisions.
- Trade rejection is tied to risk-control breaches, stale-data TTL breaches, or latency-cap breaches, not pending simulation completion.
- Fast Loop peak-load decision stack targets p99 <= 8 ms as stretch, with enforced degrade or bypass safeguards above 10 ms for in-process compute stages.
- Slow Loop latency (100 to 500 ms typical) is monitored separately and cannot directly bypass Fast Loop safety gates.
- Release gates include jitter and tail-latency tests (p99.9) for decision-path determinism during peak-load simulations.

### 7.6 Safe RL Training and Deployment
Acceptance Criteria:
- Online learning is gated behind offline training and paper trading validation.
- Exploration is constrained by risk limits and hard safety filters.
- Promotion to live trading requires risk committee approval and rollback readiness.
- Optional online RL micro-updates (15 to 30 minute cadence) are deferred to Tier 3 in the current cycle; re-evaluate only after Tier 1 and Tier 2 completion.

## 8. Portfolio Construction and Risk Budgeting
Dedicated allocation and exposure management.
Acceptance Criteria:
- Risk budgets per asset class and sector are defined.
- Correlation and concentration limits are enforced.
- Position sizing ties to forecast uncertainty and liquidity.
- Inventory and exposure state model tracks gross, net, and hedge-adjusted risk even when strategy is directional.
- Hedging policy engine is a first-class action type with cross-asset correlation checks and explicit risk-unwind logic.
- Capacity ceiling is defined: initial live notional ₹5–10 Cr, with scale to ₹25 Cr only after 6 months of live validation; stress-tested at 1x, 2x, and 3x capacity, with a hard cap at 2x if impact exceeds 0.25 percent at 3x.
- Max participation limits are documented and enforced at execution time.
- Downside risk emphasis is enforced during panic regimes.

### 8.1 Market-Making Readiness Module (Dormant by Default)
Readiness module for future quoting strategies without changing current directional deployment scope.
Acceptance Criteria:
- Inventory controls are reusable for both directional and quoting modes.
- Hedging action library includes partial unwind, cross-asset hedge, and emergency neutralization playbooks.
- Enablement flag is default-off in production until dedicated market-making validation gates are passed.

## 9. Stress Testing Framework
Dedicated stress testing framework to validate resilience under adverse conditions.
Acceptance Criteria:
- A maintained scenario library exists with versioned definitions and expected outcomes.
- Explicit shock scenarios include an RBI surprise rate hike, INR flash move, and liquidity drought.
- Historical scenario set explicitly includes 2008 Global Financial Crisis data and 2013 Taper Tantrum regime shocks.
- Stress tests cover historical crises and synthetic shocks at 1x, 2x, and 3x capacity.
- Results are reviewed quarterly and trigger risk budget or capacity adjustments.
- Impossible-scenario tests are mandatory, including correlation inversion, frozen constituent prices, and simultaneous multi-asset liquidity vacuum.
- Data poisoning and feed-freeze simulations are included, with checks for correct behavior when no ticks or no book updates arrive for defined windows.
- Validation confirms the engine distinguishes zero volume from missing data and enters safe mode when feed integrity is uncertain.
- Snapback tests measure how many ticks it takes to clip net exposure after flash-shock conditions; smoothing parameters are capped if recovery is too slow.
- Quote-response simulator tests are included to measure reaction speed, inventory drift, and unwind behavior under rapid microstructure changes.

## 10. Execution and Market Impact
Order routing and slippage management tailored to NSE.
Acceptance Criteria:
- Execution supports partial fills, pacing, and participation limits.
- Slippage model is validated against intraday data.
- Circuit breaker and halt handling is implemented and tested.
- Execution metrics include fill rate, slippage vs model, and latency.
- Market impact model is defined and linked to capacity assumptions.
- Real-time routing health checks are enforced, with automatic degradation to reduce-only or close-only on execution infrastructure failure.
- Real-time impact monitor tracks realized slippage against participation and ADV in near-real-time; position sizing is reduced automatically when impact breaches thresholds.

## 11. Phase 4: Independent Risk Overseer
Safety controls and oversight independent of trading logic.
Acceptance Criteria:
- Kill switch and loss limits enforced at broker and system levels.
- Controls include max drawdown, daily loss, and position size constraints.
- Extreme negative sentiment spikes trigger protective actions.
- Automated drift detection uses ADDM or equivalent.
- XAI logging captures top features and agent contributions using SHAP.
- Kill switch hierarchy is documented with layered triggers (model, portfolio, broker, manual) and escalation order.
- XAI coverage thresholds are defined (e.g., >= 80 percent of trades with top-k explanations logged).
- Operating modes are explicitly defined as normal, reduce-only, close-only, and kill-switch with trigger and recovery criteria.
- Dynamic risk budgets are volatility-aware; if realized volatility exceeds configured sigma thresholds, exposure caps auto-scale down before hard kill-switch triggers.
- Data provenance reliability scoring dynamically adjusts position sizing instead of fixed haircut rules.
- OOD or alien-state triggers enforce staged de-risking before full neutralization unless hard limits are breached.
- Crisis taxonomy explicitly defines full crisis, agent divergence, and slow-crash or freeze states, each mapped to mode transitions and overrides.
- Sentiment-to-price mismatch circuit breaker can downgrade sentiment to neutral during suspected manipulation or panic slang bursts.
- Crisis entry requires multi-condition confirmation (volatility break, liquidity deterioration, and agent confidence threshold).
- Crisis triggers must persist for configured ticks or seconds before activation, with explicit cooldown and hysteresis rules.
- Crisis mode has max-duration expiry and auto-reverts unless revalidated by trigger conditions.
- Agent divergence has explicit neutral-hold duration and staged re-risking rules after alignment recovery.
- Risk overseer can trigger hedge or unwind actions directly when exposure or correlation thresholds breach limits.
- Live PnL attribution is available per agent and per signal family to support real-time diagnosis during stress windows.

## 11.5 Phase 4.5: Hardware Acceleration and Low-Latency Track (Optional, Deferred for Current Cycle)
Targeted hardware evaluation path for execution-critical latency improvements.
Acceptance Criteria:
- Status is deferred for current-cycle implementation and is re-opened only after Tier 1 and Tier 2 milestones are complete.
- Candidate components for acceleration are limited to execution-path critical steps (for example market data decode/normalization and selected pricing kernels).
- Candidate components can include order-book parsing and routing adapters implemented in Rust/C++ microservices before FPGA escalation.
- FPGA evaluation includes deterministic latency benchmarking: p50, p95, p99, p99.9, jitter, and worst-case tail behavior.
- Hardware path has parity tests versus software baseline for correctness and failure handling.
- Promotion requires measured latency/efficiency gain with no regression in risk controls, observability, or operational recovery.
- Policy is enforced: non-critical workloads remain CPU/GPU to avoid unnecessary hardware complexity.

## 12. Phase 5: Validation and Evolution Loop
Backtesting, paper trading, and continuous learning.
Acceptance Criteria:
- Vectorbt or equivalent framework is used for realistic simulations.
- Backtests span 2018 or 2019 to present, covering COVID crash, budget cycles, elections, and Oct-Nov 2022 volatility.
- Walk-forward evaluation and time-based CV are required.
- Leakage tests include timestamp audits and feature lag checks.
- Survivorship bias controls are enforced using point-in-time universe data, including delisted symbols.
- Nightly incremental learning uses TTM or equivalent lightweight models.
- PEARL-based meta-learning adapts to the most recent 1 to 3 month regimes.
- Training stability tests run via automated scripts like test_retraining.py.
- Weekly sentiment re-fine-tuning is validated against SEntFiN or equivalent.
- Synthetic data usage policy is documented, including labeling, scope boundaries for text and price data, and exclusion rules for performance claims unless validated on real data.
- Stress library includes structural breaks and synthetic payment or policy shock scenarios with clear labels for synthetic-origin data.
- Dual-loop boundary tests verify that text floods alone cannot flip portfolio direction without confirming technical or risk conditions.
- Validation harness includes quote-response simulation slices for readiness testing of future market-making behaviors.
- Phase 5 includes a dedicated research loop for rapid hypothesis testing (for example new z_t context variables) with explicit accept or reject criteria.

## 13. MLOps and Model Governance
Versioning, reproducibility, and controlled deployment.
Acceptance Criteria:
- Model registry includes data snapshot and code hash.
- Champion-challenger promotion gates are codified.
- Rollback policy exists and is tested.
- Training pipelines are reproducible from raw data.
- Human oversight gates are defined, including review cadence and approval authority.
- Material strategy changes (latency bypass, dual-speed sentiment, novelty safeguards) require ablation evidence before promotion.
- Distributed training posture is defined for scale-up (parallelized nightly TTM/PEARL jobs) with cost and reproducibility controls.
- Research-to-production lead time is measured end-to-end (idea -> backtest -> paper -> shadow -> live) and reviewed at least monthly.
- Experiment throughput is tracked (experiments per day/week) by pod and strategy family.
- Compute cost per approved strategy update is tracked and budget-guarded.
- Experiment failure taxonomy is reported (data-integrity failures vs model-logic failures) with corrective-action ownership.
- Fixed uplift claims (for example fixed Sharpe, drawdown, accuracy, or speedup multipliers) are prohibited in approval artifacts unless backed by controlled benchmark or shadow A/B evidence.
- Technology-change approvals (language/runtime/hardware) must report measured p99 and p99.9 deltas versus baseline plus correctness, observability, and recovery parity evidence.

## 14. Rollout Strategy and Live Monitoring
Controlled deployment with shadow testing and gating.
Acceptance Criteria:
- Shadow mode runs for a fixed minimum window.
- Promotion requires challenger outperforming champion on risk-adjusted metrics.
- Shadow A/B enforcement for major strategy or model changes in USD/INR and gold tracks is Tier 2 and activates after three months of paper trading; before activation, A/B is strongly recommended with explicit rationale if skipped.
- Monitoring dashboards include PnL attribution, drift, and risk limits.
- Promotion gates require no regression in Sharpe, drawdown, and slippage, plus acceptable rates for mode switching and false kill-switch events.
- Crisis logic and override routing must complete shadow-mode validation before live activation.
- Override rollout is sequential: enable one override at a time, validate, then promote the next.
- Every production change must include expected impact statement, measured impact window, and rollback trigger before deployment approval.
- Expected impact statement uses standardized evidence-first wording: unknown until controlled benchmark or shadow A/B; target is non-regression plus measurable improvement.
- Every change request must state go criterion as all applicable phase-gate checks passing.
- Every change request must state no-go criterion as any latency, risk, or correctness gate failure.
- Every promotion request must include completed governance templates for release record and CI benchmark evidence checklist.
- Mandatory 48-hour post-deploy review covers latency, slippage, risk events, drift, and incident count.

## 15. Operations, Security, and Resilience
Reliability, incident response, and secure operations.
Acceptance Criteria:
- Secrets management is centralized and audited.
- DR plan defines RPO and RTO targets.
- Centralized logging with retention policies is implemented.
- Incident response runbook exists and is exercised.
- System time sync is enforced via NTP or equivalent with drift alerts.
- Observability includes metrics, traces, and alerts for data latency, inference time, and execution failures.
- Operational dashboards track decision staleness, feature lag, mode-switch frequency, OOD trigger rate, kill-switch false positives, and MTTR.
- Full cross-functional pod operating model is deferred to Tier 3 in the current cycle.
- Interim operating model uses single owner plus reviewer accountability with explicit metric ownership.

## 16. Milestones and Deliverables
Phased execution with measurable outcomes.
Acceptance Criteria:
- Each phase has exit criteria, owners, and timelines.
- Go-live criteria include performance, stability, and compliance readiness.
- Every milestone artifact has pod-level ownership with a primary DRI and backup reviewer.

### 16.1 Phase Go or No-Go Benchmark Gates

| Phase | GO Benchmarks (all required) | NO-GO Trigger |
| --- | --- | --- |
| Phase 1: Data Orchestration | Data uptime >= 99.5 percent during NSE hours for at least 10 consecutive trading days; core symbol completeness >= 99.0 percent; provenance tagging coverage = 100 percent; leakage tests = 100 percent pass | Any critical data integrity failure unresolved, or any required benchmark not met |
| Phase 2: Analyst Board | Baseline and ablation packs are complete for each active model family; sentiment precision/recall thresholds and timestamp alignment tests pass; OOD and regime transition validations pass documented thresholds | Missing baseline evidence, failed sentiment/OOD checks, or unresolved model-data alignment defects |
| Phase 3: Strategic Executive | Fast Loop decision stack meets latency release gates (stretch p99 <= 8 ms, enforced degrade path above 10 ms, and p99.9 jitter limits); teacher-student agreement threshold passes on crisis slices; rollback path is tested | Latency gate breach without compliant degrade behavior, failed crisis-slice agreement, or untested rollback |
| Phase 4: Independent Risk Overseer | Kill-switch hierarchy drills pass (model, portfolio, broker, manual); dynamic volatility-scaled budget controls function in stress drills; hedge and unwind triggers behave as specified | Any failed kill-switch drill, risk limit breach without protective transition, or unresolved false-trigger escalation path |
| Phase 4.5: Hardware Acceleration (Optional) | Candidate component shows pre-defined benchmark improvement versus software baseline on p99 and p99.9; correctness parity and failover parity are validated; observability and incident recovery remain compliant | No measured latency benefit, any correctness mismatch, or degraded recovery and observability |
| Phase 5: Validation and Evolution | Walk-forward, time-based CV, leakage, and dual-loop boundary tests pass; shadow A/B non-regression is confirmed for risk-adjusted metrics and slippage; stress and impossible-scenario library passes signed review | Any validation failure, statistically material regression in shadow A/B, or unreviewed stress-test exceptions |
| Phase 17: Gold Expansion | Core stack has already passed go-live checklist; COMEX shock scenario limits and gold-specific notional/participation controls pass drills; gold shadow window meets risk and stability gates | Base stack not go-live eligible, failed gold stress controls, or unresolved gold-specific execution risk defects |

Go-Live Checklist (Minimums):
- Minimum 3 calendar months of paper trading with >= 80 percent uptime.
- Annualized Sharpe >= 1.8.
- Sortino >= 2.0.
- Max Drawdown <= 8 percent.
- Win rate >= 52 percent.
- Profit factor >= 1.5.
- Average realized slippage <= model estimate + 20 bps.
- Data uptime >= 99.5 percent during NSE trading hours.
- Zero critical compliance violations and zero broker or SEBI flags or order rejections.

## 17. Gold Expansion (MCX Bullion Track)
Add exchange-traded gold alongside existing USD/INR and NSE instruments using the same multi-agent stack.
Scope and Assumptions:
- Instruments: MCX `GOLD` and `GOLDM` futures in Phase 1; options only after stable futures rollout.
- Venue: Indian exchange-traded commodity segment only.
- Dependency: Existing core stack has passed current go-live checklist.
- Gold risk policy includes scenario-based notional and participation limits during COMEX-linked shock windows.


## 18. Next Iteration Enhancements (Nice-to-have)
Planned upgrades to be prioritized after initial production stability.
Acceptance Criteria:
- Backlog is reviewed quarterly and scoped only after six months of stable live operation.
- Any new enhancements require documented risk review and rollback plans before deployment.

Prioritization Framework:
- Tier 1 (implement first, next 4 to 8 weeks): real-time impact and slippage monitoring with automatic position reduction; dynamic volatility-scaled risk budgets; order-book imbalance in Fast Loop; tightened Fast Loop latency discipline (8 ms stretch, 10 ms degrade safeguard).
- Tier 2 (implement after three months of paper trading): shadow A/B enforcement for major changes; basic crisis-weighted voting with 60 to 70 percent cap.
- Tier 3 (strongly defer for current cycle): hardware acceleration and FPGA track; market-making module; online RL micro-updates; full cross-functional pod ownership model.

### 18.1 Current-Cycle Priority Table
| Tier | Initiative | Window | Status |
| --- | --- | --- | --- |
| Tier 1 | Real-time impact and slippage monitoring with auto reduction | Next 4 to 8 weeks | Implement now |
| Tier 1 | Dynamic volatility-scaled risk budgets | Next 4 to 8 weeks | Implement now |
| Tier 1 | Order-book imbalance reinforcement in Fast Loop | Next 4 to 8 weeks | Implement now |
| Tier 1 | Tightened Fast Loop latency discipline (8 ms stretch) | Next 4 to 8 weeks | Implement now |
| Tier 2 | Shadow A/B enforcement for major changes | After three months paper trading | Phase in later |
| Tier 2 | Basic crisis-weighted voting (60 to 70 percent cap) | After three months paper trading | Phase in later |
| Tier 3 | Hardware acceleration and FPGA track | Deferred | Do not implement in current cycle |
| Tier 3 | Market-making module | Deferred | Do not implement in current cycle |
| Tier 3 | Online RL micro-updates | Deferred | Do not implement in current cycle |
| Tier 3 | Full cross-functional pod ownership model | Deferred | Do not implement in current cycle |

### 18.2 Tier 1 Task Breakdown (Execution Backlog)

#### Tier 1-A Real-Time Impact and Slippage Monitor with Auto Reduction
Tasks:
- Define impact thresholds by instrument bucket (for example liquid index names, mid-liquidity names, and gold futures).
- Add live metrics pipeline: realized slippage, participation, and ADV-linked impact score.
- Implement automatic position-sizing reducer on threshold breach with cooldown and hysteresis.
- Add alerting and dashboard panels for threshold breaches and auto-reduction events.
Completion Checks:
- Replay and paper-trading drills show threshold events trigger expected size reduction behavior.
- No correctness regressions in order intent, fill tracking, and risk logs.
- Post-deploy review template captures slippage-impact incident counts and outcomes.

#### Tier 1-B Dynamic Volatility-Scaled Risk Budgets
Tasks:
- Define volatility regime calculation and sigma thresholds by asset cluster.
- Map each regime to exposure caps (normal, reduced, protective).
- Implement automatic cap adjustment in risk overseer before hard kill-switch thresholds.
- Add risk state telemetry and alerts for regime transitions and cap changes.
Completion Checks:
- Stress drills confirm exposure caps scale down before hard kill-switch events.
- Risk cap transitions are deterministic and auditable.
- False-trigger rate remains within rollout acceptance limits.

#### Tier 1-C Order-Book Imbalance in Fast Loop
Tasks:
- Define canonical imbalance and queue-pressure features with schema versioning.
- Publish features via non-blocking snapshot path for Fast Loop consumption.
- Add feature quality guardrails (staleness, missing book levels, and source-quality flags).
- Validate inference-time impact on decision latency.
Completion Checks:
- Fast Loop p99 and p99.9 latency remain within release gates after feature enablement.
- Feature quality failures degrade safely without blocking execution.
- Shadow comparison shows non-regression on slippage and risk controls.

#### Tier 1-D Tightened Fast Loop Latency Discipline (8 ms Stretch)
Tasks:
- Instrument full decision path timing (p50, p95, p99, p99.9, and jitter).
- Enforce CI benchmark gate for execution-path pull requests.
- Validate degrade safeguard behavior when latency breaches > 10 ms.
- Add weekly latency regression report with owner and remediation actions.
Completion Checks:
- CI produces benchmark artifacts for replay and peak-load on every qualifying change.
- Degrade-path tests pass consistently under synthetic latency stress.
- Latency trend remains stable or improving over the Tier 1 window.

## Appendix A: Metrics and Promotion Gates
Acceptance Criteria:
- Core metrics include Sharpe, Sortino, Calmar, MDD, turnover, slippage, and capacity.
- Promotion gates include statistical significance thresholds vs baseline.
- Risk budget adherence is validated before production promotion.

## Appendix B: Data Quality SLAs
Acceptance Criteria:
- Feed uptime thresholds are defined and monitored.
- Max permissible latency is defined per data source.
- Data anomaly alerts trigger automatic fallback or pause.

## Appendix C: Indian Market Optimization Summary
| Area | Optimization |
| --- | --- |
| Data and Macros | RBI policy, FII/DII flows, domestic inflation metrics, FX reserves, and sentiment sources |
| Regime Detection | RBI inner and outer band monitoring with event calendar tracking |
| Volatility Modeling | GARCH quantiles tuned for NSE circuit breakers and FII movement patterns |
| Reward Emphasis | Sortino and Calmar weighting during panic regimes |
| Consensus Signal | Bayesian LSTAR or ESTAR hybrid with sentiment integration |
| Trading Execution | SAC-led ensemble for regime-switch robustness |
| Infrastructure | Mumbai VPS deployment with static IP whitelisting |
| Hardware Path | Optional FPGA acceleration for execution-critical components only |
| Operating Model | Interim single owner plus reviewer model; full pod model deferred for current cycle |
| Latency Discipline | Fast Loop stretch p99 <= 8 ms with mandatory degrade/bypass above 10 ms |
