# Multi-Agent AI Trading System
Updated Integrated Plan for Indian Market (USD/INR + NSE Stocks & Indices)
Version 1.2.1 | February 2026

## 0. Document Control
Purpose: maintain versioning, accountability, and auditability.
Acceptance Criteria:
- Single owner and reviewer list are defined.
- Versioning and changelog are maintained in this document.
- Each model and data artifact is traceable to a plan version.
- Version 1.2 change set captures dual-loop execution, sentiment cache decoupling, crisis override voting, and impossible-scenario stress tests.
- Version 1.2.1 applies minimum-safe scope: ship dual-loop, sentiment cache, basic crisis mode, and impossible stress testing; defer aggressive alien-state automation, snapback smoothing tuning, and full winner-takes-all automation.

## 1. Executive Summary
This updated plan refines the existing multi-agent trading system for Indian markets by adding concrete governance, execution, and validation controls. It preserves the core multi-agent architecture while tightening scope, compliance, data integrity, and rollout safety. The plan incorporates a dedicated Sentiment Agent in Phase 2 using a fine-tuned FinBERT model to improve event-driven robustness. The system targets early 2026 deployment, a simulated Sharpe ratio of 1.8 to 2.5+, and a 10 to 20 percent improvement in event-driven accuracy attributable to sentiment integration.

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
- Fast Loop target is p99 <= 10 ms and processes only technical microstructure, order-imbalance features, and cache reads.
- Slow Loop supports heavier inference (100 to 500 ms typical) and is asynchronous, never blocking order placement.
- Volatility bypass thresholds use hysteresis and cooldown windows to prevent rapid mode flapping.
- Deployment topology (Mumbai-based low-latency VPS) is specified.
- Minimum-safe release profile is explicit: advanced overrides are disabled by default and only basic crisis-weighted controls are active.

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
- Basic crisis mode uses crisis-weighted routing with capped dominance (max 60 to 70 percent for the crisis agent), not full winner-takes-all.
- Agent divergence (fundamental disagreement between agents) is modeled as a dedicated regime signal that triggers staged risk reduction or temporary freeze.
- Full winner-takes-all crisis automation is deferred until false-positive and stability thresholds are met in shadow mode.
- Snapback tests are retained for measurement; automatic Bayesian smoothing retuning from snapback output is deferred.

## 7. Phase 3: Strategic Executive (Trader Ensemble)
RL ensemble for trade execution decisions.

### 7.1 Ensemble Composition
Acceptance Criteria:
- Ensemble includes SAC, PPO, and TD3 or justified alternatives.
- Each policy has defined training cadence and evaluation metrics.

### 7.2 Decision Mechanism
Acceptance Criteria:
- Decision policy uses a maximum entropy framework.
- Multi-threshold genetic algorithm is specified for action selection.
- Genetic algorithm evolution and search run offline in training and calibration.
- Live execution uses teacher-student distillation, where the student policy mimics teacher decisions under strict latency budgets.
- Student promotion requires teacher-student agreement thresholds on crisis slices, not only average-day performance.
- Offline teacher monitoring remains active in production analytics, and student drift beyond threshold triggers automatic demotion.

### 7.3 Observation Space
Acceptance Criteria:
- Observation space includes outputs from Technical, Regime, Sentiment, and Consensus agents.
- Features include regime probabilities, VaR and ES levels, macro differentials, and sentiment scores.
- Observation schema is versioned and validated.

### 7.4 Reward and Utility Functions
Acceptance Criteria:
- Reward library includes RA-DRL composite, Sharpe, Sortino, Calmar, Omega, and Kelly variants.
- Dynamic weighting is tied to regime state and sentiment signals.
- Downside penalties increase during RBI outer band or negative sentiment spikes.
- Formulas are documented in an appendix or code reference.

### 7.5 Deliberation Process
Acceptance Criteria:
- Critical-path thinker logic is lightweight and deterministic; heavy reasoning remains in the Slow Loop.
- Simulations and policy search are removed from the execution-critical path and run asynchronously.
- Execution consumes the latest validated policy snapshot, with scheduled refresh (e.g., every 10 minutes) or earlier on async completion.
- Deliberation bypass is enabled under high-volatility triggers and does not block order decisions.
- Trade rejection is tied to risk-control breaches, stale-data TTL breaches, or latency-cap breaches, not pending simulation completion.
- Fast Loop peak-load decision stack target is hard-capped with p99 <= 10 ms for in-process compute stages.
- Slow Loop latency (100 to 500 ms typical) is monitored separately and cannot directly bypass Fast Loop safety gates.

### 7.6 Safe RL Training and Deployment
Acceptance Criteria:
- Online learning is gated behind offline training and paper trading validation.
- Exploration is constrained by risk limits and hard safety filters.
- Promotion to live trading requires risk committee approval and rollback readiness.

## 8. Portfolio Construction and Risk Budgeting
Dedicated allocation and exposure management.
Acceptance Criteria:
- Risk budgets per asset class and sector are defined.
- Correlation and concentration limits are enforced.
- Position sizing ties to forecast uncertainty and liquidity.
- Capacity ceiling is defined: initial live notional ₹5–10 Cr, with scale to ₹25 Cr only after 6 months of live validation; stress-tested at 1x, 2x, and 3x capacity, with a hard cap at 2x if impact exceeds 0.25 percent at 3x.
- Max participation limits are documented and enforced at execution time.
- Downside risk emphasis is enforced during panic regimes.

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

## 10. Execution and Market Impact
Order routing and slippage management tailored to NSE.
Acceptance Criteria:
- Execution supports partial fills, pacing, and participation limits.
- Slippage model is validated against intraday data.
- Circuit breaker and halt handling is implemented and tested.
- Execution metrics include fill rate, slippage vs model, and latency.
- Market impact model is defined and linked to capacity assumptions.
- Real-time routing health checks are enforced, with automatic degradation to reduce-only or close-only on execution infrastructure failure.

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
- Data provenance reliability scoring dynamically adjusts position sizing instead of fixed haircut rules.
- OOD or alien-state triggers enforce staged de-risking before full neutralization unless hard limits are breached.
- Crisis taxonomy explicitly defines full crisis, agent divergence, and slow-crash or freeze states, each mapped to mode transitions and overrides.
- Sentiment-to-price mismatch circuit breaker can downgrade sentiment to neutral during suspected manipulation or panic slang bursts.
- Crisis entry requires multi-condition confirmation (volatility break, liquidity deterioration, and agent confidence threshold).
- Crisis triggers must persist for configured ticks or seconds before activation, with explicit cooldown and hysteresis rules.
- Crisis mode has max-duration expiry and auto-reverts unless revalidated by trigger conditions.
- Agent divergence has explicit neutral-hold duration and staged re-risking rules after alignment recovery.

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

## 13. MLOps and Model Governance
Versioning, reproducibility, and controlled deployment.
Acceptance Criteria:
- Model registry includes data snapshot and code hash.
- Champion-challenger promotion gates are codified.
- Rollback policy exists and is tested.
- Training pipelines are reproducible from raw data.
- Human oversight gates are defined, including review cadence and approval authority.
- Material strategy changes (latency bypass, dual-speed sentiment, novelty safeguards) require ablation evidence before promotion.

## 14. Rollout Strategy and Live Monitoring
Controlled deployment with shadow testing and gating.
Acceptance Criteria:
- Shadow mode runs for a fixed minimum window.
- Promotion requires challenger outperforming champion on risk-adjusted metrics.
- Live A/B is optional and gated by risk approval.
- Monitoring dashboards include PnL attribution, drift, and risk limits.
- Promotion gates require no regression in Sharpe, drawdown, and slippage, plus acceptable rates for mode switching and false kill-switch events.
- Crisis logic and override routing must complete shadow-mode validation before live activation.
- Override rollout is sequential: enable one override at a time, validate, then promote the next.

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

## 16. Milestones and Deliverables
Phased execution with measurable outcomes.
Acceptance Criteria:
- Each phase has exit criteria, owners, and timelines.
- Go-live criteria include performance, stability, and compliance readiness.

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


## 18. Next Iteration Enhancements (Nice-to-have)
Planned upgrades to be prioritized after initial production stability.
Acceptance Criteria:
- Backlog is reviewed quarterly and scoped only after six months of stable live operation.
- Any new enhancements require documented risk review and rollback plans before deployment.

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
