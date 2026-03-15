# Phase 2: Analyst Board — Weekly Execution Plan

**Phase window**: Monday, March 9, 2026 to Sunday, April 13, 2026 (5 weeks)
**Alignment**: Multi-Agent Plan v1.3.7 §6 (Phase 2: Analyst Board)
**Team**: Owner + Partner (2-person team)
**Prerequisite**: Phase 1 Data Orchestration is complete and GO gate passed.

---

## Phase 2 Goal

Build the four modeling agents (Technical, Regime, Sentiment, Consensus) that consume Phase 1 data and produce actionable signals for the Strategic Executive (Phase 3). Each agent must have baseline models, validation evidence, and paper-trading readiness by end of phase.

---

## Phase 2 GO Benchmarks (from §16.1)

| Benchmark | Required Evidence |
|---|---|
| Baseline and ablation packs complete for each active model family | Model cards + backtest reports |
| Sentiment precision/recall thresholds pass | Classification report on Indian test set |
| Timestamp alignment tests pass | Alignment audit log |
| OOD and regime transition validations pass documented thresholds | Regime transition test report |

---

## Week 1 (March 9–15): Technical Agent

**Owner**: You | **Focus**: Time-series models for price prediction

### Day 1 (Mon): Setup & Data Pipeline
- Set up `src/agents/technical/` package skeleton.
- Define the Technical Agent's input contract (consuming Gold-tier OHLCV features from Phase 1).
- Implement data loader that reads from Silver/Gold DB for model training.
- Output: Package skeleton + data loader + input schema doc.

### Day 2 (Tue): ARIMA-LSTM Hybrid Baseline
- Implement ARIMA component for linear trend capture.
- Implement LSTM component for non-linear pattern capture.
- Build the hybrid combiner (ARIMA residuals → LSTM).
- Output: Working hybrid model with training pipeline.

### Day 3 (Wed): 2D CNN Price Pattern Model
- Implement 2D CNN that treats OHLCV windows as image-like inputs.
- Define the sliding window feature encoder.
- Train on historical data and log metrics.
- Output: CNN model + training metrics.

### Day 4 (Thu): GARCH Quantiles for VaR/ES
- Implement GARCH(1,1) model for volatility forecasting.
- Compute VaR and ES quantiles from GARCH output.
- Validate against historical volatility data.
- Output: GARCH model + VaR/ES pipeline.

### Day 5 (Fri): Backtesting & Ablation
- Run walk-forward backtest for each model (2019–present).
- Compute Sharpe, Sortino, MDD for each model family.
- Run ablation tests (remove each feature group, measure impact).
- Output: Backtest report + ablation results.

### Day 6 (Sat): Integration & Model Cards
- Integrate all three models into the Technical Agent's `predict()` interface.
- Write Model Card metadata for each model.
- Run leakage and timestamp alignment tests.
- Output: Unified Technical Agent + 3 model cards.

### Day 7 (Sun): Review & Handoff
- Review test results, fix issues.
- Document the Technical Agent's API for downstream consumption.
- Prepare Week 2 handoff.

---

## Week 2 (March 16–22): Regime Agent

**Owner**: Partner | **Focus**: Regime detection and transition modeling

### Day 1 (Mon): Regime Definitions & Data
- Define regime states: Bull, Bear, Sideways, Crisis, RBI-Band transitions.
- Map RBI inner/outer band zones to regime boundaries.
- Prepare historical regime-labeled training data.
- Output: Regime taxonomy + labeled dataset.

### Day 2 (Tue): Hidden Markov Model Baseline
- Implement HMM for regime state estimation.
- Train on historical OHLCV + macro indicators.
- Validate regime transitions against known events (COVID crash, 2022 volatility).
- Output: HMM regime model.

### Day 3 (Wed): PEARL Meta-Learning Model
- Implement PEARL or equivalent meta-learning approach for regime adaptation.
- Configure the 1–3 month adaptation window.
- Train on multi-regime historical slices.
- Output: PEARL model + adaptation pipeline.

### Day 4 (Thu): OOD & Alien-State Detection
- Implement statistical distance thresholds (Mahalanobis, KL-divergence) for novelty detection.
- Define alien-state flag trigger conditions.
- Map alien-state to staged de-risking levels (full risk → reduced → neutral/cash).
- Output: OOD detector + de-risking escalation logic.

### Day 5 (Fri): Transition Validation
- Test regime transitions on historical structural breaks (2008, 2013 taper, COVID, RBI policy shifts).
- Validate transition stability during high-volatility windows.
- Output: Transition test report.

### Day 6 (Sat): Calibration & Model Cards
- Calibrate confidence thresholds.
- Write Model Cards.
- Integrate into Regime Agent's unified `detect_regime()` interface.
- Output: Calibrated Regime Agent + model cards.

### Day 7 (Sun): Review & Handoff
- Review and fix issues.
- Prepare Week 3 handoff.

---

## Week 3 (March 23–29): Sentiment Agent

**Owner**: You | **Focus**: FinBERT fine-tuning and dual-speed sentiment

### Day 1 (Mon): FinBERT Setup & Indian Data
- Set up ProsusAI/finbert as base model.
- Prepare fine-tuning datasets: harixn/indian_news_sentiment, SEntFiN.
- Define precision/recall thresholds per sentiment class.
- Output: Fine-tuning pipeline setup.

### Day 2 (Tue): Fine-Tuning & Evaluation
- Fine-tune FinBERT on Indian market sentiment data.
- Evaluate precision, recall, F1 per class (positive/negative/neutral).
- Implement Bayesian priors for stability.
- Output: Fine-tuned model + classification report.

### Day 3 (Wed): Fast Lane (Intraday Sentiment)
- Build lightweight keyword-rule model for real-time headline scoring.
- Target: ≤ 100ms from headline arrival to score.
- Implement sentiment cache with TTL, confidence, and freshness tracking.
- Output: Fast lane scorer + cache layer.

### Day 4 (Thu): Slow Lane (Deep Sentiment)
- Build nightly deep sentiment pipeline using fine-tuned FinBERT.
- Implement daily aggregate sentiment variable (z_t) for regime logic.
- Add spam/adversarial text filtering and pump-and-dump detection.
- Output: Slow lane pipeline + z_t aggregator.

### Day 5 (Fri): Hinglish & Robustness
- Integrate code-mixed/Hinglish handling from Phase 1 textual services.
- Test sentiment accuracy on Hinglish and code-mixed samples.
- Add source deduplication and noise handling for social media.
- Output: Robustness test report.

### Day 6 (Sat): Cache Policy & Integration
- Implement deterministic cache policy: fresh → use, stale → downweight, expired → ignore.
- Implement cache-failure fallback: technical-only reduced-risk mode.
- Integrate fast and slow lanes into unified Sentiment Agent.
- Output: Integrated Sentiment Agent.

### Day 7 (Sun): Review & Handoff
- Review precision/recall against thresholds.
- Write Model Cards.
- Prepare Week 4 handoff.

---

## Week 4 (March 30 – April 5): Consensus Agent

**Owner**: Partner | **Focus**: LSTAR/ESTAR signal aggregation

### Day 1 (Mon): Consensus Framework Setup
- Set up `src/agents/consensus/` package.
- Define input contracts: Technical signals, Regime state, Sentiment z_t.
- Implement LSTAR (Logistic Smooth Transition) base model.
- Output: Package skeleton + LSTAR implementation.

### Day 2 (Tue): ESTAR & Bayesian Estimation
- Implement ESTAR (Exponential Smooth Transition) variant.
- Implement Bayesian estimation of smoothness and location parameters.
- Output: ESTAR model + Bayesian parameter estimation.

### Day 3 (Wed): Transition Function Design
- Build transition function incorporating: volatility, macro differentials, RBI signals, sentiment quantiles.
- Define weighted consensus with explicit safety bias toward protective signals.
- Output: Transition function + safety bias logic.

### Day 4 (Thu): Crisis Mode Routing
- Implement basic crisis-weighted routing (max 60–70% cap for crisis agent).
- Implement agent divergence detection (fundamental disagreement → staged risk reduction).
- Output: Crisis routing + divergence handler.

### Day 5 (Fri): Stability Testing
- Test consensus stability during high-volatility windows.
- Validate that text floods alone cannot flip portfolio direction without confirming technical/risk conditions.
- Run dual-loop boundary tests.
- Output: Stability test report.

### Day 6 (Sat): Integration & Model Cards
- Integrate into unified Consensus Agent.
- Write Model Cards.
- Output: Integrated Consensus Agent + model cards.

### Day 7 (Sun): Review & Handoff
- Review and fix issues.
- Prepare Week 5 handoff.

---

## Week 5 (April 6–12): Integration & Paper-Trading Validation

**Owner**: Both | **Focus**: End-to-end integration and paper-trading setup

### Day 1 (Mon): Integration Wiring
- Wire all four agents: Technical → Regime → Sentiment → Consensus.
- Verify data flows from Phase 1 Gold tier into Phase 2 agents.
- Output: End-to-end signal pipeline.

### Day 2 (Tue): Unified Backtest
- Run walk-forward backtest with all agents active (2019–present).
- Compute combined Sharpe, Sortino, MDD.
- Compare vs individual agent baselines.
- Output: Combined backtest report.

### Day 3 (Wed): Timestamp & Leakage Audit
- Run full timestamp alignment audit across all agents.
- Verify no leakage in the combined pipeline.
- Output: Alignment audit log.

### Day 4 (Thu): Paper-Trading Setup
- Configure paper-trading environment with live Phase 1 data feeds.
- Deploy all four agents in paper-trading mode.
- Output: Paper-trading environment running.

### Day 5 (Fri): Phase 2 Gate Evidence
- Collect all Phase 2 GO benchmark evidence.
- Compile model cards, backtest reports, ablation packs, classification reports.
- Output: Phase 2 gate evidence package.

### Day 6 (Sat): GO/NO-GO Assessment
- Evaluate all Phase 2 benchmarks from §16.1.
- Document any at-risk items with mitigation plans.
- Output: GO/NO-GO assessment.

### Day 7 (Sun): Handoff & Phase 3 Planning
- Package Phase 2 delivery.
- Begin Phase 3 (Strategic Executive) planning.
- Output: Phase 2 handoff package.

---

## Phase 2 Exit Criteria

| Criterion | Evidence Required |
|---|---|
| Baseline + ablation packs for each model family | Backtest reports + ablation results |
| Sentiment precision/recall thresholds met | Classification report on Indian test set |
| OOD and regime transition validations pass | Transition test report + OOD detection log |
| Timestamp alignment tests pass | Alignment audit log |
| Dual-loop boundary tests pass | Boundary test report |
| All model cards complete | Model card metadata files |
| Paper-trading environment running | Deployment confirmation |
