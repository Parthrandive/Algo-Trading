# SKILL: Analyst Board — Phase 2 Modeling
**Project:** Multi-Agent AI Trading System — Indian Market
**Applies To:** Technical Agent · Regime Agent · Sentiment Agent · Consensus Agent
**Version:** aligned with Plan v1.3.7

---

## PURPOSE
This skill governs how all Phase 2 analyst models are trained, validated, promoted, and consumed by downstream execution. All model families listed here are **reference implementations** — any equivalent alternative that passes the same validation gates is acceptable.

---

## 1. TECHNICAL AGENT

### 1.1 Model Families (Reference, Not Mandatory)
| Family | Role | Validation Metric |
|--------|------|------------------|
| ARIMA-LSTM Hybrid | Price direction / level forecasting | Directional accuracy, MAE vs naive baseline |
| 2D CNN | Pattern recognition on price image representations | Precision / recall on target labels |
| GARCH Quantile | VaR and ES estimation | Coverage tests (Kupiec / Christoffersen) |

### 1.2 Mandatory Validation Steps
1. **Baseline**: naive persistence and simple moving-average benchmark must exist for every model
2. **Ablation**: each feature group removed independently; performance delta logged
3. **Walk-forward CV**: non-overlapping folds aligned to calendar quarters; no shuffling
4. **Leakage test**: pass rate = 100 % (see Data Orchestration Skill)
5. **Backtest period**: 2018/2019 to present; must include COVID crash (Feb–Apr 2020), budget cycles, Indian general elections, Oct–Nov 2022 volatility window

### 1.3 Training Cadence
- Nightly incremental updates via TTM or equivalent lightweight adapter
- Full retraining triggered by: regime structural break, data schema change, or champion-challenger demotion
- All training runs are reproducible from raw data via versioned pipeline

### 1.4 Phase Gate (Phase 2 GO Criterion)
- Baseline and ablation packs complete for every active model family
- No missing baseline evidence; all model-data alignment defects resolved before promotion

---

## 2. REGIME AGENT

### 2.1 Model (Reference)
- PEARL or equivalent meta-learning model
- Must capture: trending, mean-reverting, high-volatility, low-volatility, RBI inner band, RBI outer band regimes

### 2.2 Regime Definitions
| Regime | Key Signals | Risk Action |
|--------|-------------|-------------|
| Normal / Trending | Stable vol, directional macro | Full risk budget |
| High Volatility | Realised vol >> GARCH quantile | Reduce exposure caps |
| RBI Inner Band | INR within comfort zone | Standard operation |
| RBI Outer Band | INR approaching intervention zone | Heightened hedge trigger |
| Alien / OOD | Statistical distance > threshold | Staged de-risking |
| Crisis | Multi-condition confirmation (see Section 2.4) | Crisis mode protocol |

### 2.3 OOD / Alien-State Detection
- Use explicit statistical distance threshold (e.g., Mahalanobis distance on feature space; KL divergence on regime posterior)
- Thresholds must be documented and versioned
- On alien-state flag:
  1. Step 1: Reduce to 50 % of normal risk budget
  2. Step 2 (if persists > N ticks): Reduce to 25 %
  3. Step 3 (if persists > M ticks): Move to neutral / cash
- Document N and M in config; log every transition

### 2.4 Crisis Entry Rules (Multi-Condition Required)
All three conditions must be true before entering crisis mode:
1. Realised volatility breaks configured sigma threshold
2. Liquidity deterioration: impact cost > 2× normal
3. Agent confidence threshold drops below configured floor

- Conditions must persist for configured ticks/seconds before activation (hysteresis)
- Crisis mode has max-duration expiry; auto-reverts unless conditions revalidated
- Cooldown window enforced after crisis exit to prevent rapid re-entry

### 2.5 Stress Validation (Mandatory Scenarios)
- 2008 Global Financial Crisis data
- 2013 Taper Tantrum regime shocks
- COVID crash window (Feb–Apr 2020)
- RBI policy discontinuities / surprise rate hikes
- INR flash move scenarios
- Structural breaks (correlation inversion, liquidity vacuum)

### 2.6 Calibration & Stability
- Calibration tests run after every retrain
- Transition probability stability across rolling 3-month windows
- Regime frequency audit: flag if any regime is never or always active for > 30 consecutive days

---

## 3. SENTIMENT AGENT

### 3.1 Base Model
- ProsusAI/finbert or documented equivalent
- Fine-tuning datasets: `harixn/indian_news_sentiment`, SEntFiN, and curated internal corpus
- Bayesian priors or regularisation must be defined to prevent catastrophic forgetting

### 3.2 Sentiment as Threshold Variable
- Daily sentiment aggregate forms `z_t` (standardised sentiment score)
- `z_t` is consumed by Regime Agent as a transition modifier
- `z_t` computation is deterministic and versioned

### 3.3 Precision / Recall Requirements
| Class | Min Precision | Min Recall |
|-------|--------------|-----------|
| Positive | Defined at design time | Defined at design time |
| Negative | Defined at design time | Defined at design time |
| Neutral | — | — |
- Thresholds must be documented; breach triggers model demotion flag

### 3.4 Cache Architecture (Non-Negotiable)
- All Fast Loop consumers read sentiment via O(1) cache lookup (Redis or equivalent)
- **Never** perform fresh NLP inference inside the execution-critical path
- Cache record schema: `{score, confidence, source, timestamp, TTL, freshness_flag}`
- Intraday fast-lane target: <= 100 ms from headline arrival to cache write (outside execution path)
- Weekly re-fine-tuning validated against SEntFiN or equivalent holdout

### 3.5 Robustness Controls
- Spam / adversarial text filtered before scoring
- Source deduplication enforced
- Code-mixed (Hinglish) handling strategy documented
- Pump-and-dump / slang-scam detection: match pattern → sentiment downgraded to `neutral`, event logged as `MANIPULATION_DETECTED`

### 3.6 Timestamp Alignment
- Every sentiment record carries `headline_timestamp` and `score_timestamp`
- Alignment test: `score_timestamp` must be >= `headline_timestamp`; no backdating permitted
- Pass rate = 100 % before model promotion

---

## 4. CONSENSUS AGENT

### 4.1 Aggregation Model (Reference)
- Bayesian LSTAR (Logistic Smooth Transition Auto-Regression) or ESTAR hybrid
- Logistic and exponential transition functions both specified
- Smoothness (γ) and location (c) parameters estimated via Bayesian posterior

### 4.2 Transition Function Inputs
- Realised volatility level
- Macro differentials (India–US spread, FII flow direction)
- RBI stance signal
- Sentiment quantile (`z_t`)
- Regime posterior from Regime Agent

### 4.3 Consensus Modes
| Mode | Description | Activation |
|------|-------------|-----------|
| Default Weighted | Weighted signal average with safety bias toward protective signals | Always active |
| Crisis-Weighted | Crisis agent weight capped at max 60–70 % | Tier 2; after 3 months paper trading |
| Agent Divergence Hold | Freeze new positions when agents fundamentally disagree | Immediate on divergence flag |
| Full Winner-Takes-All | NOT PERMITTED in current cycle | Deferred |

### 4.4 Agent Divergence Protocol
- Divergence = fundamental disagreement between >= 2 major agents on direction
- On divergence flag:
  1. Emit `AGENT_DIVERGENCE` event
  2. Freeze new position opens for configured hold duration
  3. Staged re-risking after alignment recovery (require >= 2 consecutive aligned signals)
- Log divergence event with agent outputs and timestamp

### 4.5 Stability Testing
- Run consensus under high-volatility replay windows
- Snapback test: measure ticks-to-clip after flash shock; log; do NOT auto-tune smoothing parameters in current cycle (manual review only)
- Shadow A/B for major Consensus changes: Tier 2 (after 3 months paper trading); strongly recommended even before Tier 2 activation

### 4.6 Output Schema
```json
{
  "consensus_signal": "BUY | SELL | NEUTRAL | HOLD",
  "confidence": 0.0–1.0,
  "regime": "string",
  "dominant_agent": "string",
  "divergence_flag": true/false,
  "mode": "default | crisis | divergence_hold",
  "as_of": "ISO8601 timestamp",
  "version": "string"
}
```

---

## ANALYST BOARD AGENT CHECKLIST
- [ ] Baseline and ablation evidence complete for all active model families
- [ ] OOD thresholds documented and tested against structural break scenarios
- [ ] Sentiment precision/recall thresholds met; timestamp alignment = 100 %
- [ ] Sentiment cache writes confirmed; no NLP inference in execution path
- [ ] Manipulation detection active and logging events
- [ ] Consensus output schema versioned; divergence protocol tested
- [ ] All stress scenarios in Section 2.5 pass signed review
- [ ] No fixed Sharpe/accuracy uplift claims in promotion artifacts
