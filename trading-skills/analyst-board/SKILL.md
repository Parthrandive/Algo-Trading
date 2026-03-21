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

## 5. WORKED EXAMPLES

These procedures were validated on real tasks. Follow the exact steps when performing similar work.

### 5.1 Training All Technical Agent Models (Single Symbol)

**Validated:** March 2026 — trained on TATASTEEL.NS, RELIANCE.NS, INFY.NS, USDINR=X, LT.NS, POWERGRID.NS

**Model artifacts directory:** `data/models/{model_name}/` or `data/reports/training_runs/{run_id}/`

**Steps:**
1. **ARIMA-LSTM Hybrid** (price direction forecasting):
   ```bash
   python scripts/train_arima_lstm.py --symbol RELIANCE.NS --interval 1h --epochs 50 --seed 42
   ```
   Output: `data/models/arima_lstm/training_meta.json` + `lstm_weights.pt`
   Pass criteria: `val_loss` < `train_loss` (no overfitting), `epochs_run` > 5 (not trivially early-stopped)

2. **CNN Pattern Classifier** (directional pattern recognition):
   ```bash
   python scripts/train_cnn_pattern.py --symbol RELIANCE.NS --interval 1h --epochs 20 --seed 42
   ```
   Output: `data/models/cnn_pattern/training_meta.json` + `cnn_weights.pt`
   Pass criteria: `val_acc` > 33% (better than random 3-class guess)

3. **GARCH VaR** (volatility / risk estimation):
   ```bash
   python scripts/train_garch_var.py --symbol RELIANCE.NS --interval 1d --run-backtest --seed 42
   ```
   Output: `data/models/garch_var/training_meta.json`
   Pass criteria: `convergence_flag` = 0 and Kupiec test p-value > 0.05

4. **Verify all** — check that `training_meta.json` exists for each model:
   ```bash
   ls data/models/*/training_meta.json
   ```

**Gotcha — Forex symbols:** For USDINR=X, use `--dist t` for GARCH (auto-applied) and the CNN auto-adjusts `neutral-threshold` to 0.0002.

### 5.2 Training Regime-Aware Multi-Model

**Validated:** March 2026 — regime-aware models trained with walk-forward CV on multiple symbols.

```bash
python scripts/train_regime_aware.py --symbol RELIANCE.NS --interval 1h --seed 42
```
Output: `data/reports/training_runs_full150/regime_aware_{interval}_{timestamp}/`
- Contains: `train.log`, model weights, confusion matrices, fold-level metrics
- Pass criteria: balanced accuracy > baseline (check `train.log` summary)

**Run when:** After changing `features.py`, `data_loader.py`, regime definitions, or macro columns.

### 5.3 Running the Full Test Suite

**Test files by agent:**
| Agent | Test files | Run command |
|---|---|---|
| Technical | `test_technical_day1.py` → `test_technical_day6.py` | `pytest tests/test_technical_day*.py -v` |
| Regime | `test_regime_day2.py` → `test_regime_day7.py` | `pytest tests/test_regime_day*.py -v` |
| Textual/Sentiment | `test_textual_day1.py` → `test_textual_day6.py` | `pytest tests/test_textual_day*.py -v` |
| Preprocessing | `test_leakage.py`, `test_loaders.py` | `pytest tests/agents/preprocessing/ -v` |
| Macro | `test_pipeline.py`, `test_parsers.py`, `test_freshness.py` | `pytest tests/agents/macro/ -v` |

**Run everything:**
```bash
pytest tests/ -v --tb=short 2>&1 | tail -20
```
Pass criteria: All tests pass (exit code 0). Any failure blocks model promotion.

**Run leakage test specifically** (mandatory before any feature change):
```bash
pytest tests/agents/preprocessing/test_leakage.py -v
```
Pass criteria: 100% pass rate. Any leakage detection = **hard block** on promotion.

### 5.4 Adding a New Model Family

**Steps:**
1. Create model class in `src/agents/technical/models/new_model.py` with `fit()`, `predict()`, `save()`, `load()`
2. Create training script: `scripts/train_new_model.py` following the pattern in existing scripts:
   - Use `DataLoader` + `engineer_features()` for data
   - 80/20 chronological split (no shuffling)
   - Early stopping with patience
   - Save `training_meta.json` with hyperparameters and metrics
3. Add the model to the `§1.1 Model Families` table above with its validation metric
4. Create baseline comparison: train naive persistence + SMA benchmark on same data
5. Run ablation: remove each feature group one at a time, log delta
6. Run leakage test: `pytest tests/agents/preprocessing/test_leakage.py -v`
7. Validate on crisis windows: ensure backtest period includes Feb–Apr 2020, Oct–Nov 2022
8. Generate model card: record all metrics, baselines, and ablation results in `training_meta.json`

---

## 6. PERFORMANCE BASELINE (Skill vs No-Skill)

Measured on: **March 2026 — Multi-symbol model training task**

### Without this skill
```
Task: "Train all models for RELIANCE.NS"
- Agent searches codebase for training scripts → finds 6 files, unsure which to run
- Agent guesses wrong argument names (--ticker instead of --symbol)
- Agent runs CNN with default neutral_threshold on forex → bad class distribution
- Agent doesn't run leakage test after feature changes
- Agent doesn't check training_meta.json for required fields
- Agent skips GARCH backtest (doesn't know --run-backtest flag exists)

Result:
  Scripts found: 6 (confused by train_models.py vs individual scripts)
  Training runs with correct args: 1 out of 3
  Validation performed: none
  Gotchas hit: forex threshold, missing --run-backtest
  Leakage test: skipped
```

### With this skill (§5.1 procedure)
```
Task: "Train all models for RELIANCE.NS"
- Agent reads §5.1 → exact commands for all 3 models
- Agent reads Gotcha → auto-adjusts for forex symbols
- Agent runs all 3 training scripts with correct args
- Agent verifies training_meta.json exists with required fields
- Agent runs leakage test (§5.3)

Result:
  Commands run: 3 (one per model) + 1 validation
  Training runs with correct args: 3 out of 3
  Validation: training_meta.json checked, leakage test passed
  Gotchas avoided: forex threshold, GARCH distribution
```

### Success criteria
| Metric | Without Skill | With Skill | Target |
|---|---|---|---|
| Correct training commands on first try | 1/3 | 3/3 | 3/3 |
| Post-training validation | None | Meta JSON + leakage test | Always |
| Forex-specific gotchas avoided | 0 | 2 (threshold + distribution) | All documented |
| Test suite run after changes | No | Yes (`pytest tests/skills/`) | Always |
| training_meta.json schema valid | Unknown | Verified (30/30 tests pass) | Always |

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
