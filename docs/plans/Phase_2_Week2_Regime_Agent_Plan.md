# Week 2 Plan: Regime Agent (Phase 2 — Analyst Board)

**Week window**: Monday, March 16, 2026 to Sunday, March 22, 2026
**Alignment**: Phase 2 Analyst Board §6.2 (Regime Agent)
**Owner**: Partner
**Branch**: `feature/regime-agent`

---

## Week 2 Goal

Build the Regime Agent that detects and models market regimes and transitions. Deliver baseline models (Hidden Markov Model, PEARL Meta-Learning), implement Out-Of-Distribution (OOD) and alien-state detection for de-risking, validate transitions, and produce model cards.

---

## Input Contract (From Phase 1 & Technical Agent)

The Regime Agent reads from the Silver/Gold DB and Technical Agent:
- **OHLCV Bars & Macro Indicators**: Historical and live data.
- **Technical Signals**: Price forecasts, volatility estimates, and VaR/ES quantiles from the Technical Agent.
- **RBI Zones**: Inner/outer band zones mapped to regime boundaries.

## Output Contract (To Downstream Agents)

The Regime Agent produces:
- `regime_state`: Current detected regime (Bull, Bear, Sideways, Crisis, RBI-Band transition, Alien).
- `transition_probability`: Likelihood of shifting to a new regime.
- `confidence`: Model confidence score (0.0–1.0).
- `risk_level`: Staged de-risking level (e.g., full risk, reduced risk, neutral/cash) based on alien-state triggers.
- `model_id`: Which model produced this prediction.

---

## Daily Execution Plan

### Day 1: Monday, March 16 — Regime Definitions & Data Preparation

- Define regime states:
  - Bull, Bear, Sideways, Crisis, RBI-Band transitions.
- Map RBI inner/outer band zones to regime boundaries.
- Prepare historical regime-labeled training data from Gold-tier.
- Create `src/agents/regime/` package with:
  - `__init__.py`
  - `regime_agent.py` — agent skeleton with `detect_regime()` interface
  - `data_loader.py` — regime-specific data preparation
  - `schemas.py` — output schema (`RegimePrediction`)

**Output**: Regime taxonomy + labeled dataset + package skeleton.

---

### Day 2: Tuesday, March 17 — Hidden Markov Model (HMM) Baseline

- Implement `src/agents/regime/models/hmm_regime.py`:
  - HMM for regime state estimation.
- Train on historical OHLCV + macro indicators.
- Validate regime transitions against known events (e.g., COVID crash, 2022 volatility).
- Write Day 2 validation tests: `tests/test_regime_day2.py`.

**Output**: HMM regime model + validation against known historical events.

---

### Day 3: Wednesday, March 18 — PEARL Meta-Learning Model

- Implement `src/agents/regime/models/pearl_meta.py`:
  - PEARL (or equivalent meta-learning approach) for regime adaptation.
  - Configure the 1–3 month adaptation window.
- Train on multi-regime historical slices.
- Write Day 3 tests validating adaptation capabilities.

**Output**: PEARL model + adaptation pipeline.

---

### Day 4: Thursday, March 19 — OOD & Alien-State Detection

- Implement `src/agents/regime/models/ood_detector.py`:
  - Statistical distance thresholds (Mahalanobis, KL-divergence) for novelty detection.
- Define alien-state flag trigger conditions.
- Map alien-state to staged de-risking levels:
  - normal -> full risk
  - warning -> reduced risk
  - alien/crisis -> neutral/cash
- Write Day 4 tests testing distance boundaries.

**Output**: OOD detector + de-risking escalation logic.

---

### Day 5: Friday, March 20 — Transition Validation

- Test regime transitions on historical structural breaks:
  - 2008 Financial Crisis
  - 2013 Taper Tantrum
  - 2020 COVID Crash
  - Recent RBI policy shifts
- Validate transition stability during high-volatility windows.
- Generate transition test report: `docs/reports/regime_transition_test_report.md`

**Output**: Transition test report.

---

### Day 6: Saturday, March 21 — Calibration & Model Cards

- Calibrate confidence thresholds for all models.
- Integrate into Regime Agent's unified `detect_regime()` interface.
  - Model selection / ensemble logic for final regime output.
- Write Model Card metadata files (JSON):
  - `data/models/hmm_regime/model_card.json`
  - `data/models/pearl_meta/model_card.json`
  - `data/models/ood_detector/model_card.json`
- Ensure model cards include model_id, logic, thresholds, and performance metrics.

**Output**: Calibrated Regime Agent + integrated pipeline + model cards.

---

### Day 7: Sunday, March 22 — Review, Fix & Handoff

- Run full test suite for Regime Agent: `pytest tests/test_regime_*.py -v`
- Review and fix any failing tests or threshold issues.
- Prepare handoff document for Week 3 (Sentiment Agent):
  - Document what Regime Agent produces (especially `regime_state` and `risk_level`).
  - Define the interface for Sentiment and Consensus agents.
- Commit and push all Day 7 changes.

**Output**: All tests passing + Week 3 handoff doc.

---

## Week 2 Exit Criteria

| Criterion | Evidence Required |
|---|---|
| Regime Agent package created | `src/agents/regime/` exists with all modules |
| Labeling & Taxonomy complete | Documented regime definitions and generated labeled dataset |
| HMM Baseline model trainable | Day 2 tests pass, validates against historical events |
| PEARL Meta-learning model trainable | Day 3 tests pass |
| OOD / Alien-state detection active | Day 4 tests pass, thresholds defined |
| Transition validation complete | Transition test report covering structural breaks |
| All model cards complete | JSON metadata files in `data/models/` |
| All tests pass | `pytest tests/test_regime_*.py` green |

## Dependencies

### Python Packages Required
- `hmmlearn` or `pomegranate` — Hidden Markov Models
- `torch` — PEARL / Meta-learning
- `scipy`, `scikit-learn` — statistical distances (Mahalanobis, KL), preprocessing
- `pandas`, `numpy` — data manipulation
- `pydantic` — schemas (already installed)
