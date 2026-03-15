# Week 1 Plan: Technical Agent (Phase 2 — Analyst Board)

**Week window**: Monday, March 9, 2026 to Sunday, March 15, 2026
**Alignment**: Phase 2 Analyst Board §6.1 (Technical Agent)
**Owner**: You
**Branch**: `feature1`

---

## Week 1 Goal

Build the Technical Agent that consumes Gold-tier OHLCV data from Phase 1 and produces price forecasts, volatility estimates, and VaR/ES quantiles. Deliver baseline models (ARIMA-LSTM hybrid, 2D CNN, GARCH), run walk-forward backtests, and produce ablation evidence and model cards.

---

## Input Contract (From Phase 1)

The Technical Agent reads from the Silver/Gold DB via `SilverDBRecorder`:
- **OHLCV Bars** (`Bar` schema): `symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`, `interval`
- **Macro Indicators** (`MacroIndicator` schema): `indicator_name`, `value`, `release_date`, `is_stale`
- **Corporate Actions** (`CorporateAction` schema): `action_type`, `ratio`, `value`, `ex_date`

All records carry provenance (`source_type`, `ingestion_timestamp_utc/ist`, `schema_version`, `quality_status`).

## Output Contract (To Downstream Agents)

The Technical Agent produces:
- `price_forecast`: Predicted direction and magnitude for the next horizon (hourly).
- `volatility_estimate`: GARCH-based realized and forecasted volatility.
- `var_es`: Value-at-Risk and Expected Shortfall quantiles (95%, 99%).
- `confidence`: Model confidence score (0.0–1.0).
- `model_id`: Which model produced this prediction.

---

## Daily Execution Plan

### Day 1: Monday, March 9 — Setup, Schema, and Data Loader

- Create `src/agents/technical/` package with:
  - `__init__.py` — package init
  - `technical_agent.py` — agent skeleton with `predict()` interface
  - `data_loader.py` — loads OHLCV bars from DB/Parquet for training
  - `schemas.py` — output schema (`TechnicalPrediction`, `VolatilityEstimate`, `VaRES`)
- Define the Technical Agent's prediction output schema (Pydantic model):
  ```python
  class TechnicalPrediction(BaseModel):
      symbol: str
      timestamp: datetime
      price_forecast: float
      direction: Literal["up", "down", "neutral"]
      volatility_estimate: float
      var_95: float
      var_99: float
      es_95: float
      es_99: float
      confidence: float  # 0.0–1.0
      model_id: str
      schema_version: str = "1.0"
  ```
- Implement `DataLoader` class that reads historical OHLCV from DB and returns pandas DataFrames.
- Write Day 1 tests: `tests/test_technical_day1.py`
  - Schema validation tests for `TechnicalPrediction`
  - DataLoader smoke test (reads from test DB or sample CSV)

**Implementation snapshot targets:**
- Package skeleton: `src/agents/technical/`
- Output schema: `src/agents/technical/schemas.py`
- Data loader: `src/agents/technical/data_loader.py`
- Day 1 tests: `tests/test_technical_day1.py`

**Output**: Technical Agent skeleton + schema + data loader + Day 1 tests passing.

---

### Day 2: Tuesday, March 10 — ARIMA-LSTM Hybrid Model

- Implement `src/agents/technical/models/arima_lstm.py`:
  - ARIMA component: fit on historical close prices, capture linear trend.
  - LSTM component: train on ARIMA residuals to capture non-linear patterns.
  - Hybrid combiner: ARIMA prediction + LSTM residual prediction = final forecast.
- Use `statsmodels` for ARIMA and `torch` (or `tensorflow`) for LSTM.
- Implement training pipeline:
  - Walk-forward split: 80% train / 20% validation.
  - Feature engineering: lag features, rolling means, RSI, MACD.
- Save model artifacts (weights + hyperparameters) to `data/models/arima_lstm/`.
- Write Day 2 tests: `tests/test_technical_day2.py`
  - Model trains without error on sample data.
  - Output shape and dtype validation.
  - Prediction falls within reasonable price range.

**Implementation snapshot targets:**
- ARIMA-LSTM model: `src/agents/technical/models/arima_lstm.py`
- Feature engineering utils: `src/agents/technical/features.py`
- Day 2 tests: `tests/test_technical_day2.py`

**Output**: Working ARIMA-LSTM hybrid with training pipeline.

---

### Day 3: Wednesday, March 11 — 2D CNN Price Pattern Model

- Implement `src/agents/technical/models/cnn_pattern.py`:
  - Treat OHLCV windows as 2D "images" (rows = time steps, columns = OHLCV features).
  - 2D CNN architecture: Conv2D → BatchNorm → ReLU → Pool → FC → output.
  - Sliding window encoder: last N bars → (N × 5) matrix → prediction.
- Configure:
  - Window size: 20 bars (hourly) = ~1 trading day.
  - Target: next-bar close direction (up/down/neutral classification).
- Train on historical data, log accuracy and loss curves.
- Write Day 3 tests: `tests/test_technical_day3.py`
  - CNN forward pass produces correct output shape.
  - Training loop runs for 2 epochs without error.
  - Classification output sums to 1.0 (softmax check).

**Implementation snapshot targets:**
- CNN model: `src/agents/technical/models/cnn_pattern.py`
- Day 3 tests: `tests/test_technical_day3.py`

**Output**: 2D CNN model + training pipeline + tests.

---

### Day 4: Thursday, March 12 — GARCH Quantiles for VaR/ES

- Implement `src/agents/technical/models/garch_var.py`:
  - GARCH(1,1) model using `arch` library for volatility forecasting.
  - VaR computation at 95% and 99% confidence levels (parametric and historical).
  - Expected Shortfall (CVaR) computation from GARCH distribution tails.
- Implement rolling volatility estimation:
  - Fit GARCH on rolling 252-bar (trading day equivalent) windows.
  - Produce 1-step-ahead volatility forecast.
- Validate against historical drawdowns:
  - Check VaR breach frequency matches expected confidence level.
  - Backtest VaR accuracy with Kupiec POF test.
- Write Day 4 tests: `tests/test_technical_day4.py`
  - GARCH fits without convergence failure on sample data.
  - VaR values are negative (loss metric).
  - ES values are more extreme than corresponding VaR.
  - Breach rate within expected bounds.

**Implementation snapshot targets:**
- GARCH model: `src/agents/technical/models/garch_var.py`
- Day 4 tests: `tests/test_technical_day4.py`

**Output**: GARCH VaR/ES pipeline + validation tests.

---

### Day 5: Friday, March 13 — Walk-Forward Backtest & Ablation

- Implement `src/agents/technical/backtest.py`:
  - Walk-forward backtesting framework:
    - Rolling training window (e.g., 6 months train, 1 month test, step forward 1 month).
    - Cover period: 2019–present (including COVID crash, 2022 volatility, budget cycles).
  - Per-model metrics computation:
    - Sharpe ratio, Sortino ratio, Max Drawdown, Win Rate, Profit Factor.
    - Directional accuracy for classification models.
- Run backtests for each model family:
  1. ARIMA-LSTM hybrid
  2. 2D CNN pattern
  3. GARCH VaR/ES
- Run ablation tests:
  - Remove each feature group (volume, RSI, MACD, macro) one at a time.
  - Measure impact on Sharpe and accuracy.
  - Document which features are essential vs marginal.
- Generate backtest report: `docs/reports/technical_agent_backtest.md`

**Implementation snapshot targets:**
- Backtest framework: `src/agents/technical/backtest.py`
- Backtest report: `docs/reports/technical_agent_backtest.md`
- Ablation results: `docs/reports/technical_agent_ablation.md`

**Output**: Walk-forward backtest report + ablation results for all 3 models.

---

### Day 6: Saturday, March 14 — Integration, Model Cards & Leakage Tests

- Integrate all three models into `TechnicalAgent.predict()`:
  - Agent runs all three models in parallel.
  - Outputs a `TechnicalPrediction` per symbol with the chosen model's forecast.
  - Model selection strategy: ensemble average or best-performing model per regime.
- Write Model Card metadata files (JSON):
  - `data/models/arima_lstm/model_card.json`
  - `data/models/cnn_pattern/model_card.json`
  - `data/models/garch_var/model_card.json`
  - Each card: model_id, version, owner, algorithm, hyperparameters, dataset_id, time_range, features, metrics.
- Run leakage and timestamp alignment tests:
  - Verify no future data leaks into predictions.
  - Verify all timestamps are strictly lagged (prediction time > latest input time).
  - Run `tests/test_technical_day6.py` with explicit leakage checks.
- Document the Technical Agent's API for downstream consumption in `src/agents/technical/README.md`.

**Implementation snapshot targets:**
- Unified agent: `src/agents/technical/technical_agent.py` (updated)
- Model cards: `data/models/*/model_card.json`
- README: `src/agents/technical/README.md`
- Day 6 tests: `tests/test_technical_day6.py`

**Output**: Integrated Technical Agent + model cards + leakage tests + README.

---

### Day 7: Sunday, March 15 — Review, Fix, and Week 2 Handoff

- Run full test suite: `pytest tests/test_technical_*.py -v`
- Review and fix any failing tests or edge cases.
- Review backtest/ablation results — flag any concerns.
- Prepare handoff document for Week 2 (Regime Agent):
  - Document what Technical Agent produces.
  - Define the interface that Regime Agent will consume.
  - List any open issues or deferred items.
- Commit and push all Day 7 changes.

**Output**: All tests passing + Week 2 handoff doc.

---

## Week 1 Exit Criteria

| Criterion | Evidence Required |
|---|---|
| Technical Agent package created | `src/agents/technical/` exists with all modules |
| ARIMA-LSTM model trainable | Day 2 tests pass, training pipeline runs |
| 2D CNN model trainable | Day 3 tests pass, forward pass correct |
| GARCH VaR/ES produces valid quantiles | Day 4 tests pass, VaR breach rate validated |
| Walk-forward backtest complete (2019–present) | Backtest report with Sharpe, MDD, win rate |
| Ablation tests complete | Ablation report with feature importance ranking |
| No leakage in predictions | Day 6 leakage tests pass |
| Model cards for all 3 models | JSON metadata files in `data/models/` |
| All tests pass | `pytest tests/test_technical_*.py` green |

## Dependencies

### Python Packages Required
- `statsmodels` — ARIMA
- `torch` or `tensorflow` — LSTM, CNN
- `arch` — GARCH
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — metrics, preprocessing
- `pydantic` — schemas (already installed)
