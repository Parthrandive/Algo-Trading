# Week 1 Technical Agent to Week 2 Regime Agent Handoff

Prepared on: 2026-03-14 (IST)
Scope: Phase 2, Week 1 Day 7 handoff from Technical Agent to Regime Agent.

## 1) What is complete

- Unified technical inference entrypoint is available at `TechnicalAgent.predict(symbol)`.
- Ensemble components are integrated:
  - ARIMA-LSTM (price magnitude forecast)
  - CNN pattern classifier (direction and confidence)
  - GARCH VaR/ES (volatility and downside risk quantiles)
- Model cards exist for all three models in `data/models/*/model_card.json`.
- Leakage and timestamp alignment tests are in place (`tests/test_technical_day6.py`).
- Day 7 regression hardening added:
  - Inference now ignores unrelated all-NaN optional columns (for example `vwap`) instead of dropping all rows.

## 2) Output contract for Regime Agent

Producer:
- `src/agents/technical/technical_agent.py`

Schema:
- `src/agents/technical/schemas.py` (`TechnicalPrediction`, schema_version `1.0`)

Invocation contract:
- Input: `symbol: str`
- Return: `Optional[TechnicalPrediction]`
- `None` is a valid output when data/features are insufficient or upstream loading fails.

`TechnicalPrediction` fields consumed by Regime Agent:
- `symbol: str`
- `timestamp: datetime` (UTC generation time)
- `price_forecast: float` (next-horizon magnitude proxy)
- `direction: "up" | "down" | "neutral"`
- `volatility_estimate: float`
- `var_95: float`
- `var_99: float`
- `es_95: float`
- `es_99: float`
- `confidence: float` in [0, 1]
- `model_id: str` (currently `ensemble_arima_cnn_garch_v1.0`)
- `schema_version: str` (default `1.0`)

Consumer handling requirements:
- Treat `None` as `NO_SIGNAL` and activate reduced-risk fallback logic.
- Treat risk metrics (`var_*`, `es_*`) as downside measures (expected to be negative under loss convention).
- Do not assume non-zero `price_forecast`; fallback defaults may appear if ARIMA inference errors.

## 3) Validation evidence (Day 7 gate)

- Technical suite gate command:
  - `pytest tests/test_technical_*.py -v`
- Current status:
  - All tests passing.
- Additional Day 7 regression:
  - `test_technical_agent_ignores_all_nan_optional_columns` ensures optional all-NaN columns do not suppress predictions.

## 4) Backtest and ablation review (concerns to carry into Week 2)

Sources:
- `docs/reports/technical_agent_backtest.md`
- `docs/reports/technical_agent_ablation.md`

Observed concerns:
- ARIMA-LSTM backtest coverage is `0.0000` with `0` predictions (no usable strategy output).
- CNN performance is weak:
  - Sharpe: `-0.0482`
  - Directional accuracy: `0.4286`
- GARCH strategy performance is poor and sparse:
  - Sharpe: `-2.8008`
  - Coverage: `0.1596`
- Ablation report is not yet informative because ARIMA-LSTM baseline is `N/A` with zero predictions.

Implication for Regime Agent:
- Week 2 should initially use technical outputs as weak evidence and apply conservative confidence weighting until technical model coverage/performance is improved.

## 5) Open issues and deferred items

Open issues:
1. ARIMA-LSTM backtest path producing zero predictions needs root-cause analysis.
2. CNN and GARCH trading metrics are below deployment quality thresholds (negative Sharpe).
3. Current model selection is static (`model_id` fixed) and does not yet adapt by market regime.

Deferred items:
1. Regime-conditional model routing (choose model family by detected state).
2. Confidence calibration against out-of-sample regime segments.
3. Expanded ablation beyond ARIMA feature groups after ARIMA coverage is restored.

## 6) Week 2 integration checklist for Regime Agent

1. Wire `TechnicalAgent.predict(symbol)` into `detect_regime()` input pipeline.
2. Define explicit behavior for `None` predictions and low-confidence predictions.
3. Use `direction`, `confidence`, `volatility_estimate`, and `var_99` as primary technical factors for early regime scoring.
4. Track signal availability and quality metrics per symbol/day:
   - prediction coverage
   - confidence distribution
   - rate of fallback/default outputs
5. Open a follow-up issue to repair ARIMA-LSTM backtest coverage before Regime Week 2 Day 6 calibration.
