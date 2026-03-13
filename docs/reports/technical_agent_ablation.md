# Technical Agent Ablation Report (Day 5)

- Status: Ablation framework implemented, awaiting real-data execution window.
- Source plan: `docs/plans/Phase_2_Week1_Technical_Agent_Plan.md` (Day 5).

## What Is Implemented

- Feature-group ablation flow in `src/agents/technical/backtest.py`.
- Baseline + one-at-a-time removals for ARIMA-LSTM feature groups:
  - `volume`
  - `rsi`
  - `macd`
  - `macro` (drops `macro_*` columns when present)
- Delta tracking versus baseline for:
  - Sharpe Ratio
  - Directional Accuracy

## Run Command (Once Real Data Is Attached)

Use `TechnicalBacktester.run_ablation(market_df)` and then `write_reports(...)` to regenerate this report with live ablation deltas.

## Notes

- This document is a Day 5 scaffold output.
- Production ablation rankings will be generated after Day 5 real data is available.
