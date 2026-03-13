# Technical Agent Backtest Report (Day 5)

- Status: Framework implemented, awaiting real-data execution window.
- Source plan: `docs/plans/Phase_2_Week1_Technical_Agent_Plan.md` (Day 5).

## What Is Implemented

- Walk-forward backtesting engine in `src/agents/technical/backtest.py`.
- Rolling train/test splits (`6m train / 1m test / 1m step` configurable).
- Backtests for:
  - `arima_lstm`
  - `cnn_pattern`
  - `garch_var`
- Metrics:
  - Sharpe Ratio
  - Sortino Ratio
  - Max Drawdown
  - Win Rate
  - Profit Factor
  - Directional Accuracy

## Run Command (Once Real Data Is Attached)

Use `TechnicalBacktester.run_model_backtests(market_df)` and then `write_reports(...)` to regenerate this report with live metrics.

## Notes

- Current repository includes Day 5 framework and tests.
- Final production numbers should be generated after the real OHLCV attachment for Day 5.
