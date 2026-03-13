# Technical Agent Backtest Report (Day 5)

- Generated: `2026-03-13T18:48:00.216978+00:00`
- Walk-forward config: `{'train_months': 6, 'test_months': 1, 'step_months': 1, 'train_days': 3, 'test_days': 1, 'step_days': 1, 'start_date': '2026-02-09'}`

## Aggregate Metrics

| Model | Sharpe | Sortino | Max Drawdown | Win Rate | Profit Factor | Directional Accuracy | Coverage | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arima_lstm | N/A | N/A | N/A | N/A | N/A | N/A | 0.0000 | 0 |
| cnn_pattern | N/A | N/A | N/A | N/A | N/A | N/A | 0.0000 | 0 |
| garch_var | N/A | N/A | N/A | N/A | N/A | N/A | 0.0000 | 0 |

## Notes

- Metrics are computed on strategy returns from walk-forward predictions.
- GARCH strategy uses a volatility-scaled directional proxy from fitted mean/trailing drift.
- This report is generated from whichever dataset is passed to `TechnicalBacktester`.