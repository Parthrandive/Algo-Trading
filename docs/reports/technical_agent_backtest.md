# Technical Agent Backtest Report (Day 5)

- Generated: `2026-03-14T06:52:05.846714+00:00`
- Walk-forward config: `{'train_months': 6, 'test_months': 1, 'step_months': 1, 'train_days': 15, 'test_days': 3, 'step_days': 3, 'start_date': '2026-02-13'}`

## Aggregate Metrics

| Model | Sharpe | Sortino | Max Drawdown | Win Rate | Profit Factor | Directional Accuracy | Coverage | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arima_lstm | 5.7958 | 8.1806 | -0.0158 | 0.5455 | 1.5260 | 0.5455 | 0.7333 | 33 |
| cnn_pattern | 5.7026 | 9.8319 | -0.0177 | 0.5111 | 1.5681 | 0.5897 | 1.0000 | 45 |
| garch_var | N/A | N/A | N/A | N/A | N/A | N/A | 0.0000 | 0 |

## Notes

- Metrics are computed on strategy returns from walk-forward predictions.
- GARCH strategy uses a volatility-scaled directional proxy from fitted mean/trailing drift.
- This report is generated from whichever dataset is passed to `TechnicalBacktester`.