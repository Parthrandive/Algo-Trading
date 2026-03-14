# Technical Agent Backtest Report (Day 5)

- Generated: `2026-03-14T07:09:49.676538+00:00`
- Walk-forward config: `{'train_months': 6, 'test_months': 1, 'step_months': 1, 'train_days': 15, 'test_days': 3, 'step_days': 3, 'start_date': '2024-03-25'}`

## Aggregate Metrics

| Model | Sharpe | Sortino | Max Drawdown | Win Rate | Profit Factor | Directional Accuracy | Coverage | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arima_lstm | N/A | N/A | N/A | N/A | N/A | N/A | 0.0000 | 0 |
| cnn_pattern | -0.0482 | -0.0060 | -0.0023 | 0.0026 | 0.9600 | 0.4286 | 1.0000 | 3490 |
| garch_var | -2.8008 | -2.9297 | -0.0381 | 0.4668 | 0.7866 | 0.4668 | 0.1596 | 557 |

## Notes

- Metrics are computed on strategy returns from walk-forward predictions.
- GARCH strategy uses a volatility-scaled directional proxy from fitted mean/trailing drift.
- This report is generated from whichever dataset is passed to `TechnicalBacktester`.