# Technical Agent Backtest Report (Day 5)

- Generated: `2026-03-20T18:12:26.811400+00:00`
- Walk-forward config: `{'train_months': 6, 'test_months': 1, 'step_months': 1, 'train_days': None, 'test_days': None, 'step_days': None, 'start_date': '2019-01-01'}`

## Aggregate Metrics

| Model | Sharpe | Sortino | Max Drawdown | Win Rate | Profit Factor | Directional Accuracy | Coverage | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arima_lstm | N/A | N/A | N/A | N/A | N/A | N/A | 0.0000 | 0 |
| cnn_pattern | 0.0062 | 0.0090 | -0.5107 | 0.4927 | 1.0012 | 0.4927 | 1.0000 | 1652 |
| garch_var | 0.1732 | 0.2650 | -0.2965 | 0.5018 | 1.0311 | 0.5018 | 1.0000 | 1652 |

## Notes

- Metrics are computed on strategy returns from walk-forward predictions.
- GARCH strategy uses a volatility-scaled directional proxy from fitted mean/trailing drift.
- This report is generated from whichever dataset is passed to `TechnicalBacktester`.