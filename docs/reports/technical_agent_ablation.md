# Technical Agent Ablation Report (Day 5)

- Generated: `2026-03-14T07:09:49.678311+00:00`
- Walk-forward config: `{'train_months': 6, 'test_months': 1, 'step_months': 1, 'train_days': 15, 'test_days': 3, 'step_days': 3, 'start_date': '2024-03-25'}`

## ARIMA-LSTM Feature Group Ablation

| Group | Sharpe | Directional Accuracy | Sharpe Delta vs Baseline | Accuracy Delta vs Baseline | Predictions | Dropped Columns |
|---|---:|---:|---:|---:|---:|---|
| baseline | N/A | N/A | N/A | N/A | 0 | None |
| volume | N/A | N/A | N/A | N/A | 0 | volume |
| rsi | N/A | N/A | N/A | N/A | 0 | rsi |
| macd | N/A | N/A | N/A | N/A | 0 | macd, macd_hist, macd_signal |
| macro | N/A | N/A | N/A | N/A | 0 | None |

## Notes

- Ablation currently targets ARIMA-LSTM feature engineering groups.
- `macro` group removes columns prefixed with `macro_` when present.