# Technical Agent Ablation Report (Day 5)

- Generated: `2026-03-13T18:48:00.217617+00:00`
- Walk-forward config: `{'train_months': 6, 'test_months': 1, 'step_months': 1, 'train_days': 3, 'test_days': 1, 'step_days': 1, 'start_date': '2026-02-09'}`

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