# Technical Agent Ablation Report (Day 5)

- Generated: `2026-03-14T06:52:05.847534+00:00`
- Walk-forward config: `{'train_months': 6, 'test_months': 1, 'step_months': 1, 'train_days': 15, 'test_days': 3, 'step_days': 3, 'start_date': '2026-02-13'}`

## ARIMA-LSTM Feature Group Ablation

| Group | Sharpe | Directional Accuracy | Sharpe Delta vs Baseline | Accuracy Delta vs Baseline | Predictions | Dropped Columns |
|---|---:|---:|---:|---:|---:|---|
| baseline | 2.2230 | 0.5152 | 0.0000 | 0.0000 | 33 | None |
| volume | 4.7557 | 0.5152 | 2.5327 | 0.0000 | 33 | volume |
| rsi | -0.1193 | 0.4545 | -2.3423 | -0.0606 | 33 | rsi |
| macd | 2.6520 | 0.5455 | 0.4290 | 0.0303 | 33 | macd, macd_hist, macd_signal |
| macro | -2.1931 | 0.4545 | -4.4161 | -0.0606 | 33 | None |

## Notes

- Ablation currently targets ARIMA-LSTM feature engineering groups.
- `macro` group removes columns prefixed with `macro_` when present.