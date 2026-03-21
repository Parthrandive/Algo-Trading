# Technical Agent Ablation Report (Day 5)

- Generated: `2026-03-20T18:12:26.812915+00:00`
- Walk-forward config: `{'train_months': 6, 'test_months': 1, 'step_months': 1, 'train_days': None, 'test_days': None, 'step_days': None, 'start_date': '2019-01-01'}`

## ARIMA-LSTM Feature Group Ablation

| Group | Sharpe | Directional Accuracy | Sharpe Delta vs Baseline | Accuracy Delta vs Baseline | Predictions | Dropped Columns |
|---|---:|---:|---:|---:|---:|---|
| baseline | N/A | N/A | N/A | N/A | 0 | None |
| volume | N/A | N/A | N/A | N/A | 0 | volume |
| rsi | N/A | N/A | N/A | N/A | 0 | rsi |
| macd | N/A | N/A | N/A | N/A | 0 | macd, macd_hist, macd_signal |
| macro | N/A | N/A | N/A | N/A | 0 | macro_coverage_ratio, macro_directional_flag, macro_regime_index, macro_regime_shock |

## Notes

- Ablation currently targets ARIMA-LSTM feature engineering groups.
- `macro` group removes columns prefixed with `macro_` when present.