# USDINR + Two Stocks Training (2026-03-17)

## Run setup
- Script: `scripts/train_regime_aware.py`
- Interval: `1h`
- Symbols requested: `USDINR=X`, `RELIANCE.NS`, `INFY.NS`
- Additional stock run: `TATASTEEL.NS` (`--min-rows 100`)

## Results
| Symbol | Rows | Train Rows | Val Rows | CNN Val Accuracy | CNN Val Balanced Accuracy | ARIMA-LSTM Val RMSE | ARIMA-LSTM Val MAE | ARIMA-LSTM Directional Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| USDINR=X | 3500 | 2800 | 700 | 0.9757 | 0.3694 | 0.1412 | 0.0764 | 0.5600 |
| RELIANCE.NS | 1842 | 1473 | 369 | 0.8211 | 0.3290 | 6.4827 | 3.9812 | 0.4851 |
| TATASTEEL.NS | 172 | 137 | 35 | 0.1714 | 0.1686 | 1.6337 | 1.0820 | 0.4286 |

## Notes
- `INFY.NS` failed in the first run due insufficient usable rows after preprocessing (`Only 14 rows available; need at least 300`).
- `TATASTEEL.NS` required lowering `--min-rows` to `100` to train.
- Model heavily predicts neutral class on `USDINR=X` and `RELIANCE.NS` (high raw accuracy but low balanced accuracy).

## Artifact paths
- `data/reports/training_runs/regime_aware_1h_20260317_170043` (USDINR + RELIANCE + INFY fail)
- `data/reports/training_runs/regime_aware_1h_20260317_170657` (TATASTEEL)
