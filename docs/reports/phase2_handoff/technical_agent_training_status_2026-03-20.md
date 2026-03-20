# Technical Agent Training Status (Phase 2)

Date: 2026-03-20 (IST)  
Scope: Phase 2 Technical Agent training pipeline status after latest reruns.

## Executed in this cycle

- Ran unified pipeline for `HDFCBANK.NS`:
  - `python scripts/train_models.py --symbol HDFCBANK.NS --interval 1d --no-auto-prepare-data`
- Fixed backtest window auto-adjust behavior for non-intraday data:
  - `scripts/run_backtest.py`
- Re-ran backtest/ablation after fix:
  - `python scripts/run_backtest.py --symbol HDFCBANK.NS --interval 1d`
- Verified Technical test gates:
  - `pytest tests/test_technical_day5.py -q` (3 passed)
  - `pytest tests/test_technical_day6.py -q` (4 passed)

## Complete

- Train-ready daily NSE symbols available in quality table: `INFY.NS`, `RELIANCE.NS`, `TATASTEEL.NS`, `TCS.NS`, `HDFCBANK.NS`.
- HDFCBANK training artifacts refreshed for all three Technical model families:
  - ARIMA-LSTM: `data/models/arima_lstm/training_meta.json`
  - CNN pattern: `data/models/cnn_pattern/training_meta.json`
  - GARCH VaR/ES: `data/models/garch_var/training_meta.json`
- GARCH fit converged (`convergence_flag=0`) and produced VaR/ES with Kupiec backtest output.
- Walk-forward report now uses month-based windows for `1d` data (no forced `15/3` day windows).
- CNN and GARCH backtest coverage restored to `1.0000` for the latest HDFCBANK run:
  - `docs/reports/technical_agent_backtest.md`

## Pending

- ARIMA-LSTM backtest coverage remains `0.0000` with zero predictions:
  - `docs/reports/technical_agent_backtest.md`
- ARIMA ablation remains non-informative (`N/A` baseline and deltas):
  - `docs/reports/technical_agent_ablation.md`
- Cross-symbol CNN acceptance thresholds are not met in latest recheck summaries (all symbols in `WARN` state):
  - `data/reports/training_runs_updated/recheck_prev_20260320_223124_summary.json`
  - `data/reports/training_runs_updated/recheck_new_symbol_hdfcbank_20260320_224816_summary.json`
- GARCH 99% Kupiec test p-value is low (`0.0012`), indicating calibration stress at tail confidence:
  - `data/models/garch_var/training_meta.json`
- Model governance/registry completeness is still pending against MLOps checklist:
  - missing explicit `training_data_snapshot_hash`, `code_hash`, and reproducibility hash evidence in model artifacts.

## Immediate next technical blockers

- Root-cause ARIMA walk-forward zero-prediction path inside `TechnicalBacktester._run_arima_lstm_split`.
- Regenerate ablation only after ARIMA baseline produces non-zero predictions.
- Raise cross-symbol CNN balanced accuracy and down-recall to pass thresholds before promotion.
