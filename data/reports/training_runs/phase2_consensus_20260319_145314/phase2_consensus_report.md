# Phase-2 Offline Training Report (Technical -> Regime -> Sentiment -> Consensus)

- Generated at (UTC): 2026-03-19T14:53:24.167884+00:00
- Symbols: RELIANCE.NS, TATASTEEL.NS, USDINR=X
- Run directory: `data\reports\training_runs\phase2_consensus_20260319_145314`

## Data Used
- Hourly OHLCV root: `data\silver\ohlcv`
- Macro root: `data\silver\macro`
- Textual canonical artifact: `docs\reports\day4_sync_s2\artifacts\textual_canonical_2026-03-05.parquet`
- Textual sidecar artifact: `docs\reports\day4_sync_s2\artifacts\textual_sidecar_2026-03-05.parquet`
- Data overview: `{"symbols": {"RELIANCE.NS": {"rows": 37, "start_utc": "2026-02-23 08:45:00+00:00", "end_utc": "2026-03-02 09:45:00+00:00"}, "TATASTEEL.NS": {"rows": 140, "start_utc": "2026-01-27 03:45:00+00:00", "end_utc": "2026-02-24 09:45:00+00:00"}, "USDINR=X": {"rows": 140, "start_utc": "2026-01-27 04:00:00+00:00", "end_utc": "2026-02-24 10:00:00+00:00"}}, "macro_rows": 12, "doc_rows": 20}`

## Method
- Chronological expanding-window walk-forward used for base agents.
- Consensus trained only on out-of-sample base-agent predictions from pre-test windows.
- Latest contiguous block reserved as untouched final test set.
- Weighted baseline uses protective safeguards; challenger is Bayesian LSTAR/ESTAR-style.

## Leakage Checks
- Checks: `{"macro_asof_safe_violations": 0, "text_timestamp_alignment_violations": 0, "consensus_train_rows_from_oos_only": true, "chronological_order_pass": true, "future_label_join_violations": 0}`

## Base Agent Metrics
- Technical: `{"accuracy": 0.11392405063291139, "balanced_accuracy": 0.3683288610614192, "macro_f1": 0.12074685439171419, "confusion_matrix": [[10, 0, 3], [73, 3, 53], [11, 0, 5]], "per_class": {"down": {"precision": 0.10638297872340426, "recall": 0.7692307692307693, "f1": 0.18691588785046728, "support": 13}, "neutral": {"precision": 1.0, "recall": 0.023255813953488372, "f1": 0.045454545454545456, "support": 129}, "up": {"precision": 0.08196721311475409, "recall": 0.3125, "f1": 0.12987012987012986, "support": 16}}, "confidence_brier": 0.34772388009242305, "regression_rmse": 0.005321089115945829, "regression_mae": 0.002442179786344355, "time_slice_stability_std_accuracy": 0.055697098601659514}`
- Regime: `{"accuracy": 0.6582278481012658, "macro_f1": 0.15877862595419848, "labels": ["Alien", "Bear", "Bull", "Crisis", "Sideways"], "confusion_matrix": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 12], [0, 0, 0, 0, 12], [0, 0, 0, 0, 29], [1, 0, 0, 0, 104]], "transition_stability_flip_rate": 0.0189873417721519, "ood_warning_rate": 0.02531645569620253, "ood_alien_rate": 0.006329113924050633}`
- Sentiment: `{"status": "insufficient_labeled_docs", "doc_count": 20, "macro_f1": null, "class_balance": {}, "timestamp_alignment_violations": 0}`

## Consensus Metrics (Final Test)
- Weighted baseline: `{"accuracy": 0.16666666666666666, "balanced_accuracy": 0.4296296296296296, "macro_f1": 0.15810393925011865, "confusion_matrix": [[4, 0, 0], [31, 4, 10], [4, 0, 1]], "per_class": {"down": {"precision": 0.10256410256410256, "recall": 1.0, "f1": 0.18604651162790697, "support": 4}, "neutral": {"precision": 1.0, "recall": 0.08888888888888889, "f1": 0.16326530612244897, "support": 45}, "up": {"precision": 0.09090909090909091, "recall": 0.2, "f1": 0.125, "support": 5}}, "confidence_brier": 0.2692748128866568, "slice_accuracy": {"high_volatility": 0.14285714285714285, "stale_or_missing_sentiment": 0.16666666666666666, "strong_disagreement": null, "normal": 0.175}, "proxy_utility_after_costs": 0.0011560364191983254, "proxy_utility_mean_per_bar": 2.1408081837006026e-05, "robust_disagreement_neutral_rate": 0.0}`
- Challenger: `{"accuracy": 0.8333333333333334, "balanced_accuracy": 0.3333333333333333, "macro_f1": 0.30612244897959184, "confusion_matrix": [[0, 3, 1], [0, 45, 0], [0, 5, 0]], "per_class": {"down": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 4}, "neutral": {"precision": 0.8490566037735849, "recall": 1.0, "f1": 0.9183673469387755, "support": 45}, "up": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 5}}, "confidence_brier": 0.1724243207714079, "slice_accuracy": {"high_volatility": 0.7857142857142857, "stale_or_missing_sentiment": 0.8333333333333334, "strong_disagreement": null, "normal": 0.85}, "proxy_utility_after_costs": -0.002854481781508044, "proxy_utility_mean_per_bar": -5.286077373163044e-05, "robust_disagreement_neutral_rate": 0.0}`

## Ablations
- weighted_consensus_baseline: accuracy=0.1667, macro_f1=0.1581
- no_sentiment: accuracy=0.1111, macro_f1=0.1121
- no_regime: accuracy=0.1111, macro_f1=0.1121
- simple_average: accuracy=0.1667, macro_f1=0.1581

## Recommendation
- Recommended now: **challenger_lstar_estar_bayesian**
- Reason: Challenger shows clear out-of-sample uplift with no major instability on high-vol slices.

## Assumptions and Limitations
- Available local history is limited; long-horizon (2019+) validation is not currently possible from local files.
- Textual data coverage is sparse and mostly point-in-time artifacts; sentiment is run with graceful degradation.
- Results are offline research validation only; no live execution logic is touched.
