# Cross-Agent Replay Report (Day 4 Sync S2) - 2026-03-05

- **Run date (UTC)**: 2026-03-05T18:28:29.194830+00:00
- **Trading day replayed**: 2026-03-05
- **Cutoff**: 2026-03-05T23:59:59+00:00
- **dataset_snapshot_id**: `snapshot_20260305_182828_UTC`
- **Preprocessing Gold hash**: `0542a912ffe2e595bf0892715091e6a34893299a0bc4feddf721b8fc923500fc`
- **Deterministic replay (A == B)**: `True`

## Stream Coverage
| Stream | dataset_snapshot_id | Record Count | Trace Reference |
| --- | --- | --- | --- |
| sentinel_ohlcv_silver | snapshot_20260305_182828_UTC | 14 | /Users/juhi/Desktop/algo-trading/docs/reports/day4_sync_s2/artifacts/market_slice_2026-03-05.parquet |
| sentinel_corporate_actions_silver | snapshot_20260305_182828_UTC | 8 | /Users/juhi/Desktop/algo-trading/docs/reports/day4_sync_s2/artifacts/corp_slice_asof_2026-03-05.parquet |
| macro_indicator_tables | snapshot_20260305_182828_UTC | 16 | /Users/juhi/Desktop/algo-trading/docs/reports/day4_sync_s2/artifacts/macro_slice_asof_2026-03-05.parquet |
| textual_silver_plus_sidecar | snapshot_20260305_182828_UTC | 4 | local_textual_replay_slice |
| preprocessing_gold_features | snapshot_20260305_182828_UTC | 14 | 0542a912ffe2e595bf0892715091e6a34893299a0bc4feddf721b8fc923500fc |

## Snapshot ID Cross-Reference
- **Alignment pass**: `True`

## Text Sidecar Metadata Validation
| source_id | confidence float | ttl_seconds int | manipulation_risk_score float |
| --- | --- | --- | --- |
| nse_news_2026-03-05_001 | True | True | True |
| x_post_2026-03-05_001 | True | True | True |

- **All required sidecar fields typed correctly**: `True`

## Status
- **S2 decision status**: `GO_WITH_CONDITIONS`
- **Note**: Partner textual replay evidence is tracked in defect log until external artifact is attached.
