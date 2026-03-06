# Cross-Agent Replay Report (Day 4 Sync S2) - 2026-03-05

- **Run date (UTC)**: 2026-03-06T15:17:46.871403+00:00
- **Trading day replayed**: 2026-03-05
- **Cutoff**: 2026-03-05T23:59:59+00:00
- **dataset_snapshot_id**: `snapshot_20260306_151746_UTC`
- **Preprocessing Gold hash**: `d18c31c657bf37eb882fc5a0c33827a2b78b1a0cde64a3f0ca1c276cd6861ade`
- **Deterministic replay (A == B)**: `True`

## Stream Coverage
| Stream | dataset_snapshot_id | Record Count | Trace Reference |
| --- | --- | --- | --- |
| sentinel_ohlcv_silver | snapshot_20260306_151746_UTC | 14 | /Users/juhi/Desktop/algo-trading/docs/reports/day4_sync_s2/artifacts/market_slice_2026-03-05.parquet |
| sentinel_corporate_actions_silver | snapshot_20260306_151746_UTC | 8 | /Users/juhi/Desktop/algo-trading/docs/reports/day4_sync_s2/artifacts/corp_slice_asof_2026-03-05.parquet |
| macro_indicator_tables | snapshot_20260306_151746_UTC | 16 | /Users/juhi/Desktop/algo-trading/docs/reports/day4_sync_s2/artifacts/macro_slice_asof_2026-03-05.parquet |
| textual_silver_plus_sidecar | snapshot_20260306_151746_UTC | 90 | /Users/juhi/Desktop/algo-trading/docs/reports/day4_sync_s2/artifacts/textual_canonical_2026-03-05.parquet|/Users/juhi/Desktop/algo-trading/docs/reports/day4_sync_s2/artifacts/textual_sidecar_2026-03-05.parquet |
| preprocessing_gold_features | snapshot_20260306_151746_UTC | 14 | d18c31c657bf37eb882fc5a0c33827a2b78b1a0cde64a3f0ca1c276cd6861ade |

## Snapshot ID Cross-Reference
- **Alignment pass**: `True`

## Text Sidecar Metadata Validation
| source_id | confidence float | ttl_seconds int | manipulation_risk_score float |
| --- | --- | --- | --- |
| nse_news_eb77ac94137a2ade | True | True | True |
| nse_news_614c8e13cdd5359c | True | True | True |
| nse_news_371613b00c41fbc4 | True | True | True |
| nse_news_4fcf03bf260e289f | True | True | True |
| nse_news_374825b94e040621 | True | True | True |
| nse_news_272473817149126e | True | True | True |
| nse_news_404b2d86ca30310d | True | True | True |
| nse_news_9184456fe70b661e | True | True | True |
| nse_news_0de5e2749aa10dea | True | True | True |
| nse_news_98096bbc5713386b | True | True | True |
| nse_news_525bbddc9ff4b505 | True | True | True |
| nse_news_d334407e1516da16 | True | True | True |
| nse_news_f15b3e69f48be100 | True | True | True |
| nse_news_92fbed6a3221d877 | True | True | True |
| nse_news_2e269d4f9b34fa5f | True | True | True |
| nse_news_e7e6c68a4220076a | True | True | True |
| nse_news_787f7ec7cb7feb71 | True | True | True |
| nse_news_f0302eec0366bec2 | True | True | True |
| nse_news_a172a9717550e429 | True | True | True |
| nse_news_ed705e70b655bf79 | True | True | True |
| et_news_3baa3cce97d98bd2 | True | True | True |
| et_news_6c9c79e1cdb8b200 | True | True | True |
| et_news_2ad582253c3c6406 | True | True | True |
| et_news_c3cc9d970610a5e9 | True | True | True |
| et_news_9ce1d5ad8c9a293f | True | True | True |
| et_news_ded418aeff076afc | True | True | True |
| et_news_73115e47fbe7d3c5 | True | True | True |
| et_news_4e705873e1e45bc7 | True | True | True |
| et_news_b0579f3db88b0a49 | True | True | True |
| et_news_de17b7b49c933d84 | True | True | True |
| et_news_74a53477f001cbc4 | True | True | True |
| et_news_06f046942ce35a7d | True | True | True |
| et_news_c640ab31151923a0 | True | True | True |
| et_news_781f59e0490cecc2 | True | True | True |
| et_news_3c2c204a928b2e06 | True | True | True |
| et_news_15d63a948a7568a3 | True | True | True |
| et_news_1d0839dd8fb0d816 | True | True | True |
| et_news_373b63877c13a62d | True | True | True |
| et_news_4531a692f2c9a239 | True | True | True |
| et_news_5800c5bec22f21ef | True | True | True |
| et_news_4235e8265a7a54e2 | True | True | True |
| et_news_2ea84cf110e06865 | True | True | True |
| et_news_90f80d671f688a38 | True | True | True |
| et_news_72a0441f6ebeed90 | True | True | True |
| et_news_ed82ee70dea4bcc1 | True | True | True |

- **All required sidecar fields typed correctly**: `True`
- **Textual replay source mode**: `persisted_silver_artifacts`

## Status
- **S2 decision status**: `GO_WITH_CONDITIONS`
- **Note**: Textual slice is generated from live/cached adapters and persisted in artifact parquet files.
