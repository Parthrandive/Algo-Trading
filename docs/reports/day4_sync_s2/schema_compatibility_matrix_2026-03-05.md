# Schema Compatibility Matrix - 2026-03-05

- **dataset_snapshot_id**: `snapshot_20260305_182828_UTC`
- **Conflict pair count**: `0`

## Pairwise Matrix
| Left Table | Right Table | Shared Columns | Conflict Columns | Status |
| --- | --- | --- | --- | --- |
| gold_preprocessing_features | macro_indicator_table | 7 | - | compatible |
| gold_preprocessing_features | text_silver_canonical | 7 | - | compatible |
| gold_preprocessing_features | text_sidecar_metadata | 3 | - | compatible |
| gold_preprocessing_features | corp_actions_table | 9 | - | compatible |
| macro_indicator_table | text_silver_canonical | 7 | - | compatible |
| macro_indicator_table | text_sidecar_metadata | 3 | - | compatible |
| macro_indicator_table | corp_actions_table | 8 | - | compatible |
| text_silver_canonical | text_sidecar_metadata | 4 | - | compatible |
| text_silver_canonical | corp_actions_table | 7 | - | compatible |
| text_sidecar_metadata | corp_actions_table | 3 | - | compatible |

## Per-Table Field Maps

### gold_preprocessing_features
| Column | Type |
| --- | --- |
| CPI | float64 |
| DII_FLOW | float64 |
| FII_FLOW | float64 |
| FX_RESERVES | float64 |
| IIP | float64 |
| INDIA_US_10Y_SPREAD | float64 |
| RBI_BULLETIN | float64 |
| WPI | float64 |
| close | float64 |
| close_log_return | float64 |
| close_log_return_zscore | float64 |
| close_log_return_zscore_is_anomaly | int64 |
| close_log_return_zscore_is_regime_shift | int64 |
| dataset_snapshot_id | string |
| exchange | string |
| high | float64 |
| ingestion_timestamp_ist | datetime |
| ingestion_timestamp_utc | datetime |
| interval | string |
| low | float64 |
| macro_directional_flag | int64 |
| open | float64 |
| quality_status | string |
| schema_version | string |
| source_type | string |
| symbol | string |
| timestamp | datetime |
| volume | int64 |
| vwap | float64 |

### macro_indicator_table
| Column | Type |
| --- | --- |
| dataset_snapshot_id | string |
| indicator_name | string |
| ingestion_timestamp_ist | datetime |
| ingestion_timestamp_utc | datetime |
| period | string |
| quality_status | string |
| region | string |
| schema_version | string |
| source_type | string |
| timestamp | datetime |
| unit | string |
| value | float64 |

### text_silver_canonical
| Column | Type |
| --- | --- |
| author | string |
| content | string |
| dataset_snapshot_id | string |
| embedding | unknown |
| entities | unknown |
| headline | string |
| ingestion_timestamp_ist | datetime |
| ingestion_timestamp_utc | datetime |
| language | string |
| likes | float64 |
| platform | string |
| publisher | string |
| quality_status | string |
| schema_version | string |
| sentiment_score | unknown |
| shares | float64 |
| source_id | string |
| source_type | string |
| timestamp | datetime |
| url | string |

### text_sidecar_metadata
| Column | Type |
| --- | --- |
| compliance_reason | unknown |
| compliance_status | string |
| confidence | float64 |
| dataset_snapshot_id | string |
| ingestion_timestamp_utc | datetime |
| manipulation_risk_score | float64 |
| quality_flags | array |
| source_id | string |
| source_route_detail | string |
| source_type | string |
| ttl_seconds | int64 |

### corp_actions_table
| Column | Type |
| --- | --- |
| action_type | string |
| dataset_snapshot_id | string |
| ex_date | datetime |
| exchange | string |
| ingestion_timestamp_ist | datetime |
| ingestion_timestamp_utc | datetime |
| quality_status | string |
| ratio | float64 |
| record_date | datetime |
| schema_version | string |
| source_type | string |
| symbol | string |
| timestamp | datetime |
| value | float64 |
