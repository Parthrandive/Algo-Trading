# NSE Sentinel Connector Spec v1
Date: 2026-02-19
Scope: Week 2 Day 8 contract lock for Phase 1 (NSE Sentinel Agent)

## 1. Connector Contracts

### Interface
All market connectors implement `NSEClientInterface`:
- `get_stock_quote(symbol) -> Tick`
- `get_historical_data(symbol, start_date, end_date, interval=\"1h\") -> list[Bar]`

Reference: `/Users/juhi/Desktop/algo-trading/src/agents/sentinel/client.py`.

### Feed Roles
- Primary feed: `official_api` (current implementation: yfinance connector)
- Secondary feed: `broker_api` (REST broker connector)
- Fallback feed: `fallback_scraper` (nsepython wrapper)

## 2. Symbol Universe + Session Rules (Runtime Baseline)

Runtime baseline file:
- `/Users/juhi/Desktop/algo-trading/configs/nse_sentinel_runtime_v1.json`

Core symbol universe (`week2-core-v1`):
- Equities: `RELIANCE.NS`, `TATASTEEL.NS`
- FX: `USDINR=X`
- Index reference: `^NSEI`

Session calendar/rules:
- Timezone: `Asia/Kolkata`
- Trading days: Monday-Friday (`0..4`)
- Regular session: `09:15` to `15:30` IST
- Pre-open start: `09:00` IST
- Holiday baseline: `2026-01-26`

## 3. Bronze -> Silver Canonical Field Mapping

### `market.tick.v1`
| Bronze input key | Silver key | Rule |
| :--- | :--- | :--- |
| `symbol` | `symbol` | Required |
| `timestamp` | `timestamp` | Must be timezone-aware |
| `price` | `price` | `> 0` |
| `volume` | `volume` | `>= 0` |
| `source_type` / `source` | `source_type` | Canonical enum |
| `ingestion_timestamp` | `ingestion_timestamp_utc` | Backward-compatible alias |
| n/a | `ingestion_timestamp_ist` | Auto-derived / validated |
| n/a | `schema_version` | Default `1.0` |
| `quality_flag` | `quality_status` | Backward-compatible alias |

### `market.bar.v1`
| Bronze input key | Silver key | Rule |
| :--- | :--- | :--- |
| `symbol` | `symbol` | Required |
| `timestamp` | `timestamp` | Must be timezone-aware |
| `interval` | `interval` | Required |
| `open` | `open` | `> 0` |
| `high` | `high` | `>= max(open, low, close)` |
| `low` | `low` | `<= min(open, high, close)` |
| `close` | `close` | `> 0` |
| `volume` | `volume` | `>= 0` |
| `source_type` / `source` | `source_type` | Canonical enum |
| `ingestion_timestamp` | `ingestion_timestamp_utc` | Backward-compatible alias |
| n/a | `ingestion_timestamp_ist` | Auto-derived / validated |
| n/a | `schema_version` | Default `1.0` |
| `quality_flag` | `quality_status` | Backward-compatible alias |

### `market.corporate_action.v1`
| Bronze input key | Silver key | Rule |
| :--- | :--- | :--- |
| `symbol` | `symbol` | Required |
| `action_type` | `action_type` | Enum: dividend/split/bonus/rights |
| `ratio` | `ratio` | Optional |
| `value` | `value` | Optional numeric |
| `ex_date` | `ex_date` | Required timezone-aware datetime |
| `record_date` | `record_date` | Optional timezone-aware datetime |
| `source_type` / `source` | `source_type` | Canonical enum |
| `ingestion_timestamp` | `ingestion_timestamp_utc` | Backward-compatible alias |
| n/a | `ingestion_timestamp_ist` | Auto-derived / validated |
| n/a | `schema_version` | Default `1.0` |
| `quality_flag` | `quality_status` | Backward-compatible alias |

## 4. Runtime Config Loader

Runtime config model + loader:
- `/Users/juhi/Desktop/algo-trading/src/agents/sentinel/config.py`

This module provides:
- Strongly-typed source priorities and retry/rate-limit policy
- Versioned symbol universe access (`all_symbols`)
- Session rule check (`is_trading_session`)
- Config loading from `configs/nse_sentinel_runtime_v1.json`
