# Service Level Agreements (SLAs)

This document defines the Service Level Agreements (SLAs) for the Multi-Agent AI Trading System. These SLAs serve as the baseline for system health and performance.

## 1. Latency SLAs
*Criticality: High*

| Metric | Target (p95) | Max Limit (p99) | Measurement Point |
| :--- | :--- | :--- | :--- |
| **Tick-to-Trade** | < 50ms | < 100ms | Time from market data ingest to order generation signal. |
| **Order Execution** | < 200ms | < 500ms | Time from order generation to broker ACK. |
| **Internal Msg** | < 5ms | < 10ms | Intra-agent message passing latency (ZeroMQ/Queue). |

## 2. Data Freshness SLAs
*Criticality: Critical*

| Data Type | Max Age (Normal) | Max Age (Degraded) | Action on Breach |
| :--- | :--- | :--- | :--- |
| **L1 Market Data** | < 1s | < 3s | Switch to Reduce-Only if > 3s. Halt if > 5s. |
| **Positions/Balances** | < 5s | < 10s | Block new orders until refreshed. |
| **Macro Indicators** | < 1h | < 4h | Warn. Use stale data with penalty weight. |

## 3. Data Quality SLAs
*Criticality: High*

| Metric | Threshold | Action on Breach |
| :--- | :--- | :--- |
| **Missing/Null/NaN** | 0% for OHLC | Reject ingest. Log error. |
| **Outlier Detection** | < 3 sigma | Flag as suspect. Require manual override or secondary source confirmation. |
| **Schema Compliance** | 100% | Reject record immediately. |

## 4. System Availability
| Component | Uptime Target | Maint. Window |
| :--- | :--- | :--- |
| **Core Trading Loop** | 99.9% (Market Hours) | Weekends / Post-Market |
| **Data Ingestion** | 99.5% | Weekends |
| **Dashboard/Logs** | 99.0% | Any time non-critical |
