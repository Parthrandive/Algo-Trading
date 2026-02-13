# System Failure Modes & Circuit Breakers

## 1. Agent Failure Behaviors

### NSE Sentinel (Market Data)
| Failure Condition | Response Action | Recovery |
| :--- | :--- | :--- |
| **Primary API Down** | Switch to **Fallback Scraper** (tier-2 source). Flag data with `source_type=fallback_scraper`. | Automatic switchback after 15 mins of stable primary pings. |
| **All Sources Down** | Trigger **CRITICAL_DATA_LOSS** event. System enters **CLOSE_ONLY** mode. | Manual intervention or auto-recovery of >1 source. |
| **Data Lag > 30s** | Mark data as `stale`. Downstream agents hold previous valid signal or widen confidence intervals. | Lag drops below 5s. |

### Macro Monitor
| Failure Condition | Response Action | Recovery |
| :--- | :--- | :--- |
| **Source Unavailable** | Use "Last Known Good" value. detailed flag `macro_age` incremented. | New value arrival. |
| **Outlier Value** | (> 3-sigma from median). Quarantine value. Assert `stale` flag to downstream. | Manual review or next valid data point. |

### Preprocessing Agent
| Failure Condition | Response Action | Recovery |
| :--- | :--- | :--- |
| **Schema Mismatch** | **Reject** Batch. Halt pipeline for that symbol. Alert On-Call. | Fix upstream schema or update registry. |
| **Validation Fail** | (e.g., High < Low). Quarantine record. Emit `quality_flag=fail`. | N/A (Bad data discarded). |

## 2. System Circuit Breakers

### Global Risk States

| State | Trigger | Allowed Actions |
| :--- | :--- | :--- |
| **NORMAL** | All systems nominal. | Buy, Sell, Short, Cover. |
| **REDUCE_ONLY** | Fallback source active OR Macro data stale > 24h. | Close positions, Reduce exposure. **No New Entries.** |
| **CLOSE_ONLY** | Critical Data Loss OR Preprocessing Lag > 5 mins. | Market Close orders ONLY. |
| **KILL_SWITCH** | Broker Disconnect OR PnL Drawdown > Limit. | **Cancel All Open Orders.** Halt all execution. |

### Automated Circuit Breaker Logic
1.  **Feed Integrity Monitor**: Continuously checks heartbeat of NSE Sentinel. If heartbeat misses > 3 beats (15s), trip to `CLOSE_ONLY`.
2.  **Asset Specific Halt**: If a single symbol fails validation > 50% of time in 1 hour, blacklist symbol for remainder of day.
