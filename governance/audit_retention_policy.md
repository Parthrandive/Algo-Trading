# Data Audit & Retention Policy

## 1. Data Retention Schedule

| Data Type | Description | Retention Period | Storage Tier |
| :--- | :--- | :--- | :--- |
| **Raw Market Data** | Tick/Trade data as received from provider. | 7 Days (Hot) <br> 1 Year (Cold) | Local Disk -> S3/GCS Archive |
| **Aggregated OHLCV** | Processed bars (1m, 5m, 1h). | Indefinite | Database (Hot/Warm) |
| **Trade Logs** | Execution reports, order updates. | 7 Years (Regulatory) | Database + Cold Archive (WORM) |
| **System Logs** | Application logs, error traces. | 30 Days | Log Aggregator / Local File |
| **Model Artifacts** | Trained model binaries, scalers. | Indefinite (Versioned) | Model Registry / Object Storage |

## 2. Storage Tiers

- **Hot Strategy**: Low latency, high cost. Used for active trading and recent analysis.
    - *Media*: NVMe SSDs, In-Memory DB (Redis).
    - *Access*: Immediate.

- **Warm Strategy**: Moderate latency, moderate cost. Used for backtesting and research.
    - *Media*: HDD Arrays, Standard Cloud Storage.
    - *Access*: < 1 min.

- **Cold Strategy**: High latency, low cost. Used for compliance and disaster recovery.
    - *Media*: Glacier / Deep Archive.
    - *Access*: 12-48 hours.

## 3. Audit Logging Requirements

All critical system actions must generate an audit trail entry containing:
- **Timestamp**: UTC ISO8601
- **Actor**: User ID / System Agent ID
- **Action**: CREATE / UPDATE / DELETE / EXECUTE
- **Target**: Resource ID (Order ID, Config File, etc.)
- **Outcome**: SUCCESS / FAILURE
- **Context**: IP Address, Request ID

### Auditable Events
- **Configuration Changes**: Strategy parameter updates.
- **Manual Overrides**: Force close positions, service restarts.
- **User Access**: Login/Logout, permission changes.
- **Deployments**: Code updates, model promotions.
