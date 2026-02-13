# Operational Runbook v1.0

## 1. Incident Severity Levels

| Severity | Description | Impact | Response Time (SLA) | Update Frequency |
| :--- | :--- | :--- | :--- | :--- |
| **SEV1 (Critical)** | System Down / Data Loss / Major Financial Risk | Trading halted, substantial financial loss possible. | Immediate (< 15m) | Every 30m |
| **SEV2 (High)** | Degradation of Critical Service | Trading continues withreduced functionality or delayed data throughout. | < 1h | Every 2h |
| **SEV3 (Minor)** | Non-Critical Bug / Warning | Minor data delay, UI glitch, no financial risk. | < 4h | Daily |
| **SEV4 (Cosmetic)**| Documentation / Low Priority | No operational impact. | N/A | Weekly |

## 2. Incident Response Workflow

### Phase 1: Detection & Classification
- **Trigger**: Alert from monitoring system (e.g., Data Freshness alert, Service Down).
- **Action**: On-call engineer acknowledges alert.
- **Classification**: Assign Severity Level based on table above.

### Phase 2: Containment & Mitigation
- **Goal**: Stop the bleeding. Prioritize service restoration over root cause analysis.
- **Runbook Actions**:
    - **Data Stale**: Check upstream vendor status -> Restart Ingestion Service -> Switch to Backup Provider (if avail).
    - **Execution Failure**: Engage "Close-Only" mode (see Degradation Policy) -> Cancel open orders.
    - **System Crash**: Rollback to last known good build -> Restart containers.

### Phase 3: Resolution & Recovery
- **Goal**: Restore full service functionality.
- **Action**: Apply permanent fix, verify data integrity, resume normal trading.

## 3. Escalation Matrix

| Role | Contact | Trigger |
| :--- | :--- | :--- |
| **L1 Support** | On-Call Engineer | Initial Alert |
| **L2 Lead Dev** | Lead Engineer | SEV1 > 30m, SEV2 > 2h |
| **L3 Stakeholder**| Project Manager / CTO | SEV1 > 1h, Financial Loss > $X |

## 4. Post-Incident Review (PIR)
*Required for all SEV1 and SEV2 incidents.*

**Template:**
- **Date/Time**:
- **Severity**:
- **Impact**:
- **Root Cause**:
- **Timeline**:
    - [Time] Alert fired
    - [Time] Acknowledged
    - [Time] Mitigation applied
    - [Time] Resolved
- **Corrective Actions**: (Jira tickets / Tasks)
    1. [Immediate Fix]
    2. [Long-term Prevention]
