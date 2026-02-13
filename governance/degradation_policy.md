# System Degradation Policy

This document defines the operational states of the trading system and the triggers for transitioning between them.

## 1. Operational States

### A. Normal Operation (Green)
- **Description**: All systems nominal. All SLAs met.
- **Allowed Actions**: Full trading capabilities (Entry, Exit, Scaling).
- **Position Sizing**: 100% of model recommendation.

### B. Degraded State (Yellow) - Reduce-Only
- **Description**: Minor SLA breaches or data warnings.
- **Triggers**:
  - One or more Data Freshness SLAs missed (e.g., latency > 1s but < 3s).
  - Non-critical outlier detected.
  - API error rate > 1% but < 5%.
- **Allowed Actions**: 
  - **NO NEW ENTRIES**.
  - Open positions can be managed or closed.
  - Risk reduction logic active.
- **Position Sizing**: 0% for new trades.

### C. Critical Failure (Red) - Halt / Close-Only
- **Description**: Major system failure or critical data loss.
- **Triggers**:
  - Critical Data Freshness SLA missed (> 5s).
  - API error rate > 5%.
  - Circuit Breaker tripped (Drawdown limit).
  - Unauthorized access attempt detected.
- **Allowed Actions**:
  - **EMERGENCY CLOSE ONLY** (if safely possible).
  - System Halt.
  - Notification to Human Operator immediately.

## 2. Recovery Protocol
- **From Red to Yellow**: Requires manual human intervention and verification.
- **From Yellow to Green**: Automatic recovery allowed if metrics stabilize for > 5 minutes.
