# Controls Matrix

This matrix maps operational risks to specific controls, evidence verification, and ownership.

## 1. Access & Security
| ID | Control Objective | Control Activity | Evidence | Owner |
| :--- | :--- | :--- | :--- | :--- |
| **SEC-01** | Prevent unauthorized trading access. | Static IP Whitelisting (Mumbai VPS API only). | Broker Console Logs / IP Logs | Owner |
| **SEC-02** | Secure API Credentials. | Secrets management (no hardcoded keys). | Code Review / Secret Manager Audit | Owner |

## 2. Data Integrity
| ID | Control Objective | Control Activity | Evidence | Owner |
| :--- | :--- | :--- | :--- | :--- |
| **DAT-01** | Ensure market data accuracy. | Multi-source validation (NSE Source vs Broker). | Data Logs / Comparison Reports | Owner |
| **DAT-02** | Prevent processing of stale data. | Timestamp checks on every tick ingest. | Ingestion Logs (checking `timestamp` vs `now`) | Owner |

## 3. Trading Risk
| ID | Control Objective | Control Activity | Evidence | Owner |
| :--- | :--- | :--- | :--- | :--- |
| **RSK-01** | Prevent exceeding capital limits. | Pre-trade check of max position size vs Account Balance. | Order Manager Logs | Owner |
| **RSK-02** | Stop trading during market anomalies. | Volatility circuit breakers (VIX spike / Flash crash). | System State Logs (Normal -> Halt) | Owner |
| **RSK-03** | Ensure Strategy Adherence. | Daily reconciliation of trades vs signals. | End-of-Day Report | Partner |
