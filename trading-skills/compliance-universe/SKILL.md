# SKILL: Compliance, Universe Selection & Regulatory Controls
**Project:** Multi-Agent AI Trading System — Indian Market (NSE / USD-INR / MCX Gold)
**Applies To:** All agents; especially Pre-trade Checks, Order Routing, Audit Trail
**Version:** aligned with Plan v1.3.7

---

## PURPOSE
This skill governs SEBI compliance, broker rules, universe eligibility, pre-trade checks, audit trails, and data usage policy. Any agent that generates or routes orders **must** consult this skill before submission.

---

## 1. REGULATORY FRAMEWORK

### 1.1 SEBI Obligations
- Algorithmic trading notification or approval: confirm status and maintain documentation of any applicable SEBI algorithmic trading registration/notification
- Periodic audit: schedule and document compliance audits per SEBI's mandated frequency
- Audit trail: retained for minimum period required by SEBI (maintain evidence of retention policy)
- Reportable events: define list of events that must be reported to SEBI/broker; automate reporting where possible

### 1.2 Data Usage Policy (Hard Rule)
- **ONLY** publicly released or contractually licensed data feeds are permitted
- **PROHIBITED**: unpublished data, embargoed information, material non-public information (MNPI), insider information of any kind
- Every data source must have a documented licensing or public-availability justification in the data registry

### 1.3 Broker Compliance
- Static Mumbai IP whitelisting: enforced in production; document whitelisted IPs and review quarterly
- Broker API access controls: credentials in centralised secrets manager; never in code or logs
- Order rejection tracking: alert if broker rejection rate exceeds threshold; log every rejection with reason code

---

## 2. UNIVERSE ELIGIBILITY FILTERS

Applied quarterly by the Preprocessing Agent. A symbol must pass **all** filters to be included:

| Filter | Threshold | Data Source |
|--------|-----------|------------|
| Average Daily Turnover | >= ₹10 Cr (6-month rolling average) | NSE Bhav Copy |
| Impact Cost | <= 0.50 % | NSE published impact cost data |
| Free Float | >= 20 % (NSE IWF methodology) | NSE |
| Price | >= ₹50 | NSE Bhav Copy |
| Listing Age | >= 12 months from listing date | NSE |
| Circuit Hit Count | <= 5 circuit breaker hits in last 3 months | NSE |

### 2.1 Universe Change Protocol
1. Run filters on quarter-end data
2. Identify additions and deletions
3. Log change set with date and filter reason for each change
4. Increment universe version number
5. Propagate new universe to all downstream agents before next trading session

### 2.2 Point-in-Time Integrity
- Backtests and paper trading must use point-in-time universe snapshots (no forward-looking universe)
- Delisted symbols retained in historical data for survivorship bias control
- Universe version is recorded on every model artifact and every trade log

---

## 3. PRE-TRADE COMPLIANCE CHECKS (Every Order)

All checks must pass before order submission. Any failure → reject order and log reason.

### 3.1 Position Limit Checks
- [ ] Post-trade position size <= max per-instrument limit
- [ ] Post-trade sector concentration <= sector concentration limit
- [ ] Post-trade gross exposure <= current exposure cap (from Risk Overseer)
- [ ] Post-trade notional <= current capacity ceiling

### 3.2 Margin & Margin Rules
- [ ] Available margin >= required margin for the order
- [ ] Margin utilisation post-trade <= broker-imposed margin limit
- [ ] F&O lot size compliance: order quantity is exact multiple of lot size

### 3.3 Circuit Breaker Rules
- [ ] Instrument is not currently in a circuit breaker halt
- [ ] Instrument has not hit upper/lower circuit in current session (flag and reduce participation)
- [ ] Market-wide circuit breaker status checked before session open

### 3.4 Order Type Compliance
- [ ] Order type is permitted for the instrument and segment (NSE equity, NSE F&O, NSE currency, MCX commodity)
- [ ] Shorting: only where allowed; F&O segment only for short index/stock futures
- [ ] Leverage: within broker-permitted and SEBI-mandated leverage limits

### 3.5 Participation Limits
- [ ] Order quantity <= max participation rate × ADV for the instrument bucket
- [ ] If impact monitor signals threshold breach, apply auto-reduction before submission (see Execution Skill Section 4.3)

---

## 4. AUDIT TRAIL REQUIREMENTS

Every trade event must be logged with the following fields:

```
{
  "event_id": "unique ID",
  "event_type": "ORDER_INTENT | ORDER_SUBMITTED | FILL | PARTIAL_FILL | CANCELLATION | REJECTION",
  "timestamp_utc": "ISO8601",
  "instrument": "symbol",
  "direction": "BUY | SELL",
  "quantity": integer,
  "price": float,
  "order_type": "LIMIT | MARKET | SL | SL-M",
  "model_version": "string",
  "signal_source": "agent_name",
  "universe_version": "string",
  "plan_version": "string",
  "pre_trade_checks_passed": true/false,
  "rejection_reason": "string or null",
  "data_source_tags": ["primary_api", "fallback_scraper"],
  "risk_mode": "NORMAL | REDUCE_ONLY | CLOSE_ONLY | KILL_SWITCH"
}
```

- Audit logs are **immutable** after write; no deletion or modification permitted
- Retention: per SEBI retention policy; minimum 5 years recommended
- Logs are stored separately from trading database with independent access controls

---

## 5. TRADING CONSTRAINTS BY PRODUCT

| Product | Shorting | Leverage | Lot Size | Order Types |
|---------|---------|---------|---------|-------------|
| NSE Equity (Cash) | No (intraday MIS only via broker) | Per broker MIS margin | 1 share | Limit, Market, SL, SL-M |
| NSE Index Futures | Yes | SEBI SPAN margin | Exchange-defined | Limit, Market, SL |
| NSE Stock Futures | Yes | SEBI SPAN margin | Exchange-defined | Limit, Market, SL |
| NSE Currency Futures (USD/INR) | Yes | SEBI currency margin | 1000 USD per lot | Limit, Market, SL |
| MCX Gold Futures | Yes | MCX SPAN margin | 1 kg (GOLD), 100 g (GOLDM) | Limit, Market, SL |

---

## 6. INFRASTRUCTURE & SECURITY COMPLIANCE

### 6.1 Mumbai VPS Requirements (Production Only)
- Deployment: Mumbai-based VPS with static IP
- IP whitelisting: registered with broker API; changes require formal update process
- System time: NTP-synced; drift alert if > 100 ms
- DR plan: RPO and RTO targets documented and tested

### 6.2 Secrets Management
- All credentials (broker API keys, database passwords) in centralised secrets manager
- No credentials in source code, config files checked into version control, or log output
- Access audit: who accessed which credential and when; reviewed quarterly

### 6.3 Logging & Retention
- Centralised logging with defined retention policy
- Log categories: trade events, risk events, system errors, compliance events, model versions
- Logs shipped to secure off-system storage daily

---

## 7. REPORTING AUTOMATION

- Daily end-of-day: trade summary, PnL, position snapshot, compliance check results
- Weekly: slippage report, circuit breaker encounters, rejection summary
- Monthly: regulatory reporting (where mandated), risk budget utilisation, universe changes
- Quarterly: SEBI audit preparation, stress test review, universe rebalance

---

## COMPLIANCE AGENT CHECKLIST
- [ ] Data sources: all have documented licensing / public-availability justification
- [ ] No MNPI or embargoed data in any pipeline
- [ ] Pre-trade checks: all 5 categories implemented and tested
- [ ] Audit trail: all 16 required fields present in every trade event
- [ ] Audit logs: immutable, retained per policy, independently stored
- [ ] Universe filters: all 6 applied quarterly; change log versioned
- [ ] Point-in-time universe used in all backtests (delisted symbols retained)
- [ ] IP whitelisting registered with broker; reviewed quarterly
- [ ] Secrets in centralised manager; no credentials in code or logs
- [ ] DR plan documented; RPO/RTO tested
- [ ] Reporting automation scheduled: daily, weekly, monthly, quarterly
