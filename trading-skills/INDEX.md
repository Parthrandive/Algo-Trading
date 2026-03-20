# Trading System Skills — Master Index
**Project:** Multi-Agent AI Trading System (NSE / USD-INR / MCX Gold)  
**Plan Version:** v1.3.7 | Last Updated: March 2026

---

## HOW TO USE THESE SKILLS

Each skill file is a self-contained instruction set for a specific agent or layer of the system. When an agent starts a task, it must load the relevant skill(s) first and follow all rules within.

> **Rule**: If a skill rule conflicts with a general instruction, the skill rule takes precedence within its domain. The Risk Overseer Skill always takes precedence over all other skills on risk decisions.

---

## SKILL DIRECTORY

| Skill | File | Applies To | Must Read If... |
|-------|------|-----------|----------------|
| **Data Orchestration** | `data-orchestration/SKILL.md` | NSE Sentinel, Macro Monitor, Preprocessing, Textual Data agents | Acquiring, validating, normalising, or storing any market data |
| **Analyst Board** | `analyst-board/SKILL.md` | Technical, Regime, Sentiment, Consensus agents | Training, validating, or running any Phase 2 model |
| **Execution & Fast Loop** | `execution-fastloop/SKILL.md` | Trader Ensemble, order routing, latency management | Placing orders, managing the Fast/Slow Loop, teacher-student inference |
| **Risk Overseer** | `risk-overseer/SKILL.md` | Independent Risk Overseer, kill-switch, stress testing | Any risk control decision, kill-switch drill, stress test run |
| **MLOps & Governance** | `mlops-governance/SKILL.md` | All model training, promotion, and rollback workflows | Promoting any model artifact to shadow or production |
| **Compliance & Universe** | `compliance-universe/SKILL.md` | All agents; especially pre-trade and order routing | Submitting any order, changing universe, or handling regulatory data |

---

## CROSS-SKILL DEPENDENCIES

```
Data Orchestration  ──►  Analyst Board  ──►  Execution Fast Loop
        │                      │                      │
        │                      ▼                      ▼
        └──────────────►  Risk Overseer  ◄──────────────┘
                               │
                               ▼
                    MLOps & Governance
                    Compliance & Universe (all layers)
```

- **Data Orchestration** must complete feed integrity checks before Analyst Board consumes data
- **Analyst Board** outputs (regime, sentiment, consensus) must be cached before Fast Loop reads them
- **Risk Overseer** has veto authority over all execution decisions; its rules override all other agents
- **MLOps & Governance** governs every model promotion across all phases
- **Compliance & Universe** applies to every order submitted; no exceptions

---

## CURRENT CYCLE TIER STATUS (v1.3.7)

| Tier | Items | Status |
|------|-------|--------|
| Tier 1 | Real-time impact monitor, dynamic vol-scaled budgets, order-book imbalance in Fast Loop, latency discipline | **Implement now** |
| Tier 2 | Shadow A/B enforcement, crisis-weighted voting (60–70 % cap) | After 3 months paper trading |
| Tier 3 | FPGA/hardware, market-making module, online RL, full pod model | **DEFERRED — DO NOT IMPLEMENT** |

---

## GO-LIVE MINIMUM CRITERIA (all must pass)

| Metric | Target |
|--------|--------|
| Paper trading duration | >= 3 calendar months |
| Paper trading uptime | >= 80 % |
| Annualised Sharpe | >= 1.8 |
| Sortino | >= 2.0 |
| Max Drawdown | <= 8 % |
| Win Rate | >= 52 % |
| Profit Factor | >= 1.5 |
| Avg realised slippage | <= model estimate + 20 bps |
| Data uptime (NSE hours) | >= 99.5 % |
| Critical compliance violations | Zero |
| Broker/SEBI flags or order rejections | Zero |
